//! Stage one: basis-16 products and the clear-projection source link.

use akita_error::AkitaError;
use jolt_field::{Field as _, One as _, Ring as _, Zero as _};

use super::transcript::PioTranscript;
use crate::substrate::OpeningField as E;

use super::algebra::eq_evaluation;
use super::projection::{self, ForwardProjection, ProjectionOutputs, OUTPUT_COUNT};
use super::source_tables::{CodedSource, PreparedCoefficients, SourceTables};
use super::stage::{append_round, evaluate_compressed};
use super::{FalconDegree, FalconWitness};

mod basis16_product;
mod projection_link;
use basis16_product::Prover as Basis16Prover;
use projection_link::{projection_weights_at, ProjectionLinkProver};

const GAMMA_SITE: &str = "aerie/falcon/v2/stage1/gamma";
const TERMINAL_SITE: &str = "aerie/falcon/v2/stage1/terminal";

/// Wire message for stage one.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage1Proof {
    /// Ordinary cubic messages `(g0, g2, g3)` in stage round order.
    pub rounds: Vec<[E; 3]>,
    /// Low and high basis-16 claims in canonical order `a1, a2, [u2]`.
    pub basis16_claims: Vec<[E; 2]>,
    /// Projection claims `(v_S1, v_e, v_Delta)`.
    pub projection_claims: [E; 3],
}

/// Claims and points passed from stage one to the later stages and opening.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage1Output {
    /// Common stage challenge sequence `r^(1)`.
    pub point: Vec<E>,
    /// Basis-16 child claims in canonical source order.
    pub basis16_claims: Vec<[E; 2]>,
    /// Canonically reindexed coefficient point `r_Pi`.
    pub projection_point: Vec<E>,
    /// Canonically reindexed slack point `r_Pi,e`.
    pub slack_projection_point: Vec<E>,
    /// Projection claims `(v_S1, v_e, v_Delta)`.
    pub projection_claims: [E; 3],
}

#[derive(Clone, Copy)]
struct ProverRound {
    coefficient: [E; 3],
    budget: Option<[E; 3]>,
    link_coefficient: [E; 3],
    slack: [E; 3],
}

/// Prove stage one, including output absorption and the fresh batching
/// challenges required by Section 10.1.
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn prove(
    batch: &FalconWitness,
    tables: &SourceTables,
    projection: &ForwardProjection,
    transcript: &mut impl PioTranscript,
) -> Result<(Stage1Proof, Stage1Output), AkitaError> {
    let prepared = PreparedCoefficients::new(tables)?;
    prove_prepared(batch, tables, &prepared, projection, transcript)
}

pub(crate) fn prove_prepared(
    batch: &FalconWitness,
    tables: &SourceTables,
    prepared: &PreparedCoefficients<'_>,
    projection: &ForwardProjection,
    transcript: &mut impl PioTranscript,
) -> Result<(Stage1Proof, Stage1Output), AkitaError> {
    tracing::info_span!("stage1_driver_setup")
        .in_scope(|| validate_prover_shape(batch, tables, projection))?;
    let (lambda, logn, rounds, tau, basis_count, gammas) = tracing::info_span!(
        "stage1_driver_transcript",
        phase = "initial"
    )
    .in_scope(|| -> Result<_, AkitaError> {
        projection::absorb_outputs(projection.outputs(), transcript)?;
        let lambda = sample_lambda(transcript)?;
        let logn = FalconDegree::from_degree(batch.degree())
            .map_err(invalid)?
            .logn();
        let rounds = logn + tables.records().trailing_zeros() as usize;
        let tau = transcript.challenge_extensions("aerie/falcon/v2/tau", rounds)?;
        let basis_count = 2 + usize::from(batch.degree() == FalconDegree::Falcon1024.degree());
        let gammas = transcript.challenge_extensions(GAMMA_SITE, basis_count + 1)?;
        Ok((lambda, logn, rounds, tau, basis_count, gammas))
    })?;

    let target = tracing::info_span!("stage1_driver_claim_combine", phase = "initial")
        .in_scope(|| output_claim(projection.outputs(), &lambda));
    let setup_products = || -> Result<_, AkitaError> {
        let mut coefficient_product =
            tracing::info_span!("basis16_product_setup", source_count = 2, rounds).in_scope(
                || {
                    Basis16Prover::new(
                        vec![prepared.source(1), prepared.source(2)],
                        gammas[..2].to_vec(),
                        &tau,
                        0,
                    )
                },
            )?;
        let budget_source = (basis_count == 3)
            .then(|| CodedSource::new(tables.budget(2), -8, 7))
            .transpose()?;
        let mut budget_product = if basis_count == 3 {
            Some(
                tracing::info_span!(
                    "basis16_product_setup",
                    source_count = 1,
                    rounds = rounds - logn
                )
                .in_scope(|| {
                    Basis16Prover::new(
                        vec![budget_source.as_ref().expect("Falcon-1024 budget source")],
                        vec![gammas[2]],
                        &tau[logn..],
                        logn,
                    )
                })?,
            )
        } else {
            None
        };
        let coefficient =
            tracing::info_span!("basis16_product_round", round = 0, phase = "coefficient")
                .in_scope(|| coefficient_product.round(0));
        let budget = budget_product.as_mut().map(|instance| {
            tracing::info_span!("basis16_product_round", round = 0, phase = "inactive")
                .in_scope(|| instance.round(0))
        });
        Ok((coefficient_product, budget_product, coefficient, budget))
    };
    let setup_link = || -> Result<_, AkitaError> {
        let link = tracing::info_span!("projection_link_setup", rounds)
            .in_scope(|| ProjectionLinkProver::new(batch, projection, &lambda, target))?;
        let coefficient = tracing::info_span!(
            "projection_link_coefficient_round",
            round = 0,
            phase = link.coefficient_round_phase(0)
        )
        .in_scope(|| link.coefficient_round(0));
        let slack =
            tracing::info_span!("projection_link_slack_round", round = 0, phase = "inactive")
                .in_scope(|| link.slack_round(0));
        Ok((link, coefficient, slack))
    };
    let (products, link) = super::join_if(true, setup_products, setup_link);
    let (
        mut coefficient_product,
        mut budget_product,
        first_coefficient_message,
        first_budget_message,
    ) = products?;
    let (mut link, first_link_coefficient_message, first_slack_message) = link?;
    debug_assert_eq!(link.initial_claim(), target);
    let mut claim = tracing::info_span!("stage1_driver_claim_combine")
        .in_scope(|| gammas[basis_count] * target);
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);

    finish_prover_round(
        0,
        logn,
        gammas[basis_count],
        ProverRound {
            coefficient: first_coefficient_message,
            budget: first_budget_message,
            link_coefficient: first_link_coefficient_message,
            slack: first_slack_message,
        },
        &mut coefficient_product,
        &mut budget_product,
        &mut link,
        &mut claim,
        &mut messages,
        &mut point,
        transcript,
    )?;

    for stage_round in 1..rounds {
        let phase = if stage_round < logn {
            "coefficient"
        } else {
            "row"
        };
        let projection_phase = link.coefficient_round_phase(stage_round);
        let coefficient_message =
            tracing::info_span!("basis16_product_round", round = stage_round, phase)
                .in_scope(|| coefficient_product.round(stage_round));
        let budget_message = budget_product.as_mut().map(|instance| {
            tracing::info_span!(
                "basis16_product_round",
                round = stage_round,
                phase = if stage_round < logn {
                    "inactive"
                } else {
                    "active"
                }
            )
            .in_scope(|| instance.round(stage_round))
        });
        let link_coefficient_message = tracing::info_span!(
            "projection_link_coefficient_round",
            round = stage_round,
            phase = projection_phase
        )
        .in_scope(|| link.coefficient_round(stage_round));
        let slack_message = tracing::info_span!(
            "projection_link_slack_round",
            round = stage_round,
            phase = if stage_round < link.inactive_slack_rounds() {
                "inactive"
            } else {
                "active"
            }
        )
        .in_scope(|| link.slack_round(stage_round));
        finish_prover_round(
            stage_round,
            logn,
            gammas[basis_count],
            ProverRound {
                coefficient: coefficient_message,
                budget: budget_message,
                link_coefficient: link_coefficient_message,
                slack: slack_message,
            },
            &mut coefficient_product,
            &mut budget_product,
            &mut link,
            &mut claim,
            &mut messages,
            &mut point,
            transcript,
        )?;
    }

    let mut basis_claims = coefficient_product.terminal_pairs()?;
    if let Some(budget) = budget_product {
        basis_claims.extend(budget.terminal_pairs()?);
    }
    let projection_claims = link.terminal_claims();
    tracing::info_span!("stage1_driver_transcript", phase = "terminal")
        .in_scope(|| absorb_terminal(transcript, &basis_claims, &projection_claims))?;

    let degree = FalconDegree::from_degree(batch.degree()).map_err(invalid)?;
    let (projection_point, slack_projection_point) = reindex_projection_points(degree, &point);
    Ok((
        Stage1Proof {
            rounds: messages,
            basis16_claims: basis_claims.clone(),
            projection_claims,
        },
        Stage1Output {
            point,
            basis16_claims: basis_claims,
            projection_point,
            slack_projection_point,
            projection_claims,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn finish_prover_round(
    stage_round: usize,
    logn: usize,
    link_weight: E,
    round: ProverRound,
    coefficient_product: &mut Basis16Prover<'_>,
    budget_product: &mut Option<Basis16Prover<'_>>,
    link: &mut ProjectionLinkProver<'_>,
    claim: &mut E,
    messages: &mut Vec<[E; 3]>,
    point: &mut Vec<E>,
    transcript: &mut impl PioTranscript,
) -> Result<(), AkitaError> {
    let phase = if stage_round < logn {
        "coefficient"
    } else {
        "row"
    };
    let projection_phase = link.coefficient_round_phase(stage_round);
    let message =
        tracing::info_span!("stage1_driver_claim_combine", round = stage_round).in_scope(|| {
            let mut message = [E::zero(); 3];
            add_scaled(&mut message, round.coefficient, E::one());
            if let Some(local_message) = round.budget {
                add_scaled(&mut message, local_message, E::one());
            }
            add_scaled(&mut message, round.link_coefficient, link_weight);
            add_scaled(&mut message, round.slack, link_weight);
            message
        });
    let challenge = tracing::info_span!(
        "stage1_driver_transcript",
        phase = "round",
        round = stage_round
    )
    .in_scope(|| append_round(transcript, "stage1", stage_round, message))?;
    *claim = tracing::info_span!(
        "stage1_driver_claim_combine",
        round = stage_round,
        phase = "bind_claim"
    )
    .in_scope(|| evaluate_compressed(message, *claim, challenge));
    tracing::info_span!("basis16_product_bind", round = stage_round, phase)
        .in_scope(|| coefficient_product.bind(stage_round, challenge, round.coefficient));
    if let (Some(instance), Some(local_message)) = (budget_product.as_mut(), round.budget) {
        tracing::info_span!(
            "basis16_product_bind",
            round = stage_round,
            phase = if stage_round < logn {
                "inactive"
            } else {
                "active"
            }
        )
        .in_scope(|| instance.bind(stage_round, challenge, local_message));
    }
    tracing::info_span!(
        "projection_link_coefficient_bind",
        round = stage_round,
        phase = projection_phase
    )
    .in_scope(|| link.coefficient_bind(stage_round, challenge, round.link_coefficient));
    tracing::info_span!(
        "projection_link_slack_bind",
        round = stage_round,
        phase = if stage_round < link.inactive_slack_rounds() {
            "inactive"
        } else {
            "active"
        }
    )
    .in_scope(|| link.slack_bind(stage_round, challenge, round.slack));
    point.push(challenge);
    messages.push(message);
    Ok(())
}

/// Verify stage one and return its three local terminal points.
pub fn verify(
    degree: FalconDegree,
    records: usize,
    live_rows: usize,
    seed: &[u8],
    outputs: &ProjectionOutputs,
    proof: &Stage1Proof,
    transcript: &mut impl PioTranscript,
) -> Result<Stage1Output, AkitaError> {
    if records == 0
        || !records.is_power_of_two()
        || live_rows == 0
        || live_rows > records
        || degree.padded_records(live_rows) != Ok(records)
    {
        return Err(invalid("stage-one batch shape is not canonically padded"));
    }
    let source_count = 2 + usize::from(degree == FalconDegree::Falcon1024);
    let rounds = degree.logn() + records.trailing_zeros() as usize;
    if proof.rounds.len() != rounds || proof.basis16_claims.len() != source_count {
        return Err(invalid("stage-one proof shape mismatch"));
    }
    projection::check_outputs(degree, outputs)?;
    projection::absorb_outputs(outputs, transcript)?;
    let lambda = sample_lambda(transcript)?;
    let tau = transcript.challenge_extensions("aerie/falcon/v2/tau", rounds)?;
    let gammas = transcript.challenge_extensions(GAMMA_SITE, source_count + 1)?;
    let target = output_claim(outputs, &lambda);
    let mut claim = gammas[source_count] * target;
    let mut point = Vec::with_capacity(rounds);
    for (round, message) in proof.rounds.iter().enumerate() {
        let challenge = append_round(transcript, "stage1", round, *message)?;
        claim = evaluate_compressed(*message, claim, challenge);
        point.push(challenge);
    }
    absorb_terminal(transcript, &proof.basis16_claims, &proof.projection_claims)?;

    let mut expected = E::zero();
    let half = E::from_u64(2).inverse().expect("two is invertible");
    for (index, pair) in proof.basis16_claims.iter().enumerate() {
        let (local_tau, local_point, inactive) = if index + 1 == source_count {
            if degree == FalconDegree::Falcon1024 {
                (
                    &tau[degree.logn()..],
                    &point[degree.logn()..],
                    degree.logn(),
                )
            } else {
                (&tau[..], &point[..], 0)
            }
        } else {
            (&tau[..], &point[..], 0)
        };
        let scale = (0..inactive).fold(E::one(), |value, _| value * half);
        expected +=
            gammas[index] * scale * eq_evaluation(local_tau, local_point) * pair[0] * pair[1];
    }

    let (projection_point, slack_projection_point) = reindex_projection_points(degree, &point);
    let weights = projection_weights_at(
        degree,
        records,
        live_rows,
        seed,
        &lambda,
        &projection_point,
        &slack_projection_point,
    )?;
    let slack_scale = (0..(degree.logn() - 2)).fold(E::one(), |value, _| value * half);
    let [v_s1, v_e, v_delta] = proof.projection_claims;
    expected += gammas[source_count]
        * (weights[0] * v_s1 + slack_scale * weights[1] * v_e + weights[2] * v_delta);
    if claim != expected {
        return Err(AkitaError::InvalidProof);
    }

    Ok(Stage1Output {
        point,
        basis16_claims: proof.basis16_claims.clone(),
        projection_point,
        slack_projection_point,
        projection_claims: proof.projection_claims,
    })
}

fn validate_prover_shape(
    batch: &FalconWitness,
    tables: &SourceTables,
    projection: &ForwardProjection,
) -> Result<(), AkitaError> {
    let degree = FalconDegree::from_degree(batch.degree()).map_err(invalid)?;
    if tables.degree() != batch.degree()
        || tables.records() != batch.padded_rows
        || projection.live_rows != batch.live_rows
        || projection.geometry.records != batch.padded_rows
        || projection.geometry.degree != degree
    {
        return Err(invalid("stage-one inputs have inconsistent batch shapes"));
    }
    Ok(())
}

fn absorb_terminal(
    transcript: &mut impl PioTranscript,
    basis: &[[E; 2]],
    projection: &[E; 3],
) -> Result<(), AkitaError> {
    let claims: Vec<_> = basis.iter().flatten().chain(projection).copied().collect();
    transcript.absorb_extensions(TERMINAL_SITE, &claims)?;
    Ok(())
}

fn sample_lambda(transcript: &mut impl PioTranscript) -> Result<Vec<E>, AkitaError> {
    let mu = transcript.challenge_extensions("aerie/falcon/v2/mu", 8)?;
    Ok((0..OUTPUT_COUNT)
        .map(|index| {
            (0..8).fold(E::one(), |weight, bit| {
                if (index >> bit) & 1 == 1 {
                    weight * mu[bit]
                } else {
                    weight
                }
            })
        })
        .collect())
}

fn output_claim(outputs: &ProjectionOutputs, lambda: &[E]) -> E {
    lambda
        .iter()
        .zip(outputs.values())
        .fold(E::zero(), |sum, (&weight, value)| {
            sum + weight * E::from_u64(value.to_canonical_u64())
        })
}

fn reindex_projection_points(degree: FalconDegree, stage_point: &[E]) -> (Vec<E>, Vec<E>) {
    let logn = degree.logn();
    let record_bits = stage_point.len() - logn;
    let low_bits = 12 - (logn + 1);
    let rest_bits = record_bits - low_bits;

    let mut coefficient = stage_point[rest_bits..rest_bits + logn].to_vec();
    coefficient.extend_from_slice(&stage_point[rest_bits + logn..]);
    coefficient.extend_from_slice(&stage_point[..rest_bits]);

    let inactive = logn - 2;
    let local = &stage_point[inactive..];
    let mut slack = local[rest_bits..rest_bits + 2].to_vec();
    slack.extend_from_slice(&local[rest_bits + 2..]);
    slack.extend_from_slice(&local[..rest_bits]);
    (coefficient, slack)
}

fn add_scaled(target: &mut [E; 3], message: [E; 3], coefficient: E) {
    for (target, value) in target.iter_mut().zip(message) {
        *target += coefficient * value;
    }
}

fn invalid(message: &str) -> AkitaError {
    AkitaError::InvalidInput(message.to_owned())
}

#[cfg(test)]
mod tests;
