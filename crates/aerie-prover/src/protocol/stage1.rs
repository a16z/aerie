//! Stage one: basis-16 products and the clear-projection source link.

use crate::error::AggregationError;
use jolt_field::{One as _, Zero as _};

use super::transcript::PioTranscript;
use crate::substrate::OpeningField as E;

use super::projection::{self, ForwardProjection};
use super::source_tables::{CodedSource, PreparedCoefficients, SourceTables};
use super::stage::{append_round, evaluate_compressed, StageShapes};
use super::{FalconDegree, FalconWitness};

mod basis16_product;
mod projection_link;
use basis16_product::Prover as Basis16Prover;
use projection_link::ProjectionLinkProver;

pub use aerie_falcon::protocol::stage1::*;

#[derive(Clone, Copy)]
struct ProverRound {
    coefficient: [E; 3],
    budget: Option<[E; 3]>,
    link_coefficient: [E; 3],
    slack: [E; 3],
}

/// Prove stage one, including output absorption and the fresh batching
/// challenges required by Section 10.1.
#[cfg(any(test, feature = "internals"))]
pub fn prove(
    batch: &FalconWitness,
    tables: &SourceTables,
    projection: &ForwardProjection,
    transcript: &mut impl PioTranscript,
) -> Result<(Stage1Proof, Stage1Output), AggregationError> {
    let prepared = PreparedCoefficients::new(tables)?;
    prove_prepared(batch, tables, &prepared, projection, transcript)
}

pub(crate) fn prove_prepared(
    batch: &FalconWitness,
    tables: &SourceTables,
    prepared: &PreparedCoefficients<'_>,
    projection: &ForwardProjection,
    transcript: &mut impl PioTranscript,
) -> Result<(Stage1Proof, Stage1Output), AggregationError> {
    tracing::info_span!("stage1_driver_setup")
        .in_scope(|| validate_prover_shape(batch, tables, projection))?;
    let (lambda, logn, rounds, tau, basis_count, gammas) = tracing::info_span!(
        "stage1_driver_transcript",
        phase = "initial"
    )
    .in_scope(|| -> Result<_, AggregationError> {
        projection::absorb_outputs(projection.outputs(), transcript)?;
        let lambda = sample_lambda(transcript)?;
        let degree = FalconDegree::from_degree(batch.degree())?;
        let shapes = StageShapes::for_degree(degree);
        let logn = degree.logn();
        let rounds = logn + tables.records().trailing_zeros() as usize;
        let tau = transcript.challenge_extensions("aerie/falcon/v1/tau", rounds)?;
        let basis_count = shapes.basis16_sources;
        let gammas = transcript.challenge_extensions(GAMMA_SITE, shapes.stages[0].gamma_count)?;
        Ok((lambda, logn, rounds, tau, basis_count, gammas))
    })?;

    let target = tracing::info_span!("stage1_driver_claim_combine", phase = "initial")
        .in_scope(|| output_claim(projection.outputs(), &lambda));
    let setup_products = || -> Result<_, AggregationError> {
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
                .in_scope(|| coefficient_product.round(0))?;
        let budget = budget_product
            .as_mut()
            .map(|instance| {
                tracing::info_span!("basis16_product_round", round = 0, phase = "inactive")
                    .in_scope(|| instance.round(0))
            })
            .transpose()?;
        Ok((coefficient_product, budget_product, coefficient, budget))
    };
    let setup_link = || -> Result<_, AggregationError> {
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
                .in_scope(|| coefficient_product.round(stage_round))?;
        let budget_message = budget_product
            .as_mut()
            .map(|instance| {
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
            })
            .transpose()?;
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

    let degree = FalconDegree::from_degree(batch.degree())?;
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

fn validate_prover_shape(
    batch: &FalconWitness,
    tables: &SourceTables,
    projection: &ForwardProjection,
) -> Result<(), AggregationError> {
    let degree = FalconDegree::from_degree(batch.degree())?;
    if tables.degree() != batch.degree()
        || tables.records() != batch.padded_rows
        || projection.live_rows != batch.live_rows
        || projection.geometry.records != batch.padded_rows
        || projection.geometry.degree != degree
    {
        return Err(AggregationError::internal(
            "stage-one inputs have inconsistent batch shapes",
        ));
    }
    Ok(())
}

fn add_scaled(target: &mut [E; 3], message: [E; 3], coefficient: E) {
    for (target, value) in target.iter_mut().zip(message) {
        *target += coefficient * value;
    }
}

#[cfg(test)]
mod tests;

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
) -> Result<(), AggregationError> {
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
        .in_scope(|| coefficient_product.bind(stage_round, challenge, round.coefficient))?;
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
        .in_scope(|| instance.bind(stage_round, challenge, local_message))?;
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
