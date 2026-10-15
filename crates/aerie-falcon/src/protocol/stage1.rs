//! Stage one: basis-16 products and the clear-projection source link.

use crate::error::{AggregationError, MalformedProof, RejectionStage, StatementMismatch};
use jolt_field::{Field as _, One as _, Ring as _, Zero as _};

use super::transcript::PioTranscript;
use crate::substrate::OpeningField as E;

use super::algebra::eq_evaluation;
use super::projection::reverse::projection_weights_at;
use super::projection::{self, ProjectionOutputs, OUTPUT_COUNT};
use super::stage::{append_round, evaluate_compressed, StageShapes};
use super::FalconDegree;

pub const GAMMA_SITE: &str = "aerie/falcon/v1/stage1/gamma";
pub const TERMINAL_SITE: &str = "aerie/falcon/v1/stage1/terminal";

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

/// Verify stage one and return its three local terminal points.
pub fn verify(
    degree: FalconDegree,
    records: usize,
    live_rows: usize,
    seed: &[u8],
    outputs: &ProjectionOutputs,
    proof: &Stage1Proof,
    transcript: &mut impl PioTranscript,
) -> Result<Stage1Output, AggregationError> {
    if records == 0
        || !records.is_power_of_two()
        || live_rows == 0
        || live_rows > records
        || degree.padded_records(live_rows) != Ok(records)
    {
        return Err(AggregationError::StatementMismatch(
            StatementMismatch::PublicInputShape,
        ));
    }
    let shapes = StageShapes::for_degree(degree);
    let source_count = shapes.basis16_sources;
    let rounds = degree.logn() + records.trailing_zeros() as usize;
    if proof.rounds.len() != rounds || proof.basis16_claims.len() != source_count {
        return Err(AggregationError::MalformedProof(MalformedProof::StageShape));
    }
    if !projection::check_outputs(degree, outputs) {
        return Err(AggregationError::ProofRejected {
            stage: RejectionStage::Stage1,
        });
    }
    projection::absorb_outputs(outputs, transcript)?;
    let lambda = sample_lambda(transcript)?;
    let tau = transcript.challenge_extensions("aerie/falcon/v1/tau", rounds)?;
    let gammas = transcript.challenge_extensions(GAMMA_SITE, shapes.stages[0].gamma_count)?;
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
        return Err(AggregationError::ProofRejected {
            stage: RejectionStage::Stage1,
        });
    }

    Ok(Stage1Output {
        point,
        basis16_claims: proof.basis16_claims.clone(),
        projection_point,
        slack_projection_point,
        projection_claims: proof.projection_claims,
    })
}

#[doc(hidden)]
#[inline]
pub fn absorb_terminal(
    transcript: &mut impl PioTranscript,
    basis: &[[E; 2]],
    projection: &[E; 3],
) -> Result<(), AggregationError> {
    let claims: Vec<_> = basis.iter().flatten().chain(projection).copied().collect();
    transcript.absorb_extensions(TERMINAL_SITE, &claims)?;
    Ok(())
}

#[doc(hidden)]
#[inline]
pub fn sample_lambda(transcript: &mut impl PioTranscript) -> Result<Vec<E>, AggregationError> {
    let mu = transcript.challenge_extensions("aerie/falcon/v1/mu", 8)?;
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

#[doc(hidden)]
#[inline]
#[must_use]
pub fn output_claim(outputs: &ProjectionOutputs, lambda: &[E]) -> E {
    lambda
        .iter()
        .zip(outputs.values())
        .fold(E::zero(), |sum, (&weight, value)| {
            sum + weight * E::from_u64(value.to_canonical_u64())
        })
}

#[doc(hidden)]
#[inline]
#[must_use]
pub fn reindex_projection_points(degree: FalconDegree, stage_point: &[E]) -> (Vec<E>, Vec<E>) {
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
