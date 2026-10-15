//! Shared transcript and ordinary-round mechanics for the three stages.
//!
//! Relation provers live in their named stage modules. This module deliberately
//! contains no prover registry: it only frames the common compressed round,
//! samples challenges in transcript order, and replays ordinary sumchecks.

use crate::error::{AggregationError, MalformedProof};
use jolt_field::{Field, Ring, Zero};

use super::transcript::PioTranscript;
use super::FalconDegree;
use crate::substrate::OpeningField as E;

/// Shared challenge and transmitted-field counts for one stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageShape {
    pub gamma_count: usize,
    pub terminal_count: usize,
    pub round_message_width: usize,
}

/// Degree-dependent geometry, independent of the padded record domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageShapes {
    pub basis16_sources: usize,
    pub stages: [StageShape; 3],
}

impl StageShapes {
    #[must_use]
    pub const fn for_degree(degree: FalconDegree) -> Self {
        let (basis16_sources, stage2_gamma_count) = match degree {
            FalconDegree::Falcon512 => (2, 8),
            FalconDegree::Falcon1024 => (3, 9),
        };
        Self {
            basis16_sources,
            stages: [
                StageShape {
                    gamma_count: basis16_sources + 1,
                    terminal_count: 2 * basis16_sources + 3,
                    round_message_width: 3,
                },
                StageShape {
                    gamma_count: stage2_gamma_count,
                    terminal_count: 6,
                    round_message_width: 4,
                },
                StageShape {
                    gamma_count: 12,
                    terminal_count: 9,
                    round_message_width: 3,
                },
            ],
        }
    }
}

/// Reconstruct the omitted linear coefficient of an ordinary degree-three
/// message and evaluate it at the next challenge.
#[inline]
#[must_use]
pub fn evaluate_compressed([c0, c2, c3]: [E; 3], claim: E, x: E) -> E {
    let c1 = claim - c0 - c0 - c2 - c3;
    ((c3 * x + c2) * x + c1) * x + c0
}

/// The constant polynomial for one leading inactive variable. Repeated use
/// halves the local claim at each challenge, implementing the specified
/// `2^-k` lift without expanding the witness table.
#[inline]
#[must_use]
pub fn inactive_round(claim: E) -> [E; 3] {
    [
        claim * E::from_u64(2).inverse().expect("two is invertible"),
        E::zero(),
        E::zero(),
    ]
}

/// Append one compressed ordinary message and sample its challenge.
#[inline]
pub fn append_round(
    transcript: &mut impl PioTranscript,
    stage: &str,
    round: usize,
    message: [E; 3],
) -> Result<E, AggregationError> {
    transcript.absorb_extensions(
        &format!("aerie/falcon/v1/{stage}/round/{round}/message"),
        &message,
    )?;
    Ok(transcript.challenge_extensions(
        &format!("aerie/falcon/v1/{stage}/round/{round}/challenge"),
        1,
    )?[0])
}

/// Verify an ordinary compressed sumcheck, returning its point and terminal
/// claim for relation-specific checks.
pub fn verify_stage(
    messages: &[[E; 3]],
    rounds: usize,
    mut claim: E,
    transcript: &mut impl PioTranscript,
    stage: &str,
) -> Result<(Vec<E>, E), AggregationError> {
    if messages.len() != rounds {
        return Err(AggregationError::MalformedProof(MalformedProof::StageShape));
    }
    let mut point = Vec::with_capacity(rounds);
    for (round, &message) in messages.iter().enumerate() {
        let challenge = append_round(transcript, stage, round, message)?;
        claim = evaluate_compressed(message, claim, challenge);
        point.push(challenge);
    }
    Ok((point, claim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{
        relations::quartic::QuarticPolynomial,
        stage1::Stage1Proof,
        stage2::{range_leaf, Stage2Proof},
        stage3::Stage3Proof,
    };
    use jolt_field::One;

    #[test]
    fn stage_shapes_match_wire_proof_fields() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            let shapes = StageShapes::for_degree(degree);
            let basis16_sources = range_leaf::instances(degree)
                .iter()
                .filter(|(_, polynomial)| *polynomial == QuarticPolynomial::High)
                .count();
            assert_eq!(shapes.basis16_sources, basis16_sources);
            // The wire codec infers array widths from these proof fields.
            let stage1 = Stage1Proof {
                rounds: vec![Default::default()],
                basis16_claims: vec![Default::default(); basis16_sources],
                projection_claims: Default::default(),
            };
            let stage2 = Stage2Proof {
                rounds: vec![Default::default()],
                folded_digits: Default::default(),
            };
            let stage3 = Stage3Proof {
                rounds: vec![Default::default()],
                terminal_evaluations: Default::default(),
            };
            assert_eq!(
                shapes.stages.map(|stage| stage.round_message_width),
                [
                    stage1.rounds[0].len(),
                    stage2.rounds[0].len(),
                    stage3.rounds[0].len()
                ]
            );
            assert_eq!(
                shapes.stages.map(|stage| stage.terminal_count),
                [
                    stage1
                        .basis16_claims
                        .iter()
                        .map(|claims| claims.len())
                        .sum::<usize>()
                        + stage1.projection_claims.len(),
                    stage2.folded_digits.len(),
                    stage3.terminal_evaluations.len(),
                ]
            );
        }
    }

    #[test]
    fn inactive_rounds_keep_the_local_sum_and_apply_the_terminal_scale() {
        let half = E::from_u64(2).inverse().unwrap();
        let local_sum = E::from_u64(29);
        let mut claim = local_sum;
        for challenge in [E::from_u64(7), E::from_u64(19), E::from_u64(31)] {
            let message = inactive_round(claim);
            assert_eq!(evaluate_compressed(message, claim, E::zero()), claim * half);
            assert_eq!(evaluate_compressed(message, claim, E::one()), claim * half);
            claim = evaluate_compressed(message, claim, challenge);
        }
        assert_eq!(claim, local_sum * half * half * half);
    }
}
