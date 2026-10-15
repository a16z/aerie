//! Stage two: one normalized shared-equality sumcheck for all range leaves.
pub mod range_leaf;
use super::transcript::PioTranscript;
use super::{stage::StageShapes, stage1::Stage1Output, FalconDegree};
use crate::error::{AggregationError, MalformedProof, RejectionStage};
use crate::protocol::relations::quartic::QuarticPolynomial;
use crate::substrate::OpeningField as E;
use jolt_field::Zero;
use range_leaf::{advance, instances};
pub const GAMMA_SITE: &str = "aerie/falcon/v1/stage2/gamma";
pub const TERMINAL_SITE: &str = "aerie/falcon/v1/stage2/terminal";
#[derive(Clone, Debug, PartialEq)]
pub struct Stage2Proof {
    pub rounds: Vec<[E; 4]>,
    pub folded_digits: [E; 6],
}
#[derive(Clone, Debug, PartialEq)]
pub struct Stage2Output {
    pub point: Vec<E>,
    pub folded_digits: [E; 6],
}
#[doc(hidden)]
#[inline]
#[must_use]
pub fn has_input_shape(degree: FalconDegree, records: usize, stage1: &Stage1Output) -> bool {
    records.is_power_of_two()
        && stage1.point.len() == degree.logn() + records.trailing_zeros() as usize
        && stage1.basis16_claims.len() == StageShapes::for_degree(degree).basis16_sources
}

#[doc(hidden)]
#[inline]
pub fn setup(
    degree: FalconDegree,
    stage1: &Stage1Output,
    t: &mut impl PioTranscript,
) -> Result<(Vec<E>, E), AggregationError> {
    let gamma = t.challenge_extensions(
        GAMMA_SITE,
        StageShapes::for_degree(degree).stages[1].gamma_count,
    )?;
    let claim = stage1
        .basis16_claims
        .iter()
        .flatten()
        .zip(&gamma)
        .fold(E::zero(), |s, (&v, &g)| s + g * v);
    Ok((gamma, claim))
}
pub fn verify(
    degree: FalconDegree,
    records: usize,
    stage1: &Stage1Output,
    proof: &Stage2Proof,
    t: &mut impl PioTranscript,
) -> Result<Stage2Output, AggregationError> {
    if !has_input_shape(degree, records, stage1) {
        return Err(AggregationError::MalformedProof(MalformedProof::StageShape));
    }
    let (gamma, mut claim) = setup(degree, stage1, t)?;
    if proof.rounds.len() != stage1.point.len() {
        return Err(AggregationError::MalformedProof(MalformedProof::StageShape));
    }
    let mut point = Vec::with_capacity(proof.rounds.len());
    for (round, (&message, &alpha)) in proof.rounds.iter().zip(&stage1.point).enumerate() {
        t.absorb_extensions(
            &format!("aerie/falcon/v1/stage2/round/{round}/message"),
            &message,
        )?;
        let challenge = t.challenge_extensions(
            &format!("aerie/falcon/v1/stage2/round/{round}/challenge"),
            1,
        )?[0];
        claim = advance(claim, alpha, message, challenge);
        point.push(challenge);
    }
    if !check_terminal(degree, &gamma, &proof.folded_digits, claim)? {
        return Err(AggregationError::ProofRejected {
            stage: RejectionStage::Stage2,
        });
    }
    t.absorb_extensions(TERMINAL_SITE, &proof.folded_digits)?;
    Ok(Stage2Output {
        point,
        folded_digits: proof.folded_digits,
    })
}
#[doc(hidden)]
#[inline]
pub fn pair_relations<'a>(
    relations: &'a [(usize, QuarticPolynomial)],
    gamma: &'a [E],
) -> Result<impl Iterator<Item = (&'a (usize, QuarticPolynomial), &'a E)> + Clone, AggregationError>
{
    if relations.len() != gamma.len() {
        return Err(AggregationError::internal(
            "range leaf gamma count mismatch",
        ));
    }
    Ok(relations.iter().zip(gamma))
}

#[doc(hidden)]
#[inline]
pub fn check_terminal(
    degree: FalconDegree,
    gamma: &[E],
    values: &[E; 6],
    claim: E,
) -> Result<bool, AggregationError> {
    let relations = instances(degree);
    let expected = pair_relations(&relations, gamma)?.fold(E::zero(), |s, (&(index, q), &g)| {
        s + g * q.evaluate(values[index])
    });
    Ok(claim == expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Zero;
    #[test]
    fn range_leaf_count_matches_stage_shape() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            assert_eq!(
                instances(degree).len(),
                StageShapes::for_degree(degree).stages[1].gamma_count
            );
        }
    }

    #[test]
    fn range_leaf_pairing_rejects_gamma_count_mismatches() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            let relations = instances(degree);
            for count in [relations.len() - 1, relations.len() + 1] {
                let gamma = vec![E::zero(); count];
                assert!(matches!(
                    pair_relations(&relations, &gamma),
                    Err(AggregationError::Backend(_))
                ));
                assert!(matches!(
                    check_terminal(degree, &gamma, &[E::zero(); 6], E::zero()),
                    Err(AggregationError::Backend(_))
                ));
            }
        }
    }

    #[test]
    fn verifier_preserves_transcript_faults_and_checks_shapes_explicitly() {
        use crate::protocol::transcript::{ChallengeSource, TranscriptOrderError};

        struct RegistryFault;
        impl ChallengeSource for RegistryFault {
            fn challenge_extensions(
                &mut self,
                _site: &str,
                _count: usize,
            ) -> Result<Vec<E>, TranscriptOrderError> {
                Err(TranscriptOrderError::EventOrder { index: 17 })
            }
        }
        impl PioTranscript for RegistryFault {
            fn absorb_proof(
                &mut self,
                _site: &str,
                _atom_count: usize,
                _payload: &[u8],
            ) -> Result<(), TranscriptOrderError> {
                unreachable!("the initial challenge fails before absorbing proof messages")
            }
        }

        let degree = FalconDegree::Falcon512;
        let mut stage1 = Stage1Output {
            point: vec![E::zero(); degree.logn() + 1],
            basis16_claims: vec![[E::zero(); 2]; 2],
            projection_point: Vec::new(),
            slack_projection_point: Vec::new(),
            projection_claims: [E::zero(); 3],
        };
        let proof = Stage2Proof {
            rounds: vec![[E::zero(); 4]; stage1.point.len()],
            folded_digits: [E::zero(); 6],
        };
        let error = verify(degree, 2, &stage1, &proof, &mut RegistryFault).unwrap_err();
        let cause = std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<TranscriptOrderError>());
        assert!(matches!(
            cause,
            Some(TranscriptOrderError::EventOrder { index: 17 })
        ));

        stage1.point.pop();
        let error = verify(degree, 2, &stage1, &proof, &mut RegistryFault).unwrap_err();
        assert!(matches!(
            error,
            AggregationError::MalformedProof(MalformedProof::StageShape)
        ));
    }
}
