//! Stage three: source links, Falcon predicates, and opening-point reductions.

use crate::error::{AggregationError, MalformedProof, RejectionStage, StatementMismatch};
use jolt_field::{Field, One, Ring, Zero};

use super::transcript::PioTranscript;
use crate::protocol::{
    stage::{verify_stage, StageShapes},
    stage1::Stage1Output,
    stage2::Stage2Output,
    FalconDegree,
};
use crate::substrate::OpeningField as E;

pub mod encoding_budget;
pub mod norm_equality;
pub mod range_source_link;
pub mod ring_functional;
pub mod s1_reduction;
pub mod sign_check;
pub mod slack_reduction;

pub const GAMMA_SITE: &str = "aerie/falcon/v1/stage3/gamma";
pub const TERMINAL_SITE: &str = "aerie/falcon/v1/stage3/terminal";
pub const TERMINAL_EVALUATION_COUNT: usize = 9;
pub const INSTANCE_COUNT: usize = 12;
const _: () = {
    assert!(
        StageShapes::for_degree(FalconDegree::Falcon512).stages[2].gamma_count == INSTANCE_COUNT
    );
    assert!(
        StageShapes::for_degree(FalconDegree::Falcon1024).stages[2].gamma_count == INSTANCE_COUNT
    );
};

/// Stage-three round messages and nine committed opening values.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage3Proof {
    pub rounds: Vec<[E; 3]>,
    pub terminal_evaluations: [E; TERMINAL_EVALUATION_COUNT],
}

/// Canonical stage-three point and committed terminal evaluations.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage3Output {
    pub point: Vec<E>,
    pub terminal_evaluations: [E; TERMINAL_EVALUATION_COUNT],
}

#[doc(hidden)]
#[inline]
#[must_use]
pub fn inverse_power_of_two(power: usize) -> E {
    let half = E::from_u64(2).inverse().expect("two is invertible");
    (0..power).fold(E::one(), |value, _| value * half)
}

#[doc(hidden)]
#[inline]
pub fn gammas(
    degree: FalconDegree,
    transcript: &mut impl PioTranscript,
) -> Result<[E; INSTANCE_COUNT], AggregationError> {
    Ok(transcript
        .challenge_extensions(
            GAMMA_SITE,
            StageShapes::for_degree(degree).stages[2].gamma_count,
        )?
        .try_into()
        .expect("registry fixes gamma width"))
}

#[doc(hidden)]
#[inline]
#[must_use]
pub fn has_stage1_shape(degree: FalconDegree, records: usize, stage1: &Stage1Output) -> bool {
    if !records.is_power_of_two() {
        return false;
    }
    let rounds = degree.logn() + records.trailing_zeros() as usize;
    stage1.point.len() == rounds
        && stage1.projection_point.len() == rounds
        && stage1.slack_projection_point.len() == rounds - degree.logn() + 2
}

#[doc(hidden)]
#[inline]
#[must_use]
pub fn has_point_shapes(
    degree: FalconDegree,
    records: usize,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
) -> bool {
    has_stage1_shape(degree, records, stage1) && stage2.point.len() == stage1.point.len()
}

#[doc(hidden)]
#[inline]
#[must_use]
pub fn stage_claim(
    gamma: &[E; INSTANCE_COUNT],
    stage2: &Stage2Output,
    degree: FalconDegree,
    live_rows: usize,
    stage1: &Stage1Output,
    target: E,
) -> E {
    let range_claim = stage2
        .folded_digits
        .iter()
        .zip(&gamma[..6])
        .fold(E::zero(), |sum, (&claim, &weight)| sum + claim * weight);
    range_claim
        + gamma[7]
            * E::from_u64(degree.beta_squared())
            * crate::protocol::algebra::live_prefix_evaluation(
                live_rows,
                &stage2.point[degree.logn()..],
            )
        + gamma[8] * encoding_budget::input_claim(degree)
        + gamma[9] * (target + stage1.projection_claims[2] - stage1.projection_claims[0])
        + gamma[10] * stage1.projection_claims[0]
        + gamma[11] * stage1.projection_claims[1]
}

/// Verify the stage-three sumcheck and its nine committed terminal values,
/// adding the streamed public-target evaluation time to
/// `target_evaluation_ns`.
pub fn verify<C>(
    records: usize,
    public: &crate::protocol::public_inputs::PublicInputs<'_, C>,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    proof: &Stage3Proof,
    transcript: &mut impl PioTranscript,
    target_evaluation_ns: &mut u128,
) -> Result<Stage3Output, AggregationError>
where
    C: Copy + Sync + Into<i64>,
{
    let degree = public.degree;
    let live_rows = public.records;
    let h = public.h;
    let targets = public.t;
    if !has_point_shapes(degree, records, stage1, stage2) {
        return Err(AggregationError::MalformedProof(MalformedProof::StageShape));
    }
    if live_rows == 0 || live_rows > records {
        return Err(AggregationError::StatementMismatch(
            StatementMismatch::PublicInputShape,
        ));
    }
    h.validate_shape_of(degree.degree(), live_rows, records)?;
    targets.validate_shape_of(degree.degree(), live_rows, records)?;
    if proof.rounds.len() != stage1.point.len() {
        return Err(AggregationError::MalformedProof(MalformedProof::StageShape));
    }

    let gamma = gammas(degree, transcript)?;
    let target = aerie_falcon_statement::staged!(
        "v1.stages.verify.targets",
        *target_evaluation_ns,
        crate::protocol::ring::target_evaluation(
            degree,
            records,
            live_rows,
            targets,
            &stage1.projection_point,
        )
    );
    let claim = stage_claim(&gamma, stage2, degree, live_rows, stage1, target);
    let (point, final_claim) = verify_stage(
        &proof.rounds,
        stage1.point.len(),
        claim,
        transcript,
        "stage3",
    )?;
    transcript.absorb_extensions(TERMINAL_SITE, &proof.terminal_evaluations)?;

    let range_gammas: [E; 6] = gamma[..6].try_into().expect("six range coefficients");
    let expected = range_source_link::terminal(
        &range_gammas,
        stage2,
        &proof.terminal_evaluations,
        &point,
        degree,
    ) + sign_check::terminal(
        gamma[6],
        &stage2.point,
        &point,
        proof.terminal_evaluations[5],
    ) + norm_equality::terminal(
        gamma[7],
        degree,
        &stage2.point,
        &proof.terminal_evaluations,
        &point,
    ) + encoding_budget::terminal(
        gamma[8],
        degree,
        &stage2.point,
        &proof.terminal_evaluations,
        &point,
    ) + ring_functional::terminal(
        gamma[9],
        h,
        records,
        &stage1.projection_point,
        &point,
        &proof.terminal_evaluations,
    ) + s1_reduction::terminal(
        gamma[10],
        &stage1.projection_point,
        &point,
        &proof.terminal_evaluations,
    ) + slack_reduction::terminal(
        gamma[11],
        degree,
        &stage1.slack_projection_point,
        &point,
        &proof.terminal_evaluations,
    );
    if final_claim != expected {
        return Err(AggregationError::ProofRejected {
            stage: RejectionStage::Stage3,
        });
    }
    Ok(Stage3Output {
        point,
        terminal_evaluations: proof.terminal_evaluations,
    })
}
