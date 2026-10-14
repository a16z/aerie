//! Stage three: source links, Falcon predicates, and opening-point reductions.

use akita_error::AkitaError;
use jolt_field::{Field, One, Ring, Zero};

use super::transcript::PioTranscript;
use crate::protocol::{
    source_tables::{PreparedCoefficients, SourceTables},
    stage::{append_round, evaluate_compressed, verify_stage},
    stage1::Stage1Output,
    stage2::Stage2Output,
    FalconDegree, FalconWitness,
};
use crate::substrate::OpeningField as E;

mod compact;
pub mod encoding_budget;
mod fused_rounds;
pub mod norm_equality;
pub mod range_source_link;
pub mod ring_functional;
pub mod s1_reduction;
pub mod sign_check;
pub mod slack_reduction;
#[cfg(test)]
mod source_folds;

const GAMMA_SITE: &str = "aerie/falcon/v2/stage3/gamma";
const TERMINAL_SITE: &str = "aerie/falcon/v2/stage3/terminal";
const TERMINAL_EVALUATION_COUNT: usize = 9;
const INSTANCE_COUNT: usize = 12;

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

fn invalid(message: &str) -> AkitaError {
    AkitaError::InvalidInput(message.to_owned())
}

fn inverse_power_of_two(power: usize) -> E {
    let half = E::from_u64(2).inverse().expect("two is invertible");
    (0..power).fold(E::one(), |value, _| value * half)
}

fn gammas(transcript: &mut impl PioTranscript) -> Result<[E; INSTANCE_COUNT], AkitaError> {
    Ok(transcript
        .challenge_extensions(GAMMA_SITE, INSTANCE_COUNT)?
        .try_into()
        .expect("registry fixes gamma width"))
}

fn check_stage1_shape(
    degree: FalconDegree,
    records: usize,
    stage1: &Stage1Output,
) -> Result<(), AkitaError> {
    if !records.is_power_of_two() || records == 0 {
        return Err(invalid(
            "stage-three records must be a nonzero power of two",
        ));
    }
    let rounds = degree.logn() + records.trailing_zeros() as usize;
    if stage1.point.len() != rounds
        || stage1.projection_point.len() != rounds
        || stage1.slack_projection_point.len() != rounds - degree.logn() + 2
        || stage1.projection_claims.len() != 3
    {
        return Err(invalid("stage-three stage-one shape is invalid"));
    }
    Ok(())
}

fn check_prover_shape(
    degree: FalconDegree,
    records: usize,
    tables: &SourceTables,
    stage1: &Stage1Output,
) -> Result<(), AkitaError> {
    check_stage1_shape(degree, records, stage1)?;
    if tables.degree() != degree.degree() || tables.records() != records {
        return Err(invalid("stage-three source tables do not match the batch"));
    }
    Ok(())
}

fn check_points(
    degree: FalconDegree,
    records: usize,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
) -> Result<(), AkitaError> {
    check_stage1_shape(degree, records, stage1)?;
    let rounds = degree.logn() + records.trailing_zeros() as usize;
    if stage2.point.len() != rounds || stage2.folded_digits.len() != 6 {
        return Err(invalid("stage-three stage-two shape is invalid"));
    }
    Ok(())
}

fn stage_claim(
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

pub(crate) struct Prepared {
    degree: FalconDegree,
    records: usize,
    target: E,
    magnitude: Vec<i16>,
    ring: ring_functional::Prover,
}

/// Build the stage-three witness-only ring table and public target evaluation.
/// Both depend on stage one, but neither depends on the stage-two transcript.
pub(crate) fn prepare(
    batch: &FalconWitness,
    tables: &SourceTables,
    stage1: &Stage1Output,
) -> Result<Prepared, AkitaError> {
    let degree = FalconDegree::from_degree(batch.degree()).map_err(invalid)?;
    let records = batch.padded_rows;
    check_prover_shape(degree, records, tables, stage1)?;
    let target =
        tracing::info_span!("stage3_driver_setup", phase = "target_evaluation").in_scope(|| {
            crate::protocol::ring::target_evaluation_i16(
                degree,
                &batch.t,
                batch.live_rows,
                records,
                &stage1.projection_point,
            )
        });
    let magnitude = norm_equality::magnitude_table(tables);
    let ring = tracing::info_span!("ring_functional_setup").in_scope(|| {
        ring_functional::Prover::new(
            batch,
            tables,
            &magnitude,
            degree,
            &stage1.projection_point,
            target + stage1.projection_claims[2] - stage1.projection_claims[0],
        )
    })?;
    Ok(Prepared {
        degree,
        records,
        target,
        magnitude,
        ring,
    })
}

/// Prove the seven named stage-three relations in canonical gamma order.
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn prove(
    batch: &FalconWitness,
    tables: &SourceTables,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    transcript: &mut impl PioTranscript,
) -> Result<(Stage3Proof, Stage3Output), AkitaError> {
    let coefficients = PreparedCoefficients::new(tables)?;
    let prepared = prepare(batch, tables, stage1)?;
    prove_prepared(
        batch,
        tables,
        &coefficients,
        stage1,
        stage2,
        prepared,
        transcript,
    )
}

pub(crate) fn prove_prepared(
    batch: &FalconWitness,
    tables: &SourceTables,
    coefficients: &PreparedCoefficients<'_>,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    prepared: Prepared,
    transcript: &mut impl PioTranscript,
) -> Result<(Stage3Proof, Stage3Output), AkitaError> {
    let degree = FalconDegree::from_degree(batch.degree()).map_err(invalid)?;
    let records = batch.padded_rows;
    tracing::info_span!("stage3_driver_setup", phase = "validate").in_scope(
        || -> Result<(), AkitaError> {
            check_prover_shape(degree, records, tables, stage1)?;
            check_points(degree, records, stage1, stage2)?;
            if prepared.degree != degree || prepared.records != records {
                return Err(invalid("stage-three preparation does not match the batch"));
            }
            Ok(())
        },
    )?;

    let gamma = tracing::info_span!("stage3_driver_transcript", phase = "gamma_sampling")
        .in_scope(|| gammas(transcript))?;
    let target = prepared.target;
    let mut running_claim = tracing::info_span!("stage3_driver_claim_combine", phase = "initial")
        .in_scope(|| stage_claim(&gamma, stage2, degree, batch.live_rows, stage1, target));
    let rounds = stage1.point.len();
    let mut engine = tracing::info_span!("stage3_fused_setup").in_scope(|| {
        fused_rounds::Engine::new(batch, tables, coefficients, stage1, stage2, gamma, prepared)
    })?;
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let message = engine.message();
        let challenge = tracing::info_span!("stage3_driver_transcript", phase = "round", round)
            .in_scope(|| append_round(transcript, "stage3", round, message))?;
        running_claim = evaluate_compressed(message, running_claim, challenge);
        tracing::info_span!("stage3_fused_round", round).in_scope(|| engine.bind(challenge))?;
        point.push(challenge);
        messages.push(message);
    }
    debug_assert_eq!(running_claim, engine.claim());
    let terminal_evaluations = engine.terminal_evaluations();
    tracing::info_span!("stage3_driver_transcript", phase = "terminal")
        .in_scope(|| transcript.absorb_extensions(TERMINAL_SITE, &terminal_evaluations))?;
    let proof = Stage3Proof {
        rounds: messages,
        terminal_evaluations,
    };
    let output = Stage3Output {
        point,
        terminal_evaluations,
    };
    Ok((proof, output))
}

/// Verify the stage-three sumcheck and its nine committed terminal values,
/// adding the streamed public-target evaluation time to
/// `target_evaluation_ns`.
#[allow(clippy::too_many_arguments)]
pub fn verify<C>(
    degree: FalconDegree,
    records: usize,
    public: &crate::protocol::public_inputs::PublicInputs<'_, C>,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    proof: &Stage3Proof,
    transcript: &mut impl PioTranscript,
    target_evaluation_ns: &mut u128,
) -> Result<Stage3Output, AkitaError>
where
    C: Copy + Sync + Into<i64>,
{
    let live_rows = public.records;
    let h = public.h;
    let targets = public.t;
    check_points(degree, records, stage1, stage2)?;
    if live_rows == 0 || live_rows > records {
        return Err(invalid("stage-three live row count is out of range"));
    }
    h.validate_shape_of(degree.degree(), live_rows, records)
        .map_err(invalid)?;
    targets
        .validate_shape_of(degree.degree(), live_rows, records)
        .map_err(invalid)?;
    if proof.rounds.len() != stage1.point.len() {
        return Err(invalid("incorrect stage-three round count"));
    }

    let gamma = gammas(transcript)?;
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
        return Err(invalid("stage-three terminal equation failed"));
    }
    Ok(Stage3Output {
        point,
        terminal_evaluations: proof.terminal_evaluations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{projection, source_tables::SourceTables, transcript::TestTranscript};
    use aerie_falcon_native::Falcon512;
    use aerie_falcon_statement::DirectRow;

    #[test]
    fn prepared_and_direct_paths_match_whole_stage() {
        let rows = (0..2)
            .map(|row| DirectRow::<Falcon512>::synthetic(0x51a6_0000 + row).unwrap())
            .collect::<Vec<_>>();
        let mut batch = FalconWitness::from_rows(&rows);
        crate::protocol::pad_witness(&mut batch).unwrap();
        let tables = SourceTables::build(&batch).unwrap();
        let coefficients = PreparedCoefficients::new(&tables).unwrap();
        let projected =
            projection::project_unchecked(&batch, b"stage3 single chunk paths").unwrap();

        let run = |use_prepared| {
            let mut transcript = TestTranscript::new(b"stage3 single chunk paths");
            let (_, first) =
                crate::protocol::stage1::prove(&batch, &tables, &projected, &mut transcript)
                    .unwrap();
            let (_, second) = crate::protocol::stage2::prove(
                FalconDegree::Falcon512,
                &tables,
                &first,
                &mut transcript,
            )
            .unwrap();
            if use_prepared {
                let prepared = prepare(&batch, &tables, &first).unwrap();
                prove_prepared(
                    &batch,
                    &tables,
                    &coefficients,
                    &first,
                    &second,
                    prepared,
                    &mut transcript,
                )
                .unwrap()
            } else {
                prove(&batch, &tables, &first, &second, &mut transcript).unwrap()
            }
        };

        let serial = run(false);
        assert_eq!(run(true), serial);
        #[cfg(feature = "parallel")]
        {
            let forced_parallel = fused_rounds::with_forced_parallel_single_chunk(|| run(false));
            assert_eq!(forced_parallel, serial);
        }
    }
}
