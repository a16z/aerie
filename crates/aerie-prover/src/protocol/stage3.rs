//! Stage three: source links, Falcon predicates, and opening-point reductions.

use crate::error::AggregationError;

use super::transcript::PioTranscript;
use crate::protocol::{
    source_tables::{PreparedCoefficients, SourceTables},
    stage::{append_round, evaluate_compressed},
    stage1::Stage1Output,
    stage2::Stage2Output,
    FalconDegree, FalconWitness,
};
use crate::substrate::OpeningField as E;

mod compact;
#[cfg(not(test))]
pub use aerie_falcon::protocol::stage3::encoding_budget;
#[cfg(test)]
pub mod encoding_budget {
    pub use aerie_falcon::protocol::stage3::encoding_budget::*;
    #[cfg(test)]
    mod tests {
        use crate::protocol::{stage3::fused_rounds::tests::check_relations, FalconDegree};

        #[test]
        fn coefficient_and_budget_rounds_match_direct_tables_with_eq_endpoints() {
            for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
                for mode in [0, 1, 4, 5] {
                    check_relations(&[5, 6], degree, 4, mode);
                }
            }
        }
    }
}

mod fused_rounds;
pub mod norm_equality;
pub mod range_source_link;
pub mod ring_functional;
#[cfg(all(not(test), feature = "internals"))]
pub use aerie_falcon::protocol::stage3::s1_reduction;
#[cfg(test)]
pub mod s1_reduction {
    #[cfg(feature = "internals")]
    pub use aerie_falcon::protocol::stage3::s1_reduction::*;
    #[cfg(test)]
    mod tests {
        use crate::protocol::{stage3::fused_rounds::tests::check_relations, FalconDegree};

        #[test]
        fn direct_rounds_match_the_native_relation_at_equality_endpoints() {
            for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
                for mode in 0..4 {
                    check_relations(&[8], degree, 4, mode);
                }
            }
        }
    }
}

#[cfg(all(not(test), feature = "internals"))]
pub use aerie_falcon::protocol::stage3::sign_check;
#[cfg(test)]
pub mod sign_check {
    #[cfg(feature = "internals")]
    pub use aerie_falcon::protocol::stage3::sign_check::*;
    #[cfg(test)]
    mod tests {
        use crate::protocol::{stage3::fused_rounds::tests::check_relations, FalconDegree};

        #[test]
        fn direct_rounds_match_the_native_relation_at_equality_endpoints() {
            for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
                for mode in 0..4 {
                    check_relations(&[2], degree, 4, mode);
                }
            }
        }
    }
}

#[cfg(all(not(test), feature = "internals"))]
pub use aerie_falcon::protocol::stage3::slack_reduction;
#[cfg(test)]
pub mod slack_reduction {
    #[cfg(feature = "internals")]
    pub use aerie_falcon::protocol::stage3::slack_reduction::*;
    #[cfg(test)]
    mod tests {
        use crate::protocol::{stage3::fused_rounds::tests::check_relations, FalconDegree};

        #[test]
        fn direct_rounds_match_the_native_relation_at_equality_endpoints() {
            for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
                for mode in 0..4 {
                    check_relations(&[9], degree, 4, mode);
                }
            }
        }
    }
}

#[cfg(test)]
mod source_folds;

#[cfg(any(test, feature = "internals"))]
pub use aerie_falcon::protocol::stage3::verify;
pub use aerie_falcon::protocol::stage3::{
    gammas, has_point_shapes, has_stage1_shape, inverse_power_of_two, stage_claim, Stage3Output,
    Stage3Proof, TERMINAL_SITE,
};
#[cfg(feature = "internals")]
pub use aerie_falcon::protocol::stage3::{GAMMA_SITE, INSTANCE_COUNT, TERMINAL_EVALUATION_COUNT};

fn check_prover_shape(
    degree: FalconDegree,
    records: usize,
    tables: &SourceTables,
    stage1: &Stage1Output,
) -> Result<(), AggregationError> {
    if !has_stage1_shape(degree, records, stage1) {
        return Err(AggregationError::internal(
            "stage-three stage-one shape mismatch",
        ));
    }
    if tables.degree() != degree.degree() || tables.records() != records {
        return Err(AggregationError::internal(
            "stage-three source tables do not match the batch",
        ));
    }
    Ok(())
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
) -> Result<Prepared, AggregationError> {
    let degree = FalconDegree::from_degree(batch.degree())?;
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

/// Prove seven relation families as twelve independently weighted instances.
/// Canonical gamma order is six range links, sign, norm, encoding budget,
/// ring functional, s1 reduction, then slack reduction. The proof ends in
/// nine terminal evaluations.
#[cfg(any(test, feature = "internals"))]
pub fn prove(
    batch: &FalconWitness,
    tables: &SourceTables,
    stage1: &Stage1Output,
    stage2: &Stage2Output,
    transcript: &mut impl PioTranscript,
) -> Result<(Stage3Proof, Stage3Output), AggregationError> {
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
) -> Result<(Stage3Proof, Stage3Output), AggregationError> {
    let degree = FalconDegree::from_degree(batch.degree())?;
    let records = batch.padded_rows;
    tracing::info_span!("stage3_driver_setup", phase = "validate").in_scope(
        || -> Result<(), AggregationError> {
            check_prover_shape(degree, records, tables, stage1)?;
            if !has_point_shapes(degree, records, stage1, stage2) {
                return Err(AggregationError::internal(
                    "stage-three stage-two shape mismatch",
                ));
            }
            if prepared.degree != degree || prepared.records != records {
                return Err(AggregationError::internal(
                    "stage-three preparation does not match the batch",
                ));
            }
            Ok(())
        },
    )?;

    let gamma = tracing::info_span!("stage3_driver_transcript", phase = "gamma_sampling")
        .in_scope(|| gammas(degree, transcript))?;
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{projection, source_tables::SourceTables, transcript::TestTranscript};
    use aerie_falcon_native::Falcon512;
    use aerie_falcon_witness::DirectRow;

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
