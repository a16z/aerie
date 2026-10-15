//! PCS-backed witness-soundness tests.

use aerie_invariants::invariants::{
    BatchColumn, OutputLeg, ProofField, ProofStage, SourceCell, WitnessSoundness, WitnessTamper,
};
use aerie_invariants::Invariant;

use super::{check, fixture, original_a0};
use crate::support::{assert_holds, pcs_guard};

#[test]
fn out_of_alphabet_source_cells_are_rejected() {
    for parameter_set in [9, 10] {
        let _pcs = pcs_guard();
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::SourceCell {
                    table: SourceCell::A0,
                    index: 5,
                    value: 4,
                }],
            ),
            "a0 = 4 (above [-4,3])",
        );
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::SourceCell {
                    table: SourceCell::A0,
                    index: 5,
                    value: -5,
                }],
            ),
            "a0 = -5 (below [-4,3])",
        );
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::SourceCell {
                    table: SourceCell::A1,
                    index: 5,
                    value: 8,
                }],
            ),
            "a1 = 8 (above [-8,7])",
        );
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::SourceCell {
                    table: SourceCell::Sigma,
                    index: 21,
                    value: 0,
                }],
            ),
            "sigma = 0 (outside {-1,1})",
        );
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::SourceCell {
                    table: SourceCell::A2,
                    index: 34,
                    value: 8,
                }],
            ),
            "a2 = 8 (above [-8,7])",
        );
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::SourceCell {
                    table: SourceCell::U0,
                    index: 0,
                    value: 4,
                }],
            ),
            "u0 = 4 (above [-4,3])",
        );
    }
}

#[test]
fn predicate_failing_outputs_are_rejected() {
    for parameter_set in [9, 10] {
        let _pcs = pcs_guard();
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::Output {
                    leg: OutputLeg::Z,
                    index: 3,
                    delta: 1,
                }],
            ),
            "z[3] += 1 (off a multiple of q)",
        );
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::Output {
                    leg: OutputLeg::Y,
                    index: 0,
                    delta: 1 << 40,
                }],
            ),
            "y[0] += 2^40 (past the norm cap)",
        );
    }
}

#[test]
fn out_of_range_batch_cells_are_rejected() {
    for parameter_set in [9, 10] {
        let _pcs = pcs_guard();
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::BatchCell {
                    column: BatchColumn::S2,
                    index: 9,
                    value: 2048,
                }],
            ),
            "s2 = 2048 (past the Falcon limit)",
        );
        // Moving s1 by q keeps the residual divisible by q (the re-derived
        // quotient shifts by one) but leaves the Falcon norm range.
        let s1 = i64::from(fixture(parameter_set).s1[9]);
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::BatchCell {
                    column: BatchColumn::S1,
                    index: 9,
                    value: s1 + 12_289,
                }],
            ),
            "s1 + q (congruence kept, past the norm bound)",
        );
    }
}

#[test]
fn in_range_ring_and_norm_slack_failures_reach_the_pcs_prover() {
    for parameter_set in [9, 10] {
        let _pcs = pcs_guard();
        let batch = fixture(parameter_set);
        for (column, value) in [
            (BatchColumn::S1, i64::from(batch.s1[9]) + 1),
            (BatchColumn::S2, i64::from(batch.s2[9]) + 1),
            (BatchColumn::Slack, i64::from(batch.slack[9]) + 1),
        ] {
            assert_holds(
                check(
                    parameter_set,
                    &[WitnessTamper::BatchCell {
                        column,
                        index: 9,
                        value,
                    }],
                ),
                &format!("in-range {column:?} identity failure"),
            );
        }
    }
}

#[test]
fn all_1024_row_is_an_in_scope_encodability_failure() {
    for parameter_set in [9, 10] {
        let _pcs = pcs_guard();
        let candidate = (0..(1 << parameter_set))
            .map(|index| WitnessTamper::BatchCell {
                column: BatchColumn::S2,
                index,
                value: 1024,
            })
            .collect::<Vec<_>>();
        assert_holds(
            check(parameter_set, &candidate),
            "all-1024 row exceeds the parameter-set unary budget",
        );
    }
}

#[test]
fn derived_norm_bounds_are_in_scope() {
    for parameter_set in [9, 10] {
        let _pcs = pcs_guard();
        let outside_bound = match parameter_set {
        9 => <aerie_falcon::Falcon512 as aerie_falcon_statement::DirectParams>::NORM_COEFFICIENT_ABS_BOUND + 1,
        10 => <aerie_falcon::Falcon1024 as aerie_falcon_statement::DirectParams>::NORM_COEFFICIENT_ABS_BOUND + 1,
        _ => unreachable!(),
    };
        for (column, value) in [
            (BatchColumn::S1, outside_bound),
            (BatchColumn::Slack, -outside_bound),
        ] {
            assert_holds(
                check(
                    parameter_set,
                    &[WitnessTamper::BatchCell {
                        column,
                        index: 0,
                        value,
                    }],
                ),
                &format!("{column:?} exceeds derived bound"),
            );
        }
    }
}

#[test]
fn in_alphabet_source_and_predicate_passing_output_failures_are_in_scope() {
    for parameter_set in [9, 10] {
        let _pcs = pcs_guard();
        let original = original_a0(parameter_set, 5);
        let changed = if original == 3 { 2 } else { original + 1 };
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::SourceCell {
                    table: SourceCell::A0,
                    index: 5,
                    value: changed,
                }],
            ),
            "in-alphabet source differs from decoded s2",
        );
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::SourceCell {
                    table: SourceCell::U0,
                    index: 0,
                    value: 1,
                }],
            ),
            "in-alphabet budget misdeclares the full slack",
        );
        // A synthetic row has all unary parts zero, so canonical u0 is -4.
        // Changing it to 1 above deterministically violates the budget identity.

        // Outputs that pass both verifier predicates but are not the seeded
        // projection: z moved by q, and y moved by one.
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::Output {
                    leg: OutputLeg::Z,
                    index: 3,
                    delta: 12_289,
                }],
            ),
            "z[3] += q (predicate-passing)",
        );
        assert_holds(
            check(
                parameter_set,
                &[WitnessTamper::Output {
                    leg: OutputLeg::Y,
                    index: 7,
                    delta: 1,
                }],
            ),
            "y[7] += 1 (predicate-passing for this small fixture)",
        );
    }
}

#[test]
fn full_width_claim_deltas_can_accompany_a_witness_defect() {
    for parameter_set in [9, 10] {
        let _pcs = pcs_guard();
        assert_holds(
            check(
                parameter_set,
                &[
                    WitnessTamper::SourceCell {
                        table: SourceCell::A0,
                        index: 5,
                        value: 8,
                    },
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage2,
                        field: ProofField::FoldedDigits,
                        slot: 0,
                        delta: i64::MIN,
                    },
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage2,
                        field: ProofField::FoldedDigits,
                        slot: 0,
                        delta: i64::MAX,
                    },
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage2,
                        field: ProofField::FoldedDigits,
                        slot: 0,
                        delta: 1,
                    },
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage1,
                        field: ProofField::Rounds,
                        slot: 0,
                        delta: 1,
                    },
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage1,
                        field: ProofField::Basis16Claims,
                        slot: 0,
                        delta: 1,
                    },
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage1,
                        field: ProofField::ProjectionClaims,
                        slot: 0,
                        delta: 1,
                    },
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage2,
                        field: ProofField::Rounds,
                        slot: 0,
                        delta: 1,
                    },
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage3,
                        field: ProofField::Rounds,
                        slot: 0,
                        delta: 1,
                    },
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage3,
                        field: ProofField::TerminalEvaluations,
                        slot: 0,
                        delta: 1,
                    },
                ],
            ),
            "current stage fields and full signed deltas accompany a surviving witness defect",
        );
    }
}

#[test]
fn coordinated_witness_defect_is_rejected() {
    for parameter_set in [9, 10] {
        let _pcs = pcs_guard();
        // The bad cell survives an in-alphabet edit at a neighboring index.
        assert_holds(
            check(
                parameter_set,
                &[
                    WitnessTamper::SourceCell {
                        table: SourceCell::A0,
                        index: 5,
                        value: 8,
                    },
                    WitnessTamper::SourceCell {
                        table: SourceCell::A0,
                        index: 6,
                        value: 3,
                    },
                ],
            ),
            "masked out-of-alphabet cell",
        );
    }
}

// Every candidate carries a surviving witness defect and must be rejected
// during proving or verification.
#[test]
fn invalid_witness_regression_battery_is_not_accepted() {
    for parameter_set in [9, 10] {
        use WitnessTamper::{Output, ProofClaim, SourceCell as Src};
        let _pcs = pcs_guard();

        let src = |index, value| Src {
            table: SourceCell::A0,
            index,
            value,
        };
        let battery: Vec<(Vec<WitnessTamper>, &str)> = vec![
            (
                vec![src(5, 8), src(5, 3), src(7, 9)],
                "overwrite plus a surviving bad cell",
            ),
            (
                vec![
                    src(5, 8),
                    ProofClaim {
                        stage: ProofStage::Stage2,
                        field: ProofField::FoldedDigits,
                        slot: 0,
                        delta: 1,
                    },
                ],
                "bad cell plus norm-scalar bump",
            ),
            (
                vec![
                    src(5, 3),
                    Output {
                        leg: OutputLeg::Z,
                        index: 3,
                        delta: 12_289,
                    },
                ],
                "output defect under source noise",
            ),
        ];

        for (input, label) in battery {
            assert_holds(
                check(parameter_set, &input),
                &format!("parameter_set {parameter_set}: {label}"),
            );
        }
    }
}

#[test]
fn benign_seed_corpus_holds() {
    let inv = super::invariant();
    let seeds = <WitnessSoundness as Invariant>::seed_corpus(inv);
    assert_eq!(seeds.len(), 14, "total seed corpus members");
    for parameter_set in [9, 10] {
        let _pcs = pcs_guard();
        assert_eq!(
            seeds
                .iter()
                .filter(|seed| seed.parameter_set == Some(parameter_set))
                .count(),
            7,
            "parameter_set {parameter_set}: seed corpus members",
        );
        for (index, seed) in seeds
            .iter()
            .enumerate()
            .filter(|(_, seed)| seed.parameter_set == Some(parameter_set))
        {
            assert_holds(
                inv.check(seed),
                &format!("parameter_set {parameter_set}: seed {index}"),
            );
        }
    }
}
