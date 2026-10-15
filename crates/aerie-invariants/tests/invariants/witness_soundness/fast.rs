//! Fast witness-soundness tests.

use aerie_invariants::invariants::{
    BatchColumn, OutputLeg, ProofField, ProofStage, SourceCell, WitnessTamper,
};
use aerie_invariants::{Invariant, Outcome};

use super::{check, fixture, original_a0};
use crate::support::assert_invalid;

#[test]
fn unchanged_or_equivalent_witnesses_are_out_of_scope() {
    for parameter_set in [9, 10] {
        let batch = fixture(parameter_set);
        assert_invalid(
            check(
                parameter_set,
                &[
                    WitnessTamper::BatchCell {
                        column: BatchColumn::S2,
                        index: 9,
                        value: i64::from(batch.s2[9]),
                    },
                    WitnessTamper::SourceCell {
                        table: SourceCell::A0,
                        index: 5,
                        value: original_a0(parameter_set, 5),
                    },
                ],
            ),
            "unchanged witness",
        );
        assert_invalid(
            check(
                parameter_set,
                &[WitnessTamper::BatchCell {
                    column: BatchColumn::Slack,
                    index: 0,
                    value: -i64::from(batch.slack[0]),
                }],
            ),
            "equivalent slack norm witness",
        );
    }
}

#[test]
fn restored_batch_identity_is_out_of_scope() {
    for parameter_set in [9, 10] {
        let batch = fixture(parameter_set);
        assert_invalid(
            check(
                parameter_set,
                &[
                    WitnessTamper::BatchCell {
                        column: BatchColumn::S1,
                        index: 9,
                        value: i64::from(batch.s1[9]) + 1,
                    },
                    WitnessTamper::BatchCell {
                        column: BatchColumn::S1,
                        index: 9,
                        value: i64::from(batch.s1[9]),
                    },
                ],
            ),
            "restored ring identity",
        );
    }
}

#[test]
fn malformed_output_indices_are_out_of_scope_even_with_a_batch_defect() {
    for parameter_set in [9, 10] {
        for (leg, index, label) in [
            (OutputLeg::Y, 156, "y index past the 156 norm outputs"),
            (OutputLeg::Z, 31, "z index past the 31 residual outputs"),
        ] {
            assert_invalid(
                check(
                    parameter_set,
                    &[
                        WitnessTamper::BatchCell {
                            column: BatchColumn::S2,
                            index: 9,
                            value: 2048,
                        },
                        WitnessTamper::Output {
                            leg,
                            index,
                            delta: 1,
                        },
                    ],
                ),
                label,
            );
        }
    }
}

#[test]
fn proof_only_tampers_are_out_of_scope() {
    for parameter_set in [9, 10] {
        assert_invalid(
            check(
                parameter_set,
                &[WitnessTamper::ProofClaim {
                    stage: ProofStage::Stage1,
                    field: ProofField::ProjectionClaims,
                    slot: 0,
                    delta: 1,
                }],
            ),
            "Stage1 projection_claims[0] += 1",
        );
        assert_invalid(
            check(
                parameter_set,
                &[WitnessTamper::ProofClaim {
                    stage: ProofStage::Stage2,
                    field: ProofField::FoldedDigits,
                    slot: 0,
                    delta: 3,
                }],
            ),
            "Stage2 folded_digits[0] += 3",
        );
        assert_invalid(
            check(
                parameter_set,
                &[WitnessTamper::ProofClaim {
                    stage: ProofStage::Stage3,
                    field: ProofField::TerminalEvaluations,
                    slot: 2,
                    delta: 1,
                }],
            ),
            "Stage3 terminal_evaluations[2] += 1",
        );
        assert_invalid(
            check(
                parameter_set,
                &[WitnessTamper::ProofClaim {
                    stage: ProofStage::Stage1,
                    field: ProofField::FoldedDigits,
                    slot: 0,
                    delta: 1,
                }],
            ),
            "field does not belong to Stage1",
        );
    }
}

#[test]
fn proof_field_slots_match_the_current_stage_shapes() {
    for parameter_set in [9, 10] {
        let rounds = usize::from(parameter_set) + 3;
        let (basis16_last_valid, basis16_first_invalid) = match parameter_set {
            9 => (3, 4),
            10 => (5, 6),
            _ => unreachable!(),
        };
        let mut shapes = vec![
            (ProofStage::Stage1, ProofField::Rounds, rounds * 3 - 1, true),
            (ProofStage::Stage1, ProofField::Rounds, rounds * 3, false),
            (
                ProofStage::Stage1,
                ProofField::Basis16Claims,
                basis16_last_valid,
                true,
            ),
            (
                ProofStage::Stage1,
                ProofField::Basis16Claims,
                basis16_first_invalid,
                false,
            ),
            (ProofStage::Stage1, ProofField::ProjectionClaims, 2, true),
            (ProofStage::Stage1, ProofField::ProjectionClaims, 3, false),
            (ProofStage::Stage2, ProofField::Rounds, rounds * 4 - 1, true),
            (ProofStage::Stage2, ProofField::Rounds, rounds * 4, false),
            (ProofStage::Stage2, ProofField::FoldedDigits, 5, true),
            (ProofStage::Stage2, ProofField::FoldedDigits, 6, false),
            (ProofStage::Stage3, ProofField::Rounds, rounds * 3 - 1, true),
            (ProofStage::Stage3, ProofField::Rounds, rounds * 3, false),
            (ProofStage::Stage3, ProofField::TerminalEvaluations, 8, true),
            (
                ProofStage::Stage3,
                ProofField::TerminalEvaluations,
                9,
                false,
            ),
        ];
        if parameter_set == 10 {
            shapes.push((ProofStage::Stage1, ProofField::Basis16Claims, 4, true));
        }
        for (stage, field, slot, valid) in shapes {
            let result = check(
                parameter_set,
                &[WitnessTamper::ProofClaim {
                    stage,
                    field,
                    slot,
                    delta: 1,
                }],
            );
            let context = format!("parameter_set {parameter_set}: {stage:?}/{field:?} slot {slot}");
            let outcome =
                result.unwrap_or_else(|error| panic!("{context}: checker failed: {error}"));
            let Outcome::Invalid(reason) = outcome else {
                panic!("{context}: expected Invalid, got {outcome:?}");
            };
            let expected = if valid {
                "the net witness satisfies the semantic predicates; proof-only changes are out of scope"
                    .to_owned()
            } else {
                format!("tamper 0: no proof field {stage:?}/{field:?} at slot {slot}")
            };
            assert_eq!(reason, expected, "{context}");
        }
    }
}

#[test]
fn commitment_swap_is_not_in_the_input_schema() {
    for parameter_set in [9, 10] {
        let input = serde_json::json!({"parameter_set":parameter_set,"tampers":[{"kind": "commitment_swap", "a": 0, "b": 1}]});
        assert!(
            serde_json::from_value::<aerie_invariants::invariants::WitnessInput>(input).is_err(),
            "commitment swaps must not deserialize as witness-soundness tampers"
        );
    }
}

#[test]
fn proof_only_edits_to_current_stage3_terminals_are_out_of_scope() {
    for parameter_set in [9, 10] {
        let candidate = (0..9)
            .map(|slot| WitnessTamper::ProofClaim {
                stage: ProofStage::Stage3,
                field: ProofField::TerminalEvaluations,
                slot,
                delta: i64::try_from(slot).unwrap() - 4,
            })
            .collect::<Vec<_>>();
        assert_invalid(
            check(parameter_set, &candidate),
            "proof-only changes to Stage3 terminal evaluations",
        );
    }
}

#[test]
fn benign_coordinated_tampers_are_out_of_scope() {
    for parameter_set in [9, 10] {
        // Valid Stage1 claim edits without a witness defect are outside this
        // witness-soundness property.
        assert_invalid(
            check(
                parameter_set,
                &[
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage1,
                        field: ProofField::ProjectionClaims,
                        slot: 0,
                        delta: 1,
                    },
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage1,
                        field: ProofField::ProjectionClaims,
                        slot: 1,
                        delta: 1,
                    },
                ],
            ),
            "coordinated Stage1 projection-claim move",
        );
        // A no-op source write plus a zero proof edit is still benign.
        assert_invalid(
            check(
                parameter_set,
                &[
                    WitnessTamper::SourceCell {
                        table: SourceCell::A0,
                        index: 5,
                        value: original_a0(parameter_set, 5),
                    },
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage1,
                        field: ProofField::ProjectionClaims,
                        slot: 0,
                        delta: 0,
                    },
                ],
            ),
            "all-benign coordinated list",
        );
        assert_invalid(check(parameter_set, &[]), "empty tamper list");
    }
}

#[test]
fn cancelling_tampers_are_out_of_scope() {
    for parameter_set in [9, 10] {
        // +1 then -1 on the same claim slot is both proof-only and net-zero.
        assert_invalid(
            check(
                parameter_set,
                &[
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage1,
                        field: ProofField::ProjectionClaims,
                        slot: 0,
                        delta: 1,
                    },
                    WitnessTamper::ProofClaim {
                        stage: ProofStage::Stage1,
                        field: ProofField::ProjectionClaims,
                        slot: 0,
                        delta: -1,
                    },
                ],
            ),
            "net-zero claim delta on one slot",
        );
        // Last-write-wins restores the original decoded source value.
        assert_invalid(
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
                        index: 5,
                        value: original_a0(parameter_set, 5),
                    },
                ],
            ),
            "out-of-alphabet cell restored",
        );
        // Output deltas that cancel leave the honest projection.
        assert_invalid(
            check(
                parameter_set,
                &[
                    WitnessTamper::Output {
                        leg: OutputLeg::Z,
                        index: 4,
                        delta: 12_289,
                    },
                    WitnessTamper::Output {
                        leg: OutputLeg::Z,
                        index: 4,
                        delta: -12_289,
                    },
                ],
            ),
            "net-zero output delta",
        );
    }
}

#[test]
fn unbounded_tamper_lists_are_out_of_scope() {
    for parameter_set in [9, 10] {
        let input = vec![
            WitnessTamper::SourceCell {
                table: SourceCell::A0,
                index: 5,
                value: original_a0(parameter_set, 5),
            };
            1025
        ];
        assert_invalid(
            check(parameter_set, &input),
            "tamper count past the resource cap",
        );
    }
}

#[test]
fn benign_regression_battery_is_out_of_scope() {
    for parameter_set in [9, 10] {
        use WitnessTamper::{BatchCell, ProofClaim, SourceCell as Src};

        let claim = |slot, delta| ProofClaim {
            stage: ProofStage::Stage1,
            field: ProofField::ProjectionClaims,
            slot,
            delta,
        };
        let src = |index, value| Src {
            table: SourceCell::A0,
            index,
            value,
        };
        let battery: Vec<(Vec<WitnessTamper>, &str)> = vec![
            (vec![claim(0, 2), claim(0, -1)], "net +1 on one slot"),
            (
                vec![claim(0, 1), claim(1, 1), claim(2, -2)],
                "three-slot move",
            ),
            (
                vec![
                    claim(0, 5),
                    claim(0, -5),
                    ProofClaim {
                        stage: ProofStage::Stage3,
                        field: ProofField::TerminalEvaluations,
                        slot: 0,
                        delta: 1,
                    },
                ],
                "cancelling pair plus a real terminal bump",
            ),
            (
                vec![BatchCell {
                    column: BatchColumn::S2,
                    index: 9,
                    value: i64::from(fixture(parameter_set).s2[9]),
                }],
                "unchanged s2",
            ),
            (
                vec![
                    src(5, original_a0(parameter_set, 5)),
                    claim(0, 1),
                    claim(0, -1),
                ],
                "all benign",
            ),
        ];

        for (input, label) in battery {
            assert_invalid(check(parameter_set, &input), label);
        }
    }
}

#[test]
fn missing_or_unknown_parameter_sets_are_out_of_scope() {
    let omitted: aerie_invariants::invariants::WitnessInput = serde_json::from_value(
        serde_json::json!({"tampers":[{"kind":"source_cell","table":"a0","index":5,"value":8}]}),
    )
    .unwrap();
    assert_invalid(super::invariant().check(&omitted), "missing parameter set");
    for parameter_set in [0, 8, 11, u8::MAX] {
        assert_invalid(
            check(parameter_set, &omitted.tampers),
            "unknown parameter set",
        );
    }
}
