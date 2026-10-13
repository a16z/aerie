//! Fast witness-soundness tests.

use aerie_invariants::invariants::{
    BatchColumn, OutputLeg, ProofField, ProofStage, SourceCell, WitnessTamper,
};
use aerie_invariants::Invariant;

use super::{fixture, invariant, original_a0};
use crate::support::assert_invalid;

#[test]
fn unchanged_or_equivalent_witnesses_are_out_of_scope() {
    let batch = fixture();
    assert_invalid(
        invariant().check(&vec![
            WitnessTamper::BatchCell {
                column: BatchColumn::S2,
                index: 9,
                value: i64::from(batch.s2[9]),
            },
            WitnessTamper::SourceCell {
                table: SourceCell::A0,
                index: 5,
                value: original_a0(5),
            },
        ]),
        "unchanged witness",
    );
    assert_invalid(
        invariant().check(&vec![WitnessTamper::BatchCell {
            column: BatchColumn::Slack,
            index: 0,
            value: -i64::from(batch.slack[0]),
        }]),
        "equivalent slack norm witness",
    );
}

#[test]
fn restored_batch_identity_is_out_of_scope() {
    let batch = fixture();
    assert_invalid(
        invariant().check(&vec![
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
        ]),
        "restored ring identity",
    );
}

#[test]
fn malformed_output_indices_are_out_of_scope_even_with_a_batch_defect() {
    for (leg, index, label) in [
        (OutputLeg::Y, 156, "y index past the 156 norm outputs"),
        (OutputLeg::Z, 31, "z index past the 31 residual outputs"),
    ] {
        assert_invalid(
            invariant().check(&vec![
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
            ]),
            label,
        );
    }
}

#[test]
fn proof_only_tampers_are_out_of_scope() {
    let inv = invariant();
    assert_invalid(
        inv.check(&vec![WitnessTamper::ProofClaim {
            stage: ProofStage::Stage1,
            field: ProofField::ProjectionClaims,
            slot: 0,
            delta: 1,
        }]),
        "Stage1 projection_claims[0] += 1",
    );
    assert_invalid(
        inv.check(&vec![WitnessTamper::ProofClaim {
            stage: ProofStage::Stage2,
            field: ProofField::FoldedDigits,
            slot: 0,
            delta: 3,
        }]),
        "Stage2 folded_digits[0] += 3",
    );
    assert_invalid(
        inv.check(&vec![WitnessTamper::ProofClaim {
            stage: ProofStage::Stage3,
            field: ProofField::TerminalEvaluations,
            slot: 2,
            delta: 1,
        }]),
        "Stage3 terminal_evaluations[2] += 1",
    );
    assert_invalid(
        inv.check(&vec![WitnessTamper::ProofClaim {
            stage: ProofStage::Stage1,
            field: ProofField::FoldedDigits,
            slot: 0,
            delta: 1,
        }]),
        "field does not belong to Stage1",
    );
}

#[test]
fn proof_field_slots_match_the_current_stage_shapes() {
    for (stage, field, slot, valid) in [
        (ProofStage::Stage1, ProofField::Rounds, 35, true),
        (ProofStage::Stage1, ProofField::Rounds, 36, false),
        (ProofStage::Stage1, ProofField::Basis16Claims, 3, true),
        (ProofStage::Stage1, ProofField::Basis16Claims, 4, false),
        (ProofStage::Stage1, ProofField::ProjectionClaims, 2, true),
        (ProofStage::Stage1, ProofField::ProjectionClaims, 3, false),
        (ProofStage::Stage2, ProofField::Rounds, 47, true),
        (ProofStage::Stage2, ProofField::Rounds, 48, false),
        (ProofStage::Stage2, ProofField::FoldedDigits, 5, true),
        (ProofStage::Stage2, ProofField::FoldedDigits, 6, false),
        (ProofStage::Stage3, ProofField::Rounds, 35, true),
        (ProofStage::Stage3, ProofField::Rounds, 36, false),
        (ProofStage::Stage3, ProofField::TerminalEvaluations, 8, true),
        (
            ProofStage::Stage3,
            ProofField::TerminalEvaluations,
            9,
            false,
        ),
    ] {
        let result = super::invariant().check(&vec![WitnessTamper::ProofClaim {
            stage,
            field,
            slot,
            delta: 1,
        }]);
        if valid {
            assert_invalid(result, "valid slot shape is a proof-only candidate");
        } else {
            assert_invalid(result, "slot outside the current proof shape is malformed");
        }
    }
}

#[test]
fn commitment_swap_is_not_in_the_input_schema() {
    let input = serde_json::json!([{"kind": "commitment_swap", "a": 0, "b": 1}]);
    assert!(
        serde_json::from_value::<Vec<WitnessTamper>>(input).is_err(),
        "commitment swaps must not deserialize as witness-soundness tampers"
    );
}

#[test]
fn proof_only_edits_to_current_stage3_terminals_are_out_of_scope() {
    let candidate = (0..9)
        .map(|slot| WitnessTamper::ProofClaim {
            stage: ProofStage::Stage3,
            field: ProofField::TerminalEvaluations,
            slot,
            delta: i64::try_from(slot).unwrap() - 4,
        })
        .collect::<Vec<_>>();
    assert_invalid(
        invariant().check(&candidate),
        "proof-only changes to Stage3 terminal evaluations",
    );
}

#[test]
fn benign_coordinated_tampers_are_out_of_scope() {
    let inv = invariant();
    // Valid Stage1 claim edits without a witness defect are outside this
    // witness-soundness property.
    assert_invalid(
        inv.check(&vec![
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
        ]),
        "coordinated Stage1 projection-claim move",
    );
    // A no-op source write plus a zero proof edit is still benign.
    assert_invalid(
        inv.check(&vec![
            WitnessTamper::SourceCell {
                table: SourceCell::A0,
                index: 5,
                value: original_a0(5),
            },
            WitnessTamper::ProofClaim {
                stage: ProofStage::Stage1,
                field: ProofField::ProjectionClaims,
                slot: 0,
                delta: 0,
            },
        ]),
        "all-benign coordinated list",
    );
    assert_invalid(inv.check(&vec![]), "empty tamper list");
}

#[test]
fn cancelling_tampers_are_out_of_scope() {
    let inv = invariant();
    // +1 then -1 on the same claim slot is both proof-only and net-zero.
    assert_invalid(
        inv.check(&vec![
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
        ]),
        "net-zero claim delta on one slot",
    );
    // Last-write-wins restores the original decoded source value.
    assert_invalid(
        inv.check(&vec![
            WitnessTamper::SourceCell {
                table: SourceCell::A0,
                index: 5,
                value: 8,
            },
            WitnessTamper::SourceCell {
                table: SourceCell::A0,
                index: 5,
                value: original_a0(5),
            },
        ]),
        "out-of-alphabet cell restored",
    );
    // Output deltas that cancel leave the honest projection.
    assert_invalid(
        inv.check(&vec![
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
        ]),
        "net-zero output delta",
    );
}

#[test]
fn unbounded_tamper_lists_are_out_of_scope() {
    let input = vec![
        WitnessTamper::SourceCell {
            table: SourceCell::A0,
            index: 5,
            value: original_a0(5),
        };
        1025
    ];
    assert_invalid(
        invariant().check(&input),
        "tamper count past the resource cap",
    );
}

#[test]
fn benign_regression_battery_is_out_of_scope() {
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
                value: i64::from(fixture().s2[9]),
            }],
            "unchanged s2",
        ),
        (
            vec![src(5, original_a0(5)), claim(0, 1), claim(0, -1)],
            "all benign",
        ),
    ];

    for (input, label) in battery {
        assert_invalid(invariant().check(&input), label);
    }
}
