//! Fast statement-binding tests.

use aerie_falcon::FalconDegree;
use aerie_invariants::invariants::{
    sample_statement, supported_statement_record_counts, StatementBinding, StatementPair,
};
use aerie_invariants::{lookup, Invariant, Outcome};

use crate::support::assert_invalid;

#[test]
fn seed_corpus_contains_both_parameter_sets() {
    let corpus = StatementBinding.seed_corpus();
    assert_eq!(corpus.len(), 2);
    assert_eq!(
        corpus
            .iter()
            .map(|pair| pair.a.parameter_set)
            .collect::<Vec<_>>(),
        [Some(9), Some(10)]
    );
    assert_eq!(
        corpus
            .iter()
            .map(|pair| pair.b.parameter_set)
            .collect::<Vec<_>>(),
        [Some(9), Some(10)]
    );
    for pair in corpus {
        assert_ne!(pair.a.statement, pair.b.statement);
    }
}

#[test]
fn identical_mismatched_and_oversize_pairs_are_out_of_scope() {
    let inv = StatementBinding;
    for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
        let eight = sample_statement(degree, 8, None).expect("8-record statement");
        assert_invalid(
            inv.check(&StatementPair {
                a: eight.clone(),
                b: eight.clone(),
            }),
            &format!("{degree:?}: identical statements"),
        );
        // Different record counts cannot share a proof shape.
        let five = sample_statement(degree, 5, None).expect("5-record statement");
        assert_invalid(
            inv.check(&StatementPair { a: eight, b: five }),
            &format!("{degree:?}: mismatched record counts"),
        );
        let big = sample_statement(degree, 17, None).expect("17-record statement");
        assert_invalid(
            inv.check(&StatementPair {
                a: big.clone(),
                b: big,
            }),
            &format!("{degree:?}: records over the cap"),
        );
    }
}

#[test]
fn unsupported_record_counts_are_rejected_on_either_side() {
    for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
        let supported = supported_statement_record_counts(degree);
        let reference = sample_statement(degree, supported[0], None).unwrap();
        for records in (1..=17).filter(|records| !supported.contains(records)) {
            let unsupported = sample_statement(degree, records, None).unwrap();
            for (a, b) in [
                (&unsupported, &unsupported),
                (&unsupported, &reference),
                (&reference, &unsupported),
            ] {
                let pair = StatementPair {
                    a: a.clone(),
                    b: b.clone(),
                };
                let Outcome::Invalid(reason) = StatementBinding.check(&pair).unwrap() else {
                    panic!("{degree:?}: {records} records must be out of scope");
                };
                assert_eq!(
                    reason,
                    format!("record count {records} is out of scope; supported live record counts are {supported:?}")
                );
            }
        }
    }
}

#[test]
fn missing_unknown_and_mismatched_parameter_sets_are_invalid_on_either_side() {
    let registered = lookup("statement-binding").expect("registered statement-binding invariant");
    for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
        let a = sample_statement(degree, 1, None).expect("constructed A");
        let b = sample_statement(degree, 1, Some(7)).expect("honest B");
        assert_ne!(a.statement, b.statement);
        let other = match degree {
            FalconDegree::Falcon512 => FalconDegree::Falcon1024,
            FalconDegree::Falcon1024 => FalconDegree::Falcon512,
        };
        for side in ["a", "b"] {
            for parameter_set in [None, Some(0), Some(other.parameter_set())] {
                let mut pair = StatementPair {
                    a: a.clone(),
                    b: b.clone(),
                };
                let candidate = if side == "a" {
                    &mut pair.a
                } else {
                    &mut pair.b
                };
                candidate.parameter_set = parameter_set;
                let context = format!("{degree:?}: invalid {side} parameter set {parameter_set:?}");
                let Outcome::Invalid(reason) = StatementBinding.check(&pair).unwrap() else {
                    panic!("{context}: expected Invalid");
                };
                assert_eq!(
                    reason,
                    "A and B must name the same parameter_set: 9 (Falcon-512) or 10 (Falcon-1024)",
                    "{context}"
                );
                let mut json = serde_json::to_value(pair).unwrap();
                if parameter_set.is_none() {
                    json[side].as_object_mut().unwrap().remove("parameter_set");
                }
                let Outcome::Invalid(reason) =
                    registered.resolve_and_adjudicate(&json).result.unwrap()
                else {
                    panic!("{context}: registered adjudication expected Invalid");
                };
                assert_eq!(
                    reason,
                    "A and B must name the same parameter_set: 9 (Falcon-512) or 10 (Falcon-1024)",
                    "{context}"
                );
            }
        }
    }
}

#[test]
fn different_parameter_sets_cannot_share_a_proof() {
    let a = sample_statement(FalconDegree::Falcon512, 1, None).unwrap();
    let b = sample_statement(FalconDegree::Falcon1024, 1, None).unwrap();
    for (a, b) in [(&a, &b), (&b, &a)] {
        let pair = StatementPair {
            a: a.clone(),
            b: b.clone(),
        };
        assert_invalid(StatementBinding.check(&pair), "different parameter sets");
    }
}
