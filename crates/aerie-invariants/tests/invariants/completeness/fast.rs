//! Fast completeness tests.

use aerie_falcon::FalconDegree;
use aerie_invariants::invariants::{
    sample_statement, supported_statement_record_counts, Completeness, StatementInput,
};
use aerie_invariants::{lookup, Invariant, Outcome};

use crate::support::assert_invalid;

#[test]
fn seed_corpus_contains_both_parameter_sets() {
    let corpus = Completeness.seed_corpus();
    assert_eq!(corpus.len(), 2);
    assert_eq!(
        corpus
            .iter()
            .map(|input| input.parameter_set)
            .collect::<Vec<_>>(),
        [Some(9), Some(10)]
    );
}

#[test]
fn natively_invalid_and_oversize_inputs_are_out_of_scope() {
    let inv = Completeness;
    for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
        // The signatures must verify against the supplied keys and messages.
        let constructed = sample_statement(degree, 8, None).expect("constructed");
        let honest = sample_statement(degree, 8, Some(1)).expect("honest");
        assert_invalid(
            inv.check(&StatementInput {
                parameter_set: Some(degree.parameter_set()),
                statement: constructed.statement,
                witness: honest.witness,
            }),
            &format!("{degree:?}: signatures do not verify against the statement"),
        );
        assert_invalid(
            inv.check(&sample_statement(degree, 17, None).expect("17-record statement")),
            &format!("{degree:?}: records over the cap"),
        );
        assert_invalid(
            inv.check(&StatementInput {
                parameter_set: Some(degree.parameter_set()),
                statement: vec![0_u8; 8].into(),
                witness: vec![0_u8; 8].into(),
            }),
            &format!("{degree:?}: malformed statement bytes"),
        );
    }
}

#[test]
fn unsupported_record_counts_are_rejected_by_domain_validation() {
    for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
        let supported = supported_statement_record_counts(degree);
        if degree == FalconDegree::Falcon512 {
            assert_eq!(supported, (1..=16).collect::<Vec<_>>());
        }
        for records in (1..=17).filter(|records| !supported.contains(records)) {
            let input = sample_statement(degree, records, None).unwrap();
            let Outcome::Invalid(reason) = Completeness.check(&input).unwrap() else {
                panic!("{degree:?}: {records} records must be out of scope");
            };
            assert_eq!(
                reason,
                format!("record count {records} is out of scope; supported live record counts are {supported:?}")
            );
        }
    }
}

#[test]
fn missing_unknown_and_mismatched_parameter_sets_are_invalid() {
    let registered = lookup("completeness").expect("registered completeness invariant");
    for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
        let input = sample_statement(degree, 1, None).expect("statement");
        let other = match degree {
            FalconDegree::Falcon512 => FalconDegree::Falcon1024,
            FalconDegree::Falcon1024 => FalconDegree::Falcon512,
        };
        for parameter_set in [None, Some(0), Some(other.parameter_set())] {
            let mut candidate = input.clone();
            candidate.parameter_set = parameter_set;
            let context = format!("{degree:?}: invalid parameter set {parameter_set:?}");
            assert_invalid(Completeness.check(&candidate), &context);
            let mut json = serde_json::to_value(candidate).unwrap();
            if parameter_set.is_none() {
                json.as_object_mut().unwrap().remove("parameter_set");
            }
            assert_invalid(registered.resolve_and_adjudicate(&json).result, &context);
        }
    }
}
