//! Fast completeness tests.

use aerie_invariants::invariants::{
    sample_statement, supported_statement_record_counts, Completeness, StatementInput,
};
use aerie_invariants::{Invariant, Outcome};

use crate::support::assert_invalid;

#[test]
fn natively_invalid_and_oversize_inputs_are_out_of_scope() {
    let inv = Completeness;
    // A statement whose signatures do not match it (constructed keys/messages
    // paired with signatures from another statement) parses but is not
    // natively valid.
    let constructed = sample_statement(8, None).expect("constructed");
    let honest = sample_statement(8, Some(1)).expect("honest");
    assert_invalid(
        inv.check(&StatementInput {
            statement: constructed.statement,
            witness: honest.witness,
        }),
        "signatures do not verify against the statement",
    );
    // Over the record cap.
    assert_invalid(
        inv.check(&sample_statement(17, None).expect("17-record statement")),
        "records over the cap",
    );
    // Malformed bytes.
    assert_invalid(
        inv.check(&StatementInput {
            statement: vec![0_u8; 8].into(),
            witness: vec![0_u8; 8].into(),
        }),
        "malformed statement bytes",
    );
}

#[test]
fn unsupported_record_counts_are_rejected_by_domain_validation() {
    let supported = supported_statement_record_counts();
    assert_eq!(supported, (1..=16).collect::<Vec<_>>());
    for records in (1..=17).filter(|records| !supported.contains(records)) {
        let input = sample_statement(records, None).unwrap();
        let Outcome::Invalid(reason) = Completeness.check(&input).unwrap() else {
            panic!("{records} records must be out of scope");
        };
        assert_eq!(
            reason,
            format!("record count {records} is out of scope; supported live record counts are {supported:?}")
        );
    }
}
