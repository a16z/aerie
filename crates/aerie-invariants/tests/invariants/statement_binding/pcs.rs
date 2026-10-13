//! PCS-backed statement-binding tests.

use aerie_falcon_statement::{AggregationRecord, AggregationStatement};
use aerie_invariants::invariants::{
    sample_statement, supported_statement_record_counts, StatementBinding, StatementPair,
};
use aerie_invariants::Invariant;

use crate::support::{assert_holds, pcs_guard};

#[test]
fn a_proof_does_not_verify_under_a_second_statement() {
    let _pcs = pcs_guard();
    let inv = StatementBinding;
    // Constructed A vs honest B, and two distinct honest statements: A's proof
    // must not verify when re-presented as the distinct statement B.
    for &records in supported_statement_record_counts() {
        let a = sample_statement(records, None).expect("constructed A");
        let b = sample_statement(records, Some(7)).expect("honest B");
        assert_holds(
            inv.check(&StatementPair { a, b }),
            &format!("constructed A vs honest B at {records} records"),
        );
    }
    let a = sample_statement(8, Some(1)).expect("honest A");
    let b = sample_statement(8, Some(2)).expect("honest B");
    assert_holds(
        inv.check(&StatementPair { a, b }),
        "two distinct honest statements",
    );
}

#[test]
fn a_proof_does_not_verify_when_only_the_external_message_changes() {
    let _pcs = pcs_guard();
    let a = sample_statement(8, None).expect("constructed A");
    let statement =
        AggregationStatement::<aerie_falcon::Falcon512>::parse_canonical(&a.statement).unwrap();
    let mut records = statement.records().to_vec();
    let mut message = records[0].message().to_vec();
    message.push(0x42);
    records[0] = AggregationRecord::new(records[0].public_key().clone(), message).unwrap();
    let altered = AggregationStatement::new(records).unwrap();
    let mut b = a.clone();
    b.statement = altered.canonical_encoding().unwrap().into();

    // Keys, signatures, and transported salts are unchanged. Only the public
    // message passed separately to verify_aggregate differs; B need not have
    // a valid witness for the binding check.
    assert_holds(
        StatementBinding.check(&StatementPair { a, b }),
        "only B's externally supplied message differs",
    );
}
