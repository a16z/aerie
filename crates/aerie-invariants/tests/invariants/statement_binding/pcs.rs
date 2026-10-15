//! PCS-backed statement-binding tests.

use aerie_falcon::{Falcon1024, Falcon512, FalconDegree, FalconParams};
use aerie_falcon_statement::{AggregationRecord, AggregationStatement};
use aerie_invariants::invariants::{
    sample_statement, supported_statement_record_counts, StatementBinding, StatementInput,
    StatementPair,
};
use aerie_invariants::Invariant;

use crate::support::{assert_holds, pcs_guard};

#[test]
fn a_proof_does_not_verify_under_a_second_statement() {
    let _pcs = pcs_guard();
    let inv = StatementBinding;
    for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
        let supported = supported_statement_record_counts(degree);
        // A's proof must reject B's distinct public statement.
        for &records in supported {
            let a = sample_statement(degree, records, None).expect("constructed A");
            let b = sample_statement(degree, records, Some(7)).expect("honest B");
            assert_holds(
                inv.check(&StatementPair { a, b }),
                &format!("{degree:?}: constructed A vs honest B at {records} records"),
            );
        }
        let a = sample_statement(degree, 8, Some(1)).expect("honest A");
        let b = sample_statement(degree, 8, Some(2)).expect("honest B");
        let checked = inv.check(&StatementPair { a, b });
        let context = format!("{degree:?}: two distinct honest statements");
        assert_holds(checked, &context);
    }
}

#[test]
fn a_proof_does_not_verify_when_only_the_external_message_changes() {
    let _pcs = pcs_guard();
    for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
        let a = sample_statement(degree, 8, None).expect("constructed A");
        let b = match degree {
            FalconDegree::Falcon512 => alter_external_message::<Falcon512>(&a),
            FalconDegree::Falcon1024 => alter_external_message::<Falcon1024>(&a),
        };

        // Keys, signatures, and transported salts are unchanged. Only the public
        // message in the statement passed to verify_aggregate differs; B need not have
        // a valid witness for the binding check.
        let checked = StatementBinding.check(&StatementPair { a, b });
        let context = format!("{degree:?}: only B's externally supplied message differs");
        assert_holds(checked, &context);
    }
}

fn alter_external_message<P: FalconParams>(a: &StatementInput) -> StatementInput {
    let statement = AggregationStatement::<P>::parse_canonical(&a.statement).unwrap();
    let mut records = statement.to_records();
    let mut message = records[0].message().to_vec();
    message.push(0x42);
    records[0] = AggregationRecord::new(records[0].public_key().clone(), message).unwrap();
    let altered = AggregationStatement::new(records).unwrap();
    let mut b = a.clone();
    b.statement = altered.canonical_encoding().unwrap().into();
    b
}
