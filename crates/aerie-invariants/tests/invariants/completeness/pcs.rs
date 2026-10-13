//! PCS-backed completeness tests.

use aerie_invariants::invariants::{
    sample_statement, supported_statement_record_counts, Completeness,
};
use aerie_invariants::Invariant;

use crate::support::{assert_holds, pcs_guard};

/// Real FN-DSA and constructed statements both prove and verify end to end
/// through the current production-profile path.
#[test]
fn valid_statements_verify_end_to_end() {
    let _pcs = pcs_guard();
    let inv = Completeness;
    for &records in supported_statement_record_counts() {
        assert_holds(
            inv.check(&sample_statement(records, Some(7)).expect("honest statement")),
            &format!("honest {records}-record batch through the production wire path"),
        );
    }
    assert_holds(
        inv.check(&sample_statement(8, None).expect("constructed statement")),
        "constructed 8-record batch through the production wire path",
    );
}
