//! PCS-backed completeness tests.

use aerie_falcon::FalconDegree;
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
    for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
        let supported = supported_statement_record_counts(degree);
        for &records in supported {
            assert_holds(
                inv.check(&sample_statement(degree, records, Some(7)).expect("honest statement")),
                &format!(
                    "{degree:?}: honest {records}-record batch through the production wire path"
                ),
            );
        }
        let constructed = sample_statement(degree, 8, None).expect("constructed statement");
        let checked = inv.check(&constructed);
        let context =
            format!("{degree:?}: constructed 8-record batch through the production wire path");
        assert_holds(checked, &context);
    }
}
