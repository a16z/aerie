//! Fast statement-binding tests.

use aerie_invariants::invariants::{
    sample_statement, supported_statement_record_counts, StatementBinding, StatementPair,
};
use aerie_invariants::{Invariant, Outcome};

use crate::support::assert_invalid;

#[test]
fn identical_mismatched_and_oversize_pairs_are_out_of_scope() {
    let inv = StatementBinding;
    let eight = sample_statement(8, None).expect("8-record statement");
    // Same statement on both sides.
    assert_invalid(
        inv.check(&StatementPair {
            a: eight.clone(),
            b: eight.clone(),
        }),
        "identical statements",
    );
    // Different record counts cannot share a proof shape.
    let five = sample_statement(5, None).expect("5-record statement");
    assert_invalid(
        inv.check(&StatementPair { a: eight, b: five }),
        "mismatched record counts",
    );
    // Over the record cap.
    let big = sample_statement(17, None).expect("17-record statement");
    assert_invalid(
        inv.check(&StatementPair {
            a: big.clone(),
            b: big,
        }),
        "records over the cap",
    );
}

#[test]
fn unsupported_record_counts_are_rejected_on_either_side() {
    let eight = sample_statement(8, None).unwrap();
    let supported = supported_statement_record_counts();
    for records in (1..=17).filter(|records| !supported.contains(records)) {
        let unsupported = sample_statement(records, None).unwrap();
        for (a, b) in [
            (&unsupported, &unsupported),
            (&unsupported, &eight),
            (&eight, &unsupported),
        ] {
            let pair = StatementPair {
                a: a.clone(),
                b: b.clone(),
            };
            let Outcome::Invalid(reason) = StatementBinding.check(&pair).unwrap() else {
                panic!("{records} records must be out of scope");
            };
            assert_eq!(
                reason,
                format!("record count {records} is out of scope; supported live record counts are {supported:?}")
            );
        }
    }
}
