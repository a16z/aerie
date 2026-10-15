use std::sync::{Mutex, MutexGuard};

use aerie_invariants::{CheckResult, Outcome};

/// Serialize tests that invoke the real PCS to bound peak memory.
pub(crate) fn pcs_guard() -> MutexGuard<'static, ()> {
    static PCS: Mutex<()> = Mutex::new(());
    PCS.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[track_caller]
pub(crate) fn assert_holds(checked: CheckResult, context: &str) {
    let outcome = checked.unwrap_or_else(|error| panic!("{context}: checker failed: {error}"));
    match outcome {
        Outcome::Holds => {}
        Outcome::Invalid(reason) => panic!("{context}: expected Holds, got Invalid: {reason}"),
        Outcome::Violated(violation) => {
            panic!(
                "{context}: expected Holds, got VIOLATION: {}",
                violation.detail
            )
        }
    }
}

#[track_caller]
pub(crate) fn assert_invalid(checked: CheckResult, context: &str) {
    let outcome = checked.unwrap_or_else(|error| panic!("{context}: checker failed: {error}"));
    match outcome {
        Outcome::Invalid(_) => {}
        Outcome::Holds => panic!("{context}: expected Invalid, got Holds"),
        Outcome::Violated(violation) => {
            panic!(
                "{context}: expected Invalid, got VIOLATION: {}",
                violation.detail
            )
        }
    }
}

#[track_caller]
pub(crate) fn assert_not_violated(checked: CheckResult, context: &str) {
    let outcome = checked.unwrap_or_else(|error| panic!("{context}: checker failed: {error}"));
    if let Outcome::Violated(violation) = outcome {
        panic!(
            "{context}: FALSE POSITIVE — reported a violation on the sound verifier: {}",
            violation.detail
        );
    }
}
