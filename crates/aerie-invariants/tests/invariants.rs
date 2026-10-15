//! Behavioral tests for the four invariants against the real Falcon
//! prover/verifier. Fast scope/schema tests and expensive PCS tests live in
//! separate modules, but remain in this single integration-test binary so the
//! process-wide PCS lock serializes their large-memory work.
//!
//! ```text
//! # Fast schema, scope, and reducer checks only:
//! cargo test -p aerie-invariants --test invariants --profile ci fast
//!
//! # Real prover/verifier checks only:
//! cargo test -p aerie-invariants --test invariants --profile ci pcs -- --test-threads=1
//! ```
//!
//! Every assertion here confirms the *correct* system's expected verdict:
//! genuine attacks are rejected (the invariant Holds), degenerate inputs are
//! Invalid. A Violated verdict would mean a real protocol bug.

#[path = "invariants/support.rs"]
mod support;

#[path = "invariants/completeness/mod.rs"]
mod completeness;
#[path = "invariants/statement_binding/mod.rs"]
mod statement_binding;
#[path = "invariants/verifier_totality/mod.rs"]
mod verifier_totality;
#[path = "invariants/witness_soundness/mod.rs"]
mod witness_soundness;
