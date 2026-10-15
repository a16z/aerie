//! `aerie-invariants` — machine-checkable security invariants for Aerie Falcon.
//!
//! Each [`Invariant`] states a security property of the protocol as a
//! deterministic predicate over a JSON-serializable counterexample candidate,
//! returning [`Outcome::Holds`], [`Outcome::Violated`], or [`Outcome::Invalid`]
//! (skip), with checker infrastructure failures reported separately as
//! [`CheckError`]. A confirmed violation is a reproducible artifact.
//!
//! The implemented invariants:
//!
//! - [`invariants::Completeness`] — canonical, native-verifier-valid
//!   statement/witness pairs always verify end to end under the current
//!   production profile.
//! - [`invariants::WitnessSoundness`] — no accepting proof carries a witness
//!   that violates a checked predicate.
//! - [`invariants::StatementBinding`] — no proof verifies under two statements.
//! - [`invariants::VerifierTotality`] — the verifier never panics on hostile
//!   bytes.
//!
//! This is a deliberately lightweight take on Jolt's `jolt-eval`: the same
//! three-way verdict, but no proc-macro, no fuzz targets, no `schemars`, and
//! object-safe erasure ([`DynInvariant`]) instead of a hand-written dispatch
//! enum. External search or fuzzing drivers can use this interface; no such
//! driver ships in this repository.

pub mod bytes;
pub mod invariant;
pub mod invariants;
mod support;

pub use bytes::Bytes;
pub use invariant::{
    lookup, registry, Adjudication, Briefing, CheckError, CheckResult, DynInvariant, Invariant,
    Outcome, Violation,
};
pub use support::configure_worker_threads;
