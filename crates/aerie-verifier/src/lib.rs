//! Verification API for the Aerie Falcon aggregation protocol.
//!
//! Verification consumes a public statement, verifier setup, and proof bytes.
//! Protocol support and verifier replay have one implementation in `aerie-falcon`.

#![forbid(unsafe_code)]

pub use aerie_falcon::{
    verify_aggregate, AggregationError, AggregationParams, AggregationRecord, AggregationStatement,
    BackendError, CanonicalPublicKeyBytes, CanonicalSignatureBytes, Falcon1024, Falcon512,
    FalconParams, FalconVerifierSetup, MalformedProof, RejectionStage, SaltMode, Stability,
    StatementError, StatementMismatch, UnsupportedBatch, PROTOCOL_DOMAIN, PROTOCOL_STABILITY,
    WIRE_VERSION,
};

#[cfg(feature = "internals")]
pub use aerie_falcon::{verify_aggregate_measured, AggregateVerifyMetrics};
