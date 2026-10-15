//! Falcon aggregate proving over verifier-safe common protocol support.
pub(crate) use aerie_falcon::error;
pub(crate) mod protocol;
pub(crate) mod substrate;
pub use aerie_falcon::{
    verify_aggregate, AggregationError, AggregationParams, AggregationRecord, AggregationStatement,
    BackendError, CanonicalPublicKeyBytes, CanonicalSignatureBytes, Falcon1024, Falcon512,
    FalconParams, FalconVerifierSetup, MalformedProof, RejectionStage, SaltMode, Stability,
    StatementError, StatementMismatch, UnsupportedBatch, PROTOCOL_DOMAIN, PROTOCOL_STABILITY,
    WIRE_VERSION,
};
#[cfg(feature = "internals")]
pub use aerie_falcon::{
    verify_aggregate_measured, AggregateVerifyMetrics, BaseField, GroupOpening, GroupRole,
    OpeningField, OpeningLedger, PublicInputs, VerifiedOpening,
};
pub use aerie_falcon_witness::AggregationWitness;
#[cfg(feature = "internals")]
pub use aerie_falcon_witness::PreprocessingMetrics;
pub use protocol::pcs::FalconAggregateSetup;
#[cfg(feature = "internals")]
pub use protocol::pcs::PcsStageMetrics;
pub use protocol::transport::{prove_aggregate, prove_aggregate_with_mode};
#[cfg(feature = "internals")]
pub use protocol::transport::{prove_aggregate_measured_with_mode, AggregateProveMetrics};
#[cfg(feature = "internals")]
pub use protocol::{
    algebra, instance, manifest, openings, pad_witness, padded_records, pcs, projection,
    public_inputs, ring, source_tables, stage1, stage2, stage3, transcript, transport, wire,
    zero_extension, CommitmentCellCounts, FalconDegree, FalconWitness, MAX_RECORDS,
};
