//! Aerie Falcon signature aggregation protocol.
//!
//! The protocol remains under development until its acceptance criteria
//! are complete. Development proofs are disposable. Two development builds
//! interoperate only when they use the same pinned commit.
//! Version 1 is the current wire. It freezes at the first public release;
//! until then, it may change without a wire-version bump.
//!
//! One wire version and one root transcript domain cover both standardized
//! parameter sets, Falcon-512 and Falcon-1024; the parameter-set byte in the
//! wire header and in every transcript binding separates them.
//!
//! Four native Akita commitment groups authenticate nine terminal evaluations.
//! Three batched sumchecks enforce the range, projection, sign, norm, encoding,
//! and direct negacyclic ring relations specified in `specs/falcon.md`.

mod error;
pub(crate) mod protocol;
pub(crate) mod substrate;

pub use aerie_falcon_native::source::{CanonicalPublicKeyBytes, CanonicalSignatureBytes};
pub use aerie_falcon_native::{Falcon1024, Falcon512, FalconParams};
pub use aerie_falcon_statement::{
    AggregationRecord, AggregationStatement, AggregationWitness, PreprocessingMetrics,
};
pub use error::AggregationError;
pub use protocol::manifest::GroupRole;
pub use protocol::openings::{GroupOpening, OpeningLedger};
pub use protocol::pcs::{
    FalconAggregateSetup, FalconVerifierSetup, PcsStageMetrics, VerifiedOpening,
};
pub use protocol::transport::{
    prove_aggregate, verify_aggregate, verify_aggregate_measured, AggregateProveMetrics,
    AggregateVerifyMetrics,
};
pub use protocol::AggregationParams;

// These modules support executable harnesses and adversarial tests. They are
// deliberately absent from the default API.
#[cfg(feature = "adversarial-internals")]
pub use protocol::pad_witness;
#[cfg(feature = "adversarial-internals")]
pub use protocol::public_inputs::PublicInputs;
#[cfg(any(feature = "bench-internals", feature = "adversarial-internals"))]
pub use protocol::{
    manifest, openings, padded_records, pcs, projection, ring, source_tables, stage1, stage2,
    stage3, transcript, transport, wire, zero_extension, CommitmentCellCounts, FalconDegree,
    FalconWitness, MAX_RECORDS,
};
#[cfg(any(feature = "bench-internals", feature = "adversarial-internals"))]
pub use substrate::{BaseField, OpeningField};

/// Falcon aggregation wire version developed by this crate, shared by both
/// parameter sets.
pub const WIRE_VERSION: u16 = 1;
/// Root Fiat Shamir domain for Falcon wire version 1, shared by both
/// parameter sets.
pub const PROTOCOL_DOMAIN: &[u8] = b"aerie/falcon/v1";

/// Release status of a protocol version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stability {
    Development,
    Stable,
}

/// Version 1 stays in development until the acceptance checklist is complete.
pub const PROTOCOL_STABILITY: Stability = Stability::Development;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::FalconDegree;

    #[test]
    fn wire_identity_remains_explicitly_in_development() {
        assert_eq!(WIRE_VERSION, 1);
        assert_eq!(
            FalconDegree::Falcon512.identity_bytes()[..2],
            WIRE_VERSION.to_le_bytes()
        );
        assert_eq!(
            FalconDegree::Falcon1024.identity_bytes()[..2],
            WIRE_VERSION.to_le_bytes()
        );
        assert_eq!(PROTOCOL_DOMAIN, b"aerie/falcon/v1");
        assert_eq!(PROTOCOL_STABILITY, Stability::Development);
    }
}
