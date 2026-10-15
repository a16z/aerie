//! Aerie Falcon signature aggregation protocol.
//!
//! Falcon v1.2 padded signature format with key-prefixed message framing
//! (raw mode, empty context), as implemented by fn-dsa 0.3.0, is supported.
//! Original `r || m` framing is unsupported; compatibility with all v1.2
//! signers or final FIPS 206 is not claimed.
//!
//! Wire version 1 is frozen: incompatible transcript or encoding changes
//! require a new wire version. Proof interchange requires the same pinned
//! implementation and profile. Neither parameter set has passed the production
//! acceptance criteria of `specs/release-policy.md`.
//!
//! One wire version and one root transcript domain cover both standardized
//! parameter sets, Falcon-512 and Falcon-1024; the parameter-set byte in the
//! wire header and in every transcript binding separates them.
//!
//! Four native Akita commitment groups authenticate nine terminal evaluations.
//! Three batched sumchecks enforce the range, projection, sign, norm, encoding,
//! and direct negacyclic ring relations specified in `specs/falcon.md`.

#[doc(hidden)]
pub mod error;
#[doc(hidden)]
pub mod protocol;
#[doc(hidden)]
pub mod substrate;

pub use aerie_falcon_native::source::{CanonicalPublicKeyBytes, CanonicalSignatureBytes};
pub use aerie_falcon_native::{Falcon1024, Falcon512, FalconParams};
pub use aerie_falcon_statement::{AggregationRecord, AggregationStatement};
pub use error::{
    AggregationError, BackendError, MalformedProof, RejectionStage, StatementError,
    StatementMismatch, UnsupportedBatch,
};
#[cfg(feature = "internals")]
pub use protocol::manifest::GroupRole;
#[cfg(feature = "internals")]
pub use protocol::openings::{GroupOpening, OpeningLedger};
pub use protocol::pcs::FalconVerifierSetup;
#[cfg(feature = "internals")]
pub use protocol::pcs::VerifiedOpening;
pub use protocol::transport::verify_aggregate;
#[cfg(feature = "internals")]
pub use protocol::transport::{verify_aggregate_measured, AggregateVerifyMetrics};
pub use protocol::wire::SaltMode;
pub use protocol::AggregationParams;

// The internals capability supports executable harnesses and adversarial tests.
// These exports are unsupported internal API and absent from the default API.
#[cfg(feature = "internals")]
pub use protocol::public_inputs::PublicInputs;
#[cfg(feature = "internals")]
pub use protocol::{
    manifest, openings, padded_records, pcs, projection, ring, stage1, stage2, stage3, transcript,
    transport, wire, zero_extension, CommitmentCellCounts, FalconDegree, MAX_RECORDS,
};
#[cfg(feature = "internals")]
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

/// Wire version 1 is frozen. Proof interchange requires the same pinned
/// implementation and profile.
pub const PROTOCOL_STABILITY: Stability = Stability::Stable;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::FalconDegree;

    #[test]
    fn wire_version_1_is_stable() {
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
        assert_eq!(PROTOCOL_STABILITY, Stability::Stable);
    }
}
