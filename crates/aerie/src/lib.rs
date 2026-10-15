//! Public facade for Aerie signature aggregation.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "prover")]
//! # fn main() {
//! use aerie::falcon::{
//!     prove_aggregate, verify_aggregate, CanonicalPublicKeyBytes,
//!     CanonicalSignatureBytes, Falcon512, FalconAggregateSetup,
//! };
//! use aerie::falcon::{AggregationRecord, AggregationStatement, AggregationWitness};
//!
//! type P = Falcon512; // or aerie::falcon::Falcon1024
//!
//! /// Each record is (encoded public key, padded signature, message) for a
//! /// signature that verifies under Falcon v1.2 padded signature format with
//! /// key-prefixed message framing (raw mode, empty context), as implemented by
//! /// fn-dsa 0.3.0. Original `r || m` framing is unsupported; compatibility with
//! /// all v1.2 signers or final FIPS 206 is not claimed.
//! fn aggregate(
//!     records: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
//! ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
//!     let mut statement_records = Vec::new();
//!     let mut signatures = Vec::new();
//!     for (public_key, signature, message) in records {
//!         let public_key = CanonicalPublicKeyBytes::<P>::parse(public_key)?;
//!         let record = AggregationRecord::new(public_key, message.clone())?;
//!         statement_records.push(record);
//!         signatures.push(CanonicalSignatureBytes::<P>::parse(signature)?);
//!     }
//!     let statement = AggregationStatement::new(statement_records)?;
//!     let witness = AggregationWitness::new(signatures)?;
//!
//!     // Depends only on the record count; build once and reuse per batch size.
//!     let setup = FalconAggregateSetup::<P>::new(records.len())?;
//!     let proof = prove_aggregate(&setup, &statement, &witness)?;
//!
//!     // The verifier sees the public statement and the proof.
//!     verify_aggregate(setup.verifier(), &statement, &proof)?;
//!     Ok(proof)
//! }
//! # }
//! # #[cfg(not(feature = "prover"))]
//! # fn main() {}
//! ```

#![forbid(unsafe_code)]

/// The Aerie Falcon signature aggregation protocol.
///
/// Falcon v1.2 padded signature format with key-prefixed message framing
/// (raw mode, empty context), as implemented by fn-dsa 0.3.0, is supported.
/// Original `r || m` framing is unsupported; compatibility with all v1.2
/// signers or final FIPS 206 is not claimed.
///
/// Wire version 1 is frozen: incompatible transcript or encoding changes
/// require a new wire version. Proof interchange requires the same pinned
/// implementation and profile.
pub mod falcon {
    #[cfg(feature = "prover")]
    pub use aerie_prover::{
        prove_aggregate, prove_aggregate_with_mode, AggregationWitness, FalconAggregateSetup,
    };
    pub use aerie_verifier::{
        verify_aggregate, AggregationError, AggregationParams, AggregationRecord,
        AggregationStatement, BackendError, CanonicalPublicKeyBytes, CanonicalSignatureBytes,
        Falcon1024, Falcon512, FalconParams, FalconVerifierSetup, MalformedProof, RejectionStage,
        SaltMode, Stability, StatementError, StatementMismatch, UnsupportedBatch, PROTOCOL_DOMAIN,
        PROTOCOL_STABILITY, WIRE_VERSION,
    };
}
