//! Public facade for Aerie signature aggregation.
//!
//! # Example
//!
//! ```no_run
//! use aerie::falcon::{
//!     prove_aggregate, verify_aggregate, CanonicalPublicKeyBytes,
//!     CanonicalSignatureBytes, Falcon512, FalconAggregateSetup,
//! };
//! use aerie::{AggregationRecord, AggregationStatement, AggregationWitness};
//!
//! type P = Falcon512; // or aerie::falcon::Falcon1024
//!
//! /// Each record is (encoded public key, padded signature, message) for a
//! /// signature that verifies under standard Falcon with an empty context.
//! fn aggregate(
//!     records: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
//! ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
//!     let mut public_keys = Vec::new();
//!     let mut messages = Vec::new();
//!     let mut statement_records = Vec::new();
//!     let mut signatures = Vec::new();
//!     for (public_key, signature, message) in records {
//!         let public_key = CanonicalPublicKeyBytes::<P>::parse(public_key)?;
//!         let record = AggregationRecord::new(public_key.clone(), message.clone())?;
//!         statement_records.push(record);
//!         signatures.push(CanonicalSignatureBytes::<P>::parse(signature)?);
//!         public_keys.push(public_key);
//!         messages.push(message.clone());
//!     }
//!     let statement = AggregationStatement::new(statement_records)?;
//!     let witness = AggregationWitness::new(signatures)?;
//!
//!     // Depends only on the record count; build once and reuse per batch size.
//!     let setup = FalconAggregateSetup::<P>::new(records.len())?;
//!     let (proof, _metrics) = prove_aggregate(&setup, &statement, &witness)?;
//!
//!     // The verifier sees only the public keys, the messages and the proof.
//!     verify_aggregate(setup.verifier(), &public_keys, &messages, &proof)?;
//!     Ok(proof)
//! }
//! ```

#![forbid(unsafe_code)]

/// The Aerie Falcon signature aggregation protocol.
///
/// Version 1 is the current wire. It will be frozen at the first public
/// release; until then it may change without a version bump, so proofs are not
/// durable across pinned commits.
pub mod falcon {
    pub use aerie_falcon::{
        prove_aggregate, verify_aggregate, verify_aggregate_measured, AggregateProveMetrics,
        AggregateVerifyMetrics, AggregationError, AggregationParams, AggregationRecord,
        AggregationStatement, AggregationWitness, CanonicalPublicKeyBytes, CanonicalSignatureBytes,
        Falcon1024, Falcon512, FalconAggregateSetup, FalconParams, FalconVerifierSetup,
        GroupOpening, GroupRole, OpeningLedger, PcsStageMetrics, PreprocessingMetrics, Stability,
        VerifiedOpening, PROTOCOL_DOMAIN, PROTOCOL_STABILITY, WIRE_VERSION,
    };
    #[cfg(feature = "bench-internals")]
    pub use aerie_falcon::{transport, wire, FalconDegree};
}

#[cfg(feature = "bench-internals")]
pub use aerie_falcon_statement::split_source_statement;
pub use falcon::{
    AggregationRecord, AggregationStatement, AggregationWitness, PreprocessingMetrics,
};

#[cfg(test)]
mod tests {
    #[test]
    fn falcon_is_available() {
        assert_eq!(crate::falcon::WIRE_VERSION, 2);
        assert_eq!(
            crate::falcon::PROTOCOL_STABILITY,
            crate::falcon::Stability::Development,
        );
    }
}
