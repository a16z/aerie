//! Common proof wire data and public-statement PCS verification.

use super::openings::OpeningLedger;
use super::projection::ProjectionOutputs;
use super::{stage1::Stage1Proof, stage2::Stage2Proof, stage3::Stage3Proof};
use crate::substrate::BaseField;
use crate::{AggregationError as Error, MalformedProof};
use akita_serialization::{AkitaSerialize, Compress};
use akita_types::Commitment;

mod setup;
mod verifier;

pub use setup::FalconVerifierSetup;
#[doc(hidden)]
pub use setup::{
    commitment_coefficient_counts_of, group_cells_of, max_physical_vars, shared_setup_requirements,
    total_group_polynomials, verifier_group_profiles_of,
};
pub use verifier::verify;
#[doc(hidden)]
pub use verifier::verify_opening;

/// Complete wire data of the version 1 protocol with real commitments.
#[derive(Clone, Debug)]
pub struct AggregateProof {
    /// The commitment payloads in canonical manifest order.
    pub commitments: Vec<Commitment<BaseField>>,
    /// The public projection outputs `(y, z)`, absorbed after the seed.
    pub projection_outputs: ProjectionOutputs,
    pub stage1: Stage1Proof,
    pub stage2: Stage2Proof,
    pub stage3: Stage3Proof,
    /// Inline canonical unsigned-LEB128 PIOP proof-of-work nonces.
    pub piop_work: Vec<u8>,
    /// The batched Akita opening: Akita's native Spongefish argument string.
    pub pcs: Vec<u8>,
}

/// Verifier output: the discharged opening ledger and the verify timings.
#[cfg_attr(
    not(feature = "internals"),
    allow(dead_code, reason = "read only by the measured entry points")
)]
#[derive(Clone, Debug)]
pub struct VerifiedOpening {
    pub ledger: OpeningLedger,
    /// Wall time spent streaming the public targets and contracting them at `r_Pi`.
    pub target_evaluation_ns: u128,
    /// Wall time for protocol-block verification excluding target evaluation.
    pub stage_verify_ns: u128,
    pub pcs_verify_ns: u128,
}

#[doc(hidden)]
pub fn serialized_commitment(commitment: &Commitment<BaseField>) -> Result<Vec<u8>, Error> {
    if commitment.rows().coeff_len() != 16 {
        return Err(Error::MalformedProof(MalformedProof::CommitmentShape));
    }
    let mut bytes = Vec::with_capacity(128);
    commitment
        .serialize_with_mode(&mut bytes, Compress::Yes)
        .map_err(Error::backend)?;
    debug_assert_eq!(bytes.len(), 128);
    Ok(bytes)
}
