//! Field, configuration, and PCS wire-shape substrate for wire version 1.
//!
//! These aliases select Akita's fp64 proof-optimized field pair. The wire-shape
//! helper mirrors the exact transmitted-coefficient layout of the pinned Akita
//! commitment objects.

use aerie_pcs_config::configs::{Fp64SignedWidth14, Fp64SignedWidth4};
use akita_error::AkitaError;
use akita_params::{CompressionChainPlan, GroupCommitPhaseParams};

/// Width-14 family for the S1 and slack groups.
#[doc(hidden)]
pub type Width14Config = Fp64SignedWidth14;
pub type BaseField = akita_config::proof_optimized::fp64::Field;
/// The base-field modulus `2^64 - 59`.
pub const FIELD_MODULUS: u64 = ((1_u128
    << <BaseField as jolt_field::CanonicalEncoding>::MODULUS_BITS)
    - <BaseField as jolt_field::PseudoMersenne>::OFFSET) as u64;
// The projection certificate check reads the modulus from this literal.
const _: () = assert!(FIELD_MODULUS == u64::MAX - 58);
pub type OpeningField = akita_config::proof_optimized::fp64::ExtensionField;
/// Width-4 family for the source and budget groups.
#[doc(hidden)]
pub type Width4Config = Fp64SignedWidth4;
pub(crate) fn commitment_coefficients(
    profile: &GroupCommitPhaseParams,
) -> Result<usize, AkitaError> {
    let source_coefficients = profile.outer_slice_count.complete_source_coefficients(
        profile.outer.matrix.output_rank(),
        profile.outer.matrix.ring_dimension(),
    )?;
    Ok(CompressionChainPlan::for_complete_source(
        profile.outer.matrix.sis_table_key().modulus_profile,
        source_coefficients,
    )?
    .terminal_coefficients())
}
