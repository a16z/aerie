//! Field, configuration, and PCS wire-shape substrate for wire version 1.
//!
//! These aliases select Akita's fp64 proof-optimized field pair. The wire-shape
//! helper mirrors the exact transmitted-coefficient layout of the pinned Akita
//! commitment objects.

use aerie_pcs_config::configs::{Fp64SignedWidth14, Fp64SignedWidth4};
use akita_error::AkitaError;
use akita_params::{CompressionChainPlan, GroupCommitPhaseParams};
use akita_pcs::AkitaCommitmentScheme;

pub type Config = Fp64SignedWidth14;
pub type BaseField = akita_config::proof_optimized::fp64::Field;
pub type OpeningField = akita_config::proof_optimized::fp64::ExtensionField;
/// Width-4 family for the digit and budget groups.
pub type DigitRangeConfig = Fp64SignedWidth4;
/// Proving config for the single multi-group root opening: the FINAL
/// group's (width-4) config; every precommitted group carries its own
/// frozen [`GroupCommitPhaseParams`].
pub type MergedConfig = DigitRangeConfig;
pub type MergedPcs = AkitaCommitmentScheme<MergedConfig>;
/// Committing config for the width-14 precommitted groups.
pub type ConservativeMainConfig = Config;

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
