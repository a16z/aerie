//! Producer field and commitment substrate.

pub(crate) use aerie_falcon::substrate::*;
use akita_pcs::AkitaCommitmentScheme;

/// The multi-group opening uses the final group's width-4 catalog; each
/// precommitted group carries its own frozen [`akita_params::GroupCommitPhaseParams`].
pub(crate) type GroupOpeningScheme = AkitaCommitmentScheme<Width4Config>;
