//! Clear structured projections and their public predicates.
//!
//! Stage one absorbs the canonical Rice section of the projection outputs,
//! then proves the source-link identity with its other basis-16 instances.
//! This module owns the unchanged projection profile, core derivation,
//! forward maps, predicates, and the Rice wire codec.

mod codec;
pub(crate) mod cores;
pub(crate) mod forward;
mod predicates;
pub(crate) mod profile;
pub(crate) mod reverse;
mod sources;
#[cfg(test)]
mod tests;

use super::transcript::PioTranscript;
use crate::substrate::BaseField as F;
use akita_error::AkitaError;

use super::FalconDegree;

#[cfg(any(feature = "bench-internals", feature = "adversarial-internals"))]
pub use codec::{CodecError, MAX_ENCODED_BYTES, MAX_RICE_PARAMETER};
#[cfg(any(feature = "bench-internals", feature = "adversarial-internals"))]
pub use cores::{CORE_XOF_LABEL, TERMINAL_GROUP};
pub use forward::ForwardProjection;
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub use predicates::{centered, FIELD_MODULUS};
pub use predicates::{norm_predicate, residual_predicate};
pub use profile::{
    norm_cap, ProjectionProfile, FALCON_Q, NORM_TERMINAL_ROWS, OUTPUT_COUNT, RESIDUAL_WIDTH,
};
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub use profile::{
    NORM_CAP_FALCON_1024, NORM_CAP_FALCON_512, NORM_CHANNELS, NORM_GROUPS, NORM_PREFIX_LAYERS,
    NORM_PREFIX_WIDTH, PROFILE_BINDING_LABEL, RESIDUAL_LAYERS,
};

#[cfg(any(feature = "bench-internals", feature = "adversarial-internals"))]
pub use profile::{
    CoreDistribution, NormLeg, NormSourceOrder, ResidualLeg, FIRST_CHANNELS, RESIDUAL_BOUND,
};

use cores::ProjectionCores;
use profile::Geometry;

/// Public projection outputs, in normative transcript order `y` then `z`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionOutputs {
    /// Norm-leg output `y`.
    pub y: [F; NORM_TERMINAL_ROWS],
    /// Residual-leg output `z`.
    pub z: [F; RESIDUAL_WIDTH],
}

impl ProjectionOutputs {
    /// `y` followed by `z`.
    pub fn values(&self) -> impl Iterator<Item = F> + '_ {
        self.y.iter().chain(&self.z).copied()
    }
}

fn invalid(reason: &str) -> AkitaError {
    AkitaError::InvalidInput(reason.to_owned())
}

/// Shared shape validation of a canonically padded batch.
fn geometry(
    degree: FalconDegree,
    records: usize,
    live_rows: usize,
) -> Result<Geometry, AkitaError> {
    if live_rows == 0 || live_rows > records || degree.padded_records(live_rows) != Ok(records) {
        return Err(invalid("the projection needs a canonically padded batch"));
    }
    Geometry::new(degree, records).map_err(invalid)
}

/// Stable prefix of the projection-rejection error in Section 6.1.
pub const PROJECTION_REJECTED: &str = "projection rejected";

/// Derive the cores, run the forward projection and check both predicates.
#[tracing::instrument(skip_all, name = "projection::project")]
pub fn project(batch: &super::FalconWitness, seed: &[u8]) -> Result<ForwardProjection, AkitaError> {
    let degree = FalconDegree::from_degree(batch.degree()).map_err(invalid)?;
    let geometry = geometry(degree, batch.padded_rows, batch.live_rows)?;
    let cores = tracing::info_span!("projection::cores")
        .in_scope(|| ProjectionCores::derive(seed, &geometry, batch.live_rows));
    let projection = forward::forward(batch, &geometry, cores);
    check_outputs(degree, projection.outputs()).map_err(|_| {
        AkitaError::InvalidInput(format!(
            "{PROJECTION_REJECTED}: the outputs fail a Section 6.1 predicate"
        ))
    })?;
    Ok(projection)
}

/// [`project`] without its predicate check for adversarial prover hooks.
pub(crate) fn project_unchecked(
    batch: &super::FalconWitness,
    seed: &[u8],
) -> Result<ForwardProjection, AkitaError> {
    let degree = FalconDegree::from_degree(batch.degree()).map_err(invalid)?;
    let geometry = geometry(degree, batch.padded_rows, batch.live_rows)?;
    let cores = ProjectionCores::derive(seed, &geometry, batch.live_rows);
    Ok(forward::forward(batch, &geometry, cores))
}

/// Absorb the canonical Rice section, including both parameters and padding,
/// as one kind-1 byte-string atom. The wire codec owns this encoding.
pub fn absorb_outputs(
    outputs: &ProjectionOutputs,
    transcript: &mut impl PioTranscript,
) -> Result<(), AkitaError> {
    let mut bytes = Vec::new();
    outputs.encode_wire(&mut bytes).map_err(invalid)?;
    transcript.absorb_proof("aerie/falcon/v2/projection/output", 1, &bytes)?;
    Ok(())
}

/// Check the public norm and residual predicates.
pub fn check_outputs(degree: FalconDegree, outputs: &ProjectionOutputs) -> Result<(), AkitaError> {
    if !norm_predicate(&outputs.y, norm_cap(degree)) {
        return Err(AkitaError::InvalidProof);
    }
    if !residual_predicate(&outputs.z, FALCON_Q) {
        return Err(AkitaError::InvalidProof);
    }
    Ok(())
}
