//! Clear structured projections and their public predicates.
//!
//! Stage one absorbs the canonical Rice section of the projection outputs,
//! then proves the source-link identity with its other basis-16 instances.
//! This module owns the unchanged projection profile, core derivation,
//! forward maps, predicates, and the Rice wire codec.

mod codec;
#[doc(hidden)]
pub mod cores;
mod predicates;
#[doc(hidden)]
pub mod profile;
#[doc(hidden)]
pub mod reverse;
#[cfg(test)]
mod tests;

use super::transcript::PioTranscript;
use crate::substrate::BaseField as F;
use crate::{AggregationError, UnsupportedBatch};

use super::FalconDegree;

#[cfg(feature = "internals")]
pub use codec::{CodecError, MAX_ENCODED_BYTES, MAX_RICE_PARAMETER};
#[cfg(feature = "internals")]
pub use cores::{CORE_XOF_LABEL, TERMINAL_GROUP};
#[cfg(any(test, feature = "internals"))]
pub use predicates::{centered, FIELD_MODULUS};
pub use predicates::{norm_predicate, residual_predicate};
pub use profile::{
    norm_cap, ProjectionProfile, FALCON_Q, NORM_TERMINAL_ROWS, OUTPUT_COUNT, RESIDUAL_WIDTH,
};
#[cfg(any(test, feature = "internals"))]
pub use profile::{
    NORM_CAP_FALCON_1024, NORM_CAP_FALCON_512, NORM_CHANNELS, NORM_GROUPS, NORM_PREFIX_LAYERS,
    NORM_PREFIX_WIDTH, PROFILE_BINDING_LABEL, RESIDUAL_LAYERS,
};

#[cfg(feature = "internals")]
pub use profile::{
    CoreDistribution, NormLeg, NormSourceOrder, ResidualLeg, FIRST_CHANNELS, RESIDUAL_BOUND,
};

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

/// Shared shape validation of a canonically padded batch.
pub fn geometry(
    degree: FalconDegree,
    records: usize,
    live_rows: usize,
) -> Result<Geometry, AggregationError> {
    if live_rows == 0 || live_rows > records || degree.padded_records(live_rows) != Ok(records) {
        return Err(UnsupportedBatch::Shape {
            degree: degree.degree(),
            records,
        }
        .into());
    }
    Geometry::new(degree, records).map_err(Into::into)
}

/// Absorb the canonical Rice section, including both parameters and padding,
/// as one kind-1 byte-string atom. The wire codec owns this encoding.
pub fn absorb_outputs(
    outputs: &ProjectionOutputs,
    transcript: &mut impl PioTranscript,
) -> Result<(), AggregationError> {
    let mut bytes = Vec::new();
    outputs
        .encode_wire(&mut bytes)
        .map_err(|_| crate::MalformedProof::ProjectionEncoding)?;
    transcript.absorb_proof("aerie/falcon/v1/projection/output", 1, &bytes)?;
    Ok(())
}

/// Check the public norm and residual predicates.
#[must_use]
pub fn check_outputs(degree: FalconDegree, outputs: &ProjectionOutputs) -> bool {
    norm_predicate(&outputs.y, norm_cap(degree)) && residual_predicate(&outputs.z, FALCON_Q)
}
