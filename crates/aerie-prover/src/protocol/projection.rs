//! Producer clear projection.
use super::FalconDegree;
use crate::AggregationError;
pub use aerie_falcon::protocol::projection::*;
use cores::ProjectionCores;
pub(crate) mod forward;
mod sources;
pub use forward::ForwardProjection;
/// Derive the cores, run the forward projection and check both predicates.
#[tracing::instrument(skip_all, name = "projection::project")]
pub fn project(
    batch: &super::FalconWitness,
    seed: &[u8],
) -> Result<ForwardProjection, AggregationError> {
    let degree = FalconDegree::from_degree(batch.degree())?;
    let geometry = geometry(degree, batch.padded_rows, batch.live_rows)?;
    let cores = tracing::info_span!("projection::cores")
        .in_scope(|| ProjectionCores::derive(seed, &geometry, batch.live_rows));
    let projection = forward::forward(batch, &geometry, cores);
    if !check_outputs(degree, projection.outputs()) {
        return Err(AggregationError::ProjectionRejected);
    }
    Ok(projection)
}

/// [`project`] without its predicate check for adversarial prover hooks.
#[cfg(any(test, feature = "internals"))]
pub(crate) fn project_unchecked(
    batch: &super::FalconWitness,
    seed: &[u8],
) -> Result<ForwardProjection, AggregationError> {
    let degree = FalconDegree::from_degree(batch.degree())?;
    let geometry = geometry(degree, batch.padded_rows, batch.live_rows)?;
    let cores = ProjectionCores::derive(seed, &geometry, batch.live_rows);
    Ok(forward::forward(batch, &geometry, cores))
}

#[cfg(test)]
mod tests;
