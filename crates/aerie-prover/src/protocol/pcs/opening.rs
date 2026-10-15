//! The grouped Akita opening that discharges the PIOP's group claims.
//!
//! The prover collects one [`OpeningGroup`] per committed polynomial group
//! and proves them in one batched opening; the verifier reconstructs the same
//! per-polynomial claims and checks that opening.
use akita_error::AkitaError;

use akita_pcs::{BasisMode, SelectedProverOpeningData};
use akita_types::{CommittedGroup, OpeningClaims, PolynomialGroupClaims};

use super::GroupHandle;
use super::ProverCore;
pub(super) use crate::protocol::zero_extension::physical_point;
use crate::substrate::{BaseField, GroupOpeningScheme, OpeningField, Width4Config};

/// One committed polynomial group and the PIOP's claims on its members.
pub(super) struct OpeningGroup {
    pub(super) committed: CommittedGroup<BaseField>,
    pub(super) handle: GroupHandle,
    pub(super) point: Vec<OpeningField>,
    pub(super) values: Vec<OpeningField>,
}

/// Prove the batched multi-group opening of every committed group.
pub(super) fn open_groups(
    groups: Vec<OpeningGroup>,
    setup: &ProverCore,
    session: &[u8; 32],
) -> Result<Vec<u8>, AkitaError> {
    let mut claims = Vec::with_capacity(groups.len());
    let mut handles = Vec::with_capacity(groups.len());
    for group in groups {
        claims.push(PolynomialGroupClaims::new(
            group.point,
            group.values,
            group.committed,
        )?);
        handles.push(group.handle);
    }
    let claims = OpeningClaims::from_groups(claims)?;
    let catalog = Width4Config::catalog()?;
    let opening =
        SelectedProverOpeningData::from_committed_claims::<Width4Config>(claims, handles, catalog)?;
    GroupOpeningScheme::new(catalog.clone()).batched_prove(
        &setup.prover,
        opening,
        &setup.backend,
        session,
        BasisMode::Lagrange,
    )
}
