//! The grouped Akita opening that discharges the PIOP's group claims.
//!
//! The prover collects one [`OpeningGroup`] per committed polynomial group
//! and proves them in one batched opening; the verifier reconstructs the same
//! per-polynomial claims as a [`ReducedClaims`] and checks that opening.
use super::{
    BaseField, BasisMode, CommittedGroup, Error, GroupHandle, MergedConfig, MergedPcs,
    OpeningClaims, OpeningField, OpeningLedger, OpeningScheduleSelection, PolynomialGroupClaims,
    ProverCore, SelectedProverOpeningData,
};

/// One committed polynomial group and the PIOP's claims on its members.
pub(super) struct OpeningGroup {
    pub(super) committed: CommittedGroup<BaseField>,
    pub(super) handle: GroupHandle,
    pub(super) point: Vec<OpeningField>,
    pub(super) values: Vec<OpeningField>,
}

/// The verifier's reconstruction of every PIOP block: the committed groups
/// and their per-polynomial claims. Nothing is accepted until the opening
/// verifies against these claims.
pub(super) struct ReducedClaims {
    pub(super) session: [u8; 32],
    pub(super) groups: Vec<CommittedGroup<BaseField>>,
    pub(super) claims: Vec<(Vec<OpeningField>, Vec<OpeningField>)>,
    pub(super) ledger: OpeningLedger,
    pub(super) selection: OpeningScheduleSelection,
    pub(super) target_evaluation_ns: u128,
    pub(super) stage_verify_ns: u128,
}

/// Prove the batched multi-group opening of every committed group.
pub(super) fn open_groups(
    groups: Vec<OpeningGroup>,
    setup: &ProverCore,
    session: &[u8; 32],
) -> Result<Vec<u8>, Error> {
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
    let catalog = MergedConfig::catalog()?;
    let opening =
        SelectedProverOpeningData::from_committed_claims::<MergedConfig>(claims, handles, catalog)?;
    MergedPcs::new(catalog.clone())
        .batched_prove(
            &setup.prover,
            opening,
            &setup.backend,
            session,
            BasisMode::Lagrange,
        )
        .map_err(Error::from)
}
