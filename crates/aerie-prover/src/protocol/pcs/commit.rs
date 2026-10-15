//! Manifest-ordered group commitments and early S2 precommitment.

use akita_error::AkitaError;
use akita_pcs::{DensePoly, GroupContext};
use akita_types::CommittedGroup;

use super::group::SignedGroup;
use super::serialized_commitment;
use super::ProverCore;
use super::{Error, FalconAggregateSetup, GroupHandle};
use crate::protocol::manifest::{GroupRole, PcsFamily};
use crate::protocol::source_tables::{slack_group, SourceTables};
use crate::protocol::zero_extension::pcs_length;
use crate::protocol::{AggregationParams, FalconWitness};
use crate::substrate::{BaseField, Width14Config, Width4Config};

/// One committed group: the transcript payload, the committed group, the
/// retained handle the opening reuses, and the commit wall time.
pub(super) struct PreparedCommitment {
    pub(super) payload: Vec<u8>,
    pub(super) committed: CommittedGroup<BaseField>,
    pub(super) handle: GroupHandle,
    pub(super) group_ns: u128,
}

/// Commit one polynomial under its own family's catalog and return the
/// committed group with a handle owned by the proving backend.
fn commit_polynomials(
    setup: &ProverCore,
    polynomials: Vec<DensePoly<BaseField>>,
    width14: bool,
    context: GroupContext<'_>,
) -> Result<(CommittedGroup<BaseField>, GroupHandle), AkitaError> {
    let backend = &setup.backend;
    let source = tracing::info_span!(
        "v1.commit.group.import_source",
        width = if width14 { 14 } else { 4 }
    )
    .in_scope(|| backend.import_source(polynomials))?;
    let out = tracing::info_span!(
        "v1.commit.group.backend_commit",
        width = if width14 { 14 } else { 4 }
    )
    .in_scope(|| {
        if width14 {
            backend.commit(Width14Config::catalog()?, &source, context)
        } else {
            backend.commit(Width4Config::catalog()?, &source, context)
        }
    })?;
    Ok((out.committed_group, out.private_handle))
}

/// Pure per-group commit: transcript-free, so groups commit in parallel and
/// their payloads are absorbed afterwards in canonical order (the transcript
/// sees exactly the sequential byte stream). The final manifest group
/// commits as the final group against the setup's precommitted profiles.
pub(super) fn commit_group(
    setup: &ProverCore,
    role: GroupRole,
    cells: &SignedGroup<'_>,
) -> Result<PreparedCommitment, Error> {
    let group_start = std::time::Instant::now();
    let descriptor = setup
        .manifest
        .groups()
        .iter()
        .find(|descriptor| descriptor.role == role)
        .ok_or_else(|| Error::internal("committed group is absent from the manifest"))?;
    let width14 = descriptor.family == PcsFamily::Width14;
    let is_final = setup.manifest.groups().last().map(|last| last.role) == Some(role);
    let context = if is_final {
        GroupContext::scheduler_with_precommitted_groups(&setup.precommitted)
    } else {
        GroupContext::scheduler_without_precommitted_groups()
    };
    tracing::info_span!(
        "v1.commit.group",
        role = ?role,
        logical_cells_per_polynomial = cells.logical_cells(),
        polynomial_count = cells.polynomial_count(),
        physical_cells_per_polynomial = pcs_length(cells.logical_cells()),
        width = if width14 { 14 } else { 4 }
    )
    .in_scope(|| {
        if descriptor.logical_cells != cells.logical_cells()
            || descriptor.logical_polynomials != cells.polynomial_count()
        {
            return Err(Error::internal(
                "committed source shape differs from the manifest",
            ));
        }
        let polynomials =
            tracing::info_span!("v1.commit.group.materialize").in_scope(|| cells.polynomials())?;
        let (committed, handle) = tracing::info_span!("v1.commit.group.pcs")
            .in_scope(|| commit_polynomials(setup, polynomials, width14, context))?;
        let payload = tracing::info_span!("v1.commit.group.serialize")
            .in_scope(|| serialized_commitment(committed.commitment()))?;
        Ok::<_, Error>(PreparedCommitment {
            payload,
            committed,
            handle,
            group_ns: group_start.elapsed().as_nanos(),
        })
    })
}

/// The two groups that depend on `s2` alone (`S2Sources`, `Budget`), committed
/// ahead of native verification so the memory-bound Akita commits overlap
/// the compute-bound ingestion. The transcript is untouched here: the
/// prover absorbs these payloads in canonical slot order once the `S1` and
/// `slack` commits exist, so the proof bytes are those of the sequential
/// prover.
pub struct S2Precommit {
    pub(super) tables: SourceTables,
    pub(super) groups: Vec<(GroupRole, PreparedCommitment)>,
    pub(super) encodability_ns: u128,
    pub(super) commit_ns: u128,
}

/// Build the source tables from the padded `s2` table and commit the two
/// `s2`-only groups for [`super::prove_prepared_with_precommit`].
pub fn precommit_s2_groups<P: AggregationParams>(
    setup: &FalconAggregateSetup<P>,
    padded_rows: usize,
    s2: &[i16],
) -> Result<S2Precommit, Error> {
    let mut encodability_ns: u128 = 0;
    let tables = aerie_falcon_statement::staged!(
        "v1.prove.witness.encodability",
        encodability_ns,
        SourceTables::build_from_s2(P::DEGREE, padded_rows, s2,)?
    );
    let groups: Vec<(GroupRole, SignedGroup<'_>)> = setup
        .core
        .manifest
        .groups()
        .iter()
        .map(|descriptor| descriptor.role)
        .filter(|role| is_s2_only_role(*role))
        .filter_map(|role| s2_group(role, &tables).map(|group| (role, group)))
        .collect();
    let mut commit_ns: u128 = 0;
    let committed = aerie_falcon_statement::staged!("v1.commit.s2_early", commit_ns, {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            groups
                .par_iter()
                .map(|(role, cells)| {
                    commit_group(&setup.core, *role, cells).map(|group| (*role, group))
                })
                .collect::<Result<Vec<_>, Error>>()?
        }
        #[cfg(not(feature = "parallel"))]
        {
            groups
                .iter()
                .map(|(role, cells)| {
                    commit_group(&setup.core, *role, cells).map(|group| (*role, group))
                })
                .collect::<Result<Vec<_>, Error>>()?
        }
    });
    drop(groups);
    Ok(S2Precommit {
        tables,
        groups: committed,
        encodability_ns,
        commit_ns,
    })
}

/// Whether a group is built entirely from the decomposed `s2` source tables.
pub(super) const fn is_s2_only_role(role: GroupRole) -> bool {
    matches!(role, GroupRole::S2Sources | GroupRole::Budget)
}

/// The committed table of an `s2`-only group, borrowed from the source
/// tables in their own storage.
fn s2_group(role: GroupRole, tables: &SourceTables) -> Option<SignedGroup<'_>> {
    match role {
        GroupRole::S2Sources => Some(SignedGroup::S2Sources {
            digits: [tables.a0_view(), tables.a1_view(), tables.a2_view()],
            signs: tables.sigma_view(),
        }),
        GroupRole::Budget => Some(SignedGroup::MultiI8Three([
            tables.budget(0),
            tables.budget(1),
            tables.budget(2),
        ])),
        GroupRole::S1 | GroupRole::Slack => None,
    }
}

/// The committed table of any manifest group.
pub(super) fn signed_group<'a>(
    role: GroupRole,
    batch: &'a FalconWitness,
    tables: &'a SourceTables,
) -> SignedGroup<'a> {
    match role {
        GroupRole::S1 => SignedGroup::I16(&batch.s1),
        GroupRole::Slack => SignedGroup::I16(slack_group(batch)),
        _ => s2_group(role, tables).expect("every other role is s2-only"),
    }
}
