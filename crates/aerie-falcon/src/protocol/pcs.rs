//! Real Akita manifest-defined commitments and the grouped multi-group opening.
//!
//! Production aggregation splits into a serializable [`prove`] half and a wire-style
//! [`verify`] half. Both parameter sets commit the same four groups before
//! any challenge (`specs/falcon.md` Section 5). Every group except the last
//! is precommitted in transcript order (width-14 for `S_1` and slack,
//! width-4 for the grouped source tables), and the last manifest group is
//! the width-4 Budget group as final. The precommitted profiles are
//! public catalog data, so the final group commits concurrently with the
//! precommits. Groups below the `2^13` floor commit zero-extended, and each
//! opening point appends all-zero pad coordinates, where the zero-extension
//! tie is the exact identity `P_phys~(r, 0) = P_logical~(r)`.
//!
//! Trust discipline: the verifier reconstructs every committed-group profile and the public schedule
//! selection from the statement shape and the generated catalogs, never
//! from prover-supplied data; only the commitment payloads and proof
//! messages cross the wire. Commitment payloads are absorbed as their
//! canonical compressed serialization under the group's own label, so the
//! prover and a byte-level verifier bind identical transcripts.

use std::marker::PhantomData;
use std::sync::Arc;

use aerie_falcon_statement::prepare_direct_witness_convolution;
use akita_params::{
    CommittedGroupBatchProfile, GroupCommitPhaseParams, OpeningScheduleSelection,
    PolynomialGroupLayout, PrecommittedGroupProfiles, ScheduleLookupKey,
};
use akita_pcs::{
    new_prover_setup, AkitaProverSetup, AkitaVerifier, BasisMode, CommitmentHandle, CpuBackend,
    DensePoly, GroupContext, SelectedProverOpeningData, SetupRequirements,
};
use akita_serialization::{AkitaSerialize, Compress};
use akita_types::{
    Commitment, CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims,
};
use jolt_field::{Ring as _, Zero as _};

use crate::substrate::{BaseField, ConservativeMainConfig, MergedConfig, MergedPcs, OpeningField};

use super::manifest::{CommitmentManifest, GroupRole, PcsFamily, GROUP_COUNT};
use super::openings::{opening_ledger, OpeningLedger};
use super::projection::{self, ProjectionOutputs};
use super::public_inputs::PublicInputs;
use super::{
    stage1::{self, Stage1Proof},
    stage2::{self, Stage2Proof},
    stage3::{self, Stage3Proof},
};

#[cfg(test)]
use super::ring::{PublicRingKeys, PublicTargets};
use super::source_tables::{slack_group, PreparedCoefficients, SourceTables};
use super::transcript::{ProtocolTranscript, VerifierProtocolTranscript};
use super::zero_extension::{pad_vars, pcs_length};
use super::{instance, wire::SaltMode};
use super::{AggregationParams, FalconDegree, FalconWitness};

mod group;
mod opening;
use group::{delta_evaluation, SignedGroup};
use opening::{open_groups, OpeningGroup, ReducedClaims};

// Measurements showed that smaller padded batches do not repay the scheduling
// cost of overlapping stage two with stage-three preparation.
#[cfg(feature = "parallel")]
const STAGE3_PREPARE_OVERLAP_MIN_ROWS: usize = 256;

struct PiopOutput {
    stage1_proof: Stage1Proof,
    stage1_ns: u128,
    stage2_proof: Stage2Proof,
    stage2_ns: u128,
    stage3_proof: Stage3Proof,
    third: stage3::Stage3Output,
    stage3_ns: u128,
}

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

/// Prover-side stage instrumentation (Section 12). Nanosecond fields
/// follow the `aerie_falcon_statement::staged!` convention; timings measured in managed
/// sessions are not comparable to host numbers.
#[derive(Clone, Copy, Debug, Default)]
pub struct PcsStageMetrics {
    pub records: usize,
    pub padded_records: usize,
    pub encodability_ns: u128,
    pub group_materialization_ns: u128,
    pub setup_ns: u128,
    pub commit_ns: u128,
    pub projection_ns: u128,
    pub stage1_ns: u128,
    pub stage2_ns: u128,
    pub stage3_ns: u128,
    pub pcs_prove_ns: u128,
    /// Public-target evaluation in the configured Rayon pool.
    pub target_evaluation_ns: u128,
    /// Protocol-block verification excluding public-target evaluation.
    pub stage_verify_ns: u128,
    pub pcs_verify_ns: u128,
    /// The same verifier replayed inside a dedicated one-worker Rayon pool.
    pub target_evaluation_single_thread_ns: u128,
    pub stage_verify_single_thread_ns: u128,
    pub pcs_verify_single_thread_ns: u128,
    /// Per-group commit wall time in canonical manifest order (Section
    /// 26.4). Groups commit inside a parallel fan-out, so entries overlap
    /// and do not sum to `commit_ns`.
    pub commit_group_ns: [u128; GROUP_COUNT],
    pub commitment_bytes: usize,
    pub pcs_proof_bytes: usize,
}

/// A retained group commitment, owned by the backend that proves the
/// opening.
type GroupHandle = CommitmentHandle<BaseField, OpeningField>;

/// Parameter-set-agnostic prover state: the Akita prover setup, the owning
/// CPU backend over its expanded matrices, and the logical group widths.
/// The typed [`FalconAggregateSetup`] owns one; the untyped [`prove`] path builds
/// one from the witness's runtime degree.
pub(crate) struct ProverCore {
    degree: FalconDegree,
    records: usize,
    #[cfg(test)]
    instance_mutation: Option<usize>,
    #[cfg(test)]
    session_mutation: bool,
    #[cfg(test)]
    site_mutation: bool,
    manifest: CommitmentManifest,
    logical_cells: Vec<usize>,
    /// The catalog profiles of every precommitted group in manifest order.
    /// They are public shape data, so the final group commits against them
    /// without waiting for the precommits.
    precommitted: PrecommittedGroupProfiles,
    prover: AkitaProverSetup<BaseField>,
    /// Commits every group, each under its own family's catalog, and proves
    /// the batched opening.
    backend: CpuBackend<BaseField, OpeningField>,
}

/// One Akita prover setup large enough for both commitment families.
///
/// The width-4 and width-14 groups commit on the same family-agnostic
/// backend, so the setup is sized by the union of both catalogs'
/// requirements rather than by either catalog alone.
fn shared_prover_setup(
    max_physical_vars: usize,
    polys: usize,
) -> Result<AkitaProverSetup<BaseField>, Error> {
    let requirements =
        SetupRequirements::from_catalog(MergedConfig::catalog()?, max_physical_vars, polys)?
            .union(SetupRequirements::from_catalog(
                ConservativeMainConfig::catalog()?,
                max_physical_vars,
                polys,
            )?)?;
    new_prover_setup(&requirements).map_err(Into::into)
}

impl ProverCore {
    #[tracing::instrument(skip_all, fields(records), name = "v1.setup.total")]
    fn new(degree: FalconDegree, records: usize) -> Result<Self, Error> {
        tracing::info_span!("v1.setup.witness_ntt").in_scope(prepare_direct_witness_convolution);
        let manifest = target_manifest(degree, records)?;
        let logical_cells = group_cells_of(&manifest);
        let max_physical_vars = max_physical_vars(&logical_cells);
        let prover = shared_prover_setup(max_physical_vars, total_group_polynomials(&manifest))?;
        let backend = CpuBackend::new(Arc::clone(&prover.expanded))?;
        let (precommitteds, final_group, _) = verifier_group_profiles_of(degree, records)?;
        let schedule = MergedConfig::catalog()?.resolve_key(&ScheduleLookupKey {
            final_group: final_group.group,
            precommitteds: precommitteds.clone(),
        })?;
        backend.prewarm(schedule.schedule())?;
        let precommitted = PrecommittedGroupProfiles::from_profiles(precommitteds)?;
        tracing::info_span!("v1.setup.compression_caches")
            .in_scope(|| warm_compression_caches(&backend, &manifest, &precommitted))?;
        Ok(Self {
            degree,
            records,
            #[cfg(test)]
            instance_mutation: None,
            #[cfg(test)]
            session_mutation: false,
            #[cfg(test)]
            site_mutation: false,
            manifest,
            logical_cells,
            precommitted,
            prover,
            backend,
        })
    }

    /// The public verifier derived from this prover setup.
    fn verifier(&self) -> Result<AkitaVerifier<MergedConfig>, Error> {
        let pcs = MergedPcs::new(MergedConfig::catalog()?.clone());
        Ok(pcs.verifier(pcs.setup_verifier(&self.prover)?)?)
    }
}

/// Reusable, shape-dependent Akita state for proving and verifying
/// aggregates of parameter set `P` at one live record count.
///
/// Construction depends on the parameter set, the padded record count, and
/// the generated PCS catalogs, never on a public key, signature, salt, or
/// message. Callers should build one value outside online prove/verify
/// timing and reuse it for every statement of the same size.
pub struct FalconAggregateSetup<P: AggregationParams> {
    core: ProverCore,
    verifier: FalconVerifierSetup<P>,
}

impl<P: AggregationParams> FalconAggregateSetup<P> {
    /// Whether the generated PCS catalog supports this live record count.
    /// Uses the same padded geometry and schedule lookup as setup and the
    /// verifier, without allocating or preparing prover state.
    #[must_use]
    pub fn supports(records: usize) -> bool {
        FalconVerifierSetup::<P>::supports(records)
    }

    /// Build reusable prover and verifier state for one live record count.
    pub fn new(records: usize) -> Result<Self, Error> {
        let core = ProverCore::new(P::DEGREE_ID, records)?;
        let verifier = FalconVerifierSetup {
            records,
            akita: core.verifier()?,
            params: PhantomData,
        };
        Ok(Self { core, verifier })
    }

    /// The live record count this setup was built for.
    #[must_use]
    pub const fn records(&self) -> usize {
        self.core.records
    }

    /// The public verifier half.
    #[must_use]
    pub const fn verifier(&self) -> &FalconVerifierSetup<P> {
        &self.verifier
    }
}

/// The public verifier setup for aggregates of parameter set `P` at one
/// live record count. Holds no prover state; a verifier builds it with
/// [`FalconVerifierSetup::new`] or borrows it from a [`FalconAggregateSetup`].
pub struct FalconVerifierSetup<P: AggregationParams> {
    records: usize,
    akita: AkitaVerifier<MergedConfig>,
    params: PhantomData<fn() -> P>,
}

impl<P: AggregationParams> FalconVerifierSetup<P> {
    /// Whether the generated PCS catalog supports this live record count.
    #[must_use]
    pub fn supports(records: usize) -> bool {
        verifier_group_profiles_of(P::DEGREE_ID, records).is_ok()
    }

    /// Build the public verifier setup for one live record count.
    pub fn new(records: usize) -> Result<Self, Error> {
        let manifest = target_manifest(P::DEGREE_ID, records)?;
        let cells = group_cells_of(&manifest);
        let setup = shared_prover_setup(
            max_physical_vars(&cells),
            total_group_polynomials(&manifest),
        )?;
        let pcs = MergedPcs::new(MergedConfig::catalog()?.clone());
        Ok(Self {
            records,
            akita: pcs.verifier(pcs.setup_verifier(&setup)?)?,
            params: PhantomData,
        })
    }

    /// The live record count this setup was built for.
    #[must_use]
    pub const fn records(&self) -> usize {
        self.records
    }

    /// Reconstruct the canonical instance for adversarial transcript tests.
    #[cfg(feature = "adversarial-internals")]
    #[doc(hidden)]
    pub fn transcript_instance<C: Copy + Sync + Into<i64>>(
        &self,
        public: &PublicInputs<'_, C>,
    ) -> Result<[u8; instance::INSTANCE_BYTES], Error> {
        let degree = public.degree()?;
        let manifest = CommitmentManifest::for_batch_of(degree, public.records)?;
        let (_, _, selection) = verifier_group_profiles_of(degree, public.records)?;
        verifier_instance(public, &manifest, selection, &self.akita)
    }
}

/// Verifier output: the discharged opening ledger and the verify timings.
#[derive(Clone, Debug)]
pub struct VerifiedOpening {
    pub ledger: OpeningLedger,
    /// Wall time spent streaming the public targets and contracting them at `r_Pi`.
    pub target_evaluation_ns: u128,
    /// Wall time for protocol-block verification excluding target evaluation.
    pub stage_verify_ns: u128,
    pub pcs_verify_ns: u128,
}

use crate::AggregationError as Error;

fn serialized_commitment(commitment: &Commitment<BaseField>) -> Result<Vec<u8>, Error> {
    if commitment.rows().coeff_len() != 16 {
        return Err("v1 commitments must contain 16 canonical base coefficients".into());
    }
    let mut bytes = Vec::with_capacity(128);
    commitment.serialize_with_mode(&mut bytes, Compress::Yes)?;
    debug_assert_eq!(bytes.len(), 128);
    Ok(bytes)
}

/// One committed group: the transcript payload, the committed group, the
/// retained handle the opening reuses, and the commit wall time.
struct PreparedCommitment {
    payload: Vec<u8>,
    committed: CommittedGroup<BaseField>,
    handle: GroupHandle,
    group_ns: u128,
}

/// Commit one polynomial under its own family's catalog and return the
/// committed group with a handle owned by the proving backend.
fn commit_polynomials(
    setup: &ProverCore,
    polynomials: Vec<DensePoly<BaseField>>,
    width14: bool,
    context: GroupContext<'_>,
) -> Result<(CommittedGroup<BaseField>, GroupHandle), Error> {
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
            backend.commit(ConservativeMainConfig::catalog()?, &source, context)
        } else {
            backend.commit(MergedConfig::catalog()?, &source, context)
        }
    })?;
    Ok((out.committed_group, out.private_handle))
}

/// Pure per-group commit: transcript-free, so groups commit in parallel and
/// their payloads are absorbed afterwards in canonical order (the transcript
/// sees exactly the sequential byte stream). The final manifest group
/// commits as the final group against the setup's precommitted profiles.
fn commit_group(
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
        .ok_or_else(|| format!("{role:?} is not a manifest group"))?;
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
            return Err(format!(
                "{role:?} source shape is {} cells by {} polynomials, manifest requires {} by {}",
                cells.logical_cells(),
                cells.polynomial_count(),
                descriptor.logical_cells,
                descriptor.logical_polynomials
            )
            .into());
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
    tables: SourceTables,
    groups: Vec<(GroupRole, PreparedCommitment)>,
    encodability_ns: u128,
    commit_ns: u128,
}

/// Build the source tables from the padded `s2` table and commit the two
/// `s2`-only groups for [`prove_prepared_with_precommit`].
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
const fn is_s2_only_role(role: GroupRole) -> bool {
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
fn signed_group<'a>(
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

fn group_polynomial<C>(cells: &[C]) -> Result<DensePoly<BaseField>, Error>
where
    C: Copy + Sync + Into<i64>,
{
    let physical_len = pcs_length(cells.len());
    let mut evaluations = tracing::info_span!(
        "v1.commit.group.field_conversion",
        logical_cells = cells.len(),
        physical_cells = physical_len
    )
    .in_scope(|| {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            cells
                .par_iter()
                .map(|&cell| BaseField::from_i64(cell.into()))
                .collect::<Vec<_>>()
        }
        #[cfg(not(feature = "parallel"))]
        {
            cells
                .iter()
                .map(|&cell| BaseField::from_i64(cell.into()))
                .collect::<Vec<_>>()
        }
    });
    tracing::info_span!(
        "v1.commit.group.dense_poly",
        logical_cells = cells.len(),
        physical_cells = physical_len
    )
    .in_scope(|| {
        evaluations.resize(physical_len, BaseField::zero());
        DensePoly::from_field_evals(physical_len.trailing_zeros() as usize, evaluations)
            .map_err(Into::into)
    })
}

/// The physical opening point of one ledger claim: the logical point with
/// all-zero trailing pad coordinates, making the zero-extension tie exact.
fn physical_point(logical_point: &[OpeningField], logical_cells: usize) -> Vec<OpeningField> {
    let mut point = logical_point.to_vec();
    point.extend(std::iter::repeat_n(
        OpeningField::zero(),
        pad_vars(logical_cells),
    ));
    point
}

/// Commit zero-valued groups of every production shape once, serially, at
/// setup time. The Akita CPU backend builds its compression NTT cache
/// lazily per digit width under a `OnceLock` whose initializer itself runs
/// rayon work. When the group commits race for an
/// uninitialized cache from inside the pool, the waiters pin the workers
/// the initializer needs and the prover deadlocks at zero CPU (reproduced
/// three times out of three on a 14-core host; the 4-vCPU CI runners never
/// hit it). Warming every (shape, width) the prover commits keeps the
/// parallel commit path and moves the one-time cache cost into setup,
/// outside every prover timer. Commitments are transcript-free, so
/// the proof bytes are unchanged.
fn warm_compression_caches(
    backend: &CpuBackend<BaseField, OpeningField>,
    manifest: &CommitmentManifest,
    precommitted: &PrecommittedGroupProfiles,
) -> Result<(), Error> {
    // Manifest order, families and commit contexts: the cache shape depends
    // only on the physical arity, the width and the scheduled commit
    // parameters, so zero tables of each logical length suffice.
    let final_slot = manifest.group_count() - 1;
    for (slot, descriptor) in manifest.groups().iter().enumerate() {
        let group = SignedGroup::OwnedI64(vec![
            vec![0_i64; descriptor.logical_cells];
            descriptor.logical_polynomials
        ]);
        let source = backend.import_source(group.polynomials()?)?;
        let context = if slot == final_slot {
            GroupContext::scheduler_with_precommitted_groups(precommitted)
        } else {
            GroupContext::scheduler_without_precommitted_groups()
        };
        match descriptor.family {
            PcsFamily::Width14 => {
                backend.commit(ConservativeMainConfig::catalog()?, &source, context)?;
            }
            PcsFamily::Width4 => {
                backend.commit(MergedConfig::catalog()?, &source, context)?;
            }
        }
    }
    Ok(())
}

/// The canonical manifest of a parameter set for the target representation
/// profiles; group lengths and families do not depend on the profiles.
fn target_manifest(degree: FalconDegree, records: usize) -> Result<CommitmentManifest, Error> {
    let manifest = CommitmentManifest::for_batch_of(degree, records)?;
    Ok(manifest)
}

/// Exact per-polynomial logical cell counts of every group in canonical order.
fn group_cells_of(manifest: &CommitmentManifest) -> Vec<usize> {
    manifest
        .groups()
        .iter()
        .map(|descriptor| descriptor.logical_cells)
        .collect()
}

fn max_physical_vars(logical_cells: &[usize]) -> usize {
    logical_cells
        .iter()
        .map(|&cells| pcs_length(cells).trailing_zeros() as usize)
        .max()
        .expect("a manifest has at least one group")
}

fn total_group_polynomials(manifest: &CommitmentManifest) -> usize {
    manifest
        .groups()
        .iter()
        .map(|descriptor| descriptor.logical_polynomials)
        .sum()
}

/// Verifier-derived group profiles and the public schedule selection,
/// computed from the statement shape and the generated catalogs alone
/// ("verifier-derivable; never prover-supplied"). Every group but the last
/// is a precommitted group in manifest order; the last is the final group.
pub(crate) fn verifier_group_profiles_of(
    degree: FalconDegree,
    records: usize,
) -> Result<
    (
        Vec<GroupCommitPhaseParams>,
        GroupCommitPhaseParams,
        OpeningScheduleSelection,
    ),
    Error,
> {
    let manifest = target_manifest(degree, records)?;
    let groups = manifest.groups();
    let layout = |descriptor: &super::manifest::GroupDescriptor| {
        PolynomialGroupLayout::new(
            pcs_length(descriptor.logical_cells).trailing_zeros() as usize,
            descriptor.logical_polynomials,
        )
    };
    let (final_group, precommitted) = groups
        .split_last()
        .ok_or("a manifest has at least one group")?;
    let mut precommits = Vec::with_capacity(precommitted.len());
    for descriptor in precommitted {
        let profile = match descriptor.family {
            PcsFamily::Width14 => ConservativeMainConfig::commitment_profile(layout(descriptor))?,
            PcsFamily::Width4 => MergedConfig::commitment_profile(layout(descriptor))?,
        };
        precommits.push(profile);
    }
    if final_group.family != PcsFamily::Width4 {
        return Err("the final group must commit under the width-4 family".into());
    }
    let key = ScheduleLookupKey {
        final_group: layout(final_group),
        precommitteds: precommits.clone(),
    };
    let row = MergedConfig::catalog()?.resolve_key(&key)?;
    Ok((precommits, row.profiles().final_group, row.selection()))
}

/// Shaped wire context for the codec: the manifest's commitment coefficient
/// counts in canonical order, derived from the statement shape and the
/// generated catalogs.
pub(crate) fn commitment_coefficient_counts_of(
    degree: FalconDegree,
    records: usize,
) -> Result<Vec<usize>, Error> {
    let (precommits, final_profile, _) = verifier_group_profiles_of(degree, records)?;
    let mut coefficient_counts = Vec::with_capacity(precommits.len() + 1);
    for profile in precommits.iter().chain(std::iter::once(&final_profile)) {
        coefficient_counts.push(crate::substrate::commitment_coefficients(profile)?);
    }
    Ok(coefficient_counts)
}

/// Prove the complete version 1 protocol with real commitments, building
/// a one-off setup for the witness's parameter set and record count.
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn prove(
    batch: &FalconWitness,
    public_records: &[u8],
) -> Result<(AggregateProof, PcsStageMetrics), Error> {
    let setup_start = std::time::Instant::now();
    let degree = FalconDegree::from_degree(batch.degree())?;
    let setup = tracing::info_span!("v1.setup.prover")
        .in_scope(|| ProverCore::new(degree, batch.live_rows))?;
    let mut result = prove_prepared_impl(
        batch,
        public_records,
        SaltMode::PerRecord,
        &setup,
        None,
        None,
        None,
        None,
    )?;
    result.1.setup_ns = setup_start.elapsed().as_nanos();
    Ok(result)
}

/// Prove with reusable shape-dependent PCS state.
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn prove_prepared<P: AggregationParams>(
    batch: &FalconWitness,
    public_records: &[u8],
    setup: &FalconAggregateSetup<P>,
) -> Result<(AggregateProof, PcsStageMetrics), Error> {
    prove_prepared_impl(
        batch,
        public_records,
        SaltMode::PerRecord,
        &setup.core,
        None,
        None,
        None,
        None,
    )
}

/// [`prove_prepared`] with the `s2`-only groups already committed by
/// [`precommit_s2_groups`], typically concurrently with the ingestion that
/// produced `batch`. The tables and commitments must have been built for
/// this batch's degree, padded size and profiles; the proof bytes are
/// identical to [`prove_prepared`]'s.
pub fn prove_prepared_with_precommit<P: AggregationParams>(
    batch: &FalconWitness,
    public_records: &[u8],
    salt_mode: SaltMode,
    setup: &FalconAggregateSetup<P>,
    precommit: S2Precommit,
) -> Result<(AggregateProof, PcsStageMetrics), Error> {
    prove_prepared_impl(
        batch,
        public_records,
        salt_mode,
        &setup.core,
        None,
        None,
        None,
        Some(precommit),
    )
}

/// Adversarial-test hook: apply a cheating witness consistently to both the
/// committed groups and the PIOP before producing the real PCS opening proof.
///
/// Source-table edits happen before the four commitments. Output edits see
/// the projection seed and replace the forward projection's `(y, z)` before
/// they are absorbed, so stage one proves against the edited outputs; with
/// an output hook the prover skips its own predicate check.
/// Unlike [`prove_with_group_tamper`], this hook proves against the edited
/// witness rather than intentionally creating a commitment/ledger mismatch.
#[cfg(feature = "adversarial-internals")]
#[doc(hidden)]
pub fn prove_prepared_with_witness_tamper<P: AggregationParams>(
    batch: &FalconWitness,
    public_records: &[u8],
    setup: &FalconAggregateSetup<P>,
    tamper_tables: impl Fn(&mut SourceTables),
    tamper_outputs: impl Fn(&[u8], &mut ProjectionOutputs),
) -> Result<(AggregateProof, PcsStageMetrics), Error> {
    prove_prepared_impl(
        batch,
        public_records,
        SaltMode::PerRecord,
        &setup.core,
        None,
        Some(&tamper_tables),
        Some(&tamper_outputs),
        None,
    )
}

/// Adversarial-test hook (Section 12): commit tampered group cells while
/// proving the PIOP against the honest witness. The batched opening proves
/// the committed (tampered, zero-extended) polynomials' true evaluations,
/// which the Akita prover requires, so a proof is always emitted; the
/// verifier's ledger claim then no longer ties to the committed physical
/// table, and the multi-group PCS verifier must reject.
#[cfg(feature = "adversarial-internals")]
#[doc(hidden)]
pub fn prove_with_group_tamper(
    batch: &FalconWitness,
    public_records: &[u8],
    tamper: &dyn Fn(GroupRole, &mut Vec<i64>),
) -> Result<(AggregateProof, PcsStageMetrics), Error> {
    let setup_start = std::time::Instant::now();
    let degree = FalconDegree::from_degree(batch.degree())?;
    let setup = tracing::info_span!("v1.setup.prover")
        .in_scope(|| ProverCore::new(degree, batch.live_rows))?;
    let mut result = prove_prepared_impl(
        batch,
        public_records,
        SaltMode::PerRecord,
        &setup,
        Some(tamper),
        None,
        None,
        None,
    )?;
    result.1.setup_ns = setup_start.elapsed().as_nanos();
    Ok(result)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn prove_prepared_impl(
    batch: &FalconWitness,
    public_records: &[u8],
    salt_mode: SaltMode,
    setup: &ProverCore,
    tamper: Option<&dyn Fn(GroupRole, &mut Vec<i64>)>,
    tamper_tables: Option<&dyn Fn(&mut SourceTables)>,
    tamper_outputs: Option<&dyn Fn(&[u8], &mut ProjectionOutputs)>,
    precommit: Option<S2Precommit>,
) -> Result<(AggregateProof, PcsStageMetrics), Error> {
    let records = batch.live_rows;
    if setup.records != records {
        return Err(format!(
            "PCS setup has {} records but the witness has {records}",
            setup.records
        )
        .into());
    }
    if setup.degree.degree() != batch.degree() {
        return Err(format!(
            "PCS setup is for Falcon-{} but the witness has degree {}",
            setup.degree.degree(),
            batch.degree()
        )
        .into());
    }
    let mut padded_batch;
    let batch = if batch.padded_rows < setup.manifest.padded_records() {
        padded_batch = batch.clone();
        super::pad_witness(&mut padded_batch)?;
        &padded_batch
    } else {
        batch
    };
    let manifest = setup.manifest;
    if manifest.padded_records() != batch.padded_rows {
        return Err("batch is not canonically padded".into());
    }
    let mut encodability_ns: u128 = 0;
    let mut early_commit_ns: u128 = 0;
    let (mut tables, early_groups) = if let Some(early) = precommit {
        if tamper.is_some() || tamper_tables.is_some() {
            return Err("tampering hooks need the uncommitted source tables".into());
        }
        if early.tables.degree() != batch.degree() || early.tables.records() != batch.padded_rows {
            return Err("early s2 commitments were built for another batch shape".into());
        }
        encodability_ns = early.encodability_ns;
        early_commit_ns = early.commit_ns;
        (early.tables, Some(early.groups))
    } else {
        let tables = aerie_falcon_statement::staged!(
            "v1.prove.witness.encodability",
            encodability_ns,
            SourceTables::build(batch)?
        );
        (tables, None)
    };
    if let Some(tamper) = tamper_tables {
        tamper(&mut tables);
    }
    // Compact encodings are prepared after any adversarial mutation and
    // before the PIOP timer so every proof path uses the same validated data.
    let coefficients = PreparedCoefficients::new(&tables)?;
    let logical_cells = &setup.logical_cells;

    let (_, _, selection) = verifier_group_profiles_of(setup.degree, records)?;
    let instance = instance::record(
        &manifest,
        salt_mode,
        public_records,
        [
            akita_types::setup_seed_digest(&setup.prover.expanded.descriptor.setup_seed)?,
            MergedConfig::catalog()?.catalog().catalog_digest(),
            *selection.row_digest.as_bytes(),
        ],
    )?;
    #[cfg(test)]
    let registry_instance = instance;
    #[cfg(test)]
    let instance = {
        let mut instance = instance;
        if let Some(offset) = setup.instance_mutation {
            instance[offset] ^= 1;
        }
        instance
    };
    #[cfg(not(test))]
    let mut bound = ProtocolTranscript::new(&manifest, &instance)?;
    #[cfg(test)]
    let mut bound =
        ProtocolTranscript::new_with_test_record(&manifest, &instance, &registry_instance)?;
    #[cfg(test)]
    if setup.site_mutation {
        bound.override_next_site_for_test(b"aerie/falcon/v1/commit/changed-s1");
    }

    let mut commit_group_ns = [0_u128; GROUP_COUNT];
    // Adversarial group tamper only: each role's sparse deltas per polynomial, so the
    // opening proves the tampered polynomials' true evaluations.
    let mut group_deltas: Vec<(GroupRole, Vec<Vec<(usize, i64)>>)> = Vec::new();

    // Every group not already committed by the early `s2` path, in
    // canonical order.
    let mut group_materialization_ns: u128 = 0;
    let pending =
        aerie_falcon_statement::staged!("v1.prove.witness.groups", group_materialization_ns, {
            let mut groups: Vec<(GroupRole, SignedGroup<'_>)> = manifest
                .groups()
                .iter()
                .map(|descriptor| descriptor.role)
                .filter(|role| early_groups.is_none() || !is_s2_only_role(*role))
                .map(|role| (role, signed_group(role, batch, &tables)))
                .collect();
            if let Some(tamper) = tamper {
                for (role, group) in &mut groups {
                    group_deltas.push((*role, group.tamper(*role, tamper)?));
                }
            }
            groups
        });
    let mut commit_ns: u128 = 0;
    // Commitment is the first prover phase. Keep the overall prover span
    // open through the final PCS opening proof.
    let prove_span = tracing::info_span!("v1.prove.protocol").entered();
    let committed = aerie_falcon_statement::staged!("v1.commit", commit_ns, {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            pending
                .par_iter()
                .map(|(role, cells)| commit_group(setup, *role, cells).map(|group| (*role, group)))
                .collect::<Result<Vec<_>, Error>>()?
        }
        #[cfg(not(feature = "parallel"))]
        {
            pending
                .iter()
                .map(|(role, cells)| commit_group(setup, *role, cells).map(|group| (*role, group)))
                .collect::<Result<Vec<_>, Error>>()?
        }
    });
    drop(pending);
    // Canonical slot order S1, slack, S2 sources, budget: the early groups,
    // when present, are exactly the final two.
    let mut ordered = committed;
    if let Some(early) = early_groups {
        ordered.extend(early);
    }
    if ordered.len() != manifest.group_count() {
        return Err("the prover committed the wrong number of groups".into());
    }
    let mut groups: Vec<CommittedGroup<BaseField>> = Vec::with_capacity(GROUP_COUNT);
    let mut handles = Vec::with_capacity(GROUP_COUNT);
    for (slot, (role, group)) in ordered.into_iter().enumerate() {
        if manifest.groups()[slot].role != role {
            return Err("group commitments are out of canonical order".into());
        }
        bound.absorb_group_commitment(role, &group.payload)?;
        groups.push(group.committed);
        handles.push(group.handle);
        commit_group_ns[slot] = group.group_ns;
    }
    // The early commit ran concurrently with the ingestion; its wall time is
    // added so `commit_ns` still accounts every commit, but the prover total
    // no longer equals the sum of its stage timers.
    commit_ns += early_commit_ns;
    // Sanity: the committed profiles are the catalog profiles the final
    // group was scheduled against, and they resolve to a generated catalog
    // row. The verifier re-derives its own selection from the catalogs.
    let (final_group, precommitted) = groups.split_last().expect("four groups");
    if precommitted.iter().map(|group| *group.profile()).ne(setup
        .precommitted
        .as_slice()
        .iter()
        .copied())
    {
        return Err("a precommitted group left its catalog profile".into());
    }
    MergedConfig::catalog()?.resolve_profiles(&CommittedGroupBatchProfile {
        final_group: *final_group.profile(),
        precommitteds: setup.precommitted.as_slice().to_vec(),
    })?;
    let commitment_bytes = groups
        .iter()
        .map(|group| group.commitment().compressed_size())
        .sum::<usize>();

    let seed = bound.derive_projection_seed()?;
    let mut transcript = bound.finish()?;

    // Forward projection and the public outputs `(y, z)`.
    let mut projection_ns: u128 = 0;
    let projected = aerie_falcon_statement::staged!("v1.projection", projection_ns, {
        // A cheating prover may emit any outputs; the predicates are then
        // the verifier's to check.
        match tamper_outputs {
            Some(tamper) => {
                let mut projected = projection::project_unchecked(batch, &seed)?;
                tamper(&seed, &mut projected.outputs);
                projected
            }
            None => projection::project(batch, &seed)?,
        }
    });

    // The PIOP blocks.
    #[cfg(feature = "parallel")]
    let piop_parent = tracing::Span::current();
    let piop_span = tracing::info_span!("v1.piop.prove");
    let degree = setup.degree;
    let mut piop = || -> Result<_, Error> {
        let _piop_span = piop_span.enter();
        let mut stage1_ns = 0;
        let (stage1_proof, first) = aerie_falcon_statement::staged!(
            "v1.stage1.prove",
            stage1_ns,
            stage1::prove_prepared(batch, &tables, &coefficients, &projected, &mut transcript)?
        );
        let mut stage2_ns = 0;
        let (stage2_result, stage3_prepared) =
            aerie_falcon_statement::staged!("v1.stage2.prove_with_stage3_prepare", stage2_ns, {
                #[cfg(feature = "parallel")]
                let parallel = batch.padded_rows >= STAGE3_PREPARE_OVERLAP_MIN_ROWS
                    && rayon::current_num_threads() > 1;
                #[cfg(not(feature = "parallel"))]
                let parallel = false;
                super::join_if(
                    parallel,
                    || {
                        stage2::prove_prepared(
                            degree,
                            &tables,
                            &coefficients,
                            &first,
                            &mut transcript,
                        )
                    },
                    || stage3::prepare(batch, &tables, &first),
                )
            });
        let (stage2_proof, second) = stage2_result?;
        let stage3_prepared = stage3_prepared?;
        let mut stage3_ns = 0;
        let (stage3_proof, third) = aerie_falcon_statement::staged!(
            "v1.stage3.prove",
            stage3_ns,
            stage3::prove_prepared(
                batch,
                &tables,
                &coefficients,
                &first,
                &second,
                stage3_prepared,
                &mut transcript,
            )?
        );
        Ok(PiopOutput {
            stage1_proof,
            stage1_ns,
            stage2_proof,
            stage2_ns,
            stage3_proof,
            third,
            stage3_ns,
        })
    };
    #[cfg(feature = "parallel")]
    let piop_result = if rayon::current_num_threads() > 1 {
        // Start the PIOP on a pool worker so nested parallel iterators and the
        // stage-two/prepare join can be stolen without cold pool injection.
        // Re-enter the caller span because tracing context is thread-local.
        rayon::scope(|_| piop_parent.in_scope(&mut piop))
    } else {
        piop()
    };
    #[cfg(not(feature = "parallel"))]
    let piop_result = piop();
    let PiopOutput {
        stage1_proof,
        stage1_ns,
        stage2_proof,
        stage2_ns,
        stage3_proof,
        third,
        stage3_ns,
    } = piop_result?;
    #[cfg(test)]
    if setup.session_mutation {
        transcript.alter_export_prefix_for_test();
    }
    let pcs_session = transcript.pcs_session()?;
    transcript.ensure_complete()?;
    let piop_work = transcript.work_bytes().to_vec();
    let projection_outputs = projected.outputs;
    let ledger = opening_ledger(degree, &third);
    drop(tables);
    drop(piop_span);

    let wire_commitments = groups.iter().map(|g| g.commitment().clone()).collect();
    let groups = groups
        .into_iter()
        .zip(handles)
        .zip(ledger.groups.iter().zip(logical_cells))
        .zip(manifest.groups())
        .map(|(((committed, handle), (claim, cells)), descriptor)| {
            let point = physical_point(&claim.point, *cells);
            let values = group_deltas
                .iter()
                .filter(|(role, _)| *role == descriptor.role)
                .fold(claim.values.clone(), |values, (_, deltas)| {
                    values
                        .iter()
                        .zip(deltas)
                        .map(|(&value, polynomial_deltas)| {
                            value + delta_evaluation(polynomial_deltas, &point)
                        })
                        .collect()
                });
            OpeningGroup {
                committed,
                handle,
                point,
                values,
            }
        })
        .collect();
    let mut pcs_prove_ns = 0;
    let pcs_proof = aerie_falcon_statement::staged!(
        "v1.pcs.prove",
        pcs_prove_ns,
        open_groups(groups, setup, &pcs_session)?
    );
    let pcs_proof_bytes = pcs_proof.len();
    drop(prove_span);

    Ok((
        AggregateProof {
            commitments: wire_commitments,
            projection_outputs,
            stage1: stage1_proof,
            stage2: stage2_proof,
            stage3: stage3_proof,
            piop_work,
            pcs: pcs_proof,
        },
        PcsStageMetrics {
            records,
            padded_records: batch.padded_rows,
            encodability_ns,
            group_materialization_ns,
            setup_ns: 0,
            commit_ns,
            projection_ns,
            stage1_ns,
            stage2_ns,
            stage3_ns,
            pcs_prove_ns,
            target_evaluation_ns: 0,
            stage_verify_ns: 0,
            pcs_verify_ns: 0,
            target_evaluation_single_thread_ns: 0,
            stage_verify_single_thread_ns: 0,
            pcs_verify_single_thread_ns: 0,
            commit_group_ns,
            commitment_bytes,
            pcs_proof_bytes,
        },
    ))
}

/// Verify the complete version 1 protocol from public inputs and wire data.
///
/// Every group profile and the schedule selection are re-derived from the
/// statement shape and the generated catalogs; only commitment payloads
/// and proof messages are consumed from the proof.
///
/// The public inputs must be of the setup's parameter set and record count.
pub fn verify<P: AggregationParams, C>(
    public: &PublicInputs<'_, C>,
    verifier_setup: &FalconVerifierSetup<P>,
    proof: &AggregateProof,
) -> Result<VerifiedOpening, Error>
where
    C: Copy + Sync + Into<i64>,
{
    if public.degree()? != P::DEGREE_ID {
        return Err(format!(
            "verifier setup is for Falcon-{} but the public inputs are not",
            P::DEGREE
        )
        .into());
    }
    if public.records != verifier_setup.records {
        return Err(format!(
            "verifier setup has {} records but the public inputs have {}",
            verifier_setup.records, public.records
        )
        .into());
    }
    verify_opening(public, &verifier_setup.akita, proof)
}

#[allow(clippy::too_many_lines)]
#[tracing::instrument(skip_all, name = "v1.verify.protocol")]
fn verify_opening<C>(
    public: &PublicInputs<'_, C>,
    verifier: &AkitaVerifier<MergedConfig>,
    proof: &AggregateProof,
) -> Result<VerifiedOpening, Error>
where
    C: Copy + Sync + Into<i64>,
{
    let reduced = reduce_pcs(public, verifier, proof)?;
    let groups = reduced
        .groups
        .iter()
        .zip(&reduced.claims)
        .map(|(group, (point, values))| {
            PolynomialGroupClaims::new(point.as_slice(), values.clone(), group)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let claims = OpeningClaims::from_groups(groups)?;
    let mut pcs_verify_ns = 0;
    aerie_falcon_statement::staged!(
        "v1.pcs.verify",
        pcs_verify_ns,
        verifier
            .batched_verify(
                &proof.pcs,
                &reduced.session,
                GroupBatchStatement::new(reduced.selection, claims)?,
                BasisMode::Lagrange,
            )
            .map_err(Error::PcsOpening)?
    );
    Ok(VerifiedOpening {
        ledger: reduced.ledger,
        target_evaluation_ns: reduced.target_evaluation_ns,
        stage_verify_ns: reduced.stage_verify_ns,
        pcs_verify_ns,
    })
}

fn verifier_instance<C: Copy + Sync + Into<i64>>(
    public: &PublicInputs<'_, C>,
    manifest: &CommitmentManifest,
    selection: OpeningScheduleSelection,
    verifier: &AkitaVerifier<MergedConfig>,
) -> Result<[u8; instance::INSTANCE_BYTES], Error> {
    instance::record(
        manifest,
        public.salt_mode,
        public.public_records,
        [
            akita_types::setup_seed_digest(&verifier.setup().expanded().descriptor.setup_seed)?,
            MergedConfig::catalog()?.catalog().catalog_digest(),
            *selection.row_digest.as_bytes(),
        ],
    )
}

/// Replay every PIOP block and reconstruct the group claims the batched
/// opening must discharge.
#[allow(clippy::too_many_lines)]
fn reduce_pcs<C>(
    public: &PublicInputs<'_, C>,
    verifier: &AkitaVerifier<MergedConfig>,
    proof: &AggregateProof,
) -> Result<ReducedClaims, Error>
where
    C: Copy + Sync + Into<i64>,
{
    let preprocessing_span = tracing::info_span!("v1.verify.preprocessing").entered();
    let live_rows = public.records;
    let degree = public.degree()?;
    let manifest = CommitmentManifest::for_batch_of(degree, live_rows)?;
    let records = manifest.padded_records();
    public
        .h
        .validate_shape_of(degree.degree(), live_rows, records)?;
    public
        .t
        .validate_shape_of(degree.degree(), live_rows, records)?;
    let group_count = manifest.group_count();
    if proof.commitments.len() != group_count {
        return Err(format!("the proof must carry {group_count} commitments").into());
    }

    let logical_cells = group_cells_of(&manifest);
    let (precommit_profiles, final_profile, opening_selection) =
        verifier_group_profiles_of(degree, live_rows)?;
    let (final_commitment, precommitted) = proof
        .commitments
        .split_last()
        .expect("the group count is positive");
    let groups: Vec<CommittedGroup<BaseField>> = precommit_profiles
        .into_iter()
        .zip(precommitted)
        .map(|(profile, commitment)| CommittedGroup::new(profile, commitment.clone()))
        .collect();
    let final_group = CommittedGroup::new(final_profile, final_commitment.clone());

    let instance = verifier_instance(public, &manifest, opening_selection, verifier)?;
    let mut bound = VerifierProtocolTranscript::new(&manifest, &instance, &proof.piop_work)?;
    for (descriptor, commitment) in manifest.groups().iter().zip(&proof.commitments) {
        bound.absorb_group_commitment(descriptor.role, &serialized_commitment(commitment)?)?;
    }
    let seed = bound.derive_projection_seed()?;
    let mut transcript = bound.finish()?;
    drop(preprocessing_span);

    let mut stage_verify_inclusive_ns: u128 = 0;
    let mut target_evaluation_ns: u128 = 0;
    let ledger = aerie_falcon_statement::staged!("v1.stages.verify", stage_verify_inclusive_ns, {
        projection::check_outputs(degree, &proof.projection_outputs)?;
        let first = stage1::verify(
            degree,
            records,
            live_rows,
            &seed,
            &proof.projection_outputs,
            &proof.stage1,
            &mut transcript,
        )?;
        let second = stage2::verify(degree, records, &first, &proof.stage2, &mut transcript)?;
        let third = stage3::verify(
            degree,
            records,
            public,
            &first,
            &second,
            &proof.stage3,
            &mut transcript,
            &mut target_evaluation_ns,
        )?;
        opening_ledger(degree, &third)
    });

    let mut all_groups = groups;
    all_groups.push(final_group);
    let claims = ledger
        .groups
        .iter()
        .zip(&logical_cells)
        .map(|(opening, cells)| {
            (
                physical_point(&opening.point, *cells),
                opening.values.clone(),
            )
        })
        .collect();
    let stage_verify_ns = stage_verify_inclusive_ns
        .checked_sub(target_evaluation_ns)
        .expect("target evaluation timing is nested inside block verification");
    let session = transcript.pcs_session()?;
    transcript.ensure_complete()?;
    Ok(ReducedClaims {
        session,
        groups: all_groups,
        claims,
        ledger,
        selection: opening_selection,
        target_evaluation_ns,
        stage_verify_ns,
    })
}

/// Prove and verify in one call, merging the stage timings.
#[cfg(test)]
#[tracing::instrument(skip_all, name = "v1.roundtrip")]
pub fn pcs_roundtrip(
    batch: &FalconWitness,
    public_records: &[u8],
) -> Result<PcsStageMetrics, Error> {
    let mut padded_batch = batch.clone();
    super::pad_witness(&mut padded_batch)?;
    let batch = &padded_batch;
    let setup_start = std::time::Instant::now();
    let degree = FalconDegree::from_degree(batch.degree())?;
    let (setup, verifier) = tracing::info_span!("v1.setup").in_scope(|| {
        let setup = ProverCore::new(degree, batch.live_rows)?;
        let verifier = setup.verifier()?;
        Ok::<_, Error>((setup, verifier))
    })?;
    let setup_ns = setup_start.elapsed().as_nanos();
    let (proof, mut metrics) = prove_prepared_impl(
        batch,
        public_records,
        SaltMode::PerRecord,
        &setup,
        None,
        None,
        None,
        None,
    )?;
    metrics.setup_ns = setup_ns;
    let public = PublicInputs {
        records: batch.live_rows,
        salt_mode: crate::protocol::wire::SaltMode::PerRecord,
        public_records,
        h: PublicRingKeys::ExpandedOf(FalconDegree::from_degree(batch.degree())?, &batch.h),
        t: PublicTargets::Expanded(&batch.t),
    };
    let verified = tracing::info_span!("v1.verify").in_scope(|| {
        let verified = tracing::info_span!("v1.verify.configured_pool")
            .in_scope(|| verify_opening(&public, &verifier, &proof))?;
        Ok::<_, Error>(verified)
    })?;
    metrics.target_evaluation_ns = verified.target_evaluation_ns;
    metrics.stage_verify_ns = verified.stage_verify_ns;
    metrics.pcs_verify_ns = verified.pcs_verify_ns;

    #[cfg(feature = "parallel")]
    let single_thread_pool =
        tracing::info_span!("v1.verify.single_thread.setup").in_scope(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .thread_name(|_| "aerie-v1-verifier-single".to_owned())
                .build()
                .map_err(|error| -> Error { error.to_string().into() })
        })?;
    #[cfg(feature = "parallel")]
    let verified_single = tracing::info_span!("v1.verify.single_thread")
        .in_scope(|| {
            single_thread_pool.install(|| {
                verify_opening(&public, &verifier, &proof).map_err(|error| error.to_string())
            })
        })
        .map_err(|error| -> Error { error.into() })?;
    #[cfg(not(feature = "parallel"))]
    let verified_single = tracing::info_span!("v1.verify.single_thread")
        .in_scope(|| verify_opening(&public, &verifier, &proof))?;
    if verified.ledger != verified_single.ledger {
        return Err(
            "configured-pool and single-thread verifiers produced different ledgers".into(),
        );
    }
    metrics.target_evaluation_single_thread_ns = verified_single.target_evaluation_ns;
    metrics.stage_verify_single_thread_ns = verified_single.stage_verify_ns;
    metrics.pcs_verify_single_thread_ns = verified_single.pcs_verify_ns;
    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::kernels::{DigitSource, SignSource};
    use aerie_falcon_native::{Falcon1024, Falcon512};
    use aerie_falcon_statement::DirectRow;

    #[test]
    fn supported_record_counts_follow_the_generated_catalog() {
        let small_counts: Vec<_> = (0..=16)
            .filter(|&records| FalconAggregateSetup::<Falcon512>::supports(records))
            .collect();
        assert_eq!(small_counts, (1..=16).collect::<Vec<_>>());
        for exponent in 3..=16 {
            let records = 1 << exponent;
            assert!(
                FalconAggregateSetup::<Falcon512>::supports(records),
                "{records}"
            );
        }
        for records in [(1 << 16) + 1, usize::MAX] {
            assert!(
                !FalconAggregateSetup::<Falcon512>::supports(records),
                "{records}"
            );
        }
    }

    fn fixture(records: usize) -> FalconWitness {
        let rows = (0..records)
            .map(|seed| {
                DirectRow::<aerie_falcon_native::Falcon512>::synthetic(seed as u64 + 151)
                    .expect("synthetic row")
            })
            .collect::<Vec<_>>();
        FalconWitness::from_rows(&rows)
    }

    fn fixture_1024(records: usize) -> FalconWitness {
        let rows = (0..records)
            .map(|seed| DirectRow::<Falcon1024>::synthetic(seed as u64 + 0x0102_4151).expect("row"))
            .collect::<Vec<_>>();
        FalconWitness::from_rows(&rows)
    }

    fn assert_group_profiles(degree: FalconDegree, records: usize, vars: [usize; 4]) {
        let (precommits, final_group, _) =
            verifier_group_profiles_of(degree, records).expect("cataloged group shapes");
        let expected_precommits = [
            PolynomialGroupLayout::new(vars[0], 1),
            PolynomialGroupLayout::new(vars[1], 1),
            PolynomialGroupLayout::new(vars[2], 4),
        ];
        assert_eq!(precommits.len(), expected_precommits.len());
        for (profile, expected) in precommits.iter().zip(expected_precommits) {
            assert_eq!(profile.group, expected, "precommitted native group shape");
        }
        assert_eq!(
            final_group.group,
            PolynomialGroupLayout::new(vars[3], 3),
            "final native budget group shape"
        );
    }

    /// Native group layouts are resolved at every padded Falcon-1024 batch.
    #[test]
    fn falcon_1024_catalog_covers_the_production_batches() {
        let small_counts: Vec<_> = (0..=16)
            .filter(|&records| FalconAggregateSetup::<Falcon1024>::supports(records))
            .collect();
        assert_eq!(small_counts, (1..=16).collect::<Vec<_>>());
        for records in [32_usize, 64, 128, 256, 512, 1024, 2048] {
            assert!(
                FalconAggregateSetup::<Falcon1024>::supports(records),
                "{records}"
            );
        }
        let cases = [
            (16, [14, 13, 14, 13]),
            (1 << 12, [22, 14, 22, 13]),
            (1 << 13, [23, 15, 23, 13]),
            (1 << 14, [24, 16, 24, 14]),
            (1 << 15, [25, 17, 25, 15]),
            (1 << 16, [26, 18, 26, 16]),
        ];
        for (records, expected_vars) in cases {
            assert_group_profiles(FalconDegree::Falcon1024, records, expected_vars);
        }
        // The Falcon-512 rows do not answer for Falcon-1024 shapes.
        let (precommits_512, _, _) =
            verifier_group_profiles_of(FalconDegree::Falcon512, 1 << 14).expect("Falcon-512 row");
        assert_eq!(precommits_512.len(), 3);
    }

    /// End-to-end Falcon-1024: four real commitments, the blocks and one
    /// multi-group opening against the generated
    /// minimum-batch budget-final row (the Falcon-1024 shapes at
    /// `N_sig* = 4` equal the Falcon-512 shapes at `N_sig* = 8`).
    #[test]
    fn falcon_1024_pcs_roundtrip_at_the_minimum_batch() {
        let metrics = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| {
                let batch = fixture_1024(3);
                pcs_roundtrip(&batch, &[0x10; 32]).map_err(|error| error.to_string())
            })
            .expect("spawn pcs roundtrip")
            .join()
            .expect("join pcs roundtrip")
            .expect("Falcon-1024 PCS roundtrip");
        assert_eq!(metrics.records, 3);
        assert_eq!(metrics.padded_records, 4);
        assert!(
            metrics.commit_group_ns[3] > 0,
            "the final group was committed"
        );
        assert!(metrics.commitment_bytes > 0);
        assert!(metrics.pcs_proof_bytes > 0);
    }

    /// A Falcon-1024 proof must carry four commitments in manifest order: a
    /// swapped or missing final-group commitment rejects, and a Falcon-512
    /// setup refuses a Falcon-1024 witness.
    #[test]
    fn falcon_1024_pcs_verify_rejects_reordered_commitments() {
        std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| {
                let batch = fixture_1024(8);
                let setup = FalconAggregateSetup::<Falcon1024>::new(8).expect("setup");
                assert_eq!(setup.records(), 8);
                let (proof, _) =
                    prove_prepared(&batch, b"falcon-1024 records             ", &setup)
                        .expect("Falcon-1024 prover");
                assert_eq!(proof.commitments.len(), 4);
                let public = PublicInputs {
                    records: 8,
                    salt_mode: crate::protocol::wire::SaltMode::PerRecord,
                    public_records: b"falcon-1024 records             ",
                    h: PublicRingKeys::Expanded(&batch.h),
                    t: PublicTargets::Expanded(&batch.t),
                };
                assert_eq!(public.degree(), Ok(FalconDegree::Falcon1024));
                let verified = verify(&public, setup.verifier(), &proof).expect("verifies");
                assert_eq!(verified.ledger.groups.len(), 4);
                assert_eq!(verified.ledger.groups[3].role, GroupRole::Budget);

                let mut swapped = proof.clone();
                swapped.commitments.swap(2, 3);
                assert!(verify(&public, setup.verifier(), &swapped).is_err());
                let mut truncated = proof.clone();
                truncated.commitments.pop();
                assert!(verify(&public, setup.verifier(), &truncated).is_err());

                let wrong_setup =
                    FalconAggregateSetup::<Falcon512>::new(8).expect("Falcon-512 setup");
                assert!(verify(&public, wrong_setup.verifier(), &proof).is_err());
                assert!(
                    prove_prepared(&batch, b"falcon-1024 records             ", &wrong_setup,)
                        .is_err()
                );
            })
            .expect("spawn")
            .join()
            .expect("join");
    }

    fn public(batch: &FalconWitness, records: usize) -> PublicInputs<'_, i16> {
        PublicInputs {
            records,
            salt_mode: crate::protocol::wire::SaltMode::PerRecord,
            public_records: b"pcs roundtrip records           ",
            h: PublicRingKeys::Expanded(&batch.h),
            t: PublicTargets::Expanded(&batch.t),
        }
    }

    #[test]
    fn compact_group_materialization_matches_the_semantic_i64_table() {
        let compact = [i16::MIN, -8192, -1, 0, 1, 8191, 16_384, i16::MAX];
        let wide = compact.map(i64::from);
        assert!(
            group_polynomial(&compact).expect("compact polynomial")
                == group_polynomial(&wide).expect("wide polynomial")
        );
    }

    #[test]
    fn native_groups_keep_polynomials_independent_and_split_tamper_deltas() {
        let first = [-8_i8, -1, 0, 7];
        let second = [7_i8, 0, -1, -8];
        let third = [-4_i8, 1, 2, 3];
        let fourth = [3_i8, 2, 1, -4];
        let mut group = SignedGroup::S2Sources {
            digits: [&first, &second, &third].map(|cells| DigitSource::Direct(cells)),
            signs: SignSource::Direct(&fourth),
        };
        assert_eq!(group.logical_cells(), first.len());
        assert_eq!(group.polynomial_count(), 4);
        assert_eq!(
            group.polynomials().expect("four source polynomials"),
            vec![
                group_polynomial(&first).expect("a0"),
                group_polynomial(&second).expect("a1"),
                group_polynomial(&third).expect("a2"),
                group_polynomial(&fourth).expect("sigma"),
            ]
        );
        let deltas = group
            .tamper(GroupRole::S2Sources, &|role, cells| {
                assert_eq!(role, GroupRole::S2Sources);
                cells[0] -= 1;
                *cells.last_mut().expect("nonempty group") += 2;
            })
            .expect("fixed-shape adversarial edit");
        assert_eq!(deltas, vec![vec![(0, -1)], vec![], vec![], vec![(3, 2)]]);
        let expected = vec![
            vec![-9, -1, 0, 7],
            vec![7, 0, -1, -8],
            vec![-4, 1, 2, 3],
            vec![3, 2, 1, -2],
        ];
        match &group {
            SignedGroup::OwnedI64(polys) => assert_eq!(polys, &expected),
            _ => panic!("tampering must retain polynomial-major group shape"),
        }

        let budget = SignedGroup::MultiI8Three([&first, &second, &third]);
        assert_eq!(budget.polynomial_count(), 3);
        assert_eq!(budget.logical_cells(), first.len());
        assert_eq!(
            budget
                .polynomials()
                .expect("three native budget polynomials"),
            [&first, &second, &third]
                .into_iter()
                .map(|cells| group_polynomial(cells).expect("semantic narrow polynomial"))
                .collect::<Vec<_>>()
        );
    }

    /// Every cataloged production power-of-two batch has the four native
    /// group layouts in the runtime profile derivation path.
    #[test]
    fn catalog_covers_production_power_of_two_batches() {
        let cases = [
            (1 << 12, [21, 14, 21, 13]),
            (1 << 13, [22, 15, 22, 13]),
            (1 << 14, [23, 16, 23, 14]),
            (1 << 15, [24, 17, 24, 15]),
            (1 << 16, [25, 18, 25, 16]),
        ];
        for (records, expected_vars) in cases {
            let expected_layouts = [
                PolynomialGroupLayout::new(expected_vars[0], 1),
                PolynomialGroupLayout::new(expected_vars[1], 1),
                PolynomialGroupLayout::new(expected_vars[2], 4),
            ];
            let precommits = expected_layouts
                .iter()
                .enumerate()
                .map(|(index, &layout)| match index {
                    0 | 1 => ConservativeMainConfig::commitment_profile(layout),
                    _ => MergedConfig::commitment_profile(layout),
                })
                .collect::<Result<Vec<_>, _>>()
                .expect("cataloged precommit profiles");
            MergedConfig::catalog()
                .expect("trusted test catalog")
                .resolve_key(&ScheduleLookupKey {
                    final_group: PolynomialGroupLayout::new(expected_vars[3], 3),
                    precommitteds: precommits,
                })
                .expect("cataloged requested batch");
            assert_group_profiles(FalconDegree::Falcon512, records, expected_vars);
        }
    }

    /// End-to-end: four real commitments, the blocks, and one multi-group
    /// opening against the generated minimum-batch budget-final row. Requires the
    /// regenerated schedule catalogs; runs on a deep stack like the other
    /// PCS compositions.
    #[test]
    fn pcs_roundtrip_at_the_minimum_batch() {
        let metrics = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| {
                let batch = fixture(8);
                pcs_roundtrip(&batch, b"pcs roundtrip records           ")
                    .map_err(|error| error.to_string())
            })
            .expect("spawn pcs roundtrip")
            .join()
            .expect("join pcs roundtrip")
            .expect("PCS roundtrip");
        assert_eq!(metrics.records, 8);
        assert_eq!(metrics.padded_records, 8);
        assert!(metrics.commitment_bytes > 0);
        assert!(metrics.pcs_proof_bytes > 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn stage3_prepare_overlap_preserves_proof_bytes() {
        std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| {
                // 129 is the smallest live batch whose canonical padded domain
                // reaches the 256-row overlap gate.
                let mut batch = fixture(129);
                crate::protocol::pad_witness(&mut batch).expect("pad overlap fixture");
                assert_eq!(batch.padded_rows, STAGE3_PREPARE_OVERLAP_MIN_ROWS);
                let setup =
                    FalconAggregateSetup::<Falcon512>::new(batch.live_rows).expect("overlap setup");
                let prove_in_pool = |threads| {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .stack_size(512 * 1024 * 1024)
                        .build()
                        .expect("overlap proof pool")
                        .install(|| {
                            prove_prepared(&batch, b"stage3 prepare overlap proof    ", &setup)
                                .expect("overlap proof")
                                .0
                        })
                };
                let single = prove_in_pool(1);
                let parallel = prove_in_pool(4);
                let salts =
                    vec![[0x5a; crate::protocol::wire::TRANSPORT_SALT_BYTES]; batch.live_rows];
                let encode = |proof| {
                    crate::protocol::wire::encode_pcs_proof(
                        FalconDegree::Falcon512,
                        batch.live_rows,
                        &salts,
                        proof,
                    )
                    .expect("encode overlap proof")
                };
                assert_eq!(encode(&single), encode(&parallel));
            })
            .expect("spawn overlap proof")
            .join()
            .expect("join overlap proof");
    }

    #[test]
    fn native_transcript_rejects_proofs_bound_to_another_instance_or_session() {
        std::thread::Builder::new().stack_size(512 * 1024 * 1024).spawn(|| {
            let run = || {
                for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
                    let batch = if degree == FalconDegree::Falcon512 { fixture(8) } else { fixture_1024(4) };
                    let mut setup = ProverCore::new(degree, batch.live_rows).unwrap();
                    let verifier = setup.verifier().unwrap();
                    let public = PublicInputs {
                        records: batch.live_rows,
                        salt_mode: SaltMode::PerRecord,
                        public_records: &[0x91; 32],
                        h: PublicRingKeys::ExpandedOf(degree, &batch.h),
                        t: PublicTargets::Expanded(&batch.t),
                    };
                    for offset in [None, Some(2), Some(27), Some(28), Some(60), Some(302), Some(334), Some(366)] {
                        setup.instance_mutation = offset;
                        let (proof, _) = prove_prepared_impl(&batch, public.public_records, SaltMode::PerRecord, &setup, None, None, None, None).unwrap();
                        let result = verify_opening(&public, &verifier, &proof);
                        assert_eq!(result.is_ok(), offset.is_none(), "instance field {offset:?}");
                    }
                    setup.instance_mutation = None;
                    setup.site_mutation = true;
                    let (proof, _) = prove_prepared_impl(&batch, public.public_records, SaltMode::PerRecord, &setup, None, None, None, None).unwrap();
                    assert!(verify_opening(&public, &verifier, &proof).is_err(), "changed commitment site must reject");
                    setup.site_mutation = false;
                    setup.session_mutation = true;
                    let (proof, _) = prove_prepared_impl(&batch, public.public_records, SaltMode::PerRecord, &setup, None, None, None, None).unwrap();
                    assert!(matches!(verify_opening(&public, &verifier, &proof), Err(Error::PcsOpening(_))), "same PIOP with changed PCS session must reach and fail PCS verification");
                }
            };
            #[cfg(feature = "parallel")]
            {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(1).stack_size(512 * 1024 * 1024).build().unwrap();
                pool.install(run);
            }
            #[cfg(not(feature = "parallel"))]
            run();
        }).unwrap().join().unwrap();
    }

    /// The split halves are wire-style: the verifier reconstructs every
    /// profile from the catalogs and rejects swapped or mismatched
    /// commitment payloads and wrong statements.
    #[test]
    fn split_verify_rejects_tampered_statements_and_payloads() {
        let outcome = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| -> Result<(), String> {
                let batch = fixture(8);
                let (proof, _) = prove(&batch, b"pcs roundtrip records           ")
                    .map_err(|error| error.to_string())?;
                let verifier_setup =
                    FalconVerifierSetup::<Falcon512>::new(8).map_err(|error| error.to_string())?;
                verify(&public(&batch, 8), &verifier_setup, &proof)
                    .map_err(|error| error.to_string())?;

                // Swapped commitment payloads change the bound transcript.
                let mut swapped = proof.clone();
                swapped.commitments.swap(0, 1);
                assert!(verify(&public(&batch, 8), &verifier_setup, &swapped).is_err());

                // A wrong public live count is refused by the eight-record
                // setup, and under a seven-record setup (same padded domain)
                // it rebinds the manifest.
                let mut wrong_live = public(&batch, 8);
                wrong_live.records = 7;
                assert!(verify(&wrong_live, &verifier_setup, &proof).is_err());
                let seven_setup =
                    FalconVerifierSetup::<Falcon512>::new(7).map_err(|error| error.to_string())?;
                assert!(verify(&wrong_live, &seven_setup, &proof).is_err());

                // Tampered public targets break the ring scalar.
                let mut tampered_targets = batch.t.clone();
                tampered_targets[3] += 1;
                let mut wrong_targets = public(&batch, 8);
                wrong_targets.t = PublicTargets::Expanded(&tampered_targets);
                assert!(verify(&wrong_targets, &verifier_setup, &proof).is_err());
                Ok(())
            })
            .expect("spawn split test")
            .join()
            .expect("join split test");
        outcome.expect("split verification checks");
    }

    #[test]
    fn prepared_setup_rejects_a_different_live_record_count() {
        let batch = fixture(5);
        let setup = FalconAggregateSetup::<Falcon512>::new(8).expect("setup for eight records");
        let result = prove_prepared(&batch, b"setup mismatch                  ", &setup);
        assert!(result.is_err());
    }

    /// The production batch against the `falcon-512-batch-16384-budget-final`
    /// catalog row. Ignored by default: minutes of unoptimized serial
    /// sumcheck work and a multi-gigabyte peak. Timings measured in managed
    /// sessions are not comparable to host numbers.
    #[test]
    #[ignore = "target-batch roundtrip; slow and memory heavy"]
    fn pcs_roundtrip_at_the_target_batch() {
        let metrics = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| {
                let batch = fixture(1 << 14);
                pcs_roundtrip(&batch, b"pcs target roundtrip records    ")
                    .map_err(|error| error.to_string())
            })
            .expect("spawn target roundtrip")
            .join()
            .expect("join target roundtrip")
            .expect("PCS roundtrip at the target batch");
        assert_eq!(metrics.records, 1 << 14);
        println!(
            "target batch: commitment bytes {}, pcs proof bytes {}",
            metrics.commitment_bytes, metrics.pcs_proof_bytes
        );
    }
}
