//! Offline regeneration of aerie's fp64 generated schedule catalogs.
//!
//! Upstream akita resolves runtime schedules exclusively from generated
//! catalogs, so aerie's presets ship audited `.aks` artifacts under
//! `crates/aerie-pcs-config/src/schedules/`. This harness drives the same
//! `akita_planner::emit` API the in-tree `gen_schedule_tables` binary uses
//! ("Jolt can invoke the same API"; so can aerie).
//!
//! Regenerate with the catalog feature OFF:
//!
//! ```text
//! cargo test -p aerie-pcs-config --release --no-default-features \
//!     --test schedule_catalog_regen -- --ignored regenerate
//! ```
//!
//! Check drift without rewriting anything, one shard at a time (`0/1` checks
//! every row; `AKITA_SCHEDULE_GEN_JOBS` bounds the planner workers):
//!
//! ```text
//! AERIE_CATALOG_SHARD=0/8 cargo test -p aerie-pcs-config --release --no-default-features \
//!     --test schedule_catalog_regen -- --ignored --exact verify_shard
//! ```
//!
//! After adding or pruning a key-list entry, `update_catalog_rows` rewrites
//! both artifacts to the requested rows, planning only the new ones.

use aerie_pcs_config::configs::{Fp64SignedWidth14, Fp64SignedWidth4};
use akita_config::{policy_of, CommitmentConfig, TrustedScheduleCatalog, ValidatedScheduleCatalog};
use akita_params::{
    CommittedGroupBatchProfile, FoldSchedule, FoldScheduleEstimate, GroupCommitPhaseParams,
    PolynomialGroupLayout, ScheduleLookupKey, ScheduleSisBound,
};
use akita_planner::emit::{
    bounded_parallel_filter_map, offline_planning_worker_count, publish_artifact_outputs,
    render_schedule_artifact_outputs_with_validation, MaterializationDiagnostics,
};
use akita_planner::emit::{GroupedGenerationRequest, PrecommittedProducer};
use akita_planner::{find_schedule, EmitSpec};
use std::any::TypeId;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Complete single-group producer rows. Variable counts include the PCS floor.
const WIDTH14_PRECOMMIT_KEYS: &[(usize, usize)] = &[
    (13, 1),
    (14, 1),
    (15, 1),
    (16, 1),
    (17, 1),
    (18, 1),
    (19, 1),
    (20, 1),
    (21, 1),
    (22, 1),
    (23, 1),
    (24, 1),
    (25, 1),
    (26, 1),
];

const WIDTH4_SCALAR_KEYS: &[(usize, usize)] = &[
    (13, 4),
    (14, 4),
    (15, 4),
    (16, 4),
    (17, 4),
    (18, 4),
    (19, 4),
    (20, 4),
    (21, 4),
    (22, 4),
    (23, 4),
    (24, 4),
    (25, 4),
    (26, 4),
];

use PrecommitKey::{Width14 as W14, Width4 as W4};
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrecommitKey {
    Width14(usize, usize),
    Width4(usize, usize),
}
impl PrecommitKey {
    fn producer(self) -> PrecommittedProducer {
        match self {
            Self::Width14(nv, polys) => {
                precommitted_producer::<Fp64SignedWidth14>(PolynomialGroupLayout::new(nv, polys))
            }
            Self::Width4(nv, polys) => {
                precommitted_producer::<Fp64SignedWidth4>(PolynomialGroupLayout::new(nv, polys))
            }
        }
    }
}

const FALCON_V1_GROUPED_ROWS: &[(&str, (usize, usize), &[PrecommitKey])] = &[
    (
        "falcon-512-batch-8-budget-final",
        (13, 3),
        &[W14(13, 1), W14(13, 1), W4(13, 4)],
    ),
    (
        "falcon-512-batch-32-budget-final",
        (13, 3),
        &[W14(14, 1), W14(13, 1), W4(14, 4)],
    ),
    (
        "falcon-512-batch-64-budget-final",
        (13, 3),
        &[W14(15, 1), W14(13, 1), W4(15, 4)],
    ),
    (
        "falcon-512-batch-128-budget-final",
        (13, 3),
        &[W14(16, 1), W14(13, 1), W4(16, 4)],
    ),
    (
        "falcon-512-batch-256-budget-final",
        (13, 3),
        &[W14(17, 1), W14(13, 1), W4(17, 4)],
    ),
    (
        "falcon-512-batch-512-budget-final",
        (13, 3),
        &[W14(18, 1), W14(13, 1), W4(18, 4)],
    ),
    (
        "falcon-512-batch-1024-budget-final",
        (13, 3),
        &[W14(19, 1), W14(13, 1), W4(19, 4)],
    ),
    (
        "falcon-512-batch-2048-budget-final",
        (13, 3),
        &[W14(20, 1), W14(13, 1), W4(20, 4)],
    ),
    (
        "falcon-512-batch-4096-budget-final",
        (13, 3),
        &[W14(21, 1), W14(14, 1), W4(21, 4)],
    ),
    (
        "falcon-512-batch-8192-budget-final",
        (13, 3),
        &[W14(22, 1), W14(15, 1), W4(22, 4)],
    ),
    (
        "falcon-512-batch-16384-budget-final",
        (14, 3),
        &[W14(23, 1), W14(16, 1), W4(23, 4)],
    ),
    (
        "falcon-512-batch-32768-budget-final",
        (15, 3),
        &[W14(24, 1), W14(17, 1), W4(24, 4)],
    ),
    (
        "falcon-512-batch-65536-budget-final",
        (16, 3),
        &[W14(25, 1), W14(18, 1), W4(25, 4)],
    ),
];
const FALCON_1024_GROUPED_ROWS: &[(&str, (usize, usize), &[PrecommitKey])] = &[
    (
        "falcon-1024-batch-2048-budget-final",
        (13, 3),
        &[W14(21, 1), W14(13, 1), W4(21, 4)],
    ),
    (
        "falcon-1024-batch-4096-budget-final",
        (13, 3),
        &[W14(22, 1), W14(14, 1), W4(22, 4)],
    ),
    (
        "falcon-1024-batch-8192-budget-final",
        (13, 3),
        &[W14(23, 1), W14(15, 1), W4(23, 4)],
    ),
    (
        "falcon-1024-batch-16384-budget-final",
        (14, 3),
        &[W14(24, 1), W14(16, 1), W4(24, 4)],
    ),
    (
        "falcon-1024-batch-32768-budget-final",
        (15, 3),
        &[W14(25, 1), W14(17, 1), W4(25, 4)],
    ),
    (
        "falcon-1024-batch-65536-budget-final",
        (16, 3),
        &[W14(26, 1), W14(18, 1), W4(26, 4)],
    ),
];
fn grouped_request(
    (final_nv, final_polys): (usize, usize),
    precommits: &[PrecommitKey],
) -> GroupedGenerationRequest {
    GroupedGenerationRequest::new(
        PolynomialGroupLayout::new(final_nv, final_polys),
        precommits.iter().map(|&key| key.producer()).collect(),
    )
}

/// Session-scoped cache of exact scalar DP results, mirroring upstream's
/// `GenerationPreplans`: when a complete scalar schedule exists, its root
/// profile and schedule are reused byte-identically.
static SCALAR_PREPLANS: Mutex<Vec<(TypeId, PolynomialGroupLayout, FoldSchedule)>> =
    Mutex::new(Vec::new());

#[derive(Clone)]
struct PlannerObservation {
    family: &'static str,
    key: ScheduleLookupKey,
    estimate: FoldScheduleEstimate,
}

static PLANNER_OBSERVATIONS: Mutex<Vec<PlannerObservation>> = Mutex::new(Vec::new());

thread_local! {
    // The coverage test must fail on missing producers, never launch the DP.
    // Other tests, including explicit offline regeneration, keep their own mode.
    static CATALOG_COVERAGE_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn preplanned_scalar<Cfg: CommitmentConfig + 'static>(
    key: PolynomialGroupLayout,
) -> Result<FoldSchedule, akita_error::AkitaError> {
    let source = TypeId::of::<Cfg>();
    let cached = {
        let cache = SCALAR_PREPLANS
            .lock()
            .expect("scalar preplan cache poisoned");
        cache
            .iter()
            .find(|(cached_source, cached_key, _)| *cached_source == source && *cached_key == key)
            .map(|(_, _, schedule)| schedule.clone())
    };
    if let Some(schedule) = cached {
        return Ok(schedule);
    }
    assert!(
        !CATALOG_COVERAGE_ONLY.get(),
        "missing checked-in producer for {} {key:?}",
        Cfg::schedule_family_name()
    );
    eprintln!("preplanning {} {key:?}", Cfg::schedule_family_name());
    let schedule = regen::<Cfg>(key)?;
    let mut cache = SCALAR_PREPLANS
        .lock()
        .expect("scalar preplan cache poisoned");
    if let Some((_, _, cached)) = cache
        .iter()
        .find(|(cached_source, cached_key, _)| *cached_source == source && *cached_key == key)
    {
        return Ok(cached.clone());
    }
    cache.push((source, key, schedule.clone()));
    Ok(schedule)
}

fn preplanned_scalar_for_keys<Cfg: CommitmentConfig + 'static>(
    keys: &[(usize, usize)],
) -> Vec<(PolynomialGroupLayout, FoldSchedule)> {
    let source = TypeId::of::<Cfg>();
    let cache = SCALAR_PREPLANS
        .lock()
        .expect("scalar preplan cache poisoned");
    keys.iter()
        .filter_map(|&(nv, polys)| {
            let key = PolynomialGroupLayout::new(nv, polys);
            cache
                .iter()
                .find(|(cached_source, cached_key, _)| {
                    *cached_source == source && *cached_key == key
                })
                .map(|(_, _, schedule)| (key, schedule.clone()))
        })
        .collect()
}

fn all_scalar_preplan_keys() -> Vec<PrecommitKey> {
    WIDTH14_PRECOMMIT_KEYS
        .iter()
        .map(|&(nv, polys)| W14(nv, polys))
        .chain(WIDTH4_SCALAR_KEYS.iter().map(|&(nv, polys)| W4(nv, polys)))
        .collect()
}

type GroupedCatalogRow = (&'static str, (usize, usize), &'static [PrecommitKey]);

/// Every width-4 grouped row for the budget-final catalog, in Falcon shape order.
fn grouped_catalog_rows() -> Vec<GroupedCatalogRow> {
    FALCON_V1_GROUPED_ROWS
        .iter()
        .chain(FALCON_1024_GROUPED_ROWS)
        .copied()
        .collect()
}

fn selected_grouped_producer_keys() -> Vec<PrecommitKey> {
    grouped_catalog_rows()
        .into_iter()
        .flat_map(|(_, _, precommits)| precommits.iter().copied())
        .collect()
}

/// Plan every scalar catalog row through Akita's bounded worker pool before
/// grouped requests ask for their exact producer profiles. This preserves the
/// fresh-regeneration contract while avoiding a serial planning phase ahead
/// of the renderer's own parallel grouped-row planning.
fn prewarm_scalar_preplans() {
    let keys = all_scalar_preplan_keys();
    let workers = offline_planning_worker_count(keys.len());
    eprintln!(
        "preplanning {} scalar schedule rows with {workers} workers",
        keys.len()
    );
    bounded_parallel_filter_map(&keys, workers, |key| {
        let result = match *key {
            PrecommitKey::Width14(nv, polys) => {
                preplanned_scalar::<Fp64SignedWidth14>(PolynomialGroupLayout::new(nv, polys))
            }
            PrecommitKey::Width4(nv, polys) => {
                preplanned_scalar::<Fp64SignedWidth4>(PolynomialGroupLayout::new(nv, polys))
            }
        };
        result
            .map(|_| Some(()))
            .map_err(|error| format!("scalar preplanning failed for {key:?}: {error}"))
    })
    .expect("prewarm scalar schedule plans");
}

/// Resolve a catalog authorized precommit profile from its complete scalar row.
fn planned_precommit_profile<Cfg: CommitmentConfig + 'static>(
    layout: PolynomialGroupLayout,
) -> GroupCommitPhaseParams {
    let schedule = preplanned_scalar::<Cfg>(layout)
        .unwrap_or_else(|error| panic!("precommit profile planning failed: {error}"));
    GroupCommitPhaseParams::try_from_params(layout, &schedule.root.params)
        .expect("scalar derived precommit profile")
}

fn precommitted_producer<Cfg: CommitmentConfig + 'static>(
    layout: PolynomialGroupLayout,
) -> PrecommittedProducer {
    PrecommittedProducer::from_config::<Cfg>(planned_precommit_profile::<Cfg>(layout))
        .expect("precommitted producer")
}

/// Pure DP regeneration for `Cfg` — mirrors
/// `akita_planner::generated_families::regen`, which is feature-gated behind
/// `catalog-gen` upstream.
fn regen<Cfg: CommitmentConfig>(
    key: PolynomialGroupLayout,
) -> Result<FoldSchedule, akita_error::AkitaError> {
    let planned = find_schedule(
        &ScheduleLookupKey::single(key),
        Cfg::committed_source_contract()?,
        &[],
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    planned.schedule.validate_structure()?;
    PLANNER_OBSERVATIONS
        .lock()
        .expect("planner observation cache poisoned")
        .push(PlannerObservation {
            family: Cfg::schedule_family_name(),
            key: ScheduleLookupKey::single(key),
            estimate: planned.estimate,
        });
    Ok(planned.schedule)
}

#[allow(clippy::needless_pass_by_value)]
fn regen_group_batch<Cfg: CommitmentConfig>(
    request: GroupedGenerationRequest,
) -> Result<FoldSchedule, akita_error::AkitaError> {
    let key = request.key();
    let planned = find_schedule(
        &key,
        Cfg::committed_source_contract()?,
        &request.source_contracts(),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )?;
    planned.schedule.validate_structure()?;
    PLANNER_OBSERVATIONS
        .lock()
        .expect("planner observation cache poisoned")
        .push(PlannerObservation {
            family: Cfg::schedule_family_name(),
            key: key.clone(),
            estimate: planned.estimate.clone(),
        });
    eprintln!(
        "planned {} grouped final {:?} + {} precommits: estimated proof payload {} B, setup {} field elements",
        Cfg::schedule_family_name(),
        key.final_group,
        key.precommitteds.len(),
        planned.estimate.estimated_proof_payload_bytes()?,
        planned.estimate.estimated_num_setup_field_elements,
    );
    Ok(planned.schedule)
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("schedules")
}

const WIDTH14_FAMILY: &str = "aerie_fp64_signed_width14";
const WIDTH4_FAMILY: &str = "aerie_fp64_signed_width4";
/// Akita revision whose planner emitted the checked-in catalogs and audit.
const AUDIT_AKITA_REVISION: &str = "e023b6a46143f7b23199ccecb2684192baf7b640";

fn width14_spec(output_dir: PathBuf) -> EmitSpec {
    // Reuse complete scalar rows as precommit profiles for every width-14
    // layout in this catalog. Partial-ring layouts, when added, belong only in
    // grouped requests and are planned by `planned_precommit_profile`.
    let mut keys: Vec<PolynomialGroupLayout> = WIDTH14_PRECOMMIT_KEYS
        .iter()
        .map(|&(nv, polys)| PolynomialGroupLayout::new(nv, polys))
        .collect();
    keys.sort_by_key(|layout| (layout.num_vars(), layout.num_polynomials()));
    keys.dedup();
    EmitSpec {
        family_name: WIDTH14_FAMILY,
        policy: policy_of::<Fp64SignedWidth14>(),
        source_contract: Fp64SignedWidth14::committed_source_contract()
            .expect("width-14 source contract"),
        preplanned_scalar: preplanned_scalar_for_keys::<Fp64SignedWidth14>(WIDTH14_PRECOMMIT_KEYS),
        keys,
        grouped_requests: Vec::new(),
        output_dir,
        regen: regen::<Fp64SignedWidth14>,
        regen_group_batch: regen_group_batch::<Fp64SignedWidth14>,
        ring_challenge_config: <Fp64SignedWidth14 as CommitmentConfig>::ring_challenge_config,
    }
}

fn width4_grouped_requests() -> Vec<GroupedGenerationRequest> {
    grouped_catalog_rows()
        .into_iter()
        .map(|(_, final_group, precommits)| grouped_request(final_group, precommits))
        .collect()
}

fn width4_spec(output_dir: PathBuf) -> EmitSpec {
    let grouped_requests = width4_grouped_requests();
    let mut keys: Vec<PolynomialGroupLayout> = WIDTH4_SCALAR_KEYS
        .iter()
        .map(|&(nv, polys)| PolynomialGroupLayout::new(nv, polys))
        .collect();
    keys.sort_by_key(|layout| (layout.num_vars(), layout.num_polynomials()));
    keys.dedup();
    EmitSpec {
        family_name: WIDTH4_FAMILY,
        policy: policy_of::<Fp64SignedWidth4>(),
        source_contract: Fp64SignedWidth4::committed_source_contract()
            .expect("width-4 source contract"),
        preplanned_scalar: preplanned_scalar_for_keys::<Fp64SignedWidth4>(WIDTH4_SCALAR_KEYS),
        keys,
        grouped_requests,
        output_dir,
        regen: regen::<Fp64SignedWidth4>,
        regen_group_batch: regen_group_batch::<Fp64SignedWidth4>,
        ring_challenge_config: <Fp64SignedWidth4 as CommitmentConfig>::ring_challenge_config,
    }
}

fn seed_scalar_preplans<Cfg: CommitmentConfig + 'static>(catalog: &TrustedScheduleCatalog<Cfg>) {
    let source = TypeId::of::<Cfg>();
    let mut cache = SCALAR_PREPLANS
        .lock()
        .expect("scalar preplan cache poisoned");
    for row in catalog.rows() {
        if row.profiles().precommitteds.is_empty() {
            let key = row.profiles().final_group.group;
            if !cache.iter().any(|(cached_source, cached_key, _)| {
                *cached_source == source && *cached_key == key
            }) {
                cache.push((source, key, row.schedule().clone()));
            }
        }
    }
}

fn rows_of<Cfg: CommitmentConfig>(
    catalog: &TrustedScheduleCatalog<Cfg>,
) -> Vec<(CommittedGroupBatchProfile, FoldSchedule)> {
    catalog
        .rows()
        .map(|row| (row.profiles().clone(), row.schedule().clone()))
        .collect()
}

fn width14_artifact() -> Vec<u8> {
    std::fs::read(output_dir().join("aerie_fp64_signed_width14.aks")).expect("width14 artifact")
}
fn width4_artifact() -> Vec<u8> {
    std::fs::read(output_dir().join("aerie_fp64_signed_width4.aks")).expect("width4 artifact")
}

fn checked_in_catalogs() -> (
    TrustedScheduleCatalog<Fp64SignedWidth14>,
    TrustedScheduleCatalog<Fp64SignedWidth4>,
) {
    (
        TrustedScheduleCatalog::from_artifact_bytes(&width14_artifact())
            .expect("load checked-in width-14 catalog"),
        TrustedScheduleCatalog::from_artifact_bytes(&width4_artifact())
            .expect("load checked-in width-4 catalog"),
    )
}

/// Assert that `catalog` holds one row per requested key and nothing else.
/// Full regeneration rejects duplicate grouped keys when it builds the
/// catalog, so duplicates are drift here too.
fn assert_catalog_holds_exactly(catalog: &ValidatedScheduleCatalog, keys: &[ScheduleLookupKey]) {
    let family = catalog.family_name();
    for (index, key) in keys.iter().enumerate() {
        assert!(
            !keys[..index].contains(key),
            "{family}: duplicate requested key {key:?}"
        );
        catalog.resolve_key(key).unwrap_or_else(|error| {
            panic!("{family}: requested key {key:?} has no checked-in row: {error}")
        });
    }
    assert_eq!(
        keys.len(),
        catalog.len(),
        "{family}: catalog contains unrequested rows"
    );
}

/// Re-audit the decoded rows and require the canonical encoding to reproduce
/// the checked-in bytes exactly.
fn assert_rows_reencode<Cfg: CommitmentConfig>(
    catalog: &TrustedScheduleCatalog<Cfg>,
    artifact: &[u8],
) {
    let audited = ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        rows_of(catalog),
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )
    .expect("audit every retained schedule");
    assert_eq!(
        audited.to_artifact_bytes().expect("encode audited rows"),
        artifact,
        "{}: re-encoded rows differ from the checked-in artifact",
        Cfg::schedule_family_name()
    );
}

fn assert_width4_inner_decomposition(profile: &GroupCommitPhaseParams, label: &str) {
    assert_eq!(
        profile.inner.digits.log_basis, 4,
        "{label}: width-4 group must use inner basis 2^4"
    );
    assert_eq!(
        profile.inner.digits.num_digits, 1,
        "{label}: width-4 group must use one inner digit"
    );
}

/// Check pruning and emitted-key coverage without rerunning the offline DP.
///
/// This is the cheap half of the sharded drift check (see [`verify_shard`]):
/// each family's artifact holds exactly the requested lookup keys, once
/// each, and re-auditing and re-encoding its rows reproduces the checked-in
/// bytes. No drift can therefore hide in row selection, ordering, or
/// encoding, and the shards together cover the whole artifact.
#[test]
fn retained_catalog_matches_emitted_shapes() {
    CATALOG_COVERAGE_ONLY.set(true);
    let (wide_catalog, narrow_catalog) = checked_in_catalogs();
    // `regenerate` names its families by these literals.
    assert_eq!(Fp64SignedWidth14::schedule_family_name(), WIDTH14_FAMILY);
    assert_eq!(Fp64SignedWidth4::schedule_family_name(), WIDTH4_FAMILY);
    seed_scalar_preplans(&wide_catalog);
    seed_scalar_preplans(&narrow_catalog);

    let mut wide_keys = Vec::new();
    let mut narrow_keys = Vec::new();
    for key in all_scalar_preplan_keys() {
        match key {
            W14(nv, polys) => wide_keys.push(ScheduleLookupKey::single(
                PolynomialGroupLayout::new(nv, polys),
            )),
            W4(nv, polys) => narrow_keys.push(ScheduleLookupKey::single(
                PolynomialGroupLayout::new(nv, polys),
            )),
        }
    }
    narrow_keys.extend(
        width4_grouped_requests()
            .iter()
            .map(GroupedGenerationRequest::key),
    );
    assert_catalog_holds_exactly(&wide_catalog, &wide_keys);
    assert_catalog_holds_exactly(&narrow_catalog, &narrow_keys);
    for (row_index, row) in narrow_catalog.rows().enumerate() {
        let profiles = row.profiles();
        assert_width4_inner_decomposition(
            &profiles.final_group,
            &format!("width-4 row {row_index} final group"),
        );
        for (group_index, profile) in profiles.precommitteds.iter().enumerate() {
            if profile.group.num_polynomials() >= 3 {
                assert_width4_inner_decomposition(
                    profile,
                    &format!("width-4 row {row_index} precommit group {group_index}"),
                );
            }
        }
    }
    assert_rows_reencode(&wide_catalog, &width14_artifact());
    assert_rows_reencode(&narrow_catalog, &width4_artifact());
    CATALOG_COVERAGE_ONLY.set(false);
}

/// One requested catalog row. `regenerate`, the coverage check, and the
/// drift shards all derive their rows from the key lists above, so pruning a
/// list entry prunes it everywhere.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CatalogRow {
    /// A complete scalar row in its own family.
    Scalar(PrecommitKey),
    /// A budget-final width-4 grouped row.
    Grouped(GroupedCatalogRow),
}

impl CatalogRow {
    fn label(self) -> String {
        match self {
            Self::Scalar(W14(nv, polys)) => format!("{WIDTH14_FAMILY} scalar ({nv}, {polys})"),
            Self::Scalar(W4(nv, polys)) => format!("{WIDTH4_FAMILY} scalar ({nv}, {polys})"),
            Self::Grouped((name, _, _)) => format!("{WIDTH4_FAMILY} {name}"),
        }
    }

    /// Rough planner seconds per row on a two-vCPU runner with two workers,
    /// fitted to the 2026-09-22 main regeneration log. Only shard balance
    /// depends on this estimate; coverage never does.
    fn estimated_planner_seconds(self) -> usize {
        match self {
            Self::Scalar(W14(nv, _) | W4(nv, _)) if nv < 16 => (50 * nv).saturating_sub(620),
            Self::Scalar(W14(nv, _) | W4(nv, _)) => 10 * nv - 20,
            Self::Grouped((_, (final_nv, _), _)) => 100 + 10 * final_nv.saturating_sub(13),
        }
    }
}

/// Every row `regenerate` requests, in its request order.
fn requested_catalog_rows() -> Vec<CatalogRow> {
    all_scalar_preplan_keys()
        .into_iter()
        .map(CatalogRow::Scalar)
        .chain(grouped_catalog_rows().into_iter().map(CatalogRow::Grouped))
        .collect()
}

/// Shard `index` of a deterministic longest-processing-time partition of
/// `rows` into `count` shards by estimated planner cost. The shard is
/// returned in descending cost so the bounded worker pool starts its slowest
/// rows first.
fn catalog_shard(rows: &[CatalogRow], index: usize, count: usize) -> Vec<CatalogRow> {
    assert!(index < count, "shard index {index} is outside 0..{count}");
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&row| {
        (
            std::cmp::Reverse(rows[row].estimated_planner_seconds()),
            row,
        )
    });
    let mut loads = vec![0; count];
    let mut shard = Vec::new();
    for row in order {
        // `min_by_key` keeps the first minimum, so ties go to the lowest shard.
        let lightest = (0..count)
            .min_by_key(|&candidate| loads[candidate])
            .expect("at least one shard");
        loads[lightest] += rows[row].estimated_planner_seconds();
        if lightest == index {
            shard.push(rows[row]);
        }
    }
    shard
}

fn catalog_shard_from_env() -> (usize, usize) {
    let value = std::env::var("AERIE_CATALOG_SHARD")
        .expect("set AERIE_CATALOG_SHARD=<index>/<count>, zero-based; 0/1 checks every row");
    let parsed = value.split_once('/').and_then(|(index, count)| {
        Some((
            index.trim().parse::<usize>().ok()?,
            count.trim().parse::<usize>().ok()?,
        ))
    });
    match parsed {
        Some((index, count)) if index < count => (index, count),
        _ => panic!("AERIE_CATALOG_SHARD={value:?} is not <index>/<count> with index < count"),
    }
}

/// Describe the first byte difference between two serializations.
fn serialized_drift(label: &str, part: &str, checked_in: &[u8], fresh: &[u8]) -> Option<String> {
    if checked_in == fresh {
        return None;
    }
    let offset = checked_in
        .iter()
        .zip(fresh)
        .position(|(left, right)| left != right)
        .unwrap_or(checked_in.len().min(fresh.len()));
    let window = |bytes: &[u8]| {
        let start = offset.saturating_sub(80);
        let end = (offset + 80).min(bytes.len());
        String::from_utf8_lossy(&bytes[start..end]).into_owned()
    };
    Some(format!(
        "{label}: {part} drifted at byte {offset} (checked-in {} bytes, fresh {} bytes)\n  \
         checked-in: ...{}...\n  fresh:      ...{}...",
        checked_in.len(),
        fresh.len(),
        window(checked_in),
        window(fresh),
    ))
}

/// The profiles `render_family_artifact` stores beside a planned schedule:
/// the final profile comes from the root parameters, the precommitted
/// profiles from the request key.
fn rendered_profiles(
    key: &ScheduleLookupKey,
    schedule: &FoldSchedule,
) -> Result<CommittedGroupBatchProfile, akita_error::AkitaError> {
    Ok(CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(
            key.final_group,
            &schedule.root.params,
        )?,
        precommitteds: key.precommitteds.clone(),
    })
}

/// Compare one freshly planned row with the checked-in row for its key,
/// byte-exactly on the serialized profile and schedule. Returns a diagnostic
/// on drift.
fn catalog_row_drift(
    label: &str,
    key: &ScheduleLookupKey,
    fresh: Result<FoldSchedule, akita_error::AkitaError>,
    checked_in: &ValidatedScheduleCatalog,
) -> Option<String> {
    let schedule = match fresh {
        Ok(schedule) => schedule,
        // Full regeneration drops an unsupported row and fails on any other
        // planner error; either way the checked-in artifact would change.
        Err(error) => return Some(format!("{label}: fresh planning failed: {error}")),
    };
    let profiles = match rendered_profiles(key, &schedule) {
        Ok(profiles) => profiles,
        Err(error) => return Some(format!("{label}: derive final profile: {error}")),
    };
    let row = match checked_in.resolve_key(key) {
        Ok(row) => row,
        Err(error) => return Some(format!("{label}: no checked-in row: {error}")),
    };
    let encode = |value: Result<Vec<u8>, serde_json::Error>| {
        value.unwrap_or_else(|error| format!("<encoding failed: {error}>").into_bytes())
    };
    serialized_drift(
        label,
        "profile",
        &encode(serde_json::to_vec(row.profiles())),
        &encode(serde_json::to_vec(&profiles)),
    )
    .or_else(|| {
        serialized_drift(
            label,
            "schedule",
            &encode(serde_json::to_vec(row.schedule())),
            &encode(serde_json::to_vec(&schedule)),
        )
    })
}

/// Sharded drift check: `AERIE_CATALOG_SHARD=<index>/<count>` freshly plans
/// one cost-balanced shard of the requested rows and compares each with the
/// checked-in row for its key. `0/1` checks every row.
///
/// Every shard passing, together with [`retained_catalog_matches_emitted_shapes`],
/// is equivalent to `regenerate` reproducing the checked-in bytes:
///
/// - `regenerate` renders each family as the canonical encoding of the
///   `ValidatedScheduleCatalog` built from its materialized
///   `(profile, schedule)` rows. The catalog orders rows by row digest, so
///   the bytes depend only on the row set. Each row is planned alone: a
///   scalar row by `regen(layout)` and a grouped row by
///   `regen_group_batch(request)`, a fresh `find_schedule` search with its
///   own memo. No row reads another row's result.
/// - A grouped request depends on the scalar rows only through its producer
///   descriptors, and `planned_precommit_profile` derives each descriptor
///   from the scalar schedule's `root.params` alone. This test seeds those
///   producers from the checked-in scalar rows. Each producer is itself a
///   requested scalar row (`every_grouped_producer_has_a_fresh_scalar_preplan`)
///   that some shard plans fresh and compares byte-exactly. When every shard
///   passes, the seeded producers equal the fresh ones full regeneration
///   derives, so each grouped row here equals the row `regenerate` plans.
/// - With each fresh row equal to the checked-in row for its key, and the
///   artifact holding exactly the requested keys, the regenerated row set
///   equals the checked-in row set, and so do the canonical bytes. If a
///   scalar row drifts, its own shard fails even when a grouped row seeded
///   from its stale checked-in profile still matches.
#[test]
#[ignore = "offline DP drift check; set AERIE_CATALOG_SHARD=<index>/<count>"]
fn verify_shard() {
    let (index, count) = catalog_shard_from_env();
    let (wide_catalog, narrow_catalog) = checked_in_catalogs();
    seed_scalar_preplans(&wide_catalog);
    seed_scalar_preplans(&narrow_catalog);
    let rows = catalog_shard(&requested_catalog_rows(), index, count);
    let estimate: usize = rows.iter().map(|row| row.estimated_planner_seconds()).sum();
    eprintln!(
        "catalog shard {index}/{count}: {} rows, about {estimate} s of two-vCPU planner time",
        rows.len()
    );

    // Build grouped requests here, on the thread whose coverage mode forbids
    // planning: a producer must come from a checked-in scalar row.
    CATALOG_COVERAGE_ONLY.set(true);
    let jobs: Vec<(CatalogRow, Option<GroupedGenerationRequest>)> = rows
        .into_iter()
        .map(|row| match row {
            CatalogRow::Scalar(_) => (row, None),
            CatalogRow::Grouped((_, final_group, precommits)) => {
                (row, Some(grouped_request(final_group, precommits)))
            }
        })
        .collect();
    CATALOG_COVERAGE_ONLY.set(false);

    let workers = offline_planning_worker_count(jobs.len());
    eprintln!("verifying {} rows with {workers} workers", jobs.len());
    let started = std::time::Instant::now();
    let drift = bounded_parallel_filter_map(&jobs, workers, |(row, request)| {
        let label = row.label();
        let row_started = std::time::Instant::now();
        let drift = match (*row, request) {
            (CatalogRow::Scalar(W14(nv, polys)), None) => {
                let layout = PolynomialGroupLayout::new(nv, polys);
                catalog_row_drift(
                    &label,
                    &ScheduleLookupKey::single(layout),
                    regen::<Fp64SignedWidth14>(layout),
                    &wide_catalog,
                )
            }
            (CatalogRow::Scalar(W4(nv, polys)), None) => {
                let layout = PolynomialGroupLayout::new(nv, polys);
                catalog_row_drift(
                    &label,
                    &ScheduleLookupKey::single(layout),
                    regen::<Fp64SignedWidth4>(layout),
                    &narrow_catalog,
                )
            }
            (CatalogRow::Grouped(_), Some(request)) => catalog_row_drift(
                &label,
                &request.key(),
                regen_group_batch::<Fp64SignedWidth4>(request.clone()),
                &narrow_catalog,
            ),
            _ => Some(format!("{label}: internal shard job mismatch")),
        };
        let verdict = if drift.is_some() { "DRIFT" } else { "ok" };
        eprintln!("{verdict} {label} in {:.2?}", row_started.elapsed());
        Ok(drift)
    })
    .expect("catalog shard workers");
    eprintln!(
        "catalog shard {index}/{count} finished in {:.2?}",
        started.elapsed()
    );
    assert!(
        drift.is_empty(),
        "{} of {} rows in catalog shard {index}/{count} drifted from the checked-in artifact:\n{}",
        drift.len(),
        jobs.len(),
        drift.join("\n")
    );
}

#[test]
fn catalog_shards_partition_requested_rows() {
    for (name, (final_nv, final_polys), _) in grouped_catalog_rows() {
        assert!(name.ends_with("-budget-final"));
        assert_eq!(final_polys, 3, "{name} must commit the budget group last");
        assert!(final_nv >= 13, "{name} has a supported logical shape");
    }
    let rows = requested_catalog_rows();
    for (index, row) in rows.iter().enumerate() {
        assert!(!rows[..index].contains(row), "duplicate row {row:?}");
    }
    for count in 1..=16 {
        let mut covered = Vec::new();
        for index in 0..count {
            let shard = catalog_shard(&rows, index, count);
            assert!(!shard.is_empty(), "shard {index}/{count} is empty");
            covered.extend(shard);
        }
        assert_eq!(covered.len(), rows.len(), "{count} shards");
        for row in &rows {
            assert!(covered.contains(row), "{row:?} is in no shard of {count}");
        }
    }
}

fn write_catalog<Cfg: CommitmentConfig>(
    rows: Vec<(CommittedGroupBatchProfile, FoldSchedule)>,
    destination: &Path,
) {
    let catalog = ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        rows,
        &policy_of::<Cfg>(),
        Cfg::ring_challenge_config,
    )
    .expect("validate updated catalog");
    std::fs::write(
        destination,
        catalog.to_artifact_bytes().expect("encode updated catalog"),
    )
    .expect("write updated catalog");
    println!("wrote {} ({} rows)", destination.display(), catalog.len());
}

fn estimate_json(estimate: &FoldScheduleEstimate) -> serde_json::Value {
    serde_json::json!({
        "estimated_proof_payload_bytes": estimate.estimated_proof_payload_bytes().expect("proof byte estimate"),
        "estimated_num_setup_field_elements": estimate.estimated_num_setup_field_elements,
    })
}

macro_rules! json_value_with_exact_fallback {
    ($value:expr) => {{
        let value = $value;
        match serde_json::to_value(value) {
            Ok(value) => value,
            Err(error) => serde_json::json!({
                "serialization_error": error.to_string(),
                "exact_json": serde_json::to_string(value).expect("serialize exact JSON"),
            }),
        }
    }};
}

fn u128_json_value(value: u128) -> serde_json::Value {
    match u64::try_from(value) {
        Ok(value) => serde_json::Value::from(value),
        Err(_) => serde_json::Value::String(value.to_string()),
    }
}

fn decomposition_audit(log_basis: u32, num_digits: usize) -> serde_json::Value {
    serde_json::json!({
        "log_basis": log_basis,
        "num_digits": num_digits,
    })
}

fn group_role_label(total_groups: usize, index: usize, num_polynomials: usize) -> &'static str {
    if total_groups == 4 {
        match index {
            0 => "S1",
            1 => "e",
            _ if num_polynomials == 4 => "a0,a1,a2,sigma",
            _ if num_polynomials == 3 => "u0,u1,u2",
            _ => "unknown grouped role",
        }
    } else {
        match num_polynomials {
            1 => "width14 single-polynomial producer",
            3 => "width4 budget producer",
            4 => "width4 source producer",
            _ => "unknown standalone producer",
        }
    }
}

fn group_profile_audit(profile: &GroupCommitPhaseParams, role: &'static str) -> serde_json::Value {
    serde_json::json!({
        "role": role,
        "profile": json_value_with_exact_fallback!(profile),
        "inner_decomposition": decomposition_audit(
            profile.inner.digits.log_basis,
            profile.inner.digits.num_digits,
        ),
        "outer_decomposition": decomposition_audit(
            profile.outer.digits.log_basis,
            profile.outer.digits.num_digits,
        ),
    })
}

fn group_profiles_audit(profiles: &CommittedGroupBatchProfile) -> Vec<serde_json::Value> {
    let total_groups = profiles.precommitteds.len() + 1;
    let mut groups: Vec<_> = profiles
        .precommitteds
        .iter()
        .enumerate()
        .map(|(index, profile)| {
            group_profile_audit(
                profile,
                group_role_label(total_groups, index, profile.group.num_polynomials()),
            )
        })
        .collect();
    let final_index = profiles.precommitteds.len();
    let final_profile = &profiles.final_group;
    groups.push(group_profile_audit(
        final_profile,
        group_role_label(
            total_groups,
            final_index,
            final_profile.group.num_polynomials(),
        ),
    ));
    groups
}

fn audit_occurrences(schedule: &FoldSchedule) -> Vec<serde_json::Value> {
    schedule
        .sis_occurrences()
        .expect("audited canonical SIS occurrences")
        .into_iter()
        .map(|occurrence| {
            let bound = match occurrence.bound {
                ScheduleSisBound::Linf(value) => serde_json::json!({
                    "kind": "linf",
                    "value": u128_json_value(value),
                }),
                ScheduleSisBound::L2Squared(value) => serde_json::json!({
                    "kind": "l2_squared",
                    "value": u128_json_value(value),
                }),
            };
            serde_json::json!({
                "location": occurrence.location,
                "role": format!("{:?}", occurrence.role),
                "output_rank": occurrence.output_rank,
                "input_width": occurrence.input_width,
                "ring_dimension": occurrence.ring_dimension,
                "modulus_profile": format!("{:?}", occurrence.modulus_profile),
                "bound": bound,
            })
        })
        .collect()
}

fn audited_family<Cfg: CommitmentConfig>(
    family: &'static str,
    catalog: &TrustedScheduleCatalog<Cfg>,
    observations: &[PlannerObservation],
    recorded_estimates: &BTreeMap<(String, String), serde_json::Value>,
) -> serde_json::Value {
    let rows: Vec<_> = catalog
        .rows()
        .map(|row| {
            let profiles = row.profiles();
            let key = ScheduleLookupKey {
                final_group: profiles.final_group.group,
                precommitteds: profiles.precommitteds.clone(),
            };
            let observation = observations
                .iter()
                .rev()
                .find(|observation| observation.family == family && observation.key == key)
                .map(|observation| estimate_json(&observation.estimate));
            let key_value = serde_json::to_value(&key).expect("serialize audit key");
            let key_json = serde_json::to_string(&key_value).expect("encode audit key");
            let estimate_key = (family.to_owned(), key_json);
            let planner_estimate = observation
                .or_else(|| recorded_estimates.get(&estimate_key).cloned())
                .unwrap_or_else(|| panic!("no recorded planner estimate for {family} {key:?}"));
            let schedule = row.schedule();
            serde_json::json!({
                "key": key,
                "profiles": json_value_with_exact_fallback!(profiles),
                "group_profiles": group_profiles_audit(profiles),
                "schedule_json": serde_json::to_string(schedule).expect("serialize full schedule"),
                "fold_count": schedule.recursive_folds.len() + 2,
                "planner_estimate": planner_estimate,
                "sis_occurrences": audit_occurrences(schedule),
            })
        })
        .collect();
    serde_json::json!({"name": family, "rows": rows})
}

/// The `.aks` artifact does not store these two planner metrics. For a cheap
/// audit rewrite, retain their already-recorded values by exact family/key;
/// regeneration instead supplies fresh metrics through `observations`.
fn recorded_planner_estimates(
    report: &serde_json::Value,
) -> BTreeMap<(String, String), serde_json::Value> {
    let mut estimates = BTreeMap::new();
    for family in report["families"].as_array().into_iter().flatten() {
        let Some(family_name) = family["name"].as_str() else {
            continue;
        };
        for row in family["rows"].as_array().into_iter().flatten() {
            if let (Ok(key), Some(estimate)) = (
                serde_json::to_string(&row["key"]),
                row.get("planner_estimate"),
            ) {
                estimates.insert(
                    (family_name.to_owned(), key),
                    serde_json::json!({
                        "estimated_proof_payload_bytes": estimate["estimated_proof_payload_bytes"],
                        "estimated_num_setup_field_elements": estimate["estimated_num_setup_field_elements"],
                    }),
                );
            }
        }
    }
    estimates
}

fn catalog_audit_report(
    observations: &[PlannerObservation],
    recorded_estimates: &BTreeMap<(String, String), serde_json::Value>,
) -> serde_json::Value {
    let wide_catalog =
        TrustedScheduleCatalog::<Fp64SignedWidth14>::from_artifact_bytes(&width14_artifact())
            .expect("width14 audit catalog");
    let narrow_catalog =
        TrustedScheduleCatalog::<Fp64SignedWidth4>::from_artifact_bytes(&width4_artifact())
            .expect("width4 audit catalog");
    let families = [
        audited_family(
            WIDTH14_FAMILY,
            &wide_catalog,
            observations,
            recorded_estimates,
        ),
        audited_family(
            WIDTH4_FAMILY,
            &narrow_catalog,
            observations,
            recorded_estimates,
        ),
    ];
    serde_json::json!({
        "schema": "aerie-pcs-catalog-audit-v1",
        "akita_revision": AUDIT_AKITA_REVISION,
        "families": families,
    })
}

fn write_catalog_audit(path: &Path) {
    let observations = PLANNER_OBSERVATIONS
        .lock()
        .expect("planner observation cache poisoned")
        .clone();
    let recorded_estimates = std::fs::read(path)
        .ok()
        .map(|bytes| {
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap_or_else(|error| {
                panic!("parse prior catalog audit {}: {error}", path.display())
            })
        })
        .map_or_else(BTreeMap::new, |report| recorded_planner_estimates(&report));
    let report = catalog_audit_report(&observations, &recorded_estimates);
    std::fs::write(
        path,
        serde_json::to_vec(&report).expect("serialize compact catalog audit"),
    )
    .unwrap_or_else(|error| panic!("write catalog audit {}: {error}", path.display()));
    println!("wrote catalog audit {}", path.display());
}

/// The checked-in row for `key`, or a freshly planned one when the artifact
/// lacks it.
fn retained_or_planned_row(
    label: &str,
    catalog: &ValidatedScheduleCatalog,
    key: &ScheduleLookupKey,
    plan: impl FnOnce() -> Result<FoldSchedule, akita_error::AkitaError>,
) -> (CommittedGroupBatchProfile, FoldSchedule) {
    if let Ok(row) = catalog.resolve_key(key) {
        return (row.profiles().clone(), row.schedule().clone());
    }
    let started = std::time::Instant::now();
    let schedule = plan().unwrap_or_else(|error| panic!("plan {label}: {error}"));
    let profiles = rendered_profiles(key, &schedule)
        .unwrap_or_else(|error| panic!("derive {label} profile: {error}"));
    println!("planned {label} in {:.2?}", started.elapsed());
    (profiles, schedule)
}

/// Print every checked-in row that no requested key selects.
fn report_pruned_rows(catalog: &ValidatedScheduleCatalog, requested: &[ScheduleLookupKey]) {
    for row in catalog.rows() {
        let key = ScheduleLookupKey {
            final_group: row.profiles().final_group.group,
            precommitteds: row.profiles().precommitteds.clone(),
        };
        if !requested.contains(&key) {
            println!(
                "pruned {} row: final {:?}, precommits {:?}",
                catalog.family_name(),
                key.final_group,
                key.precommitteds
                    .iter()
                    .map(|profile| profile.group)
                    .collect::<Vec<_>>()
            );
        }
    }
}

/// Rewrite both artifacts to hold exactly the requested rows without
/// re-planning the retained ones: a requested row already checked in is
/// kept byte-for-byte, an unrequested row is dropped, and only a missing
/// requested row runs the offline DP (grouped producers come from the
/// checked-in or newly planned scalar rows). Rows are planned independently
/// and the artifact bytes depend only on the row set (see [`verify_shard`]),
/// so the result is what `regenerate` writes whenever every retained row
/// passes `verify_shard`.
#[test]
#[ignore = "offline incremental catalog update; rewrites src/schedules/"]
fn update_catalog_rows() {
    let (wide_catalog, narrow_catalog) = checked_in_catalogs();
    seed_scalar_preplans(&wide_catalog);
    seed_scalar_preplans(&narrow_catalog);
    let mut wide_rows = Vec::new();
    let mut narrow_rows = Vec::new();
    let mut wide_keys = Vec::new();
    let mut narrow_keys = Vec::new();
    // Scalar rows come first, so a new grouped row's producers are cached.
    for row in requested_catalog_rows() {
        let label = row.label();
        match row {
            CatalogRow::Scalar(W14(nv, polys)) => {
                let layout = PolynomialGroupLayout::new(nv, polys);
                let key = ScheduleLookupKey::single(layout);
                wide_rows.push(retained_or_planned_row(&label, &wide_catalog, &key, || {
                    preplanned_scalar::<Fp64SignedWidth14>(layout)
                }));
                wide_keys.push(key);
            }
            CatalogRow::Scalar(W4(nv, polys)) => {
                let layout = PolynomialGroupLayout::new(nv, polys);
                let key = ScheduleLookupKey::single(layout);
                narrow_rows.push(retained_or_planned_row(
                    &label,
                    &narrow_catalog,
                    &key,
                    || preplanned_scalar::<Fp64SignedWidth4>(layout),
                ));
                narrow_keys.push(key);
            }
            CatalogRow::Grouped((_, final_group, precommits)) => {
                let request = grouped_request(final_group, precommits);
                let key = request.key();
                narrow_rows.push(retained_or_planned_row(
                    &label,
                    &narrow_catalog,
                    &key,
                    || regen_group_batch::<Fp64SignedWidth4>(request),
                ));
                narrow_keys.push(key);
            }
        }
    }
    report_pruned_rows(&wide_catalog, &wide_keys);
    report_pruned_rows(&narrow_catalog, &narrow_keys);

    let output_dir = output_dir();
    write_catalog::<Fp64SignedWidth14>(
        wide_rows,
        &output_dir.join(format!("{WIDTH14_FAMILY}.aks")),
    );
    write_catalog::<Fp64SignedWidth4>(
        narrow_rows,
        &output_dir.join(format!("{WIDTH4_FAMILY}.aks")),
    );
}

#[test]
#[ignore = "offline DP regeneration; rewrites src/schedules/"]
fn regenerate() {
    let output_dir = output_dir();
    PLANNER_OBSERVATIONS
        .lock()
        .expect("planner observation cache poisoned")
        .clear();
    prewarm_scalar_preplans();
    // Construct width-4 first so every grouped request resolves its exact
    // producer profile from the fresh scalar schedules warmed above. Both
    // specs then snapshot those same rows into `preplanned_scalar`.
    let narrow_spec = width4_spec(output_dir.clone());
    let wide_spec = width14_spec(output_dir.clone());
    let specs = [wide_spec, narrow_spec];
    let outputs = render_schedule_artifact_outputs_with_validation(
        &specs,
        MaterializationDiagnostics { row_progress: true },
        |_, _| Ok(()),
    )
    .expect("render generated outputs");
    for dest in publish_artifact_outputs(outputs).expect("publish generated outputs") {
        println!("wrote {}", dest.display());
    }
    if let Some(path) = std::env::var_os("AERIE_CATALOG_AUDIT_JSON") {
        write_catalog_audit(Path::new(&path));
    }
}

#[test]
fn every_grouped_producer_has_a_fresh_scalar_preplan() {
    let scalar_keys = all_scalar_preplan_keys();
    assert_eq!(
        scalar_keys.len(),
        WIDTH14_PRECOMMIT_KEYS.len() + WIDTH4_SCALAR_KEYS.len()
    );
    for (index, key) in scalar_keys.iter().enumerate() {
        assert!(
            !scalar_keys[..index].contains(key),
            "duplicate scalar preplan key {key:?}"
        );
    }

    let producers = selected_grouped_producer_keys();
    for producer in &producers {
        assert!(
            scalar_keys.contains(producer),
            "selected grouped producer {producer:?} is absent from scalar regeneration"
        );
    }
    // The runtime resolves scalar rows only as precommit profiles, so a
    // scalar row that no emitted grouped row commits is dead weight.
    for key in &scalar_keys {
        assert!(
            producers.contains(key),
            "scalar row {key:?} is no emitted grouped row's producer"
        );
    }
}

/// The ledger consumes the same audited schedules the runtime consumes.
#[test]
fn ledger_audit_matches_catalogs() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../specs/scripts/falcon_pcs_catalog_audit.json");
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("checked-in PCS ledger audit"))
            .expect("PCS audit JSON");
    assert_eq!(report["akita_revision"], AUDIT_AKITA_REVISION);
    let recorded_estimates = recorded_planner_estimates(&report);
    let rebuilt = catalog_audit_report(&[], &recorded_estimates);
    assert_eq!(
        serde_json::to_vec(&rebuilt).expect("serialize rebuilt audit"),
        std::fs::read(&path).expect("checked-in PCS ledger audit"),
        "compact audit writer must reproduce the checked-in audit from catalog rows and its recorded planner estimate scalars",
    );
    let families = report["families"].as_array().expect("audit families");
    assert_eq!(families.len(), 2);
    let (wide_catalog, narrow_catalog) = checked_in_catalogs();
    check_ledger_family(&wide_catalog, &families[0]);
    check_ledger_family(&narrow_catalog, &families[1]);
}

fn check_ledger_family<C: CommitmentConfig>(
    catalog: &TrustedScheduleCatalog<C>,
    family: &serde_json::Value,
) {
    assert_eq!(family["name"], C::schedule_family_name());
    let rows = family["rows"].as_array().expect("audit rows");
    assert_eq!(rows.len(), catalog.len());
    for (row, exported) in catalog.rows().zip(rows) {
        assert_eq!(
            exported["profiles"],
            serde_json::to_value(row.profiles()).unwrap()
        );
        assert_eq!(
            exported["schedule_json"].as_str().unwrap(),
            serde_json::to_string(row.schedule()).unwrap()
        );
        assert_eq!(
            exported["sis_occurrences"],
            serde_json::to_value(audit_occurrences(row.schedule())).unwrap()
        );
        assert_eq!(
            exported["fold_count"].as_u64().unwrap(),
            (row.schedule().recursive_folds.len() + 2) as u64
        );
    }
}
