//! Stage-level harness for the Falcon aggregation proof roundtrip.
//!
//! Runs the real-commitment roundtrip at the requested live batch sizes
//! and prints one TSV row per batch with the Section 12
//! stage decomposition. Every record is generated through the reference
//! `fn-dsa` key-generation and signing algorithms for the selected parameter
//! set (Falcon-512 by default, Falcon-1024 with `--degree=1024`). Native validation
//! and witness derivation are part of the measured raw-record-to-proof path.
//! The deterministic fixture RNG exists only to make benchmark vectors
//! reproducible; its keys are not production key material.
//!
//! Usage:
//! ```text
//! cargo bench -p aerie-bench \
//!   --features falcon-bench \
//!   --bench falcon -- 16384
//! cargo bench -p aerie-bench \
//!   --features falcon-bench \
//!   --bench falcon -- 16384 --degree=1024
//! ```
//!
//! For tracing and memory/CPU profiling:
//! ```text
//! cargo bench -p aerie-bench \
//!   --features falcon-bench,profiling \
//!   --bench falcon -- 16384 --profile
//! ```
//! `--profile` writes the trace under `target/traces`; `--profile=<dir>`
//! chooses the directory. Profiling requires exactly one explicit batch size because
//! the subscriber is process-global. Profiled timings include
//! instrumentation overhead and are not citable benchmark results.
//! `AERIE_PROFILE_FILTER` overrides the subscriber filter, and
//! `AERIE_MONITOR_INTERVAL_MS` sets the sampling interval (50–10,000 ms).
//!
//! Batch sizes default to `8 1024 16384` when none are given. Peak RSS is
//! best measured externally, e.g. `/usr/bin/time -l` on macOS.
//!
//! The first run generates and caches a canonical honest corpus under
//! `target/honest-fixtures/`. Later runs reload it but still parse every
//! record, enforce distinct keys/signatures/messages and the selected salt
//! mode, and natively
//! verify every signature while deriving the measured proof witness. Pass
//! `--regenerate-honest-fixture` to replace the cached corpus. Set
//! `AERIE_HONEST_FIXTURE_DIR` to use another cache directory.
//! Pass `--common-salt` to benchmark shared-salt signatures, or add
//! `--generate-honest-fixtures-only` to write fixtures without proving.
//!
//! Fixture acquisition is outside timing. Reusable, shape-dependent PCS setup
//! is reported separately. `prove_ms` starts from canonical public keys,
//! signatures, and messages and ends after wire encoding of the proof.
//! Parallel and single-thread verification each start from public keys plus
//! proof bytes and include decoding, verifier-local target derivation,
//! streaming public-key evaluation, and the full protocol verifier.
//! The benchmark's global allocator also counts verifier heap: the bytes a
//! standalone verifier setup keeps, and for each verification the most bytes
//! held at once beyond its inputs (public keys, messages, proof bytes, and
//! the setup).
//! With the `heap-profile` feature, `--heap-profile=prove`,
//! `--heap-profile=verify` or `--heap-profile=verifier-setup` records every
//! allocation of that phase with dhat
//! and writes `target/heap-profiles/dhat-falcon<degree>-<records>-<phase>.json`
//! (view it at <https://nnethercote.github.io/dh_view/dh_view.html>). Build
//! with `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only` for readable stacks.
//! A heap-profiled run's timings are not citable.
//! Each row is followed on stderr by a byte breakdown: salt transport,
//! header plus non-salt PIOP, and PCS commitments/opening with its length word.

#[path = "falcon/args.rs"]
mod args;

use args::{BenchArgs, HeapPhase};

use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use aerie_falcon::AggregationStatement;
use aerie_falcon::{
    verify_aggregate_measured,
    wire::{SaltMode, TRANSPORT_SALT_BYTES},
    AggregateVerifyMetrics, AggregationParams, FalconDegree, FalconVerifierSetup,
};
use aerie_falcon_native::{Falcon1024, Falcon512};
use aerie_falcon_witness::honest::{honest_aggregation, HonestParams};
use aerie_falcon_witness::native::{record::DirectRecord, witness::verify_signature_native};
use aerie_prover::{
    prove_aggregate_measured_with_mode, AggregateProveMetrics, AggregationWitness,
    FalconAggregateSetup, PreprocessingMetrics,
};

/// The parameter set under benchmark and its honest corpus source.
trait BenchParams: AggregationParams + HonestParams {
    /// Fixture file stem and report label.
    const LABEL: &'static str;
    /// Fresh honest keys and signatures from the reference signer.
    fn honest_corpus(
        records: usize,
        seed: u64,
        common_salt: bool,
    ) -> Result<(AggregationStatement<Self>, AggregationWitness<Self>), Box<dyn Error>> {
        let salt_mode = if common_salt {
            aerie_falcon_witness::honest::SaltMode::Common
        } else {
            aerie_falcon_witness::honest::SaltMode::PerRecord
        };
        Ok(honest_aggregation::<Self>(records, seed, salt_mode)?)
    }
}

impl BenchParams for Falcon512 {
    const LABEL: &'static str = "falcon512";
}

impl BenchParams for Falcon1024 {
    const LABEL: &'static str = "falcon1024";
}

/// Peak-heap metering for the verifier. The counters change only while a
/// measurement is active, so proving pays one relaxed load per allocation.
#[allow(unsafe_code)]
#[cfg_attr(feature = "heap-profile", allow(dead_code))]
mod heap_meter {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};

    pub struct MeteredAllocator;

    static ACTIVE: AtomicBool = AtomicBool::new(false);
    static LIVE: AtomicIsize = AtomicIsize::new(0);
    static PEAK: AtomicIsize = AtomicIsize::new(0);

    fn record(delta: isize) {
        if ACTIVE.load(Ordering::Relaxed) {
            let live = LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
    }

    // SAFETY: every call forwards to `System` with the caller's arguments
    // and only updates counters afterwards.
    unsafe impl GlobalAlloc for MeteredAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let pointer = unsafe { System.alloc(layout) };
            if !pointer.is_null() {
                record(layout.size().cast_signed());
            }
            pointer
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let pointer = unsafe { System.alloc_zeroed(layout) };
            if !pointer.is_null() {
                record(layout.size().cast_signed());
            }
            pointer
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer, layout) };
            record(-layout.size().cast_signed());
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let resized = unsafe { System.realloc(pointer, layout, new_size) };
            if !resized.is_null() {
                record(new_size.cast_signed() - layout.size().cast_signed());
            }
            resized
        }
    }

    /// Net heap allocated by a metered call.
    #[derive(Clone, Copy)]
    pub struct HeapUse {
        /// The most bytes held at once during the call.
        pub peak_bytes: usize,
        /// Bytes still held when the call returned, including its result.
        pub retained_bytes: usize,
    }

    /// Run `f` and return its result with the net heap it allocated. Frees
    /// of memory allocated before the call lower the running total, so `f`
    /// should not drop large values it did not allocate.
    pub fn metered<T>(f: impl FnOnce() -> T) -> (T, HeapUse) {
        LIVE.store(0, Ordering::SeqCst);
        PEAK.store(0, Ordering::SeqCst);
        ACTIVE.store(true, Ordering::SeqCst);
        let value = f();
        ACTIVE.store(false, Ordering::SeqCst);
        let heap = HeapUse {
            peak_bytes: PEAK.load(Ordering::SeqCst).max(0).cast_unsigned(),
            retained_bytes: LIVE.load(Ordering::SeqCst).max(0).cast_unsigned(),
        };
        (value, heap)
    }
}

#[cfg(not(feature = "heap-profile"))]
#[global_allocator]
static ALLOCATOR: heap_meter::MeteredAllocator = heap_meter::MeteredAllocator;

// dhat replaces the metering allocator, so a heap-profiled run reports zero
// verifier heap in its TSV row.
#[cfg(feature = "heap-profile")]
#[global_allocator]
static ALLOCATOR: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "heap-profile")]
type HeapProfiler = dhat::Profiler;
#[cfg(not(feature = "heap-profile"))]
type HeapProfiler = ();

/// Start a dhat profile when `phase` is the one requested; dropping the
/// returned value writes the profile.
fn start_heap_profile<P: BenchParams>(
    requested: Option<HeapPhase>,
    phase: HeapPhase,
    records: usize,
) -> Option<HeapProfiler> {
    if requested != Some(phase) {
        return None;
    }
    #[cfg(feature = "heap-profile")]
    {
        let directory = workspace_root().join("target/heap-profiles");
        fs::create_dir_all(&directory).expect("create heap profile directory");
        let path = directory.join(format!(
            "dhat-falcon{}-{records}-{}.json",
            P::DEGREE_ID.degree(),
            match phase {
                HeapPhase::Prove => "prove",
                HeapPhase::Verify => "verify",
                HeapPhase::VerifierSetup => "verifier-setup",
            }
        ));
        eprintln!("# heap profile: {}", path.display());
        Some(
            dhat::Profiler::builder()
                .file_name(path)
                .trim_backtraces(Some(64))
                .build(),
        )
    }
    #[cfg(not(feature = "heap-profile"))]
    {
        let _ = records;
        None
    }
}

const HONEST_FIXTURE_SEED: u64 = 0xa3e1_e5f1_6e51_6e31;
const HONEST_CACHE_MAGIC: &[u8; 4] = b"AHF1";
const BENCHMARK_SCHEMA: u32 = 8;

struct HonestCorpus<P: BenchParams> {
    statement: AggregationStatement<P>,
    witness: AggregationWitness<P>,
    cache_path: PathBuf,
    cache_hit: bool,
    source_ms: f64,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("aerie-bench crate must be inside the workspace crates directory")
        .to_path_buf()
}

fn honest_fixture_path<P: BenchParams>(records: usize, common_salt: bool) -> PathBuf {
    let directory = env::var_os("AERIE_HONEST_FIXTURE_DIR").map_or_else(
        || workspace_root().join("target/honest-fixtures"),
        PathBuf::from,
    );
    let salt_mode = if common_salt { "-common-salt" } else { "" };
    directory.join(format!(
        "fn-dsa-0.3.0-statement-records-v1-{}{salt_mode}-{records}-{HONEST_FIXTURE_SEED:016x}.bin",
        P::LABEL
    ))
}

fn validate_honest_corpus<P: BenchParams>(
    statement: &AggregationStatement<P>,
    witness: &AggregationWitness<P>,
    common_salt: bool,
) -> Result<(), io::Error> {
    if statement.len() != witness.signatures().len() {
        return Err(io::Error::other(
            "honest fixture statement and witness lengths differ",
        ));
    }
    let unique_keys = statement
        .public_keys()
        .iter()
        .map(|public_key| public_key.as_bytes().as_ref().to_vec())
        .collect::<HashSet<_>>();
    let unique_signatures = witness
        .signatures()
        .iter()
        .map(|signature| signature.as_bytes().as_ref().to_vec())
        .collect::<HashSet<_>>();
    let unique_salts = witness
        .signatures()
        .iter()
        .map(|signature| *signature.salt())
        .collect::<HashSet<_>>();
    let unique_messages = statement
        .messages()
        .iter()
        .map(Vec::as_slice)
        .collect::<HashSet<_>>();
    if unique_keys.len() != statement.len() {
        return Err(io::Error::other(
            "honest fixture contains a repeated public key",
        ));
    }
    if unique_signatures.len() != witness.signatures().len() {
        return Err(io::Error::other(
            "honest fixture contains a repeated signature",
        ));
    }
    let expected_salts = if common_salt {
        usize::from(!witness.signatures().is_empty())
    } else {
        witness.signatures().len()
    };
    if unique_salts.len() != expected_salts {
        let message = if common_salt {
            "common-salt fixture does not use exactly one salt"
        } else {
            "honest fixture contains a repeated salt"
        };
        return Err(io::Error::other(message));
    }
    if unique_messages.len() != statement.len() {
        return Err(io::Error::other(
            "honest fixture contains a repeated message",
        ));
    }
    Ok(())
}

fn write_honest_fixture<P: BenchParams>(
    cache_path: &Path,
    statement: &AggregationStatement<P>,
    witness: &AggregationWitness<P>,
) -> Result<(), Box<dyn Error>> {
    let statement_bytes = statement.canonical_encoding()?;
    let witness_bytes = witness.canonical_encoding()?;
    let mut encoded = Vec::with_capacity(12 + statement_bytes.len() + witness_bytes.len());
    encoded.extend_from_slice(HONEST_CACHE_MAGIC);
    encoded.extend_from_slice(&u64::try_from(statement_bytes.len())?.to_le_bytes());
    encoded.extend_from_slice(&statement_bytes);
    encoded.extend_from_slice(&witness_bytes);
    let parent = cache_path
        .parent()
        .ok_or_else(|| io::Error::other("honest fixture cache path has no parent"))?;
    fs::create_dir_all(parent)?;
    let temporary = cache_path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, encoded)?;
    fs::rename(temporary, cache_path)?;
    Ok(())
}

fn load_or_generate_honest_statement<P: BenchParams>(
    records: usize,
    common_salt: bool,
    regenerate: bool,
) -> Result<HonestCorpus<P>, Box<dyn Error>> {
    let cache_path = honest_fixture_path::<P>(records, common_salt);
    let source_start = Instant::now();
    if !regenerate && cache_path.is_file() {
        let _cache_span = tracing::info_span!(
            target: "aerie_falcon_statement",
            "statement.honest.cache_load",
            records = records
        )
        .entered();
        let encoded = fs::read(&cache_path)?;
        if encoded.len() < 12 || &encoded[..4] != HONEST_CACHE_MAGIC {
            return Err(io::Error::other("honest fixture cache has an invalid header").into());
        }
        let statement_len = usize::try_from(u64::from_le_bytes(encoded[4..12].try_into()?))?;
        let statement_end = 12_usize
            .checked_add(statement_len)
            .ok_or_else(|| io::Error::other("honest fixture cache length overflow"))?;
        if statement_end > encoded.len() {
            return Err(io::Error::other("honest fixture cache is truncated").into());
        }
        let statement = AggregationStatement::<P>::parse_canonical(&encoded[12..statement_end])?;
        let witness = AggregationWitness::<P>::parse_canonical(&encoded[statement_end..])?;
        if statement.len() != records {
            return Err(io::Error::other("honest fixture cache has the wrong batch size").into());
        }
        validate_honest_corpus(&statement, &witness, common_salt)?;
        return Ok(HonestCorpus {
            statement,
            witness,
            cache_path,
            cache_hit: true,
            source_ms: source_start.elapsed().as_secs_f64() * 1.0e3,
        });
    }

    let _generation_span = tracing::info_span!(
        target: "aerie_falcon_statement",
        "statement.honest.generate",
        records = records
    )
    .entered();
    let (statement, witness) = P::honest_corpus(records, HONEST_FIXTURE_SEED, common_salt)?;
    validate_honest_corpus(&statement, &witness, common_salt)?;
    let source_ms = source_start.elapsed().as_secs_f64() * 1.0e3;
    write_honest_fixture(&cache_path, &statement, &witness)?;
    Ok(HonestCorpus {
        statement,
        witness,
        cache_path,
        cache_hit: false,
        source_ms,
    })
}

fn generate_honest_fixtures_only<P: BenchParams>(
    sizes: &[usize],
    common_salt: bool,
    regenerate: bool,
) -> Result<(), Box<dyn Error>> {
    let missing = sizes
        .iter()
        .copied()
        .filter(|records| regenerate || !honest_fixture_path::<P>(*records, common_salt).is_file())
        .collect::<HashSet<_>>();
    let Some(&max_records) = missing.iter().max() else {
        eprintln!("# all requested honest fixtures are already cached");
        return Ok(());
    };

    let started = Instant::now();
    let (full_statement, full_witness) =
        P::honest_corpus(max_records, HONEST_FIXTURE_SEED, common_salt)?;
    validate_honest_corpus(&full_statement, &full_witness, common_salt)?;
    for ((public_key, message), signature) in full_statement
        .public_keys()
        .iter()
        .zip(full_statement.messages())
        .zip(full_witness.signatures())
    {
        let record = DirectRecord::<P>::new(public_key.clone(), signature.clone(), message.clone());
        verify_signature_native(record.as_record_ref())?;
    }
    for records in sizes.iter().copied().filter(|n| missing.contains(n)) {
        let statement =
            AggregationStatement::<P>::new(full_statement.to_records()[..records].to_vec())?;
        let witness = AggregationWitness::<P>::new(full_witness.signatures()[..records].to_vec())?;
        validate_honest_corpus(&statement, &witness, common_salt)?;
        let path = honest_fixture_path::<P>(records, common_salt);
        write_honest_fixture(&path, &statement, &witness)?;
        eprintln!(
            "# generated honest fixture: degree={} records={records} salt_mode={} path={}",
            P::LABEL,
            if common_salt { "common" } else { "per-record" },
            path.display()
        );
    }
    eprintln!(
        "# generated requested fixture prefixes from one {max_records}-record signer corpus in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn report_honest_corpus<P: BenchParams>(records: usize, corpus: &HonestCorpus<P>) {
    eprintln!(
        "# honest inputs: records={records} source={} source_ms={:.1} cache={} \
         (fixture acquisition excluded from setup/prove/verify)",
        if corpus.cache_hit {
            "canonical-cache"
        } else {
            "reference-keygen-sign"
        },
        corpus.source_ms,
        corpus.cache_path.display(),
    );
}

fn milliseconds(nanoseconds: u128) -> f64 {
    nanoseconds as f64 / 1.0e6
}

/// The canonical group order of `PcsStageMetrics::commit_group_ns`, the
/// same four groups for both parameter sets.
const GROUP_LABELS: [&str; 4] = ["s1", "slack", "digits", "budget"];

#[allow(clippy::similar_names)]
#[allow(clippy::too_many_arguments)]
fn report(
    records: usize,
    group_count: usize,
    setup_ms: f64,
    prove: &AggregateProveMetrics,
    verify_parallel: &AggregateVerifyMetrics,
    verify_single: &AggregateVerifyMetrics,
    verifier_setup_bytes: usize,
    verify_heap: VerifyHeap,
    wire_bytes: usize,
) {
    let PreprocessingMetrics {
        record_materialize_ns,
        native_verify_ns,
        direct_witness_ns,
        pack_ns,
        ..
    } = prove.preprocessing;
    let metrics = &prove.protocol;
    let commit_groups = GROUP_LABELS
        .iter()
        .zip(metrics.commit_group_ns)
        .take(group_count)
        .map(|(label, ns)| format!("{label}:{:.1}", milliseconds(ns)))
        .collect::<Vec<_>>()
        .join(";");
    println!(
        "{records}\t{setup_ms:.1}\t{prove_total:.1}\t\
         {transport_materialize:.1}\t{transport_binding:.1}\t{record_materialize:.1}\t{native_verify:.1}\t\
         {direct_witness:.1}\t{pack:.1}\t\
         {encodability:.1}\t{groups:.1}\t\
         {commit:.1}\t{projection:.1}\t{stage1:.1}\t{stage2:.1}\t{stage3:.1}\t\
         {pcs_prove:.1}\t{output_encode:.1}\t{verify_parallel_total:.1}\t{verify_parallel_decode:.1}\t\
         {verify_parallel_h2p:.1}\t{verify_parallel_binding:.1}\t\
         {stage_verify:.1}\t{pcs_verify:.1}\t{verify_single_total:.1}\t\
         {stage_verify_single:.1}\t{pcs_verify_single:.1}\t{verifier_setup_heap}\t\
         {verify_parallel_heap}\t{verify_single_heap}\t{commit_bytes}\t\
         {proof_bytes}\t{wire_bytes}\t{commit_groups}",
        prove_total = milliseconds(prove.total_ns),
        transport_materialize = milliseconds(prove.transport_materialize_ns),
        transport_binding = milliseconds(prove.transport_binding_ns),
        record_materialize = milliseconds(record_materialize_ns),
        native_verify = milliseconds(native_verify_ns),
        direct_witness = milliseconds(direct_witness_ns),
        pack = milliseconds(pack_ns),
        encodability = milliseconds(metrics.encodability_ns),
        groups = milliseconds(metrics.group_materialization_ns),
        commit = milliseconds(metrics.commit_ns),
        projection = milliseconds(metrics.projection_ns),
        stage1 = milliseconds(metrics.stage1_ns),
        stage2 = milliseconds(metrics.stage2_ns),
        stage3 = milliseconds(metrics.stage3_ns),
        pcs_prove = milliseconds(metrics.pcs_prove_ns),
        output_encode = milliseconds(prove.encode_ns),
        verify_parallel_total = milliseconds(verify_parallel.total_ns),
        verify_parallel_decode = milliseconds(verify_parallel.decode_ns),
        verify_parallel_h2p = milliseconds(verify_parallel.hash_to_point_ns),
        verify_parallel_binding = milliseconds(verify_parallel.binding_ns),
        stage_verify = milliseconds(verify_parallel.stage_verify_ns),
        pcs_verify = milliseconds(verify_parallel.pcs_verify_ns),
        verify_single_total = milliseconds(verify_single.total_ns),
        stage_verify_single = milliseconds(verify_single.stage_verify_ns),
        pcs_verify_single = milliseconds(verify_single.pcs_verify_ns),
        verifier_setup_heap = verifier_setup_bytes,
        verify_parallel_heap = verify_heap.all_threads,
        verify_single_heap = verify_heap.one_thread,
        commit_bytes = metrics.commitment_bytes,
        proof_bytes = metrics.pcs_proof_bytes,
    );
}

fn report_byte_breakdown(
    records: usize,
    salt_mode: SaltMode,
    prove: &AggregateProveMetrics,
    wire_bytes: usize,
) {
    let metrics = &prove.protocol;
    let salts_bytes = match salt_mode {
        SaltMode::PerRecord => records * TRANSPORT_SALT_BYTES,
        SaltMode::Common => TRANSPORT_SALT_BYTES,
    };
    // The PCS bucket includes the group commitments, its length word, and
    // the opening argument; the remainder is header and non-salt PIOP data.
    let pcs_bytes = metrics.commitment_bytes + 8 + metrics.pcs_proof_bytes;
    let non_salt_piop_bytes = wire_bytes - salts_bytes - pcs_bytes;
    eprintln!(
        "# byte_breakdown records={records} salt_mode={} salts_bytes={salts_bytes} \
         header_piop_bytes={non_salt_piop_bytes} pcs_bytes={pcs_bytes}",
        salt_mode_name(salt_mode),
    );
}

#[cfg(feature = "profiling")]
fn profile_path(dir: &str, records: usize) -> PathBuf {
    let requested = PathBuf::from(dir);
    let directory = if requested.is_absolute() {
        requested
    } else {
        workspace_root().join(requested)
    };
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    directory.join(format!(
        "falcon-{records}-{timestamp}-{}.json",
        std::process::id()
    ))
}

#[cfg(feature = "profiling")]
fn profiling_interval() -> std::time::Duration {
    let milliseconds = env::var("AERIE_MONITOR_INTERVAL_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(50)
        .clamp(50, 10_000);
    std::time::Duration::from_millis(milliseconds)
}

/// Name the global Rayon workers used by the benchmark.
fn configure_rayon() -> Result<(), std::io::Error> {
    rayon::ThreadPoolBuilder::new()
        .thread_name(|index| format!("aerie-rayon-{index}"))
        .build_global()
        .map_err(|error| std::io::Error::other(error.to_string()))
}

/// Peak heap bytes of the all-threads and one-thread verifications.
#[derive(Clone, Copy)]
struct VerifyHeap {
    all_threads: usize,
    one_thread: usize,
}

fn run_once<P: BenchParams>(
    statement: &AggregationStatement<P>,
    witness: &AggregationWitness<P>,
    setup: &Arc<FalconAggregateSetup<P>>,
    single_thread_pool: &Arc<rayon::ThreadPool>,
    salt_mode: SaltMode,
    heap_phase: Option<HeapPhase>,
) -> Result<
    (
        AggregateProveMetrics,
        AggregateVerifyMetrics,
        AggregateVerifyMetrics,
        VerifyHeap,
        usize,
    ),
    String,
> {
    let records = statement.len();
    let (proof_bytes, prove) = {
        let _heap_profile = start_heap_profile::<P>(heap_phase, HeapPhase::Prove, records);
        prove_aggregate_measured_with_mode(setup, statement, witness, salt_mode)
            .map_err(|error| error.to_string())?
    };
    let (verified_parallel, verify_parallel) = {
        let _heap_profile = start_heap_profile::<P>(heap_phase, HeapPhase::Verify, records);
        tracing::info_span!(target: "aerie_falcon", "v1.verify.parallel")
            .in_scope(|| verify_aggregate_measured(setup.verifier(), statement, &proof_bytes))
            .map_err(|error| error.to_string())?
    };
    let (verified_single, verify_single) =
        tracing::info_span!(target: "aerie_falcon", "v1.verify.single_thread").in_scope(|| {
            single_thread_pool.install(|| {
                verify_aggregate_measured(setup.verifier(), statement, &proof_bytes)
                    .map_err(|error| error.to_string())
            })
        })?;
    if verified_parallel.ledger != verified_single.ledger {
        return Err("parallel and single-thread verifiers produced different ledgers".to_owned());
    }
    // Heap is metered on separate passes, so the timed verifications
    // above carry no counting overhead.
    let verify_once = || {
        verify_aggregate_measured(setup.verifier(), statement, &proof_bytes)
            .map(|_| ())
            .map_err(|error| error.to_string())
    };
    let (all_threads, all_threads_heap) = heap_meter::metered(verify_once);
    all_threads?;
    let (one_thread, one_thread_heap) =
        heap_meter::metered(|| single_thread_pool.install(verify_once));
    one_thread?;
    let verify_heap = VerifyHeap {
        all_threads: all_threads_heap.peak_bytes,
        one_thread: one_thread_heap.peak_bytes,
    };
    Ok((
        prove,
        verify_parallel,
        verify_single,
        verify_heap,
        proof_bytes.len(),
    ))
}

fn run_batches<P: BenchParams>(args: &BenchArgs) -> Result<(), Box<dyn Error>> {
    let group_count = P::GROUP_COUNT;
    let salt_mode = args.salt_mode;
    let heap_phase = args.heap_phase;
    for &records in &args.sizes {
        let padded = P::DEGREE_ID.padded_records(records)?;
        println!("# batch_shape\tlive={records}\tpadded={padded}");
        #[cfg(feature = "profiling")]
        let trace = if let Some(dir) = &args.profile_dir {
            Some(aerie_profiling::install(
                &profile_path(dir, records),
                profiling_interval(),
            )?)
        } else {
            None
        };
        let corpus = load_or_generate_honest_statement::<P>(
            records,
            salt_mode == SaltMode::Common,
            args.regenerate,
        )?;
        report_honest_corpus(records, &corpus);
        // A standalone verifier setup, built only to count the heap a
        // verifier keeps; verification below borrows the shared setup.
        let (verifier_setup, verifier_setup_heap) = heap_meter::metered(|| {
            // The profile ends while the setup is alive, so its final
            // snapshot is exactly what the setup keeps.
            let _heap_profile =
                start_heap_profile::<P>(heap_phase, HeapPhase::VerifierSetup, records);
            FalconVerifierSetup::<P>::new(records)
        });
        drop(verifier_setup?);
        let setup_start = Instant::now();
        let setup = Arc::new(FalconAggregateSetup::<P>::new(records)?);
        let setup_ms = setup_start.elapsed().as_secs_f64() * 1.0e3;
        let single_thread_pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .thread_name(|_| "aerie-v1-verifier-single".to_owned())
                .build()?,
        );
        let (prove, verify_parallel, verify_single, verify_heap, wire_bytes) = run_once::<P>(
            &corpus.statement,
            &corpus.witness,
            &setup,
            &single_thread_pool,
            salt_mode,
            heap_phase,
        )?;
        report(
            records,
            group_count,
            setup_ms,
            &prove,
            &verify_parallel,
            &verify_single,
            verifier_setup_heap.retained_bytes,
            verify_heap,
            wire_bytes,
        );
        report_byte_breakdown(records, salt_mode, &prove, wire_bytes);
        #[cfg(feature = "profiling")]
        if let Some(trace) = trace {
            let path = trace.finish()?;
            eprintln!("# trace written to {}", path.display());
        }
    }
    Ok(())
}

const USAGE: &str = "Usage: falcon [--bench] [positive-batch-size ...] [--degree=512|1024] \
                     [--common-salt] [--generate-honest-fixtures-only] [--regenerate-honest-fixture] \
                     [--profile[=<dir>] (requires profiling feature; exactly one batch size)] \
                     [--heap-profile=prove|verify|verifier-setup (requires heap-profile feature)]";

fn salt_mode_name(mode: SaltMode) -> &'static str {
    match mode {
        SaltMode::Common => "common",
        SaltMode::PerRecord => "per-record",
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    if env::args_os()
        .skip(1)
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!("{USAGE}");
        return Ok(());
    }
    let args = BenchArgs::parse(env::args_os().skip(1)).unwrap_or_else(|error| {
        eprintln!("error: {error}\n{USAGE}");
        std::process::exit(2);
    });
    configure_rayon()?;
    let degree = args.degree;
    let salt_mode = args.salt_mode;
    println!("# benchmark_schema\t{BENCHMARK_SCHEMA}");
    println!("# profiled\t{}", args.profile_dir.is_some());
    println!("# rayon_threads\t{}", rayon::current_num_threads());
    println!(
        "# simd_backend\tsmall_signed_dot={};key_unpack={}",
        aerie_simd::SmallSignedDotPlan::best_available()
            .unwrap_or_else(aerie_simd::SmallSignedDotPlan::scalar)
            .name(),
        aerie_simd::KeyUnpackPlan::best_available().name()
    );
    println!("# keccak_backend\t{}", aerie_keccak::backend_name());
    println!("# rustc\t{}", env!("AERIE_BENCH_RUSTC"));
    println!("# akita_rev\t{}", env!("AERIE_BENCH_AKITA_REV"));
    println!("# git_revision\t{}", env!("AERIE_BENCH_BUILD_REVISION"));
    println!(
        "# heap_profile\t{}",
        match args.heap_phase {
            None => "none",
            Some(HeapPhase::Prove) => "prove",
            Some(HeapPhase::Verify) => "verify",
            Some(HeapPhase::VerifierSetup) => "verifier-setup",
        }
    );
    println!("# parameter_set\tFalcon-{}", degree.degree());
    println!("# salt_mode\t{}", salt_mode_name(salt_mode));
    if args.generate_only {
        match degree {
            FalconDegree::Falcon512 => generate_honest_fixtures_only::<Falcon512>(
                &args.sizes,
                salt_mode == SaltMode::Common,
                args.regenerate,
            )?,
            FalconDegree::Falcon1024 => generate_honest_fixtures_only::<Falcon1024>(
                &args.sizes,
                salt_mode == SaltMode::Common,
                args.regenerate,
            )?,
        }
        return Ok(());
    }
    println!("# byte_breakdown_columns\tsalts_bytes\theader_piop_bytes\tpcs_bytes");
    println!(
        "records\tsetup_ms\tprove_ms\ttransport_materialize_ms\ttransport_binding_ms\t\
         record_materialize_ms\tnative_verify_ms\t\
         direct_witness_ms\tpack_ms\tencodability_ms\t\
         group_materialization_ms\tcommit_ms\t\
         projection_ms\tstage1_ms\tstage2_ms\tstage3_ms\tpcs_prove_ms\t\
         output_encode_ms\t\
         verify_parallel_ms\tverify_decode_ms\tverify_hash_to_point_ms\t\
         verify_binding_ms\tstage_verify_ms\tpcs_verify_ms\tverify_single_thread_ms\t\
         stage_verify_single_thread_ms\tpcs_verify_single_thread_ms\t\
         verifier_setup_heap_bytes\tverify_peak_heap_bytes\t\
         verify_single_thread_peak_heap_bytes\t\
         commitment_bytes\tpcs_proof_bytes\tenvelope_bytes\tcommit_groups_ms"
    );
    match degree {
        FalconDegree::Falcon512 => run_batches::<Falcon512>(&args)?,
        FalconDegree::Falcon1024 => run_batches::<Falcon1024>(&args)?,
    }
    println!(
        "# honest FN-DSA fixtures: distinct reference-generated keypairs, signatures, and messages; \
         salt_mode={}; deterministic benchmark randomness is not production key material",
        salt_mode_name(salt_mode),
    );
    Ok(())
}
