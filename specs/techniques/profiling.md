# Spec: profiling (native spans, Perfetto traces, memory/CPU counters)

| Field         | Value                                        |
|---------------|----------------------------------------------|
| Author(s)     | Michael Zhu |
| Created       | 2026-08-05                                   |
| Status        | implemented (span taxonomy v8) |

## 1. Current profiling support

The proof crates emit native `tracing` spans around Falcon protocol and
arithmetic-kernel work. Examples verified in the code include `v1.setup.total`,
`v1.prove.total`, `v1.verify.total`, `projection::project`, and
`stage1_driver_setup`. The `aerie-profiling` crate installs a Chrome-format tracing
subscriber and samples process memory and CPU for the Falcon benchmark when
the `aerie` crate's off-by-default `profiling` feature is enabled and the
`--profile` flag is passed. The trace can be opened in
[Perfetto](https://ui.perfetto.dev).

Span instrumentation is present in the proof crates independently of the
profiling feature. Without a subscriber, it does not write a trace; the
monitor is started only by the benchmark's profiling path. The ordinary
benchmark continues to report its `Instant`-based timing columns.

## 2. Architecture

The workspace member `crates/aerie-profiling` has no proof-system dependencies.
Only `aerie` depends on it, behind the off-by-default `profiling` feature:

```
crates/aerie-profiling
├── src/lib.rs        install() -> ProfilingGuard; env-filter defaults
├── src/monitor.rs    MetricsMonitor (background sampling thread)
└── src/rewrite.rs    post-run counter rewrite (Perfetto "ph":"C" tracks)
```

- `install(trace_path, interval)` builds
  `tracing_subscriber::registry()` with an `EnvFilter`
  (overridable with `AERIE_PROFILE_FILTER`) and a `tracing_chrome::ChromeLayer`
  (`include_args(true)`), sets it as the global default, starts the
  monitor, and returns a `ProfilingGuard` holding the chrome
  `FlushGuard`, the monitor, and the trace path.
- `ProfilingGuard::finish()` (also on `Drop`): stops the monitor,
  drops the `FlushGuard` (flushing the trace), then runs the counter
  rewrite in place. The bench must hold the guard in `main()` and drop
  it only after the roundtrip thread is joined. Argument validation runs
  before `install()`, so validation failures never leave a partial trace.
- The instrumented proof crates (`aerie-falcon-statement`, `aerie-falcon-native`, and `aerie-falcon`)
  carry an unconditional `tracing`
  dependency and span instrumentation at the boundaries listed in
  Section 4, via the `staged!` boundary helper (Section 3) where an
  Instant region exists and plain span guards where the region is
  span-only. They do not depend on `crates/aerie-profiling`; span emission
  with no subscriber installed compiles to a static-branch check and
  is free in every existing invocation.

The default `EnvFilter` enables the following Aerie targets and seven Akita
crates that emit spans or events at the pinned revision:

```
warn,aerie_falcon_statement=debug,aerie_pcs_config=debug,aerie_falcon=debug,\
aerie_profiling=debug,aerie_falcon_native=warn,akita_pcs=trace,akita_prover=trace,akita_cpu_backend=trace,\
akita_setup=trace,akita_types=trace,akita_algebra=trace,akita_verifier=trace
```

(`DEFAULT_ENV_FILTER` in `crates/aerie-profiling/src/lib.rs`.)

`aerie_falcon` owns the Falcon v1 protocol-stage and compact arithmetic-kernel
spans. This target is required for a complete v1 range-proof trace.

(The monitor's events carry the module-path target `aerie_profiling::monitor`
rather than a custom one: tracing's macros cannot combine an explicit
`target:` with dotted `counters.*` field names.)

Because the subscriber is the process-global default, spans reach it
from the 512 MB-stack roundtrip thread and from rayon workers without
per-thread setup. tracing-chrome assigns events to per-thread Perfetto
lanes; the roundtrip root span must be opened *inside* the spawned
closure so the stage tree sits in the worker's lane. Rayon tasks
appear in their own lanes without a parent span unless the submitting code
captures the current span and re-enters it on the worker. The PIOP scope does
so before opening its stage spans, preserving the `v1.piop.prove` hierarchy.

## 3. Contract: online work is complete and spans mirror timed regions

The taxonomy is versioned (`SPAN_TAXONOMY_VERSION = 9`). Accounting spans
either wrap exactly one existing Instant-timed region or form a sibling set
whose durations sum to one region. Nested diagnostic spans may subdivide an
accounting span without a separate timer. Ancestors and descendants overlap
and must not be summed; diagnostic siblings under the same parent are
disjoint. A Perfetto trace and a TSV row must tell the same story. Composite
TSV fields reconstruct from accounting-span sums.

Falcon v1 additionally enforces three non-overlapping top-level boundaries:

- `v1.setup.total` contains only state derived from the record count and PCS
  catalogs. It excludes all keys, signatures, salts, and messages and may be
  reused for every statement with the same shape.
- `v1.prove.total` starts with canonical public keys, signatures, and messages
  and ends after transport-envelope encoding. It includes native Falcon
  validation and every derived witness, table, group, commitment, proof stage,
  and PCS opening proof.
- `v1.verify.total` starts with canonical public keys and envelope bytes and
  ends at acceptance. It includes envelope decoding, verifier-local `hpk`
  derivation, deferred streaming `HashToPoint` evaluation, streaming
  public-key evaluation, transcript binding, stage replay, and PCS
  verification.

Fixture generation, fixture file I/O, and benchmark-only uniqueness checks are
input acquisition, not protocol work. The configured-pool and single-threaded
verifiers are separate measurements; neither is included in `prove_ms`, and
the single-threaded replay is not included in `verify_parallel_ms`.

**The contract is enforced by construction, not discipline.** The
instrumented crates share one boundary helper, the `staged!` macro,
exported from `aerie-falcon-statement` (the lowest instrumented crate, so
`aerie-falcon` inherits it as an existing dependency):

```rust
let value = aerie_falcon_statement::staged!("v1.commit.pre_seed", commit_ns, {
    // region body
});
```

One invocation defines both the Instant region (accumulating
`elapsed().as_nanos()` into the named `u128`, preserving the existing
`+=` semantics for multi-region fields) and the entered span, so the
span and the TSV column cannot drift apart: a stage that moves takes
its span with it. Span-only regions, such as the range kernel
diagnostics of Section 4, use a plain `info_span!(...).entered()` guard
instead.

**Dead timings are pruned, not mirrored.** Instant plumbing that
feeds no TSV column, no published sum, and no assertion is deleted
rather than given a span. Verify liveness by call graph, then delete or
mirror, never both-and-drift.

Instrumentation is observational only: no transcript-visible operation
moves (main commits stay before the matrix seed; challenge sampling
order is untouched), and proof bytes remain byte-identical with and
without a subscriber.

## 4. Span taxonomy v9

Accounting spans are named `<area>.<stage>[.<substage>]`; the crate's module
path is the tracing target. Rust-kernel diagnostics use `Type::method` names
to match Akita's executor traces.

**Roots.** The benchmark's top-level spans are `v1.setup.total`,
`v1.prove.total`, `v1.verify.parallel` and `v1.verify.single_thread`
(`crates/aerie/benches/falcon.rs`). `v1.roundtrip` in
`crates/aerie-falcon/src/protocol/pcs.rs` is a test-only root. Stage spans
are prefixed by name rather than discriminated by fields, so flat by-name
aggregation over a trace never conflates stages.

**Falcon v1 online path** (`transport.rs`, `production.rs`, `pcs.rs`):

| Span | Included work |
|---|---|
| `v1.setup.total` | Shape-dependent Akita prover setup, CPU preparation, NTT prewarming, and verifier setup |
| `v1.setup.witness_ntt` | Input-independent degree-1024 direct-witness NTT plans, twiddles, mixed-width CRT constants, and host-kernel selection |
| `v1.prove.total` | Complete canonical-record-to-envelope prover |
| `v1.prove.input.transport_materialize` | Canonical key/message views and direct salt extraction for the envelope |
| `v1.prove.input.transport_binding` | Public key, salt, and message transcript binding |
| `v1.prove.witness.records` | Pair public records with their private signatures |
| `v1.prove.witness.native_verify` | Decode, `HashToPoint`, `s1` derivation, ranges, and Falcon norm |
| `v1.prove.witness.direct` | Exact mixed-width CRT+NTT convolution, lift `v/w`, norm slack, relation checks, and direct writes into the final dense columns |
| `v1.prove.witness.direct.chunk` | One 256-record Rayon worker chunk inside `witness.direct`; chunks overlap and MUST NOT be summed |
| `v1.prove.witness.pack` | Allocation and zero-padding of the final dense witness columns; no intermediate `DirectRow` table is materialized by the production path |
| `v1.prove.witness.encodability` | Four `s2` source tables and three budget digits |
| `v1.prove.witness.groups` | Physical pre-seed group materialization |
| `v1.commit.group.field_conversion` | Base-field source conversion for one committed group |
| `v1.commit.group.dense_poly` | Dense PCS polynomial construction for one group |
| `v1.commit.group.import_source` | Import of one materialized group into the PCS source representation |
| `v1.commit.group.backend_commit` | Backend commitment work for one group |
| `v1.prove.protocol` | Commitments through the final PCS opening proof |
| `v1.piop.prove` | The complete three-stage PIOP, re-entered beneath the caller span when it runs on a Rayon worker |
| `v1.stage1.prove` | Stage-one proof generation |
| `v1.stage2.prove_with_stage3_prepare` | Stage-two proof generation together with overlapped stage-three preparation; this region backs `stage2_ms` |
| `v1.stage3.prove` | Stage-three proof generation from the prepared tables |
| `v1.prove.output.encode` | Canonical transport-envelope encoding |
| `v1.verify.parallel`, `v1.verify.single_thread` | Separate verifier executions |
| `v1.verify.total` | Complete public-input-and-envelope verifier |
| `v1.verify.input.decode` | Envelope decoding and shape validation |
| `v1.verify.input.hashed_keys` | Fixed-width `hpk` derivation needed by public-record binding before stage one |
| `v1.verify.input.binding` | Canonical public-record binding |
| `v1.verify.protocol` | Transcript replay, proof stages, and PCS verification |
| `v1.stages.verify.targets` | Streamed `HashToPoint` target contraction at `r_Pi`, inside `v1.stages.verify` |

The equality-factored range executors emit kernel spans inside the stage
timers. Their concrete kernel types are `ProductGroup`, `QuarticGroup`, and
`QuadraticGroup`; the last supplies the compact prefix for stage-three
range-source links. Stage 1 has relation-specific setup, round, and bind
spans. Stage 3 uses a fused driver, with one compact prefix round followed by
fused setup, round, and contraction spans. The driver and kernel spans are:

| Span | Included work |
|---|---|
| `stage1_driver_setup` | Stage-one shape validation and setup shared across relations |
| `stage1_driver_claim_combine`, `stage1_driver_transcript` | Stage-one claim aggregation and transcript work, with `round` and `phase` fields |
| `basis16_product_setup`, `basis16_product_round`, `basis16_product_bind` | Stage-one grouped basis-16 product setup and one active or inactive round/bind; round spans carry `round` and `phase` |
| `projection_link_setup`, `projection_link_coefficient_round`, `projection_link_slack_round`, `projection_link_coefficient_bind`, `projection_link_slack_bind` | Stage-one projection transpose and native slack setup, relation rounds, and folds; round spans carry `round` and `phase` |
| `projection_link_rest_weights`, `projection_link_first_slack_tables`, `projection_link_slack_weights` | Setup diagnostics for the projection transpose, first slack channels, and native slack weights |
| `stage3_driver_setup` | Stage-three validation (`phase=validate`) and prover target evaluation (`phase=target_evaluation`) |
| `stage3_driver_claim_combine`, `stage3_driver_transcript` | Initial and per-round claim aggregation; gamma, round, and terminal transcript work, distinguished by `phase` |
| `stage3_fused_setup`, `stage3_fused_round`, `stage3_compact_round0`, `stage3_fused_contract` | Fused stage-three engine setup, compact first-round computation, challenge binding, and per-round contraction |
| `ring_functional_setup`, `stage3::ring_functional.lambda`, `stage3::ring_functional.row_weights`, `stage3::ring_functional.key_dot` | Ring-functional setup, challenge-derived weights, and public-key dot products |
| `ProductGroup::new`, `QuarticGroup::new`, `QuadraticGroup::new` | Construct the stage-one product, stage-two quartic, and stage-three quadratic range kernels |
| `product_group_round`, `quartic_group_round`, `range_source_round` | One sumcheck round, with stable shape fields plus `round` and `phase` fields distinguishing bivariate, compact pair/quartet/octet, and dense work |
| `product_group_bind`, `quartic_group_bind`, `range_source_bind` | One challenge ingestion, with stable shape fields plus `round` and `phase` fields distinguishing compact retention, materialization, and dense folding |
| `ProductGroup::accumulate_factored_pattern_weights`, `ProductGroup::accumulate_pattern_weights`, `QuarticGroup::accumulate_factored_pattern_weights`, `QuarticGroup::accumulate_pattern_weights`, `QuadraticGroup::accumulate_factored_pattern_weights`, `QuadraticGroup::accumulate_pattern_weights` | Equality-weight accumulation inside compact rounds |
| `ProductGroup::build_pair_fold_lut`, `ProductGroup::build_quartet_fold_lut`, `QuarticGroup::build_quartet_fold_lut`, `QuadraticGroup::build_quartet_fold_lut` | Challenge-dependent compact lookup construction |
| `ProductGroup::materialize_after_first_round`, `ProductGroup::materialize_after_second_round`, `ProductGroup::materialize_after_third_round`, `QuarticGroup::materialize_after_first_round`, `QuarticGroup::materialize_after_second_round`, `QuarticGroup::materialize_after_third_round`, `QuadraticGroup::materialize_compact_sources`, `QuadraticGroup::materialize_and_contract_after_third_round`, `kernels::fold_dense_rows` | Compact-to-dense transitions and later in-place folds |

These spans are diagnostic children, not additional benchmark regions. Sum
only disjoint siblings at one nesting level; never add a child to its parent.
The compact and fused spans may overlap with their enclosing stage spans. No
per-pair, per-row, or per-Rayon-chunk span is created. Rayon work continues to
appear on worker lanes without a parent span. Benchmark schema 7 removes the
unused `statement_binding_ms` column. A profiled run remains non-citable.

The Falcon benchmark reports `stage1_ms`, `stage2_ms`, and `stage3_ms`
separately, along with per-group commitment timings.
The wire verifier constructs no
padded `h` table: `stage3::ring_functional` unpacks each canonical
public key directly into the `falcon.md` Section 8 functional evaluation, and padded key
rows are implicit zeros. Packed-key work is therefore included in
`stage_verify_ms`.
The verifier also materializes no `HashToPoint` target table: it streams each
record's `C` from the bound public inputs into the contraction at `r_Pi`.
Verification records fixed-width hashed-key derivation in
`v1.verify.input.hashed_keys` and the streamed target contraction in
`v1.stages.verify.targets` (metric `target_evaluation_ns`). Together these
populate `verify_hash_to_point_ms`. The target span nests inside
`v1.stages.verify`, and `stage_verify_ms` excludes it.

The `PreprocessingMetrics` timing fields map one-to-one to the four
`v1.prove.witness.*` spans.
`AggregateProveMetrics` separately exposes transport binding and envelope
encoding, while `PcsStageMetrics` exposes encodability and group materialization.
A material unnamed gap under either top-level online span is an instrumentation
bug.

The direct witness requires the *integer* ordinary convolution of centered
`h` and `s2`, not only Falcon's negacyclic product modulo 12289. Reduction in
the Falcon field would erase the quotient needed to derive and check `v` and
`w`. The dense path therefore zero-pads both degree-512 polynomials to 1024,
runs a cyclic NTT under one 30-bit Akita prime and a second NTT under Falcon's
14-bit prime 12289, and reconstructs the unique centered integer coefficient.
The CRT modulus is 13,194,609,258,497. This exceeds twice both the honest-input
bound `512 * 6144 * 2047 = 6,439,305,216` and the stronger full-range `i16`
bound `512 * 32768 * 32768 = 549,755,813,888`. Plans and twiddles belong to
`v1.setup.witness_ntt`; transforms of statement-dependent `h` and `s2` belong
to `v1.prove.witness.direct`.

The `functional_kernel_table` child of `v1.stage3.prove` is one materialization. Perfetto
stores its begin and end as two events with the same name; those events do not
represent two computations. The prover retains the resulting extension-field
vector and folds it in place through the stage-three ring sumcheck. At Falcon-512 with 16,384 padded
records it has 8,388,608 `Fp64²` cells and occupies 134,217,728 bytes (128 MiB).
The table cannot use an `i16` or `i32` carrier because its cells depend on the
extension-field challenge `r_Pi`.

`crates/aerie-falcon-native` emits two native-verification child spans,
`v1.prove.witness.native_verify.hash_to_point` and
`v1.prove.witness.native_verify.verify_records`, from
`crates/aerie-falcon-native/src/witness.rs`. The default filter explicitly keeps
`aerie_falcon_native` at warn because the `aerie_falcon=debug` directive also
matches longer target prefixes. These info-level spans therefore require an
`AERIE_PROFILE_FILTER` override that
includes `aerie_falcon_native=info`. The override replaces the entire filter; retain the
default directives above to include the other protocol and dependency spans.

## 5. The metrics monitor

`MetricsMonitor` follows jolt's `monitor.rs`, reduced:

- `MetricsMonitor::start(interval: Duration)` spawns one named thread
  (`aerie-monitor`); default interval 50 ms, clamped to
  [50 ms, 10 s]. Stop flag is an `Arc<AtomicBool>`; the sleep is
  chunked so `Drop` (signal + join) returns within ~50 ms rather than
  a full interval. `#[must_use]`.
- Per sample, via `sysinfo` (refreshing only the needed components):
  process RSS, global CPU utilization %, average active cores
  (cpu% × core count / 100), count of cores above a 0.1% activity
  threshold, and — Linux only, via `/proc/self/task` `read_dir` —
  live thread count (absent on macOS rather than fabricated).
- Each sample is emitted as one `tracing::debug!` event (module-path
  target `aerie_profiling::monitor`) with structured fields named
  `counters.memory_gib`, `counters.cpu_percent`,
  `counters.cores_active_avg`, `counters.cores_active`,
  `counters.thread_count`. With `include_args(true)` these land in the
  chrome trace as instant events carrying `args`.
- The monitor emits **no derived peak counter**: the memory track is a
  sampled time series (it can miss transients between ticks), and a
  sampled maximum would invite quoting as a peak figure.

All monitor code is safe Rust (`unsafe_code = forbid` holds
workspace-wide); no `libc`/`getrusage` FFI, which is also why the
monitor does not attempt a true high-water-mark reading.

## 6. Trace postprocessing: counter tracks

Perfetto renders `"ph": "C"` (counter) events as continuous
memory/CPU tracks; tracing-chrome emits the monitor's samples only as
instant events. Following jolt's `scripts/postprocess_trace.py`, but in
Rust inside `ProfilingGuard::finish()` so the artifact is complete the
moment the process exits:

1. read the flushed trace JSON;
2. for every event whose `args` contain keys prefixed `counters.`:
   for each such key, append `{"ph": "C", "ts": <same>, "pid": <same>,
   "tid": <same>, "name": <key minus prefix>, "args": {<name>:
   <value>}}`; drop the original instant event;
3. write atomically (temp file + rename), so a crash mid-rewrite
   cannot destroy a long run's trace (jolt design decision, kept).

The result is one self-contained artifact: aerie stage spans + akita
internal spans + memory/CPU counter tracks, on one timebase, loadable
in Perfetto or `chrome://tracing` with no external scripts.

## 7. CLI and control

One flag on `crates/aerie/benches/falcon.rs`, for either parameter set:

```
cargo bench -p aerie --features falcon-bench,profiling --bench falcon -- 16384 --profile
cargo bench -p aerie --features falcon-bench,profiling --bench falcon -- 16384 --profile --degree=1024
```

- `--profile[=<dir>]`: installs the subscriber + monitor around one run of
  the first requested batch. Trace
  path: `<dir>/falcon-<records>-<unix_secs>-<pid>.json` (the pid
  keeps concurrent same-config runs from clobbering each other),
  printed to stderr on completion. `<dir>` defaults to
  `target/traces/` — inside the already-gitignored, `cargo
  clean`-able target dir (honoring `CARGO_TARGET_DIR`), never the
  bench cwd, which cargo sets to the package root; traces are
  short-lived eyeballing artifacts, and anything worth keeping is
  saved via an explicit `--profile=<dir>`. A bare `--profile=` maps to
  the default. `AERIE_PROFILE_FILTER` overrides the complete profiling
  filter. Ambient `RUST_LOG` is ignored because the profiler requires its
  repository span targets and developer tools may set `RUST_LOG` for their
  own logging;
  `AERIE_MONITOR_INTERVAL_MS` overrides the sampling interval (clamped
  to [50 ms, 10 s]).
- Without the aerie `profiling` cargo feature, `--profile` fails with a
  pointer to the feature.
- Threads are named for Perfetto lanes: rayon
  workers are `aerie-rayon-<n>`, the single-thread verifier pool is
  `aerie-v1-verifier-single`, and the monitor is `aerie-monitor`.
- The benchmark uses a global Rayon pool with 256 MiB worker stacks and a
  512 MiB roundtrip worker stack, whether or not profiling is enabled.
- Profiled timings include tracing and monitor overhead. Use the trace to
  inspect execution shape; do not report its timing row as a benchmark result.
- The benchmark runs verification twice over identical inputs:
  `v1.verify.parallel` on the configured pool first, then
  `v1.verify.single_thread` in a dedicated one-worker pool whose
  construction is outside the timer.
