# Use data-parallel Falcon HashToPoint on stable Rust

Aerie derives several independent Falcon `HashToPoint` targets in one Keccak
permutation. One stable Rust 1.95 binary detects the CPU at runtime and selects
the widest supported backend. No nightly compiler, global target CPU flag, or
machine-specific binary is required.

This changes scheduling only. It does not change the SHAKE256 transcript,
Falcon framing, or per-record public inputs.

## Stream targets into the ring evaluation

Public-record binding needs each fixed-width `hpk` before block `L`, but the
ring scalar does not need a target until block `L` has sampled $`r_L`$. The wire
verifier therefore performs these operations at their dependency boundaries:

1. derive all `hpk` values with the active Rayon pool and Keccak backend;
2. bind `hpk`, salt, and message in the public-record digest;
3. retain one aligned borrowed batch view over the `hpk`, salt, and message
   collections through block `L`;
4. sample at most one backend-width group of target rows;
5. contract every completed row into $`t_L`$ and discard it immediately.

At 16,384 records this removes the 16 MiB padded `i16` target table for
Falcon-512 (32 MiB for Falcon-1024) and its later full read. Scratch is one
reusable backend-width buffer per outer Rayon chunk, so scratch memory is
bounded independently of the aggregate size. The outer grain scales with the active worker count, is aligned to the SIMD width, and
is capped at 64 records; it is not tuned to one machine's core count.

The coefficient contraction prepackages `n=512` or `n=1024` extension-field
equality weights. Each base-field component is represented by two signed
`i32` digits, `low + 2^32 high`, congruent to its canonical residue modulo
`p = 2^64 - 59`. The representative is the residue itself when it fits the
digit pair's upper bound `2^63 - 2^31 - 1`, and the residue minus `p`
otherwise. Four digit streams cover the two base-field components.

NEON, AVX2, and AVX-512 compute exact signed `i32 × i16` products and bounded
signed sums. Each product has magnitude at most `2^46`; the primitive allows
at most `2^16` terms, so each `i64` accumulator lane has magnitude at most
`2^62`. Reconstruction as `sum_low + 2^32 sum_high` is exact in `i128` and
has magnitude below `2^95`. The protocol reduces once per base-field
component. The packed primitive is protocol-neutral and remains behind a
fixed-size safe plan selected once per target evaluation. Architecture-specific
dot-product operations stay confined to `crates/aerie-simd`, which selects
numeric kernels independently of Keccak's private permutation dispatch.

## Runtime backend selection

| Runtime target | Backend | Independent SHAKE256 streams |
|---|---|---:|
| x86 or x86-64 without supported AVX | scalar platform fallback | 1 |
| ARMv8 with SHA3 instructions | `ARMv8` SHA3, selected at runtime | 2 |
| Apple AArch64 with SHA3 instructions | hybrid: two SHA3 vector states and one scalar state | 3 |
| x86-64 with AVX2 | AVX2, selected at runtime | 4 |
| x86-64 with AVX-512F | AVX-512, selected at runtime | 8 |

The default order on x86-64 is AVX-512, AVX2, then scalar. On Apple AArch64
cores the hybrid kernel is the default because the two-state SHA3 kernel is
limited by vector issue and leaves the integer pipes idle; other AArch64
cores keep the two-state kernel unless the hybrid is requested. Runtime feature
checks occur before a target-specific function can execute. The resulting
binary remains portable across x86-64 machines.

Target contraction independently selects the widest runtime-safe packed AVX2,
AVX-512, or NEON plan. Targets without one of those vector kernels keep the
field's delayed accumulator.

The AVX2 and AVX-512 permutation kernels use stable `std::arch::x86_64`
intrinsics. HashToPoint rejection sampling additionally uses ordered SIMD
compaction: mandatory NEON on AArch64, AVX2 on x86-64, and AVX-512BW with
AVX-512VBMI2 masked compression when both extensions and AVX2 are available. An
AVX-512F permutation backend falls back to AVX2 or scalar sampling if the CPU
lacks those sampling extensions.

Architecture-specific unsafe operations are isolated in `crates/aerie-keccak`
for permutation and compaction kernels and `crates/aerie-simd` for reusable
numeric kernels; the protocol-facing `aerie-falcon-native` crate continues to forbid unsafe
code. The sampling API preserves candidate order while fusing
big-endian decoding, rejection below `5q`, and centered reduction modulo the
compile-time modulus.

## Build and test

Use the repository's pinned stable toolchain for every backend:

```console
cargo test --release --locked -p aerie-keccak
cargo test --release --locked -p aerie-simd
cargo test --release --locked -p aerie-falcon-native --all-features
cargo test --release --locked -p aerie-falcon packed_small_dot
cargo test --release --locked -p aerie-falcon streamed_target_matches_expanded
```

The first command compares every available x86 SIMD permutation against the
scalar permutation. The Falcon suite additionally compares complete batched
`HashToPoint` outputs with the public scalar API across SHAKE256 rate
boundaries, unequal message lengths, partial batches, and sparse rejection
masks. A separate sampler test feeds synthetic squeeze blocks with interspersed
rejections and checks that sampling continues into the extra blocks. The SIMD suite
checks the scalar dot plan and NEON, AVX2, and AVX-512 plans whenever the
host supports them.
The final test compares streamed and expanded target evaluation at a
non-power-of-two live count in one-, two-, and four-worker Rayon pools.

No `RUSTFLAGS`, `target-cpu`, or toolchain override is necessary.

## Pin a backend for testing

Normally, leave backend selection automatic. The two override interfaces are
independent and effective only when the corresponding crate's
`backend-override` feature is enabled. Builds without that feature ignore the
corresponding environment variable.

- Keccak accepts `AERIE_KECCAK_BACKEND=auto|platform|hybrid|avx2|avx512` with
  `aerie-keccak/backend-override`. `auto` uses runtime detection. `platform`
  uses RustCrypto dispatch: scalar on x86-64, potentially ARM SHA3 on
  AArch64. Forced `hybrid` requires AArch64 with SHA3. Forced `avx2` and
  `avx512` require x86-64 and the requested CPU features (AVX2 and AVX-512F,
  respectively); unsupported requests panic before a vector kernel executes.
- Numeric kernels accept `AERIE_SIMD_BACKEND=auto|scalar|neon|avx2|avx512`
  with `aerie-simd/backend-override`. `auto` uses the widest detected ISA.
  A supported request can lower the process-wide CPU ceiling; each kernel
  family selects its widest available implementation at or below that
  ceiling, checking its required ISA subsets. Unsupported requests panic.
  A scalar ceiling makes target contraction use the field's delayed
  accumulator rather than a packed vector dot plan.

For example, pin Keccak's platform backend or numeric kernels' scalar ceiling
in separate test processes:

```console
AERIE_KECCAK_BACKEND=platform \
  cargo test --release --locked -p aerie-keccak --features backend-override

AERIE_SIMD_BACKEND=scalar \
  cargo test --release --locked -p aerie-simd --features backend-override
```

These overrides are intended for tests and benchmarks. An arbitrary production
build does not honor them unless it enables the corresponding feature.

## Reproduce the HashToPoint microbenchmark

The focused Falcon-512 benchmark uses canonical public-key bytes plus
synthetic salts and messages. It measures `HashToPoint` preprocessing only.
It does not generate or verify signatures and is not an aggregate-proof
benchmark. Set
`MODE=verifier` to measure the composition verifier transport uses: parallel
hashed-key derivation, then streamed prehashed targets over the canonical
Rayon grain. Here, prehashed means public-key `hpk` derivation; messages
remain in raw-message mode with the fixed empty context.

Run 16,384 targets on one worker with automatic runtime selection:

```console
RECORDS=16384 SAMPLES=11 THREADS=1 \
  cargo bench --locked -p aerie-falcon-native --features parallel --bench hash-to-point
```

Prefix the command with one of the `AERIE_KECCAK_BACKEND` settings above and
add `aerie-keccak/backend-override` to `--features` to compare backends in
separate processes. The output reports the selected
backend, stream width, and median milliseconds for each measured path.
`THREADS=1` holds the worker count fixed when comparing backends.

## Timing results

No timing results are reported here; run the benchmark above to measure the
current implementation.

## How batching composes with threads

Each SIMD lane carries a complete, independent SHAKE256 state. This is
single-threaded data parallelism; it does not split one transcript between
lanes. Falcon owns the single canonical Rayon grain derived from the live record count, the
active worker count, and the selected backend width. The grain is aligned to
that SIMD width and capped at 64 records, so production-sized batches retain
the cross-machine stable cap while small batches avoid oversized chunks. The
final incomplete SIMD group uses the scalar public path.

The compactor accepts a caller-bounded output. Full-capacity blocks take a
branch-free SIMD path. The rare final block compacts into one fixed stack
buffer and copies only the remaining accepted coefficients, so the Falcon
caller needs no separate scratch allocation or partial-copy branch.

Messages may have different lengths. Contiguous and segmented inputs share one
cursor-driven absorb scheduler. Common full SHAKE256 blocks are permuted
together. Additional full blocks for a longer message use the backend's scalar
permutation before all lanes resume together for padding and squeezing.
