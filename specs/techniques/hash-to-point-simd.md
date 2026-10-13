# Use data-parallel Falcon HashToPoint on stable Rust

Aerie derives several independent Falcon `HashToPoint` targets in one Keccak
permutation. One stable Rust 1.95 binary detects the CPU at runtime and selects
the fastest supported backend. No nightly compiler, global target CPU flag, or
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

At 16,384 records this removes the 16 MiB padded `i16` target table and its
later full read. Scratch is one reusable backend-width buffer per outer Rayon
chunk, so memory is bounded independently of the aggregate size. The outer
grain scales with the active worker count, is aligned to the SIMD width, and
is capped at 64 records; it is not tuned to one machine's core count.

The coefficient contraction prepackages the 512 extension-field equality
weights as two `u32` limbs for each of their two base-field components. NEON,
AVX2, and AVX-512 then compute exact `u32 × |i16| -> u64` products. Signed sums
fit in `i128` for Falcon's centered coefficients and are reduced once per
extension component. The packed primitive is protocol-neutral and remains
behind a fixed-size safe plan selected once per target evaluation.
Architecture-specific dot-product operations stay confined to
`crates/aerie-simd`, which selects numeric kernels independently of Keccak's
private permutation dispatch.

## Runtime backend selection

| Runtime target | Backend | Independent SHAKE256 streams |
|---|---|---:|
| x86 or x86-64 without supported AVX | scalar platform fallback | 1 |
| ARMv8 with SHA3 instructions | `ARMv8` SHA3, selected at runtime | 2 |
| x86-64 with AVX2 | AVX2, selected at runtime | 4 |
| x86-64 with AVX-512F | AVX-512, selected at runtime | 8 |

The default order on x86-64 is AVX-512, AVX2, then scalar. Runtime feature
checks occur before a target-specific function can execute. The resulting
binary remains portable across x86-64 machines.

Target contraction independently selects the widest runtime-safe packed AVX2,
AVX-512, or NEON plan. Targets without one of those vector kernels keep the
field's delayed accumulator, which is faster than scalar `i128` limb
reconstruction.

The AVX2 and AVX-512 permutation kernels use stable `std::arch::x86_64`
intrinsics. HashToPoint rejection sampling additionally uses ordered SIMD
compaction: mandatory NEON on AArch64, AVX2 on x86-64, and AVX-512BW with
AVX-512VBMI2 masked compression when both extensions are available. An
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
boundaries, unequal message lengths, partial batches, sparse rejection masks,
and a deterministic target that requires a ninth squeeze block. The SIMD suite
checks platform, AVX2, and AVX-512 dot plans whenever the host supports them.
The final test compares streamed and expanded target evaluation at a
non-power-of-two live count in one-, two-, and four-worker Rayon pools.

No `RUSTFLAGS`, `target-cpu`, or toolchain override is necessary.

## Pin a backend for testing

Normally, leave backend selection automatic. To compare implementations on an
x86-64 test machine, set `AERIE_KECCAK_BACKEND` to `platform`, `avx2`, or
`avx512`:

```console
AERIE_KECCAK_BACKEND=platform \
  cargo test --release --locked -p aerie-falcon-native --all-features

AERIE_KECCAK_BACKEND=avx2 \
  cargo test --release --locked -p aerie-falcon-native --all-features

AERIE_KECCAK_BACKEND=avx512 \
  cargo test --release --locked -p aerie-falcon-native --all-features
```

On x86-64, `platform` selects the scalar fallback. A forced AVX backend exits
before running the kernel when the CPU lacks the requested feature. These
overrides are intended for tests and benchmarks, not deployment.

## Reproduce the HashToPoint microbenchmark

The focused benchmark uses canonical public-key bytes plus synthetic salts and
messages. It measures `HashToPoint` preprocessing only. It does not generate or
verify signatures and is not an aggregate-proof benchmark. Set
`MODE=verifier` to measure the composition verifier transport uses: parallel
hashed-key derivation, then streamed prehashed targets over the canonical
Rayon grain.

Run 16,384 targets on one worker with automatic runtime selection:

```console
RECORDS=16384 SAMPLES=11 THREADS=1 \
  cargo bench --locked -p aerie-falcon-native --features parallel --bench hash-to-point
```

Prefix the command with one of the `AERIE_KECCAK_BACKEND` settings above to
compare backends in separate processes. The output reports the selected
backend, stream width, median milliseconds, and speedup. `THREADS=1` confirms
that any SIMD improvement comes from data parallelism rather than concurrent
workers.

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
