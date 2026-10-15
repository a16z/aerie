# Aerie

Aerie aggregates Falcon-512 or Falcon-1024 signatures into one succinct
proof. It is built on the [Akita](https://eprint.iacr.org/2026/1983)
lattice polynomial commitment scheme.

This implementation selects Akita's `dev-protocol` feature at
[`e023b6a46143f7b23199ccecb2684192baf7b640`](https://github.com/LayerZero-Labs/akita/commit/e023b6a46143f7b23199ccecb2684192baf7b640),
the full revision pinned in [Cargo.toml](Cargo.toml). The feature name is not a
compatibility guarantee.

Each signature in a batch keeps its own public key, salt and message. One
proof covers one parameter set; Falcon-512 and Falcon-1024 signatures cannot
be mixed.

The paper, [*Aerie: Efficient Aggregation of Falcon Signatures*](paper/aerie.pdf),
describes the protocol, proves its security and reports the measurements
below.

> **Status:** research release. Wire version 1 is frozen: incompatible
> transcript or encoding changes require a new wire version. Proof interchange
> requires the same pinned implementation and profile. Neither parameter set
> has passed the production acceptance criteria of the
> [release policy](specs/release-policy.md).

## What a proof shows

An accepting proof shows that, for every record, a signature exists that:

- satisfies the Falcon verification equation for that record's public key,
  salt and message;
- meets the Falcon norm bound;
- fits Falcon's canonical compressed signature encoding.

The verifier is given the public keys and messages. The default
`prove_aggregate` produces per-record 40-byte salts. The wire format and verifier
also support common-salt mode: all signatures use the same salt, and the proof
carries one 40-byte salt expanded per record for binding. The verifier computes
`HashToPoint` itself; hashing is not proved inside the proof. The full statement is in
[`specs/falcon.md`](specs/falcon.md).

Signatures are never sent to the verifier. Witness omission does not establish
zero knowledge; zero knowledge is not claimed.

Aerie supports Falcon v1.2 padded signature format with key-prefixed message
framing (raw mode, empty context), as implemented by fn-dsa 0.3.0. The
`HashToPoint` input is `salt || SHAKE256(pk,64) || 0x00 || 0x00 || message`.
Original `r || m` framing is unsupported. This is not a compatibility claim
for all v1.2 signers or final FIPS 206 (FN-DSA).

## Signers

The two salt modes require different signers. With per-record salts, each
signature comes from a signer that draws a fresh salt on every signing
attempt, as FN-DSA and the `fn-dsa` 0.3.0 signer do. Fouque et al. reduce the
unforgeability of this signer to a lattice assumption in the random-oracle
model ([ePrint 2024/1769](https://eprint.iacr.org/2024/1769)).

In common-salt mode the signer receives the salt and, as Falcon v1.2 does,
retries the norm and encoding checks at that fixed target. It must return the
same signature whenever it is asked again to sign the same salt and message,
either from a cache or by deriving its sampler randomness from a dedicated
secret key, the salt and the message
([`specs/falcon.md`](specs/falcon.md), Section 10.2). Unforgeability of this
fixed-target signer is an assumption of common-salt mode. No reduction to a
standard lattice assumption is known, and no attack is known.

Aerie includes no common-salt signer. The `honest-vectors` fixtures in
`aerie-falcon-witness` replay one salt to `fn-dsa` from a seeded generator to
build test and benchmark batches; their keys are not production key material.

## Build and test

Aerie requires Rust 1.95.

```sh
cargo test --workspace --locked --profile ci
```

[CONTRIBUTING.md](CONTRIBUTING.md) lists the full checks.

## Usage

The API is generic over the parameter set `P`, either
`aerie::falcon::Falcon512` or `aerie::falcon::Falcon1024`. Values built for one
parameter set do not type-check against the other.

This example uses the `aerie` facade:

```rust
# #[cfg(feature = "prover")]
# fn main() {
use aerie::falcon::{
    prove_aggregate, verify_aggregate, CanonicalPublicKeyBytes,
    CanonicalSignatureBytes, Falcon512, FalconAggregateSetup,
};
use aerie::falcon::{AggregationRecord, AggregationStatement, AggregationWitness};

type P = Falcon512; // or aerie::falcon::Falcon1024

/// Each record is (encoded public key, padded signature, message) for a
/// signature that verifies under Falcon v1.2 padded signature format with
/// key-prefixed message framing (raw mode, empty context), as implemented by
/// fn-dsa 0.3.0. Original `r || m` framing is unsupported; compatibility with
/// all v1.2 signers or final FIPS 206 is not claimed.
fn aggregate(
    records: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut statement_records = Vec::new();
    let mut signatures = Vec::new();
    for (public_key, signature, message) in records {
        let public_key = CanonicalPublicKeyBytes::<P>::parse(public_key)?;
        let record = AggregationRecord::new(public_key, message.clone())?;
        statement_records.push(record);
        signatures.push(CanonicalSignatureBytes::<P>::parse(signature)?);
    }
    let statement = AggregationStatement::new(statement_records)?;
    let witness = AggregationWitness::new(signatures)?;

    // Depends only on the record count; build once and reuse per batch size.
    let setup = FalconAggregateSetup::<P>::new(records.len())?;
    let proof = prove_aggregate(&setup, &statement, &witness)?;

    // The verifier sees the public statement and the proof.
    verify_aggregate(setup.verifier(), &statement, &proof)?;
    Ok(proof)
}
# }
# #[cfg(not(feature = "prover"))]
# fn main() {}
```

The default facade enables `prover` and `parallel`.

`prove_aggregate` uses per-record salts. To prove a batch in which every
signature uses the same salt, call
`prove_aggregate_with_mode(&setup, &statement, &witness, SaltMode::Common)`;
the verifier reads the mode from the proof. The `aerie::falcon` facade is the
supported public API. The `internals` features of `aerie-prover` and
`aerie-falcon` exist for the benchmark harness and adversarial tests and are
not a stable API.

### Verifier-only builds

For verification only, use `aerie` with
`default-features = false, features = ["verifier"]`, or depend on
`aerie-verifier` directly with default features disabled. Both exclude proving,
witness construction, the Falcon signer, producer Akita crates, and benchmark
and profiling dependencies; scalar builds also exclude Rayon. Construct
`FalconVerifierSetup::<P>::new(records)` and call `verify_aggregate` through
`aerie::falcon` or `aerie_verifier`. Add `parallel` for parallel verification.

## Batch sizes

A proof covers 1 to 65,536 records. Batches are padded to a power of two, with
a minimum of 8 rows for Falcon-512 and 4 for Falcon-1024, so smaller batches
cost the same as the minimum.

Total wire size depends on the padded batch, parameter set and salt mode. For
comparison, a padded Falcon-512 signature is 666 bytes and a padded
Falcon-1024 signature is 1280 bytes. The benchmark below prints the measured
PCS proof bytes and full wire bytes for each batch.

## Benchmarks

```sh
cargo bench --locked -p aerie-bench --features falcon-bench --bench falcon -- 8
```

Append `--degree=1024` to select Falcon-1024 or `--common-salt` to benchmark
common-salt mode. The benchmark proves real Falcon signatures made by the
`fn-dsa` 0.3.0 signer from test keys. On pull requests that touch the prover, the
[Falcon Benchmarks workflow](.github/workflows/falcon-bench.yml) posts head
timings as a comment, with a base comparison when the base provides the same
benchmark schema and package. For traces, see
[`specs/techniques/profiling.md`](specs/techniques/profiling.md).

### Results

Measured on this release's code on an Apple M4 Max (12 performance and 4 efficiency
cores, 64 GiB, macOS 27.0.1) with Rust 1.95.0, using honestly generated
batches with one salt per signature. Each timing is the median of seven runs;
parallel runs use 16 worker threads. Sizes include salts and headers but not
the verifier-supplied keys and messages. The common-salt size keeps one salt
in place of the per-record salts. Peak memory is the benchmark process's
maximum resident set size. One KiB is 1024 bytes.

Falcon-512:

| Signatures | Prove, 16 threads | Verify, 1 thread | Verify, 16 threads | Aggregate | Common-salt aggregate | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 132 ms | 5.5 ms | 4.6 ms | 102 KiB | 62 KiB | 198 MiB |
| 2,048 | 172 ms | 7.5 ms | 5.2 ms | 143 KiB | 63 KiB | 280 MiB |
| 4,096 | 207 ms | 11.1 ms | 6.0 ms | 223 KiB | 63 KiB | 418 MiB |
| 8,192 | 263 ms | 16.9 ms | 6.7 ms | 385 KiB | 65 KiB | 658 MiB |
| 16,384 | 386 ms | 29.0 ms | 8.2 ms | 705 KiB | 65 KiB | 1,126 MiB |
| 32,768 | 645 ms | 53.6 ms | 11.6 ms | 1,346 KiB | 66 KiB | 2,113 MiB |
| 65,536 | 1,071 ms | 101.2 ms | 15.8 ms | 2,627 KiB | 67 KiB | 4,127 MiB |

Falcon-1024:

| Signatures | Prove, 16 threads | Verify, 1 thread | Verify, 16 threads | Aggregate | Common-salt aggregate | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 169 ms | 7.2 ms | 5.2 ms | 103 KiB | 63 KiB | 286 MiB |
| 2,048 | 202 ms | 10.2 ms | 5.6 ms | 143 KiB | 63 KiB | 411 MiB |
| 4,096 | 263 ms | 16.1 ms | 6.3 ms | 225 KiB | 65 KiB | 658 MiB |
| 8,192 | 384 ms | 26.9 ms | 7.9 ms | 385 KiB | 65 KiB | 1,126 MiB |
| 16,384 | 633 ms | 49.4 ms | 11.1 ms | 706 KiB | 66 KiB | 2,106 MiB |
| 32,768 | 1,070 ms | 92.1 ms | 15.3 ms | 1,347 KiB | 67 KiB | 4,108 MiB |
| 65,536 | 1,861 ms | 177.0 ms | 23.8 ms | 2,627 KiB | 67 KiB | 7,947 MiB |

For comparison, checking 32,768 Falcon-512 signatures one by one on one core
of the same machine takes 519 ms with PQClean's aarch64 implementation
([`0586a824`](https://github.com/PQClean/PQClean/tree/0586a824fc0d49df0b6b6e9179d8d15d06d0974f))
and 536 ms with `fn-dsa` 0.3.0. Aerie's one-thread verification is about 10
times faster than PQClean. Proving the same batch on one thread takes 4.3 s,
about 8 times as long as checking the signatures with PQClean.

## Security

See [SECURITY.md](SECURITY.md) for the security policy and vulnerability reporting.

The protocol and its security argument are specified in `specs/`:

- [`falcon.md`](specs/falcon.md): the protocol;
- [`falcon-projection-theorem.md`](specs/falcon-projection-theorem.md): the
  projection soundness theorem;
- [`falcon-pcs-extraction.md`](specs/falcon-pcs-extraction.md): commitment
  extraction;
- [`falcon-ledger.md`](specs/falcon-ledger.md) and
  [`falcon-1024-ledger.md`](specs/falcon-1024-ledger.md): the soundness error
  for each parameter set.

The [release policy](specs/release-policy.md) sets the criteria for a
production release, and Section 12.2 of [`falcon.md`](specs/falcon.md) lists
the open obligations.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `crates/aerie` | Public API (`aerie::falcon`) |
| `crates/aerie-bench` | Benchmark harness, build provenance and optional profiling (unpublished) |
| `crates/aerie-falcon` | Verifier-safe common protocol support, public math, transcript and wire codec |
| `crates/aerie-verifier` | Verification API backed by common replay |
| `crates/aerie-prover` | Aggregate proving, setup, forward projection and commitment execution |
| `crates/aerie-falcon-native` | Falcon encodings and public `HashToPoint` preprocessing |
| `crates/aerie-falcon-statement` | Public statements, metadata, shared errors and checked framing |
| `crates/aerie-falcon-witness` | Validated native arithmetic, witness preparation and fixtures |
| `crates/aerie-pcs-config` | Akita commitment presets and fold-schedule catalogs |
| `crates/aerie-keccak`, `crates/aerie-simd` | Keccak backends and SIMD kernels |
| `crates/aerie-invariants` | Cross-crate and adversarial tests (unpublished) |
| `crates/aerie-profiling` | Optional tracing (unpublished) |

## Documentation

Start with the [specification index](specs/README.md). The
[paper](paper/aerie.pdf) gives the security proofs and the evaluation.

## Related work

- Bolboceanu, Bootle, Lyubashevsky, Merino-Gallardo, and Seiler, *Orthus:
  Practical Sublinear Batch-Verification of Lattice Relations from Standard
  Assumptions*, [ePrint 2026/398](https://eprint.iacr.org/2026/398).
- Aardal et al., *Aggregating Falcon Signatures with LaBRADOR*,
  [ePrint 2024/311](https://eprint.iacr.org/2024/311).
- Lyubashevsky et al., *The LaZer Library: Lattice-Based Zero Knowledge and
  Succinct Proofs for Quantum-Safe Privacy*,
  [ePrint 2024/1846](https://eprint.iacr.org/2024/1846).
- Beullens and Seiler, *LaBRADOR: Compact Proofs for R1CS from Module-SIS*,
  [ePrint 2022/1341](https://eprint.iacr.org/2022/1341).
- Fouque et al., *Falcon: Fast-Fourier Lattice-based Compact Signatures over
  NTRU*, [falcon-sign.info](https://falcon-sign.info/).
- Fouque, Gajland, de Groote, Janneck, and Kiltz, *A Closer Look at Falcon*,
  [ePrint 2024/1769](https://eprint.iacr.org/2024/1769).
- Lazar and Peikert, *Deterministic Falcon Signatures*,
  [algorand/falcon](https://github.com/algorand/falcon/blob/a24293c8/falcon-det.pdf).
- Dao, Bodaghi, Khajehpour, Vitto, Badakhshan, Georghiades, Liu, Zhang, and
  Thaler, *Akita: A High-Performance Lattice-Based Polynomial Commitment
  Scheme*, [ePrint 2026/1983](https://eprint.iacr.org/2026/1983).
- [LayerZero-Labs/akita](https://github.com/LayerZero-Labs/akita), the pinned
  lattice multilinear PCS dependency.

## Acknowledgements

Aerie is built on [Akita](https://github.com/LayerZero-Labs/akita), developed
at LayerZero Labs. Part of Quang Dao's work on Aerie was done at LayerZero
Labs, which also provided equipment and funded AI model subscriptions and API
usage. Part of Alberto Centelles's work on Aerie was done at a16z crypto
research, which also funded AI model usage.

We thank the Akita maintainers, Omid Bodaghi, Amirhossein Khajehpour, Taghi
Badakhshan and Giuseppe Vitto, for ongoing discussions about Aerie and for
designing, implementing or reviewing the Akita changes it depends on. These
include:

- committing groups of polynomials with different sizes and coefficient
  bounds in one batch
  ([30505610](https://github.com/LayerZero-Labs/akita/commit/30505610b),
  [2322d485](https://github.com/LayerZero-Labs/akita/commit/2322d485b));
- one prover backend that holds the setup and serves several parameter sets
  ([#36](https://github.com/LayerZero-Labs/akita/pull/36),
  [#76](https://github.com/LayerZero-Labs/akita/pull/76));
- the Spongefish proof stream that Aerie's opening proof uses
  ([#37](https://github.com/LayerZero-Labs/akita/pull/37));
- schedule planning from declared coefficient bounds, and the prepared
  verifier ([#40](https://github.com/LayerZero-Labs/akita/pull/40),
  [#41](https://github.com/LayerZero-Labs/akita/pull/41),
  [#85](https://github.com/LayerZero-Labs/akita/pull/85)).

## License

Aerie is dual-licensed under the [MIT License](LICENSE-MIT) and the
[Apache License 2.0](LICENSE-APACHE), at your option. Unless stated otherwise,
contributions intentionally submitted for inclusion are dual-licensed on the
same terms. The paper in [`paper/`](paper/) is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
