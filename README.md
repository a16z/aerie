# Aerie

Aerie aggregates Falcon-512 or Falcon-1024 signatures into one succinct
proof. It is built on the [Akita](https://eprint.iacr.org/2026/1983)
lattice polynomial commitment scheme.

Each signature in a batch keeps its own public key, salt and message. One
proof covers one parameter set; Falcon-512 and Falcon-1024 signatures cannot
be mixed.

> **Status:** pre-release. The version 1 wire format may change before the
> first public release.

## What a proof shows

An accepting proof shows that, for every record, a signature exists that:

- satisfies the Falcon verification equation for that record's public key,
  salt and message;
- meets the Falcon norm bound;
- fits Falcon's canonical compressed signature encoding.

The verifier is given the public keys and messages, and the proof carries the
40-byte salts. The verifier computes `HashToPoint` itself; hashing is not
proved inside the proof. The full statement is in
[`specs/falcon.md`](specs/falcon.md).

Signatures are never sent to the verifier. Zero knowledge is not claimed.

Aerie follows Falcon semantics. If the final FIPS 206 (FN-DSA) encoding or
verification rules differ, FN-DSA will need its own protocol identity.

## Build and test

Aerie requires Rust 1.95.

```sh
RUST_MIN_STACK=536870912 cargo test --workspace --locked --profile ci
```

[CONTRIBUTING.md](CONTRIBUTING.md) lists the full checks.

## Usage

The API is generic over the parameter set `P`, either
`aerie::falcon::Falcon512` or `aerie::falcon::Falcon1024`. Values built for one
parameter set do not type-check against the other.

This example uses the `aerie` facade:

```rust
use aerie::falcon::{
    prove_aggregate, verify_aggregate, CanonicalPublicKeyBytes,
    CanonicalSignatureBytes, Falcon512, FalconAggregateSetup,
};
use aerie::{AggregationRecord, AggregationStatement, AggregationWitness};

type P = Falcon512; // or aerie::falcon::Falcon1024

/// Each record is (encoded public key, padded signature, message) for a
/// signature that verifies under standard Falcon with an empty context.
fn aggregate(
    records: &[(Vec<u8>, Vec<u8>, Vec<u8>)],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut public_keys = Vec::new();
    let mut messages = Vec::new();
    let mut statement_records = Vec::new();
    let mut signatures = Vec::new();
    for (public_key, signature, message) in records {
        let public_key = CanonicalPublicKeyBytes::<P>::parse(public_key)?;
        let record = AggregationRecord::new(public_key.clone(), message.clone())?;
        statement_records.push(record);
        signatures.push(CanonicalSignatureBytes::<P>::parse(signature)?);
        public_keys.push(public_key);
        messages.push(message.clone());
    }
    let statement = AggregationStatement::new(statement_records)?;
    let witness = AggregationWitness::new(signatures)?;

    // Depends only on the record count; build once and reuse per batch size.
    let setup = FalconAggregateSetup::<P>::new(records.len())?;
    let (proof, _metrics) = prove_aggregate(&setup, &statement, &witness)?;

    // The verifier sees only the public keys, the messages and the proof.
    verify_aggregate(setup.verifier(), &public_keys, &messages, &proof)?;
    Ok(proof)
}
```

The prover needs a larger thread stack than Rust's default; set
`RUST_MIN_STACK=536870912`. A verifier that does not prove can build
`FalconVerifierSetup::<P>::new(records)` directly.

## Batch sizes

A proof covers 1 to 65,536 records. Batches are padded to a power of two, with
a minimum of 8 rows for Falcon-512 and 4 for Falcon-1024, so smaller batches
cost the same as the minimum.

A one-record proof is about 64 KB and grows slowly with the batch. Counting
only the 40-byte salt per extra record, a proof is smaller than the signatures
it replaces from about 103 Falcon-512 or 52 Falcon-1024 signatures.

## Benchmarks

```sh
# Falcon-512
cargo bench --locked -p aerie --features falcon-bench --bench falcon -- 16384
# Falcon-1024
cargo bench --locked -p aerie --features falcon-bench --bench falcon -- 16384 --degree=1024
```

The benchmark proves real Falcon signatures made by the reference `fn-dsa`
signer from test keys. On pull requests that touch the prover, the
[Falcon Benchmarks workflow](.github/workflows/falcon-bench.yml) posts head and
base timings as a comment. For traces, see
[`specs/techniques/profiling.md`](specs/techniques/profiling.md).

## Security

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
production release.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `crates/aerie` | Public API (`aerie::falcon`) and benchmarks |
| `crates/aerie-falcon` | The aggregation protocol |
| `crates/aerie-falcon-native` | Falcon encodings, `HashToPoint` and native verification |
| `crates/aerie-falcon-statement` | Statements, witnesses and fixtures |
| `crates/aerie-pcs-config` | Akita commitment presets and fold-schedule catalogs |
| `crates/aerie-keccak`, `crates/aerie-simd` | Keccak backends and SIMD kernels |
| `crates/aerie-invariants` | Cross-crate and adversarial tests |
| `crates/aerie-profiling` | Optional tracing |

## Documentation

Start with the [specification index](specs/README.md).

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
same terms.
