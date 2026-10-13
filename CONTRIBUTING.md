# Contributing to Aerie

Use a focused pull request for fixes, small improvements, and documentation.
For protocol, transcript, serialization, public API, or crate-boundary changes,
describe the proposed design, invariants, and validation plan before implementation.
Keep the relevant [specifications](specs/README.md) consistent with the code.

## Setup and checks

Install Rust through rustup. [rust-toolchain.toml](rust-toolchain.toml) selects
Rust 1.95 with rustfmt and Clippy. Python 3 runs the specification and repository
script checks without third-party packages. Run commands from the repository root.

```sh
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
RUST_MIN_STACK=536870912 cargo test --workspace --lib --bins --tests \
  --features aerie-falcon/adversarial-internals --profile ci
python3 -m unittest discover -s scripts/tests -p 'test_*.py'
```

Use `cargo test -p <crate>` for focused iteration, for example
`cargo test -p aerie-falcon-native` for canonical Falcon bytes or
`cargo test -p aerie-pcs-config` for PCS presets and catalogs. The larger stack above is
needed by production PCS tests. CI uses cargo-nextest and also checks feature
variants and explicitly ignored large tests; see the exact commands in
[the Rust workflow](.github/workflows/rust.yml).

Run the specification arithmetic checks when changing protocol parameters,
bounds, or their implementation:

```sh
python3 specs/scripts/falcon_projection_certificate.py --check
python3 specs/scripts/falcon_projection_wire_bound.py
python3 specs/scripts/falcon_composition_ledger.py
python3 specs/scripts/falcon_msis_audit.py
python3 specs/scripts/falcon_1024_parameter_checks.py
python3 specs/scripts/falcon_1024_composition_ledger.py
```

[The spec-checks workflow](.github/workflows/spec-checks.yml) owns this check list.

## Benchmarks

The [Falcon harness](crates/aerie/benches/falcon.rs) measures a real-commitment
roundtrip using signatures from the reference FN-DSA signer. For a small batch:

```sh
cargo bench -p aerie --no-default-features --features falcon-bench --bench falcon -- 8
cargo bench -p aerie --no-default-features --features falcon-bench --bench falcon -- 8 --degree=1024
```

The positional argument is the live batch size. Omitting it runs `8 1024 16384`.
For profiling, add `profiling` to the feature list and pass `--profile` after `--`.
Record the machine, parameter set, batch size, features, and fixture source with
performance results. Do not present synthetic relation-positive rows as a
production aggregate-signature benchmark.

## Pull requests

Explain the problem, resulting behavior, and validation. Permanent tests should
assert the current contract against independent vectors or mathematical
properties. Preserve explicit protocol and coverage limitations.

PR titles follow `<type>[optional scope][optional !]: <description>`, for example
`fix(wire): reject a truncated transport frame`. The
[title workflow](.github/workflows/pr-title.yml) accepts `feat`, `fix`, `chore`,
`docs`, `refactor`, `perf`, `test`, `build`, `ci`, `style`, `revert`, `spec`, `specs`,
`bench`, `schedules`, and `deps`; titles beginning with `Revert ` are exempt.

The public facade exports an explicit typed API. Put helpers used only by
benchmarks, fuzz targets, or invariant tests behind `bench-internals` or
`adversarial-internals`. Test proof mutations against the production PCS prover
and verifier. Keep the transcript and wire version unchanged during API-only
refactors.
