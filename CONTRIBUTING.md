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
cargo test --workspace --lib --bins --tests \
  --features aerie-prover/internals --profile ci
```

Use `cargo test -p <crate>` for focused iteration, for example
`cargo test -p aerie-falcon-native` for canonical Falcon bytes or
`cargo test -p aerie-falcon-witness` for native verification and witness construction, or
`cargo test -p aerie-pcs-config` for PCS presets and catalogs. Use
`cargo test -p aerie-prover --features internals` for producer-backed protocol
tests and `cargo test -p aerie-falcon` for common protocol and verifier math. CI uses
cargo-nextest and also checks feature variants and explicitly ignored large tests; see the exact commands in
[the Rust workflow](.github/workflows/rust.yml).

Run the specification and repository script checks when changing protocol
parameters, bounds, their implementation, or repository scripts:

```sh
python3 specs/scripts/falcon_projection_certificate.py --check
python3 specs/scripts/falcon_projection_lean.py --check
python3 specs/scripts/falcon_projection_wire_bound.py
python3 specs/scripts/falcon_composition_ledger.py
python3 specs/scripts/falcon_msis_audit.py
python3 specs/scripts/falcon_1024_parameter_checks.py
python3 specs/scripts/falcon_1024_composition_ledger.py
python3 specs/scripts/falcon_transcript_grinding.py --check
python3 -m unittest discover -s scripts/tests -p "test_*.py"
test -x scripts/check_serial_deps.sh
bash -n scripts/check_serial_deps.sh
```

[The spec-checks workflow](.github/workflows/spec-checks.yml) owns this check list.
Run `scripts/check_verifier_deps.sh` and `scripts/check_verifier_consumer.sh` to check verifier dependency isolation and standalone use.

## Code conventions

Comments state invariants and reasons, not history.
Permanent tests assert the current contract against independent ground truth or
properties, and a superseded implementation is not kept as an oracle.
Malformed verifier inputs are validated before arithmetic, indexing or allocation
that could panic.
Unsafe code documents the feature, length and alignment invariants that make it
valid.

## Benchmarks

The [Falcon harness](crates/aerie-bench/benches/falcon.rs) measures a real-commitment
roundtrip using signatures from the `fn-dsa` 0.3.0 signer. For a small batch:

```sh
cargo bench --locked -p aerie-bench --features falcon-bench --bench falcon -- 8
```

The positional argument is the live batch size. Omitting it runs `8 1024 16384`.
Append `--degree=1024` to select Falcon-1024 or `--common-salt` for common-salt mode.
For profiling, add `profiling` to the feature list and pass `--profile` after `--`.
Record the machine, parameter set, batch size, features, and fixture source with
performance results. Do not present synthetic relation-positive rows as a
production aggregate-signature benchmark.

## Pull requests

Explain the problem, resulting behavior, and validation. Preserve explicit
protocol and coverage limitations.

PR titles follow `<type>[optional scope][optional !]: <description>`, for example
`fix(wire): reject a truncated transport frame`. The
[title workflow](.github/workflows/pr-title.yml) accepts `feat`, `fix`, `chore`,
`docs`, `refactor`, `perf`, `test`, `build`, `ci`, `style`, `revert`, `spec`, `specs`,
`bench`, `schedules`, and `deps`; titles beginning with `Revert ` are exempt.

The public facade exports an explicit typed API. Put helpers used only by
benchmarks, fuzz targets, or invariant tests behind the owning role
capability: `aerie-prover/internals` for producer execution and
`aerie-falcon/internals` for common protocol support. These capabilities
expose unsupported internal API; the `aerie` facade provides no access to it. Test proof mutations against the
production PCS prover and verifier. Keep the transcript and wire version
unchanged during API-only refactors.
