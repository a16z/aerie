# Aerie Falcon wire fuzzing

`seeds/wire_pcs_proof` contains deterministic, honestly generated aggregate
proofs at the smallest supported live record count. It covers both Falcon
parameter sets and both `v1p` (per-record salt) and `v1c` (common salt) frames.

Regenerate the corpus from the pinned honest-vector and prover dependencies:

```console
cargo run --locked -p aerie-prover --profile ci \
  --example generate_wire_fuzz_seeds \
  --features internals,honest-vectors -- OUTPUT_DIRECTORY
```

The example requires `OUTPUT_DIRECTORY` not to exist and verifies every proof
before writing it; copy the four files over `seeds/wire_pcs_proof` afterwards.
