# Aerie Falcon wire fuzzing

`seeds/wire_pcs_proof` contains deterministic, honestly generated aggregate
proofs at the smallest supported live record count. It covers both Falcon
parameter sets and both `v2p` (per-record salt) and `v2c` (common salt) frames.

Regenerate the corpus from the pinned honest-vector and prover dependencies:

```console
RUST_MIN_STACK=16777216 cargo test -p aerie-falcon generate_wire_fuzz_seeds \
  --profile ci -- --ignored --nocapture
```

The ignored test verifies every generated proof before replacing these files.

Version-1 fixtures are retained to test rejection of the previous hash domains.
