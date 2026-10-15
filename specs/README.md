# Aerie protocol specifications

[falcon.md](falcon.md) is the standalone normative specification for
public-salt Falcon-512 and Falcon-1024 aggregation. It defines three
batched sumcheck stages, four multi-polynomial Akita commitment groups,
and one batched opening. Both parameter sets use wire version 1
and transcript root `aerie/falcon/v1`; parameter bytes 9 and 10 separate
them.

`crates/aerie-falcon` owns shared protocol math, verifier replay, transcript
and wire grammar. `crates/aerie-prover` implements producer execution for the
three-stage PIOP and four commitment groups; `crates/aerie-verifier` exposes
the verification API. Neither parameter set has passed production acceptance.

## Document set

| Document | Scope |
| --- | --- |
| [Falcon](falcon.md) | Relation, parameters, widths, projection, three stages, wire and acceptance |
| [Projection theorem](falcon-projection-theorem.md) | Self-contained norm and residual projection proof with its exact-rational certificate |
| [PCS extraction](falcon-pcs-extraction.md) | Four groups and the extraction boundary; catalog costs |
| [Falcon-512 ledger](falcon-ledger.md) | Three-stage PIOP and projection errors; four-group PCS accounting |
| [Falcon-1024 ledger](falcon-1024-ledger.md) | Parameter-specific ledger; four-group PCS accounting |
| [Release policy](release-policy.md) | Development compatibility and production acceptance |

## Checks

The [projection theorem](falcon-projection-theorem.md)'s written proof and
exact-rational certificate, checked by the in-repository scripts, form the
self-contained proof under H1 to H4. Nothing in the written proof depends
on Lean. The separate, currently unpublished library certified-projections
provides an additional formalization; it is not published with this
release and this repository's checks do not run it. The constants/text
consistency check does not machine-check its proofs or validate its axiom
audit.

Numerical and consistency checks live in [scripts/](scripts/). Spec CI runs
these eight checks:

| Script | Check |
| --- | --- |
| `falcon_projection_certificate.py --check` | Recomputes the exact projection certificate and compares it with the pinned report |
| `falcon_projection_lean.py --check` | Compares local Rust constants with the certificate and generated Lean constants-file text with its recorded SHA-256; does not run Lean |
| `falcon_projection_wire_bound.py` | Checks Rice-coded projection output size bounds for decodable and accepted proofs at both parameter sets |
| `falcon_composition_ledger.py` | Checks the Falcon-512 three-stage PIOP and projection accounting and reports current four-group PCS terms |
| `falcon_msis_audit.py` | Checks the current four-group PCS catalog report's provenance and structure and counts its SIS matrix occurrences; does not turn the estimator gate into a concrete MSIS advantage |
| `falcon_1024_parameter_checks.py` | Checks Falcon-1024 constants, encoding budget, projection geometry, lifting bounds, four-group shapes and three-stage PIOP arithmetic |
| `falcon_1024_composition_ledger.py` | Checks the Falcon-1024 three-stage PIOP and projection accounting and reports current four-group PCS terms |
| `falcon_transcript_grinding.py --check` | Recomputes the Section 10.1 field-event and nonce allocation for all 29 admitted padded shapes and compares it with the vendored whole-protocol tree evidence |

The whole-protocol evidence is vendored at
[evidence/whole-protocol-tree-checks.json.gz](evidence/whole-protocol-tree-checks.json.gz).
`falcon_transcript_grinding.py` checks the SHA-256 of its decompressed bytes
before comparing the field events and allocations, including the root
`EvaluationBatch` site. This makes the allocation evidence reproducible;
it does not prove the paper's whole-protocol theorem.

`python3 specs/scripts/falcon_slack_width_check.py` exhaustively checks
bounded four-square decompositions above $`8191^2`$ through $`70{,}265{,}242`$.
It is deliberately not part of CI. Use its optional range for a short run.

## Implementation techniques

- [HashToPoint on stable Rust](techniques/hash-to-point-simd.md).
- [Compact range kernels and normalized rounds](techniques/round-zero-integer-paths.md).
- [Profiling](techniques/profiling.md).

These describe the current three-stage implementation and its optimized
prover kernels.

## Scope

The verifier receives public keys and messages and derives HashToPoint
targets from the public salts in the proof. Raw private witnesses are not
sent. Zero knowledge is not required or claimed.
