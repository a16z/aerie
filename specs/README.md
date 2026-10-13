# Aerie protocol specifications

[falcon.md](falcon.md) is the standalone normative specification for
public-salt Falcon-512 and Falcon-1024 aggregation. It defines three
batched sumcheck stages, four multi-polynomial Akita commitment groups,
and one batched opening. Both parameter sets use development wire version
1 and transcript root `aerie/falcon/v1`; parameter bytes 9 and 10 separate
them.

`crates/aerie-falcon` implements the three-stage PIOP and four commitment
groups specified here. Neither parameter set has passed production acceptance.

## Document set

| Document | Scope |
| --- | --- |
| [Falcon](falcon.md) | Relation, parameters, widths, projection, three stages, wire and acceptance |
| [Projection theorem](falcon-projection-theorem.md) | Norm and residual projection proofs and certified constants |
| [PCS extraction](falcon-pcs-extraction.md) | Four groups and the extraction boundary; catalog costs |
| [Falcon-512 ledger](falcon-ledger.md) | Three-stage PIOP and projection errors; four-group PCS accounting |
| [Falcon-1024 ledger](falcon-1024-ledger.md) | Parameter-specific ledger; four-group PCS accounting |
| [Release policy](release-policy.md) | Development compatibility and production acceptance |

## Checks

Numerical checks live in [scripts/](scripts/). Spec CI checks the pinned
projection certificate, Rice wire bound, both composition ledgers,
Falcon-1024 parameter arithmetic, the retained MSIS audit and the Section 10.1
transcript grinding allocation. The grinding check compares every admitted
padded shape with the paper's whole-protocol tree evidence, vendored and
SHA-256 pinned in [evidence/](evidence/). Historical catalog checks do not
certify the target four-group catalogs.

`python3 specs/scripts/falcon_slack_width_check.py` exhaustively checks
bounded four-square decompositions above $`8191^2`$ through $`70{,}265{,}242`$.
It is deliberately not part of CI. Use its optional range for a short run.

## Implementation techniques

- [HashToPoint on stable Rust](techniques/hash-to-point-simd.md).
- [Round-zero integer paths](techniques/round-zero-integer-paths.md).
- [Profiling](techniques/profiling.md).

These describe the current three-stage implementation and its optimized
prover kernels.

## Scope

The verifier receives public keys and messages and derives HashToPoint
targets from the public salts in the proof. Raw private witnesses are not
sent. Zero knowledge is not required or claimed.
