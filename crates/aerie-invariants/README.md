# aerie-invariants

Machine-checkable security invariants for Aerie Falcon v1, in the spirit of
Jolt's [`jolt-eval`](https://github.com/a16z/jolt/blob/main/jolt-eval/src/invariant/soundness.rs):
each property is a deterministic predicate over a JSON-serializable
counterexample candidate. Completeness, witness soundness, and statement
binding exercise Falcon-512. Verifier totality also checks Falcon-1024 decoding
and verification when the input frame names that parameter set.

A candidate is checked mechanically and lands in one of three buckets:

| verdict | meaning |
| --- | --- |
| `Holds` | the property was upheld on this input |
| `Violated` | a confirmed, reproducible counterexample |
| `Invalid` | degenerate / out of scope — skipped, so a no-op can't count as a pass |

## The invariants

| name | property | a counterexample is… |
| --- | --- | --- |
| `completeness` | canonical, native-verifier-valid statement/witness pairs verify end to end under the production profile | a native-verifier-valid pair that the production `prove_aggregate`→`verify_aggregate` path rejects or panics on |
| `witness-soundness` | no accepting proof carries a witness that breaks a checked predicate | a tamper the full PCS verifier **accepts** despite a net witness violating an exact semantic predicate |
| `statement-binding` | no proof verifies under two statements | a pair A≠B where a proof first verified under A and still verifies when re-presented under B's keys+salts+messages |
| `verifier-totality` | `verify_aggregate` / `decode_pcs_proof` never panic | any bytes that make either function unwind |

`completeness` and `witness-soundness` run the real Akita PCS under the current
production signed width-14 and width-4 profiles; `witness-soundness` uses its
supported eight-record shape. `statement-binding` and `verifier-totality` also
run the real PCS. Every check that proves or verifies runs on a 512 MiB stack.

`completeness` and `statement-binding` accept live record counts **1–16** with
the current PCS catalog. `supported_statement_record_counts()` derives their
shared domain from the catalog within the harness's 16-record resource cap;
validation and briefings use that same list. Other counts are explicitly
out of scope before native verification or PCS setup. Statement binding also
requires A and B to have the same record count.

`completeness` intentionally defines its scope by canonical decoding, matching
record and signature counts, and native FN-DSA verification. It does not require
the keys or signatures to have been produced by the harness's key-generation or
signing helpers.

`statement-binding` requires A to be provable and baseline-verifiable, including
constructed relation-positive fixtures. B needs well-formed public data with a
different key, salt, or message. The checker first verifies A's proof under A,
then checks it again under B's public data: B's keys and messages are supplied
to the verifier, while B's salts replace A's in the transport frame.

`witness-soundness` reports a violation only when the full PCS verifier accepts
an independently invalid witness. The shared statement validator checks the
exact negacyclic ring residual, norm/epsilon identity, per-row high-part budget,
and derived bounds (`|s2| <= 2047`, `|s1|, |epsilon| <= 5833`, `|v| <= 65998`).
Protocol validators check source alphabets, decoded `s2` and budget consistency,
and the projection outputs `(y, z)` against the clear projection under the
actual commitment-derived seed. In-range edits that break these identities are in scope.

Writes are last-write-wins. Restored or semantically equivalent witnesses and
proof-only candidates are `Invalid`. Proof claims can accompany an invalid
witness and accept any signed 64-bit delta. Candidates are limited to 1024
edits on eight synthetic records under the centered, signed `S2` encoding. This is
bounded adversarial coverage, not a general soundness proof: record-count,
live-mask, and semantic-padding-dependent failures are not exercised, even
though the shared batch validator also checks zero semantic padding.

## Tests

```sh
# Fast unit tests (no protocol):
cargo test -p aerie-invariants --lib

# Per-invariant behavior against the real prover/verifier (release speed):
cargo test -p aerie-invariants --test invariants --profile ci -- --test-threads=1
```

`--test-threads=1` keeps the verifier-totality check's panic-hook trap from
racing across concurrent checks.

## Extending

Implement `Invariant` (associated `Input`, `NAME`, `briefing`, `seed_corpus`,
`check`) and add it to `invariant::registry`. The object-safe `DynInvariant`
blanket impl picks it up automatically — no macro, no registration
boilerplate.
