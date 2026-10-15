# Release and compatibility policy

Aerie names its supported scheme **Falcon**. The supported mode is
**public-salt**, at both standardized parameter sets, Falcon-512 and
Falcon-1024, which share one wire version. API and command names describe
these capabilities; protocol iteration numbers are not product names.

The supported framing is Falcon v1.2 padded signature format with key-prefixed
message framing (raw mode, empty context), as implemented by fn-dsa 0.3.0.
Original `r || m` framing is unsupported; compatibility with all v1.2 signers
or final FIPS 206 is not claimed.

## Package releases

The workspace has one Cargo SemVer version, declared in
`[workspace.package]`. The current version `0.2.0` is a development release:
it does not declare a production-ready release or publish packages.
Registry packaging, release artifacts and the production acceptance review
remain separate work.

The Falcon API is generic over `P: AggregationParams` (`Falcon512` or
`Falcon1024`); generic types have no implicit Falcon-512 parameter.

The supported facade is `aerie::falcon`. Its default exports are explicit:
parameter-set and setup types, generic statement/record/witness and canonical
byte types, aggregate prover/verifier entry points, metrics and verification
results, `AggregationError`, and wire identity/stability constants. Protocol modules and transcript
implementation details are private. The private `aerie-bench` harness accesses
producer internals through `aerie-prover/internals` and common verification
internals through `aerie-falcon/internals`; the default facade remains
the supported public API. With default features disabled, the `verifier`
selection omits proving and witness preparation. `parallel` enables parallel
verification without enabling the optional `prover` feature.

## Wire compatibility

`crates/aerie-falcon` owns the common protocol support and verifier replay
for the normative [Falcon specification](falcon.md). `crates/aerie-prover`
executes its three batched sumcheck stages and four multi-polynomial Akita
groups. The transcript state machine and canonical wire grammar remain common.
The ledgers include the four-group catalog costs.

Both parameter sets use wire version 1. PCS frames use the
16-byte magic `aerie/falcon/v1p` for per-record salts or
`aerie/falcon/v1c` for common salts. The transcript
root/domain is `aerie/falcon/v1` (`lib.rs::PROTOCOL_DOMAIN`). The header carries
LE16 version at offset 16 and parameter byte 9 or 10 at offset 18. The 187
projection outputs use the canonical Rice section defined in the Falcon
specification.

Wire version 1 was frozen at Aerie's first public release. Incompatible
transcript or encoding changes require a new wire version. Proof interchange
requires the same pinned implementation and profile.
Verifiers must continue rejecting unsupported identifiers and malformed or
trailing data.

## Production acceptance

The requested disclosure contract is **raw witnesses are not sent to the
verifier**. Zero knowledge is not a release requirement. Witness omission does
not establish zero knowledge or guarantee that proof observations reveal no
information; documentation must not imply either property.

Before declaring a parameter set production-ready, its acceptance evidence
must cover:

- The exact supported statement, current dependency revisions, parameter
  catalogs and applicable soundness assumptions, carrying forward existing
  audits where the audited implementation is unchanged.
- A supported verifier interface and complete canonical proof codec, with
  explicit public inputs and no raw private-witness transport.
- Mutation, malformed-input and parameter-set/identifier-separation tests, plus a
  reviewed compatibility and release contract.
- Resolution or explicit acceptance of the remaining normative protocol
  review obligations. A release number alone does not close them.

Falcon-512 and Falcon-1024 may reach these criteria separately. Public-salt
proofs intentionally transport their salts as public data. The open obligations for
both parameter sets are listed in [falcon.md](falcon.md) Section 12.2. Zero knowledge must not be added back as an implicit gate.
