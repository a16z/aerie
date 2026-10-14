# Release and compatibility policy

Aerie names its supported scheme **Falcon**. The supported mode is
**public-salt**, at both standardized parameter sets, Falcon-512 and
Falcon-1024, which share one wire version. API and command names describe
these capabilities; protocol iteration numbers are not product names.

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
implementation details are private; facade harness access requires
`bench-internals`.

## Wire compatibility

`crates/aerie-falcon` implements the normative [Falcon specification](falcon.md):
three batched sumcheck stages, four multi-polynomial Akita groups, and their
canonical transcript and wire grammar. The ledgers include the four-group
catalog costs.

Both parameter sets use development wire version 2. PCS frames use the
16-byte magic `aerie/falcon/v2p` for per-record salts or
`aerie/falcon/v2c` for common salts. The transcript
root/domain is `aerie/falcon/v2` (`lib.rs::PROTOCOL_DOMAIN`). The header carries
LE16 version at offset 16 and parameter byte 9 or 10 at offset 18. The 187
projection outputs use the canonical Rice section defined in the Falcon
specification.

Version 2 separates the projection streams, record leaves and instance digest
from raw Falcon SHAKE256 using cSHAKE256. It changes proof bytes and transcript
domains; version-1 frames are rejected. Signature and key encodings are unchanged.

Version 2 is the current development wire. It will be frozen at the first
public release; until then it may change without a version bump, and proof
interchange requires the same pinned implementation and profile. After that
release, a change to accepted statements, transcript bytes, proof grammar,
field choices or layout must use an explicit wire/profile compatibility change.
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
