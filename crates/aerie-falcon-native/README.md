# aerie-falcon-native

Canonical Falcon byte handling and public preprocessing for Aerie, for both
standardized parameter sets (Falcon-512 and Falcon-1024).

The supported framing is Falcon v1.2 padded signature format with key-prefixed
message framing (raw mode, empty context), as implemented by fn-dsa 0.3.0.
The `HashToPoint` input is
`salt || SHAKE256(pk,64) || 0x00 || 0x00 || message`. Original `r || m`
framing is unsupported; compatibility with all v1.2 signers or final FIPS 206
is not claimed. Prehashed-key helpers cache only the public-key hash; they are
not a message-prehash API.

Every byte-layer type is generic over the sealed `FalconParams` trait
(`Falcon512`, `Falcon1024`). This crate owns canonical public-key and
padded-signature decoding and public `HashToPoint` preprocessing. Malformed
residues, compressed coefficients, negative zero, and nonzero signature
padding are rejected by the codecs. `tests/fn_dsa_differential.rs` checks
codec behavior against reproducible fn-dsa signatures for both parameter sets.

Validated native arithmetic, `DirectRecord`, native signature verification,
and encodability witnesses belong to
[`aerie-falcon-witness`](../aerie-falcon-witness/src/lib.rs). Its arithmetic
constructors remain crate-private; verified values are created through
`native::witness::verify_signature_native` and its batched form. Its
[differential suite](../aerie-falcon-witness/tests/fn_dsa_differential.rs)
compares verification, batching, norms, and encodability against fn-dsa.

The crate has no proof system code; `fn-dsa` is a test-only dependency.
`aerie-falcon-statement` consumes it for public statement ingestion.
