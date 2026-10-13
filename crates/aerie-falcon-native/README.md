# aerie-falcon-native

Falcon byte handling and native verification for Aerie, for both
standardized parameter sets (Falcon-512 and Falcon-1024). Each record has its
own public key, salt, message or context, and HashToPoint target.

Every byte-layer and native-verification type is generic over the sealed
`FalconParams` trait (`Falcon512`, `Falcon1024`). Specify the parameter set
explicitly, for example `DirectRecord<Falcon512>` or
`DirectRecord<Falcon1024>`. `tests/fn_dsa_differential.rs` checks both
parameter sets against real `fn-dsa` key generation, signing, and verification.

This crate owns the canonical-byte trust boundary. It implements canonical
public-key and padded-signature decoding, direct-record construction,
`HashToPoint` preprocessing, source-bound native
verification (NTT-based `s1` reconstruction, exact norm check), and the
exact padded and four-cell encodability witnesses derived from verified
signatures.

Malformed residues, compressed coefficients, negative zero, and nonzero
signature padding are rejected before record construction; prehash mode is
represented but rejected until a differential vector pins its framing.

The decoded arithmetic types expose read-only accessors, but their
constructors are crate-private: the public paths to verified arithmetic
values are `witness::verify_signature_native` and its batched form
`witness::verify_signatures_native`.

The crate has no proof system code; `fn-dsa` is a test-only dependency.
`aerie-falcon` consumes it as the canonical Falcon layer.
