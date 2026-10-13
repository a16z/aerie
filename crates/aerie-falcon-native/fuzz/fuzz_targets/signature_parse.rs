#![no_main]
use aerie_falcon_native::params::FALCON_DEGREE;
use aerie_falcon_native::source::{
    CanonicalSignatureBytes, FALCON_512_PADDED_BODY_BITS, FALCON_COMPRESSED_COEFFICIENT_LIMIT,
};
use aerie_falcon_native::Falcon512;
use libfuzzer_sys::fuzz_target;

/// Independent re-encoder: 0x39 header, 40-byte salt, then each coefficient
/// as sign bit, 7 low magnitude bits, unary high bits, terminator, padded
/// with zero bits. Byte equality with the accepted input is what makes loss
/// of the negative-zero or zero-padding canonicality checks a detectable
/// mismatch instead of silent signature-byte malleability.
fn reencode(salt: &[u8; 40], s2: &[i16; FALCON_DEGREE]) -> Vec<u8> {
    let mut bits: Vec<bool> = Vec::with_capacity(FALCON_512_PADDED_BODY_BITS);
    for &coefficient in s2 {
        let magnitude = coefficient.unsigned_abs();
        bits.push(coefficient < 0);
        let low = magnitude & 127;
        for shift in (0..7).rev() {
            bits.push((low >> shift) & 1 != 0);
        }
        bits.extend(std::iter::repeat_n(false, usize::from(magnitude >> 7)));
        bits.push(true);
    }
    bits.resize(FALCON_512_PADDED_BODY_BITS, false);
    let mut encoded = vec![0x39_u8];
    encoded.extend_from_slice(salt);
    for chunk in bits.chunks(8) {
        encoded.push(chunk.iter().fold(0_u8, |byte, &bit| (byte << 1) | u8::from(bit)));
    }
    encoded
}

// `parse` is the trust boundary for attacker-supplied padded-signature
// bytes; `decode` relies on parse-time validation for its internal
// `expect`s, and the compressed-coefficient limit bounds every |s2[i]|.
fuzz_target!(|data: &[u8]| {
    let Ok(canonical) = CanonicalSignatureBytes::<Falcon512>::parse(data) else {
        return;
    };
    assert_eq!(canonical.as_bytes().as_slice(), data);
    let decoded = canonical.decode();
    assert_eq!(decoded.salt().as_slice(), &data[1..41]);
    for &coefficient in decoded.s2() {
        assert!(coefficient.unsigned_abs() <= FALCON_COMPRESSED_COEFFICIENT_LIMIT);
    }
    // Injectivity oracle: the decoded salt and s2 re-encode to exactly the
    // accepted bytes, so two distinct byte strings can never decode to the
    // same signature.
    assert_eq!(reencode(decoded.salt(), decoded.s2()), data);
    // Accepted bytes must re-parse identically (canonical form is stable).
    let reparsed =
        CanonicalSignatureBytes::<Falcon512>::parse(data).expect("accepted bytes re-parse");
    assert_eq!(reparsed.decode().s2(), decoded.s2());
});
