#![no_main]
use aerie_falcon_native::params::FalconParams;
use aerie_falcon_native::source::{CanonicalSignatureBytes, FALCON_COMPRESSED_COEFFICIENT_LIMIT};
use aerie_falcon_native::Falcon1024;
use libfuzzer_sys::fuzz_target;

/// Independent Falcon-1024 re-encoder: 0x3A header, 40-byte salt, then each
/// coefficient as sign bit, 7 low magnitude bits, unary high bits,
/// terminator, padded with zero bits to the Falcon-1024 padded body. Byte
/// equality with the accepted input is what makes loss of the negative-zero
/// or zero-padding canonicality checks a detectable mismatch instead of
/// silent signature-byte malleability.
fn reencode(salt: &[u8; 40], s2: &[i16]) -> Vec<u8> {
    assert_eq!(s2.len(), Falcon1024::DEGREE);
    let mut bits: Vec<bool> = Vec::with_capacity(Falcon1024::PADDED_BODY_BITS);
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
    bits.resize(Falcon1024::PADDED_BODY_BITS, false);
    let mut encoded = vec![0x3A_u8];
    encoded.extend_from_slice(salt);
    for chunk in bits.chunks(8) {
        encoded.push(chunk.iter().fold(0_u8, |byte, &bit| (byte << 1) | u8::from(bit)));
    }
    encoded
}

// Falcon-1024 counterpart of `signature_parse`: `parse` is the trust
// boundary for attacker-supplied padded-signature bytes, and the
// compressed-coefficient limit bounds every |s2[i]|.
fuzz_target!(|data: &[u8]| {
    let Ok(canonical) = CanonicalSignatureBytes::<Falcon1024>::parse(data) else {
        return;
    };
    assert_eq!(canonical.as_bytes().as_slice(), data);
    let decoded = canonical.decode();
    assert_eq!(decoded.salt().as_slice(), &data[1..41]);
    let s2: &[i16] = decoded.s2().as_ref();
    for &coefficient in s2 {
        assert!(coefficient.unsigned_abs() <= FALCON_COMPRESSED_COEFFICIENT_LIMIT);
    }
    // Injectivity oracle: the decoded salt and s2 re-encode to exactly the
    // accepted bytes.
    assert_eq!(reencode(decoded.salt(), s2), data);
    let reparsed =
        CanonicalSignatureBytes::<Falcon1024>::parse(data).expect("accepted bytes re-parse");
    let reparsed = reparsed.decode();
    let reparsed_s2: &[i16] = reparsed.s2().as_ref();
    assert_eq!(reparsed_s2, s2);
});
