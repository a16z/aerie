#![no_main]
use aerie_falcon_native::params::{FALCON_DEGREE, FALCON_Q};
use aerie_falcon_native::source::CanonicalPublicKeyBytes;
use aerie_falcon_native::Falcon512;
use libfuzzer_sys::fuzz_target;

/// Independent re-encoder: 0x09 header, then 512 x 14-bit residues packed
/// MSB-first. Byte equality with the accepted input is what makes loss of
/// a parse-time canonicality check (residue < q) a detectable mismatch
/// instead of silent encoding malleability.
fn reencode(h_centered: &[i16; FALCON_DEGREE]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(FALCON_DEGREE * 14);
    for &coefficient in h_centered {
        let residue = if coefficient < 0 {
            (coefficient + FALCON_Q as i16) as u16
        } else {
            coefficient as u16
        };
        for shift in (0..14).rev() {
            bits.push((residue >> shift) & 1 != 0);
        }
    }
    let mut encoded = vec![0x09_u8];
    for chunk in bits.chunks(8) {
        encoded.push(chunk.iter().fold(0_u8, |byte, &bit| (byte << 1) | u8::from(bit)));
    }
    encoded
}

// `parse` is the trust boundary for attacker-supplied public-key bytes;
// `decode` relies on parse-time validation for its internal `expect`s.
fuzz_target!(|data: &[u8]| {
    let Ok(canonical) = CanonicalPublicKeyBytes::<Falcon512>::parse(data) else {
        return;
    };
    assert_eq!(canonical.as_bytes().as_slice(), data);
    let decoded = canonical.decode();
    for &coefficient in decoded.h_centered() {
        assert!((-6_144..=6_144).contains(&coefficient));
    }
    // Injectivity oracle: the decoded polynomial re-encodes to exactly the
    // accepted bytes, so two distinct byte strings can never decode to the
    // same key.
    assert_eq!(reencode(decoded.h_centered()), data);
    // Accepted bytes must re-parse identically (canonical form is stable).
    let reparsed =
        CanonicalPublicKeyBytes::<Falcon512>::parse(data).expect("accepted bytes re-parse");
    assert_eq!(reparsed.decode().h_centered(), decoded.h_centered());
});
