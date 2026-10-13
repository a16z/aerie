#![no_main]
use aerie_falcon_native::params::{FalconParams, FALCON_Q};
use aerie_falcon_native::source::CanonicalPublicKeyBytes;
use aerie_falcon_native::Falcon1024;
use libfuzzer_sys::fuzz_target;

/// Independent Falcon-1024 re-encoder: 0x0A header, then 1024 x 14-bit
/// residues packed MSB-first. Byte equality with the accepted input is what
/// makes loss of a parse-time canonicality check (residue < q) a detectable
/// mismatch instead of silent encoding malleability.
fn reencode(h_centered: &[i16]) -> Vec<u8> {
    assert_eq!(h_centered.len(), Falcon1024::DEGREE);
    let mut bits = Vec::with_capacity(Falcon1024::DEGREE * 14);
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
    let mut encoded = vec![0x0A_u8];
    for chunk in bits.chunks(8) {
        encoded.push(chunk.iter().fold(0_u8, |byte, &bit| (byte << 1) | u8::from(bit)));
    }
    encoded
}

// Falcon-1024 counterpart of `pubkey_parse`: `parse` is the trust boundary
// for attacker-supplied public-key bytes, and `decode` relies on parse-time
// validation for its internal `expect`s.
fuzz_target!(|data: &[u8]| {
    let Ok(canonical) = CanonicalPublicKeyBytes::<Falcon1024>::parse(data) else {
        return;
    };
    assert_eq!(canonical.as_bytes().as_slice(), data);
    let decoded = canonical.decode();
    let h_centered: &[i16] = decoded.h_centered().as_ref();
    for &coefficient in h_centered {
        assert!((-6_144..=6_144).contains(&coefficient));
    }
    // Injectivity oracle: the decoded polynomial re-encodes to exactly the
    // accepted bytes.
    assert_eq!(reencode(h_centered), data);
    let reparsed =
        CanonicalPublicKeyBytes::<Falcon1024>::parse(data).expect("accepted bytes re-parse");
    let reparsed = reparsed.decode();
    let reparsed_h: &[i16] = reparsed.h_centered().as_ref();
    assert_eq!(reparsed_h, h_centered);
});
