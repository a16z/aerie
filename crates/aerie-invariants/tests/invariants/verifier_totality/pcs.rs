//! PCS-backed verifier-totality tests.

use aerie_falcon::wire::HEADER_PARAMETER_SET_OFFSET;
use aerie_invariants::invariants::{
    sample_envelope, sample_envelope_1024, TotalityInput, VerifierTotality,
};
use aerie_invariants::Invariant;

use crate::support::{assert_holds, assert_not_violated, pcs_guard};

fn feed(records: usize, bytes: Vec<u8>) -> TotalityInput {
    TotalityInput {
        records,
        bytes: bytes.into(),
    }
}

#[test]
fn valid_patched_and_garbage_inputs_never_panic() {
    let _pcs = pcs_guard();
    let inv = VerifierTotality;
    // A real constructed envelope decodes and verifies deeply — no panic.
    let envelope = sample_envelope(8).expect("build a sample envelope");
    assert_holds(inv.check(&feed(8, envelope.clone())), "pristine envelope");
    // A byte flipped deep in the proof region: rejected, no panic.
    let mut flipped = envelope;
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xFF;
    assert_not_violated(inv.check(&feed(8, flipped)), "one deep byte flip");
    // Shallow inputs: the decoder and verifier must reject without panicking.
    assert_not_violated(inv.check(&feed(8, Vec::new())), "empty bytes");
    assert_not_violated(inv.check(&feed(8, vec![0xAB; 64])), "pure garbage");
    assert_not_violated(
        inv.check(&feed(8, b"aerie/falcon/v1p".to_vec())),
        "magic then nothing",
    );
}

/// A Falcon-1024 envelope reaches the deep Falcon-1024 verifier (its header
/// names parameter set 10 and five records is a Falcon-1024 catalog row);
/// the pristine frame, a deep flip, and a relabeled parameter set must not
/// panic either verifier.
#[test]
fn falcon_1024_envelopes_never_panic() {
    let _pcs = pcs_guard();
    let inv = VerifierTotality;
    let envelope = sample_envelope_1024(5).expect("build a Falcon-1024 sample envelope");
    assert_eq!(envelope[HEADER_PARAMETER_SET_OFFSET], 10);
    assert_holds(
        inv.check(&feed(5, envelope.clone())),
        "pristine Falcon-1024 envelope",
    );
    let mut flipped = envelope.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xFF;
    assert_not_violated(
        inv.check(&feed(5, flipped)),
        "one deep Falcon-1024 byte flip",
    );
    let mut relabeled = envelope;
    relabeled[HEADER_PARAMETER_SET_OFFSET] = 9;
    assert_not_violated(
        inv.check(&feed(5, relabeled)),
        "a Falcon-1024 body labeled Falcon-512",
    );
}

/// Both file-path spellings must feed the exact envelope through the verifier.
#[test]
fn accepts_envelope_from_both_file_path_spellings() {
    let _pcs = pcs_guard();
    let envelope = sample_envelope(8).expect("build a sample envelope");
    let path = std::env::temp_dir().join(format!(
        "aerie-verifier-totality-envelope-{}.bin",
        std::process::id()
    ));
    std::fs::write(&path, &envelope).expect("write the envelope");
    let display = path.display().to_string();

    for spelling in [
        serde_json::json!({ "records": 8, "bytes": { "file": display.clone() } }),
        serde_json::json!({ "records": 8, "bytes": display.clone() }),
    ] {
        let parsed: TotalityInput = serde_json::from_value(spelling).expect("parse");
        assert_eq!(
            *parsed.bytes, envelope,
            "the file's bytes reached the input"
        );
        assert_holds(VerifierTotality.check(&parsed), "envelope supplied by path");
    }

    let _ = std::fs::remove_file(&path);
}
