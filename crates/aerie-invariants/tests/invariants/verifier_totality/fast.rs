//! Fast verifier-totality tests.

use aerie_invariants::invariants::{TotalityInput, VerifierTotality};
use aerie_invariants::Invariant;

use crate::support::assert_invalid;

fn feed(records: usize, bytes: Vec<u8>) -> TotalityInput {
    TotalityInput {
        records,
        bytes: bytes.into(),
    }
}

#[test]
fn resource_caps_are_out_of_scope_before_verification() {
    let inv = VerifierTotality;
    assert_invalid(inv.check(&feed(0, Vec::new())), "zero records");
    assert_invalid(
        inv.check(&feed(257, Vec::new())),
        "records just over the cap",
    );
    assert_invalid(
        inv.check(&feed(usize::MAX, Vec::new())),
        "records far over the cap",
    );
    assert_invalid(
        inv.check(&feed(8, vec![0_u8; 4 * 1024 * 1024 + 1])),
        "bytes just over the cap",
    );
}

#[test]
fn bytes_deserialize_from_both_file_path_spellings() {
    let expected = vec![7_u8, 8, 9];
    let path = std::env::temp_dir().join(format!(
        "aerie-verifier-totality-schema-{}.bin",
        std::process::id()
    ));
    std::fs::write(&path, &expected).expect("write the byte fixture");
    let display = path.display().to_string();

    for spelling in [
        serde_json::json!({ "records": 8, "bytes": { "file": display.clone() } }),
        serde_json::json!({ "records": 8, "bytes": display.clone() }),
    ] {
        let parsed: TotalityInput = serde_json::from_value(spelling).expect("parse");
        assert_eq!(
            *parsed.bytes, expected,
            "the file's bytes reached the input"
        );
    }

    // A path that does not exist is a schema error, not a panic or violation.
    let missing = serde_json::json!({
        "records": 8,
        "bytes": { "file": "/nonexistent/x.bin" }
    });
    assert!(serde_json::from_value::<TotalityInput>(missing).is_err());

    let _ = std::fs::remove_file(&path);
}
