use aerie_falcon::wire::{decode_pcs_proof, encode_pcs_proof_from_salt_frame};
use aerie_falcon::FalconDegree;

/// Canonical wire-proof fuzz oracle shared with the regression tests.
pub fn check_wire_pcs_proof(data: &[u8]) {
    let mut accepted = 0;
    for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
        let Ok(decoded) = decode_pcs_proof(degree, data) else {
            continue;
        };
        accepted += 1;
        assert_eq!(decoded.degree, degree);
        let reencoded = encode_pcs_proof_from_salt_frame(
            decoded.degree,
            decoded.records,
            &decoded.salts,
            &decoded.proof,
        )
        .expect("an accepted proof re-encodes");
        assert_eq!(reencoded.as_slice(), data);
    }
    assert!(accepted <= 1, "one frame decoded under both parameter sets");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_salt_seed_roundtrips_through_the_fuzz_oracle() {
        check_wire_pcs_proof(include_bytes!("../seeds/wire_pcs_proof/falcon512-v1c"));
    }
}
