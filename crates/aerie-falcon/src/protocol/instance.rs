//! Canonical public instance binding for the Falcon v2 PIOP.

use sha3_legacy::{
    digest::{ExtendableOutput, Update, XofReader},
    CShake256, CShake256Core,
};

use super::{manifest::CommitmentManifest, wire::SaltMode};
use crate::AggregationError;

pub(crate) const INSTANCE_BYTES: usize = 398;

/// cSHAKE256 over the two length-prefixed byte strings used by protocol domains.
pub(crate) fn digest(label: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hash = framed_hash(label, bytes.len());
    hash.update(bytes);
    finish(hash)
}

fn framed_hash(label: &[u8], len: usize) -> CShake256 {
    let mut hash = CShake256::from_core(CShake256Core::new_with_function_name(
        b"",
        b"aerie/falcon/v2/instance",
    ));
    hash.update(&(label.len() as u64).to_le_bytes());
    hash.update(label);
    hash.update(&(len as u64).to_le_bytes());
    hash
}

fn finish(hash: CShake256) -> [u8; 32] {
    let mut digest = [0; 32];
    hash.finalize_xof().read(&mut digest);
    digest
}

/// Bind the setup's deterministic matrix stream and the trusted catalog row.
pub(crate) fn record(
    manifest: &CommitmentManifest,
    mode: SaltMode,
    public_records: &[u8],
    identities: [[u8; 32]; 3],
) -> Result<[u8; INSTANCE_BYTES], AggregationError> {
    let records: &[u8; 32] = public_records
        .try_into()
        .map_err(|_| "public-record binding must contain exactly 32 bytes")?;
    let binding = manifest.binding_bytes();
    let mut out = [0; INSTANCE_BYTES];
    out[..27].copy_from_slice(&binding[..27]);
    out[27] = match mode {
        SaltMode::PerRecord => 0,
        SaltMode::Common => 1,
    };
    out[28..60].copy_from_slice(records);
    out[60..302].copy_from_slice(&binding[27..]);
    for (slot, identity) in out[302..].chunks_exact_mut(32).zip(identities) {
        slot.copy_from_slice(&identity);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::FalconDegree;

    #[test]
    fn instance_digest_matches_independent_customized_keccak() {
        let label = b"aerie/falcon/v2/instance";
        let record = [0xa5; INSTANCE_BYTES];
        let mut input = Vec::new();
        input.extend_from_slice(&(label.len() as u64).to_le_bytes());
        input.extend_from_slice(label);
        input.extend_from_slice(&(record.len() as u64).to_le_bytes());
        input.extend_from_slice(&record);
        let reference =
            aerie_keccak::customized_chaining_values(&[&input], b"", b"aerie/falcon/v2/instance");
        assert_eq!(digest(label, &record), reference[0].as_bytes()[..32]);
    }

    #[test]
    fn every_instance_field_has_its_specified_position() {
        for (degree, parameter, layout, n, padded) in [
            (FalconDegree::Falcon512, 9, 5_u32, 512_u32, 8_u64),
            (FalconDegree::Falcon1024, 10, 6, 1024, 4),
        ] {
            let manifest = CommitmentManifest::for_batch_of(degree, 1).unwrap();
            let bytes = record(
                &manifest,
                SaltMode::Common,
                &[0x11; 32],
                [[0x33; 32], [0x44; 32], [0x55; 32]],
            )
            .unwrap();
            assert_eq!(bytes.len(), 398);
            assert_eq!(&bytes[0..2], &crate::WIRE_VERSION.to_le_bytes());
            assert_eq!(bytes[2], parameter);
            assert_eq!(&bytes[3..7], &layout.to_le_bytes());
            assert_eq!(&bytes[7..11], &n.to_le_bytes());
            assert_eq!(&bytes[11..19], &1_u64.to_le_bytes());
            assert_eq!(&bytes[19..27], &padded.to_le_bytes());
            assert_eq!(bytes[27], 1);
            assert_eq!(&bytes[28..60], &[0x11; 32]);
            assert_eq!(
                &bytes[60..258],
                manifest.projection_profile().binding_bytes()
            );
            for (i, (polys, cells, family)) in [
                (1, u64::from(n) * padded, 0),
                (1, 4 * padded, 0),
                (4, u64::from(n) * padded, 1),
                (3, padded, 1),
            ]
            .into_iter()
            .enumerate()
            {
                let offset = 258 + 11 * i;
                assert_eq!(bytes[offset], i as u8);
                assert_eq!(bytes[offset + 1], polys);
                assert_eq!(&bytes[offset + 2..offset + 10], &cells.to_le_bytes());
                assert_eq!(bytes[offset + 10], family);
            }
            assert_eq!(&bytes[302..334], &[0x33; 32]);
            assert_eq!(&bytes[334..366], &[0x44; 32]);
            assert_eq!(&bytes[366..398], &[0x55; 32]);
        }
    }
}
