//! Canonical public instance binding for the Falcon v1 PIOP.

use aerie_keccak::CShake256;

use super::{manifest::CommitmentManifest, wire::SaltMode};
use crate::AggregationError;

pub(crate) const BINDING_PREFIX_BYTES: usize = 27;
pub(crate) const SALT_MODE_OFFSET: usize = BINDING_PREFIX_BYTES;
pub(crate) const DIGEST_BYTES: usize = 32;
pub(crate) const IDENTITY_COUNT: usize = 3;
pub(crate) const BINDING_DIGEST_RANGE: core::ops::Range<usize> =
    SALT_MODE_OFFSET + 1..SALT_MODE_OFFSET + 1 + DIGEST_BYTES;
const MANIFEST_SUFFIX_BYTES: usize = 242;
pub(crate) const MANIFEST_SUFFIX_RANGE: core::ops::Range<usize> =
    BINDING_DIGEST_RANGE.end..BINDING_DIGEST_RANGE.end + MANIFEST_SUFFIX_BYTES;
pub(crate) const IDENTITY_START: usize = MANIFEST_SUFFIX_RANGE.end;
pub const INSTANCE_BYTES: usize = IDENTITY_START + IDENTITY_COUNT * DIGEST_BYTES;

const _: () = assert!(INSTANCE_BYTES == 398);

/// cSHAKE256 over the two length-prefixed byte strings used by protocol domains.
#[must_use]
pub fn digest(label: &[u8], bytes: &[u8]) -> [u8; DIGEST_BYTES] {
    let mut hash = framed_hash(label, bytes.len());
    hash.update(bytes);
    finish(hash)
}

fn framed_hash(label: &[u8], len: usize) -> CShake256 {
    let mut hash = CShake256::new(b"", b"aerie/falcon/v1/instance");
    hash.update(&(label.len() as u64).to_le_bytes());
    hash.update(label);
    hash.update(&(len as u64).to_le_bytes());
    hash
}

fn finish(hash: CShake256) -> [u8; DIGEST_BYTES] {
    let mut digest = [0; DIGEST_BYTES];
    hash.finalize_xof().read(&mut digest);
    digest
}

/// Bind the setup's deterministic matrix stream and the trusted catalog row.
pub fn record(
    manifest: &CommitmentManifest,
    mode: SaltMode,
    public_records: &[u8],
    identities: [[u8; DIGEST_BYTES]; IDENTITY_COUNT],
) -> Result<[u8; INSTANCE_BYTES], AggregationError> {
    let records: &[u8; DIGEST_BYTES] = public_records
        .try_into()
        .map_err(|_| crate::StatementMismatch::PublicInputShape)?;
    let binding = manifest.binding_bytes();
    let mut out = [0; INSTANCE_BYTES];
    out[..BINDING_PREFIX_BYTES].copy_from_slice(&binding[..BINDING_PREFIX_BYTES]);
    out[SALT_MODE_OFFSET] = match mode {
        SaltMode::PerRecord => 0,
        SaltMode::Common => 1,
    };
    out[BINDING_DIGEST_RANGE].copy_from_slice(records);
    out[MANIFEST_SUFFIX_RANGE].copy_from_slice(&binding[BINDING_PREFIX_BYTES..]);
    for (slot, identity) in out[IDENTITY_START..]
        .chunks_exact_mut(DIGEST_BYTES)
        .zip(identities)
    {
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
        let label = b"aerie/falcon/v1/instance";
        let record = [0xa5; INSTANCE_BYTES];
        let mut input = Vec::new();
        input.extend_from_slice(&(label.len() as u64).to_le_bytes());
        input.extend_from_slice(label);
        input.extend_from_slice(&(record.len() as u64).to_le_bytes());
        input.extend_from_slice(&record);
        let reference =
            aerie_keccak::customized_chaining_values(&[&input], b"", b"aerie/falcon/v1/instance");
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
