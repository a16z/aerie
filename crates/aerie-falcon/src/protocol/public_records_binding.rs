//! Canonical public-record encoding and domain-separated tree binding.

use aerie_falcon_native::{
    preprocess::HashedVerifyingKey, source::CanonicalPublicKeyBytes, FalconParams,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    wire::{SaltMode, TRANSPORT_SALT_BYTES},
    MAX_RECORDS,
};

/// Fixed block size of the public-record tree binding.
pub const PUBLIC_RECORDS_BINDING_BLOCK_BYTES: usize = 16 * 1024;
const LEAF_CUSTOMIZATION: &[u8] = b"aerie/falcon/v2/record-leaf";
const BINDING_CUSTOMIZATION: &[u8] = b"aerie/falcon/v2/public-records-binding";
const COMMON_BINDING_CUSTOMIZATION: &[u8] = b"aerie/falcon/v2/public-records-binding/common-salt";
const BINDING_COUNT_BYTES: usize = 8;
const BINDING_MESSAGE_LENGTH_BYTES: usize = 8;

use crate::AggregationError as Error;

/// Compute the canonical public-record binding digest of Section 2.3.
///
/// Each canonical key is first committed by FN-DSA's existing 64-byte `hpk`.
/// The ordered stream `LE64(records) || (hpk || salt || LE64(message.len()) ||
/// message)*` is then processed by a customized cSHAKE256 tree with a fixed 16-KiB
/// block size and `BINDING_CUSTOMIZATION`. The segmented implementation does
/// not materialize that logical stream or pad records to block boundaries.
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn public_records_binding(
    public_keys: &[CanonicalPublicKeyBytes<aerie_falcon_native::Falcon512>],
    salts: &[[u8; TRANSPORT_SALT_BYTES]],
    messages: &[Vec<u8>],
) -> Result<[u8; 32], Error> {
    public_records_binding_of(public_keys, salts, messages)
}

/// [`public_records_binding`] over canonical keys of any parameter set. The
/// stream hashes the 64-byte `hpk` of each key, so the digest layout does
/// not depend on the key length.
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn public_records_binding_of<P: FalconParams>(
    public_keys: &[CanonicalPublicKeyBytes<P>],
    salts: &[[u8; TRANSPORT_SALT_BYTES]],
    messages: &[Vec<u8>],
) -> Result<[u8; 32], Error> {
    validate_binding_shapes(public_keys.len(), salts, messages)?;
    let hashed_keys =
        tracing::info_span!("v1.binding.hashed_keys").in_scope(|| hashed_keys(public_keys));
    tracing::info_span!("v1.binding.stream")
        .in_scope(|| public_records_binding_from_hashed_keys(&hashed_keys, salts, messages))
}

/// The FN-DSA `hpk` of every canonical public key, in record order. The
/// prover's binding and the verifier's binding and targets share this one
/// derivation.
pub(super) fn hashed_keys<P: FalconParams>(
    public_keys: &[CanonicalPublicKeyBytes<P>],
) -> Vec<HashedVerifyingKey> {
    #[cfg(feature = "parallel")]
    let hashed_keys =
        aerie_falcon_native::preprocess::hashed_verifying_keys_batch_parallel(public_keys);
    #[cfg(not(feature = "parallel"))]
    let hashed_keys = aerie_falcon_native::preprocess::hashed_verifying_keys_batch(public_keys);
    hashed_keys
}

fn validate_binding_shapes(
    records: usize,
    salts: &[[u8; TRANSPORT_SALT_BYTES]],
    messages: &[Vec<u8>],
) -> Result<(), Error> {
    if records == 0 || records > MAX_RECORDS {
        return Err("public-record count is out of range".into());
    }
    if salts.len() != records || messages.len() != records {
        return Err("public-record binding shapes disagree".into());
    }
    Ok(())
}

#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub(super) fn public_records_binding_from_hashed_keys(
    hashed_keys: &[HashedVerifyingKey],
    salts: &[[u8; TRANSPORT_SALT_BYTES]],
    messages: &[Vec<u8>],
) -> Result<[u8; 32], Error> {
    public_records_binding_with_mode(hashed_keys, salts, messages, SaltMode::PerRecord)
}

/// Bind the mode in the digest absorbed before the first challenge. The
/// per-record customization and logical stream remain byte-identical.
pub(super) fn public_records_binding_with_mode(
    hashed_keys: &[HashedVerifyingKey],
    salts: &[[u8; TRANSPORT_SALT_BYTES]],
    messages: &[Vec<u8>],
    mode: SaltMode,
) -> Result<[u8; 32], Error> {
    mode.validate(salts)?;
    let stream = PublicBindingStream::new(hashed_keys, salts, messages)?;
    let chaining_values = stream.chaining_values();
    Ok(aerie_keccak::parallel_hash256_root(
        &chaining_values,
        PUBLIC_RECORDS_BINDING_BLOCK_BYTES,
        match mode {
            SaltMode::PerRecord => BINDING_CUSTOMIZATION,
            SaltMode::Common => COMMON_BINDING_CUSTOMIZATION,
        },
    ))
}

/// Random-access view of the canonical binding stream.
///
/// The offset index has one entry per record. A block lookup performs one
/// binary search, then walks only the fields intersecting that block. Thus a
/// record crossing a `ParallelHash` boundary is copied in two bounded pieces;
/// no record padding, prefix rescans, or full-stream allocation is required.
struct PublicBindingStream<'a> {
    hashed_keys: &'a [HashedVerifyingKey],
    salts: &'a [[u8; TRANSPORT_SALT_BYTES]],
    messages: &'a [Vec<u8>],
    record_starts: Vec<usize>,
    encoded_len: usize,
}

impl<'a> PublicBindingStream<'a> {
    fn new(
        hashed_keys: &'a [HashedVerifyingKey],
        salts: &'a [[u8; TRANSPORT_SALT_BYTES]],
        messages: &'a [Vec<u8>],
    ) -> Result<Self, Error> {
        validate_binding_shapes(hashed_keys.len(), salts, messages)?;
        let mut record_starts = Vec::with_capacity(hashed_keys.len() + 1);
        let mut encoded_len = BINDING_COUNT_BYTES;
        record_starts.push(encoded_len);
        for message in messages {
            u64::try_from(message.len()).map_err(|_| "binding message length exceeds LE64")?;
            encoded_len = encoded_len
                .checked_add(
                    HashedVerifyingKey::BYTE_LEN
                        + TRANSPORT_SALT_BYTES
                        + BINDING_MESSAGE_LENGTH_BYTES
                        + message.len(),
                )
                .ok_or("public-record binding length overflow")?;
            record_starts.push(encoded_len);
        }
        Ok(Self {
            hashed_keys,
            salts,
            messages,
            record_starts,
            encoded_len,
        })
    }

    fn block_count(&self) -> usize {
        self.encoded_len
            .div_ceil(PUBLIC_RECORDS_BINDING_BLOCK_BYTES)
    }

    fn block(&self, block: usize) -> Vec<u8> {
        let start = block * PUBLIC_RECORDS_BINDING_BLOCK_BYTES;
        let end = start
            .saturating_add(PUBLIC_RECORDS_BINDING_BLOCK_BYTES)
            .min(self.encoded_len);
        let mut output = vec![0_u8; end - start];
        self.copy_range(start, &mut output);
        output
    }

    fn copy_range(&self, mut position: usize, output: &mut [u8]) {
        debug_assert!(position + output.len() <= self.encoded_len);
        let mut written = 0;
        if position < BINDING_COUNT_BYTES {
            let count = (self.hashed_keys.len() as u64).to_le_bytes();
            let take = (BINDING_COUNT_BYTES - position).min(output.len());
            output[..take].copy_from_slice(&count[position..position + take]);
            written += take;
            position += take;
        }
        if written == output.len() {
            return;
        }

        let mut record = self
            .record_starts
            .partition_point(|&start| start <= position)
            .saturating_sub(1);
        while written < output.len() {
            let record_start = self.record_starts[record];
            let offset = position - record_start;
            let hpk = self.hashed_keys[record].as_bytes();
            let salt: &[u8] = &self.salts[record];
            let message = &self.messages[record];
            let message_length = (message.len() as u64).to_le_bytes();
            let (field, field_offset): (&[u8], usize) = if offset < hpk.len() {
                (hpk, offset)
            } else if offset < hpk.len() + salt.len() {
                (salt, offset - hpk.len())
            } else if offset < hpk.len() + salt.len() + BINDING_MESSAGE_LENGTH_BYTES {
                (&message_length, offset - hpk.len() - salt.len())
            } else {
                (
                    message,
                    offset - hpk.len() - salt.len() - BINDING_MESSAGE_LENGTH_BYTES,
                )
            };
            let take = (field.len() - field_offset).min(output.len() - written);
            output[written..written + take]
                .copy_from_slice(&field[field_offset..field_offset + take]);
            written += take;
            position += take;
            if position == self.record_starts[record + 1] {
                record += 1;
            }
        }
    }

    fn chaining_values(&self) -> Vec<aerie_keccak::ParallelHash256ChainingValue> {
        let width = aerie_keccak::backend_parallel_width();
        let groups = self.block_count().div_ceil(width);
        #[cfg(feature = "parallel")]
        let grouped = (0..groups)
            .into_par_iter()
            .map(|group| self.hash_group(group, width))
            .collect::<Vec<_>>();
        #[cfg(not(feature = "parallel"))]
        let grouped = (0..groups)
            .map(|group| self.hash_group(group, width))
            .collect::<Vec<_>>();
        grouped.into_iter().flatten().collect()
    }

    fn hash_group(
        &self,
        group: usize,
        width: usize,
    ) -> Vec<aerie_keccak::ParallelHash256ChainingValue> {
        let first = group * width;
        let end = (first + width).min(self.block_count());
        let blocks = (first..end)
            .map(|block| self.block(block))
            .collect::<Vec<_>>();
        let inputs = blocks.iter().map(Vec::as_slice).collect::<Vec<_>>();
        aerie_keccak::customized_chaining_values(&inputs, b"", LEAF_CUSTOMIZATION)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerie_falcon_native::preprocess::hashed_verifying_keys_batch;

    fn zero_public_key() -> CanonicalPublicKeyBytes<aerie_falcon_native::Falcon512> {
        let mut bytes = [0_u8; aerie_falcon_native::source::FALCON_512_PUBLIC_KEY_BYTES];
        bytes[0] = 0x09;
        CanonicalPublicKeyBytes::parse(&bytes).expect("canonical zero key")
    }

    fn crossing_fixture() -> (
        Vec<HashedVerifyingKey>,
        Vec<[u8; TRANSPORT_SALT_BYTES]>,
        Vec<Vec<u8>>,
    ) {
        let block = PUBLIC_RECORDS_BINDING_BLOCK_BYTES;
        let public_keys = vec![zero_public_key(); 5];
        let hashed_keys = hashed_verifying_keys_batch(&public_keys);
        let salts = (0_u8..5)
            .map(|record| {
                let mut salt = [0_u8; TRANSPORT_SALT_BYTES];
                for (index, byte) in salt.iter_mut().enumerate() {
                    *byte = record.wrapping_mul(41).wrapping_add(index as u8);
                }
                salt
            })
            .collect::<Vec<_>>();
        // These lengths place, in turn, an hpk, a salt, an LE64 message
        // length, and a message across four successive 16-KiB boundaries.
        let message_lengths = [block - 152, block - 164, block - 136, block - 121, 17];
        let messages = message_lengths
            .into_iter()
            .enumerate()
            .map(|(record, length)| {
                (0..length)
                    .map(|index| (record as u8).wrapping_mul(53).wrapping_add(index as u8))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        (hashed_keys, salts, messages)
    }

    #[test]
    fn common_salt_domain_changes_the_first_challenge() {
        use crate::protocol::{
            instance::record as instance_record, manifest::CommitmentManifest,
            transcript::ProtocolTranscript, FalconDegree,
        };
        let hashed_keys = hashed_verifying_keys_batch(&vec![zero_public_key(); 5]);
        let salts = vec![[7; TRANSPORT_SALT_BYTES]; 5];
        let messages = vec![b"same records".to_vec(); 5];
        let seed = |mode| {
            let binding =
                public_records_binding_with_mode(&hashed_keys, &salts, &messages, mode).unwrap();
            let manifest = CommitmentManifest::for_batch_of(FalconDegree::Falcon512, 5).unwrap();
            let record = instance_record(&manifest, mode, &binding, [[0; 32]; 3]).unwrap();
            let mut transcript = ProtocolTranscript::new(&manifest, &record).unwrap();
            for group in manifest.groups() {
                transcript
                    .absorb_group_commitment(group.role, &[0; 128])
                    .unwrap();
            }
            transcript.derive_projection_seed().unwrap()
        };
        assert_ne!(seed(SaltMode::Common), seed(SaltMode::PerRecord));
    }

    #[test]
    fn binding_stream_handles_fields_crossing_block_boundaries() {
        let (hashed_keys, salts, messages) = crossing_fixture();
        let stream = PublicBindingStream::new(&hashed_keys, &salts, &messages)
            .expect("valid binding stream");
        let mut materialized = Vec::with_capacity(stream.encoded_len);
        materialized.extend_from_slice(&(hashed_keys.len() as u64).to_le_bytes());
        for ((hpk, salt), message) in hashed_keys.iter().zip(&salts).zip(&messages) {
            materialized.extend_from_slice(hpk.as_bytes());
            materialized.extend_from_slice(salt);
            materialized.extend_from_slice(&(message.len() as u64).to_le_bytes());
            materialized.extend_from_slice(message);
        }

        let streamed = (0..stream.block_count())
            .flat_map(|index| stream.block(index))
            .collect::<Vec<_>>();
        assert_eq!(streamed, materialized);

        let reference_blocks = materialized
            .chunks(PUBLIC_RECORDS_BINDING_BLOCK_BYTES)
            .collect::<Vec<_>>();
        // Independent RustCrypto cSHAKE, including the complete root framing.
        use sha3_legacy::{
            digest::{ExtendableOutput, Update, XofReader},
            CShake256, CShake256Core,
        };
        let hash = |input: &[u8], name: &[u8], custom: &[u8], out: &mut [u8]| {
            let mut h = CShake256::from_core(CShake256Core::new_with_function_name(name, custom));
            h.update(input);
            h.finalize_xof().read(out);
        };
        let actual_values = stream.chaining_values();
        let mut root_input = vec![2, 0x40, 0]; // left_encode(16384)
        for (block, actual) in reference_blocks.iter().zip(&actual_values) {
            let mut leaf = [0; 64];
            hash(block, b"", b"aerie/falcon/v2/record-leaf", &mut leaf);
            assert_eq!(actual.as_bytes(), &leaf);
            root_input.extend_from_slice(&leaf);
        }
        assert!(reference_blocks.len() < 256);
        root_input.extend_from_slice(&[u8::try_from(reference_blocks.len()).unwrap(), 1]);
        root_input.extend_from_slice(&[1, 0, 2]); // right_encode(256)
        let mut reference = [0; 32];
        hash(
            &root_input,
            b"ParallelHash",
            b"aerie/falcon/v2/public-records-binding",
            &mut reference,
        );
        assert_eq!(
            public_records_binding_from_hashed_keys(&hashed_keys, &salts, &messages)
                .expect("binding"),
            reference
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn binding_is_invariant_across_rayon_thread_counts() {
        let (hashed_keys, salts, messages) = crossing_fixture();
        let compute = || {
            public_records_binding_from_hashed_keys(&hashed_keys, &salts, &messages)
                .expect("binding")
        };
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread pool")
            .install(compute);
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("four-thread pool")
            .install(compute);
        assert_eq!(
            single,
            [
                127, 204, 179, 156, 162, 95, 67, 240, 76, 183, 208, 57, 220, 167, 103, 152, 176,
                90, 130, 130, 173, 189, 36, 127, 186, 241, 177, 61, 104, 70, 97, 76
            ]
        );
        assert_eq!(single, parallel);
    }
}
