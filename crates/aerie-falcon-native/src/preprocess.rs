use core::marker::PhantomData;

use aerie_keccak::{
    shake256_64_batch, Backend, BackendClosure, Shake256StateBatch, SHAKE256_RATE_BYTES,
};
use hybrid_array::{typenum::Unsigned, Array};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use shake::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

use crate::{
    params::{filled, FalconParams, FALCON_Q, FALCON_SALT_BYTES},
    record::DirectRecord,
    source::CanonicalPublicKeyBytes,
};

const HASHED_VERIFYING_KEY_BYTES: usize = 64;
#[cfg(test)]
const HASH_TO_POINT_ACCEPT_BOUND: u16 = 61_445;

/// SHAKE256 digest derived only from canonical verifying-key bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HashedVerifyingKey([u8; HASHED_VERIFYING_KEY_BYTES]);

impl HashedVerifyingKey {
    /// Exact byte width of an FN-DSA hashed verifying key.
    pub const BYTE_LEN: usize = HASHED_VERIFYING_KEY_BYTES;

    #[must_use]
    pub fn derive<P: FalconParams>(public_key: &CanonicalPublicKeyBytes<P>) -> Self {
        let mut hasher = Shake256::default();
        hasher.update(public_key.as_bytes().as_ref());
        let mut reader = hasher.finalize_xof();
        let mut digest = [0_u8; Self::BYTE_LEN];
        reader.read(&mut digest);
        Self(digest)
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; Self::BYTE_LEN] {
        &self.0
    }
}

/// Verifier-local hash target bound to the source tuple that produced it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HashToPointTarget<P: FalconParams> {
    salt: [u8; FALCON_SALT_BYTES],
    message: Vec<u8>,
    t_centered: P::Array<i16>,
}

impl<P: FalconParams> HashToPointTarget<P> {
    /// Derive the Falcon target from one canonical direct record.
    ///
    /// The v0 protocol fixes raw message mode and the empty context inside the
    /// pinned Falcon framing, so the target depends only on the salt,
    /// the hashed verifying key, and the message.
    #[must_use]
    pub fn derive(record: &DirectRecord<P>) -> Self {
        let salt = record.signature().salt();
        let hpk = HashedVerifyingKey::derive(record.public_key());
        let t_centered = hash_raw_message::<P>(salt, &hpk, record.message());
        Self {
            salt: *salt,
            message: record.message().to_vec(),
            t_centered,
        }
    }

    #[must_use]
    pub const fn salt(&self) -> &[u8; FALCON_SALT_BYTES] {
        &self.salt
    }

    #[must_use]
    pub fn message(&self) -> &[u8] {
        &self.message
    }

    #[must_use]
    pub const fn t_centered(&self) -> &P::Array<i16> {
        &self.t_centered
    }
}

/// Verifier-side `HashToPoint` from public inputs alone.
///
/// This computes the same centered target that [`HashToPointTarget::derive`]
/// derives from a signed record, but takes the public salt directly instead of
/// reading it from a decoded signature. It exists for the public-salt
/// aggregation variant: Falcon salts are transmitted in the clear, so the
/// verifier recomputes the per-record target rather than trusting a committed
/// one.
#[must_use]
pub fn hash_to_point_public_raw<P: FalconParams>(
    public_key: &CanonicalPublicKeyBytes<P>,
    salt: &[u8; FALCON_SALT_BYTES],
    message: &[u8],
) -> P::Array<i16> {
    let hpk = HashedVerifyingKey::derive(public_key);
    hash_raw_message::<P>(salt, &hpk, message)
}

/// Derive FN-DSA's fixed 64-byte hashes of independent canonical verifying keys.
///
/// Full backend-width groups use the active data-parallel Keccak backend. The
/// returned values are exactly the `hpk` values consumed by `HashToPoint`.
#[must_use]
pub fn hashed_verifying_keys_batch<P: FalconParams>(
    public_keys: &[CanonicalPublicKeyBytes<P>],
) -> Vec<HashedVerifyingKey> {
    let inputs = public_keys
        .iter()
        .map(|public_key| public_key.as_bytes().as_ref())
        .collect::<Vec<_>>();
    shake256_64_batch(&inputs)
        .into_iter()
        .map(HashedVerifyingKey)
        .collect()
}

/// Derive verifying-key hashes using both the active Rayon pool and the
/// active data-parallel Keccak backend.
///
/// The grain follows the same machine-adaptive policy as parallel
/// `HashToPoint`: it scales with the current worker count, remains aligned to
/// the selected SIMD width, and is capped so large batches expose enough work
/// to Rayon without creating one task per record.
#[cfg(feature = "parallel")]
#[must_use]
pub fn hashed_verifying_keys_batch_parallel<P: FalconParams>(
    public_keys: &[CanonicalPublicKeyBytes<P>],
) -> Vec<HashedVerifyingKey> {
    let grain = hash_to_point_parallel_grain(public_keys.len());
    public_keys
        .par_chunks(grain)
        .flat_map_iter(hashed_verifying_keys_batch::<P>)
        .collect()
}

/// Public inputs for one independently framed Falcon `HashToPoint` stream.
#[derive(Clone, Copy, Debug)]
pub struct HashToPointInput<'a, P: FalconParams> {
    public_key: &'a CanonicalPublicKeyBytes<P>,
    salt: &'a [u8; FALCON_SALT_BYTES],
    message: &'a [u8],
}

impl<'a, P: FalconParams> HashToPointInput<'a, P> {
    #[must_use]
    pub const fn new(
        public_key: &'a CanonicalPublicKeyBytes<P>,
        salt: &'a [u8; FALCON_SALT_BYTES],
        message: &'a [u8],
    ) -> Self {
        Self {
            public_key,
            salt,
            message,
        }
    }

    #[must_use]
    pub const fn public_key(&self) -> &'a CanonicalPublicKeyBytes<P> {
        self.public_key
    }

    #[must_use]
    pub const fn salt(&self) -> &'a [u8; FALCON_SALT_BYTES] {
        self.salt
    }

    #[must_use]
    pub const fn message(&self) -> &'a [u8] {
        self.message
    }
}

/// Borrowed verifier inputs for a batch of Falcon `HashToPoint` streams after
/// the canonical verifying keys have already been hashed.
///
/// Public-record binding and target derivation share these prehashed keys.
/// A consumer can stream target rows or retain them when its transcript binds
/// the public tables before the first challenge. The batch does not name a
/// parameter set; the consumer selects it when it derives the targets.
#[derive(Debug)]
pub struct PrehashedHashToPointBatch<'a, M = Vec<u8>> {
    hpks: &'a [HashedVerifyingKey],
    salts: &'a [[u8; FALCON_SALT_BYTES]],
    messages: &'a [M],
}

impl<M> Copy for PrehashedHashToPointBatch<'_, M> {}

impl<M> Clone for PrehashedHashToPointBatch<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, M> PrehashedHashToPointBatch<'a, M> {
    /// Construct an aligned batch view without allocating per-record adapters.
    pub fn new(
        hpks: &'a [HashedVerifyingKey],
        salts: &'a [[u8; FALCON_SALT_BYTES]],
        messages: &'a [M],
    ) -> Result<Self, &'static str> {
        if hpks.len() != salts.len() || hpks.len() != messages.len() {
            return Err("prehashed HashToPoint batch shapes disagree");
        }
        Ok(Self {
            hpks,
            salts,
            messages,
        })
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.hpks.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.hpks.is_empty()
    }

    #[must_use]
    pub fn slice(&self, range: core::ops::Range<usize>) -> Self {
        Self {
            hpks: &self.hpks[range.clone()],
            salts: &self.salts[range.clone()],
            messages: &self.messages[range],
        }
    }
}

/// Stream independently framed Falcon targets to a caller-owned consumer.
///
/// Full backend-width groups are sampled together, then each completed row is
/// passed to `visit` in input order. At most one backend-width group of target
/// rows exists at a time; the caller can contract or otherwise consume a row
/// immediately instead of allocating the complete target table. Scratch is
/// one reusable backend-width buffer, which also keeps SIMD rows off Rayon’s
/// comparatively small worker stacks.
pub fn hash_to_point_prehashed_batch_visit_of<P: FalconParams, M: AsRef<[u8]>>(
    inputs: PrehashedHashToPointBatch<'_, M>,
    visit: impl FnMut(usize, &P::Array<i16>),
) {
    if inputs.is_empty() {
        return;
    }
    aerie_keccak::with_backend(HashToPointPrehashedBatch::<P, M, _> {
        inputs,
        visit,
        params: PhantomData,
    });
}

/// Derive independent Falcon targets using the active Keccak backend's native
/// data-parallel width.
///
/// Each lane remains a separate SHAKE256 transcript with the standardized
/// Falcon framing. Batching changes only how independent Keccak-f permutations
/// are scheduled: one lane on the portable scalar backend, two lanes on the
/// stable `ARMv8` SHA3 backend, four lanes on AVX2, and eight lanes on AVX-512.
/// Runtime feature detection selects the backend in one stable binary. A final
/// partial group uses the scalar public path.
#[must_use]
pub fn hash_to_point_public_raw_batch<P: FalconParams>(
    inputs: &[HashToPointInput<'_, P>],
) -> Vec<P::Array<i16>> {
    let mut outputs = vec![filled::<P, i16>(0); inputs.len()];
    aerie_keccak::with_backend(HashToPointBatch {
        inputs,
        outputs: &mut outputs,
    });
    outputs
}

#[cfg(feature = "parallel")]
/// Machine-adaptive, backend-aligned record grain for parallel `HashToPoint`
/// work. Consumers that fuse work into `HashToPoint` should reuse this plan so
/// worker-count and SIMD-width tuning has one source of truth.
#[must_use]
pub fn hash_to_point_parallel_grain(records: usize) -> usize {
    const TASKS_PER_WORKER: usize = 8;
    const MAX_GRAIN: usize = 64;

    let width = aerie_keccak::backend_parallel_width();
    let target_tasks = rayon::current_num_threads()
        .max(1)
        .saturating_mul(TASKS_PER_WORKER)
        .max(1);
    let grain = records.div_ceil(target_tasks).clamp(width, MAX_GRAIN);
    grain.div_ceil(width) * width
}

struct HashToPointBatch<'a, 'input, P: FalconParams> {
    inputs: &'a [HashToPointInput<'input, P>],
    outputs: &'a mut [P::Array<i16>],
}

struct HashToPointPrehashedBatch<'a, P, M, V> {
    inputs: PrehashedHashToPointBatch<'a, M>,
    visit: V,
    params: PhantomData<P>,
}

impl<P, M, V> BackendClosure for HashToPointPrehashedBatch<'_, P, M, V>
where
    P: FalconParams,
    M: AsRef<[u8]>,
    V: FnMut(usize, &P::Array<i16>),
{
    fn call_once<B: Backend>(mut self) {
        let width = B::ParSize1600::USIZE;
        let full_len = self.inputs.len() / width * width;
        let mut outputs = vec![filled::<P, i16>(0); width];
        for start in (0..full_len).step_by(width) {
            hash_to_point_prehashed_backend::<B, P, _>(
                self.inputs.slice(start..start + width),
                &mut outputs,
            );
            for (lane, output) in outputs.iter().enumerate() {
                (self.visit)(start + lane, output);
            }
        }
        for tail in full_len..self.inputs.len() {
            let output = hash_raw_message::<P>(
                &self.inputs.salts[tail],
                &self.inputs.hpks[tail],
                self.inputs.messages[tail].as_ref(),
            );
            (self.visit)(tail, &output);
        }
    }
}

impl<P: FalconParams> BackendClosure for HashToPointBatch<'_, '_, P> {
    fn call_once<B: Backend>(self) {
        let width = B::ParSize1600::USIZE;
        let full_len = self.inputs.len() / width * width;
        for start in (0..full_len).step_by(width) {
            hash_to_point_backend::<B, P>(
                &self.inputs[start..start + width],
                &mut self.outputs[start..start + width],
            );
        }
        for (input, output) in self.inputs[full_len..]
            .iter()
            .zip(&mut self.outputs[full_len..])
        {
            *output = hash_to_point_public_raw(input.public_key, input.salt, input.message);
        }
    }
}

fn hash_to_point_backend<B: Backend, P: FalconParams>(
    inputs: &[HashToPointInput<'_, P>],
    outputs: &mut [P::Array<i16>],
) {
    debug_assert_eq!(inputs.len(), B::ParSize1600::USIZE);
    debug_assert_eq!(outputs.len(), B::ParSize1600::USIZE);

    let public_keys =
        Array::<_, B::ParSize1600>::from_fn(|lane| inputs[lane].public_key.as_bytes().as_ref());
    let mut hpk_xof = Shake256StateBatch::<B>::new(&public_keys);
    let hpk_blocks = hpk_xof.squeeze_block();
    let hpks = Array::<_, B::ParSize1600>::from_fn(|lane| {
        <[u8; HashedVerifyingKey::BYTE_LEN]>::try_from(
            &hpk_blocks[lane][..HashedVerifyingKey::BYTE_LEN],
        )
        .expect("fixed SHAKE256 digest length")
    });

    let salts = Array::<_, B::ParSize1600>::from_fn(|lane| inputs[lane].salt.as_slice());
    let messages = Array::<_, B::ParSize1600>::from_fn(|lane| inputs[lane].message);
    let hpk_refs = Array::<_, B::ParSize1600>::from_fn(|lane| hpks[lane].as_slice());
    hash_to_point_prehashed_lanes::<B, P>(&hpk_refs, &salts, &messages, outputs);
}

fn hash_to_point_prehashed_backend<B: Backend, P: FalconParams, M: AsRef<[u8]>>(
    inputs: PrehashedHashToPointBatch<'_, M>,
    outputs: &mut [P::Array<i16>],
) {
    debug_assert_eq!(inputs.len(), B::ParSize1600::USIZE);
    debug_assert_eq!(outputs.len(), B::ParSize1600::USIZE);
    let hpks = Array::<_, B::ParSize1600>::from_fn(|lane| inputs.hpks[lane].as_bytes().as_slice());
    let salts = Array::<_, B::ParSize1600>::from_fn(|lane| inputs.salts[lane].as_slice());
    let messages = Array::<_, B::ParSize1600>::from_fn(|lane| inputs.messages[lane].as_ref());
    hash_to_point_prehashed_lanes::<B, P>(&hpks, &salts, &messages, outputs);
}

/// The single canonical batched implementation of Falcon raw-message framing
/// after `hpk` has been derived.
fn hash_to_point_prehashed_lanes<B: Backend, P: FalconParams>(
    hpks: &Array<&[u8], B::ParSize1600>,
    salts: &Array<&[u8], B::ParSize1600>,
    messages: &Array<&[u8], B::ParSize1600>,
    outputs: &mut [P::Array<i16>],
) {
    let framed = Array::<_, B::ParSize1600>::from_fn(|lane| {
        [salts[lane], hpks[lane], &[0_u8, 0_u8], messages[lane]]
    });
    let mut target_xof = Shake256StateBatch::<B>::new_segmented(&framed);
    squeeze_targets_parallel::<B, P>(&mut target_xof, outputs);
}

fn squeeze_targets_parallel<B: Backend, P: FalconParams>(
    xof: &mut Shake256StateBatch<B>,
    outputs: &mut [P::Array<i16>],
) {
    let mut counts = Array::<usize, B::ParSize1600>::default();
    while counts.iter().any(|&count| count < P::DEGREE) {
        let blocks = xof.squeeze_block();
        for ((block, output), count) in blocks.iter().zip(outputs.iter_mut()).zip(&mut counts) {
            if *count == P::DEGREE {
                continue;
            }
            sample_candidate_block::<P>(block, output, count);
        }
    }
}

/// Falcon `HashToPoint` framing used by the pinned implementation.
///
/// The absorbed bytes are `salt || hpk || 0x00 || 0x00 || message`: the raw
/// message-mode domain byte followed by the zero context-length byte of the
/// fixed empty context, exactly as the pinned implementation frames a raw mode
/// empty context input. The parameter set only changes how many coefficients
/// are squeezed.
fn hash_raw_message<P: FalconParams>(
    salt: &[u8; FALCON_SALT_BYTES],
    hpk: &HashedVerifyingKey,
    message: &[u8],
) -> P::Array<i16> {
    let mut hasher = Shake256::default();
    hasher.update(salt);
    hasher.update(hpk.as_bytes());
    hasher.update(&[0]);
    hasher.update(&[0]);
    hasher.update(message);
    let mut reader = hasher.finalize_xof();
    sample_hash_blocks::<P>(|| {
        let mut block = [0_u8; SHAKE256_RATE_BYTES];
        reader.read(&mut block);
        block
    })
}

fn sample_hash_blocks<P: FalconParams>(
    mut next_block: impl FnMut() -> [u8; SHAKE256_RATE_BYTES],
) -> P::Array<i16> {
    let mut output = filled::<P, i16>(0);
    let mut count = 0;
    while count < P::DEGREE {
        sample_candidate_block::<P>(&next_block(), &mut output, &mut count);
    }
    output
}

fn sample_candidate_block<P: FalconParams>(
    block: &[u8; SHAKE256_RATE_BYTES],
    output: &mut P::Array<i16>,
    count: &mut usize,
) {
    *count += aerie_keccak::compact_centered_be_u16_5q::<{ FALCON_Q as u16 }>(
        block,
        &mut output.as_mut()[*count..],
    );
}

#[cfg(test)]
fn sample_hash_words<P: FalconParams>(mut next_word: impl FnMut() -> u16) -> P::Array<i16> {
    let mut output = filled::<P, i16>(0);
    let mut count = 0;
    while count < P::DEGREE {
        let word = next_word();
        if word >= HASH_TO_POINT_ACCEPT_BOUND {
            continue;
        }
        output.as_mut()[count] = center_residue(word % FALCON_Q as u16);
        count += 1;
    }
    output
}

#[cfg(test)]
const fn center_residue(residue: u16) -> i16 {
    let signed = residue.cast_signed();
    if residue > (FALCON_Q as u16 - 1) / 2 {
        signed - 12_289
    } else {
        signed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Falcon512;
    use crate::{params::Falcon1024, record::test_records::zero_record};

    fn sampling_pins_big_endian_rejection_boundary_for<P: FalconParams>() {
        let mut words = vec![0_u16; P::DEGREE + 1];
        words[0] = 61_444;
        words[1] = 61_445;
        let mut words = words.into_iter();
        let sampled = sample_hash_words::<P>(|| words.next().expect("enough words"));
        assert_eq!(sampled.as_ref()[0], -1);
        assert_eq!(sampled.as_ref()[1], 0);
    }

    #[test]
    fn sampling_pins_big_endian_rejection_boundary() {
        sampling_pins_big_endian_rejection_boundary_for::<Falcon512>();
        sampling_pins_big_endian_rejection_boundary_for::<Falcon1024>();
    }

    fn sampling_uses_extra_blocks_after_interspersed_rejections_for<P: FalconParams>() {
        // Every block carries 68 candidates. Blocks that reject every eighth
        // candidate (indices 0, 8, ..., 64) accept 59; the final block accepts
        // all 68. Use the smallest count of lossy blocks whose accepted total
        // still needs the final block:
        let candidates = SHAKE256_RATE_BYTES / 2;
        let lossy_accepts = candidates - candidates.div_ceil(8);
        let lossy_blocks = (P::DEGREE - candidates).div_ceil(lossy_accepts);
        assert!(lossy_blocks * lossy_accepts < P::DEGREE);
        let blocks = (0..=lossy_blocks)
            .map(|block| {
                let mut encoded = [0_u8; SHAKE256_RATE_BYTES];
                for candidate in 0..candidates {
                    let value = if block < lossy_blocks && candidate % 8 == 0 {
                        HASH_TO_POINT_ACCEPT_BOUND
                    } else {
                        1
                    };
                    encoded[2 * candidate..2 * candidate + 2].copy_from_slice(&value.to_be_bytes());
                }
                encoded
            })
            .collect::<Vec<_>>();
        let mut consumed = 0;
        let sampled = sample_hash_blocks::<P>(|| {
            let block = blocks[consumed];
            consumed += 1;
            block
        });
        assert_eq!(consumed, lossy_blocks + 1);
        assert!(sampled.as_ref().iter().all(|&value| value == 1));
    }

    #[test]
    fn sampling_uses_extra_blocks_after_interspersed_rejections() {
        // Nine blocks for Falcon-512, eighteen for Falcon-1024.
        sampling_uses_extra_blocks_after_interspersed_rejections_for::<Falcon512>();
        sampling_uses_extra_blocks_after_interspersed_rejections_for::<Falcon1024>();
    }

    #[test]
    fn hash_target_binds_message() {
        let first = HashToPointTarget::<Falcon512>::derive(&zero_record(b"message"));
        let changed_message = HashToPointTarget::<Falcon512>::derive(&zero_record(b"message2"));
        assert_ne!(first.t_centered, changed_message.t_centered);
        let wide = HashToPointTarget::<Falcon1024>::derive(&zero_record(b"message"));
        assert_eq!(wide.t_centered().as_ref().len(), 1024);
        // Same framing, longer squeeze: the first 512 coefficients agree
        // whenever the keys hash identically, which the zero keys do not.
        assert_ne!(
            HashedVerifyingKey::derive(zero_record::<Falcon512>(b"").public_key()),
            HashedVerifyingKey::derive(zero_record::<Falcon1024>(b"").public_key())
        );
    }

    fn fixed_mode_framing_for<P: FalconParams>() {
        // The fixed mode must keep the pinned Falcon framing:
        // SHAKE256(salt || hpk || 0x00 || 0x00 || message), where 0x00 is the
        // raw-mode domain byte and 0x00 the length byte of the empty context.
        let message = b"framing pin";
        let record = zero_record::<P>(message);
        let target = HashToPointTarget::<P>::derive(&record);

        let hpk = HashedVerifyingKey::derive(record.public_key());
        let mut hasher = Shake256::default();
        hasher.update(record.signature().salt());
        hasher.update(hpk.as_bytes());
        hasher.update(&[0x00, 0x00]);
        hasher.update(message);
        let mut reader = hasher.finalize_xof();
        let expected = sample_hash_words::<P>(|| {
            let mut word = [0_u8; 2];
            reader.read(&mut word);
            u16::from_be_bytes(word)
        });

        assert_eq!(target.t_centered(), &expected);
        assert_eq!(
            hash_to_point_public_raw(record.public_key(), record.signature().salt(), message),
            expected
        );
    }

    #[test]
    fn fixed_mode_framing_absorbs_salt_hpk_and_two_zero_bytes() {
        fixed_mode_framing_for::<Falcon512>();
        fixed_mode_framing_for::<Falcon1024>();
    }

    fn batch_matches_scalar_for<P: FalconParams>() {
        // The frame before `message` is 106 bytes. These message lengths put
        // the final byte immediately before, on, and after one or more
        // SHAKE256 rate boundaries, while the odd batch length exercises a
        // partial group on every backend wider than one lane.
        let lengths = [0, 1, 29, 30, 31, 165, 166, 167, 301, 302, 303];
        let messages = lengths
            .iter()
            .enumerate()
            .map(|(record, &length)| {
                (0..length)
                    .map(|index| (record.wrapping_mul(131).wrapping_add(index) & 0xff) as u8)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let salts = lengths
            .iter()
            .enumerate()
            .map(|(record, _)| {
                std::array::from_fn(|index| {
                    (record.wrapping_mul(17).wrapping_add(index) & 0xff) as u8
                })
            })
            .collect::<Vec<[u8; FALCON_SALT_BYTES]>>();
        let record = zero_record::<P>(b"");
        let inputs = messages
            .iter()
            .zip(&salts)
            .map(|(message, salt)| HashToPointInput::new(record.public_key(), salt, message))
            .collect::<Vec<_>>();

        let batched = hash_to_point_public_raw_batch(&inputs);
        let scalar = inputs
            .iter()
            .map(|input| hash_to_point_public_raw(input.public_key, input.salt, input.message))
            .collect::<Vec<_>>();
        assert_eq!(batched, scalar);

        let public_keys = vec![record.public_key().clone(); inputs.len()];
        let hashed_keys = hashed_verifying_keys_batch(&public_keys);
        assert!(hashed_keys
            .iter()
            .all(|hpk| *hpk == HashedVerifyingKey::derive(record.public_key())));

        let prehashed_inputs = PrehashedHashToPointBatch::new(&hashed_keys, &salts, &messages)
            .expect("aligned fixture batch");
        let mut streamed = Vec::with_capacity(inputs.len());
        hash_to_point_prehashed_batch_visit_of::<P, _>(prehashed_inputs, |index, row| {
            assert_eq!(index, streamed.len());
            streamed.push(*row);
        });
        assert_eq!(streamed, scalar);

        #[cfg(feature = "parallel")]
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("test thread pool");
            let parallel_hashed_keys =
                pool.install(|| hashed_verifying_keys_batch_parallel(&public_keys));
            assert_eq!(parallel_hashed_keys, hashed_keys, "{threads} Rayon workers");
        }
    }

    #[test]
    fn batch_matches_scalar_across_rate_boundaries_and_partial_groups() {
        batch_matches_scalar_for::<Falcon512>();
        batch_matches_scalar_for::<Falcon1024>();
    }

    #[test]
    fn empty_batch_is_empty() {
        assert!(hash_to_point_public_raw_batch::<Falcon512>(&[]).is_empty());
        let mut visits = 0;
        let hpks = Vec::new();
        let salts = Vec::new();
        let messages = Vec::<Vec<u8>>::new();
        let prehashed = PrehashedHashToPointBatch::new(&hpks, &salts, &messages).unwrap();
        hash_to_point_prehashed_batch_visit_of::<Falcon1024, _>(prehashed, |_, _| visits += 1);
        assert_eq!(visits, 0);
        assert!(hashed_verifying_keys_batch::<Falcon1024>(&[]).is_empty());
        #[cfg(feature = "parallel")]
        assert!(hashed_verifying_keys_batch_parallel::<Falcon512>(&[]).is_empty());
    }
}
