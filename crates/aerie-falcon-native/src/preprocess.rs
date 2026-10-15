use aerie_keccak::{
    shake256_64_batch_with, Backend, BackendClosure, Shake256StateBatch, SHAKE256_RATE_BYTES,
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

/// Verifier-side `HashToPoint` from public inputs alone.
///
/// This computes the centered target from canonical public inputs, using the
/// public salt directly. It exists for the public-salt
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
#[inline]
pub fn hashed_verifying_keys_batch<P: FalconParams>(
    public_keys: &[CanonicalPublicKeyBytes<P>],
) -> Vec<HashedVerifyingKey> {
    let mut hashed = Vec::with_capacity(public_keys.len());
    shake256_64_batch_with(
        public_keys,
        |public_key| public_key.as_bytes().as_ref(),
        |_, digest| hashed.push(HashedVerifyingKey(*digest)),
    );
    hashed
}

/// Derive verifying-key hashes using both the active Rayon pool and the
/// active data-parallel Keccak backend.
///
/// The grain follows the same machine-adaptive policy as parallel
/// `HashToPoint`: it scales with the current worker count, remains aligned to
/// the selected SIMD width, and is capped so large batches expose enough work
/// to Rayon without creating one task per record. Each chunk writes its
/// hashes into place in the result.
#[cfg(feature = "parallel")]
#[must_use]
#[inline]
pub fn hashed_verifying_keys_batch_parallel<P: FalconParams>(
    public_keys: &[CanonicalPublicKeyBytes<P>],
) -> Vec<HashedVerifyingKey> {
    let grain = hash_to_point_parallel_grain(public_keys.len());
    let mut hashed = vec![HashedVerifyingKey([0; HASHED_VERIFYING_KEY_BYTES]); public_keys.len()];
    hashed
        .par_chunks_mut(grain)
        .zip(public_keys.par_chunks(grain))
        .for_each(|(hashed, public_keys)| {
            shake256_64_batch_with(
                public_keys,
                |public_key| public_key.as_bytes().as_ref(),
                |index, digest| hashed[index] = HashedVerifyingKey(*digest),
            );
        });
    hashed
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

#[derive(Clone, Copy, Debug)]
enum BatchSalts<'a> {
    PerRecord(&'a [[u8; FALCON_SALT_BYTES]]),
    Common(&'a [u8; FALCON_SALT_BYTES]),
}

impl<'a> BatchSalts<'a> {
    #[inline]
    fn get(self, index: usize) -> &'a [u8; FALCON_SALT_BYTES] {
        match self {
            Self::PerRecord(salts) => &salts[index],
            Self::Common(salt) => salt,
        }
    }

    #[inline]
    fn slice(self, range: core::ops::Range<usize>) -> Self {
        match self {
            Self::PerRecord(salts) => Self::PerRecord(&salts[range]),
            Self::Common(salt) => Self::Common(salt),
        }
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
    salts: BatchSalts<'a>,
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
            salts: BatchSalts::PerRecord(salts),
            messages,
        })
    }

    /// Borrow prehashed keys, messages, and one salt shared by every record.
    /// Targets equal those of a per-record batch with the salt repeated.
    pub fn new_common(
        hpks: &'a [HashedVerifyingKey],
        salt: &'a [u8; FALCON_SALT_BYTES],
        messages: &'a [M],
    ) -> Result<Self, &'static str> {
        if hpks.len() != messages.len() {
            return Err("prehashed HashToPoint batch shapes disagree");
        }
        Ok(Self {
            hpks,
            salts: BatchSalts::Common(salt),
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
            salts: self.salts.slice(range.clone()),
            messages: &self.messages[range],
        }
    }
}

/// Stream independently framed Falcon targets to a caller-owned consumer.
///
/// Each record runs on its own backend lane. A lane whose row is complete
/// takes the next record before the next permutation, so `visit` receives
/// every `(record, row)` pair exactly once, in an unspecified order. At most
/// one backend-width group of target rows exists at a time; the caller can
/// contract or otherwise consume a row immediately instead of allocating the
/// complete target table. Scratch is one reusable backend-width buffer, which
/// also keeps SIMD rows off Rayon’s comparatively small worker stacks.
pub fn hash_to_point_prehashed_batch_visit_of<P: FalconParams, M: AsRef<[u8]>>(
    inputs: PrehashedHashToPointBatch<'_, M>,
    mut visit: impl FnMut(usize, &P::Array<i16>),
) {
    const RAW_EMPTY_CONTEXT: &[u8] = &[0, 0];

    if inputs.is_empty() {
        return;
    }
    let width = aerie_keccak::backend_parallel_width();
    let mut outputs = vec![filled::<P, i16>(0); width];
    let mut counts = vec![0_usize; width];
    aerie_keccak::shake256_streams(
        inputs.len(),
        |record| {
            [
                inputs.salts.get(record).as_slice(),
                inputs.hpks[record].as_bytes().as_slice(),
                RAW_EMPTY_CONTEXT,
                inputs.messages[record].as_ref(),
            ]
        },
        |record, lane, words| {
            counts[lane] = aerie_keccak::compact_centered_rate_5q::<{ FALCON_Q as u16 }>(
                words,
                outputs[lane].as_mut(),
                counts[lane],
            );
            if counts[lane] < P::DEGREE {
                return true;
            }
            visit(record, &outputs[lane]);
            counts[lane] = 0;
            false
        },
    );
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
    hash_to_point_prehashed_lanes::<B, P>(&hpk_refs, &salts, &messages, &mut hpk_xof, outputs);
}

/// The single canonical batched implementation of Falcon raw-message framing
/// after `hpk` has been derived.
fn hash_to_point_prehashed_lanes<B: Backend, P: FalconParams>(
    hpks: &Array<&[u8], B::ParSize1600>,
    salts: &Array<&[u8], B::ParSize1600>,
    messages: &Array<&[u8], B::ParSize1600>,
    xof: &mut Shake256StateBatch<B>,
    outputs: &mut [P::Array<i16>],
) {
    let framed = Array::<_, B::ParSize1600>::from_fn(|lane| {
        [salts[lane], hpks[lane], &[0_u8, 0_u8], messages[lane]]
    });
    xof.absorb_segmented(&framed);
    squeeze_targets_parallel::<B, P>(xof, outputs);
}

fn squeeze_targets_parallel<B: Backend, P: FalconParams>(
    xof: &mut Shake256StateBatch<B>,
    outputs: &mut [P::Array<i16>],
) {
    let mut counts = Array::<usize, B::ParSize1600>::default();
    while counts.iter().any(|&count| count < P::DEGREE) {
        xof.permute();
        for (lane, (output, count)) in outputs.iter_mut().zip(&mut counts).enumerate() {
            if *count == P::DEGREE {
                continue;
            }
            *count = aerie_keccak::compact_centered_rate_5q::<{ FALCON_Q as u16 }>(
                xof.rate_words(lane),
                output.as_mut(),
                *count,
            );
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
    use crate::params::Falcon1024;
    use crate::params::Falcon512;

    fn zero_public_key<P: FalconParams>() -> CanonicalPublicKeyBytes<P> {
        let mut bytes = vec![0_u8; P::PUBLIC_KEY_BYTES];
        bytes[0] = P::PUBLIC_KEY_HEADER;
        CanonicalPublicKeyBytes::<P>::parse(&bytes).expect("canonical public key")
    }

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

    fn fixed_mode_framing_for<P: FalconParams>() {
        // The fixed mode must keep the pinned Falcon framing:
        // SHAKE256(salt || hpk || 0x00 || 0x00 || message), where 0x00 is the
        // raw-mode domain byte and 0x00 the length byte of the empty context.
        let message = b"framing pin";
        let public_key = zero_public_key::<P>();
        let salt = [0_u8; FALCON_SALT_BYTES];

        let hpk = HashedVerifyingKey::derive(&public_key);
        let mut hasher = Shake256::default();
        hasher.update(&salt);
        hasher.update(hpk.as_bytes());
        hasher.update(&[0x00, 0x00]);
        hasher.update(message);
        let mut reader = hasher.finalize_xof();
        let expected = sample_hash_words::<P>(|| {
            let mut word = [0_u8; 2];
            reader.read(&mut word);
            u16::from_be_bytes(word)
        });

        assert_eq!(
            hash_to_point_public_raw(&public_key, &salt, message),
            expected
        );
    }

    #[test]
    fn fixed_mode_framing_absorbs_salt_hpk_and_two_zero_bytes() {
        fixed_mode_framing_for::<Falcon512>();
        fixed_mode_framing_for::<Falcon1024>();
    }

    /// Collect streamed rows by record index, checking each record is visited
    /// exactly once.
    fn visited_rows<P: FalconParams>(
        batch: PrehashedHashToPointBatch<'_, Vec<u8>>,
    ) -> Vec<P::Array<i16>> {
        let mut rows = vec![None; batch.len()];
        hash_to_point_prehashed_batch_visit_of::<P, _>(batch, |index, row| {
            assert!(
                rows[index].replace(*row).is_none(),
                "record {index} visited twice"
            );
        });
        rows.into_iter()
            .enumerate()
            .map(|(index, row)| row.unwrap_or_else(|| panic!("record {index} not visited")))
            .collect()
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
        let public_key = zero_public_key::<P>();
        let inputs = messages
            .iter()
            .zip(&salts)
            .map(|(message, salt)| HashToPointInput::new(&public_key, salt, message))
            .collect::<Vec<_>>();

        let batched = hash_to_point_public_raw_batch(&inputs);
        let scalar = inputs
            .iter()
            .map(|input| hash_to_point_public_raw(input.public_key, input.salt, input.message))
            .collect::<Vec<_>>();
        assert_eq!(batched, scalar);

        let public_keys = vec![public_key.clone(); inputs.len()];
        let hashed_keys = hashed_verifying_keys_batch(&public_keys);
        assert!(hashed_keys
            .iter()
            .all(|hpk| *hpk == HashedVerifyingKey::derive(&public_key)));

        let prehashed_inputs = PrehashedHashToPointBatch::new(&hashed_keys, &salts, &messages)
            .expect("aligned fixture batch");
        assert_eq!(visited_rows::<P>(prehashed_inputs), scalar);

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

    fn common_salt_batch_matches_repeated_salt_batch_for<P: FalconParams>() {
        let width = aerie_keccak::backend_parallel_width();
        let salt = std::array::from_fn(|index| (index.wrapping_mul(17) & 0xff) as u8);
        let lengths = [0, 1, 29, 30, 31, 165, 166, 167, 301, 302, 303];
        for n in [0, 1, width - 1, width, width + 1, 37] {
            let hpks = (0..n)
                .map(|record| {
                    let mut bytes = vec![0_u8; P::PUBLIC_KEY_BYTES];
                    bytes[0] = P::PUBLIC_KEY_HEADER;
                    bytes[1] = record as u8;
                    let public_key =
                        CanonicalPublicKeyBytes::<P>::parse(&bytes).expect("canonical public key");
                    HashedVerifyingKey::derive(&public_key)
                })
                .collect::<Vec<_>>();
            let messages = (0..n)
                .map(|record| {
                    (0..lengths[record % lengths.len()])
                        .map(|index| (record.wrapping_mul(131).wrapping_add(index) & 0xff) as u8)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let salts = vec![salt; n];
            let common = PrehashedHashToPointBatch::new_common(&hpks, &salt, &messages)
                .expect("aligned common-salt fixture batch");
            let repeated = PrehashedHashToPointBatch::new(&hpks, &salts, &messages)
                .expect("aligned repeated-salt fixture batch");
            assert_eq!(common.len(), n);
            assert_eq!(common.is_empty(), n == 0);

            let visits = visited_rows::<P>;
            assert_eq!(visits(common), visits(repeated), "batch length {n}");

            // Start before the first width boundary when the batch spans it.
            let range = if n > width { width - 1..n } else { 0..n };
            let common_slice = common.slice(range.clone());
            assert!(core::ptr::eq(
                common_slice.salts.get(0),
                core::ptr::from_ref(&salt)
            ));
            assert_eq!(
                visits(common_slice),
                visits(repeated.slice(range)),
                "sliced batch length {n}"
            );

            if n > 0 {
                assert_eq!(
                    PrehashedHashToPointBatch::new_common(&hpks[..n - 1], &salt, &messages)
                        .unwrap_err(),
                    "prehashed HashToPoint batch shapes disagree"
                );
                assert_eq!(
                    PrehashedHashToPointBatch::new_common(&hpks, &salt, &messages[..n - 1])
                        .unwrap_err(),
                    "prehashed HashToPoint batch shapes disagree"
                );
            }
        }
    }

    #[test]
    fn common_salt_batch_matches_repeated_salt_batch() {
        common_salt_batch_matches_repeated_salt_batch_for::<Falcon512>();
        common_salt_batch_matches_repeated_salt_batch_for::<Falcon1024>();
    }

    fn key_batches_match_scalar_for<P: FalconParams>() {
        // An odd length leaves a partial SIMD group on every multi-lane backend.
        let public_keys = (0..129)
            .map(|index| {
                let mut bytes = vec![0_u8; P::PUBLIC_KEY_BYTES];
                bytes[0] = P::PUBLIC_KEY_HEADER;
                bytes[1] = index;
                CanonicalPublicKeyBytes::<P>::parse(&bytes).expect("canonical public key")
            })
            .collect::<Vec<_>>();
        let batch_hashes = hashed_verifying_keys_batch(&public_keys);
        let scalar_hashes = public_keys
            .iter()
            .map(HashedVerifyingKey::derive)
            .collect::<Vec<_>>();
        assert_eq!(batch_hashes, scalar_hashes);

        #[cfg(feature = "parallel")]
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("test thread pool");
            let parallel_hashes =
                pool.install(|| hashed_verifying_keys_batch_parallel(&public_keys));
            assert_eq!(parallel_hashes, scalar_hashes, "{threads} Rayon workers");
        }
    }

    #[test]
    fn key_batches_match_scalar_with_partial_groups() {
        key_batches_match_scalar_for::<Falcon512>();
        key_batches_match_scalar_for::<Falcon1024>();
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
