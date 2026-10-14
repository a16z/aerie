//! Stable runtime dispatch for parallel Keccak-f1600 permutations.
//!
//! The `RustCrypto` `keccak` crate supplies the scalar fallback and the stable
//! `ARMv8` SHA3 backend. On x86-64, this crate adds stable `std::arch` AVX2 and
//! AVX-512 implementations which process four or eight independent states in
//! one thread. Architecture-specific unsafe code is confined to the x86-64 and
//! `AArch64` modules; callers see only safe batching and candidate-compaction
//! interfaces.

use std::sync::OnceLock;

pub use hybrid_array::{typenum::Unsigned, Array};
pub use keccak::{Backend, BackendClosure, ParState1600, State1600};

mod candidate_compaction;
pub use candidate_compaction::compact_centered_be_u16_5q;

/// SHAKE256's byte-oriented sponge rate.
pub const SHAKE256_RATE_BYTES: usize = 136;
/// Length of the protocol-neutral batched SHAKE256 output.
pub const SHAKE256_OUTPUT_64_BYTES: usize = 64;
/// Aerie's fixed public-record binding output length.
pub const PARALLEL_HASH256_OUTPUT_BYTES: usize = 32;

const SHAKE_SUFFIX: u8 = 0x1f;
const CSHAKE_SUFFIX: u8 = 0x04;
const PARALLEL_HASH_FUNCTION_NAME: &[u8] = b"ParallelHash";

/// One 512-bit block digest for the `ParallelHash256` root encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParallelHash256ChainingValue([u8; SHAKE256_OUTPUT_64_BYTES]);

impl ParallelHash256ChainingValue {
    /// Borrow the 512-bit chaining value.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; SHAKE256_OUTPUT_64_BYTES] {
        &self.0
    }
}

/// Absorbed states for independent SHAKE256 streams on one backend group.
///
/// The constructor accepts exactly the active backend width. Each call to
/// [`Self::squeeze_block`] advances every independent stream by one SHAKE256
/// rate block without creating threads.
pub struct Shake256StateBatch<B: Backend> {
    states: ParState1600<B>,
}

impl<B: Backend> Shake256StateBatch<B> {
    /// Absorb and finalize one independent SHAKE256 input per backend lane.
    #[must_use]
    pub fn new(inputs: &[&[u8]]) -> Self {
        assert_eq!(
            inputs.len(),
            B::ParSize1600::USIZE,
            "SHAKE256 batch must match the backend width"
        );
        Self {
            states: absorb_parallel::<B, _>(
                Array::<_, B::ParSize1600>::from_fn(|lane| SliceCursor(inputs[lane])),
                SHAKE_SUFFIX,
            ),
        }
    }

    /// Absorb and finalize one segmented SHAKE256 input per backend lane.
    ///
    /// Segments are logically concatenated without allocating or materializing
    /// the complete input. Full rate blocks common to every lane still use the
    /// data-parallel permutation; divergent full blocks use the backend's
    /// scalar permutation exactly as [`Self::new`] does.
    #[must_use]
    pub fn new_segmented<const SEGMENTS: usize>(inputs: &[[&[u8]; SEGMENTS]]) -> Self {
        assert_eq!(
            inputs.len(),
            B::ParSize1600::USIZE,
            "SHAKE256 batch must match the backend width"
        );
        Self {
            states: absorb_parallel::<B, _>(
                Array::<_, B::ParSize1600>::from_fn(|lane| SegmentedCursor::new(&inputs[lane])),
                SHAKE_SUFFIX,
            ),
        }
    }

    /// Squeeze the next SHAKE256 rate block from every lane.
    pub fn squeeze_block(&mut self) -> Array<[u8; SHAKE256_RATE_BYTES], B::ParSize1600> {
        B::get_par_f1600()(&mut self.states);
        Array::from_fn(|lane| rate_bytes(&self.states[lane]))
    }
}

/// Hash independent byte strings to 64-byte SHAKE256 outputs.
///
/// Full backend-width groups share one data-parallel Keccak-f1600 schedule.
/// A final partial group uses the selected backend's scalar permutation.
#[must_use]
pub fn shake256_64_batch(inputs: &[&[u8]]) -> Vec<[u8; SHAKE256_OUTPUT_64_BYTES]> {
    let mut outputs = vec![[0_u8; SHAKE256_OUTPUT_64_BYTES]; inputs.len()];
    shake256_64_batch_into(inputs, &mut outputs);
    outputs
}

/// Write independent 64-byte SHAKE256 outputs into an existing buffer.
/// The output must have one entry for each input. No hash state is retained.
pub fn shake256_64_batch_into(inputs: &[&[u8]], outputs: &mut [[u8; SHAKE256_OUTPUT_64_BYTES]]) {
    assert_eq!(inputs.len(), outputs.len(), "one SHAKE256 output per input");
    with_backend(Shake256Outputs64 { inputs, outputs });
}

struct Shake256Outputs64<'a, 'input> {
    inputs: &'a [&'input [u8]],
    outputs: &'a mut [[u8; SHAKE256_OUTPUT_64_BYTES]],
}

impl BackendClosure for Shake256Outputs64<'_, '_> {
    fn call_once<B: Backend>(self) {
        let width = B::ParSize1600::USIZE;
        let full_len = self.inputs.len() / width * width;
        for (inputs, outputs) in self.inputs[..full_len]
            .chunks_exact(width)
            .zip(self.outputs[..full_len].chunks_exact_mut(width))
        {
            shake256_64_backend::<B>(inputs, outputs);
        }
        for (input, output) in self.inputs[full_len..]
            .iter()
            .zip(&mut self.outputs[full_len..])
        {
            *output = shake256_64_scalar::<B>(input);
        }
    }
}

fn shake256_64_backend<B: Backend>(
    inputs: &[&[u8]],
    outputs: &mut [[u8; SHAKE256_OUTPUT_64_BYTES]],
) {
    debug_assert_eq!(inputs.len(), B::ParSize1600::USIZE);
    debug_assert_eq!(outputs.len(), B::ParSize1600::USIZE);
    let mut batch = Shake256StateBatch::<B>::new(inputs);
    let blocks = batch.squeeze_block();
    for (block, output) in blocks.iter().zip(outputs) {
        output.copy_from_slice(&block[..SHAKE256_OUTPUT_64_BYTES]);
    }
}

fn shake256_64_scalar<B: Backend>(input: &[u8]) -> [u8; SHAKE256_OUTPUT_64_BYTES] {
    let mut state = State1600::default();
    let permutation = B::get_f1600();
    let mut chunks = input.chunks_exact(SHAKE256_RATE_BYTES);
    for block in &mut chunks {
        xor_rate_bytes(&mut state, block);
        permutation(&mut state);
    }
    pad(&mut state, chunks.remainder(), SHAKE_SUFFIX);
    permutation(&mut state);
    let rate = rate_bytes(&state);
    rate[..SHAKE256_OUTPUT_64_BYTES]
        .try_into()
        .expect("fixed 64-byte SHAKE256 output")
}

/// Hash independent `ParallelHash256` input blocks to typed chaining values.
#[must_use]
pub fn parallel_hash256_chaining_values(blocks: &[&[u8]]) -> Vec<ParallelHash256ChainingValue> {
    shake256_64_batch(blocks)
        .into_iter()
        .map(ParallelHash256ChainingValue)
        .collect()
}

/// Hash tree leaves with a distinct cSHAKE256 function name and customization.
///
/// The customization must be nonempty; the function name may be empty.
/// Unlike the standardized `ParallelHash256` leaves, these hashes occupy a
/// different sponge domain from plain SHAKE256.
/// Combining them with [`parallel_hash256_root`] defines a customized tree hash,
/// not NIST `ParallelHash256`.
#[must_use]
pub fn customized_chaining_values(
    blocks: &[&[u8]],
    function_name: &[u8],
    customization: &[u8],
) -> Vec<ParallelHash256ChainingValue> {
    assert!(!customization.is_empty());
    let prefix = cshake_prefix(function_name, customization);
    let mut outputs = vec![[0; SHAKE256_OUTPUT_64_BYTES]; blocks.len()];
    with_backend(CustomizedOutputs64 {
        inputs: blocks,
        prefix: &prefix,
        outputs: &mut outputs,
    });
    outputs
        .into_iter()
        .map(ParallelHash256ChainingValue)
        .collect()
}

struct CustomizedOutputs64<'a, 'input> {
    inputs: &'a [&'input [u8]],
    prefix: &'a [u8],
    outputs: &'a mut [[u8; SHAKE256_OUTPUT_64_BYTES]],
}

impl BackendClosure for CustomizedOutputs64<'_, '_> {
    fn call_once<B: Backend>(self) {
        let width = B::ParSize1600::USIZE;
        for (inputs, outputs) in self
            .inputs
            .chunks(width)
            .zip(self.outputs.chunks_mut(width))
        {
            // Duplicate the last input only in unused lanes of the tail group.
            let segments = Array::<_, B::ParSize1600>::from_fn(|lane| {
                [self.prefix, inputs[lane.min(inputs.len() - 1)]]
            });
            let mut batch = Shake256StateBatch::<B> {
                states: absorb_parallel::<B, _>(
                    Array::from_fn(|lane| SegmentedCursor::new(&segments[lane])),
                    CSHAKE_SUFFIX,
                ),
            };
            let blocks = batch.squeeze_block();
            for (block, output) in blocks.iter().zip(outputs) {
                output.copy_from_slice(&block[..SHAKE256_OUTPUT_64_BYTES]);
            }
        }
    }
}

/// Finish `ParallelHash256` from its ordered 512-bit block chaining values.
///
/// This is the root encoding from NIST SP 800-185 section 6.3 with a fixed
/// 256-bit output:
/// `left_encode(B) || CV[0] || ... || CV[n-1] || right_encode(n) ||
/// right_encode(256)`, finalized by cSHAKE256 with function name
/// `ParallelHash` and the supplied customization string.
/// It implements standard `ParallelHash256` only when the chaining values
/// are produced by [`parallel_hash256_chaining_values`].
#[must_use]
pub fn parallel_hash256_root(
    chaining_values: &[ParallelHash256ChainingValue],
    block_size: usize,
    customization: &[u8],
) -> [u8; PARALLEL_HASH256_OUTPUT_BYTES] {
    parallel_hash256_root_bytes::<PARALLEL_HASH256_OUTPUT_BYTES>(
        chaining_values,
        block_size,
        customization,
    )
}

fn parallel_hash256_root_bytes<const OUTPUT_BYTES: usize>(
    chaining_values: &[ParallelHash256ChainingValue],
    block_size: usize,
    customization: &[u8],
) -> [u8; OUTPUT_BYTES] {
    assert!(block_size > 0, "ParallelHash256 block size must be nonzero");
    assert!(
        OUTPUT_BYTES <= SHAKE256_RATE_BYTES,
        "the one-block ParallelHash256 root supports at most 136 output bytes"
    );
    let mut encoded = Vec::with_capacity(9 + chaining_values.len() * SHAKE256_OUTPUT_64_BYTES + 18);
    append_left_encode(&mut encoded, block_size as u64);
    for chaining_value in chaining_values {
        encoded.extend_from_slice(chaining_value.as_bytes());
    }
    append_right_encode(&mut encoded, chaining_values.len() as u64);
    append_right_encode(&mut encoded, (OUTPUT_BYTES * 8) as u64);
    cshake256::<OUTPUT_BYTES>(&encoded, PARALLEL_HASH_FUNCTION_NAME, customization)
}

fn cshake256<const OUTPUT_BYTES: usize>(
    input: &[u8],
    function_name: &[u8],
    customization: &[u8],
) -> [u8; OUTPUT_BYTES] {
    debug_assert!(!function_name.is_empty() || !customization.is_empty());
    let mut framed = cshake_prefix(function_name, customization);
    framed.extend_from_slice(input);
    let mut output = [0_u8; OUTPUT_BYTES];
    with_backend(ScalarXof {
        input: &framed,
        suffix: CSHAKE_SUFFIX,
        output: &mut output,
    });
    output
}

fn cshake_prefix(function_name: &[u8], customization: &[u8]) -> Vec<u8> {
    let mut prefix = Vec::new();
    append_left_encode(&mut prefix, SHAKE256_RATE_BYTES as u64);
    append_encoded_string(&mut prefix, function_name);
    append_encoded_string(&mut prefix, customization);
    let padded = prefix.len().div_ceil(SHAKE256_RATE_BYTES) * SHAKE256_RATE_BYTES;
    prefix.resize(padded, 0);
    prefix
}

struct ScalarXof<'a, const OUTPUT_BYTES: usize> {
    input: &'a [u8],
    suffix: u8,
    output: &'a mut [u8; OUTPUT_BYTES],
}

impl<const OUTPUT_BYTES: usize> BackendClosure for ScalarXof<'_, OUTPUT_BYTES> {
    fn call_once<B: Backend>(self) {
        let mut state = State1600::default();
        let permutation = B::get_f1600();
        let mut chunks = self.input.chunks_exact(SHAKE256_RATE_BYTES);
        for block in &mut chunks {
            xor_rate_bytes(&mut state, block);
            permutation(&mut state);
        }
        pad(&mut state, chunks.remainder(), self.suffix);
        permutation(&mut state);
        let rate = rate_bytes(&state);
        self.output.copy_from_slice(&rate[..OUTPUT_BYTES]);
    }
}

fn append_encoded_string(output: &mut Vec<u8>, value: &[u8]) {
    append_left_encode(output, (value.len() as u64) * 8);
    output.extend_from_slice(value);
}

fn append_left_encode(output: &mut Vec<u8>, value: u64) {
    let encoded = value.to_be_bytes();
    let first = encoded
        .iter()
        .position(|&byte| byte != 0)
        .unwrap_or(encoded.len() - 1);
    let length = encoded.len() - first;
    output.push(u8::try_from(length).expect("u64 encoding length fits one byte"));
    output.extend_from_slice(&encoded[first..]);
}

fn append_right_encode(output: &mut Vec<u8>, value: u64) {
    let encoded = value.to_be_bytes();
    let first = encoded
        .iter()
        .position(|&byte| byte != 0)
        .unwrap_or(encoded.len() - 1);
    let length = encoded.len() - first;
    output.extend_from_slice(&encoded[first..]);
    output.push(u8::try_from(length).expect("u64 encoding length fits one byte"));
}

trait AbsorbCursor {
    fn remaining(&self) -> usize;
    fn absorb_next(&mut self, state: &mut State1600, length: usize);
}

struct SliceCursor<'a>(&'a [u8]);

impl AbsorbCursor for SliceCursor<'_> {
    fn remaining(&self) -> usize {
        self.0.len()
    }

    fn absorb_next(&mut self, state: &mut State1600, length: usize) {
        let (next, remaining) = self.0.split_at(length);
        xor_rate_bytes(state, next);
        self.0 = remaining;
    }
}

struct SegmentedCursor<'a, const SEGMENTS: usize> {
    segments: &'a [&'a [u8]; SEGMENTS],
    segment: usize,
    offset: usize,
    remaining: usize,
}

impl<'a, const SEGMENTS: usize> SegmentedCursor<'a, SEGMENTS> {
    fn new(segments: &'a [&'a [u8]; SEGMENTS]) -> Self {
        let remaining = segments
            .iter()
            .try_fold(0_usize, |total, segment| total.checked_add(segment.len()))
            .expect("segmented SHAKE256 input length overflow");
        Self {
            segments,
            segment: 0,
            offset: 0,
            remaining,
        }
    }
}

impl<const SEGMENTS: usize> AbsorbCursor for SegmentedCursor<'_, SEGMENTS> {
    fn remaining(&self) -> usize {
        self.remaining
    }

    fn absorb_next(&mut self, state: &mut State1600, length: usize) {
        assert!(
            length <= self.remaining,
            "segmented SHAKE256 input is too short"
        );
        let mut rate = [0_u8; SHAKE256_RATE_BYTES];
        let mut written = 0;
        while written < length {
            let segment = self.segments[self.segment];
            let available = &segment[self.offset..];
            let copied = available.len().min(length - written);
            rate[written..written + copied].copy_from_slice(&available[..copied]);
            written += copied;
            self.offset += copied;
            if self.offset == segment.len() {
                self.segment += 1;
                self.offset = 0;
            }
        }
        self.remaining -= length;
        xor_rate_bytes(state, &rate[..length]);
    }
}

fn absorb_parallel<B: Backend, C: AbsorbCursor>(
    mut inputs: Array<C, B::ParSize1600>,
    suffix: u8,
) -> ParState1600<B> {
    let mut states = ParState1600::<B>::default();
    let full_blocks =
        Array::<_, B::ParSize1600>::from_fn(|lane| inputs[lane].remaining() / SHAKE256_RATE_BYTES);
    let common_blocks = full_blocks.iter().copied().min().unwrap_or(0);

    for _ in 0..common_blocks {
        for (state, input) in states.iter_mut().zip(&mut inputs) {
            input.absorb_next(state, SHAKE256_RATE_BYTES);
        }
        B::get_par_f1600()(&mut states);
    }

    let scalar_permutation = B::get_f1600();
    for ((state, input), &block_count) in states.iter_mut().zip(&mut inputs).zip(&full_blocks) {
        for _ in common_blocks..block_count {
            input.absorb_next(state, SHAKE256_RATE_BYTES);
            scalar_permutation(state);
        }
        let tail_len = input.remaining();
        input.absorb_next(state, tail_len);
        pad_absorbed(state, tail_len, suffix);
    }
    states
}

fn pad(state: &mut State1600, tail: &[u8], suffix: u8) {
    debug_assert!(tail.len() < SHAKE256_RATE_BYTES);
    xor_rate_bytes(state, tail);
    pad_absorbed(state, tail.len(), suffix);
}

fn pad_absorbed(state: &mut State1600, position: usize, suffix: u8) {
    debug_assert!(position < SHAKE256_RATE_BYTES);
    state[position / 8] ^= u64::from(suffix) << (8 * (position % 8));
    state[SHAKE256_RATE_BYTES / 8 - 1] ^= 1_u64 << 63;
}

fn xor_rate_bytes(state: &mut State1600, bytes: &[u8]) {
    debug_assert!(bytes.len() <= SHAKE256_RATE_BYTES);
    for (word, chunk) in state.iter_mut().zip(bytes.chunks(8)) {
        let mut encoded = [0_u8; 8];
        encoded[..chunk.len()].copy_from_slice(chunk);
        *word ^= u64::from_le_bytes(encoded);
    }
}

fn rate_bytes(state: &State1600) -> [u8; SHAKE256_RATE_BYTES] {
    let mut bytes = [0_u8; SHAKE256_RATE_BYTES];
    for (chunk, word) in bytes.chunks_exact_mut(8).zip(state) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod x86;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SelectedBackend {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    Platform,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BackendSelection {
    backend: SelectedBackend,
    forced: bool,
}

/// Workload preference applied only when backend selection is automatic.
///
/// An explicit `AERIE_KECCAK_BACKEND` request always takes precedence so
/// forced-backend tests and benchmarks execute the backend they name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendPreference {
    /// Use the widest runtime-safe backend.
    Widest,
    /// Prefer a backend processing at most four independent states.
    AtMostFour,
}

#[inline]
fn backend_selection() -> BackendSelection {
    static SELECTED: OnceLock<BackendSelection> = OnceLock::new();
    *SELECTED.get_or_init(|| {
        let requested = std::env::var("AERIE_KECCAK_BACKEND").ok();
        assert!(
            matches!(
                requested.as_deref(),
                None | Some("auto" | "platform" | "avx2" | "avx512")
            ),
            "AERIE_KECCAK_BACKEND must be auto, platform, avx2, or avx512"
        );
        if requested.as_deref() == Some("platform") {
            return BackendSelection {
                backend: SelectedBackend::Platform,
                forced: true,
            };
        }
        #[cfg(target_arch = "x86_64")]
        {
            if requested.as_deref() == Some("avx512") {
                assert!(
                    std::is_x86_feature_detected!("avx512f"),
                    "AERIE_KECCAK_BACKEND=avx512 requires AVX-512F"
                );
                return BackendSelection {
                    backend: SelectedBackend::Avx512,
                    forced: true,
                };
            }
            if requested.as_deref() == Some("avx2") {
                assert!(
                    std::is_x86_feature_detected!("avx2"),
                    "AERIE_KECCAK_BACKEND=avx2 requires AVX2"
                );
                return BackendSelection {
                    backend: SelectedBackend::Avx2,
                    forced: true,
                };
            }
            if std::is_x86_feature_detected!("avx512f") {
                return BackendSelection {
                    backend: SelectedBackend::Avx512,
                    forced: false,
                };
            }
            if std::is_x86_feature_detected!("avx2") {
                return BackendSelection {
                    backend: SelectedBackend::Avx2,
                    forced: false,
                };
            }
        }
        assert!(
            requested.is_none() || requested.as_deref() == Some("auto"),
            "AERIE_KECCAK_BACKEND=avx2 or avx512 requires x86-64"
        );
        BackendSelection {
            backend: SelectedBackend::Platform,
            forced: false,
        }
    })
}

#[inline]
#[cfg(target_arch = "x86_64")]
fn selected_backend() -> SelectedBackend {
    selected_backend_for(BackendPreference::Widest)
}

#[inline]
fn selected_backend_for(preference: BackendPreference) -> SelectedBackend {
    let selection = backend_selection();
    if selection.forced || preference == BackendPreference::Widest {
        return selection.backend;
    }
    match selection.backend {
        #[cfg(target_arch = "x86_64")]
        SelectedBackend::Avx512 if std::is_x86_feature_detected!("avx2") => SelectedBackend::Avx2,
        #[cfg(target_arch = "x86_64")]
        SelectedBackend::Avx512 => SelectedBackend::Platform,
        backend => backend,
    }
}

/// Execute a closure with the fastest runtime-safe Keccak backend.
///
/// This performs data parallelism only: AVX2 handles four independent states
/// per permutation and AVX-512 handles eight, without creating threads.
/// `AERIE_KECCAK_BACKEND=platform|avx2|avx512` can pin a backend for testing;
/// on x86-64, `platform` is the scalar fallback. Unsupported forced backends
/// are rejected before executing an instruction.
#[inline]
pub fn with_backend(closure: impl BackendClosure) {
    with_backend_preference(BackendPreference::Widest, closure);
}

/// Execute a closure with the backend selected for one workload preference.
///
/// Preferences refine automatic selection only. Explicit `platform`, `avx2`,
/// and `avx512` environment requests always execute the requested backend.
#[inline]
pub fn with_backend_preference(preference: BackendPreference, closure: impl BackendClosure) {
    match selected_backend_for(preference) {
        #[cfg(target_arch = "x86_64")]
        SelectedBackend::Avx512 => closure.call_once::<x86::Avx512Backend>(),
        #[cfg(target_arch = "x86_64")]
        SelectedBackend::Avx2 => closure.call_once::<x86::Avx2Backend>(),
        SelectedBackend::Platform => keccak::Keccak::new().with_backend(closure),
    }
}

/// Name of the backend selected for this process.
#[must_use]
pub fn backend_name() -> &'static str {
    backend_name_for(BackendPreference::Widest)
}

/// Name of the effective backend for one workload preference.
#[must_use]
pub fn backend_name_for(preference: BackendPreference) -> &'static str {
    match selected_backend_for(preference) {
        #[cfg(target_arch = "x86_64")]
        SelectedBackend::Avx512 => "avx512",
        #[cfg(target_arch = "x86_64")]
        SelectedBackend::Avx2 => "avx2",
        SelectedBackend::Platform => "platform",
    }
}

/// Number of independent Keccak-f1600 states processed by the active backend.
#[must_use]
pub fn backend_parallel_width() -> usize {
    backend_parallel_width_for(BackendPreference::Widest)
}

/// Number of states processed for one workload preference.
#[must_use]
pub fn backend_parallel_width_for(preference: BackendPreference) -> usize {
    struct Width<'a>(&'a mut usize);

    impl BackendClosure for Width<'_> {
        fn call_once<B: Backend>(self) {
            *self.0 = B::ParSize1600::USIZE;
        }
    }

    let mut width = 0;
    with_backend_preference(preference, Width(&mut width));
    width
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha3_legacy::{
        digest::{
            ExtendableOutput as LegacyExtendableOutput, Update as LegacyUpdate,
            XofReader as LegacyXofReader,
        },
        CShake256, CShake256Core,
    };
    use shake::digest::{
        ExtendableOutput as ShakeExtendableOutput, Update as ShakeUpdate,
        XofReader as ShakeXofReader,
    };
    use shake::Shake256;

    #[test]
    fn selected_width_matches_backend_name() {
        let width = backend_parallel_width();
        match backend_name() {
            "avx512" => assert_eq!(width, 8),
            "avx2" => assert_eq!(width, 4),
            "platform" => assert!(width == 1 || width == 2),
            name => panic!("unknown backend {name}"),
        }
    }

    #[test]
    fn workload_preference_honors_forced_backends() {
        let preference = BackendPreference::AtMostFour;
        let effective_name = backend_name_for(preference);
        let effective_width = backend_parallel_width_for(preference);
        match std::env::var("AERIE_KECCAK_BACKEND").as_deref() {
            Ok("platform" | "avx2" | "avx512") => {
                assert_eq!(effective_name, backend_name());
                assert_eq!(effective_width, backend_parallel_width());
            }
            _ => assert!((1..=4).contains(&effective_width)),
        }
    }

    #[test]
    fn forced_backend_probe() {
        if std::env::var("AERIE_KECCAK_TEST_BACKEND_PROBE").as_deref() != Ok("1") {
            return;
        }
        let requested = std::env::var("AERIE_KECCAK_BACKEND")
            .expect("backend probe requires an explicit backend");
        assert_eq!(
            backend_name_for(BackendPreference::AtMostFour),
            requested,
            "a forced backend must override the automatic workload preference"
        );
        let effective_width = backend_parallel_width_for(BackendPreference::AtMostFour);
        assert_eq!(effective_width, backend_parallel_width());
        match requested.as_str() {
            "avx512" => assert_eq!(effective_width, 8),
            "avx2" => assert_eq!(effective_width, 4),
            "platform" => assert!(effective_width == 1 || effective_width == 2),
            backend => panic!("unexpected forced backend {backend}"),
        }
    }

    #[test]
    fn forced_backend_preferences_run_the_requested_kernel() {
        #[cfg(target_arch = "x86_64")]
        let backends = {
            let mut backends = vec!["platform"];
            if std::is_x86_feature_detected!("avx2") {
                backends.push("avx2");
            }
            if std::is_x86_feature_detected!("avx512f") {
                backends.push("avx512");
            }
            backends
        };
        #[cfg(not(target_arch = "x86_64"))]
        let backends = vec!["platform"];
        for backend in backends {
            let output = std::process::Command::new(
                std::env::current_exe().expect("current test executable"),
            )
            .args(["--exact", "tests::forced_backend_probe", "--nocapture"])
            .env("AERIE_KECCAK_BACKEND", backend)
            .env("AERIE_KECCAK_TEST_BACKEND_PROBE", "1")
            .output()
            .expect("run forced-backend probe");
            assert!(
                output.status.success(),
                "forced {backend} probe failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn centered_candidate_compaction_matches_modular_reference() {
        const MODULUS: u16 = 12_289;
        let mut encoded = Vec::with_capacity(2 * 65_536);
        for value in 0_u16..=u16::MAX {
            encoded.extend_from_slice(&value.to_be_bytes());
        }
        let mut actual = vec![0_i16; 65_536];
        let count = compact_centered_be_u16_5q::<MODULUS>(&encoded, &mut actual);
        let expected = (0..MODULUS * 5)
            .map(|value| {
                let residue = value % MODULUS;
                if residue > (MODULUS - 1) / 2 {
                    residue.cast_signed() - MODULUS.cast_signed()
                } else {
                    residue.cast_signed()
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(&actual[..count], expected);
    }

    #[test]
    fn centered_candidate_compaction_preserves_every_sparse_lane_mask() {
        const MODULUS: u16 = 12_289;
        const BOUND: u16 = MODULUS * 5;
        for mask in 0_u8..=u8::MAX {
            let masks = [mask, !mask, mask.rotate_left(3), mask ^ 0xa5];
            let mut encoded = Vec::with_capacity(64);
            let mut expected = Vec::with_capacity(32);
            for (group, group_mask) in masks.into_iter().enumerate() {
                for lane in 0..8 {
                    let accepted = group_mask & (1 << lane) != 0;
                    let value = if accepted {
                        u16::try_from((group * 8 + lane) * 1_937 % usize::from(BOUND))
                            .expect("test candidate is below the u16 bound")
                    } else {
                        BOUND + u16::try_from(group * 8 + lane).expect("test lane fits u16")
                    };
                    encoded.extend_from_slice(&value.to_be_bytes());
                    if accepted {
                        let residue = value % MODULUS;
                        expected.push(if residue > (MODULUS - 1) / 2 {
                            residue.cast_signed() - MODULUS.cast_signed()
                        } else {
                            residue.cast_signed()
                        });
                    }
                }
            }

            let mut actual = [0_i16; 32];
            let count = compact_centered_be_u16_5q::<MODULUS>(&encoded, &mut actual);
            assert_eq!(&actual[..count], expected, "mask {mask:#010b}");

            for capacity in [0, 1, 7, 8, 15, expected.len()] {
                let capacity = capacity.min(expected.len());
                let count =
                    compact_centered_be_u16_5q::<MODULUS>(&encoded, &mut actual[..capacity]);
                assert_eq!(count, capacity, "bounded mask {mask:#010b}");
                assert_eq!(
                    &actual[..count],
                    &expected[..capacity],
                    "bounded mask {mask:#010b}"
                );
            }
        }
    }

    #[test]
    fn centered_candidate_compaction_handles_scalar_vector_tail() {
        const MODULUS: u16 = 12_289;
        const BOUND: u16 = MODULUS * 5;
        let values = [
            0,
            6_144,
            6_145,
            12_288,
            12_289,
            18_433,
            24_578,
            30_722,
            36_867,
            49_156,
            61_444,
            BOUND,
            u16::MAX,
        ];
        let mut encoded = Vec::with_capacity(72);
        for lane in 0..36 {
            encoded.extend_from_slice(&values[lane % values.len()].to_be_bytes());
        }
        let expected = encoded
            .chunks_exact(2)
            .map(|word| u16::from_be_bytes([word[0], word[1]]))
            .filter(|&value| value < BOUND)
            .map(|value| {
                let residue = value % MODULUS;
                if residue > (MODULUS - 1) / 2 {
                    residue.cast_signed() - MODULUS.cast_signed()
                } else {
                    residue.cast_signed()
                }
            })
            .collect::<Vec<_>>();
        let mut actual = [0_i16; 36];
        let count = compact_centered_be_u16_5q::<MODULUS>(&encoded, &mut actual);
        assert_eq!(&actual[..count], expected);
    }

    #[test]
    fn chaining_values_match_scalar_shake256_across_rate_boundaries() {
        let lengths = [0, 1, 64, 135, 136, 137, 16_384, 16_385];
        let messages = lengths
            .map(|length| {
                (0_usize..length)
                    .map(|index| {
                        u8::try_from(index.wrapping_mul(131) & 0xff)
                            .expect("test byte is masked to eight bits")
                    })
                    .collect::<Vec<_>>()
            })
            .to_vec();
        let inputs = messages.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let actual = shake256_64_batch(&inputs);
        for (input, actual) in inputs.iter().zip(actual) {
            let mut hasher = Shake256::default();
            hasher.update(input);
            let mut expected = [0_u8; SHAKE256_OUTPUT_64_BYTES];
            hasher.finalize_xof().read(&mut expected);
            assert_eq!(actual, expected, "{}-byte input", input.len());
        }
    }

    #[test]
    fn full_parallel_groups_absorb_multiple_common_rate_blocks() {
        let width = backend_parallel_width();
        let messages = (0..width)
            .map(|lane| {
                (0..16_384 + lane)
                    .map(|index| {
                        u8::try_from(index & 0xff)
                            .expect("test index is masked to one byte")
                            .wrapping_mul(131)
                            .wrapping_add(
                                u8::try_from(lane).expect("backend width fits in one byte"),
                            )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let inputs = messages.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let actual = shake256_64_batch(&inputs);
        for (input, actual) in inputs.iter().zip(actual) {
            let mut hasher = Shake256::default();
            hasher.update(input);
            let mut expected = [0_u8; SHAKE256_OUTPUT_64_BYTES];
            hasher.finalize_xof().read(&mut expected);
            assert_eq!(actual, expected, "lane input length {}", input.len());
        }
    }

    #[test]
    fn customized_leaves_match_independent_cshake_at_block_and_lane_boundaries() {
        for count in [0, 1, 2, 3, 4, 7, 8, 9, 17] {
            let lengths = [0, 1, 135, 136, 137, 271, 272, 273, 16_384];
            let messages: Vec<_> = (0..count)
                .map(|i| vec![u8::try_from(i).unwrap(); lengths[i % lengths.len()]])
                .collect();
            let inputs: Vec<_> = messages.iter().map(Vec::as_slice).collect();
            for custom in [b"aerie/falcon/v2/record-leaf".as_slice(), &[0xa5; 300]] {
                let actual = customized_chaining_values(&inputs, b"", custom);
                for (input, value) in inputs.iter().zip(actual) {
                    let mut reference =
                        CShake256::from_core(CShake256Core::new_with_function_name(b"", custom));
                    reference.update(input);
                    let mut expected = [0; 64];
                    reference.finalize_xof().read(&mut expected);
                    assert_eq!(value.as_bytes(), &expected);
                }
            }
        }
    }

    #[test]
    fn customized_leaf_does_not_alias_a_signature_hash_with_identical_input() {
        // A record message can contain a complete raw FN-DSA hash input,
        // aligned to one tree leaf. A label on the root cannot separate it.
        let mut frame = vec![0x41; 16_384];
        frame[104..106].copy_from_slice(&[0, 0]); // raw mode, empty context
        let raw = shake256_64_batch(&[&frame])[0];
        let leaf = customized_chaining_values(&[&frame], b"", b"aerie/falcon/v2/record-leaf")[0];
        let projection =
            customized_chaining_values(&[&frame], b"", b"aerie/falcon/v2/projection")[0];
        let instance = customized_chaining_values(&[&frame], b"", b"aerie/falcon/v2/instance")[0];
        assert_ne!(&raw, leaf.as_bytes());
        assert_ne!(&raw, projection.as_bytes());
        assert_ne!(&raw, instance.as_bytes());
        assert_ne!(leaf, projection);
        assert_ne!(leaf, instance);
        assert_ne!(projection, instance);
    }

    #[test]
    fn parallel_hash256_root_matches_independent_cshake_reference() {
        let blocks = [b"first block".as_slice(), b"second block".as_slice()];
        let chaining_values = parallel_hash256_chaining_values(&blocks);
        let customization = b"aerie test binding";
        let actual = parallel_hash256_root(&chaining_values, 16_384, customization);

        let mut encoded = Vec::new();
        append_left_encode(&mut encoded, 16_384);
        for value in &chaining_values {
            encoded.extend_from_slice(value.as_bytes());
        }
        append_right_encode(&mut encoded, chaining_values.len() as u64);
        append_right_encode(&mut encoded, 256);
        let core =
            CShake256Core::new_with_function_name(PARALLEL_HASH_FUNCTION_NAME, customization);
        let mut reference = CShake256::from_core(core);
        reference.update(&encoded);
        let mut expected = [0_u8; PARALLEL_HASH256_OUTPUT_BYTES];
        reference.finalize_xof().read(&mut expected);
        assert_eq!(actual, expected);
    }

    #[test]
    fn parallel_hash256_matches_nist_sample_one() {
        let input = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
            0x16, 0x17, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
        ];
        let blocks = input.chunks(8).collect::<Vec<_>>();
        let chaining_values = parallel_hash256_chaining_values(&blocks);
        let actual = parallel_hash256_root_bytes::<64>(&chaining_values, 8, b"");
        let expected = [
            0xbc, 0x1e, 0xf1, 0x24, 0xda, 0x34, 0x49, 0x5e, 0x94, 0x8e, 0xad, 0x20, 0x7d, 0xd9,
            0x84, 0x22, 0x35, 0xda, 0x43, 0x2d, 0x2b, 0xbc, 0x54, 0xb4, 0xc1, 0x10, 0xe6, 0x4c,
            0x45, 0x11, 0x05, 0x53, 0x1b, 0x7f, 0x2a, 0x3e, 0x0c, 0xe0, 0x55, 0xc0, 0x28, 0x05,
            0xe7, 0xc2, 0xde, 0x1f, 0xb7, 0x46, 0xaf, 0x97, 0xa1, 0xdd, 0x01, 0xf4, 0x3b, 0x82,
            0x4e, 0x31, 0xb8, 0x76, 0x12, 0x41, 0x04, 0x29,
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn parallel_hash256_matches_nist_sample_two_with_customization() {
        let input = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
            0x16, 0x17, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
        ];
        let blocks = input.chunks(8).collect::<Vec<_>>();
        let chaining_values = parallel_hash256_chaining_values(&blocks);
        let actual = parallel_hash256_root_bytes::<64>(&chaining_values, 8, b"Parallel Data");
        let expected = [
            0xcd, 0xf1, 0x52, 0x89, 0xb5, 0x4f, 0x62, 0x12, 0xb4, 0xbc, 0x27, 0x05, 0x28, 0xb4,
            0x95, 0x26, 0x00, 0x6d, 0xd9, 0xb5, 0x4e, 0x2b, 0x6a, 0xdd, 0x1e, 0xf6, 0x90, 0x0d,
            0xda, 0x39, 0x63, 0xbb, 0x33, 0xa7, 0x24, 0x91, 0xf2, 0x36, 0x96, 0x9c, 0xa8, 0xaf,
            0xae, 0xa2, 0x9c, 0x68, 0x2d, 0x47, 0xa3, 0x93, 0xc0, 0x65, 0xb3, 0x8e, 0x29, 0xfa,
            0xe6, 0x51, 0xa2, 0x09, 0x1c, 0x83, 0x31, 0x10,
        ];
        assert_eq!(actual, expected);
    }
}
