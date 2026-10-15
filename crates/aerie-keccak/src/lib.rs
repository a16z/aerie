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
pub use candidate_compaction::{compact_centered_be_u16_5q, compact_centered_rate_5q, RATE_WORDS};

/// SHAKE256's byte-oriented sponge rate.
pub const SHAKE256_RATE_BYTES: usize = 136;
/// Length of the protocol-neutral batched SHAKE256 output.
pub const SHAKE256_OUTPUT_64_BYTES: usize = 64;
/// Aerie's fixed public-record binding output length.
pub const PARALLEL_HASH256_OUTPUT_BYTES: usize = 32;

const SHAKE_SUFFIX: u8 = 0x1f;
const CSHAKE_SUFFIX: u8 = 0x04;
const PARALLEL_HASH_FUNCTION_NAME: &[u8] = b"ParallelHash";

/// Scalar streaming cSHAKE256 with a nonempty customization string.
///
/// Cloning an absorbed state lets streams that share a prefix absorb it once.
#[derive(Clone)]
pub struct CShake256 {
    state: State1600,
    position: usize,
    keccak: keccak::Keccak,
}

impl CShake256 {
    /// Initialize cSHAKE256 with a function name and nonempty customization.
    #[must_use]
    pub fn new(function_name: &[u8], customization: &[u8]) -> Self {
        assert!(!customization.is_empty());
        Self::from_prefix(function_name, customization)
    }

    fn from_prefix(function_name: &[u8], customization: &[u8]) -> Self {
        debug_assert!(!function_name.is_empty() || !customization.is_empty());
        let mut hasher = Self {
            state: State1600::default(),
            position: 0,
            keccak: keccak::Keccak::new(),
        };
        cshake_prefix(function_name, customization, |bytes| hasher.update(bytes));
        hasher
    }

    /// Absorb the next input segment.
    #[inline]
    pub fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            let length = input.len().min(SHAKE256_RATE_BYTES - self.position);
            xor_rate_bytes_at(&mut self.state, self.position, &input[..length]);
            self.position += length;
            input = &input[length..];
            if self.position == SHAKE256_RATE_BYTES {
                self.keccak
                    .with_f1600(|permutation| permutation(&mut self.state));
                self.position = 0;
            }
        }
    }

    /// Finalize the input and obtain a reader for the cSHAKE256 output stream.
    #[inline]
    #[must_use]
    pub fn finalize_xof(mut self) -> CShake256Reader {
        pad_absorbed(&mut self.state, self.position, CSHAKE_SUFFIX);
        CShake256Reader {
            state: self.state,
            position: SHAKE256_RATE_BYTES,
            keccak: self.keccak,
        }
    }
}

/// Reader for a finalized scalar cSHAKE256 output stream.
pub struct CShake256Reader {
    state: State1600,
    position: usize,
    keccak: keccak::Keccak,
}

impl CShake256Reader {
    /// Fill the buffer with the next bytes of the output stream.
    #[inline]
    pub fn read(&mut self, mut output: &mut [u8]) {
        while !output.is_empty() {
            if self.position == SHAKE256_RATE_BYTES {
                self.keccak
                    .with_f1600(|permutation| permutation(&mut self.state));
                self.position = 0;
            }
            let length = output.len().min(SHAKE256_RATE_BYTES - self.position);
            let block = rate_bytes(&self.state);
            output[..length].copy_from_slice(&block[self.position..self.position + length]);
            self.position += length;
            output = &mut output[length..];
        }
    }
}

impl CShake256 {
    /// Squeeze one output stream per suffix, each equal to cloning `self`,
    /// absorbing that suffix and reading from [`Self::finalize_xof`].
    ///
    /// `consume(index, block)` receives the rate blocks of stream `index` in
    /// order until it returns `false`. Streams run in backend-width groups, so
    /// their permutations share vector lanes; each group permutes until every
    /// stream in it is satisfied.
    pub fn squeeze_forks<T: AsRef<[u8]>>(
        &self,
        suffixes: &[T],
        consume: impl FnMut(usize, &[u8; SHAKE256_RATE_BYTES]) -> bool,
    ) {
        with_backend(CShakeForks {
            base: self,
            suffixes,
            consume,
        });
    }
}

struct CShakeForks<'a, T, C> {
    base: &'a CShake256,
    suffixes: &'a [T],
    consume: C,
}

impl<T, C> BackendClosure for CShakeForks<'_, T, C>
where
    T: AsRef<[u8]>,
    C: FnMut(usize, &[u8; SHAKE256_RATE_BYTES]) -> bool,
{
    fn call_once<B: Backend>(mut self) {
        let width = B::ParSize1600::USIZE;
        let scalar_permutation = B::get_f1600();
        let mut states = ParState1600::<B>::default();
        for (group, suffixes) in self.suffixes.chunks(width).enumerate() {
            // Unused lanes of the tail group repeat its last stream.
            for (lane, state) in states.iter_mut().enumerate() {
                *state = self.base.state;
                let mut position = self.base.position;
                let mut input = suffixes[lane.min(suffixes.len() - 1)].as_ref();
                while !input.is_empty() {
                    let length = input.len().min(SHAKE256_RATE_BYTES - position);
                    xor_rate_bytes_at(state, position, &input[..length]);
                    position += length;
                    input = &input[length..];
                    if position == SHAKE256_RATE_BYTES {
                        scalar_permutation(state);
                        position = 0;
                    }
                }
                pad_absorbed(state, position, CSHAKE_SUFFIX);
            }
            let mut active = Array::<bool, B::ParSize1600>::from_fn(|lane| lane < suffixes.len());
            while active.contains(&true) {
                B::get_par_f1600()(&mut states);
                for (lane, (state, active)) in states.iter().zip(&mut active).enumerate() {
                    if *active {
                        *active = (self.consume)(group * width + lane, &rate_bytes(state));
                    }
                }
            }
        }
    }
}

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

impl<B: Backend> Default for Shake256StateBatch<B> {
    /// Every lane holds the all-zero state, the initial sponge state.
    fn default() -> Self {
        Self {
            states: ParState1600::<B>::default(),
        }
    }
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
        let mut batch = Self::default();
        absorb_parallel::<B, _>(
            &mut batch.states,
            Array::<_, B::ParSize1600>::from_fn(|lane| SliceCursor(inputs[lane])),
            SHAKE_SUFFIX,
        );
        batch
    }

    /// Absorb and finalize one segmented SHAKE256 input per backend lane.
    ///
    /// Segments are logically concatenated without allocating or materializing
    /// the complete input. Full rate blocks common to every lane still use the
    /// data-parallel permutation; divergent full blocks use the backend's
    /// scalar permutation exactly as [`Self::new`] does.
    #[must_use]
    pub fn new_segmented<const SEGMENTS: usize>(inputs: &[[&[u8]; SEGMENTS]]) -> Self {
        let mut batch = Self::default();
        batch.absorb_segmented(inputs);
        batch
    }

    /// Restart every lane with one segmented SHAKE256 input, in place.
    ///
    /// The result equals [`Self::new_segmented`] on the same inputs; reusing
    /// one batch avoids moving the backend-width state.
    pub fn absorb_segmented<const SEGMENTS: usize>(&mut self, inputs: &[[&[u8]; SEGMENTS]]) {
        assert_eq!(
            inputs.len(),
            B::ParSize1600::USIZE,
            "SHAKE256 batch must match the backend width"
        );
        absorb_parallel::<B, _>(
            &mut self.states,
            Array::<_, B::ParSize1600>::from_fn(|lane| SegmentedCursor::new(inputs[lane])),
            SHAKE_SUFFIX,
        );
    }

    /// Squeeze the next SHAKE256 rate block from every lane.
    pub fn squeeze_block(&mut self) -> Array<[u8; SHAKE256_RATE_BYTES], B::ParSize1600> {
        self.permute();
        Array::from_fn(|lane| rate_bytes(&self.states[lane]))
    }

    /// Advance every lane by one SHAKE256 rate block in place; read the new
    /// blocks with [`Self::rate_words`].
    #[inline]
    pub fn permute(&mut self) {
        B::get_par_f1600()(&mut self.states);
    }

    /// The current output block of one lane as little-endian state words.
    #[inline]
    #[must_use]
    pub fn rate_words(&self, lane: usize) -> &[u64; RATE_WORDS] {
        self.states[lane][..RATE_WORDS]
            .try_into()
            .expect("the rate is a prefix of the state")
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
    shake256_64_batch_with(
        inputs,
        |input| input,
        |index, output| {
            outputs[index] = *output;
        },
    );
}

/// Run independent SHAKE256 streams through the backend lanes.
///
/// Stream `index` absorbs the concatenation of `input(index)`, then passes
/// its output blocks, as little-endian rate words, to
/// `consume(index, lane, words)` until that returns `false`. A lane whose
/// stream is satisfied takes the next stream before the next permutation, so
/// streams that need different numbers of blocks do not wait for each other.
/// Streams therefore finish in an unspecified order. At most one stream runs
/// on a lane at a time, and lanes are numbered below
/// [`backend_parallel_width`], so callers can keep scratch per lane.
pub fn shake256_streams<'a, const SEGMENTS: usize>(
    count: usize,
    input: impl Fn(usize) -> [&'a [u8]; SEGMENTS],
    consume: impl FnMut(usize, usize, &[u64; RATE_WORDS]) -> bool,
) {
    with_backend(Shake256Streams {
        count,
        input,
        consume,
    });
}

struct Shake256Streams<I, C> {
    count: usize,
    input: I,
    consume: C,
}

struct StreamLane<'a, const SEGMENTS: usize> {
    index: usize,
    cursor: SegmentedCursor<'a, SEGMENTS>,
    squeezing: bool,
}

impl<'a, I, C, const SEGMENTS: usize> BackendClosure for Shake256Streams<I, C>
where
    I: Fn(usize) -> [&'a [u8]; SEGMENTS],
    C: FnMut(usize, usize, &[u64; RATE_WORDS]) -> bool,
{
    fn call_once<B: Backend>(mut self) {
        let mut next = 0;
        let start = |next: &mut usize| {
            (*next < self.count).then(|| {
                let lane = StreamLane {
                    index: *next,
                    cursor: SegmentedCursor::new((self.input)(*next)),
                    squeezing: false,
                };
                *next += 1;
                lane
            })
        };
        let mut states = ParState1600::<B>::default();
        let mut lanes = Array::<_, B::ParSize1600>::from_fn(|_| start(&mut next));
        loop {
            let mut active = false;
            for (lane, state) in lanes.iter_mut().zip(states.iter_mut()) {
                let Some(lane) = lane else { continue };
                active = true;
                if lane.squeezing {
                    continue;
                }
                let length = lane.cursor.remaining();
                if length >= SHAKE256_RATE_BYTES {
                    lane.cursor.absorb_next(state, SHAKE256_RATE_BYTES);
                } else {
                    lane.cursor.absorb_next(state, length);
                    pad_absorbed(state, length, SHAKE_SUFFIX);
                    lane.squeezing = true;
                }
            }
            if !active {
                return;
            }
            B::get_par_f1600()(&mut states);
            for (number, (lane, state)) in lanes.iter_mut().zip(states.iter_mut()).enumerate() {
                let Some(stream) = lane else { continue };
                let words = state[..RATE_WORDS]
                    .try_into()
                    .expect("the rate is a prefix of the state");
                if stream.squeezing && !(self.consume)(stream.index, number, words) {
                    *state = State1600::default();
                    *lane = start(&mut next);
                }
            }
        }
    }
}

/// Hash `bytes(&inputs[i])` for every `i` to a 64-byte SHAKE256 output and
/// pass `(i, output)` to `write`, in input order.
///
/// This is [`shake256_64_batch`] without the input-slice and output buffers,
/// for callers that store each output in their own type.
pub fn shake256_64_batch_with<T>(
    inputs: &[T],
    bytes: impl Fn(&T) -> &[u8],
    write: impl FnMut(usize, &[u8; SHAKE256_OUTPUT_64_BYTES]),
) {
    with_backend(Shake256Outputs64 {
        inputs,
        bytes,
        write,
    });
}

struct Shake256Outputs64<'a, T, I, W> {
    inputs: &'a [T],
    bytes: I,
    write: W,
}

impl<T, I, W> BackendClosure for Shake256Outputs64<'_, T, I, W>
where
    I: Fn(&T) -> &[u8],
    W: FnMut(usize, &[u8; SHAKE256_OUTPUT_64_BYTES]),
{
    fn call_once<B: Backend>(mut self) {
        let width = B::ParSize1600::USIZE;
        let full_len = self.inputs.len() / width * width;
        let mut batch = Shake256StateBatch::<B>::default();
        for (group, inputs) in self.inputs[..full_len].chunks_exact(width).enumerate() {
            absorb_parallel::<B, _>(
                &mut batch.states,
                Array::<_, B::ParSize1600>::from_fn(|lane| {
                    SliceCursor((self.bytes)(&inputs[lane]))
                }),
                SHAKE_SUFFIX,
            );
            batch.permute();
            for (lane, state) in batch.states.iter().enumerate() {
                let mut output = [0_u8; SHAKE256_OUTPUT_64_BYTES];
                for (chunk, word) in output.chunks_exact_mut(8).zip(state) {
                    chunk.copy_from_slice(&word.to_le_bytes());
                }
                (self.write)(group * width + lane, &output);
            }
        }
        for (index, input) in self.inputs.iter().enumerate().skip(full_len) {
            (self.write)(index, &shake256_64_scalar::<B>((self.bytes)(input)));
        }
    }
}

impl<B: Backend> Shake256StateBatch<B> {
    /// Squeeze the first 64 output bytes of every lane into `outputs`.
    fn squeeze_64_into(&mut self, outputs: &mut [[u8; SHAKE256_OUTPUT_64_BYTES]]) {
        debug_assert_eq!(outputs.len(), B::ParSize1600::USIZE);
        self.permute();
        for (state, output) in self.states.iter().zip(outputs) {
            for (chunk, word) in output.chunks_exact_mut(8).zip(state) {
                chunk.copy_from_slice(&word.to_le_bytes());
            }
        }
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
    let mut prefix = Vec::new();
    cshake_prefix(function_name, customization, |bytes| {
        prefix.extend_from_slice(bytes);
    });
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
        let mut batch = Shake256StateBatch::<B>::default();
        for (inputs, outputs) in self
            .inputs
            .chunks(width)
            .zip(self.outputs.chunks_mut(width))
        {
            // Duplicate the last input only in unused lanes of the tail group.
            let segments = Array::<_, B::ParSize1600>::from_fn(|lane| {
                [self.prefix, inputs[lane.min(inputs.len() - 1)]]
            });
            absorb_parallel::<B, _>(
                &mut batch.states,
                Array::from_fn(|lane| SegmentedCursor::new(segments[lane])),
                CSHAKE_SUFFIX,
            );
            if outputs.len() == width {
                batch.squeeze_64_into(outputs);
            } else {
                let mut lanes =
                    Array::<_, B::ParSize1600>::from_fn(|_| [0; SHAKE256_OUTPUT_64_BYTES]);
                batch.squeeze_64_into(&mut lanes);
                outputs.copy_from_slice(&lanes[..outputs.len()]);
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
    let mut hasher = CShake256::from_prefix(function_name, customization);
    hasher.update(input);
    let mut output = [0_u8; OUTPUT_BYTES];
    hasher.finalize_xof().read(&mut output);
    output
}

fn cshake_prefix(function_name: &[u8], customization: &[u8], mut absorb: impl FnMut(&[u8])) {
    let mut length = 0;
    for value in [SHAKE256_RATE_BYTES as u64, (function_name.len() as u64) * 8] {
        let (encoded, encoded_len) = left_encode(value);
        absorb(&encoded[..encoded_len]);
        length += encoded_len;
    }
    absorb(function_name);
    length += function_name.len();
    let (encoded, encoded_len) = left_encode((customization.len() as u64) * 8);
    absorb(&encoded[..encoded_len]);
    absorb(customization);
    length += encoded_len + customization.len();
    let padding = (SHAKE256_RATE_BYTES - length % SHAKE256_RATE_BYTES) % SHAKE256_RATE_BYTES;
    absorb(&[0_u8; SHAKE256_RATE_BYTES][..padding]);
}

fn left_encode(value: u64) -> ([u8; 9], usize) {
    let bytes = value.to_be_bytes();
    let first = bytes
        .iter()
        .position(|&byte| byte != 0)
        .unwrap_or(bytes.len() - 1);
    let length = bytes.len() - first;
    let mut encoded = [0_u8; 9];
    encoded[0] = u8::try_from(length).expect("u64 encoding length fits one byte");
    encoded[1..=length].copy_from_slice(&bytes[first..]);
    (encoded, length + 1)
}

fn append_left_encode(output: &mut Vec<u8>, value: u64) {
    let (encoded, length) = left_encode(value);
    output.extend_from_slice(&encoded[..length]);
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

    // Called from generic batch code that is instantiated in downstream crates.
    #[inline]
    fn absorb_next(&mut self, state: &mut State1600, length: usize) {
        let (next, remaining) = self.0.split_at(length);
        xor_rate_bytes(state, next);
        self.0 = remaining;
    }
}

struct SegmentedCursor<'a, const SEGMENTS: usize> {
    segments: [&'a [u8]; SEGMENTS],
    segment: usize,
    offset: usize,
    remaining: usize,
}

impl<'a, const SEGMENTS: usize> SegmentedCursor<'a, SEGMENTS> {
    fn new(segments: [&'a [u8]; SEGMENTS]) -> Self {
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
        let mut written = 0;
        while written < length {
            let segment = self.segments[self.segment];
            let available = &segment[self.offset..];
            let copied = available.len().min(length - written);
            xor_rate_bytes_at(state, written, &available[..copied]);
            written += copied;
            self.offset += copied;
            if self.offset == segment.len() {
                self.segment += 1;
                self.offset = 0;
            }
        }
        self.remaining -= length;
    }
}

/// Reset `states` and absorb and pad one input per lane, in place.
fn absorb_parallel<B: Backend, C: AbsorbCursor>(
    states: &mut ParState1600<B>,
    mut inputs: Array<C, B::ParSize1600>,
    suffix: u8,
) {
    for state in states.iter_mut() {
        *state = State1600::default();
    }
    let full_blocks =
        Array::<_, B::ParSize1600>::from_fn(|lane| inputs[lane].remaining() / SHAKE256_RATE_BYTES);
    let common_blocks = full_blocks.iter().copied().min().unwrap_or(0);

    for _ in 0..common_blocks {
        for (state, input) in states.iter_mut().zip(&mut inputs) {
            input.absorb_next(state, SHAKE256_RATE_BYTES);
        }
        B::get_par_f1600()(states);
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

// Called from generic batch code that is instantiated in downstream crates.
#[inline]
fn xor_rate_bytes(state: &mut State1600, bytes: &[u8]) {
    xor_rate_bytes_at(state, 0, bytes);
}

/// XOR `bytes` into the rate at byte `position`: whole little-endian words
/// where the position is word-aligned, single bytes at the unaligned ends.
#[inline]
fn xor_rate_bytes_at(state: &mut State1600, mut position: usize, mut bytes: &[u8]) {
    debug_assert!(position + bytes.len() <= SHAKE256_RATE_BYTES);
    while !position.is_multiple_of(8) {
        let Some((&byte, rest)) = bytes.split_first() else {
            return;
        };
        state[position / 8] ^= u64::from(byte) << (8 * (position % 8));
        position += 1;
        bytes = rest;
    }
    let mut words = bytes.chunks_exact(8);
    for (word, chunk) in state[position / 8..].iter_mut().zip(&mut words) {
        *word ^= u64::from_le_bytes(chunk.try_into().expect("an eight-byte chunk"));
    }
    let tail = words.remainder();
    let word = (position + bytes.len() - tail.len()) / 8;
    for (offset, &byte) in tail.iter().enumerate() {
        state[word] ^= u64::from(byte) << (8 * offset);
    }
}

// Called from generic batch code that is instantiated in downstream crates.
#[inline]
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

#[cfg(all(
    target_arch = "aarch64",
    any(target_vendor = "apple", feature = "backend-override")
))]
#[allow(unsafe_code)]
mod aarch64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SelectedBackend {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(all(
        target_arch = "aarch64",
        any(target_vendor = "apple", feature = "backend-override")
    ))]
    Hybrid,
    Platform,
}

#[inline]
fn selected_backend() -> SelectedBackend {
    static SELECTED: OnceLock<SelectedBackend> = OnceLock::new();
    *SELECTED.get_or_init(|| {
        #[cfg(feature = "backend-override")]
        {
            let requested = std::env::var("AERIE_KECCAK_BACKEND").ok();
            assert!(
                matches!(
                    requested.as_deref(),
                    None | Some("auto" | "platform" | "hybrid" | "avx2" | "avx512")
                ),
                "AERIE_KECCAK_BACKEND must be auto, platform, hybrid, avx2, or avx512"
            );
            if requested.as_deref() == Some("platform") {
                return SelectedBackend::Platform;
            }
            #[cfg(target_arch = "aarch64")]
            {
                if requested.as_deref() == Some("hybrid") {
                    assert!(
                        aarch64::supported(),
                        "AERIE_KECCAK_BACKEND=hybrid requires the ARMv8 SHA3 extension"
                    );
                    return SelectedBackend::Hybrid;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if requested.as_deref() == Some("avx512") {
                    assert!(
                        std::is_x86_feature_detected!("avx512f"),
                        "AERIE_KECCAK_BACKEND=avx512 requires AVX-512F"
                    );
                    return SelectedBackend::Avx512;
                }
                if requested.as_deref() == Some("avx2") {
                    assert!(
                        std::is_x86_feature_detected!("avx2"),
                        "AERIE_KECCAK_BACKEND=avx2 requires AVX2"
                    );
                    return SelectedBackend::Avx2;
                }
            }
            assert!(
                requested.is_none() || requested.as_deref() == Some("auto"),
                "AERIE_KECCAK_BACKEND=avx2 or avx512 requires x86-64, and hybrid requires AArch64"
            );
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return SelectedBackend::Avx512;
            }
            if std::is_x86_feature_detected!("avx2") {
                return SelectedBackend::Avx2;
            }
        }
        // The hybrid kernel is measured faster than the two-state SHA3
        // kernel on Apple cores only; other AArch64 cores keep RustCrypto's.
        #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
        {
            if aarch64::supported() {
                return SelectedBackend::Hybrid;
            }
        }
        SelectedBackend::Platform
    })
}

/// Execute a closure with the fastest runtime-safe Keccak backend.
///
/// This performs data parallelism only: AVX2 handles four independent states
/// per permutation, AVX-512 handles eight, and the `AArch64` hybrid handles
/// three (two in SHA3 vector lanes, one in general-purpose registers),
/// without creating threads. The hybrid is the default on Apple `AArch64`
/// cores with SHA3.
/// With the `backend-override` feature,
/// `AERIE_KECCAK_BACKEND=platform|hybrid|avx2|avx512` can pin a backend for
/// tests and benchmarks. `platform` uses `RustCrypto` dispatch: scalar on
/// x86-64, potentially ARM SHA3 on `AArch64`. Unsupported
/// forced backends are rejected before executing a vector instruction.
/// This crate ignores the separate `AERIE_SIMD_BACKEND` setting.
#[inline]
pub fn with_backend(closure: impl BackendClosure) {
    match selected_backend() {
        #[cfg(target_arch = "x86_64")]
        SelectedBackend::Avx512 => closure.call_once::<x86::Avx512Backend>(),
        #[cfg(target_arch = "x86_64")]
        SelectedBackend::Avx2 => closure.call_once::<x86::Avx2Backend>(),
        #[cfg(all(
            target_arch = "aarch64",
            any(target_vendor = "apple", feature = "backend-override")
        ))]
        SelectedBackend::Hybrid => closure.call_once::<aarch64::HybridBackend>(),
        SelectedBackend::Platform => keccak::Keccak::new().with_backend(closure),
    }
}

/// Name of the backend selected for this process.
#[must_use]
pub fn backend_name() -> &'static str {
    match selected_backend() {
        #[cfg(target_arch = "x86_64")]
        SelectedBackend::Avx512 => "avx512",
        #[cfg(target_arch = "x86_64")]
        SelectedBackend::Avx2 => "avx2",
        #[cfg(all(
            target_arch = "aarch64",
            any(target_vendor = "apple", feature = "backend-override")
        ))]
        SelectedBackend::Hybrid => "hybrid",
        SelectedBackend::Platform => "platform",
    }
}

/// Number of independent Keccak-f1600 states processed by the active backend.
#[must_use]
pub fn backend_parallel_width() -> usize {
    struct Width<'a>(&'a mut usize);

    impl BackendClosure for Width<'_> {
        fn call_once<B: Backend>(self) {
            *self.0 = B::ParSize1600::USIZE;
        }
    }

    let mut width = 0;
    with_backend(Width(&mut width));
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
        CShake256 as ReferenceCShake256, CShake256Core,
    };
    use shake::digest::{
        ExtendableOutput as ShakeExtendableOutput, Update as ShakeUpdate,
        XofReader as ShakeXofReader,
    };
    use shake::Shake256;

    #[test]
    fn selected_width_matches_backend_name() {
        let width = backend_parallel_width();
        if let Ok(requested) = std::env::var("AERIE_KECCAK_BACKEND") {
            assert!(requested == "auto" || cfg!(feature = "backend-override"));
            if matches!(
                requested.as_str(),
                "platform" | "hybrid" | "avx2" | "avx512"
            ) {
                assert_eq!(backend_name(), requested);
            }
        }
        match backend_name() {
            "avx512" => assert_eq!(width, 8),
            "avx2" => assert_eq!(width, 4),
            "hybrid" => assert_eq!(width, 3),
            "platform" => assert!(width == 1 || width == 2),
            name => panic!("unknown backend {name}"),
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
    fn streaming_cshake_matches_independent_reference_at_rate_boundaries() {
        let lengths = [0, 1, 135, 136, 137, 272, 1000];
        for function_name in [b"".as_slice(), b"Aerie streaming test", &[0x5a; 300]] {
            for customization in [b"Aerie cSHAKE test".as_slice(), &[0xa5; 300]] {
                for input_len in lengths {
                    let input: Vec<_> = (0..input_len)
                        .map(|i| u8::try_from(i % 251).unwrap())
                        .collect();
                    let mut reference = ReferenceCShake256::from_core(
                        CShake256Core::new_with_function_name(function_name, customization),
                    );
                    reference.update(&input);
                    let mut expected = vec![0; 1000];
                    reference.finalize_xof().read(&mut expected);
                    for output_len in lengths {
                        for split in lengths {
                            let input_split = split.min(input_len);
                            let mut hasher = CShake256::new(function_name, customization);
                            hasher.update(&input[..input_split]);
                            hasher.update(&[]);
                            hasher.update(&input[input_split..]);
                            let mut reader = hasher.finalize_xof();
                            let mut actual = vec![0; output_len];
                            let output_split = split.min(output_len);
                            reader.read(&mut actual[..output_split]);
                            reader.read(&mut []);
                            reader.read(&mut actual[output_split..]);
                            assert_eq!(actual, expected[..output_len]);
                        }
                    }
                }
            }
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
            for custom in [b"aerie/falcon/v1/record-leaf".as_slice(), &[0xa5; 300]] {
                let actual = customized_chaining_values(&inputs, b"", custom);
                for (input, value) in inputs.iter().zip(actual) {
                    let mut reference = ReferenceCShake256::from_core(
                        CShake256Core::new_with_function_name(b"", custom),
                    );
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
        let leaf = customized_chaining_values(&[&frame], b"", b"aerie/falcon/v1/record-leaf")[0];
        let projection =
            customized_chaining_values(&[&frame], b"", b"aerie/falcon/v1/projection")[0];
        let instance = customized_chaining_values(&[&frame], b"", b"aerie/falcon/v1/instance")[0];
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
        let mut reference = ReferenceCShake256::from_core(core);
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
    fn shake256_streams_match_independent_readers() {
        let messages = (0..23_usize)
            .map(|stream| {
                let length = [0, 1, 135, 136, 137, 271, 272, 600][stream % 8];
                (0..length)
                    .map(|index| u8::try_from((index * 13 + stream) % 256).expect("masked"))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for count in [0, 1, 2, 3, 5, 23] {
            let blocks_wanted = |stream: usize| 1 + (stream * 7) % 5;
            let mut actual = vec![Vec::new(); count];
            let mut finished = vec![false; count];
            let mut running = vec![None; backend_parallel_width()];
            shake256_streams(
                count,
                |stream| {
                    let (head, tail) = messages[stream].split_at(messages[stream].len() / 3);
                    [head, b"", tail]
                },
                |stream, lane, words| {
                    assert!(!finished[stream]);
                    assert!(running[lane].is_none_or(|current| current == stream));
                    running[lane] = Some(stream);
                    for word in words {
                        actual[stream].extend_from_slice(&word.to_le_bytes());
                    }
                    let more = actual[stream].len() < blocks_wanted(stream) * SHAKE256_RATE_BYTES;
                    if !more {
                        finished[stream] = true;
                        running[lane] = None;
                    }
                    more
                },
            );
            assert!(finished.iter().all(|&done| done));
            for (stream, actual) in actual.iter().enumerate() {
                let mut hasher = Shake256::default();
                ShakeUpdate::update(&mut hasher, &messages[stream]);
                let mut expected = vec![0; blocks_wanted(stream) * SHAKE256_RATE_BYTES];
                hasher.finalize_xof().read(&mut expected);
                assert_eq!(actual, &expected, "stream {stream} of {count}");
            }
        }
    }

    #[test]
    fn cshake_forks_match_cloned_readers() {
        for base_len in [0, 1, 127, 128, 135, 136, 137, 300] {
            let mut base = CShake256::new(b"", b"fork fixture");
            let base_input = (0..base_len)
                .map(|index| u8::try_from(index % 251).expect("masked below 251"))
                .collect::<Vec<_>>();
            base.update(&base_input);
            for count in 0..=9 {
                let suffixes = (0..count)
                    .map(|fork: usize| {
                        let length = [0, 8, 9, 135, 136, 140][fork % 6];
                        (0..length)
                            .map(|index| u8::try_from((index * 7 + fork) % 256).expect("masked"))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let blocks_wanted = |fork: usize| 1 + fork % 4;
                let mut actual = vec![Vec::new(); count];
                base.squeeze_forks(&suffixes, |fork, block| {
                    actual[fork].extend_from_slice(block);
                    actual[fork].len() < blocks_wanted(fork) * SHAKE256_RATE_BYTES
                });
                for (fork, suffix) in suffixes.iter().enumerate() {
                    let mut reader = base.clone();
                    reader.update(suffix);
                    let mut expected = vec![0; blocks_wanted(fork) * SHAKE256_RATE_BYTES];
                    reader.finalize_xof().read(&mut expected);
                    assert_eq!(
                        actual[fork], expected,
                        "base {base_len}, fork {fork}/{count}"
                    );
                }
            }
        }
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
