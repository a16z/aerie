use akita_error::AkitaError;
#[cfg(debug_assertions)]
use jolt_field::One;
use jolt_field::{Fold, Unreduced, Zero};

use crate::substrate::OpeningField as E;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Number of nonconstant coefficients sent by quadratic and quartic rounds.
pub(super) const QUADRATIC_COEFFICIENTS: usize = 2;
pub(super) use aerie_falcon::protocol::relations::quartic::QUARTIC_COEFFICIENTS;
/// Bucketing pays only when the items dominate the pattern space.
pub(super) const BUCKET_MIN_RATIO: usize = 4;
/// Distinct images `d(d + 1)` for the validated digits in `[-8, 7]`.
pub(super) const RANGE_IMAGE_PATTERNS: usize = 8;

const SEQUENTIAL_FOLD_ROWS: usize = 1 << 12;

/// Selects whether compact kernels cache the first two rounds together.
/// Production uses the pattern-space size threshold; tests can exercise either path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrefixPolicy {
    Auto,
    #[cfg(test)]
    ForceBivariate,
    #[cfg(test)]
    ForbidBivariate,
}

#[cfg(feature = "parallel")]
const PAIR_CODE_CHUNK: usize = 1 << 14;

/// Minimum number of independently processed elements before using Rayon.
///
/// Call sites count the elements handled by their parallel iterator, rather
/// than rows or bytes, so the threshold has the same unit across kernels.
#[cfg(any(test, feature = "parallel"))]
pub(crate) use crate::protocol::PARALLEL_MIN_WORK;

/// Borrowed coefficient digits, either in canonical radix-16 pair codes or as
/// direct cells supplied by an adversarial hook.
#[derive(Clone, Copy, Debug)]
pub(crate) enum DigitSource<'a> {
    Packed { codes: &'a [u8], minimum: i8 },
    Direct(&'a [i8]),
}

impl DigitSource<'_> {
    #[must_use]
    pub(crate) const fn len(self) -> usize {
        match self {
            Self::Packed { codes, .. } => 2 * codes.len(),
            Self::Direct(cells) => cells.len(),
        }
    }

    #[must_use]
    pub(crate) fn get(self, index: usize) -> i8 {
        match self {
            Self::Packed { codes, minimum } => {
                let code = codes[index / 2];
                let digit = if index.is_multiple_of(2) {
                    code >> 4
                } else {
                    code & 0x0f
                };
                minimum + i8::try_from(digit).expect("nibble fits i8")
            }
            Self::Direct(cells) => cells[index],
        }
    }

    pub(crate) fn decode_into(self, start: usize, output: &mut [i8]) {
        assert!(start <= self.len() && output.len() <= self.len() - start);
        match self {
            Self::Packed { codes, minimum } => {
                let (head, body) = output.split_at_mut((start % 2).min(output.len()));
                if let Some(cell) = head.first_mut() {
                    *cell = minimum + (codes[start / 2] & 0x0f) as i8;
                }
                let first = start.div_ceil(2);
                let (pairs, tail) = body.split_at_mut(body.len() & !1);
                for (pair, &code) in pairs.chunks_exact_mut(2).zip(&codes[first..]) {
                    pair[0] = minimum + (code >> 4) as i8;
                    pair[1] = minimum + (code & 0x0f) as i8;
                }
                if let Some(cell) = tail.first_mut() {
                    *cell = minimum + (codes[first + pairs.len() / 2] >> 4) as i8;
                }
            }
            Self::Direct(cells) => output.copy_from_slice(&cells[start..start + output.len()]),
        }
    }
}

/// Borrowed coefficient signs, either one bit per cell or direct adversarial
/// cells.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SignSource<'a> {
    Bits(&'a [u64]),
    #[cfg(any(test, feature = "internals"))]
    Direct(&'a [i8]),
}

impl SignSource<'_> {
    #[must_use]
    pub(crate) const fn len(self) -> usize {
        match self {
            Self::Bits(words) => 64 * words.len(),
            #[cfg(any(test, feature = "internals"))]
            Self::Direct(cells) => cells.len(),
        }
    }

    #[must_use]
    pub(crate) fn get(self, index: usize) -> i8 {
        match self {
            Self::Bits(words) => {
                if (words[index / 64] >> (index % 64)) & 1 == 0 {
                    1
                } else {
                    -1
                }
            }
            #[cfg(any(test, feature = "internals"))]
            Self::Direct(cells) => cells[index],
        }
    }

    pub(crate) fn decode_into(self, start: usize, output: &mut [i8]) {
        assert!(start <= self.len() && output.len() <= self.len() - start);
        match self {
            Self::Bits(words) => {
                let mut index = start;
                let mut rest = output;
                while !rest.is_empty() {
                    let shift = index % 64;
                    let (block, tail) = rest.split_at_mut((64 - shift).min(rest.len()));
                    let word = words[index / 64] >> shift;
                    for (offset, cell) in block.iter_mut().enumerate() {
                        *cell = 1 - 2 * ((word >> offset) & 1) as i8;
                    }
                    index += block.len();
                    rest = tail;
                }
            }
            #[cfg(any(test, feature = "internals"))]
            Self::Direct(cells) => output.copy_from_slice(&cells[start..start + output.len()]),
        }
    }
}

#[inline]
pub(crate) fn record_phase(phase: &'static str) {
    tracing::Span::current().record("phase", phase);
}

/// Fold one dense multilinear row layer in place. The leading sequential
/// wave avoids overlapping mutable/source slices; later disjoint waves may
/// use Rayon. Product, quartic, and quadratic kernels share this layout.
#[tracing::instrument(
    skip_all,
    name = "kernels::fold_dense_rows",
    fields(kernel = tracing::field::Empty)
)]
pub(crate) fn fold_rows_in_place(
    table: &mut Vec<E>,
    lanes: usize,
    challenge: E,
    kernel: &'static str,
) {
    tracing::Span::current().record("kernel", kernel);
    let rows = table.len() / lanes;
    debug_assert!(rows >= 2 && rows.is_power_of_two());
    let next_rows = rows / 2;
    let fold_context = E::precompute(challenge);
    let sequential_rows = next_rows.min(SEQUENTIAL_FOLD_ROWS);
    for output_row in 0..sequential_rows {
        for lane in 0..lanes {
            let left = table[(2 * output_row) * lanes + lane];
            let right = table[(2 * output_row + 1) * lanes + lane];
            table[output_row * lanes + lane] = E::fold_one(&fold_context, left, right);
        }
    }

    #[cfg(feature = "parallel")]
    {
        let mut wave_start = sequential_rows;
        while wave_start < next_rows {
            let wave_end = (2 * wave_start).min(next_rows);
            let source_start = 2 * wave_start;
            let (output_prefix, source_suffix) = table.split_at_mut(source_start * lanes);
            let output = &mut output_prefix[wave_start * lanes..wave_end * lanes];
            let source = &source_suffix[..2 * (wave_end - wave_start) * lanes];
            output
                .par_chunks_mut(lanes)
                .zip(source.par_chunks_exact(2 * lanes))
                .for_each(|(output, source)| {
                    for lane in 0..lanes {
                        output[lane] =
                            E::fold_one(&fold_context, source[lane], source[lanes + lane]);
                    }
                });
            wave_start = wave_end;
        }
    }
    #[cfg(not(feature = "parallel"))]
    for output_row in sequential_rows..next_rows {
        for lane in 0..lanes {
            let left = table[(2 * output_row) * lanes + lane];
            let right = table[(2 * output_row + 1) * lanes + lane];
            table[output_row * lanes + lane] = E::fold_one(&fold_context, left, right);
        }
    }
    table.truncate(next_rows * lanes);
}

/// Encode every adjacent cell pair of a compact source, validating each cell.
///
/// Large sources are encoded in parallel chunks. On failure the scan is
/// repeated serially so the reported error is always the lowest failing
/// pair's, exactly as the serial encoder reports it.
pub(crate) fn encode_pairs<C>(pair_count: usize, pair_code: C) -> Result<Vec<u8>, AkitaError>
where
    C: Fn(usize) -> Result<u8, AkitaError> + Sync,
{
    #[cfg(feature = "parallel")]
    if pair_count > PAIR_CODE_CHUNK {
        let mut codes = vec![0_u8; pair_count];
        let encoded = codes
            .par_chunks_mut(PAIR_CODE_CHUNK)
            .enumerate()
            .try_for_each(|(chunk, slots)| {
                let start = chunk * PAIR_CODE_CHUNK;
                for (offset, slot) in slots.iter_mut().enumerate() {
                    *slot = pair_code(start + offset)?;
                }
                Ok::<(), AkitaError>(())
            });
        if encoded.is_ok() {
            return Ok(codes);
        }
    }
    (0..pair_count).map(pair_code).collect()
}

#[cfg(test)]
pub(super) fn in_bucket_test_pools(check: impl Fn() + Send + Sync) {
    #[cfg(feature = "parallel")]
    for threads in [1, 2, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(&check);
    }
    #[cfg(not(feature = "parallel"))]
    check();
}

/// Bucket sums over `items`, accumulated in one private table per worker.
/// The chunk count is at most `items / width` (and the thread count), with a
/// minimum of one; below `2 * width` items, accumulation uses one serial chunk.
///
/// A Rayon `fold` would allocate, zero, and reduce one full bucket table per
/// split, which dominates when the table is large (the source-link quartet round has
/// `16^4` buckets per source). Fixed contiguous chunks, at most one per
/// thread and each at least a table's width of items, bound that overhead;
/// the tables are then summed column-block by column-block. Field addition
/// is associative and commutative, so the sums equal the serial path's.
#[cfg(feature = "parallel")]
pub(crate) fn chunked_bucket_sums<A>(items: usize, width: usize, accumulate: A) -> Vec<E>
where
    A: Fn(&mut Vec<E>, usize) + Sync,
{
    const REDUCE_BLOCK: usize = 1 << 12;
    let chunks = rayon::current_num_threads()
        .min(items / width.max(1))
        .max(1);
    let per_chunk = items.div_ceil(chunks);
    let partials: Vec<Vec<E>> = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut sums = vec![E::zero(); width];
            for item in chunk * per_chunk..((chunk + 1) * per_chunk).min(items) {
                accumulate(&mut sums, item);
            }
            sums
        })
        .collect();
    let mut partials = partials.into_iter();
    let mut sums = partials.next().expect("at least one chunk");
    let rest: Vec<Vec<E>> = partials.collect();
    sums.par_chunks_mut(REDUCE_BLOCK)
        .enumerate()
        .for_each(|(block, target)| {
            let start = block * REDUCE_BLOCK;
            for partial in &rest {
                for (sum, value) in target.iter_mut().zip(&partial[start..]) {
                    *sum += *value;
                }
            }
        });
    sums
}

#[expect(
    clippy::inline_always,
    reason = "Called per element inside the round closures of every kernel"
)]
#[inline(always)]
pub(super) fn split_equality_weight(first: &[E], second: &[E], index: usize) -> E {
    let first_bits = first.len().trailing_zeros() as usize;
    first[index & (first.len() - 1)] * second[index >> first_bits]
}

#[inline]
pub(super) fn bivariate_linear_evals(first: &[E], second: &[E]) -> (E, E) {
    let table = if first.len() > 1 { first } else { second };
    table
        .chunks_exact(2)
        .fold((E::zero(), E::zero()), |(zero, one), pair| {
            (zero + pair[0], one + pair[1])
        })
}

#[expect(
    clippy::inline_always,
    reason = "Called per element inside the round closures of every kernel"
)]
#[inline(always)]
pub(super) fn bivariate_suffix_weight(first: &[E], second: &[E], index: usize) -> E {
    if first.len() > 1 {
        let first_index = 2 * index;
        let first_bits = first.len().trailing_zeros() as usize;
        (first[first_index & (first.len() - 1)] + first[(first_index & (first.len() - 1)) + 1])
            * second[first_index >> first_bits]
    } else {
        first[0] * (second[2 * index] + second[2 * index + 1])
    }
}

#[expect(
    clippy::inline_always,
    reason = "Called per element inside the round closures of every kernel"
)]
#[inline(always)]
pub(super) fn add_scaled_coefficients<const N: usize>(
    destination: &mut [E; N],
    source: [E; N],
    scale: E,
) {
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += scale * source;
    }
}

/// Contract coefficient vectors against `first[i] * second[j]` without
/// materializing that product per pair. Each block reduces its first-factor
/// sum before multiplying by the second factor; the outer sum is unreduced.
///
/// Each inner lane sums `first.len()` products and each outer lane sums
/// `second.len()` products, independently of the coefficient width `N`.
/// Neither accumulator sums their product. At the pinned jolt-field revision
/// c7c2d08, solinas/unreduced.rs documents more than 2^62 terms of headroom
/// for `FpExt2<Fp64>`, the current `OpeningField`. A valid `E` slice has at
/// most `isize::MAX / size_of::<E>()` elements, below 2^59 on a 64-bit target
/// with 16-byte `E`. The distinct exact-sum and slice-capacity assertions
/// protect this specified field contract, not arbitrary future accumulators.
pub(super) fn sum_split_coefficients<const N: usize>(
    first: &[E],
    second: &[E],
    coefficients_at: impl Fn(usize) -> [E; N] + Send + Sync,
) -> [E; N] {
    let first_len = first.len();
    const { assert!(E::SUM_IS_EXACT) };
    const {
        assert!((isize::MAX as u128) / (core::mem::size_of::<E>() as u128) < (1_u128 << 62));
    };
    let block_coefficients = |block: usize| {
        let mut inner = [<E as Unreduced>::Product::zero(); N];
        let base = block * first_len;
        for (low, &weight) in first.iter().enumerate() {
            let coefficients = coefficients_at(base + low);
            for (inner, coefficient) in inner.iter_mut().zip(coefficients) {
                *inner += weight.mul_unreduced(coefficient);
            }
        }
        let mut output = [<E as Unreduced>::Product::zero(); N];
        for (output, inner) in output.iter_mut().zip(inner) {
            *output += second[block].mul_unreduced(E::reduce_product(inner));
        }
        output
    };
    #[cfg(feature = "parallel")]
    if first_len * second.len() >= PARALLEL_MIN_WORK && second.len() > 1 {
        return (0..second.len())
            .into_par_iter()
            .map(block_coefficients)
            .reduce(
                || [<E as Unreduced>::Product::zero(); N],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    left
                },
            )
            .map(E::reduce_product);
    }
    (0..second.len())
        .map(block_coefficients)
        .fold([<E as Unreduced>::Product::zero(); N], |mut left, right| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        })
        .map(E::reduce_product)
}

/// Lookup slot of a validated digit modulo `z ~ -1-z`.
pub(super) fn range_image_index(digit: i8) -> usize {
    if digit >= 0 {
        digit as usize
    } else {
        (-i16::from(digit) - 1) as usize
    }
}

pub(super) fn zeroed_vec(len: usize, allocation_error: &'static str) -> Result<Vec<E>, AkitaError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| AkitaError::InvalidInput(allocation_error.to_owned()))?;
    values.resize(len, E::zero());
    Ok(values)
}

/// Decode a quartet in most-significant-slot-first radix order.
#[inline]
pub(super) fn decode_quartet_indices(pattern: usize, base: usize) -> [usize; 4] {
    [
        (pattern / base.pow(3)) % base,
        (pattern / base.pow(2)) % base,
        (pattern / base) % base,
        pattern % base,
    ]
}

/// Check the normalized factors used by the two-round equality cache.
#[inline]
pub(super) fn debug_assert_bivariate_factors(first: &[E], second: &[E], round_zero: bool) {
    #[cfg(debug_assertions)]
    {
        for factor in [first, second] {
            debug_assert!(!factor.is_empty() && factor.len().is_power_of_two());
            debug_assert_eq!(factor.iter().copied().sum::<E>(), E::one());
        }
        if round_zero {
            let table = if first.len() > 1 { first } else { second };
            let (even, odd) = bivariate_linear_evals(first, second);
            for pair in table.chunks_exact(2) {
                let sum = pair[0] + pair[1];
                debug_assert_eq!(pair[0], sum * even);
                debug_assert_eq!(pair[1], sum * odd);
            }
        }
    }
    #[cfg(not(debug_assertions))]
    let _ = (first, second, round_zero);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::substrate::BaseField as F;
    use jolt_field::Ring;

    fn full_field_value(index: usize) -> E {
        if index.is_multiple_of(7) {
            return E::zero();
        }
        let index = index as u64;
        E::new(
            F::from_u64(index.wrapping_mul(0xd134_2543_de82_ef95)),
            F::from_u64(
                index
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(u64::MAX),
            ),
        )
    }

    fn check_split_reducer<const N: usize>() {
        // Include singleton factors, asymmetric shapes, and work counts on
        // both sides of the parallel threshold (also with only one block).
        #[cfg(feature = "parallel")]
        let threshold = PARALLEL_MIN_WORK;
        #[cfg(not(feature = "parallel"))]
        let threshold = 1 << 12;
        let shapes = [
            (1, 1),
            (1, 7),
            (11, 1),
            (3, 5),
            (8, 3),
            (1, threshold - 1),
            (1, threshold),
            (1, threshold + 1),
            (threshold - 1, 1),
            (threshold, 1),
            (threshold + 1, 1),
            (63, 65),
            (64, 64),
            (17, 241),
        ];
        for (first_len, second_len) in shapes {
            for all_zero in [false, true] {
                let value = |index| {
                    if all_zero {
                        E::zero()
                    } else {
                        full_field_value(index)
                    }
                };
                let first: Vec<E> = (0..first_len).map(|i| value(i + 1)).collect();
                let second: Vec<E> = (0..second_len).map(|i| value(i + 19)).collect();
                let coefficients_at = |index| core::array::from_fn(|lane| value(index * N + lane));
                let mut expected = [E::zero(); N];
                for (high, &second_weight) in second.iter().enumerate() {
                    for (low, &first_weight) in first.iter().enumerate() {
                        let coefficients: [E; N] = coefficients_at(high * first_len + low);
                        for (sum, coefficient) in expected.iter_mut().zip(coefficients) {
                            *sum += (first_weight * second_weight) * coefficient;
                        }
                    }
                }
                assert_eq!(
                    sum_split_coefficients(&first, &second, coefficients_at),
                    expected,
                    "width={N}, shape={first_len}x{second_len}, all_zero={all_zero}"
                );
            }
        }
    }

    #[test]
    fn split_reducer_quadratic_matches_fully_reduced_nested_sum() {
        check_split_reducer::<2>();
    }

    #[test]
    fn split_reducer_quartic_matches_fully_reduced_nested_sum() {
        check_split_reducer::<4>();
    }

    #[test]
    fn quartet_decode_matches_independent_positional_order() {
        for base in [4, 8, 16] {
            let mut pattern = 0;
            for first in 0..base {
                for second in 0..base {
                    for third in 0..base {
                        for fourth in 0..base {
                            assert_eq!(
                                decode_quartet_indices(pattern, base),
                                [first, second, third, fourth]
                            );
                            pattern += 1;
                        }
                    }
                }
            }
            assert_eq!(pattern, base.pow(4));
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn bivariate_factor_contract_rejects_invalid_shapes() {
        use std::panic::catch_unwind;
        let one = E::from_i64(1);
        for (first, second) in [
            (vec![], vec![one]),
            (vec![one], vec![]),
            (vec![one, E::zero(), E::zero()], vec![one]),
            (vec![one], vec![one, E::zero(), E::zero()]),
        ] {
            assert!(
                catch_unwind(|| debug_assert_bivariate_factors(&first, &second, false)).is_err()
            );
        }
    }

    #[test]
    fn encode_pairs_reports_the_lowest_failing_pair() {
        for pair_count in [7, 1 << 15, (1 << 16) + 3] {
            let codes = encode_pairs(pair_count, |pair| Ok((pair % 251) as u8)).unwrap();
            assert_eq!(codes.len(), pair_count);
            assert!(codes
                .iter()
                .enumerate()
                .all(|(pair, &code)| code == (pair % 251) as u8));
            let first_bad = pair_count / 3;
            let error = encode_pairs(pair_count, |pair| {
                if pair >= first_bad && pair % 5 == first_bad % 5 {
                    Err(AkitaError::InvalidInput(format!("pair {pair}")))
                } else {
                    Ok(0)
                }
            })
            .unwrap_err();
            assert_eq!(
                error.to_string(),
                AkitaError::InvalidInput(format!("pair {first_bad}")).to_string()
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn chunked_bucket_sums_match_serial_accumulation() {
        use crate::substrate::OpeningField as E;
        use jolt_field::{Ring, Zero};
        for (items, width) in [(1 << 12, 3 << 8), (1 << 16, 3 << 12), (12345, 17)] {
            let accumulate = |sums: &mut Vec<E>, item: usize| {
                let weight = E::from_i64(((item as i64) * 0x9e37_79b9) | 1);
                sums[(item * 7) % width] += weight;
                sums[(item * 13 + 1) % width] += weight * weight;
            };
            let mut expected = vec![E::zero(); width];
            for item in 0..items {
                accumulate(&mut expected, item);
            }
            assert_eq!(chunked_bucket_sums(items, width, accumulate), expected);
        }
    }
}
