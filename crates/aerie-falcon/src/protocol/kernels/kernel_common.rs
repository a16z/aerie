use akita_error::AkitaError;
use jolt_field::Field;
use jolt_field::Fold;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const SEQUENTIAL_FOLD_ROWS: usize = 1 << 12;
#[cfg(feature = "parallel")]
const PAIR_CODE_CHUNK: usize = 1 << 14;

/// Minimum number of independently processed elements before using Rayon.
///
/// Call sites count the elements handled by their parallel iterator, rather
/// than rows or bytes, so the threshold has the same unit across kernels.
#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_MIN_WORK: usize = 1 << 12;

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
    Direct(&'a [i8]),
}

impl SignSource<'_> {
    #[must_use]
    pub(crate) const fn len(self) -> usize {
        match self {
            Self::Bits(words) => 64 * words.len(),
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
pub(crate) fn fold_rows_in_place<E>(
    table: &mut Vec<E>,
    lanes: usize,
    challenge: E,
    kernel: &'static str,
) where
    E: Field + Fold + Send + Sync,
{
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

/// Bucket sums over `items`, accumulated in one private table per worker.
///
/// A Rayon `fold` would allocate, zero, and reduce one full bucket table per
/// split, which dominates when the table is large (the source-link quartet round has
/// `16^4` buckets per source). Fixed contiguous chunks, at most one per
/// thread and each at least a table's width of items, bound that overhead;
/// the tables are then summed column-block by column-block. Field addition
/// is associative and commutative, so the sums equal the serial path's.
#[cfg(feature = "parallel")]
pub(crate) fn chunked_bucket_sums<E, A>(items: usize, width: usize, accumulate: A) -> Vec<E>
where
    E: Field + Send + Sync,
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

#[cfg(test)]
mod tests {
    use super::*;

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
