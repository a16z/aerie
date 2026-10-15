//! Reverse contraction of the batching weights `lambda` through the cores.
//!
//! `f_lambda = T^T lambda` factors as `f_lambda(alpha, rho) = sum_c A_1[c,
//! alpha] U_c(rho)` over the 203 first-core channels (`specs/falcon.md`
//! Section 6.3). The prover needs every `U_c(rho)`; the verifier needs only
//! `U_c~(r_rho)`, which it gets by contracting each later core's radix digit
//! against the equality weights of that core's slice of `r_rho`, then an
//! equality-weight dot product over the first block.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use crate::protocol::PARALLEL_MIN_WORK;

use crate::error::AggregationError;
use crate::substrate::OpeningField as E;
use jolt_field::{Unreduced as _, Zero as _};

use super::super::algebra::eq_table;
use super::super::FalconDegree;
use super::cores::{Core, ProjectionCores};
use super::profile::{
    Geometry, FIRST_CHANNELS, NORM_CHANNELS, NORM_GROUPS, NORM_PREFIX_LAYERS, NORM_PREFIX_WIDTH,
    NORM_TERMINAL_ROWS, OUTPUT_COUNT, RESIDUAL_BOUND, RESIDUAL_LAYERS, RESIDUAL_WIDTH,
};

type Product = <E as jolt_field::Unreduced>::Product;

const MULTIPLES: usize = 2 * RESIDUAL_BOUND as usize + 1;
const MULTIPLE_OFFSET: i8 = RESIDUAL_BOUND as i8;

/// `weight * e` for every entry `e` in `[-12, 12]`, indexed by `e + 12`.
fn small_multiples(weight: E) -> [E; MULTIPLES] {
    let mut table = [E::zero(); MULTIPLES];
    let offset = MULTIPLE_OFFSET as usize;
    for step in 1..=offset {
        let value = table[offset + step - 1] + weight;
        table[offset + step] = value;
        table[offset - step] = -value;
    }
    table
}

/// `out += weight * row`.
#[inline]
fn axpy_small(out: &mut [E], weight: E, row: &[i8]) {
    let multiples = small_multiples(weight);
    for (slot, &entry) in out.iter_mut().zip(row) {
        *slot += multiples[(entry + MULTIPLE_OFFSET) as usize];
    }
}

/// Rows per ternary lookup group.
const LOOKUP_ROWS: usize = 4;
/// Signed subset sums of one group: `3^LOOKUP_ROWS`.
const LOOKUP_SIZE: usize = 81;

/// Base-3 lookup indices of a stack of ternary rows.
///
/// Rows are grouped by four; for group `g` and column `col`,
/// `index[g * cols + col] = sum_i 3^i (row_{4g + i}[col] + 1)`, and a
/// missing row of the last group reads as entry zero. A column then costs
/// one lookup and one addition per four rows instead of four additions.
/// The layout is group-major so that both the build and the combination
/// stream over columns with independent work per column.
struct TernaryIndex {
    groups: usize,
    cols: usize,
    index: Vec<u8>,
}

impl TernaryIndex {
    /// `None` when some entry lies outside `{-1, 0, 1}`.
    fn new<'r>(rows: impl ExactSizeIterator<Item = &'r [i8]>, cols: usize) -> Option<Self> {
        let rows: Vec<&[i8]> = rows.collect();
        debug_assert!(rows.iter().all(|row| row.len() == cols));
        let groups = rows.len().div_ceil(LOOKUP_ROWS);
        let mut index = vec![0_u8; groups * cols];
        // `entry + 1 > 2` (as `u8`) marks an entry outside `{-1, 0, 1}`; the
        // flags are or-ed so the column loops stay branch-free.
        let mut invalid = 0_u8;
        for (digits, group_rows) in index.chunks_exact_mut(cols).zip(rows.chunks(LOOKUP_ROWS)) {
            let mut place = 1_u8;
            for row in group_rows {
                for (digit, &entry) in digits.iter_mut().zip(*row) {
                    let shifted = entry.wrapping_add(1) as u8;
                    invalid |= u8::from(shifted > 2);
                    // Wrapping: a non-ternary core is rejected below anyway.
                    *digit = digit.wrapping_add(place.wrapping_mul(shifted));
                }
                place *= 3;
            }
            // Missing rows of the last group contribute digit 1 each.
            let missing: u8 = (group_rows.len()..LOOKUP_ROWS)
                .map(|step| 3_u8.pow(step as u32))
                .sum();
            if missing != 0 {
                for digit in digits.iter_mut() {
                    *digit = digit.wrapping_add(missing);
                }
            }
        }
        (invalid == 0).then_some(Self {
            groups,
            cols,
            index,
        })
    }

    /// `out[col] = sum_r weights[r] row_r[col]` for the rows this index was
    /// built from; `tables` is scratch.
    fn combine(&self, weights: &[E], out: &mut [E], tables: &mut Vec<[E; LOOKUP_SIZE]>) {
        self.fill(weights, tables);
        self.combine_columns(tables, 0, out);
    }

    /// The signed subset sums of every group of `weights`.
    fn fill(&self, weights: &[E], tables: &mut Vec<[E; LOOKUP_SIZE]>) {
        tables.resize(self.groups, [E::zero(); LOOKUP_SIZE]);
        for (table, group) in tables.iter_mut().zip(weights.chunks(LOOKUP_ROWS)) {
            table[0] = E::zero();
            let mut len = 1;
            for digit in 0..LOOKUP_ROWS {
                let weight = group.get(digit).copied().unwrap_or_else(E::zero);
                for slot in 0..len {
                    let value = table[slot];
                    table[slot] = value - weight;
                    table[slot + len] = value;
                    table[slot + 2 * len] = value + weight;
                }
                len *= 3;
            }
        }
    }

    /// Columns `start..start + out.len()` from filled lookup tables.
    fn combine_columns(&self, tables: &[[E; LOOKUP_SIZE]], start: usize, out: &mut [E]) {
        let end = start + out.len();
        let mut groups = self.index.chunks_exact(self.cols).zip(tables);
        let Some((digits, table)) = groups.next() else {
            out.fill(E::zero());
            return;
        };
        for (slot, &digit) in out.iter_mut().zip(&digits[start..end]) {
            *slot = table[usize::from(digit)];
        }
        for (digits, table) in groups {
            for (slot, &digit) in out.iter_mut().zip(&digits[start..end]) {
                *slot += table[usize::from(digit)];
            }
        }
    }
}

/// One fiber of a transpose-applied core: `out[col] = sum_row input[row]
/// core[row][col]`, by lookups when the core is ternary.
fn backward_fiber(
    core: &Core,
    index: Option<&TernaryIndex>,
    input: &[E],
    out: &mut [E],
    tables: &mut Vec<[E; LOOKUP_SIZE]>,
) {
    if let Some(index) = index {
        index.combine(input, out, tables);
    } else {
        out.fill(E::zero());
        for (row, &weight) in input.iter().enumerate() {
            axpy_small(out, weight, core.row(row));
        }
    }
}

/// The lookup index of `core`, if it is ternary.
fn core_index(core: &Core) -> Option<TernaryIndex> {
    TernaryIndex::new((0..core.rows).map(|row| core.row(row)), core.cols)
}

/// Transpose-apply a core, keeping the contracted digit as a fiber:
/// `out[phi * cols + col] = sum_row input[phi * rows + row] core[row][col]`.
/// With `col = c + width u` the output is `[(u + radix phi)][c]`.
fn backward_keep(core: &Core, input: &[E]) -> Vec<E> {
    debug_assert!(input.len().is_multiple_of(core.rows));
    let fibers = input.len() / core.rows;
    let index = core_index(core);
    let mut out = vec![E::zero(); fibers * core.cols];
    let apply = |tables: &mut Vec<[E; LOOKUP_SIZE]>, (out, input): (&mut [E], &[E])| {
        backward_fiber(core, index.as_ref(), input, out, tables);
    };
    #[cfg(feature = "parallel")]
    if fibers > 1 && out.len() >= PARALLEL_MIN_WORK {
        out.par_chunks_mut(core.cols)
            .zip(input.par_chunks(core.rows))
            .for_each_init(Vec::new, apply);
        return out;
    }
    let mut tables = Vec::new();
    out.chunks_mut(core.cols)
        .zip(input.chunks(core.rows))
        .for_each(|item| apply(&mut tables, item));
    out
}

/// The last backward core of one leg, written straight into the rest-major
/// `U` table: input fiber `phi` of `core` (radix `radix`, `width` channels)
/// fills channels `offset..offset + width` of rest fibers `radix phi + u`.
fn backward_into_rest(core: &Core, input: &[E], out: &mut [E], offset: usize, width: usize) {
    let radix = core.cols / width;
    debug_assert_eq!(input.len() / core.rows * radix * FIRST_CHANNELS, out.len());
    let index = core_index(core);
    let apply = |(tables, fiber): &mut (Vec<[E; LOOKUP_SIZE]>, Vec<E>),
                 (out, input): (&mut [E], &[E])| {
        fiber.resize(core.cols, E::zero());
        backward_fiber(core, index.as_ref(), input, fiber, tables);
        for (rest, channels) in out
            .chunks_exact_mut(FIRST_CHANNELS)
            .zip(fiber.chunks_exact(width))
        {
            rest[offset..offset + width].copy_from_slice(channels);
        }
    };
    let chunk = radix * FIRST_CHANNELS;
    #[cfg(feature = "parallel")]
    if out.len() >= PARALLEL_MIN_WORK {
        out.par_chunks_mut(chunk)
            .zip(input.par_chunks(core.rows))
            .for_each_init(|| (Vec::new(), Vec::new()), apply);
        return;
    }
    let mut scratch = (Vec::new(), Vec::new());
    out.chunks_mut(chunk)
        .zip(input.chunks(core.rows))
        .for_each(|item| apply(&mut scratch, item));
}

/// Transpose-apply a core and contract its radix digit against `eq`:
/// `out[c] = sum_u eq[u] y[c + width u]` with `y[col] = sum_row input[row]
/// core[row][col]`.
///
/// `y` takes the four-row lookup path when `index` is the core's ternary
/// index, so a group of four rows costs one 81-entry table for the whole
/// core rather than one per radix digit. Field addition is exact, so the
/// regrouped sum is the same element.
fn backward_eq(core: &Core, index: Option<&TernaryIndex>, input: &[E], eq: &[E]) -> Vec<E> {
    debug_assert_eq!(input.len(), core.rows);
    debug_assert!(core.cols.is_multiple_of(eq.len()));
    let width = core.cols / eq.len();
    let mut transposed = vec![E::zero(); core.cols];
    backward_fiber(core, index, input, &mut transposed, &mut Vec::new());
    let mut out = vec![Product::zero(); width];
    for (&eq_weight, digit) in eq.iter().zip(transposed.chunks_exact(width)) {
        for (slot, &value) in out.iter_mut().zip(digit) {
            *slot += eq_weight.mul_unreduced(value);
        }
    }
    out.into_iter().map(E::reduce_product).collect()
}

/// Lookup indices of the cores that [`point_weights`] contracts, built once
/// and shared by its calls at different rest points.
#[doc(hidden)]
pub struct LaterCoreIndices {
    norm_terminal: Option<TernaryIndex>,
    /// `norm[group][layer - 2]` for layers `2..=NORM_PREFIX_LAYERS`.
    norm: Vec<Vec<Option<TernaryIndex>>>,
    /// `residual[layer - 2]` for layers `2..=RESIDUAL_LAYERS`.
    residual: Vec<Option<TernaryIndex>>,
}

impl LaterCoreIndices {
    #[must_use]
    pub fn new(cores: &ProjectionCores) -> Self {
        Self {
            norm_terminal: core_index(&cores.norm_terminal),
            norm: (0..NORM_GROUPS)
                .map(|group| {
                    (2..=NORM_PREFIX_LAYERS)
                        .map(|layer| core_index(cores.norm(group, layer)))
                        .collect()
                })
                .collect(),
            residual: (2..=RESIDUAL_LAYERS)
                .map(|layer| core_index(cores.residual(layer)))
                .collect(),
        }
    }
}

/// Split the rest point into the per-core slices of one leg.
fn rest_slices<'p>(point: &'p [E], bits: &[usize]) -> Vec<&'p [E]> {
    let mut slices = Vec::with_capacity(bits.len());
    let mut start = 0;
    for &count in bits {
        slices.push(&point[start..start + count]);
        start += count;
    }
    debug_assert_eq!(start, point.len());
    slices
}

/// Every `U_c(rho)` for the prover, rest-major: `out[rho * 203 + c]`.
#[doc(hidden)]
#[must_use]
pub fn rest_weights(cores: &ProjectionCores, geometry: &Geometry, lambda: &[E]) -> Vec<E> {
    let (lambda_norm, lambda_residual) = lambda.split_at(NORM_TERMINAL_ROWS);
    let fibers = geometry.rest_size();
    let r5 = geometry.norm_radices()[NORM_PREFIX_LAYERS];
    let concatenated = backward_keep(&cores.norm_terminal, lambda_norm);
    debug_assert_eq!(concatenated.len(), r5 * NORM_CHANNELS);
    let mut out = vec![E::zero(); fibers * FIRST_CHANNELS];
    for group in 0..NORM_GROUPS {
        let mut table: Vec<E> = concatenated
            .chunks_exact(NORM_CHANNELS)
            .flat_map(|channels| {
                channels[NORM_PREFIX_WIDTH * group..][..NORM_PREFIX_WIDTH]
                    .iter()
                    .copied()
            })
            .collect();
        for layer in (3..=NORM_PREFIX_LAYERS).rev() {
            table = backward_keep(cores.norm(group, layer), &table);
        }
        backward_into_rest(
            cores.norm(group, 2),
            &table,
            &mut out,
            NORM_PREFIX_WIDTH * group,
            NORM_PREFIX_WIDTH,
        );
    }
    let mut residual = lambda_residual.to_vec();
    for layer in (3..=RESIDUAL_LAYERS).rev() {
        residual = backward_keep(cores.residual(layer), &residual);
    }
    backward_into_rest(
        cores.residual(2),
        &residual,
        &mut out,
        NORM_CHANNELS,
        RESIDUAL_WIDTH,
    );
    out
}

/// `U_c~(r_rho)` for the verifier: the 172 norm then 31 residual channels,
/// or only the norm channels when `lambda_residual` is `None`.
#[doc(hidden)]
pub fn point_weights(
    cores: &ProjectionCores,
    indices: &LaterCoreIndices,
    geometry: &Geometry,
    lambda_norm: &[E],
    lambda_residual: Option<&[E]>,
    rest_point: &[E],
) -> Vec<E> {
    debug_assert_eq!(rest_point.len(), geometry.rest_bits);
    let norm_eq: Vec<Vec<E>> = rest_slices(rest_point, &geometry.norm_bits[1..])
        .into_iter()
        .map(eq_table)
        .collect();
    // norm_eq[i - 2] belongs to norm core i (2..=5); residual_eq[i - 2] to
    // residual core i (2..=3).
    let concatenated = backward_eq(
        &cores.norm_terminal,
        indices.norm_terminal.as_ref(),
        lambda_norm,
        &norm_eq[NORM_PREFIX_LAYERS - 1],
    );
    let mut out = Vec::with_capacity(FIRST_CHANNELS);
    for group in 0..NORM_GROUPS {
        let mut vector = concatenated[NORM_PREFIX_WIDTH * group..][..NORM_PREFIX_WIDTH].to_vec();
        for layer in (2..=NORM_PREFIX_LAYERS).rev() {
            vector = backward_eq(
                cores.norm(group, layer),
                indices.norm[group][layer - 2].as_ref(),
                &vector,
                &norm_eq[layer - 2],
            );
        }
        out.extend_from_slice(&vector);
    }
    let Some(lambda_residual) = lambda_residual else {
        debug_assert_eq!(out.len(), NORM_CHANNELS);
        return out;
    };
    let residual_eq: Vec<Vec<E>> = rest_slices(rest_point, &geometry.residual_bits[1..])
        .into_iter()
        .map(eq_table)
        .collect();
    let mut vector = lambda_residual.to_vec();
    for layer in (2..=RESIDUAL_LAYERS).rev() {
        vector = backward_eq(
            cores.residual(layer),
            indices.residual[layer - 2].as_ref(),
            &vector,
            &residual_eq[layer - 2],
        );
    }
    out.extend_from_slice(&vector);
    debug_assert_eq!(out.len(), FIRST_CHANNELS);
    out
}

/// Sum of weighted first-core rows: `out[col] = sum_c weights[c] core_c[col]`.
fn weighted_rows(rows: &[(&[i8], E)], cols: usize) -> Vec<E> {
    const COLUMN_CHUNK: usize = 512;
    let mut out = vec![E::zero(); cols];
    if let Some(index) = TernaryIndex::new(rows.iter().map(|&(row, _)| row), cols) {
        let weights: Vec<E> = rows.iter().map(|&(_, weight)| weight).collect();
        let mut tables = Vec::new();
        index.fill(&weights, &mut tables);
        let apply = |(chunk, out): (usize, &mut [E])| {
            index.combine_columns(&tables, chunk * COLUMN_CHUNK, out);
        };
        #[cfg(feature = "parallel")]
        if cols >= PARALLEL_MIN_WORK {
            out.par_chunks_mut(COLUMN_CHUNK).enumerate().for_each(apply);
            return out;
        }
        out.chunks_mut(COLUMN_CHUNK).enumerate().for_each(apply);
        return out;
    }
    let apply = |(chunk, out): (usize, &mut [E])| {
        let start = chunk * COLUMN_CHUNK;
        for &(row, weight) in rows {
            axpy_small(out, weight, &row[start..start + out.len()]);
        }
    };
    #[cfg(feature = "parallel")]
    if cols >= PARALLEL_MIN_WORK {
        out.par_chunks_mut(COLUMN_CHUNK).enumerate().for_each(apply);
        return out;
    }
    out.chunks_mut(COLUMN_CHUNK).enumerate().for_each(apply);
    out
}

/// Norm first-block weights `sum_{g,c} u[43 g + c] A_{g,1}[c, alpha_S]`
/// over `alpha_S = j + n a + 2n i_{<k}` (4096 entries).
#[doc(hidden)]
#[must_use]
pub fn first_norm_weights(cores: &ProjectionCores, channel_weights: &[E]) -> Vec<E> {
    let cols = cores.norm(0, 1).cols;
    let rows: Vec<(&[i8], E)> = (0..NORM_GROUPS)
        .flat_map(|group| {
            let core = cores.norm(group, 1);
            (0..NORM_PREFIX_WIDTH).map(move |channel| {
                (
                    core.row(channel),
                    channel_weights[NORM_PREFIX_WIDTH * group + channel],
                )
            })
        })
        .collect();
    weighted_rows(&rows, cols)
}

/// Residual first-block weights `sum_c u[c] B_1[c, alpha_Delta]` over
/// `alpha_Delta = j + n i_{<k}` (2048 entries).
#[doc(hidden)]
#[must_use]
pub fn first_residual_weights(cores: &ProjectionCores, channel_weights: &[E]) -> Vec<E> {
    let core = cores.residual(1);
    let rows: Vec<(&[i8], E)> = (0..RESIDUAL_WIDTH)
        .map(|channel| (core.row(channel), channel_weights[channel]))
        .collect();
    weighted_rows(&rows, core.cols)
}

fn slack_first_norm_weights(
    geometry: &Geometry,
    cores: &ProjectionCores,
    channel_weights: &[E],
) -> Vec<E> {
    let mut out = vec![E::zero(); 4 * geometry.low_records()];
    for group in 0..NORM_GROUPS {
        let core = cores.norm(group, 1);
        for channel in 0..NORM_PREFIX_WIDTH {
            let weight = channel_weights[group * NORM_PREFIX_WIDTH + channel];
            let row = core.row(channel);
            for record_low in 0..geometry.low_records() {
                for slot in 0..4 {
                    let column = geometry.n + slot + 2 * geometry.n * record_low;
                    match row[column] {
                        -1 => out[slot + 4 * record_low] -= weight,
                        0 => {}
                        1 => out[slot + 4 * record_low] += weight,
                        value => debug_assert!(
                            (-1..=1).contains(&value),
                            "derived norm core entry is not ternary"
                        ),
                    }
                }
            }
        }
    }
    out
}

/// Public transpose evaluations used by the stage-one terminal check.
/// Returns `(w_S1(r_pi), w_e(r_pi,e), w_Delta(r_pi))`.
#[doc(hidden)]
pub fn projection_weights_at(
    degree: FalconDegree,
    records: usize,
    live_rows: usize,
    seed: &[u8],
    lambda: &[E],
    coefficient_point: &[E],
    slack_point: &[E],
) -> Result<[E; 3], AggregationError> {
    if lambda.len() != OUTPUT_COUNT
        || coefficient_point.len() != degree.logn() + records.trailing_zeros() as usize
        || slack_point.len() != 2 + records.trailing_zeros() as usize
    {
        return Err(AggregationError::internal(
            "stage-one projection terminal shape mismatch",
        ));
    }
    let geometry = Geometry::new(degree, records)?;
    let cores = ProjectionCores::derive(seed, &geometry, live_rows);
    let (lambda_norm, lambda_residual) = lambda.split_at(NORM_TERMINAL_ROWS);
    let (coef, row) = coefficient_point.split_at(geometry.logn);
    let (low, high) = row.split_at(geometry.low_record_bits);
    let indices = LaterCoreIndices::new(&cores);
    let channels = point_weights(
        &cores,
        &indices,
        &geometry,
        lambda_norm,
        Some(lambda_residual),
        high,
    );
    let norm = first_norm_weights(&cores, &channels[..NORM_CHANNELS]);
    let residual = first_residual_weights(&cores, &channels[NORM_CHANNELS..]);

    let norm_point = {
        let mut point = coef.to_vec();
        point.push(E::zero());
        point.extend_from_slice(low);
        point
    };
    let residual_point = {
        let mut point = coef.to_vec();
        point.extend_from_slice(low);
        point
    };
    let w_norm_s1 = mle(&norm, &norm_point);
    let w_delta = mle(&residual, &residual_point);

    let (slack_slot, slack_row) = slack_point.split_at(2);
    let (slack_low, slack_high) = slack_row.split_at(geometry.low_record_bits);
    let norm_channels = point_weights(&cores, &indices, &geometry, lambda_norm, None, slack_high);
    let slack_table = slack_first_norm_weights(&geometry, &cores, &norm_channels);
    let mut canonical_slack_point = slack_slot.to_vec();
    canonical_slack_point.extend_from_slice(slack_low);
    let w_e = mle(&slack_table, &canonical_slack_point);
    Ok([w_norm_s1, w_e, w_delta])
}

fn mle(table: &[E], point: &[E]) -> E {
    eq_table(point)
        .iter()
        .zip(table)
        .fold(E::zero(), |sum, (&weight, &value)| sum + weight * value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::BaseField as F;
    use jolt_field::Ring as _;

    #[test]
    fn selected_slack_columns_match_full_first_norm_weights() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            let low_bits = 12 - (degree.logn() + 1);
            let geometry = Geometry::new(degree, 1 << (low_bits + 2)).unwrap();
            let cores = ProjectionCores::derive(&[0xa5; 32], &geometry, geometry.records - 1);
            let channel_weights = (0..NORM_CHANNELS)
                .map(|index| E::from_u64((index as u64 % 101) + 1))
                .collect::<Vec<_>>();

            let actual = slack_first_norm_weights(&geometry, &cores, &channel_weights);
            let full = first_norm_weights(&cores, &channel_weights);
            let expected = (0..geometry.low_records())
                .flat_map(|record_low| {
                    (0..4).map({
                        let full = &full;
                        move |slot| full[geometry.n + slot + 2 * geometry.n * record_low]
                    })
                })
                .collect::<Vec<_>>();

            assert_eq!(actual, expected, "{degree:?}");
        }
    }

    #[test]
    fn projection_ternary_lookup_matches_axpy() {
        let cols = 37;
        for rows in [1, 2, 3, 4, 5, 7, 8, 43] {
            let entries: Vec<i8> = (0..rows * cols)
                .map(|index| [1, -1, 0, 0, -1, 1, 1][(index * 5 + index / 3) % 7])
                .collect();
            let core = Core {
                rows,
                cols,
                entries,
            };
            let input: Vec<E> = (0..2 * rows)
                .map(|index| {
                    let seed = 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(index as u64 + 1);
                    E::new(F::from_u64(seed), F::from_u64(seed.rotate_left(29)))
                })
                .collect();
            let mut expected = vec![E::zero(); 2 * cols];
            for (out, input) in expected.chunks_mut(cols).zip(input.chunks(rows)) {
                for (row, &weight) in input.iter().enumerate() {
                    axpy_small(out, weight, core.row(row));
                }
            }
            assert!(core_index(&core).is_some());
            assert_eq!(backward_keep(&core, &input), expected, "rows {rows}");
        }
        let uniform = Core {
            rows: 1,
            cols: 2,
            entries: vec![2, 0],
        };
        assert!(core_index(&uniform).is_none());
    }

    /// The reference `backward_eq`: one small axpy per `(row, digit)` slice.
    fn backward_eq_reference(core: &Core, input: &[E], eq: &[E]) -> Vec<E> {
        let width = core.cols / eq.len();
        let mut out = vec![E::zero(); width];
        for (row, &weight) in input.iter().enumerate() {
            for (digit, &eq_weight) in eq.iter().enumerate() {
                axpy_small(
                    &mut out,
                    weight * eq_weight,
                    &core.row(row)[digit * width..][..width],
                );
            }
        }
        out
    }

    #[test]
    fn backward_eq_lookup_matches_axpy_for_ternary_and_wide_cores() {
        let element = |index: usize| {
            let seed = 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(index as u64 + 1);
            E::new(F::from_u64(seed), F::from_u64(seed.rotate_left(29)))
        };
        for (rows, width, radix) in [
            (1, 5, 2),
            (2, 6, 1),
            (3, 7, 4),
            (4, 43, 2),
            (5, 31, 8),
            (43, 9, 4),
        ] {
            let cols = width * radix;
            let ternary: Vec<i8> = (0..rows * cols)
                .map(|index| [1, -1, 0, 0, -1, 1, 1][(index * 5 + index / 3) % 7])
                .collect();
            // One entry outside {-1, 0, 1} forces the axpy fallback.
            let mut wide: Vec<i8> = (0..rows * cols)
                .map(|index| ((index * 7 + 3) % 25) as i8 - 12)
                .collect();
            wide[0] = -12;
            let input: Vec<E> = (0..rows).map(element).collect();
            let eq: Vec<E> = (0..radix).map(|index| element(1_000 + index)).collect();
            for (label, entries) in [("ternary", ternary), ("wide", wide)] {
                let core = Core {
                    rows,
                    cols,
                    entries,
                };
                assert_eq!(
                    core_index(&core).is_some(),
                    label == "ternary",
                    "{label} {rows}x{cols}"
                );
                assert_eq!(
                    backward_eq(&core, core_index(&core).as_ref(), &input, &eq),
                    backward_eq_reference(&core, &input, &eq),
                    "{label} {rows}x{cols}"
                );
            }
        }
    }
}
