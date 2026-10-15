//! The forward projection: `y = T (..) S` and `z = B_3 B_2 B_1 Delta`.
//!
//! The first cores dominate. They run over the witness in integer
//! arithmetic, one rest fiber `rho` (a block of `2^k` consecutive records)
//! at a time, and their outputs are retained as the `P1` tables stage one
//! reuses, as is `Delta` (see [`RetainedDelta`]). The later cores act on the
//! `P1` tables in `i64` or `i128`, exact for every `i16` witness (bounds at
//! each step below), and the outputs are reduced into `F_p` once at the end.
//! No dense matrix is formed.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::substrate::BaseField as F;
use jolt_field::Ring as _;

use super::super::FalconWitness;
use super::cores::{Core, ProjectionCores};
use super::profile::{
    Geometry, FIRST_CHANNELS, NORM_CHANNELS, NORM_GROUPS, NORM_PREFIX_LAYERS, NORM_PREFIX_WIDTH,
    NORM_TERMINAL_ROWS, RESIDUAL_WIDTH,
};
use super::sources::delta_record;
use super::ProjectionOutputs;

pub(crate) mod kernels;

/// Rest fibers per parallel job of the first-core pass.
const FIBERS_PER_JOB: usize = 4;
/// `Delta` values below this magnitude are retained as `i32` and split into two
/// `i16` halves for the first-core kernel.
const NARROW_RESIDUAL: i64 = 1 << 30;

/// `Delta` retained from the forward pass for the stage-one fold, so the negacyclic
/// product runs once per record.
///
/// A record whose `Delta` has every coefficient below [`NARROW_RESIDUAL`] in
/// magnitude (every honest record: `Delta = q v` with `|v| < 2^17`) lives in
/// `narrow`; any other record is kept exactly in `wide` and its `narrow` slots
/// are zero.
#[derive(Clone, Debug)]
pub(crate) struct RetainedDelta {
    /// `Delta` at flat index `j + n i`.
    pub(crate) narrow: Vec<i32>,
    /// `Delta_i` of the wide records `i`.
    pub(crate) wide: Vec<Option<Box<[i64]>>>,
}

/// The forward projection and its retained first-core outputs.
#[derive(Clone, Debug)]
pub struct ForwardProjection {
    pub(crate) geometry: Geometry,
    pub(crate) live_rows: usize,
    pub(crate) cores: ProjectionCores,
    pub(crate) outputs: ProjectionOutputs,
    /// `P1` before the norm slack contribution is added, rest-major. Channels
    /// `43 g + c` for `g < 4` are the norm group channels; channels `172..203`
    /// are the residual ones. Stage one links the coefficient source against
    /// this table directly.
    pub(crate) first_without_slack: Vec<i64>,
    /// `Delta`, for the fold in the projection source link.
    pub(crate) delta: RetainedDelta,
}

impl ForwardProjection {
    /// The public outputs `(y, z)`.
    #[must_use]
    pub fn outputs(&self) -> &ProjectionOutputs {
        &self.outputs
    }
}

/// First cores as `i16` matrices over the per-fiber source layouts.
///
/// - `norm_s1`: `172` rows over `j + n i_{<k}`, the fiber's contiguous `s_1`
///   (core column `j + 2n i_{<k}`, lane `a = 0`).
/// - `norm_slack`: `172` rows over `slot + 4 i_{<k}`, the fiber's
///   contiguous slack (core column `n + slot + 2n i_{<k}`, lane `a = 1`).
/// - `residual`: `31` rows over `j + n i_{<k}`, `B_1` itself.
///
/// Columns of lane `a = 1` past the four slack slots never meet a nonzero
/// source cell and are dropped.
struct FirstMatrices {
    norm_s1: Vec<i16>,
    norm_slack: Vec<i16>,
    residual: Vec<i16>,
}

impl FirstMatrices {
    fn new(cores: &ProjectionCores, geometry: &Geometry) -> Self {
        let n = geometry.n;
        let low_records = geometry.low_records();
        let mut norm_s1 = Vec::with_capacity(NORM_CHANNELS * low_records * n);
        let mut norm_slack = Vec::with_capacity(NORM_CHANNELS * 4 * low_records);
        for group in 0..NORM_GROUPS {
            let core = cores.norm(group, 1);
            for channel in 0..NORM_PREFIX_WIDTH {
                let row = core.row(channel);
                for low in 0..low_records {
                    let base = 2 * n * low;
                    norm_s1.extend(row[base..base + n].iter().map(|&e| i16::from(e)));
                    norm_slack.extend(row[base + n..base + n + 4].iter().map(|&e| i16::from(e)));
                }
            }
        }
        let residual = cores
            .residual(1)
            .entries()
            .iter()
            .map(|&e| i16::from(e))
            .collect();
        Self {
            norm_s1,
            norm_slack,
            residual,
        }
    }
}

/// Per-job scratch of the first-core pass.
struct FirstScratch {
    /// `Delta` of one record.
    record: Vec<i64>,
    /// `Delta` of one fiber with a wide record, `i64`.
    wide: Vec<i64>,
    /// Low and high `i16` halves of `Delta`, per fiber of the job.
    halves: Vec<i16>,
    /// Residual dot products of the halves.
    dots: Vec<i64>,
}

/// Split a narrow `Delta` value into `i16` halves: `w = lo + 2^16 hi`.
#[inline]
pub(crate) fn halves(w: i32) -> (i16, i16) {
    let lo = w as i16;
    let hi = ((w - i32::from(lo)) >> 16) as i16;
    (lo, hi)
}

/// The `P1` tables of both legs, and the retained `Delta`.
///
/// Norm channels: each output sums at most `2^k (n + 4) <= 2064` products
/// of a ternary entry and an `i16` cell, so it is below `2^27`. Residual
/// channels: each output sums `2^11` products of an entry of magnitude at
/// most 12 and `|Delta| < 2^41`, so it is below `2^56`. Narrow fibers use the
/// `i16` kernel on `Delta = lo + 2^16 hi` (`|lo|, |hi| <= 2^15`); a fiber with a
/// wide record uses exact `i64` dot products.
fn first_tables(
    batch: &FalconWitness,
    geometry: &Geometry,
    cores: &ProjectionCores,
) -> (Vec<i64>, Vec<i64>, RetainedDelta) {
    let n = geometry.n;
    let low_records = geometry.low_records();
    let fiber_cells = low_records * n;
    let matrices = FirstMatrices::new(cores, geometry);

    let mut first = vec![0_i64; geometry.rest_size() * FIRST_CHANNELS];
    let mut first_without_slack = vec![0_i64; first.len()];
    let mut residual = RetainedDelta {
        narrow: vec![0; geometry.records * n],
        wide: vec![None; geometry.records],
    };
    let init = || FirstScratch {
        record: vec![0; n],
        wide: vec![0; fiber_cells],
        halves: vec![0; 2 * FIBERS_PER_JOB * fiber_cells],
        dots: vec![0; 2 * FIBERS_PER_JOB * RESIDUAL_WIDTH],
    };
    let fill = |scratch: &mut FirstScratch,
                (job, (((out, without_slack), narrow), wide)): (
        usize,
        (
            ((&mut [i64], &mut [i64]), &mut [i32]),
            &mut [Option<Box<[i64]>>],
        ),
    )| {
        let fibers = out.len() / FIRST_CHANNELS;
        let first_fiber = job * FIBERS_PER_JOB;
        let record_range = first_fiber * low_records..(first_fiber + fibers) * low_records;

        // Norm channels: s_1 is contiguous per fiber, then slack.
        let s1 = &batch.s1[record_range.start * n..record_range.end * n];
        let sources: Vec<&[i16]> = s1.chunks_exact(fiber_cells).collect();
        kernels::dot_rows(
            &matrices.norm_s1,
            fiber_cells,
            &sources,
            out,
            FIRST_CHANNELS,
        );
        for (without_slack, out) in without_slack
            .chunks_exact_mut(FIRST_CHANNELS)
            .zip(out.chunks_exact(FIRST_CHANNELS))
        {
            without_slack[..NORM_CHANNELS].copy_from_slice(&out[..NORM_CHANNELS]);
        }
        let slack = &batch.slack[4 * record_range.start..4 * record_range.end];
        for (out, slack) in out
            .chunks_exact_mut(FIRST_CHANNELS)
            .zip(slack.chunks_exact(4 * low_records))
        {
            for (slot, row) in out[..NORM_CHANNELS]
                .iter_mut()
                .zip(matrices.norm_slack.chunks_exact(4 * low_records))
            {
                *slot += row
                    .iter()
                    .zip(slack)
                    .map(|(&a, &b)| i64::from(a) * i64::from(b))
                    .sum::<i64>();
            }
        }

        // Delta: one negacyclic product per record, retained.
        let mut fiber_narrow = [true; FIBERS_PER_JOB];
        for (offset, ((narrow, wide), record)) in narrow
            .chunks_exact_mut(n)
            .zip(wide.iter_mut())
            .zip(record_range.clone())
            .enumerate()
        {
            delta_record(batch, geometry.degree, record, &mut scratch.record);
            let fiber = offset / low_records;
            if scratch.record.iter().all(|w| w.abs() < NARROW_RESIDUAL) {
                for (slot, &w) in narrow.iter_mut().zip(&scratch.record) {
                    *slot = w as i32;
                }
            } else {
                fiber_narrow[fiber] = false;
                *wide = Some(scratch.record.clone().into_boxed_slice());
            }
        }

        // Residual channels.
        let mut halves_sources: Vec<&[i16]> = Vec::with_capacity(2 * fibers);
        let mut narrow_fibers = Vec::with_capacity(fibers);
        {
            let (lo_hi, _) = scratch.halves.split_at_mut(2 * fibers * fiber_cells);
            for (fiber, (lo_hi, narrow)) in lo_hi
                .chunks_exact_mut(2 * fiber_cells)
                .zip(narrow.chunks_exact(fiber_cells))
                .enumerate()
            {
                if !fiber_narrow[fiber] {
                    continue;
                }
                let (lo, hi) = lo_hi.split_at_mut(fiber_cells);
                for ((lo, hi), &w) in lo.iter_mut().zip(hi.iter_mut()).zip(narrow) {
                    (*lo, *hi) = halves(w);
                }
                narrow_fibers.push(fiber);
            }
        }
        for &fiber in &narrow_fibers {
            let lo_hi = &scratch.halves[2 * fiber * fiber_cells..2 * (fiber + 1) * fiber_cells];
            let (lo, hi) = lo_hi.split_at(fiber_cells);
            halves_sources.push(lo);
            halves_sources.push(hi);
        }
        kernels::dot_rows(
            &matrices.residual,
            fiber_cells,
            &halves_sources,
            &mut scratch.dots,
            RESIDUAL_WIDTH,
        );
        for (index, &fiber) in narrow_fibers.iter().enumerate() {
            let lo = &scratch.dots[2 * index * RESIDUAL_WIDTH..][..RESIDUAL_WIDTH];
            let hi = &scratch.dots[(2 * index + 1) * RESIDUAL_WIDTH..][..RESIDUAL_WIDTH];
            for ((slot, &lo), &hi) in out[fiber * FIRST_CHANNELS + NORM_CHANNELS..]
                [..RESIDUAL_WIDTH]
                .iter_mut()
                .zip(lo)
                .zip(hi)
            {
                *slot = lo + (hi << 16);
            }
        }
        for fiber in (0..fibers).filter(|&fiber| !fiber_narrow[fiber]) {
            let cells = &mut scratch.wide[..fiber_cells];
            for (low, cells) in cells.chunks_exact_mut(n).enumerate() {
                let offset = fiber * low_records + low;
                match &wide[offset] {
                    Some(record) => cells.copy_from_slice(record),
                    None => {
                        for (slot, &w) in cells.iter_mut().zip(&narrow[offset * n..][..n]) {
                            *slot = i64::from(w);
                        }
                    }
                }
            }
            let cells = &scratch.wide[..fiber_cells];
            for (slot, row) in out[fiber * FIRST_CHANNELS + NORM_CHANNELS..][..RESIDUAL_WIDTH]
                .iter_mut()
                .zip(matrices.residual.chunks_exact(fiber_cells))
            {
                *slot = row.iter().zip(cells).map(|(&a, &b)| i64::from(a) * b).sum();
            }
        }
        for (without_slack, out) in without_slack
            .chunks_exact_mut(FIRST_CHANNELS)
            .zip(out.chunks_exact(FIRST_CHANNELS))
        {
            without_slack[NORM_CHANNELS..].copy_from_slice(&out[NORM_CHANNELS..]);
        }
    };
    let job_cells = FIBERS_PER_JOB * fiber_cells;
    #[cfg(feature = "parallel")]
    first
        .par_chunks_mut(FIBERS_PER_JOB * FIRST_CHANNELS)
        .zip(first_without_slack.par_chunks_mut(FIBERS_PER_JOB * FIRST_CHANNELS))
        .zip(residual.narrow.par_chunks_mut(job_cells))
        .zip(residual.wide.par_chunks_mut(FIBERS_PER_JOB * low_records))
        .enumerate()
        .for_each_init(init, fill);
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = init();
        first
            .chunks_mut(FIBERS_PER_JOB * FIRST_CHANNELS)
            .zip(first_without_slack.chunks_mut(FIBERS_PER_JOB * FIRST_CHANNELS))
            .zip(residual.narrow.chunks_mut(job_cells))
            .zip(residual.wide.chunks_mut(FIBERS_PER_JOB * low_records))
            .enumerate()
            .for_each(|item| fill(&mut scratch, item));
    }
    (first, first_without_slack, residual)
}

/// Apply one later core: `out[phi][c'] = sum_col core[c'][col] *
/// input[phi * cols + col]`, where `col = c + width * u` enumerates the
/// incoming channel `c` and the contracted radix digit `u`.
///
/// `Acc` is the accumulator; `product` forms one term from an entry and an
/// input value. Callers pick types whose bounds make both exact.
fn forward_layer<In, Acc>(
    core: &Core,
    input: &[In],
    product: impl Fn(i8, In) -> Acc + Sync,
) -> Vec<Acc>
where
    In: Copy + Send + Sync,
    Acc: Copy + Send + Default + core::iter::Sum,
{
    let block = core.cols();
    debug_assert!(input.len().is_multiple_of(block));
    let fibers = input.len() / block;
    let mut out = vec![Acc::default(); fibers * core.rows()];
    let apply = |(out, input): (&mut [Acc], &[In])| {
        for (slot, row) in out.iter_mut().zip(core.entries().chunks_exact(block)) {
            *slot = row
                .iter()
                .zip(input)
                .map(|(&entry, &value)| product(entry, value))
                .sum();
        }
    };
    #[cfg(feature = "parallel")]
    if fibers > 1 {
        out.par_chunks_mut(core.rows())
            .zip(input.par_chunks(block))
            .for_each(apply);
        return out;
    }
    out.chunks_mut(core.rows())
        .zip(input.chunks(block))
        .for_each(apply);
    out
}

/// Apply one later core through the `i16` kernel: every input value is
/// split into `limbs` balanced 16-bit digits, `x = sum_l d_l 2^(16 l)`, each
/// digit slice is contracted exactly by [`kernels::dot_rows`], and the
/// slices recombine in `i128`.
///
/// Requires `-32768 S <= x <= 32767 S`, where
/// `S = sum_{l < limbs} 65536^l`, and core entries of magnitude at most 12.
/// Every digit then fits an `i16` and every slice dot has magnitude at most
/// `cols * 12 * 2^15`. Above [`kernels::MAX_WIDTH`], the wide fallback also
/// requires every entry times input to fit in `i64`.
///
/// The norm caller's inputs to cores 2, 3 and 4 have magnitudes below
/// `2^27`, `2^36.5` and `2^46`: [`first_tables`] bounds the first table,
/// and each ternary prefix core sums at most `43 * 16` inputs. These fit
/// two, three and three digits. The residual caller uses four digits for
/// [`first_tables`]'s inputs below `2^56`. Both callers also keep products
/// with core entries in `i64`.
fn forward_layer_split(core: &Core, input: &[i64], limbs: usize) -> Vec<i128> {
    let block = core.cols();
    if block > kernels::MAX_WIDTH {
        return forward_layer(core, input, term_i64_wide);
    }
    debug_assert!(input.len().is_multiple_of(block));
    let fibers = input.len() / block;
    let matrix: Vec<i16> = core
        .entries()
        .iter()
        .map(|&entry| i16::from(entry))
        .collect();
    let mut out = vec![0_i128; fibers * core.rows()];
    let apply = |(digits, dots): &mut (Vec<i16>, Vec<i64>), (out, input): (&mut [i128], &[i64])| {
        digits.resize(limbs * block, 0);
        dots.resize(limbs * core.rows(), 0);
        for (col, &value) in input.iter().enumerate() {
            let mut rest = value;
            for limb in 0..limbs {
                let digit = rest as i16;
                digits[limb * block + col] = digit;
                rest = (rest - i64::from(digit)) >> 16;
            }
            debug_assert_eq!(rest, 0, "later-core input exceeds its digit budget");
        }
        let sources: Vec<&[i16]> = digits.chunks_exact(block).collect();
        kernels::dot_rows(&matrix, block, &sources, dots, core.rows());
        for (row, slot) in out.iter_mut().enumerate() {
            *slot = (0..limbs)
                .map(|limb| i128::from(dots[limb * core.rows() + row]) << (16 * limb))
                .sum();
        }
    };
    #[cfg(feature = "parallel")]
    if fibers > 1 {
        out.par_chunks_mut(core.rows())
            .zip(input.par_chunks(block))
            .for_each_init(|| (Vec::new(), Vec::new()), apply);
        return out;
    }
    let mut scratch = (Vec::new(), Vec::new());
    out.chunks_mut(core.rows())
        .zip(input.chunks(block))
        .for_each(|item| apply(&mut scratch, item));
    out
}

/// Exact `i64` terms summed in `i128`.
#[inline]
fn term_i64_wide(entry: i8, value: i64) -> i128 {
    i128::from(i64::from(entry) * value)
}

/// Channels `start..start + width` of every fiber of `P1`.
fn channels(first: &[i64], start: usize, width: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(first.len() / FIRST_CHANNELS * width);
    for fiber in first.chunks_exact(FIRST_CHANNELS) {
        out.extend_from_slice(&fiber[start..start + width]);
    }
    out
}

/// Forward projection of a batch under derived cores.
///
/// Later-core magnitudes, for any `i16` witness: the norm channels grow
/// from `< 2^27` by a factor of at most `43 * 16` per prefix core to `< 2^56`
/// after core 4, so cores 2 to 4 are exact in `i64`; the terminal core sums
/// at most `172 * 8` terms below `2^56`, `< 2^67`, in `i128`. The residual
/// channels grow from `< 2^56` by a factor of at most `31 * 256 * 12` per
/// core: core 2 forms `i64` terms (`< 2^60`) summed in `i128` (`< 2^73`),
/// and core 3 is exact in `i128` (`< 2^90`).
pub(crate) fn forward(
    batch: &FalconWitness,
    geometry: &Geometry,
    cores: ProjectionCores,
) -> ForwardProjection {
    let (first, first_without_slack, delta) = tracing::info_span!("projection::forward.first")
        .in_scope(|| first_tables(batch, geometry, &cores));
    let outputs = tracing::info_span!("projection::forward.later").in_scope(|| {
        // Norm leg: groups through cores 2..4, concatenation, terminal core.
        let r5 = geometry.norm_radices()[NORM_PREFIX_LAYERS];
        let group_output = |group: usize| {
            let mut table = channels(&first, NORM_PREFIX_WIDTH * group, NORM_PREFIX_WIDTH);
            const _: () = assert!(NORM_PREFIX_LAYERS == 4);
            // Inputs of cores 2, 3, 4 stay below 2^27, 2^36.5 and 2^46: two,
            // three and three digits.
            for (layer, limbs) in (2..=NORM_PREFIX_LAYERS).zip([2, 3, 3]) {
                table = forward_layer_split(cores.norm(group, layer), &table, limbs)
                    .into_iter()
                    .map(|value| value as i64)
                    .collect();
            }
            debug_assert_eq!(table.len(), r5 * NORM_PREFIX_WIDTH);
            table
        };
        #[cfg(feature = "parallel")]
        let groups: Vec<Vec<i64>> = (0..NORM_GROUPS).into_par_iter().map(group_output).collect();
        #[cfg(not(feature = "parallel"))]
        let groups: Vec<Vec<i64>> = (0..NORM_GROUPS).map(group_output).collect();
        let mut concatenated = vec![0_i64; r5 * NORM_CHANNELS];
        for (group, table) in groups.iter().enumerate() {
            for (u5, channels) in table.chunks_exact(NORM_PREFIX_WIDTH).enumerate() {
                concatenated[u5 * NORM_CHANNELS + NORM_PREFIX_WIDTH * group..][..NORM_PREFIX_WIDTH]
                    .copy_from_slice(channels);
            }
        }
        let y_integer = forward_layer(cores.norm_terminal(), &concatenated, term_i64_wide);
        debug_assert_eq!(y_integer.len(), NORM_TERMINAL_ROWS);

        // Residual leg: cores 2..3.
        let table = channels(&first, NORM_CHANNELS, RESIDUAL_WIDTH);
        // |P1| < 2^56 fits four digits.
        let table = forward_layer_split(cores.residual(2), &table, 4);
        let z_integer = forward_layer(cores.residual(3), &table, |entry, value: i128| {
            i128::from(entry) * value
        });
        debug_assert_eq!(z_integer.len(), RESIDUAL_WIDTH);
        ProjectionOutputs {
            y: core::array::from_fn(|index| F::from_i128(y_integer[index])),
            z: core::array::from_fn(|index| F::from_i128(z_integer[index])),
        }
    });
    ForwardProjection {
        geometry: *geometry,
        live_rows: batch.live_rows,
        cores,
        outputs,
        first_without_slack,
        delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_layer_matches_wide_at_digit_interval_endpoints() {
        for limbs in 1..=4 {
            let sum: i128 = (0..limbs).map(|limb| 1_i128 << (16 * limb)).sum();
            let lower = -32768 * sum;
            let upper = 32767 * sum;
            // The four-digit lower endpoint is below the i64 input range.
            let lower = lower.max(i128::from(i64::MIN));
            let input = [lower, lower + 1, upper - 1, upper]
                .map(|value| i64::try_from(value).expect("i64 endpoint"));
            let entries = if limbs == 4 {
                // Unit entries keep the wide reference's products in i64.
                vec![1, 0, -1, 1, 0, 1, 1, -1, 1, 1, 0, 0]
            } else {
                vec![12, -12, -7, 9, -11, 3, 12, -12, 0, -1, 1, 0]
            };
            let core = Core::from_test_entries(3, input.len(), entries);
            assert_eq!(
                forward_layer_split(&core, &input, limbs),
                forward_layer(&core, &input, term_i64_wide),
                "limbs {limbs}"
            );
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "later-core input exceeds its digit budget")]
    fn split_layer_rejects_input_above_digit_interval() {
        let core = Core::from_test_entries(1, 1, vec![1]);
        forward_layer_split(&core, &[32767 * (1 + 65536) + 1], 2);
    }
}
