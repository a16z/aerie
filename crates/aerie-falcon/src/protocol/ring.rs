//! Public negacyclic functional and target evaluation for stage 3.
//!
//! The prover materializes coefficient-row weights with the factorized
//! negacyclic transform. The verifier evaluates their multilinear extension
//! at the stage-three point with the linear-time carry recurrence.

use crate::protocol::kernels::SignSource;
use crate::substrate::OpeningField as E;
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
use aerie_falcon_native::{Falcon512, FalconParams};
use akita_sumcheck::reduce_signed_accum;
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
use jolt_field::Fold;
#[cfg(test)]
use jolt_field::One as _;
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
use jolt_field::Ring as _;
use jolt_field::Unreduced;
use jolt_field::Zero;
#[cfg(feature = "parallel")]
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
use rayon::prelude::*;

#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
use super::algebra::eq_table;

mod exact_convolution;
mod functional;
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub use functional::functional_evaluation;
#[cfg(any(feature = "bench-internals", feature = "adversarial-internals"))]
pub use functional::functional_evaluation_canonical;
#[cfg(feature = "bench-internals")]
pub use functional::functional_evaluation_canonical_for_bench;
pub use functional::PublicRingKeys;
mod targets;
pub(super) use targets::target_evaluation_i16;
pub use targets::{target_evaluation, PublicTargets};

/// Falcon-512 coefficient variable count, kept for the fixed-degree tests.
#[cfg(test)]
pub(super) const COEFFICIENT_VARS: usize = 9;

/// Evaluate a base-coefficient inner product in the extension field with one
/// final reduction. Akita's `FpExt2<Fp64>` accumulator has 2^64-term headroom,
/// so a 512-term Falcon row is exact.
#[inline]
pub(super) fn delayed_base_inner_product<C>(coefficients: &[C], factors: &[E]) -> E
where
    C: Copy + Into<i64>,
{
    debug_assert_eq!(coefficients.len(), factors.len());
    delayed_base_inner_product_iter(coefficients.iter().copied(), factors)
}

#[inline]
fn delayed_base_inner_product_iter<C>(coefficients: impl Iterator<Item = C>, factors: &[E]) -> E
where
    C: Into<i64>,
{
    let (positive, negative) = coefficients.zip(factors).fold(
        (
            <E as Unreduced>::SmallProduct::zero(),
            <E as Unreduced>::SmallProduct::zero(),
        ),
        |(mut positive, mut negative), (coefficient, &factor)| {
            let coefficient = coefficient.into();
            let product = factor.mul_u64_unreduced(coefficient.unsigned_abs());
            if coefficient >= 0 {
                positive += product;
            } else {
                negative += product;
            }
            (positive, negative)
        },
    );
    reduce_signed_accum::<E>(positive, negative)
}

/// Peak bytes of the materialized `w_ring` table for one padded record count
/// at Falcon-512.
///
/// Every cell is an extension-field element after `r_Pi` is sampled, so this
/// storage cannot use an integer carrier such as `i16` or `i32`.
#[must_use]
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub const fn functional_table_storage_bytes(records: usize) -> usize {
    functional_table_storage_bytes_of(records, Falcon512::DEGREE)
}

/// Peak bytes of the materialized `w_ring` table for one padded record count
/// and ring degree.
#[must_use]
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub const fn functional_table_storage_bytes_of(records: usize, degree: usize) -> usize {
    records
        .saturating_mul(degree)
        .saturating_mul(core::mem::size_of::<E>())
}

#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
type Ctx = <E as Fold>::Ctx;

/// Allocation-free recursive core for [`negacyclic_transform`]. `scratch`
/// has length at least twice the current transform length and is reused at
/// every level. `fold_contexts` precomputes multiplication by each challenge
/// once for all rows in a functional table.
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
fn negacyclic_transform_into(
    values: &[E],
    fold_contexts: &[Ctx],
    output: &mut [E],
    output_offset: usize,
    output_stride: usize,
    scale: Option<E>,
    scratch: &mut [E],
) {
    let length = values.len();
    debug_assert!(scratch.len() >= 2 * length);
    if fold_contexts.is_empty() {
        debug_assert_eq!(length, 1);
        output[output_offset] = scale.map_or(values[0], |weight| values[0] * weight);
        return;
    }

    let half = length / 2;
    let fold_context = &fold_contexts[0];
    if length == 2 {
        // Write the last butterfly directly. Most recursive calls end here;
        // routing both scalars through scratch and another call only copies
        // values before the leaf writes the same strided output cells.
        let even = E::fold_one(fold_context, values[0], -values[1]);
        let odd = E::fold_one(fold_context, values[1], values[0]);
        output[output_offset] = scale.map_or(even, |weight| even * weight);
        output[output_offset + output_stride] = scale.map_or(odd, |weight| odd * weight);
        return;
    }
    if length == 4 {
        // Expand both recursion levels directly. The first butterfly creates
        // the even and odd lanes; the terminal butterflies then write the
        // same strided output positions without routing through scratch.
        debug_assert_eq!(fold_contexts.len(), 2);
        let next_context = &fold_contexts[1];
        let even_0 = E::fold_one(fold_context, values[0], -values[3]);
        let even_1 = E::fold_one(fold_context, values[2], values[1]);
        let odd_0 = E::fold_one(fold_context, values[1], values[0]);
        let odd_1 = E::fold_one(fold_context, values[3], values[2]);
        let outputs = [
            E::fold_one(next_context, even_0, -even_1),
            E::fold_one(next_context, odd_0, -odd_1),
            E::fold_one(next_context, even_1, even_0),
            E::fold_one(next_context, odd_1, odd_0),
        ];
        for (index, value) in outputs.into_iter().enumerate() {
            output[output_offset + index * output_stride] =
                scale.map_or(value, |weight| value * weight);
        }
        return;
    }
    let (inputs, recursive_scratch) = scratch.split_at_mut(length);
    let (even_input, odd_input) = inputs.split_at_mut(half);
    for index in 0..half {
        let a0 = values[2 * index];
        let a1 = values[2 * index + 1];
        let rotated = if index == 0 {
            E::zero() - values[length - 1]
        } else {
            values[2 * index - 1]
        };
        // EOR form: a + r * (b - a). Besides halving the extension-field
        // multiplications, `fold_one` delays reduction across each product.
        even_input[index] = E::fold_one(fold_context, a0, rotated);
        odd_input[index] = E::fold_one(fold_context, a1, a0);
    }

    let (even_input, odd_input) = inputs.split_at(half);
    negacyclic_transform_into(
        even_input,
        &fold_contexts[1..],
        output,
        output_offset,
        output_stride * 2,
        scale,
        recursive_scratch,
    );
    negacyclic_transform_into(
        odd_input,
        &fold_contexts[1..],
        output,
        output_offset + output_stride,
        output_stride * 2,
        scale,
        recursive_scratch,
    );
}

/// The normative factorized negacyclic transform, Equation (8.7):
/// computes `A (*) w` where `w[u] = eq(point, bits(u))`, in
/// `O(L log L)` extension-field operations.
#[must_use]
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn negacyclic_transform(values: &[E], point: &[E]) -> Vec<E> {
    assert_eq!(values.len(), 1 << point.len(), "transform length mismatch");
    let fold_contexts: Vec<_> = point.iter().copied().map(E::precompute).collect();
    let mut output = vec![E::zero(); values.len()];
    let mut scratch = vec![E::zero(); 2 * values.len()];
    negacyclic_transform_into(
        values,
        &fold_contexts,
        &mut output,
        0,
        1,
        None,
        &mut scratch,
    );
    output
}

/// Negacyclic adjoint of one public-key row, Equation (8.6).
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
fn adjoint<C>(h_row: &[C]) -> Vec<E>
where
    C: Copy + Sync + Into<i64>,
{
    let length = h_row.len();
    let mut adjoint = Vec::with_capacity(length);
    adjoint.push(E::from_i64(h_row[0].into()));
    for index in 1..length {
        // Widen before negating so compact signed carriers cannot overflow.
        adjoint.push(E::from_i64(-h_row[length - index].into()));
    }
    adjoint
}

/// The transposed negacyclic kernel `K_{h,r} = h^dagger (*) w^(0)`.
#[must_use]
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn negacyclic_kernel<C>(h_row: &[C], coefficient_point: &[E]) -> Vec<E>
where
    C: Copy + Sync + Into<i64>,
{
    negacyclic_transform(&adjoint(h_row), coefficient_point)
}

#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
struct NegacyclicWorkspace {
    scratch: Vec<E>,
}

#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
impl NegacyclicWorkspace {
    fn new(length: usize) -> Self {
        Self {
            scratch: vec![E::zero(); 2 * length],
        }
    }

    fn kernel<C>(
        &mut self,
        h_row: &[C],
        fold_contexts: &[Ctx],
        scale: Option<E>,
        destination: &mut [E],
    ) where
        C: Copy + Sync + Into<i64>,
    {
        let length = h_row.len();
        debug_assert_eq!(fold_contexts.len(), length.trailing_zeros() as usize);
        let half = length / 2;
        let (inputs, recursive_scratch) = self.scratch.split_at_mut(length);
        let (even_input, odd_input) = inputs.split_at_mut(half);
        let fold_context = &fold_contexts[0];
        let mut previous_odd = E::zero();
        for index in 0..half {
            let even_index = 2 * index;
            let odd_index = even_index + 1;
            let a0 = if even_index == 0 {
                E::from_i64(h_row[0].into())
            } else {
                E::from_i64(-h_row[length - even_index].into())
            };
            // Widen before negating so compact signed carriers cannot overflow.
            let a1 = E::from_i64(-h_row[length - odd_index].into());
            let rotated = if index == 0 {
                E::from_i64(h_row[1].into())
            } else {
                previous_odd
            };
            even_input[index] = E::fold_one(fold_context, a0, rotated);
            odd_input[index] = E::fold_one(fold_context, a1, a0);
            previous_odd = a1;
        }

        let (even_input, odd_input) = inputs.split_at(half);
        negacyclic_transform_into(
            even_input,
            &fold_contexts[1..],
            destination,
            0,
            2,
            scale,
            recursive_scratch,
        );
        negacyclic_transform_into(
            odd_input,
            &fold_contexts[1..],
            destination,
            1,
            2,
            scale,
            recursive_scratch,
        );
    }
}

#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
fn materialize_functional_table<C>(
    h: &[C],
    records: usize,
    coefficient_point: &[E],
    row_weights: Option<&[E]>,
    live_rows: Option<usize>,
) -> Vec<E>
where
    C: Copy + Sync + Into<i64>,
{
    let degree = 1_usize << coefficient_point.len();
    assert_eq!(h.len(), records * degree, "public-key table shape");
    let fold_contexts: Vec<_> = coefficient_point
        .iter()
        .copied()
        .map(E::precompute)
        .collect();
    let mut table = vec![E::zero(); records * degree];
    let fill_row = |workspace: &mut NegacyclicWorkspace, record: usize, destination: &mut [E]| {
        let h_row = &h[record * degree..(record + 1) * degree];
        if live_rows.map_or_else(
            || h_row.iter().all(|&value| value.into() == 0),
            |live_rows| record >= live_rows,
        ) {
            return;
        }
        workspace.kernel(
            h_row,
            &fold_contexts,
            row_weights.map(|weights| weights[record]),
            destination,
        );
    };
    // Rows are independent and write disjoint chunks.
    #[cfg(feature = "parallel")]
    table.par_chunks_mut(degree).enumerate().for_each_init(
        || NegacyclicWorkspace::new(degree),
        |workspace, (record, destination)| fill_row(workspace, record, destination),
    );
    #[cfg(not(feature = "parallel"))]
    {
        let mut workspace = NegacyclicWorkspace::new(degree);
        table
            .chunks_mut(degree)
            .enumerate()
            .for_each(|(record, destination)| fill_row(&mut workspace, record, destination));
    }
    table
}

/// Prover materialization of the public functional table `w_ring`,
/// Equation (8.4): one kernel per live row scaled by the row weight.
#[must_use]
#[tracing::instrument(
    skip_all,
    name = "functional_table",
    fields(
        cells = h.len(),
        storage_bytes = h.len().saturating_mul(core::mem::size_of::<E>())
    )
)]
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn functional_table<C>(h: &[C], records: usize, projection_point: &[E]) -> Vec<E>
where
    C: Copy + Sync + Into<i64>,
{
    let (coefficient_point, row_point) = projection_point.split_at(coefficient_vars_of(h, records));
    let row_weights = eq_table(row_point);
    materialize_functional_table(h, records, coefficient_point, Some(&row_weights), None)
}

/// Stage-three materialization of the unscaled public kernels. The prover folds
/// all coefficient variables before applying each row's public equality
/// weight, avoiding one extension-field multiplication per table cell.
#[must_use]
#[tracing::instrument(
    skip_all,
    name = "functional_kernel_table",
    fields(
        cells = h.len(),
        storage_bytes = h.len().saturating_mul(core::mem::size_of::<E>())
    )
)]
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn functional_kernel_table<C>(
    h: &[C],
    records: usize,
    live_rows: usize,
    coefficient_point: &[E],
) -> Vec<E>
where
    C: Copy + Sync + Into<i64>,
{
    assert!(live_rows <= records, "live rows exceed padded rows");
    materialize_functional_table(h, records, coefficient_point, None, Some(live_rows))
}

/// Signed-coefficient specialization used by stage three. Falcon-512 and
/// Falcon-1024 use exact limb-split negacyclic convolution; other degrees use
/// the general factorized transform.
#[must_use]
#[tracing::instrument(skip_all, name = "functional_kernel_table")]
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn functional_kernel_table_i16(
    h: &[i16],
    records: usize,
    live_rows: usize,
    coefficient_point: &[E],
) -> Vec<E> {
    assert!(live_rows <= records, "live rows exceed padded rows");
    match coefficient_point.len() {
        9 => exact_convolution::table::<512>(h, records, live_rows, coefficient_point),
        10 => exact_convolution::table::<1024>(h, records, live_rows, coefficient_point),
        _ => functional_kernel_table(h, records, live_rows, coefficient_point),
    }
}

/// Materialize the signed public ring table and its first raw coefficient
/// triple in the same row-generation pass. `row_scale` contains the existing
/// `gamma * eq(row_point, row)` factors; it scales the round message only.
#[must_use]
#[tracing::instrument(
    skip_all,
    name = "functional_kernel_table_round0",
    fields(
        cells = h.len(),
        storage_bytes = h.len().saturating_mul(core::mem::size_of::<E>())
    )
)]
pub(super) fn functional_kernel_table_i16_with_round(
    h: &[i16],
    records: usize,
    live_rows: usize,
    coefficient_point: &[E],
    sigma: SignSource<'_>,
    magnitude: &[i16],
    row_scale: &[E],
) -> (Vec<E>, [E; 3]) {
    assert!(live_rows <= records, "live rows exceed padded rows");
    match coefficient_point.len() {
        9 => exact_convolution::table_with_round0::<512>(
            h,
            records,
            live_rows,
            coefficient_point,
            sigma,
            magnitude,
            row_scale,
        ),
        10 => exact_convolution::table_with_round0::<1024>(
            h,
            records,
            live_rows,
            coefficient_point,
            sigma,
            magnitude,
            row_scale,
        ),
        _ => panic!("fused ring table supports Falcon-512 and Falcon-1024"),
    }
}

/// Direct quadratic reference for [`functional_table`]: Equation (8.2)
/// summed cell by cell under the Equation (8.4) row weight. Section 12
/// benchmark comparison only; the prover MUST use the factorized
/// transform, so keep this off every proving path.
#[must_use]
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn functional_table_direct<C>(h: &[C], records: usize, projection_point: &[E]) -> Vec<E>
where
    C: Copy + Sync + Into<i64>,
{
    let (coefficient_point, row_point) = projection_point.split_at(coefficient_vars_of(h, records));
    let degree = 1_usize << coefficient_point.len();
    let row_weights = eq_table(row_point);
    let omega = eq_table(coefficient_point);
    let mut table = vec![E::zero(); records * degree];
    let fill_row = |record: usize, destination: &mut [E]| {
        let h_row = &h[record * degree..(record + 1) * degree];
        if h_row.iter().all(|&value| value.into() == 0) {
            return;
        }
        let weight = row_weights[record];
        for (j, cell) in destination.iter_mut().enumerate() {
            let mut kernel = E::zero();
            for (a, &h_a) in h_row.iter().enumerate() {
                let h_a = h_a.into();
                if h_a == 0 {
                    continue;
                }
                let term = E::from_i64(h_a) * omega[(a + j) % degree];
                if (a + j) / degree == 1 {
                    kernel -= term;
                } else {
                    kernel += term;
                }
            }
            *cell = weight * kernel;
        }
    };
    #[cfg(feature = "parallel")]
    table
        .par_chunks_mut(degree)
        .enumerate()
        .for_each(|(record, destination)| fill_row(record, destination));
    #[cfg(not(feature = "parallel"))]
    table
        .chunks_mut(degree)
        .enumerate()
        .for_each(|(record, destination)| fill_row(record, destination));
    table
}

/// `log2 n` of a row-major public-key table over `records` padded rows.
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
fn coefficient_vars_of<C>(h: &[C], records: usize) -> usize {
    assert!(records > 0, "a functional table needs at least one row");
    let degree = h.len() / records;
    assert!(
        degree.is_power_of_two() && degree * records == h.len(),
        "public-key table shape"
    );
    degree.trailing_zeros() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::algebra::{evaluate_mle, lift_table, sample_point};
    use crate::protocol::transcript::TestTranscript;
    use aerie_falcon_statement::{DirectBatch, DirectRow};

    fn random_point(label: &[u8], count: usize) -> Vec<E> {
        sample_point(
            &mut TestTranscript::new(b"aerie/falcon/v2/ring-test"),
            core::str::from_utf8(label).unwrap(),
            count,
        )
        .unwrap()
    }

    /// Direct quadratic reference for Equation (8.2).
    fn direct_kernel(h_row: &[i64], point: &[E]) -> Vec<E> {
        let length = 1_usize << point.len();
        let weights = eq_table(point);
        (0..length)
            .map(|j| {
                (0..length).fold(E::zero(), |sum, a| {
                    let wrapped = (a + j) % length;
                    let sign = if (a + j) / length == 1 {
                        E::zero() - E::one()
                    } else {
                        E::one()
                    };
                    sum + E::from_i64(h_row[a]) * sign * weights[wrapped]
                })
            })
            .collect()
    }

    #[test]
    fn factorized_transform_matches_the_direct_kernel_at_every_size() {
        for bits in 1..=9_usize {
            let length = 1_i64 << bits;
            let h_row: Vec<i64> = (0..length).map(|a| (a * 37 + 11) % 61 - 30).collect();
            let compact_h = h_row
                .iter()
                .map(|&coefficient| i16::try_from(coefficient).expect("small test coefficient"))
                .collect::<Vec<_>>();
            let point = random_point(b"kernel point", bits);
            assert_eq!(
                negacyclic_kernel(&h_row, &point),
                direct_kernel(&h_row, &point),
                "length {length}"
            );
            assert_eq!(
                negacyclic_kernel(&compact_h, &point),
                direct_kernel(&h_row, &point),
                "compact length {length}"
            );
        }
    }

    #[test]
    fn compact_kernel_widens_before_negation() {
        let compact = [17_i16, i16::MIN];
        let wide = compact.map(i64::from);
        let point = random_point(b"compact kernel point", 1);
        assert_eq!(
            negacyclic_kernel(&compact, &point),
            direct_kernel(&wide, &point),
        );
    }

    #[test]
    fn kernel_computes_the_negacyclic_correlation_functional() {
        // sum_j K[j] s2[j] must equal the MLE of the negacyclic product
        // h (*) s2 at the coefficient point.
        let bits = 9;
        let length = 1_usize << bits;
        let h_row: Vec<i64> = (0..length as i64).map(|a| (a * 13 + 5) % 21 - 10).collect();
        let s2_row: Vec<i64> = (0..length as i64).map(|a| (a * 7 + 3) % 15 - 7).collect();
        let point = random_point(b"functional point", bits);
        let kernel = negacyclic_kernel(&h_row, &point);
        let functional = kernel
            .iter()
            .zip(&s2_row)
            .fold(E::zero(), |sum, (&k, &s)| sum + k * E::from_i64(s));
        let mut product = vec![0_i64; length];
        for (a, &ha) in h_row.iter().enumerate() {
            for (b, &sb) in s2_row.iter().enumerate() {
                if a + b < length {
                    product[a + b] += ha * sb;
                } else {
                    product[a + b - length] -= ha * sb;
                }
            }
        }
        assert_eq!(functional, evaluate_mle(&lift_table(&product), &point));
    }

    #[test]
    fn ring_functional_matches_direct_negacyclic_reference_at_both_degrees() {
        for bits in [9_usize, 10] {
            let degree = 1_usize << bits;
            let h: Vec<i64> = (0..degree)
                .map(|a| i64::try_from((a * 73 + 19) % 127).unwrap() - 63)
                .collect();
            let s2: Vec<i64> = (0..degree)
                .map(|a| i64::try_from((a * 47 + 31) % 89).unwrap() - 44)
                .collect();
            let point = random_point(b"random negacyclic reference point", bits);
            let kernel = negacyclic_kernel(&h, &point);
            let functional = kernel
                .iter()
                .zip(&s2)
                .fold(E::zero(), |sum, (&weight, &coefficient)| {
                    sum + weight * E::from_i64(coefficient)
                });

            let mut product = vec![0_i64; degree];
            for (a, &h_a) in h.iter().enumerate() {
                for (b, &s2_b) in s2.iter().enumerate() {
                    let coefficient = h_a * s2_b;
                    if a + b < degree {
                        product[a + b] += coefficient;
                    } else {
                        product[a + b - degree] -= coefficient;
                    }
                }
            }
            assert_eq!(functional, evaluate_mle(&lift_table(&product), &point));
        }
    }

    #[test]
    fn direct_table_matches_the_factorized_table() {
        let records = 4_usize;
        let rows = (0..records)
            .map(|seed| {
                DirectRow::<aerie_falcon_native::Falcon512>::synthetic(seed as u64 + 91)
                    .expect("synthetic row")
            })
            .collect::<Vec<_>>();
        let batch = DirectBatch::from_rows(&rows);
        let row_vars = records.trailing_zeros() as usize;
        let projection_point = random_point(b"direct table point", COEFFICIENT_VARS + row_vars);
        assert_eq!(
            functional_table_direct(&batch.h, records, &projection_point),
            functional_table(&batch.h, records, &projection_point),
        );
    }

    #[test]
    fn carry_recurrence_matches_the_materialized_functional_table() {
        for bits in [9_usize, 10] {
            let degree = 1 << bits;
            let records = 4;
            let h: Vec<i16> = (0..degree * records)
                .map(|i| {
                    if i >= 3 * degree {
                        0
                    } else {
                        ((i * 73 + 29) % 12289) as i16 - 6144
                    }
                })
                .collect();
            let mut transcript = TestTranscript::new(b"aerie/falcon/v2/ring-test");
            for _ in 0..3 {
                let projection =
                    sample_point(&mut transcript, "projection point", bits + 2).unwrap();
                let terminal = sample_point(&mut transcript, "terminal point", bits + 2).unwrap();
                let table = functional_table(&h, records, &projection);
                let direct = evaluate_mle(&table, &terminal);
                assert_eq!(
                    functional_evaluation(&h, records, &projection, &terminal),
                    direct
                );
                let kernels = functional_kernel_table_i16(&h, records, 3, &projection[..bits]);
                let rows = eq_table(&projection[bits..]);
                let weighted: Vec<_> = kernels
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v * rows[i / degree])
                    .collect();
                assert_eq!(weighted, table);
                let wide: Vec<i64> = h.iter().copied().map(i64::from).collect();
                assert_eq!(
                    functional_evaluation(&wide, records, &projection, &terminal),
                    direct
                );
            }
        }
    }

    #[test]
    fn target_functional_table_storage_is_128_mib() {
        assert_eq!(core::mem::size_of::<E>(), 16);
        assert_eq!(functional_table_storage_bytes(1 << 14), 128 * 1024 * 1024,);
    }
}
