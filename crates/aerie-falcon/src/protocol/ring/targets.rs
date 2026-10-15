//! Public target evaluation at a canonical coefficient and row point.

use aerie_falcon_native::preprocess::{
    hash_to_point_prehashed_batch_visit_of, PrehashedHashToPointBatch,
};
use aerie_falcon_native::{Falcon1024, Falcon512, FalconParams};
use jolt_field::Unreduced;
use jolt_field::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::protocol::FalconDegree;
use crate::substrate::OpeningField as E;

use super::delayed_base_inner_product;
use crate::protocol::algebra::eq_table;
use crate::protocol::small_signed::SmallSignedFactors;

type Product = <E as Unreduced>::Product;

/// Public Falcon targets in expanded or verifier-source form.
#[derive(Debug)]
pub enum PublicTargets<'a, C = i64> {
    /// A row-major, zero-extended table over the padded record domain.
    #[cfg_attr(not(any(test, feature = "internals")), allow(dead_code))]
    Expanded(&'a [C]),
    /// Live `HashToPoint` source tuples, the wire verifier's form: each target
    /// row is sampled and contracted once, and padded rows are implicit zero.
    SourceBound(PrehashedHashToPointBatch<'a>),
}

impl<C> Copy for PublicTargets<'_, C> {}

impl<C> Clone for PublicTargets<'_, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C> PublicTargets<'_, C>
where
    C: Copy + Into<i64>,
{
    /// Validate this representation against its degree and row domains.
    pub fn validate_shape_of(
        self,
        degree: usize,
        live_rows: usize,
        padded_rows: usize,
    ) -> Result<(), crate::StatementMismatch> {
        match self {
            Self::Expanded(table) => {
                if table.len() != padded_rows * degree {
                    return Err(crate::StatementMismatch::PublicInputShape);
                }
                if table[live_rows * degree..]
                    .iter()
                    .any(|&value| value.into() != 0)
                {
                    return Err(crate::StatementMismatch::NonZeroPadding);
                }
                Ok(())
            }
            Self::SourceBound(inputs) => {
                if degree != Falcon512::DEGREE && degree != Falcon1024::DEGREE {
                    return Err(crate::StatementMismatch::PublicInputShape);
                }
                if inputs.len() != live_rows {
                    return Err(crate::StatementMismatch::PublicInputShape);
                }
                Ok(())
            }
        }
    }
}

/// Evaluate the public target rows at a canonical coefficient-then-row point.
#[must_use]
pub fn target_evaluation<C>(
    degree: FalconDegree,
    records: usize,
    live_rows: usize,
    targets: PublicTargets<'_, C>,
    point: &[E],
) -> E
where
    C: Copy + Sync + Into<i64>,
{
    assert!(records.is_power_of_two() && records > 0);
    assert!(live_rows <= records);
    assert_eq!(
        point.len(),
        degree.logn() + records.trailing_zeros() as usize
    );
    match targets {
        PublicTargets::Expanded(table) => {
            expanded_evaluation(table, degree.degree(), live_rows, records, point)
        }
        PublicTargets::SourceBound(inputs) => match degree {
            FalconDegree::Falcon512 => {
                source_evaluation_of::<Falcon512, 512>(inputs, records, point)
            }
            FalconDegree::Falcon1024 => {
                source_evaluation_of::<Falcon1024, 1024>(inputs, records, point)
            }
        },
    }
}

#[doc(hidden)]
#[must_use]
pub fn target_evaluation_i16(
    degree: FalconDegree,
    table: &[i16],
    live_rows: usize,
    records: usize,
    point: &[E],
) -> E {
    match degree {
        FalconDegree::Falcon512 => i16_evaluation_of::<512>(table, live_rows, records, point),
        FalconDegree::Falcon1024 => i16_evaluation_of::<1024>(table, live_rows, records, point),
    }
}

fn i16_evaluation_of<const N: usize>(
    table: &[i16],
    live_rows: usize,
    records: usize,
    point: &[E],
) -> E {
    assert_eq!(table.len(), records * N);
    let coefficient_vars = N.trailing_zeros() as usize;
    let (coefficient_point, row_point) = point.split_at(coefficient_vars);
    let coefficient_weights = eq_table(coefficient_point);
    let row_weights = eq_table(row_point);
    let coefficient_weights = SmallSignedFactors::<N>::new(&coefficient_weights);
    let evaluate_record = |record: usize| {
        let row: &[i16; N] = table[record * N..(record + 1) * N]
            .try_into()
            .expect("target row has the ring degree");
        row_weights[record].mul_unreduced(coefficient_weights.dot_i16(row))
    };
    #[cfg(feature = "parallel")]
    let accumulator = (0..live_rows)
        .into_par_iter()
        .map(evaluate_record)
        .reduce(Product::zero, |left, right| left + right);
    #[cfg(not(feature = "parallel"))]
    let accumulator = (0..live_rows)
        .map(evaluate_record)
        .fold(Product::zero(), |sum, term| sum + term);
    E::reduce_product(accumulator)
}

fn expanded_evaluation<C>(
    table: &[C],
    degree: usize,
    live_rows: usize,
    records: usize,
    point: &[E],
) -> E
where
    C: Copy + Sync + Into<i64>,
{
    assert_eq!(table.len(), records * degree);
    let coefficient_vars = degree.trailing_zeros() as usize;
    let (coefficient_point, row_point) = point.split_at(coefficient_vars);
    let coefficient_weights = eq_table(coefficient_point);
    let row_weights = eq_table(row_point);
    let evaluate_record = |record: usize| {
        let row = &table[record * degree..(record + 1) * degree];
        let inner = delayed_base_inner_product(row, &coefficient_weights);
        row_weights[record].mul_unreduced(inner)
    };
    #[cfg(feature = "parallel")]
    let accumulator = (0..live_rows)
        .into_par_iter()
        .map(evaluate_record)
        .reduce(Product::zero, |left, right| left + right);
    #[cfg(not(feature = "parallel"))]
    let accumulator = (0..live_rows)
        .map(evaluate_record)
        .fold(Product::zero(), |sum, term| sum + term);
    let _ = records;
    E::reduce_product(accumulator)
}

fn source_evaluation_of<P: FalconParams, const N: usize>(
    inputs: PrehashedHashToPointBatch<'_>,
    records: usize,
    point: &[E],
) -> E {
    debug_assert_eq!(P::DEGREE, N);
    let coefficient_vars = N.trailing_zeros() as usize;
    let (coefficient_point, row_point) = point.split_at(coefficient_vars);
    let coefficient_weights = eq_table(coefficient_point);
    let row_weights = eq_table(row_point);
    let coefficient_weights = SmallSignedFactors::<N>::new(&coefficient_weights);
    debug_assert!(inputs.len() <= records);

    let evaluate_chunk = |inputs: PrehashedHashToPointBatch<'_>, first_record: usize| {
        let mut accumulator = Product::zero();
        hash_to_point_prehashed_batch_visit_of::<P, _>(inputs, |record, row| {
            let row: &[i16; N] = row
                .as_ref()
                .try_into()
                .expect("a HashToPoint row has the ring degree");
            accumulator +=
                row_weights[first_record + record].mul_unreduced(coefficient_weights.dot_i16(row));
        });
        accumulator
    };

    #[cfg(feature = "parallel")]
    let accumulator = {
        let grain = aerie_falcon_native::preprocess::hash_to_point_parallel_grain(inputs.len());
        (0..inputs.len())
            .into_par_iter()
            .step_by(grain)
            .map(|first_record| {
                let end = (first_record + grain).min(inputs.len());
                evaluate_chunk(inputs.slice(first_record..end), first_record)
            })
            .reduce(Product::zero, |left, right| left + right)
    };
    #[cfg(not(feature = "parallel"))]
    let accumulator = evaluate_chunk(inputs, 0);
    E::reduce_product(accumulator)
}
