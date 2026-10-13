//! Low-level split-equality accumulators for compact stage-three tables.
//!
//! Each named relation owns its state and polynomial coefficients. These
//! contractions reduce the inner equality half before multiplying the outer
//! half, avoiding a field multiplication for every expanded equality weight.

use crate::substrate::OpeningField as E;
use akita_sumcheck::reduce_signed_accum;
use jolt_field::{One, Unreduced, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
const PARALLEL_PAIRS: usize = 1 << 14;
type SmallAccum = <E as Unreduced>::SmallProduct;
type Product = <E as Unreduced>::Product;

/// Contract signed integer coefficients without lifting the source tables.
/// Positive and negative products remain unreduced through each inner block.
pub(super) fn weighted_integer_coefficients<const D: usize, C>(
    first: &[E],
    second: &[E],
    coefficients: C,
) -> [E; D]
where
    C: Fn(usize) -> [i64; D] + Send + Sync,
{
    const { assert!(E::SUM_IS_EXACT) };
    sum_blocks(first.len(), second.len(), |block, start, end| {
        let mut positive = [SmallAccum::zero(); D];
        let mut negative = [SmallAccum::zero(); D];
        for (offset, &weight) in first[start..end].iter().enumerate() {
            let low = start + offset;
            if weight.is_zero() {
                continue;
            }
            for (lane, value) in coefficients(block * first.len() + low)
                .into_iter()
                .enumerate()
            {
                let product = weight.mul_u64_unreduced(value.unsigned_abs());
                if value < 0 {
                    negative[lane] += product;
                } else {
                    positive[lane] += product;
                }
            }
        }
        std::array::from_fn(|lane| {
            let inner = reduce_signed_accum::<E>(positive[lane], negative[lane]);
            second[block].mul_unreduced(inner)
        })
    })
}

fn sum_blocks<const D: usize, B>(first_len: usize, second_len: usize, block: B) -> [E; D]
where
    B: Fn(usize, usize, usize) -> [Product; D] + Send + Sync,
{
    let add = |mut left: [Product; D], right: [Product; D]| {
        for (left, right) in left.iter_mut().zip(right) {
            *left += right;
        }
        left
    };
    #[cfg(feature = "parallel")]
    let sums = if first_len * second_len >= PARALLEL_PAIRS {
        // A row-only equality has just one outer entry. Split its inner
        // dimension too: each entry may itself scan a whole coefficient row.
        // Reducing smaller inner sums before the outer multiplication keeps
        // the same field contraction with shorter unreduced accumulation chains.
        let chunk_len = if second_len == 1 { 256 } else { first_len };
        let chunks = first_len.div_ceil(chunk_len);
        (0..second_len * chunks)
            .into_par_iter()
            .map(|task| {
                let start = (task % chunks) * chunk_len;
                block(task / chunks, start, (start + chunk_len).min(first_len))
            })
            .reduce(|| [Product::zero(); D], add)
    } else {
        (0..second_len)
            .map(|index| block(index, 0, first_len))
            .fold([Product::zero(); D], add)
    };
    #[cfg(not(feature = "parallel"))]
    let sums = {
        (0..second_len)
            .map(|index| block(index, 0, first_len))
            .fold([Product::zero(); D], add)
    };
    sums.map(E::reduce_product)
}

/// Compose a quadratic inner relation with the current equality coordinate.
/// The equality-factored path omits the inner linear coefficient; endpoint
/// coordinates use the direct coefficient to avoid division by zero.
pub(super) fn equality_weighted_cubic(
    claim: E,
    tau: E,
    scalar: E,
    [constant, linear, quadratic]: [E; 3],
) -> [E; 3] {
    let constant = scalar * constant;
    let linear = scalar * linear;
    let quadratic = scalar * quadratic;
    if let Some(polynomial) =
        crate::protocol::algebra::eq_factored_cubic_from_constant_and_quadratic(
            tau, claim, constant, quadratic,
        )
    {
        let coefficients = polynomial.coefficients();
        [coefficients[0], coefficients[2], coefficients[3]]
    } else {
        let ell_constant = E::one() - tau;
        let ell_linear = tau + tau - E::one();
        [
            ell_constant * constant,
            ell_constant * quadratic + ell_linear * linear,
            ell_linear * quadratic,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::BaseField as F;
    use jolt_field::Ring;

    #[test]
    fn split_accumulators_match_direct_signed_and_extension_contractions() {
        for (first_len, second_len) in [(1, 1), (4, 8), (128, 128), (16_513, 1), (32_768, 1)] {
            let first: Vec<_> = (0..first_len)
                .map(|i| E::new(F::from_u64(i as u64), F::from_u64(3)))
                .collect();
            let second: Vec<_> = (0..second_len)
                .map(|i| E::new(F::from_u64((i + 1) as u64), F::from_u64(5)))
                .collect();
            let integers = |pair: usize| {
                let a = pair as i64 % 65_536 - 32_768;
                [a, a * a, -2 * a]
            };
            let fields =
                |pair: usize| integers(pair).map(|a| E::new(F::from_i64(a), F::from_i64(a + 7)));
            let mut expected_integer = [E::zero(); 3];
            let mut expected_field = [E::zero(); 3];
            for pair in 0..first_len * second_len {
                let weight = first[pair % first_len] * second[pair / first_len];
                for lane in 0..3 {
                    expected_integer[lane] += weight * E::from_i64(integers(pair)[lane]);
                    expected_field[lane] += weight * fields(pair)[lane];
                }
            }
            assert_eq!(
                weighted_integer_coefficients(&first, &second, integers),
                expected_integer
            );
            // The fused equality view has power-of-two factors. Zero padding
            // preserves the arbitrary-length reference contraction as well.
            let mut field_first = first.clone();
            field_first.resize(first.len().next_power_of_two(), E::zero());
            let padded_len = field_first.len();
            let padded_fields = |pair: usize| {
                let low = pair % padded_len;
                if low < first_len {
                    fields((pair / padded_len) * first_len + low)
                } else {
                    [E::zero(); 3]
                }
            };
            assert_eq!(
                super::super::fused_rounds::tests::contract_split_for_test(
                    &field_first,
                    &second,
                    padded_fields,
                ),
                expected_field
            );
        }
    }
}
