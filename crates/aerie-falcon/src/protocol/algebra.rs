//! Shared multilinear algebra helpers for the three-stage PIOP.
//!
//! The heavy helpers parallelize above a size threshold. Every parallel
//! path computes the exact per-cell arithmetic of its serial twin, and
//! field addition is associative and commutative, so reduction order never
//! changes a produced value: proofs stay byte-identical either way (the
//! transport module pins a proof-bytes digest as the regression gate).

#[cfg(any(test, feature = "internals"))]
use super::transcript::{ChallengeSource, TranscriptOrderError};
#[cfg(any(test, feature = "internals"))]
use jolt_field::Ring as _;
use jolt_field::{One as _, Zero as _};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::substrate::OpeningField as E;

/// Below this many cells the serial path is faster than the fork overhead.
#[cfg(feature = "parallel")]
const MIN_PARALLEL_CELLS: usize = 1 << 13;

/// Equality-weight table with coordinate `i` on index bit `i`, matching the
/// low-bit-first sumcheck binding order.
#[must_use]
pub fn eq_table(point: &[E]) -> Vec<E> {
    let mut weights = vec![E::one()];
    for &coordinate in point {
        let length = weights.len();
        #[cfg(feature = "parallel")]
        if 2 * length >= MIN_PARALLEL_CELLS {
            let mut next = vec![E::zero(); 2 * length];
            let (low, high) = next.split_at_mut(length);
            low.par_iter_mut()
                .zip(high.par_iter_mut())
                .zip(&weights)
                .for_each(|((low, high), &weight)| {
                    let scaled = weight * coordinate;
                    *low = weight - scaled;
                    *high = scaled;
                });
            weights = next;
            continue;
        }
        let mut next = Vec::with_capacity(2 * length);
        next.extend(
            weights
                .iter()
                .map(|&weight| weight * (E::one() - coordinate)),
        );
        next.extend(weights.iter().map(|&weight| weight * coordinate));
        weights = next;
    }
    weights
}

#[inline]
#[must_use]
pub fn eq_evaluation(left: &[E], right: &[E]) -> E {
    left.iter().zip(right).fold(E::one(), |product, (&a, &b)| {
        product * (a * b + (E::one() - a) * (E::one() - b))
    })
}

/// Evaluate the multilinear extension of a table at a point through the
/// equality-weight table.
#[cfg(any(test, feature = "internals"))]
#[must_use]
pub fn evaluate_mle(table: &[E], point: &[E]) -> E {
    let weights = eq_table(point);
    #[cfg(feature = "parallel")]
    if table.len() >= MIN_PARALLEL_CELLS {
        return weights
            .par_iter()
            .zip(table)
            .map(|(&weight, &value)| weight * value)
            .reduce(E::zero, |left, right| left + right);
    }
    weights
        .iter()
        .zip(table)
        .fold(E::zero(), |sum, (&weight, &value)| sum + weight * value)
}

#[cfg(any(test, feature = "internals"))]
pub fn sample(
    transcript: &mut impl ChallengeSource,
    site: &str,
) -> Result<E, TranscriptOrderError> {
    transcript
        .challenge_extensions(site, 1)
        .map(|mut values| values.remove(0))
}

#[cfg(any(test, feature = "internals"))]
pub fn sample_point(
    transcript: &mut impl ChallengeSource,
    site: &str,
    count: usize,
) -> Result<Vec<E>, TranscriptOrderError> {
    transcript.challenge_extensions(site, count)
}

/// Closed-form evaluation of the public live-row mask MLE,
/// `live~_N(r) = sum_{i < N_sig} eq(r, bits(i))`, Section 7.4 of `specs/falcon.md`.
///
/// A point `i` is below `N` exactly when some bit position `b` has
/// `i_b = 0 < N_b = 1` with all higher bits equal; the free low bits sum
/// to one, so each such position contributes its matched suffix product
/// times `1 - r_b`. Costs `O(d_sig)` field operations, no witness table,
/// and no commitment. For a fully live domain the mask is exactly one.
#[must_use]
pub fn live_prefix_evaluation(live_rows: usize, point: &[E]) -> E {
    debug_assert!(live_rows <= 1 << point.len());
    if live_rows == 1 << point.len() {
        return E::one();
    }
    let mut total = E::zero();
    let mut matched_suffix = E::one();
    for position in (0..point.len()).rev() {
        if (live_rows >> position) & 1 == 1 {
            total += matched_suffix * (E::one() - point[position]);
            matched_suffix *= point[position];
        } else {
            matched_suffix *= E::one() - point[position];
        }
    }
    total
}

#[cfg(any(test, feature = "internals"))]
pub fn lift_table<C>(table: &[C]) -> Vec<E>
where
    C: Copy + Sync + Into<i64>,
{
    #[cfg(feature = "parallel")]
    if table.len() >= MIN_PARALLEL_CELLS {
        return table
            .par_iter()
            .map(|&cell| E::from_i64(cell.into()))
            .collect();
    }
    table.iter().map(|&cell| E::from_i64(cell.into())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::transcript::TestTranscript;

    #[test]
    fn live_prefix_evaluation_matches_the_direct_prefix_sum() {
        let mut transcript = TestTranscript::new(b"live prefix test");
        for row_vars in 0..=4_usize {
            let point = sample_point(&mut transcript, "live-point", row_vars).unwrap();
            let weights = eq_table(&point);
            for live_rows in 1..=(1 << row_vars) {
                let direct = weights[..live_rows]
                    .iter()
                    .fold(E::zero(), |sum, &weight| sum + weight);
                assert_eq!(
                    live_prefix_evaluation(live_rows, &point),
                    direct,
                    "live {live_rows} of 2^{row_vars}"
                );
            }
            assert_eq!(live_prefix_evaluation(1 << row_vars, &point), E::one());
        }
    }
}
