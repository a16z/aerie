//! Public negacyclic functional and target evaluation for stage 3.
//!
//! The prover materializes coefficient-row weights with exact limb-split
//! negacyclic convolution. The verifier evaluates their multilinear extension
//! at the stage-three point with the linear-time carry recurrence.

use crate::substrate::OpeningField as E;
use akita_sumcheck::reduce_signed_accum;
use jolt_field::Unreduced;
use jolt_field::Zero;

mod functional;
#[cfg(any(test, feature = "internals"))]
pub use functional::functional_evaluation;
#[cfg(feature = "internals")]
pub use functional::functional_evaluation_canonical;
#[cfg(feature = "internals")]
pub use functional::functional_evaluation_canonical_for_bench;

pub use functional::PublicRingKeys;
mod targets;
pub use targets::target_evaluation_i16;
pub use targets::{target_evaluation, PublicTargets};

/// Falcon-512 coefficient variable count, kept for the fixed-degree tests.
#[cfg(any(test, feature = "internals"))]
pub const COEFFICIENT_VARS: usize = 9;

/// Evaluate a base-coefficient inner product in the extension field with one
/// final reduction. Akita's `FpExt2<Fp64>` accumulator has 2^64-term headroom,
/// so a 512-term Falcon row is exact.
#[inline]
pub fn delayed_base_inner_product<C>(coefficients: &[C], factors: &[E]) -> E
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::algebra::{evaluate_mle, lift_table, sample_point};
    use crate::protocol::transcript::TestTranscript;
    use aerie_falcon_native::params::FALCON_Q;
    use jolt_field::{One, Ring};

    fn random_point(label: &[u8], count: usize) -> Vec<E> {
        sample_point(
            &mut TestTranscript::new(b"aerie/falcon/v1/ring-test"),
            core::str::from_utf8(label).unwrap(),
            count,
        )
        .unwrap()
    }

    fn random_coefficients(seed: u64, count: usize) -> Vec<i16> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % FALCON_Q as u64) as i16 - 6144
            })
            .collect()
    }

    // Low-bit-first equality weights, computed from the Boolean definition.
    fn equality_weights(point: &[E]) -> Vec<E> {
        (0..1 << point.len())
            .map(|index| {
                point.iter().enumerate().fold(E::one(), |value, (bit, &r)| {
                    value
                        * if index & (1 << bit) == 0 {
                            E::one() - r
                        } else {
                            r
                        }
                })
            })
            .collect()
    }

    // The coefficient of s[j] in the evaluation of h(X)s(X) modulo X^n + 1.
    fn kernel_from_definition(h: &[i64], point: &[E]) -> Vec<E> {
        let degree = h.len();
        let weights = equality_weights(point);
        (0..degree)
            .map(|j| {
                h.iter()
                    .enumerate()
                    .fold(E::zero(), |sum, (a, &coefficient)| {
                        let term = E::from_i64(coefficient) * weights[(a + j) % degree];
                        if a + j < degree {
                            sum + term
                        } else {
                            sum - term
                        }
                    })
            })
            .collect()
    }

    // Schoolbook convolution retains both the integer lift and its residue mod q.
    // The quotient is needed because the opening field has a different modulus.
    fn schoolbook_product(h: &[i64], s: &[i64]) -> (Vec<i64>, Vec<i64>) {
        let degree = h.len();
        let mut integer = vec![0_i64; degree];
        let mut residues = vec![0_i64; degree];
        for (a, &ha) in h.iter().enumerate() {
            for (b, &sb) in s.iter().enumerate() {
                let index = (a + b) % degree;
                let term = if a + b < degree { ha * sb } else { -ha * sb };
                integer[index] += term;
                residues[index] = (residues[index] + term).rem_euclid(FALCON_Q);
            }
        }
        (integer, residues)
    }

    #[test]
    fn carry_functional_matches_the_definition_at_every_size() {
        for bits in 1..=10_usize {
            let degree = 1 << bits;
            let compact = random_coefficients(0x5249_4e47 + bits as u64, degree);
            let wide: Vec<_> = compact.iter().copied().map(i64::from).collect();
            let projection = random_point(b"kernel point", bits);
            let expected = kernel_from_definition(&wide, &projection);
            // Boolean terminal points select each coefficient of the kernel.
            for (index, &coefficient) in expected.iter().enumerate() {
                let terminal: Vec<_> = (0..bits)
                    .map(|bit| E::from_u64(((index >> bit) & 1) as u64))
                    .collect();
                assert_eq!(
                    functional_evaluation(&wide, 1, &projection, &terminal),
                    coefficient
                );
                assert_eq!(
                    functional_evaluation(&compact, 1, &projection, &terminal),
                    coefficient
                );
            }
        }
    }

    fn check_correlation(bits: usize, seed: u64) {
        let degree = 1 << bits;
        let h: Vec<_> = random_coefficients(seed, degree)
            .into_iter()
            .map(i64::from)
            .collect();
        let s: Vec<_> = random_coefficients(seed + 1, degree)
            .into_iter()
            .map(i64::from)
            .collect();
        let projection = random_point(b"functional point", bits);
        let mut actual = E::zero();
        for (index, &coefficient) in s.iter().enumerate() {
            let terminal: Vec<_> = (0..bits)
                .map(|bit| E::from_u64(((index >> bit) & 1) as u64))
                .collect();
            actual +=
                functional_evaluation(&h, 1, &projection, &terminal) * E::from_i64(coefficient);
        }
        let (integer, _) = schoolbook_product(&h, &s);
        assert_eq!(actual, evaluate_mle(&lift_table(&integer), &projection));
    }

    #[test]
    fn functional_computes_the_negacyclic_correlation() {
        for bits in 1..=5 {
            for seed in [0x5343_484f, 0x434f_4e56, 0x5155_4f54] {
                check_correlation(bits, seed);
            }
        }
        check_correlation(9, 0x4641_4c43);
    }

    #[test]
    fn ring_functional_matches_schoolbook_at_both_degrees() {
        for bits in [9, 10] {
            check_correlation(bits, 0x4e45_4741);
        }
    }
}
