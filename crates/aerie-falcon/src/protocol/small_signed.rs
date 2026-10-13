//! Extension-field factors prepared for exact SIMD contraction by small signs.

use aerie_simd::SmallSignedDotPlan;
use jolt_field::{CanonicalEncoding, Ring};

use crate::substrate::{BaseField as F, OpeningField as E};

#[derive(Clone, Debug)]
pub(crate) struct SmallSignedFactors<const N: usize> {
    plan: FactorPlan<N>,
}

#[derive(Clone, Debug)]
enum FactorPlan<const N: usize> {
    /// `[c0_low, c0_high, c1_low, c1_high]`: each coordinate as a signed
    /// representative `low + 2^32 high` of its residue.
    Packed {
        digits: [Box<[i32; N]>; 4],
        dot: SmallSignedDotPlan,
    },
    Field(Box<[E; N]>),
}

impl<const N: usize> SmallSignedFactors<N> {
    pub(crate) fn new(factors: &[E]) -> Self {
        Self::try_packed(factors).unwrap_or_else(|| {
            assert_eq!(factors.len(), N, "small-signed factor shape");
            Self {
                plan: FactorPlan::Field(Box::new(core::array::from_fn(|index| factors[index]))),
            }
        })
    }

    pub(crate) fn try_packed(factors: &[E]) -> Option<Self> {
        assert_eq!(factors.len(), N, "small-signed factor shape");
        let dot = SmallSignedDotPlan::best_available()?;
        Some(Self::packed(factors, dot))
    }

    pub(crate) fn packed(factors: &[E], dot: SmallSignedDotPlan) -> Self {
        assert_eq!(factors.len(), N, "small-signed factor shape");
        let mut digits = core::array::from_fn(|_| Box::new([0_i32; N]));
        for (index, factor) in factors.iter().enumerate() {
            for (coordinate, value) in [factor.c0(), factor.c1()].into_iter().enumerate() {
                let canonical = u64::try_from(
                    value
                        .to_u128_checked()
                        .expect("fp64 canonical encoding fits u128"),
                )
                .expect("Aerie base-field elements fit in u64");
                let (low, high) = signed_digits(canonical);
                digits[2 * coordinate][index] = low;
                digits[2 * coordinate + 1][index] = high;
            }
        }
        Self {
            plan: FactorPlan::Packed { digits, dot },
        }
    }

    #[inline]
    pub(crate) fn dot_i16(&self, coefficients: &[i16; N]) -> E {
        match &self.plan {
            FactorPlan::Packed { digits, dot } => {
                let sums = dot.dot(
                    [&digits[0], &digits[1], &digits[2], &digits[3]],
                    coefficients,
                );
                E::new(F::from_i128(sums[0]), F::from_i128(sums[1]))
            }
            FactorPlan::Field(factors) => {
                super::ring::delayed_base_inner_product(coefficients, factors.as_slice())
            }
        }
    }
}

/// The base-field modulus `2^64 - 59`.
const MODULUS: u64 = u64::MAX - 58;
/// Largest value `low + 2^32 high` reaches with `i32` digits.
const DIGIT_PAIR_MAX: u64 = (1 << 63) - (1 << 31) - 1;

/// Split a canonical base-field residue `x < p` into `i32` digits of a
/// congruent representative: `x` itself up to [`DIGIT_PAIR_MAX`], else
/// `x - p`, which lies in `[DIGIT_PAIR_MAX + 1 - p, -1]`, above the digit
/// pairs' minimum `-2^63 - 2^31`. The low digit is the sign-extended low word
/// and the high digit the exact quotient of the remainder by `2^32`.
fn signed_digits(residue: u64) -> (i32, i32) {
    let representative = if residue <= DIGIT_PAIR_MAX {
        i128::from(residue)
    } else {
        i128::from(residue) - i128::from(MODULUS)
    };
    let low = i32::from_le_bytes(
        representative.to_le_bytes()[..4]
            .try_into()
            .expect("four low bytes"),
    );
    let high = i32::try_from((representative - i128::from(low)) >> 32)
        .expect("a residue representative fits two signed digits");
    (low, high)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::algebra::sample_point;
    use crate::protocol::transcript::TestTranscript;
    use aerie_falcon_statement::FALCON_DEGREE;
    use akita_sumcheck::reduce_signed_accum;
    use jolt_field::Unreduced;
    use jolt_field::Zero;

    #[test]
    fn signed_digits_represent_every_boundary_residue() {
        assert_eq!(F::from_i128(i128::from(MODULUS)), F::from_i128(0));
        let digit_min = i128::from(i32::MIN) + (i128::from(i32::MIN) << 32);
        let digit_max = i128::from(i32::MAX) + (i128::from(i32::MAX) << 32);
        assert_eq!(digit_max, i128::from(DIGIT_PAIR_MAX));
        for residue in [
            0,
            1,
            (1 << 31) - 1,
            1 << 31,
            (1 << 32) - 1,
            1 << 32,
            DIGIT_PAIR_MAX - 1,
            DIGIT_PAIR_MAX,
            DIGIT_PAIR_MAX + 1,
            1 << 63,
            (MODULUS - 1) / 2,
            MODULUS.div_ceil(2),
            MODULUS - (1 << 32),
            MODULUS - 2,
            MODULUS - 1,
        ] {
            let (low, high) = signed_digits(residue);
            let value = i128::from(low) + (i128::from(high) << 32);
            assert!((digit_min..=digit_max).contains(&value));
            assert!(
                value == i128::from(residue) || value == i128::from(residue) - i128::from(MODULUS)
            );
            assert_eq!(
                F::from_i128(value),
                F::from_u64(residue),
                "residue {residue}"
            );
        }
    }

    #[test]
    fn every_dot_kernel_matches_the_field_path_at_boundary_factors() {
        let boundary = [
            0,
            1,
            DIGIT_PAIR_MAX,
            DIGIT_PAIR_MAX + 1,
            1 << 63,
            MODULUS - 1,
        ];
        let factors: Vec<E> = (0..FALCON_DEGREE)
            .map(|index| {
                E::new(
                    F::from_u64(boundary[index % boundary.len()]),
                    F::from_u64(boundary[(index / boundary.len()) % boundary.len()]),
                )
            })
            .collect();
        let field = SmallSignedFactors::<FALCON_DEGREE> {
            plan: FactorPlan::Field(Box::new(core::array::from_fn(|index| factors[index]))),
        };
        let plans = [
            Some(SmallSignedDotPlan::scalar()),
            SmallSignedDotPlan::neon(),
            SmallSignedDotPlan::avx2(),
            SmallSignedDotPlan::avx512(),
        ];
        for coefficients in [
            [i16::MIN; FALCON_DEGREE],
            [i16::MAX; FALCON_DEGREE],
            core::array::from_fn(|index| if index % 3 == 0 { i16::MIN } else { i16::MAX }),
            core::array::from_fn(|index| ((index * 7919 + 17) % 12_289) as i16 - 6_144),
        ] {
            let expected = field.dot_i16(&coefficients);
            for plan in plans.into_iter().flatten() {
                let packed = SmallSignedFactors::<FALCON_DEGREE>::packed(&factors, plan);
                assert_eq!(packed.dot_i16(&coefficients), expected, "{}", plan.name());
            }
        }
    }

    #[test]
    fn packed_small_dot_matches_unreduced_field_products() {
        let mut transcript = TestTranscript::new(b"packed-small-dot");
        let factors = sample_point(&mut transcript, "factors", FALCON_DEGREE).unwrap();
        let coefficients =
            core::array::from_fn(|index| ((index * 7919 + 17) % 12_289) as i16 - 6_144);
        let mut positive = <E as Unreduced>::SmallProduct::zero();
        let mut negative = <E as Unreduced>::SmallProduct::zero();
        for (&factor, &coefficient) in factors.iter().zip(&coefficients) {
            let product = factor.mul_u64_unreduced(i64::from(coefficient).unsigned_abs());
            if coefficient >= 0 {
                positive += product;
            } else {
                negative += product;
            }
        }
        let expected = reduce_signed_accum::<E>(positive, negative);
        assert_eq!(
            SmallSignedFactors::<FALCON_DEGREE>::new(&factors).dot_i16(&coefficients),
            expected
        );
    }
}
