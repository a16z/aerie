use jolt_field::Ring;
#[cfg(test)]
use jolt_field::Zero;

use crate::substrate::OpeningField as E;

pub const QUARTIC_COEFFICIENTS: usize = 4;

/// Number of [`QuarticPolynomial`] kinds.
pub const KIND_COUNT: usize = 2;

/// The leaf polynomials of the common quartic leaf. Every variant has degree
/// four in the range image `Y = Z (Z + 1)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuarticPolynomial {
    /// `Y (Y - 2) (Y - 6) (Y - 12)`: the radix-8 leaf and the low half of
    /// the radix-16 pair.
    Low,
    /// `(Y - 20) (Y - 30) (Y - 42) (Y - 56)`: the high half of the radix-16
    /// pair.
    High,
}

impl QuarticPolynomial {
    /// Every polynomial kind, in the order used by the kernel's per-kind
    /// mass accumulators.
    pub const KINDS: [Self; KIND_COUNT] = [Self::Low, Self::High];

    /// Index of this kind in [`Self::KINDS`].
    #[must_use]
    #[inline]
    pub const fn kind_index(self) -> usize {
        match self {
            Self::Low => 0,
            Self::High => 1,
        }
    }

    #[inline]
    #[must_use]
    pub fn evaluate(self, y: E) -> E {
        match self {
            Self::Low => y * (y - E::from_i64(2)) * (y - E::from_i64(6)) * (y - E::from_i64(12)),
            Self::High => {
                (y - E::from_i64(20))
                    * (y - E::from_i64(30))
                    * (y - E::from_i64(42))
                    * (y - E::from_i64(56))
            }
        }
    }

    /// Monomial coefficients `c_0, ..., c_4` in `Y`.
    #[inline]
    #[must_use]
    pub const fn monomials(self) -> [i64; 5] {
        match self {
            Self::Low => [0, -144, 108, -20, 1],
            Self::High => [1_411_200, -176_400, 7_852, -148, 1],
        }
    }

    #[inline]
    #[must_use]
    pub fn affine_coefficients(self, left: E, delta: E) -> [E; QUARTIC_COEFFICIENTS] {
        let [_, linear, quadratic, cubic, quartic] = self.monomials();
        let linear = E::from_i64(linear);
        let quadratic = E::from_i64(quadratic);
        let cubic = E::from_i64(cubic);
        let quartic = E::from_i64(quartic);
        let delta_squared = delta * delta;
        let delta_cubed = delta_squared * delta;
        let delta_fourth = delta_squared * delta_squared;
        [
            delta
                * (linear
                    + E::from_i64(2) * quadratic * left
                    + E::from_i64(3) * cubic * left * left
                    + E::from_i64(4) * quartic * left * left * left),
            delta_squared
                * (quadratic
                    + E::from_i64(3) * cubic * left
                    + E::from_i64(6) * quartic * left * left),
            delta_cubed * (cubic + E::from_i64(4) * quartic * left),
            delta_fourth * quartic,
        ]
    }

    /// Integer affine coefficients for endpoints in the range-image alphabet.
    ///
    /// The compact prefix uses only the images `i(i + 1)` for `i < 8`, so
    /// every intermediate fits comfortably in `i64`. Keeping these values as
    /// integers lets the caller use extension-by-base multiplication while it
    /// contracts pattern masses.
    #[inline]
    #[must_use]
    pub fn affine_coefficients_i64(self, left: i64, delta: i64) -> [i64; QUARTIC_COEFFICIENTS] {
        let [_, linear, quadratic, cubic, quartic] = self.monomials();
        let left_squared = left * left;
        let delta_squared = delta * delta;
        let delta_cubed = delta_squared * delta;
        let delta_fourth = delta_squared * delta_squared;
        [
            delta
                * (linear
                    + 2 * quadratic * left
                    + 3 * cubic * left_squared
                    + 4 * quartic * left_squared * left),
            delta_squared * (quadratic + 3 * cubic * left + 6 * quartic * left_squared),
            delta_cubed * (cubic + 4 * quartic * left),
            delta_fourth * quartic,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monomials_match_the_factored_forms() {
        for kind in QuarticPolynomial::KINDS {
            for y in -70..70 {
                let y = E::from_i64(y);
                let expanded = kind
                    .monomials()
                    .iter()
                    .rev()
                    .fold(E::zero(), |acc, &c| acc * y + E::from_i64(c));
                assert_eq!(kind.evaluate(y), expanded, "{kind:?}");
            }
        }
    }

    #[test]
    fn range_polynomials_accept_exactly_the_balanced_digit_alphabets() {
        for digit in -64..=64 {
            let digit_field = E::from_i64(digit);
            let image = digit_field * (digit_field + E::from_i64(1));
            let in_radix8 = (-4..=3).contains(&digit);
            assert_eq!(
                QuarticPolynomial::Low.evaluate(image).is_zero(),
                in_radix8,
                "Q8 must accept exactly a0 in [-4, 3], got {digit}"
            );

            let in_radix16 = (-8..=7).contains(&digit);
            let radix16_accepts = QuarticPolynomial::Low.evaluate(image)
                * QuarticPolynomial::High.evaluate(image)
                == E::zero();
            assert_eq!(
                radix16_accepts, in_radix16,
                "Q16^lo Q16^hi must accept exactly a 4-bit digit in [-8, 7], got {digit}"
            );
        }
    }

    #[test]
    fn each_quartic_has_exactly_its_specified_roots() {
        use jolt_field::Prime64Offset59 as F;
        for (polynomial, roots) in [
            (QuarticPolynomial::Low, [0, 2, 6, 12]),
            (QuarticPolynomial::High, [20, 30, 42, 56]),
        ] {
            for value in -100..=100 {
                assert_eq!(
                    polynomial.evaluate(E::from_i64(value)).is_zero(),
                    roots.contains(&value),
                    "{polynomial:?} at {value}"
                );
                // No element outside the base field is one of these roots.
                assert!(!polynomial
                    .evaluate(E::new(F::from_i64(value), F::from_u64(1)))
                    .is_zero());
            }
        }
    }
}
