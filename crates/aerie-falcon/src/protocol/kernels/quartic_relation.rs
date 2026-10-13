use jolt_field::{Field, Ring, Zero};

use crate::substrate::OpeningField as E;

const STORED_COEFFICIENTS: usize = 4;

/// Number of [`QuarticPolynomial`] kinds.
pub(crate) const KIND_COUNT: usize = 2;

/// The leaf polynomials of the common quartic leaf. Every variant has degree
/// four in the range image `Y = Z (Z + 1)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QuarticPolynomial {
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
    pub(crate) const KINDS: [Self; KIND_COUNT] = [Self::Low, Self::High];

    /// Index of this kind in [`Self::KINDS`].
    #[must_use]
    pub(crate) const fn kind_index(self) -> usize {
        match self {
            Self::Low => 0,
            Self::High => 1,
        }
    }

    pub(crate) fn evaluate<F>(self, y: F) -> F
    where
        F: Field + Ring,
    {
        match self {
            Self::Low => y * (y - F::from_i64(2)) * (y - F::from_i64(6)) * (y - F::from_i64(12)),
            Self::High => {
                (y - F::from_i64(20))
                    * (y - F::from_i64(30))
                    * (y - F::from_i64(42))
                    * (y - F::from_i64(56))
            }
        }
    }

    /// Monomial coefficients `c_0, ..., c_4` in `Y`.
    const fn monomials(self) -> [i64; 5] {
        match self {
            Self::Low => [0, -144, 108, -20, 1],
            Self::High => [1_411_200, -176_400, 7_852, -148, 1],
        }
    }

    pub(crate) fn affine_coefficients<F>(self, left: F, delta: F) -> [F; STORED_COEFFICIENTS]
    where
        F: Field + Ring,
    {
        let [_, linear, quadratic, cubic, quartic] = self.monomials();
        let linear = F::from_i64(linear);
        let quadratic = F::from_i64(quadratic);
        let cubic = F::from_i64(cubic);
        let quartic = F::from_i64(quartic);
        let delta_squared = delta * delta;
        let delta_cubed = delta_squared * delta;
        let delta_fourth = delta_squared * delta_squared;
        [
            delta
                * (linear
                    + F::from_i64(2) * quadratic * left
                    + F::from_i64(3) * cubic * left * left
                    + F::from_i64(4) * quartic * left * left * left),
            delta_squared
                * (quadratic
                    + F::from_i64(3) * cubic * left
                    + F::from_i64(6) * quartic * left * left),
            delta_cubed * (cubic + F::from_i64(4) * quartic * left),
            delta_fourth * quartic,
        ]
    }

    /// Integer affine coefficients for endpoints in the range-image alphabet.
    ///
    /// The compact prefix uses only the images `i(i + 1)` for `i < 8`, so
    /// every intermediate fits comfortably in `i64`. Keeping these values as
    /// integers lets the caller use extension-by-base multiplication while it
    /// contracts pattern masses.
    pub(crate) fn affine_coefficients_i64(
        self,
        left: i64,
        delta: i64,
    ) -> [i64; STORED_COEFFICIENTS] {
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

#[derive(Clone, Copy)]
pub(crate) struct WeightedQuartic {
    pub(crate) polynomial: QuarticPolynomial,
    pub(crate) weight: E,
}

#[derive(Clone, Copy)]
pub(crate) struct PreparedWeights {
    kinds: [E; KIND_COUNT],
    coefficients: [E; 5],
    three_cubic: E,
    depressed: Option<DepressedQuartic>,
}

#[derive(Clone, Copy)]
struct DepressedQuartic {
    shift: E,
    quadratic: E,
    linear: E,
    leading: E,
}

impl PreparedWeights {
    /// The aggregated weight of every polynomial kind, in [`QuarticPolynomial::KINDS`] order.
    pub(crate) fn kind_weights(self) -> [E; KIND_COUNT] {
        self.kinds
    }

    /// Taylor coefficients of the weighted polynomial at an endpoint. The
    /// quartet prefix reuses each endpoint across many octets.
    pub(crate) fn taylor(self, left: E) -> [E; STORED_COEFFICIENTS] {
        use super::opening_field_ops::mul;
        let [_, c1, c2, c3, c4] = self.coefficients;
        let c4_left = mul(c4, left);
        let twice = c4_left + c4_left;
        let four = twice + twice;
        [
            mul(mul(four + self.three_cubic, left) + c2 + c2, left) + c1,
            mul(four + twice + self.three_cubic, left) + c2,
            four + c3,
            c4,
        ]
    }

    /// Translation `s` such that the dense table stores `u = Y - s`.
    /// A polynomial whose leading coefficient cancels keeps the unshifted
    /// representation and uses the general affine path.
    pub(crate) fn shift(self) -> E {
        self.depressed.map_or(E::zero(), |quartic| quartic.shift)
    }

    pub(crate) fn new(kinds: [E; KIND_COUNT]) -> Self {
        // The batching weights stay fixed for every row and active round.
        let mut prepared = Self::new_unshifted(kinds);
        let coefficients = prepared.coefficients;
        // The relation-level cancelling-weights test covers the `c4 = 0`
        // fallback. The kernel only consumes this prepared representation, so
        // repeating the same case at the kernel level would add no new path.
        prepared.depressed = if coefficients[4].is_zero() {
            None
        } else {
            let c1 = coefficients[1];
            let c2 = coefficients[2];
            let c3 = coefficients[3];
            let c4 = coefficients[4];
            let four_c4 = c4 + c4 + c4 + c4;
            let four_c4_inverse = four_c4
                .inverse()
                .expect("a nonzero quartic coefficient is invertible");
            let shift = -c3 * four_c4_inverse;
            let shift_squared = shift * shift;
            let shift_cubed = shift_squared * shift;
            let c4_inverse = four_c4_inverse + four_c4_inverse + four_c4_inverse + four_c4_inverse;
            let quadratic =
                (c2 + (c3 + c3 + c3) * shift + (c4 + c4 + c4 + c4 + c4 + c4) * shift_squared)
                    * c4_inverse;
            let linear =
                (c1 + (c2 + c2) * shift + (c3 + c3 + c3) * shift_squared + four_c4 * shift_cubed)
                    * c4_inverse;
            Some(DepressedQuartic {
                shift,
                quadratic,
                linear,
                leading: c4,
            })
        };
        prepared
    }

    /// Prepare a transient weighted polynomial without computing its dense
    /// coordinate translation. Compact-prefix callers use original `Y`
    /// coordinates and construct one value per pattern.
    pub(crate) fn new_unshifted(kinds: [E; KIND_COUNT]) -> Self {
        let mut coefficients = [E::zero(); 5];
        for (kind, weight) in QuarticPolynomial::KINDS.into_iter().zip(kinds) {
            for (coefficient, monomial) in coefficients.iter_mut().zip(kind.monomials()) {
                *coefficient += E::from_i64(monomial) * weight;
            }
        }
        Self {
            kinds,
            coefficients,
            three_cubic: coefficients[3] + coefficients[3] + coefficients[3],
            depressed: None,
        }
    }
}

pub(crate) fn aggregate_weights(terms: &[WeightedQuartic]) -> PreparedWeights {
    let mut kinds = [E::zero(); KIND_COUNT];
    assert!(!terms.is_empty(), "quartic source terms are nonempty");
    for term in terms {
        kinds[term.polynomial.kind_index()] += term.weight;
    }
    PreparedWeights::new(kinds)
}

/// Substitute `left + delta * X` in the prepared weighted polynomial.
/// Only coefficients 1, 2, 3, and 4 are sent by this sumcheck. Hoisting
/// the batching weights and using Horner form shares `c4 * left` across
/// all four outputs, including zero or cancelling batching weights.
#[inline]
pub(crate) fn weighted_affine_coefficients(
    weights: PreparedWeights,
    left: E,
    delta: E,
) -> [E; STORED_COEFFICIENTS] {
    use super::opening_field_ops::mul;
    let [_c0, c1, c2, c3, c4] = weights.coefficients;
    let c4_left = mul(c4, left);
    let twice_c4_left = c4_left + c4_left;
    let four_c4_left = twice_c4_left + twice_c4_left;
    let six_c4_left = four_c4_left + twice_c4_left;
    let delta_squared = mul(delta, delta);
    [
        mul(
            delta,
            mul(
                mul(four_c4_left + weights.three_cubic, left) + c2 + c2,
                left,
            ) + c1,
        ),
        mul(
            delta_squared,
            mul(six_c4_left + weights.three_cubic, left) + c2,
        ),
        mul(mul(delta_squared, delta), four_c4_left + c3),
        mul(mul(delta_squared, delta_squared), c4),
    ]
}

/// Substitute `left + delta * X` after translating the dense table to the
/// depressed coordinate `u = Y - s`. The quartic path uses nine extension
/// products rather than the general form's eleven. If the weighted leading
/// coefficient cancels, the stored coordinate is unchanged and the general
/// formula remains valid.
#[inline]
pub(crate) fn weighted_affine_coefficients_shifted(
    weights: PreparedWeights,
    left: E,
    delta: E,
) -> [E; STORED_COEFFICIENTS] {
    use super::opening_field_ops::mul;
    let Some(depressed) = weights.depressed else {
        return weighted_affine_coefficients(weights, left, delta);
    };
    let delta_one = mul(depressed.leading, delta);
    let delta_two = mul(delta_one, delta);
    let delta_three = mul(delta_two, delta);
    let delta_four = mul(delta_three, delta);
    let left_squared = mul(left, left);
    let twice_quadratic = depressed.quadratic + depressed.quadratic;
    let four_left_squared = left_squared + left_squared + left_squared + left_squared;
    let six_left_squared = four_left_squared + left_squared + left_squared;
    let four_left = left + left + left + left;
    [
        mul(
            delta_one,
            mul(left, four_left_squared + twice_quadratic) + depressed.linear,
        ),
        mul(delta_two, six_left_squared + depressed.quadratic),
        mul(delta_three, four_left),
        delta_four,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(polynomial: QuarticPolynomial, weight: i64) -> WeightedQuartic {
        WeightedQuartic {
            polynomial,
            weight: E::from_i64(weight),
        }
    }

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

    #[test]
    fn aggregated_quartic_coefficients_match_independent_term_sums() {
        let cases = [
            vec![term(QuarticPolynomial::Low, 3)],
            vec![term(QuarticPolynomial::High, 5)],
            vec![
                term(QuarticPolynomial::Low, 3),
                term(QuarticPolynomial::High, 7),
            ],
            vec![
                term(QuarticPolynomial::Low, 3),
                term(QuarticPolynomial::Low, 5),
                term(QuarticPolynomial::High, 7),
            ],
            vec![
                term(QuarticPolynomial::Low, 3),
                term(QuarticPolynomial::Low, -3),
                term(QuarticPolynomial::High, 0),
            ],
        ];
        for terms in cases {
            let weights = aggregate_weights(&terms);
            for (left, delta) in [
                (E::from_i64(-9), E::from_i64(4)),
                (E::from_i64(17), E::from_i64(-6)),
            ] {
                let mut reference = [E::zero(); STORED_COEFFICIENTS];
                for term in &terms {
                    let coefficients = term.polynomial.affine_coefficients(left, delta);
                    for (reference, coefficient) in reference.iter_mut().zip(coefficients) {
                        *reference += term.weight * coefficient;
                    }
                }
                assert_eq!(
                    weighted_affine_coefficients(weights, left, delta),
                    reference
                );
            }
        }
    }

    #[test]
    fn prepared_quartics_match_full_extension_and_cancelling_weights() {
        use jolt_field::Prime64Offset59 as F;
        let mut state = 0x0087_6543_21ab_cdef_u64;
        let mut random = || {
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                F::from_u64(state)
            };
            E::new(next(), next())
        };
        for case in 0..4096 {
            let low = random();
            let high = match case % 5 {
                0 => E::zero(),
                1 => E::zero() - low,
                _ => random(),
            };
            let low = if case % 5 == 2 { E::zero() } else { low };
            let left = match case % 11 {
                0 => E::zero(),
                1 => E::from_u64(1),
                _ => random(),
            };
            let delta = match case % 13 {
                0 => E::zero(),
                1 => E::from_u64(1),
                _ => random(),
            };
            let terms = [
                WeightedQuartic {
                    polynomial: QuarticPolynomial::Low,
                    weight: low,
                },
                WeightedQuartic {
                    polynomial: QuarticPolynomial::High,
                    weight: high,
                },
            ];
            let mut expected = [E::zero(); STORED_COEFFICIENTS];
            for term in terms {
                let coefficients = term.polynomial.affine_coefficients(left, delta);
                for (sum, coefficient) in expected.iter_mut().zip(coefficients) {
                    *sum += term.weight * coefficient;
                }
            }
            assert_eq!(
                weighted_affine_coefficients(aggregate_weights(&terms), left, delta),
                expected
            );
            let prepared = aggregate_weights(&terms);
            assert_eq!(
                weighted_affine_coefficients_shifted(prepared, left - prepared.shift(), delta,),
                expected
            );
            let powers = [
                delta,
                delta * delta,
                delta * delta * delta,
                delta * delta * delta * delta,
            ];
            assert_eq!(
                core::array::from_fn::<_, 4, _>(
                    |i| aggregate_weights(&terms).taylor(left)[i] * powers[i]
                ),
                expected
            );
        }
    }

    #[test]
    fn low_quartic_depresses_to_the_expected_shifted_form() {
        let prepared = aggregate_weights(&[term(QuarticPolynomial::Low, 1)]);
        let depressed = prepared
            .depressed
            .expect("the low quartic has a nonzero leading coefficient");
        assert_eq!(depressed.shift, E::from_i64(5));
        assert_eq!(depressed.quadratic, E::from_i64(-42));
        assert_eq!(depressed.linear, E::from_i64(-64));

        for u in [E::from_i64(1), E::from_i64(-5), E::from_i64(7)] {
            let shifted =
                u * u * u * u - E::from_i64(42) * u * u - E::from_i64(64) * u + E::from_i64(105);
            assert_eq!(
                shifted,
                QuarticPolynomial::Low.evaluate(u + depressed.shift)
            );
        }
    }
}
