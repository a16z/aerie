//! Exact Falcon padded-signature encodability witness.

use crate::{
    error::{AerieError, Result},
    params::{filled, Falcon512, FalconParams, COMPRESSED_COEFFICIENT_BASE_BITS},
    source::FALCON_COMPRESSED_COEFFICIENT_LIMIT,
};

/// Bits available for all unary high parts of a Falcon-512 body after the
/// fixed coefficient cost.
pub const FALCON_512_COMPRESSED_HIGH_BITS_BUDGET: usize = Falcon512::COMPRESSED_HIGH_BITS_BUDGET;

/// Offset that centers the semantic unary digit `a2 + 8` in `[-8, 7]`.
pub const A2_OFFSET: i16 = 8;

/// The four committed cells that determine one virtual `s2` coefficient.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodedS2Coefficient {
    a0: i8,
    a1: i8,
    a2: i8,
    sigma: i8,
}

impl EncodedS2Coefficient {
    /// Validate the 3-bit low digit, two 4-bit high digits, and sign cell.
    #[inline]
    pub fn from_cells(a0: i8, a1: i8, a2: i8, sigma: i8) -> Result<Self> {
        if !(-4..=3).contains(&a0) {
            return Err(AerieError::InvalidWitness(
                "balanced radix-8 digit a0 is outside [-4, 3]".to_owned(),
            ));
        }
        if !(-8..=7).contains(&a1) {
            return Err(AerieError::InvalidWitness(
                "balanced radix-16 digit a1 is outside [-8, 7]".to_owned(),
            ));
        }
        if !(-8..=7).contains(&a2) {
            return Err(AerieError::InvalidWitness(
                "balanced radix-16 digit a2 is outside [-8, 7]".to_owned(),
            ));
        }
        if sigma != -1 && sigma != 1 {
            return Err(AerieError::InvalidWitness(
                "sign cell sigma is outside {-1, 1}".to_owned(),
            ));
        }
        Ok(Self { a0, a1, a2, sigma })
    }

    /// Canonically encode one valid Falcon coefficient. Zero uses positive sign.
    #[inline]
    pub fn encode(coefficient: i16) -> Result<Self> {
        let magnitude = coefficient.unsigned_abs();
        if magnitude > FALCON_COMPRESSED_COEFFICIENT_LIMIT {
            return Err(AerieError::InvalidWitness(format!(
                "s2 coefficient {coefficient} exceeds the Falcon compressed-coefficient limit"
            )));
        }
        let a0 = i8::try_from(magnitude % 8).expect("radix-8 digit fits i8") - 4;
        let a1 = i8::try_from((magnitude / 8) % 16).expect("radix-16 digit fits i8") - 8;
        let semantic_a2 = i8::try_from(magnitude / 128).expect("unary digit fits i8");
        let a2 = semantic_a2 - i8::try_from(A2_OFFSET).expect("offset fits i8");
        let sigma = if coefficient < 0 { -1 } else { 1 };
        Self::from_cells(a0, a1, a2, sigma)
    }

    /// Decode the nonnegative virtual magnitude using Section 3.2's offset.
    #[must_use]
    pub const fn magnitude(self) -> i16 {
        self.a0 as i16 + 8 * self.a1 as i16 + 128 * self.a2 as i16 + 1_092
    }

    /// Decode the virtual signed coefficient.
    #[must_use]
    pub const fn coefficient(self) -> i16 {
        self.sigma as i16 * self.magnitude()
    }

    /// Semantic unsigned unary high part used by the Falcon bit budget.
    #[must_use]
    pub const fn semantic_high(self) -> i16 {
        self.a2 as i16 + A2_OFFSET
    }

    #[must_use]
    pub const fn cells(self) -> [i8; 4] {
        [self.a0, self.a1, self.a2, self.sigma]
    }
}

/// Exact no-padding encodability source for one Falcon polynomial.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FourCellEncodabilityWitness<P: FalconParams> {
    a0: P::Array<i8>,
    a1: P::Array<i8>,
    a2: P::Array<i8>,
    sigma: P::Array<i8>,
    budget_digits: [i8; 3],
}

impl<P: FalconParams> FourCellEncodabilityWitness<P> {
    /// Build the four coefficient tables and three row-domain budget cells.
    pub fn build(s2: &P::Array<i16>) -> Result<Self> {
        let encodability = PaddedEncodabilityWitness::<P>::build(s2)?;
        let mut a0 = filled::<P, i8>(0);
        let mut a1 = filled::<P, i8>(0);
        let mut a2 = filled::<P, i8>(0);
        let mut sigma = filled::<P, i8>(0);
        for (index, &coefficient) in s2.as_ref().iter().enumerate() {
            let cells = EncodedS2Coefficient::encode(coefficient)?.cells();
            a0.as_mut()[index] = cells[0];
            a1.as_mut()[index] = cells[1];
            a2.as_mut()[index] = cells[2];
            sigma.as_mut()[index] = cells[3];
        }
        Ok(Self {
            a0,
            a1,
            a2,
            sigma,
            budget_digits: encode_budget_slack::<P>(encodability.slack()),
        })
    }

    #[must_use]
    pub const fn a0(&self) -> &P::Array<i8> {
        &self.a0
    }

    #[must_use]
    pub const fn a1(&self) -> &P::Array<i8> {
        &self.a1
    }

    #[must_use]
    pub const fn a2(&self) -> &P::Array<i8> {
        &self.a2
    }

    #[must_use]
    pub const fn sigma(&self) -> &P::Array<i8> {
        &self.sigma
    }

    #[must_use]
    pub const fn budget_digits(&self) -> &[i8; 3] {
        &self.budget_digits
    }

    /// Decode one coefficient from the stored tables.
    #[must_use]
    pub fn coefficient(&self, index: usize) -> Option<i16> {
        if index >= P::DEGREE {
            return None;
        }
        Some(
            EncodedS2Coefficient {
                a0: self.a0.as_ref()[index],
                a1: self.a1.as_ref()[index],
                a2: self.a2.as_ref()[index],
                sigma: self.sigma.as_ref()[index],
            }
            .coefficient(),
        )
    }

    /// Decode the unsigned budget slack from its three balanced cells.
    ///
    /// Falcon-512 uses three radix-8 digits and the constant 292; Falcon-1024
    /// uses radices `(8, 8, 16)` and the constant 548.
    #[must_use]
    pub const fn budget_slack(&self) -> i16 {
        let radices = P::BUDGET_DIGIT_RADICES;
        let weight1 = radices[0] as i16;
        let weight2 = weight1 * radices[1] as i16;
        self.budget_digits[0] as i16
            + weight1 * self.budget_digits[1] as i16
            + weight2 * self.budget_digits[2] as i16
            + P::BUDGET_CONSTANT
    }
}

fn encode_budget_slack<P: FalconParams>(slack: u16) -> [i8; 3] {
    let mut remaining = slack;
    let mut digits = [0_i8; 3];
    for (digit, radix) in digits.iter_mut().zip(P::BUDGET_DIGIT_RADICES) {
        let radix = u16::from(radix);
        *digit = i8::try_from(remaining % radix).expect("balanced digit fits i8")
            - i8::try_from(radix / 2).expect("half radix fits i8");
        remaining /= radix;
    }
    debug_assert_eq!(remaining, 0);
    digits
}

/// Deterministic auxiliaries for the exact Falcon padded encoding test.
///
/// For coefficient `x`, Falcon writes one sign bit, seven low magnitude bits,
/// `floor(abs(x) / 128)` zero bits, and one unary terminator. Consequently a
/// body fits exactly when every magnitude is at most 2047 and the sum of the
/// unary high parts is at most the parameter set's unary budget (392 bits for
/// Falcon-512, 696 bits for Falcon-1024).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PaddedEncodabilityWitness<P: FalconParams> {
    high_parts: P::Array<u8>,
    high_sum: u16,
    slack: u16,
}

impl<P: FalconParams> PaddedEncodabilityWitness<P> {
    /// Checked constructor: enforces the coefficient-magnitude limit and the
    /// unary high-part budget, so every instance is a valid witness.
    pub fn build(s2: &P::Array<i16>) -> Result<Self> {
        let mut high_parts = filled::<P, u8>(0);
        let mut high_sum = 0_usize;

        for (index, (coefficient, high_part)) in s2
            .as_ref()
            .iter()
            .zip(high_parts.as_mut().iter_mut())
            .enumerate()
        {
            let magnitude = coefficient.unsigned_abs();
            if magnitude > FALCON_COMPRESSED_COEFFICIENT_LIMIT {
                return Err(AerieError::InvalidWitness(format!(
                    "s2[{index}]={coefficient} exceeds the Falcon compressed-coefficient limit"
                )));
            }
            *high_part = (magnitude >> 7) as u8;
            high_sum = high_sum.checked_add(usize::from(*high_part)).ok_or(
                AerieError::IntegerOverflow("Falcon compressed high-part sum"),
            )?;
        }

        if high_sum > P::COMPRESSED_HIGH_BITS_BUDGET {
            return Err(AerieError::InvalidWitness(format!(
                "Falcon compressed body needs {} bits; padded capacity is {}",
                P::DEGREE * COMPRESSED_COEFFICIENT_BASE_BITS + high_sum,
                P::PADDED_BODY_BITS
            )));
        }

        Ok(Self {
            high_parts,
            high_sum: high_sum as u16,
            slack: (P::COMPRESSED_HIGH_BITS_BUDGET - high_sum) as u16,
        })
    }

    #[must_use]
    pub const fn high_parts(&self) -> &P::Array<u8> {
        &self.high_parts
    }

    #[must_use]
    pub const fn high_sum(&self) -> u16 {
        self.high_sum
    }

    #[must_use]
    pub const fn slack(&self) -> u16 {
        self.slack
    }

    #[must_use]
    pub const fn encoded_bits(&self) -> usize {
        P::DEGREE * COMPRESSED_COEFFICIENT_BASE_BITS + self.high_sum as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Falcon1024;

    fn budget_boundary_for<P: FalconParams>() {
        let budget = P::COMPRESSED_HIGH_BITS_BUDGET;
        for (total, accepts) in [(0, true), (budget, true), (budget + 1, false)] {
            let mut s2 = filled::<P, i16>(0);
            for cell in s2.as_mut().iter_mut().take(total) {
                *cell = 128;
            }
            let witness = PaddedEncodabilityWitness::<P>::build(&s2);
            if accepts {
                let witness = witness.expect("total within the budget");
                assert_eq!(witness.high_sum(), total as u16);
                assert_eq!(witness.slack(), (budget - total) as u16);
                assert_eq!(
                    witness.encoded_bits(),
                    P::DEGREE * COMPRESSED_COEFFICIENT_BASE_BITS + total
                );
            } else {
                assert!(witness.is_err(), "total {total} must exceed the budget");
            }
        }
    }

    #[test]
    fn budget_boundary_is_exact_for_both_parameter_sets() {
        assert_eq!(Falcon512::COMPRESSED_HIGH_BITS_BUDGET, 392);
        assert_eq!(Falcon1024::COMPRESSED_HIGH_BITS_BUDGET, 696);
        budget_boundary_for::<Falcon512>();
        budget_boundary_for::<Falcon1024>();
    }

    #[test]
    fn every_signed_344_digit_tuple_decodes_to_the_exact_falcon_container() {
        let mut count = 0;
        for a0 in -4..=3 {
            for a1 in -8..=7 {
                for a2 in -8..=7 {
                    for sigma in [1, -1] {
                        let encoded = EncodedS2Coefficient::from_cells(a0, a1, a2, sigma)
                            .expect("valid source tuple");
                        assert!((0..=2_047).contains(&encoded.magnitude()));
                        assert!((-2_047..=2_047).contains(&encoded.coefficient()));
                        assert_eq!(encoded.semantic_high(), encoded.magnitude() / 128);
                        count += 1;
                    }
                }
            }
        }
        assert_eq!(count, 4_096);
    }

    #[test]
    fn out_of_range_cells_are_rejected() {
        assert!(EncodedS2Coefficient::from_cells(-5, 0, 0, 1).is_err());
        assert!(EncodedS2Coefficient::from_cells(4, 0, 0, 1).is_err());
        assert!(EncodedS2Coefficient::from_cells(0, -9, 0, 1).is_err());
        assert!(EncodedS2Coefficient::from_cells(0, 8, 0, 1).is_err());
        assert!(EncodedS2Coefficient::from_cells(0, 0, -9, 1).is_err());
        assert!(EncodedS2Coefficient::from_cells(0, 0, 8, 1).is_err());
        for sigma in [0, 2, -2] {
            assert!(EncodedS2Coefficient::from_cells(0, 0, 0, sigma).is_err());
        }
    }

    #[test]
    fn canonical_encoder_round_trips_every_coefficient_and_canonicalizes_zero() {
        for coefficient in -2_047..=2_047 {
            let encoded = EncodedS2Coefficient::encode(coefficient).expect("encodable coefficient");
            assert_eq!(encoded.coefficient(), coefficient);
            if coefficient == 0 {
                assert_eq!(encoded.cells()[3], 1, "zero uses the positive sign");
                let negative_zero = EncodedS2Coefficient::from_cells(
                    encoded.cells()[0],
                    encoded.cells()[1],
                    encoded.cells()[2],
                    -1,
                )
                .expect("negative zero is a valid committed encoding");
                assert_eq!(
                    negative_zero.coefficient(),
                    0,
                    "both zero signs decode to zero"
                );
            }
        }
        assert!(EncodedS2Coefficient::encode(2_048).is_err());
        assert!(EncodedS2Coefficient::encode(-2_048).is_err());
    }

    fn full_witness_for<P: FalconParams>() {
        let mut s2 = filled::<P, i16>(0);
        s2.as_mut()[0] = -2_047;
        s2.as_mut()[1] = 2_047;
        let witness = FourCellEncodabilityWitness::<P>::build(&s2).expect("encodability witness");
        assert_eq!(witness.coefficient(0), Some(-2_047));
        assert_eq!(witness.coefficient(1), Some(2_047));
        assert_eq!(witness.coefficient(P::DEGREE), None);
        assert_eq!(
            witness.budget_slack(),
            i16::try_from(P::COMPRESSED_HIGH_BITS_BUDGET - 30).unwrap()
        );
        let radices = P::BUDGET_DIGIT_RADICES;
        for (digit, radix) in witness.budget_digits().iter().zip(radices) {
            let half = i8::try_from(radix / 2).unwrap();
            assert!(
                (-half..half).contains(digit),
                "{digit} outside radix {radix}"
            );
        }
        assert_eq!(witness.a0().as_ref().len(), P::DEGREE);
        assert_eq!(witness.a2().as_ref().len(), P::DEGREE);
        assert_eq!(witness.sigma().as_ref().len(), P::DEGREE);
    }

    #[test]
    fn full_witness_uses_native_344_tables_and_balanced_budget_digits() {
        full_witness_for::<Falcon512>();
        full_witness_for::<Falcon1024>();
    }

    fn budget_digits_round_trip_for<P: FalconParams>() {
        for slack in 0..=P::COMPRESSED_HIGH_BITS_BUDGET {
            let digits = encode_budget_slack::<P>(slack as u16);
            let witness = FourCellEncodabilityWitness::<P> {
                a0: filled::<P, i8>(0),
                a1: filled::<P, i8>(0),
                a2: filled::<P, i8>(0),
                sigma: filled::<P, i8>(0),
                budget_digits: digits,
            };
            assert_eq!(
                witness.budget_slack(),
                i16::try_from(slack).unwrap(),
                "slack {slack}"
            );
        }
    }

    #[test]
    fn budget_digits_round_trip_over_the_whole_reachable_slack() {
        budget_digits_round_trip_for::<Falcon512>();
        budget_digits_round_trip_for::<Falcon1024>();
    }
}
