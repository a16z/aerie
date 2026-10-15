//! Direct Falcon relation witness, following the representation used by `LaZer`.
//!
//! For every row we retain the exact integer lift
//! `s1 + h*s2 - t - q*v - (X^n + 1)*w = 0`.
//! Coefficients keep their integer representation and are not expanded into
//! mixed-radix one-hot selectors.
//!
//! Every row and batch type is generic over a sealed Falcon parameter set
//! ([`DirectParams`], implemented for `Falcon512` and `Falcon1024`). Specify
//! the parameter set for each row. Batches carry their degree implicitly as
//! `cells / padded_rows`.

#[cfg(any(test, feature = "test-fixtures"))]
use aerie_falcon_native::params::filled;
use aerie_falcon_native::{
    params::{FixedArray, FALCON_Q, S_CAP},
    source::FALCON_COMPRESSED_COEFFICIENT_LIMIT,
    Falcon1024, Falcon512, FalconParams,
};

mod compact;
mod convolution;
mod validation;

pub use compact::CompactDirectBatch;
#[doc(hidden)]
pub use convolution::{
    convolution_i16, convolution_negacyclic_i16, convolution_negacyclic_i16_from_mod_q,
};

use crate::native::witness::VerifiedFalconSignature;
pub use aerie_falcon_statement::{DirectError, DirectParams};

#[cfg(feature = "parallel")]
const DIRECT_CHUNK_ROWS: usize = 256;

/// Supported degrees, for batches whose degree is recovered from their shape.
pub(crate) fn supported_degree(degree: usize) -> Result<usize, DirectError> {
    if degree == Falcon512::DEGREE || degree == Falcon1024::DEGREE {
        Ok(degree)
    } else {
        Err(DirectError::InvalidLength)
    }
}

/// One exact direct-relation row for parameter set `P`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectRow<P: DirectParams> {
    pub h: Box<P::Array<i16>>,
    pub t: Box<P::Array<i16>>,
    pub s1: Box<P::Array<i16>>,
    pub s2: Box<P::Array<i16>>,
    pub v: Box<P::Array<i64>>,
    pub w: Box<P::Array<i64>>,
    pub slack: [i16; 4],
}

impl<P: DirectParams> DirectRow<P> {
    pub fn new(
        h: P::Array<i16>,
        t: P::Array<i16>,
        s1: P::Array<i16>,
        s2: P::Array<i16>,
    ) -> Result<Self, DirectError> {
        let auxiliaries =
            DirectAuxiliaries::derive::<P>(h.as_ref(), t.as_ref(), s1.as_ref(), s2.as_ref(), None)?;
        Ok(Self {
            h: Box::new(h),
            t: Box::new(t),
            s1: Box::new(s1),
            s2: Box::new(s2),
            v: Box::new(auxiliaries.v_array::<P>()),
            w: Box::new(auxiliaries.w_array::<P>()),
            slack: auxiliaries.slack,
        })
    }

    pub fn from_verified(value: &VerifiedFalconSignature<P>) -> Result<Self, DirectError> {
        let auxiliaries = DirectAuxiliaries::derive::<P>(
            value.h_centered().as_ref(),
            value.t_centered().as_ref(),
            value.s1().as_ref(),
            value.s2().as_ref(),
            Some(value.norm_squared()),
        )?;
        Ok(Self {
            h: Box::new(*value.h_centered()),
            t: Box::new(*value.t_centered()),
            s1: Box::new(*value.s1()),
            s2: Box::new(*value.s2()),
            v: Box::new(auxiliaries.v_array::<P>()),
            w: Box::new(auxiliaries.w_array::<P>()),
            slack: auxiliaries.slack,
        })
    }

    /// Deterministic relation-positive row for performance testing.
    ///
    /// This is not an honestly key-generated Falcon signature. Its sparse
    /// public polynomial makes fixture creation cheap without weakening the
    /// work measured after construction.
    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn synthetic(seed: u64) -> Result<Self, DirectError> {
        let degree = P::DEGREE;
        let mut state = seed;
        let mut h = filled::<P, i16>(0);
        for _ in 0..8 {
            let index = (splitmix64(&mut state) as usize) & (degree - 1);
            h.as_mut()[index] = ((splitmix64(&mut state) % 31) as i16) - 15;
        }
        let mut s1 = filled::<P, i16>(0);
        let mut s2 = filled::<P, i16>(0);
        for j in 0..degree {
            s1.as_mut()[j] = ((splitmix64(&mut state) % 65) as i16) - 32;
            s2.as_mut()[j] = ((splitmix64(&mut state) % 65) as i16) - 32;
        }
        let product =
            negacyclic_from_ordinary(&convolution_i16::<P>(h.as_ref(), s2.as_ref()), degree);
        let mut t = filled::<P, i16>(0);
        for (slot, (&s1, &product)) in t.as_mut().iter_mut().zip(s1.as_ref().iter().zip(&product)) {
            *slot = center_mod_q(i64::from(s1) + product);
        }
        Self::new(h, t, s1, s2)
    }
}

struct DirectAuxiliaries {
    v: Vec<i64>,
    w: Vec<i64>,
    slack: [i16; 4],
}

impl DirectAuxiliaries {
    fn derive<P: DirectParams>(
        h: &[i16],
        t: &[i16],
        s1: &[i16],
        s2: &[i16],
        verified_norm: Option<u64>,
    ) -> Result<Self, DirectError> {
        let degree = P::DEGREE;
        if [h.len(), t.len(), s1.len(), s2.len()] != [degree; 4] {
            return Err(DirectError::InvalidLength);
        }
        let norm = verified_norm.map_or_else(|| squared_norm(s1, s2), Ok)?;
        if norm > P::BETA_SQUARED {
            return Err(DirectError::NormExceeded(norm));
        }
        let product = convolution_i16::<P>(h, s2);
        let mut w = vec![0_i64; degree];
        w[..degree - 1].copy_from_slice(&product[degree..2 * degree - 1]);
        let mut v = vec![0_i64; degree];
        for j in 0..degree {
            let numerator = product[j]
                .checked_add(i64::from(s1[j]))
                .and_then(|x| x.checked_sub(i64::from(t[j])))
                .and_then(|x| x.checked_sub(w[j]))
                .ok_or(DirectError::IntegerOverflow)?;
            if numerator % FALCON_Q != 0 {
                return Err(DirectError::NonIntegralLift(j));
            }
            v[j] = numerator / FALCON_Q;
        }
        let slack = P::BETA_SQUARED - norm;
        let slack_u64 =
            four_squares(slack).ok_or(DirectError::CappedFourSquaresUnavailable(slack))?;
        let mut slack = [0_i16; 4];
        for (destination, value) in slack.iter_mut().zip(slack_u64) {
            *destination = i16::try_from(value).map_err(|_| DirectError::IntegerOverflow)?;
        }
        check_witness_ranges::<P>(s1, s2, &v, &w, slack)?;
        check_exact_lift::<P>(t, s1, &v, &w, slack, norm, &product)?;
        Ok(Self { v, w, slack })
    }

    fn v_array<P: DirectParams>(&self) -> P::Array<i64> {
        <P::Array<i64> as FixedArray<i64>>::from_slice(&self.v).expect("derived at the degree")
    }

    fn w_array<P: DirectParams>(&self) -> P::Array<i64> {
        <P::Array<i64> as FixedArray<i64>>::from_slice(&self.w).expect("derived at the degree")
    }
}

fn check_exact_lift<P: DirectParams>(
    t: &[i16],
    s1: &[i16],
    v: &[i64],
    w: &[i64],
    slack: [i16; 4],
    norm: u64,
    product: &[i64],
) -> Result<(), DirectError> {
    let degree = P::DEGREE;
    debug_assert_eq!(product.len(), 2 * degree);
    for degree_index in 0..2 * degree {
        let mut value = product[degree_index];
        if degree_index < degree {
            value += i64::from(s1[degree_index]) - i64::from(t[degree_index]);
            value -= FALCON_Q * v[degree_index];
            value -= w[degree_index];
        } else {
            value -= w[degree_index - degree];
        }
        if value != 0 {
            return Err(DirectError::RelationFailure(degree_index));
        }
    }
    let completed_norm = norm
        + slack
            .iter()
            .map(|&x| i64::from(x).unsigned_abs().pow(2))
            .sum::<u64>();
    if completed_norm != P::BETA_SQUARED {
        return Err(DirectError::NormExceeded(completed_norm));
    }
    Ok(())
}

fn check_witness_ranges<P: DirectParams>(
    s1: &[i16],
    s2: &[i16],
    v: &[i64],
    w: &[i64],
    slack: [i16; 4],
) -> Result<(), DirectError> {
    check_falcon_v1_witness_ranges::<P>(s1, s2, v, slack)?;
    check_range(
        "w",
        w.iter().copied(),
        -P::W_WITNESS_BOUND,
        P::W_WITNESS_BOUND - 1,
    )?;
    let last = P::DEGREE - 1;
    if w[last] != 0 {
        return Err(DirectError::RangeViolation {
            family: "w-fixed-zero",
            index: last,
            value: w[last],
        });
    }
    Ok(())
}

fn check_falcon_v1_witness_ranges<P: DirectParams>(
    s1: &[i16],
    s2: &[i16],
    v: &[i64],
    slack: [i16; 4],
) -> Result<(), DirectError> {
    let small_min = -P::SMALL_WITNESS_BOUND;
    let small_max = P::SMALL_WITNESS_BOUND - 1;
    check_range("s1", s1.iter().map(|&x| i64::from(x)), small_min, small_max)?;
    check_range("s2", s2.iter().map(|&x| i64::from(x)), small_min, small_max)?;
    check_range(
        "slack",
        slack.iter().map(|&x| i64::from(x)),
        small_min,
        small_max,
    )?;
    check_range(
        "v",
        v.iter().copied(),
        -P::V_WITNESS_BOUND,
        P::V_WITNESS_BOUND - 1,
    )?;
    let mut high_sum = 0_usize;
    for (index, &coefficient) in s2.iter().enumerate() {
        let magnitude = coefficient.unsigned_abs();
        if magnitude > FALCON_COMPRESSED_COEFFICIENT_LIMIT {
            return Err(DirectError::RangeViolation {
                family: "s2-encodable",
                index,
                value: i64::from(coefficient),
            });
        }
        high_sum = high_sum
            .checked_add(usize::from(magnitude >> 7))
            .ok_or(DirectError::IntegerOverflow)?;
    }
    if high_sum > P::COMPRESSED_HIGH_BITS_BUDGET {
        return Err(DirectError::EncodingBudgetExceeded(high_sum));
    }
    Ok(())
}

fn check_range(
    family: &'static str,
    values: impl Iterator<Item = i64>,
    minimum: i64,
    maximum: i64,
) -> Result<(), DirectError> {
    for (index, value) in values.enumerate() {
        if !(minimum..=maximum).contains(&value) {
            return Err(DirectError::RangeViolation {
                family,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn squared_norm(s1: &[i16], s2: &[i16]) -> Result<u64, DirectError> {
    s1.iter().chain(s2).try_fold(0_u64, |sum, &x| {
        sum.checked_add(i64::from(x).unsigned_abs().pow(2))
            .ok_or(DirectError::IntegerOverflow)
    })
}

#[cfg(any(test, feature = "test-fixtures"))]
fn negacyclic_from_ordinary(product: &[i64], degree: usize) -> Vec<i64> {
    (0..degree)
        .map(|j| product[j] - product.get(j + degree).copied().unwrap_or(0))
        .collect()
}

#[cfg(any(test, feature = "test-fixtures"))]
fn center_mod_q(value: i64) -> i16 {
    let residue = value.rem_euclid(FALCON_Q);
    let centered = if residue > (FALCON_Q - 1) / 2 {
        residue - FALCON_Q
    } else {
        residue
    };
    centered as i16
}

/// Prepare the reusable CRT primes, twiddles, reconstruction constants, and
/// host SIMD dispatch used by dense direct-witness convolution, for both
/// degrees.
pub fn prepare_direct_witness_convolution() {
    convolution::prepare();
}

fn four_squares(value: u64) -> Option<[u64; 4]> {
    let cap = (S_CAP - 1) as u64;
    for a in (0..=value.isqrt().min(cap)).rev() {
        let r1 = value - a * a;
        // The remainder grows as a decreases, so no later candidate can fit.
        if r1.isqrt() > cap {
            break;
        }
        if !is_three_square(r1) {
            continue;
        }
        for b in (0..=r1.isqrt()).rev() {
            let r2 = r1 - b * b;
            if let Some((c, d)) = two_squares(r2) {
                return Some([a, b, c, d]);
            }
        }
    }
    None
}

fn is_three_square(mut value: u64) -> bool {
    while value != 0 && value.is_multiple_of(4) {
        value /= 4;
    }
    value % 8 != 7
}

fn two_squares(value: u64) -> Option<(u64, u64)> {
    for a in (0..=value.isqrt()).rev() {
        let b2 = value - a * a;
        let b = b2.isqrt();
        if b * b == b2 {
            return Some((a, b));
        }
    }
    None
}

#[cfg(any(test, feature = "test-fixtures"))]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_row_definition<P: DirectParams>(row: &DirectRow<P>) {
        let mut product = vec![0_i64; 2 * P::DEGREE];
        for (i, &h) in row.h.as_ref().as_ref().iter().enumerate() {
            for (j, &s2) in row.s2.as_ref().as_ref().iter().enumerate() {
                product[i + j] += i64::from(h) * i64::from(s2);
            }
        }
        for index in 0..P::DEGREE {
            assert_eq!(product[index + P::DEGREE], row.w.as_ref().as_ref()[index]);
            assert_eq!(
                product[index] + i64::from(row.s1.as_ref().as_ref()[index])
                    - i64::from(row.t.as_ref().as_ref()[index]),
                FALCON_Q * row.v.as_ref().as_ref()[index] + row.w.as_ref().as_ref()[index],
            );
        }
        let norm = row
            .s1
            .as_ref()
            .as_ref()
            .iter()
            .chain(row.s2.as_ref().as_ref())
            .chain(&row.slack)
            .map(|&value| i64::from(value).unsigned_abs().pow(2))
            .sum::<u64>();
        assert_eq!(norm, P::BETA_SQUARED);
        check_witness_ranges::<P>(
            row.s1.as_ref().as_ref(),
            row.s2.as_ref().as_ref(),
            row.v.as_ref().as_ref(),
            row.w.as_ref().as_ref(),
            row.slack,
        )
        .unwrap();
    }

    fn assert_capped_squares(delta: u64) -> [u64; 4] {
        let squares = four_squares(delta).expect("admissible slack has a decomposition");
        assert_eq!(squares.iter().map(|x| x * x).sum::<u64>(), delta);
        assert!(squares.iter().all(|&x| x < S_CAP as u64));
        squares
    }

    #[test]
    fn four_squares_decomposes_every_falcon_1024_gap_above_the_cap_square() {
        // Only gaps above 8191^2 can force a component above the cap; check
        // every one of them up to the Falcon-1024 bound.
        let boundary = (S_CAP as u64 - 1).pow(2);
        for delta in boundary + 1..=Falcon1024::BETA_SQUARED {
            assert_capped_squares(delta);
        }
    }

    #[test]
    fn four_squares_decomposes_small_and_power_of_four_gaps() {
        // 8192^2 = 4 * 4096^2 and similar high powers of four rule out every
        // first component near the cap by Legendre's three-square theorem.
        for delta in [
            0,
            1,
            7,
            Falcon512::BETA_SQUARED,
            (S_CAP as u64 - 1).pow(2),
            67_108_864,
            69_206_016,
            70_254_592,
        ] {
            assert_capped_squares(delta);
        }
        let mut state = 7;
        for _ in 0..4096 {
            assert_capped_squares(splitmix64(&mut state) % (Falcon1024::BETA_SQUARED + 1));
        }
        assert_eq!(four_squares(u64::MAX), None);
    }

    #[test]
    fn maximum_slack_witness_and_shared_carrier_at_both_degrees() {
        fn check<P: DirectParams>() {
            let row = DirectRow::<P>::new(
                filled::<P, i16>(0),
                filled::<P, i16>(0),
                filled::<P, i16>(0),
                filled::<P, i16>(0),
            )
            .unwrap();
            assert_row_definition(&row);
            assert_eq!(
                row.slack.map(|x| x as u64),
                assert_capped_squares(P::BETA_SQUARED)
            );
            let batch = CompactDirectBatch::from_rows(&[row]);
            batch.validate_semantics().unwrap();
            for bad in [S_CAP, -S_CAP - 1] {
                let mut changed = batch.clone();
                changed.slack[0] = bad as i16;
                assert!(matches!(
                    changed.validate_semantics(),
                    Err(DirectError::RangeViolation {
                        family: "slack",
                        ..
                    })
                ));
                assert!(
                    check_falcon_v1_witness_ranges::<P>(&[bad as i16], &[], &[], [0; 4]).is_err()
                );
            }
            for edge in [-S_CAP, S_CAP - 1] {
                assert!(check_falcon_v1_witness_ranges::<P>(
                    &[edge as i16],
                    &[],
                    &[],
                    [edge as i16; 4]
                )
                .is_ok());
            }
        }
        check::<Falcon512>();
        check::<Falcon1024>();
    }

    #[test]
    fn synthetic_row_satisfies_exact_integer_lift_and_norm() {
        let row = DirectRow::<Falcon512>::synthetic(7).unwrap();
        assert_row_definition(&row);
        let wide = DirectRow::<Falcon1024>::synthetic(7).unwrap();
        assert_row_definition(&wide);
        assert_eq!(wide.h.as_ref().len(), 1024);
    }

    #[test]
    fn batch_padding_is_structural_zero() {
        let rows = (0..3)
            .map(|i| DirectRow::<Falcon512>::synthetic(i).unwrap())
            .collect::<Vec<_>>();
        let batch = CompactDirectBatch::from_rows(&rows);
        assert_eq!(batch.padded_rows, 4);
        assert_eq!(batch.degree(), 512);
        assert!(batch.s1[3 * Falcon512::DEGREE..].iter().all(|&x| x == 0));

        let wide_rows = (0..3)
            .map(|i| DirectRow::<Falcon1024>::synthetic(i).unwrap())
            .collect::<Vec<_>>();
        let wide = CompactDirectBatch::from_rows(&wide_rows);
        assert_eq!(wide.degree(), 1024);
        assert!(wide.s1[3 * 1024..].iter().all(|&x| x == 0));
    }

    #[test]
    fn falcon_1024_ranges_admit_the_wider_post_lift_bounds() {
        // Both degrees use the same signed width-14 carrier.
        let s1 = [8_191_i16];
        let s2 = [0_i16];
        assert!(check_falcon_v1_witness_ranges::<Falcon1024>(&s1, &s2, &[0], [0; 4]).is_ok());
        assert!(check_falcon_v1_witness_ranges::<Falcon512>(&s1, &s2, &[0], [0; 4]).is_ok());
        assert!(check_falcon_v1_witness_ranges::<Falcon1024>(&s1, &s2, &[134_109], [0; 4]).is_ok());
        assert!(
            check_falcon_v1_witness_ranges::<Falcon1024>(&s1, &s2, &[1 << 18], [0; 4]).is_err()
        );
        // 696 unary bits fit a Falcon-1024 body, 393 already overflow Falcon-512.
        let wide_s2 = vec![128_i16; 696];
        assert!(check_falcon_v1_witness_ranges::<Falcon1024>(&[], &wide_s2, &[], [0; 4]).is_ok());
        let narrow_s2 = vec![128_i16; 393];
        assert!(check_falcon_v1_witness_ranges::<Falcon512>(&[], &narrow_s2, &[], [0; 4]).is_err());
    }
}
