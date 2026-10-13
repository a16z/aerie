//! Direct Falcon relation witness, following the representation used by `LaZer`.
//!
//! For every row we retain the exact integer lift
//! `s1 + h*s2 - t - q*v - (X^n + 1)*w = 0`.
//! Unlike the original Aerie construction, coefficients are not expanded into
//! mixed-radix one-hot selectors.
//!
//! Every row and batch type is generic over a sealed Falcon parameter set
//! ([`DirectParams`], implemented for `Falcon512` and `Falcon1024`). Specify
//! the parameter set for each row. Batches carry their degree implicitly as
//! `cells / padded_rows`.

use aerie_falcon_native::{
    encodability::FALCON_512_COMPRESSED_HIGH_BITS_BUDGET,
    params::{filled, FixedArray, S_CAP},
    source::FALCON_COMPRESSED_COEFFICIENT_LIMIT,
    witness::VerifiedFalconSignature,
    Falcon1024, Falcon512, FalconParams,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

mod compact;
mod convolution;
mod validation;

pub use compact::CompactDirectBatch;
pub(crate) use convolution::ExactConvolution;

pub const FALCON_DEGREE: usize = 512;
pub const FALCON_Q: i64 = 12_289;
pub const FALCON_BETA_SQUARED: u64 = 34_034_726;
/// Integer bounds implied by the Falcon norm and ring identities (Falcon-512).
pub const FALCON_NORM_COEFFICIENT_ABS_BOUND: i64 = 5_833;
pub const FALCON_QUOTIENT_ABS_BOUND: i64 = 65_998;

#[cfg(feature = "parallel")]
const DIRECT_CHUNK_ROWS: usize = 256;

/// Host-side integer ranges of the direct witness for one parameter set.
///
/// The norm and quotient bounds are the values the Falcon protocol proves
/// after its two integer lifts (`specs/falcon.md` Section 9).
/// The carrier bounds are exclusive
/// powers of two that contain them.
pub trait DirectParams: FalconParams + ExactConvolution {
    /// `floor(sqrt(B))`: the bound on every `s1` and slack coefficient.
    const NORM_COEFFICIENT_ABS_BOUND: i64;
    /// The quotient bound after the ring lift.
    const QUOTIENT_ABS_BOUND: i64;
    /// Exclusive power-of-two carrier for `s1`, `s2`, and slack.
    const SMALL_WITNESS_BOUND: i64;
    /// Exclusive power-of-two carrier for the quotient `v`.
    const V_WITNESS_BOUND: i64;
    /// Exclusive power-of-two carrier for the negacyclic quotient `w`.
    const W_WITNESS_BOUND: i64 = 1 << 30;
}

impl DirectParams for Falcon512 {
    const NORM_COEFFICIENT_ABS_BOUND: i64 = FALCON_NORM_COEFFICIENT_ABS_BOUND;
    const QUOTIENT_ABS_BOUND: i64 = FALCON_QUOTIENT_ABS_BOUND;
    const SMALL_WITNESS_BOUND: i64 = S_CAP;
    const V_WITNESS_BOUND: i64 = 1 << 17;
}

impl DirectParams for Falcon1024 {
    const NORM_COEFFICIENT_ABS_BOUND: i64 = 8_382;
    const QUOTIENT_ABS_BOUND: i64 = 134_109;
    const SMALL_WITNESS_BOUND: i64 = S_CAP;
    const V_WITNESS_BOUND: i64 = 1 << 18;
    /// The ordinary-convolution wrap term is bounded by
    /// `ceil(6144 sqrt(n B))`, which is `1,648,054,080` at degree 1024
    /// and exceeds the Falcon-512 carrier.
    const W_WITNESS_BOUND: i64 = 1 << 31;
}

/// The dense-path wrap carriers hold the Cauchy-Schwarz bounds
/// `ceil(6144 sqrt(n B))`: `811,049,418` at degree 512 and
/// `1,648,054,080` at degree 1024. These omit the `s1` and target terms
/// included in the larger residual bounds from `specs/falcon.md` Section 9.3.
const _: () = assert!(<Falcon512 as DirectParams>::W_WITNESS_BOUND > 811_049_418);
const _: () = assert!(<Falcon1024 as DirectParams>::W_WITNESS_BOUND > 1_648_054_080);

const _: () = assert!(FALCON_512_COMPRESSED_HIGH_BITS_BUDGET == 392);
const _: () = assert!(Falcon512::BETA_SQUARED == FALCON_BETA_SQUARED);
const _: () = assert!(Falcon512::DEGREE == FALCON_DEGREE);
const _: () = assert!(
    <Falcon512 as DirectParams>::NORM_COEFFICIENT_ABS_BOUND
        < <Falcon512 as DirectParams>::SMALL_WITNESS_BOUND
);
const _: () = assert!(<Falcon1024 as DirectParams>::SMALL_WITNESS_BOUND == S_CAP);
const _: () = assert!(
    <Falcon1024 as DirectParams>::QUOTIENT_ABS_BOUND
        < <Falcon1024 as DirectParams>::V_WITNESS_BOUND
);
const _: () =
    assert!(8_382 * 8_382 <= Falcon1024::BETA_SQUARED && 8_383 * 8_383 > Falcon1024::BETA_SQUARED);

/// Supported degrees, for batches whose degree is recovered from their shape.
pub(crate) fn supported_degree(degree: usize) -> Result<usize, DirectError> {
    if degree == Falcon512::DEGREE || degree == Falcon1024::DEGREE {
        Ok(degree)
    } else {
        Err(DirectError::InvalidLength)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DirectError {
    InvalidLength,
    NormExceeded(u64),
    NormMismatch {
        expected: u64,
        actual: u64,
    },
    RelationFailure(usize),
    NonIntegralLift(usize),
    RangeViolation {
        family: &'static str,
        index: usize,
        value: i64,
    },
    EncodingBudgetExceeded(usize),
    IntegerOverflow,
    CappedFourSquaresUnavailable(u64),
}

impl core::fmt::Display for DirectError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidLength => write!(f, "Falcon polynomial has the wrong length"),
            Self::NormExceeded(value) => write!(f, "Falcon norm {value} exceeds the bound"),
            Self::NormMismatch { expected, actual } => {
                write!(f, "Falcon completed norm {actual} differs from {expected}")
            }
            Self::RelationFailure(index) => {
                write!(f, "Falcon relation fails at coefficient {index}")
            }
            Self::NonIntegralLift(index) => {
                write!(f, "Falcon lift is not divisible by q at {index}")
            }
            Self::RangeViolation {
                family,
                index,
                value,
            } => write!(
                f,
                "Falcon {family}[{index}]={value} is outside its declared witness range"
            ),
            Self::EncodingBudgetExceeded(high_sum) => write!(
                f,
                "Falcon padded signature needs {high_sum} unary high bits"
            ),
            Self::CappedFourSquaresUnavailable(value) => {
                write!(f, "no capped four-square decomposition for slack {value}")
            }
            Self::IntegerOverflow => write!(f, "integer overflow while constructing the lift"),
        }
    }
}

impl std::error::Error for DirectError {}

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
            negacyclic_from_ordinary(&P::convolution_i16(h.as_ref(), s2.as_ref()), degree);
        let mut t = filled::<P, i16>(0);
        for (slot, (&s1, &product)) in t.as_mut().iter_mut().zip(s1.as_ref().iter().zip(&product)) {
            *slot = center_mod_q(i64::from(s1) + product);
        }
        Self::new(h, t, s1, s2)
    }

    pub fn check_exact_lift(&self) -> Result<(), DirectError> {
        let product = P::convolution_i16(self.h.as_ref().as_ref(), self.s2.as_ref().as_ref());
        check_exact_lift::<P>(
            self.t.as_ref().as_ref(),
            self.s1.as_ref().as_ref(),
            self.v.as_ref().as_ref(),
            self.w.as_ref().as_ref(),
            self.slack,
            squared_norm(self.s1.as_ref().as_ref(), self.s2.as_ref().as_ref())?,
            &product,
        )
    }

    /// Host-check the mixed-radix ranges required by the no-wrap argument and
    /// the padded Falcon coefficient encoding budget.
    ///
    /// This does not replace proof-side range and encodability constraints.
    pub fn check_witness_ranges(&self) -> Result<(), DirectError> {
        check_witness_ranges::<P>(
            self.s1.as_ref().as_ref(),
            self.s2.as_ref().as_ref(),
            self.v.as_ref().as_ref(),
            self.w.as_ref().as_ref(),
            self.slack,
        )
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
        let product = P::convolution_i16(h, s2);
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

/// Dense direct-relation batch in `i64` cells. The degree is implicit:
/// `h.len() / padded_rows`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectBatch {
    pub live_rows: usize,
    pub padded_rows: usize,
    pub h: Vec<i64>,
    pub t: Vec<i64>,
    pub s1: Vec<i64>,
    pub s2: Vec<i64>,
    pub v: Vec<i64>,
    pub w: Vec<i64>,
    pub slack: Vec<i64>,
}

/// Public Falcon data needed by the algebraic relation verifier.
///
/// The public target `t` is deliberately absent. The verifier derives it with
/// `HashToPoint` from the proof-transported public salt, hashed public key, and
/// message, then evaluates it directly. The committed groups contain `S1`,
/// `Slack`, `S2Sources`, and `Budget`; no witness table is exposed here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectPublicBatch {
    pub live_rows: usize,
    pub padded_rows: usize,
    pub h: Vec<i64>,
}

impl DirectBatch {
    /// Allocate a zero Falcon-512 batch.
    #[cfg(test)]
    pub(crate) fn allocate(rows: usize) -> Self {
        Self::allocate_of::<Falcon512>(rows)
    }

    /// Allocate a zero batch for parameter set `P`.
    pub(crate) fn allocate_of<P: DirectParams>(rows: usize) -> Self {
        assert!(rows != 0, "a batch must contain at least one row");
        let padded_rows = rows.next_power_of_two();
        let len = padded_rows * P::DEGREE;
        Self {
            live_rows: rows,
            padded_rows,
            h: vec![0; len],
            t: vec![0; len],
            s1: vec![0; len],
            s2: vec![0; len],
            v: vec![0; len],
            w: vec![0; len],
            slack: vec![0; len],
        }
    }

    /// Ring degree recovered from the batch shape.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.h.len().checked_div(self.padded_rows).unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn fill_from_verified<P: DirectParams>(
        &mut self,
        verified: &[VerifiedFalconSignature<P>],
    ) -> Result<(), DirectError> {
        assert_eq!(verified.len(), self.live_rows, "verified row count changed");
        assert_eq!(self.degree(), P::DEGREE, "batch degree changed");
        let degree = P::DEGREE;
        #[cfg(feature = "parallel")]
        {
            let chunk_cells = DIRECT_CHUNK_ROWS * degree;
            self.h
                .par_chunks_mut(chunk_cells)
                .zip(self.t.par_chunks_mut(chunk_cells))
                .zip(self.s1.par_chunks_mut(chunk_cells))
                .zip(self.s2.par_chunks_mut(chunk_cells))
                .zip(self.v.par_chunks_mut(chunk_cells))
                .zip(self.w.par_chunks_mut(chunk_cells))
                .zip(self.slack.par_chunks_mut(chunk_cells))
                .zip(verified.par_chunks(DIRECT_CHUNK_ROWS))
                .enumerate()
                .try_for_each(
                    |(chunk_index, (((((((h, t), s1), s2), v), w), slack), verified))| {
                        tracing::info_span!(
                            "v1.prove.witness.direct.chunk",
                            chunk = chunk_index,
                            records = verified.len()
                        )
                        .in_scope(|| {
                            for (row_index, value) in verified.iter().enumerate() {
                                let start = row_index * degree;
                                let end = start + degree;
                                write_verified_row(
                                    &mut h[start..end],
                                    &mut t[start..end],
                                    &mut s1[start..end],
                                    &mut s2[start..end],
                                    &mut v[start..end],
                                    &mut w[start..end],
                                    &mut slack[start..end],
                                    value,
                                )?;
                            }
                            Ok::<(), DirectError>(())
                        })
                    },
                )?;
        }
        #[cfg(not(feature = "parallel"))]
        for (row_index, value) in verified.iter().enumerate() {
            let start = row_index * degree;
            let end = start + degree;
            write_verified_row(
                &mut self.h[start..end],
                &mut self.t[start..end],
                &mut self.s1[start..end],
                &mut self.s2[start..end],
                &mut self.v[start..end],
                &mut self.w[start..end],
                &mut self.slack[start..end],
                value,
            )?;
        }
        Ok(())
    }

    #[must_use]
    pub fn from_rows<P: DirectParams>(rows: &[DirectRow<P>]) -> Self {
        let degree = P::DEGREE;
        let mut batch = Self::allocate_of::<P>(rows.len());
        #[cfg(feature = "parallel")]
        {
            let widen = |destination: &mut [i64], source: fn(&DirectRow<P>) -> &[i16]| {
                destination
                    .par_chunks_mut(degree)
                    .zip(rows.par_iter())
                    .for_each(|(destination, row)| {
                        for (destination, &source) in destination.iter_mut().zip(source(row)) {
                            *destination = i64::from(source);
                        }
                    });
            };
            widen(&mut batch.h, |row| row.h.as_ref().as_ref());
            widen(&mut batch.t, |row| row.t.as_ref().as_ref());
            widen(&mut batch.s1, |row| row.s1.as_ref().as_ref());
            widen(&mut batch.s2, |row| row.s2.as_ref().as_ref());
            batch
                .v
                .par_chunks_mut(degree)
                .zip(rows.par_iter())
                .for_each(|(destination, row)| {
                    destination.copy_from_slice(row.v.as_ref().as_ref());
                });
            batch
                .w
                .par_chunks_mut(degree)
                .zip(rows.par_iter())
                .for_each(|(destination, row)| {
                    destination.copy_from_slice(row.w.as_ref().as_ref());
                });
            batch
                .slack
                .par_chunks_mut(degree)
                .zip(rows.par_iter())
                .for_each(|(destination, row)| {
                    for (destination, source) in destination.iter_mut().zip(row.slack) {
                        *destination = i64::from(source);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        for (i, row) in rows.iter().enumerate() {
            let range = i * degree..(i + 1) * degree;
            for (dst, src) in batch.h[range.clone()]
                .iter_mut()
                .zip(row.h.as_ref().as_ref())
            {
                *dst = i64::from(*src);
            }
            for (dst, src) in batch.t[range.clone()]
                .iter_mut()
                .zip(row.t.as_ref().as_ref())
            {
                *dst = i64::from(*src);
            }
            for (dst, src) in batch.s1[range.clone()]
                .iter_mut()
                .zip(row.s1.as_ref().as_ref())
            {
                *dst = i64::from(*src);
            }
            for (dst, src) in batch.s2[range.clone()]
                .iter_mut()
                .zip(row.s2.as_ref().as_ref())
            {
                *dst = i64::from(*src);
            }
            batch.v[range.clone()].copy_from_slice(row.v.as_ref().as_ref());
            batch.w[range.clone()].copy_from_slice(row.w.as_ref().as_ref());
            batch.slack[range.start..range.start + 4]
                .iter_mut()
                .zip(row.slack)
                .for_each(|(dst, src)| *dst = i64::from(src));
        }
        batch
    }

    #[must_use]
    pub fn num_vars(&self) -> usize {
        self.s1.len().trailing_zeros() as usize
    }

    #[must_use]
    pub fn public_batch(&self) -> DirectPublicBatch {
        DirectPublicBatch {
            live_rows: self.live_rows,
            padded_rows: self.padded_rows,
            h: self.h.clone(),
        }
    }

    /// Copy one contiguous live-row interval into its own structurally padded
    /// batch. Selector arguments use this to stay within Aerie's qualified
    /// per-commitment profile while the dense relation remains global.
    pub fn row_slice(&self, start: usize, count: usize) -> Result<Self, DirectError> {
        let degree = self.degree();
        let end = start
            .checked_add(count)
            .ok_or(DirectError::IntegerOverflow)?;
        if count == 0 || end > self.live_rows {
            return Err(DirectError::InvalidLength);
        }
        let padded_rows = count.next_power_of_two();
        let source_start = start * degree;
        let source_end = end * degree;
        let length = padded_rows * degree;
        let copy = |source: &[i64]| {
            let mut output = vec![0_i64; length];
            output[..count * degree].copy_from_slice(&source[source_start..source_end]);
            output
        };
        Ok(Self {
            live_rows: count,
            padded_rows,
            h: copy(&self.h),
            t: copy(&self.t),
            s1: copy(&self.s1),
            s2: copy(&self.s2),
            v: copy(&self.v),
            w: copy(&self.w),
            slack: copy(&self.slack),
        })
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn write_verified_row<P: DirectParams>(
    h: &mut [i64],
    t: &mut [i64],
    s1: &mut [i64],
    s2: &mut [i64],
    v: &mut [i64],
    w: &mut [i64],
    slack: &mut [i64],
    value: &VerifiedFalconSignature<P>,
) -> Result<(), DirectError> {
    let auxiliaries = DirectAuxiliaries::derive::<P>(
        value.h_centered().as_ref(),
        value.t_centered().as_ref(),
        value.s1().as_ref(),
        value.s2().as_ref(),
        Some(value.norm_squared()),
    )?;
    for (destination, &source) in h.iter_mut().zip(value.h_centered().as_ref()) {
        *destination = i64::from(source);
    }
    for (destination, &source) in t.iter_mut().zip(value.t_centered().as_ref()) {
        *destination = i64::from(source);
    }
    for (destination, &source) in s1.iter_mut().zip(value.s1().as_ref()) {
        *destination = i64::from(source);
    }
    for (destination, &source) in s2.iter_mut().zip(value.s2().as_ref()) {
        *destination = i64::from(source);
    }
    v.copy_from_slice(&auxiliaries.v);
    w.copy_from_slice(&auxiliaries.w);
    for (destination, source) in slack.iter_mut().zip(auxiliaries.slack) {
        *destination = i64::from(source);
    }
    Ok(())
}

fn squared_norm(s1: &[i16], s2: &[i16]) -> Result<u64, DirectError> {
    s1.iter().chain(s2).try_fold(0_u64, |sum, &x| {
        sum.checked_add(i64::from(x).unsigned_abs().pow(2))
            .ok_or(DirectError::IntegerOverflow)
    })
}

fn negacyclic_from_ordinary(product: &[i64], degree: usize) -> Vec<i64> {
    (0..degree)
        .map(|j| product[j] - product.get(j + degree).copied().unwrap_or(0))
        .collect()
}

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
            row.check_exact_lift().unwrap();
            row.check_witness_ranges().unwrap();
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
        row.check_exact_lift().unwrap();
        row.check_witness_ranges().unwrap();
        let wide = DirectRow::<Falcon1024>::synthetic(7).unwrap();
        wide.check_exact_lift().unwrap();
        wide.check_witness_ranges().unwrap();
        assert_eq!(wide.h.as_ref().len(), 1024);
    }

    #[test]
    fn batch_padding_is_structural_zero() {
        let rows = (0..3)
            .map(|i| DirectRow::<Falcon512>::synthetic(i).unwrap())
            .collect::<Vec<_>>();
        let batch = DirectBatch::from_rows(&rows);
        assert_eq!(batch.padded_rows, 4);
        assert_eq!(batch.degree(), 512);
        assert!(batch.s1[3 * FALCON_DEGREE..].iter().all(|&x| x == 0));
        assert_eq!(batch.num_vars(), 11);

        let wide_rows = (0..3)
            .map(|i| DirectRow::<Falcon1024>::synthetic(i).unwrap())
            .collect::<Vec<_>>();
        let wide = DirectBatch::from_rows(&wide_rows);
        assert_eq!(wide.degree(), 1024);
        assert_eq!(wide.num_vars(), 12);
        assert!(wide.s1[3 * 1024..].iter().all(|&x| x == 0));
    }

    #[test]
    fn row_slice_rebases_and_pads_contiguous_rows() {
        let rows = (0..5)
            .map(|i| DirectRow::<Falcon512>::synthetic(i).unwrap())
            .collect::<Vec<_>>();
        let batch = DirectBatch::from_rows(&rows);
        let tail = batch.row_slice(3, 2).unwrap();
        assert_eq!(tail.live_rows, 2);
        assert_eq!(tail.padded_rows, 2);
        assert_eq!(
            tail.s1[..FALCON_DEGREE],
            batch.s1[3 * FALCON_DEGREE..4 * FALCON_DEGREE]
        );
        assert_eq!(
            tail.s1[FALCON_DEGREE..],
            batch.s1[4 * FALCON_DEGREE..5 * FALCON_DEGREE]
        );
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
