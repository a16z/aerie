//! Shared Falcon parameter metadata and direct-relation errors.

use aerie_falcon_native::{params::S_CAP, Falcon1024, Falcon512, FalconParams};

/// Host-side integer ranges of the direct witness for one parameter set.
///
/// The norm and quotient bounds are the values the Falcon protocol proves
/// after its two integer lifts (`specs/falcon.md` Section 9).
/// The carrier bounds are exclusive
/// powers of two that contain them.
pub trait DirectParams: FalconParams {
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
    const NORM_COEFFICIENT_ABS_BOUND: i64 = 5_833;
    const QUOTIENT_ABS_BOUND: i64 = 65_998;
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
