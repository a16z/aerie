//! Verifier predicates on the projection outputs (`specs/falcon.md`
//! Section 6.1). Both act on centered integers and are never evaluated in
//! `F_p`.

use crate::substrate::BaseField as F;

pub use crate::substrate::FIELD_MODULUS;

/// `(p - 1) / 2`, the largest nonnegative centered representative.
const HALF_MODULUS: u64 = (FIELD_MODULUS - 1) / 2;

/// `ctr_p`: the representative of `x` in `[-(p - 1)/2, (p - 1)/2]`.
#[must_use]
pub fn centered(x: F) -> i128 {
    let canonical = x.to_canonical_u64();
    if canonical <= HALF_MODULUS {
        i128::from(canonical)
    } else {
        i128::from(canonical) - i128::from(FIELD_MODULUS)
    }
}

/// Norm predicate `sum_j ctr_p(y_j)^2 <= cap`.
///
/// Each square is below `2^126` and the running sum stops as soon as it
/// exceeds `cap < 2^127`, so the `u128` accumulator never overflows.
#[must_use]
pub fn norm_predicate(y: &[F], cap: u128) -> bool {
    let mut sum: u128 = 0;
    for &value in y {
        let magnitude = centered(value).unsigned_abs();
        sum += magnitude * magnitude;
        if sum > cap {
            return false;
        }
    }
    true
}

/// Residual predicate `q | ctr_p(z_j)` for every `j`, as integers.
#[must_use]
pub fn residual_predicate(z: &[F], modulus: u64) -> bool {
    z.iter()
        .all(|&value| centered(value) % i128::from(modulus) == 0)
}
