//! Exact scalar products in the Falcon opening field.

use crate::substrate::OpeningField as E;
use jolt_field::Prime64Offset59 as F;

// For p = 2^64 - 59, 2^128 equals 3481 modulo p. The input is the low
// 128 bits plus at most two carries from summing three products. The first
// fold is at most 60*2^64 + 6902; the second is below 2^64 + 3540 < 2p.
// `solinas_reduce` then returns the canonical representative.
#[expect(
    clippy::inline_always,
    reason = "Keep the measured reduction in the opening-field product kernel"
)]
#[inline(always)]
fn reduce130(low: u128, carry: u64) -> F {
    debug_assert!(carry <= 2);
    let folded = (low >> 64) * 59 + u128::from(low as u64) + u128::from(carry) * 3481;
    let settled = (folded >> 64) * 59 + u128::from(folded as u64);
    F::solinas_reduce(settled)
}

#[expect(
    clippy::inline_always,
    reason = "Keep the measured product kernel in the sumcheck inner loop"
)]
#[inline(always)]
pub(crate) fn mul(lhs: E, rhs: E) -> E {
    let a = u128::from(lhs.c0().to_canonical_u64());
    let b = u128::from(lhs.c1().to_canonical_u64());
    let c = u128::from(rhs.c0().to_canonical_u64());
    let d = u128::from(rhs.c1().to_canonical_u64());
    let ac = a * c;
    let bd = b * d;
    let ad = a * d;
    let bc = b * c;
    let (twice_bd, carry0) = bd.overflowing_add(bd);
    let (real, carry1) = ac.overflowing_add(twice_bd);
    let (imaginary, carry2) = ad.overflowing_add(bc);
    E::new(
        reduce130(real, u64::from(carry0) + u64::from(carry1)),
        reduce130(imaginary, u64::from(carry2)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Ring;

    const P: u64 = u64::MAX - 58;

    #[test]
    fn accumulated_products_match_native_akita_boundaries_and_full_field() {
        let boundary = [0, 1, 2, 58, 59, 60, P / 2, P - 2, P - 1];
        for a in boundary {
            for b in boundary {
                for c in boundary {
                    for d in boundary {
                        let lhs = E::new(F::from_u64(a), F::from_u64(b));
                        let rhs = E::new(F::from_u64(c), F::from_u64(d));
                        assert_eq!(mul(lhs, rhs), lhs * rhs);
                    }
                }
            }
        }
        let mut state = 0x0123_4567_89ab_cdef_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            F::from_u64(state)
        };
        for _ in 0..100_000 {
            let a = E::new(next(), next());
            let b = E::new(next(), next());
            assert_eq!(mul(a, b), a * b);
            assert_eq!(mul(a, a), a * a);
        }
    }
}
