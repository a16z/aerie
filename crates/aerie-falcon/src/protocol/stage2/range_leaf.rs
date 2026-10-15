//! Canonical range-leaf instance order and division-free normalized recurrence.
use crate::protocol::relations::quartic::QuarticPolynomial;
use crate::{protocol::FalconDegree, substrate::OpeningField as E};
use jolt_field::Zero;
#[must_use]
pub fn instances(degree: FalconDegree) -> Vec<(usize, QuarticPolynomial)> {
    use QuarticPolynomial::{High, Low};
    let mut out = vec![(1, Low), (1, High), (2, Low), (2, High)];
    if degree == FalconDegree::Falcon1024 {
        out.extend([(5, Low), (5, High)]);
    }
    out.extend([(0, Low), (3, Low), (4, Low)]);
    if degree == FalconDegree::Falcon512 {
        out.push((5, Low));
    }
    out
}
#[inline]
#[must_use]
pub fn advance(claim: E, alpha: E, c: [E; 4], x: E) -> E {
    let c0 = claim - alpha * c.iter().copied().fold(E::zero(), |s, v| s + v);
    (((c[3] * x + c[2]) * x + c[1]) * x + c[0]) * x + c0
}
#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{One, Ring};
    #[test]
    fn range_leaf_instances_follow_the_canonical_gamma_order() {
        use QuarticPolynomial::{High, Low};
        assert_eq!(
            instances(FalconDegree::Falcon512),
            vec![
                (1, Low),
                (1, High),
                (2, Low),
                (2, High),
                (0, Low),
                (3, Low),
                (4, Low),
                (5, Low),
            ]
        );
        assert_eq!(
            instances(FalconDegree::Falcon1024),
            vec![
                (1, Low),
                (1, High),
                (2, Low),
                (2, High),
                (5, Low),
                (5, High),
                (0, Low),
                (3, Low),
                (4, Low),
            ]
        );
    }

    #[test]
    fn normalized_recurrence_accepts_equality_endpoints() {
        let c = [2, 3, 5, 7].map(E::from_u64);
        for alpha in [E::zero(), E::one(), E::from_u64(19)] {
            let c0 = E::from_u64(11);
            let claim = c0 + alpha * c.iter().copied().fold(E::zero(), |s, v| s + v);
            for x in [E::zero(), E::one(), E::from_u64(31)] {
                assert_eq!(
                    advance(claim, alpha, c, x),
                    (((c[3] * x + c[2]) * x + c[1]) * x + c[0]) * x + c0
                );
            }
        }
    }
}
