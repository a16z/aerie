//! Terminal sign-check identity used by the fused stage-three prover.

use crate::substrate::OpeningField as E;
use jolt_field::One;

pub(super) fn terminal(gamma: E, r2: &[E], r3: &[E], sigma: E) -> E {
    gamma
        * crate::protocol::algebra::eq_evaluation(r2, r3)
        * (sigma - E::one())
        * (sigma + E::one())
}

#[cfg(test)]
mod tests {
    use crate::protocol::{stage3::fused_rounds::tests::check_relations, FalconDegree};

    #[test]
    fn direct_rounds_match_the_native_relation_at_equality_endpoints() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            for mode in 0..4 {
                check_relations(&[2], degree, 4, mode);
            }
        }
    }
}
