//! Terminal opening for the stage-one `S_1` claim.

use crate::substrate::OpeningField as E;

pub(super) fn terminal(
    gamma: E,
    projection_point: &[E],
    point: &[E],
    terminal_evaluations: &[E; 9],
) -> E {
    gamma
        * crate::protocol::algebra::eq_evaluation(projection_point, point)
        * terminal_evaluations[0]
}

#[cfg(test)]
mod tests {
    use crate::protocol::{stage3::fused_rounds::tests::check_relations, FalconDegree};

    #[test]
    fn direct_rounds_match_the_native_relation_at_equality_endpoints() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            for mode in 0..4 {
                check_relations(&[8], degree, 4, mode);
            }
        }
    }
}
