//! Terminal opening for the native stage-three slack relation.

use crate::protocol::FalconDegree;
use crate::substrate::OpeningField as E;

pub(super) fn terminal(
    gamma: E,
    degree: FalconDegree,
    projection_point: &[E],
    point: &[E],
    terminal_evaluations: &[E; 9],
) -> E {
    let coefficient_vars = degree.logn();
    let slack_point = &point[coefficient_vars - 2..];
    gamma
        * super::inverse_power_of_two(coefficient_vars - 2)
        * crate::protocol::algebra::eq_evaluation(projection_point, slack_point)
        * terminal_evaluations[1]
}

#[cfg(test)]
mod tests {
    use crate::protocol::{stage3::fused_rounds::tests::check_relations, FalconDegree};

    #[test]
    fn direct_rounds_match_the_native_relation_at_equality_endpoints() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            for mode in 0..4 {
                check_relations(&[9], degree, 4, mode);
            }
        }
    }
}
