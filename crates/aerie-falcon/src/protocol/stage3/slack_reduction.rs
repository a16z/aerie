//! Terminal opening for the native stage-three slack relation.

use crate::protocol::FalconDegree;
use crate::substrate::OpeningField as E;

#[must_use]
pub fn terminal(
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
