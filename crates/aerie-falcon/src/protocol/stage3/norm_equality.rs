use crate::{protocol::FalconDegree, substrate::OpeningField as E};
use jolt_field::Ring;

#[must_use]
pub fn terminal(
    gamma: E,
    degree: FalconDegree,
    stage2_point: &[E],
    terminal_evaluations: &[E; 9],
    point: &[E],
) -> E {
    let coefficient_vars = degree.logn();
    let row_equality = crate::protocol::algebra::eq_evaluation(
        &stage2_point[coefficient_vars..],
        &point[coefficient_vars..],
    );
    let magnitude = terminal_evaluations[2]
        + E::from_i64(8) * terminal_evaluations[3]
        + E::from_i64(128) * terminal_evaluations[4]
        + E::from_i64(1_092);
    let coefficient_terms =
        row_equality * (terminal_evaluations[0] * terminal_evaluations[0] + magnitude * magnitude);
    let slack_term = row_equality * terminal_evaluations[1] * terminal_evaluations[1];
    gamma * (coefficient_terms + super::inverse_power_of_two(coefficient_vars - 2) * slack_term)
}
