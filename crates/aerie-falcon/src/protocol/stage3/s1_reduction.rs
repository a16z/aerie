//! Terminal opening for the stage-one `S_1` claim.

use crate::substrate::OpeningField as E;

#[must_use]
pub fn terminal(gamma: E, projection_point: &[E], point: &[E], terminal_evaluations: &[E; 9]) -> E {
    gamma
        * crate::protocol::algebra::eq_evaluation(projection_point, point)
        * terminal_evaluations[0]
}
