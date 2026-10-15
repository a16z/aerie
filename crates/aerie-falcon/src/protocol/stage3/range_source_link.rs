use super::inverse_power_of_two;
use crate::protocol::{algebra::eq_evaluation, stage2::Stage2Output, FalconDegree};
use crate::substrate::OpeningField as E;
use jolt_field::{One, Zero};

#[must_use]
pub fn terminal(
    gammas: &[E; 6],
    stage2: &Stage2Output,
    terminal_evaluations: &[E; 9],
    point: &[E],
    degree: FalconDegree,
) -> E {
    let coefficient_vars = degree.logn();
    let coefficient_equality = eq_evaluation(&stage2.point, point);
    let mut expected = E::zero();
    for index in 0..3 {
        let value = terminal_evaluations[2 + index];
        expected += gammas[index] * coefficient_equality * value * (value + E::one());
    }

    let row_equality = eq_evaluation(
        &stage2.point[coefficient_vars..],
        &point[coefficient_vars..],
    );
    let inactive_scale = inverse_power_of_two(coefficient_vars);
    for index in 0..3 {
        let value = terminal_evaluations[6 + index];
        expected += gammas[index + 3] * inactive_scale * row_equality * value * (value + E::one());
    }
    expected
}
