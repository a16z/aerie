//! Terminal identities and fixed input claim for the compact encoding budget.

use crate::protocol::FalconDegree;
use crate::substrate::OpeningField as E;
use jolt_field::Ring;

pub(super) fn input_claim(degree: FalconDegree) -> E {
    E::from_i64(degree.compressed_high_bits_budget() as i64 - i64::from(degree.budget_constant()))
}

pub(super) fn terminal(
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
    let coefficient_term = row_equality * (terminal_evaluations[4] + E::from_i64(8));
    let signature_budget = terminal_evaluations[6]
        + E::from_i64(8) * terminal_evaluations[7]
        + E::from_i64(64) * terminal_evaluations[8];
    gamma
        * (coefficient_term
            + super::inverse_power_of_two(coefficient_vars) * row_equality * signature_budget)
}

#[cfg(test)]
mod tests {
    use crate::protocol::{stage3::fused_rounds::tests::check_relations, FalconDegree};

    #[test]
    fn coefficient_and_budget_rounds_match_direct_tables_with_eq_endpoints() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            for mode in [0, 1, 4, 5] {
                check_relations(&[5, 6], degree, 4, mode);
            }
        }
    }
}
