//! Magnitude materialization and terminal exact squared-norm identity.

use crate::protocol::source_tables::SourceTables;
use crate::protocol::FalconDegree;
use crate::substrate::OpeningField as E;
use jolt_field::Ring;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
const PARALLEL_COEFFICIENT_PAIRS: usize = 1 << 14;

/// Build the compact signed affine source shared by the norm and ring
/// relations. Its alphabet is exactly `[0, 2047]`, so `i16` retains every value.
pub(super) fn magnitude_table(tables: &SourceTables) -> Vec<i16> {
    let a0 = tables.a0_view();
    let a1 = tables.a1_view();
    let a2 = tables.a2_view();
    let magnitude = |index: usize| {
        i16::try_from(
            i64::from(a0.get(index))
                + 8 * i64::from(a1.get(index))
                + 128 * i64::from(a2.get(index))
                + 1_092,
        )
        .expect("Falcon magnitude is in [0, 2047]")
    };
    #[cfg(feature = "parallel")]
    if a0.len() >= PARALLEL_COEFFICIENT_PAIRS {
        return (0..a0.len()).into_par_iter().map(magnitude).collect();
    }
    (0..a0.len()).map(magnitude).collect()
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
    let magnitude = terminal_evaluations[2]
        + E::from_i64(8) * terminal_evaluations[3]
        + E::from_i64(128) * terminal_evaluations[4]
        + E::from_i64(1_092);
    let coefficient_terms =
        row_equality * (terminal_evaluations[0] * terminal_evaluations[0] + magnitude * magnitude);
    let slack_term = row_equality * terminal_evaluations[1] * terminal_evaluations[1];
    gamma * (coefficient_terms + super::inverse_power_of_two(coefficient_vars - 2) * slack_term)
}

#[cfg(test)]
mod tests {
    use crate::protocol::{stage3::fused_rounds::tests::check_relations, FalconDegree};

    #[test]
    fn coefficient_norm_rounds_match_direct_tables_at_eq_endpoints_and_random_points() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            for mode in [0, 1, 3] {
                check_relations(&[3], degree, 4, mode);
            }
        }
    }

    #[test]
    fn slack_norm_rounds_match_direct_tables_through_inactive_and_slot_variables() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            for mode in [0, 1, 3] {
                check_relations(&[4], degree, 4, mode);
            }
        }
    }
}
