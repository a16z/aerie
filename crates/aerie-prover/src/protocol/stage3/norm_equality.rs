//! Magnitude materialization and terminal exact squared-norm identity.

use crate::protocol::source_tables::SourceTables;

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
#[cfg(feature = "internals")]
pub use aerie_falcon::protocol::stage3::norm_equality::terminal;

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
