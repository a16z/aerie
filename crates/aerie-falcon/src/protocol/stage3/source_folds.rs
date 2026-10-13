//! Direct checks for the shared source lanes folded by the fused stage engine.

#[cfg(test)]
mod tests {
    use crate::protocol::{stage3::fused_rounds::tests::check_relations, FalconDegree};

    #[test]
    fn shared_compact_lanes_match_direct_affine_folds() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            for mode in [0, 1] {
                check_relations(&[2, 3, 7, 8], degree, 4, mode);
            }
        }
    }
}
