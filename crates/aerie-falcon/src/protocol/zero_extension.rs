//! Zero extension of each logical polynomial to the PCS catalog floor.
//!
//! Small groups commit the logical table followed by zeros (Section 5).
//! Under low-bit-first indexing, the extra pad variables are trailing
//! coordinates. Their multilinear extension satisfies
//! `P_phys~(r, r_pad) = eq(r_pad, 0) P_logical~(r)`.
//! Production openings fix those coordinates to zero, so the physical
//! opening equals the logical claim without another random-point check.

/// Log2 of the current catalog floor for a standalone group commitment.
pub const PCS_MIN_CELLS_LOG2: usize = 13;

/// `pcslen(L) = 2^max(13, log2 L)` for power-of-two `L`.
#[must_use]
pub fn pcs_length(logical_cells: usize) -> usize {
    debug_assert!(logical_cells.is_power_of_two());
    logical_cells.max(1 << PCS_MIN_CELLS_LOG2)
}

/// Number of PCS pad variables a logical group needs at the floor.
#[must_use]
pub fn pad_vars(logical_cells: usize) -> usize {
    (pcs_length(logical_cells) / logical_cells).trailing_zeros() as usize
}

use crate::substrate::OpeningField;
use jolt_field::Zero;

/// The physical opening point of one ledger claim: the logical point with
/// all-zero trailing pad coordinates, making the zero-extension tie exact.
#[must_use]
pub fn physical_point(logical_point: &[OpeningField], logical_cells: usize) -> Vec<OpeningField> {
    let mut point = logical_point.to_vec();
    point.extend(std::iter::repeat_n(
        OpeningField::zero(),
        pad_vars(logical_cells),
    ));
    point
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::algebra::{evaluate_mle, lift_table, sample_point};
    use crate::protocol::manifest::CommitmentManifest;
    use crate::protocol::transcript::TestTranscript;
    use crate::substrate::OpeningField as E;
    use jolt_field::{One, Zero};

    #[test]
    fn pcs_lengths_apply_the_floor_exactly() {
        assert_eq!(pcs_length(4), 1 << 13);
        assert_eq!(pad_vars(4), 11);
        assert_eq!(pcs_length(1 << 13), 1 << 13);
        assert_eq!(pad_vars(1 << 13), 0);
        assert_eq!(pcs_length(1 << 14), 1 << 14);
        assert_eq!(pad_vars(1 << 14), 0);
    }

    #[test]
    fn the_tie_identity_matches_the_zero_extended_mle() {
        // A group-like logical table below the floor: 32 cells (the
        // one polynomial in the three-polynomial `Budget` group for a 32-record batch).
        let logical: Vec<i64> = (0..32).map(|cell| cell * 7 - 100).collect();
        let mut physical = logical.clone();
        physical.resize(pcs_length(logical.len()), 0);
        assert_eq!(&physical[..logical.len()], logical.as_slice());
        assert_eq!(physical.len(), 1 << 13);
        assert!(physical[32..].iter().all(|&cell| cell == 0));

        let mut transcript = TestTranscript::new(b"zero extension tie test");
        let logical_point = sample_point(&mut transcript, "logical", 5).unwrap();
        let pad_point = sample_point(&mut transcript, "pad", 8).unwrap();
        let mut physical_point = logical_point.clone();
        physical_point.extend_from_slice(&pad_point);

        let logical_value = evaluate_mle(&lift_table(&logical), &logical_point);
        let physical_value = evaluate_mle(&lift_table(&physical), &physical_point);
        assert_eq!(
            physical_value,
            pad_point.iter().fold(logical_value, |value, &coordinate| {
                value * (E::one() - coordinate)
            }),
            "zero extension with trailing pad coordinates"
        );
        // At all-zero trailing coordinates, the physical opening equals
        // the logical claim.
        let zero_pad = vec![E::zero(); 8];
        let mut anchored = logical_point.clone();
        anchored.extend_from_slice(&zero_pad);
        assert_eq!(
            evaluate_mle(&lift_table(&physical), &anchored),
            logical_value
        );
    }

    #[test]
    fn every_manifest_group_has_a_defined_physical_length() {
        for records in [1_usize, 2, 3, 8, 1 << 14, crate::protocol::MAX_RECORDS] {
            let manifest = CommitmentManifest::for_batch(records).expect("manifest");
            for group in manifest.groups() {
                let physical = pcs_length(group.logical_cells);
                assert!(physical >= group.logical_cells);
                assert!(physical.is_power_of_two());
                assert_eq!(
                    pad_vars(group.logical_cells),
                    physical.trailing_zeros() as usize
                        - group.logical_cells.trailing_zeros() as usize
                );
            }
            if records >= 1 << 14 {
                // At production-scale batches every group clears the floor.
                for group in manifest.groups() {
                    assert_eq!(pad_vars(group.logical_cells), 0);
                }
            }
        }
    }
}
