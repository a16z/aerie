//! Falcon producer execution over the shared protocol identities.
use crate::UnsupportedBatch;
#[cfg(feature = "internals")]
pub use aerie_falcon::protocol::padded_records;
#[cfg(any(test, feature = "internals"))]
pub use aerie_falcon::protocol::public_inputs;
#[cfg(test)]
pub use aerie_falcon::protocol::MIN_PADDED_COEFFICIENTS;
#[cfg(any(test, feature = "parallel"))]
pub use aerie_falcon::protocol::PARALLEL_MIN_WORK;
pub use aerie_falcon::protocol::{
    instance, manifest, openings, public_records_binding, stage, wire, zero_extension,
};
pub use aerie_falcon::protocol::{join_if, AggregationParams, FalconDegree};
#[cfg(feature = "internals")]
pub use aerie_falcon::protocol::{CommitmentCellCounts, MAX_RECORDS};
pub use aerie_falcon_witness::CompactDirectBatch as FalconWitness;
pub mod algebra;
pub(crate) mod kernels;
pub mod pcs;
pub mod projection;
pub mod ring;
pub mod source_tables;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod transcript;
pub mod transport;
/// Accept generic statement storage or the protocol's minimum-padded storage.
/// Check all lengths before extending or indexing any witness plane.
pub(crate) fn validate_witness_shape(batch: &FalconWitness) -> Result<(), UnsupportedBatch> {
    let degree = FalconDegree::from_degree(batch.degree())?;
    let padded = degree.padded_records(batch.live_rows)?;
    if batch.padded_rows != batch.live_rows.next_power_of_two() && batch.padded_rows != padded {
        return Err(UnsupportedBatch::Shape {
            degree: batch.degree(),
            records: batch.padded_rows,
        });
    }
    let cells = batch.padded_rows * degree.degree();
    if [
        batch.h.len(),
        batch.t.len(),
        batch.s1.len(),
        batch.s2.len(),
        batch.v.len(),
    ]
    .iter()
    .any(|&len| len != cells)
        || batch.slack.len() != 4 * batch.padded_rows
    {
        return Err(UnsupportedBatch::Shape {
            degree: batch.degree(),
            records: batch.padded_rows,
        });
    }
    Ok(())
}

/// Extend generic statement storage to the protocol domain. Existing dead rows
/// and the new suffix remain governed by the same live-row selector.
pub fn pad_witness(batch: &mut FalconWitness) -> Result<(), UnsupportedBatch> {
    let degree = FalconDegree::from_degree(batch.degree())?;
    let padded = degree.padded_records(batch.live_rows)?;
    validate_witness_shape(batch)?;
    let cells = padded * degree.degree();
    batch.h.resize(cells, 0);
    batch.t.resize(cells, 0);
    batch.s1.resize(cells, 0);
    batch.s2.resize(cells, 0);
    batch.v.resize(cells, 0);
    batch.slack.resize(4 * padded, 0);
    batch.padded_rows = padded;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerie_falcon_native::{Falcon512, FalconParams};
    use aerie_falcon_witness::DirectRow;

    #[test]
    fn compact_witness_contains_the_defined_row_values_and_padding() {
        let rows = (0..3)
            .map(|seed| DirectRow::<Falcon512>::synthetic(seed + 701).expect("synthetic row"))
            .collect::<Vec<_>>();
        let compact = FalconWitness::from_rows(&rows);
        for (index, row) in rows.iter().enumerate() {
            let start = index * Falcon512::DEGREE;
            let end = start + Falcon512::DEGREE;
            assert_eq!(&compact.h[start..end], row.h.as_ref().as_slice());
            assert_eq!(
                compact.v[start..end]
                    .iter()
                    .map(|&value| i64::from(value))
                    .collect::<Vec<_>>(),
                row.v.as_ref().as_slice()
            );
            assert_eq!(compact.slack[4 * index..4 * index + 4], row.slack);
        }
        assert_eq!(compact.slack.len(), 4 * compact.padded_rows);
        assert_eq!(
            compact.retained_coefficient_bytes(),
            (4 * Falcon512::DEGREE + 4) * compact.padded_rows * size_of::<i16>()
                + Falcon512::DEGREE * compact.padded_rows * size_of::<i32>()
        );
        assert!(compact.h[3 * Falcon512::DEGREE..]
            .iter()
            .all(|&cell| cell == 0));
        assert!(compact.v[3 * Falcon512::DEGREE..]
            .iter()
            .all(|&cell| cell == 0));
        assert!(compact.slack[12..].iter().all(|&cell| cell == 0));
    }
}
