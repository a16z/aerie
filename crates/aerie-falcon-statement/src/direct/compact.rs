use core::mem::size_of;

use aerie_falcon_native::witness::VerifiedFalconSignature;
#[cfg(test)]
use aerie_falcon_native::Falcon512;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use super::DIRECT_CHUNK_ROWS;
use super::{
    four_squares, supported_degree, DirectBatch, DirectError, DirectParams, DirectRow, FALCON_Q,
};

/// Dense direct-relation data retained by protocols that do not prove the
/// validation-only `w` plane.
///
/// Slack is stored at its semantic width of four cells per record. The five
/// other vectors keep the coefficient-major width of one ring degree per
/// record; the degree is implicit as `h.len() / padded_rows` and is either
/// 512 or 1,024. Falcon-native planes use `i16`; the quotient uses `i32`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactDirectBatch {
    pub live_rows: usize,
    pub padded_rows: usize,
    pub h: Vec<i16>,
    pub t: Vec<i16>,
    pub s1: Vec<i16>,
    pub s2: Vec<i16>,
    pub v: Vec<i32>,
    pub slack: Vec<i16>,
}

impl CompactDirectBatch {
    /// Allocate a zero Falcon-512 batch.
    #[cfg(test)]
    pub(crate) fn allocate(rows: usize) -> Self {
        Self::allocate_of::<Falcon512>(rows)
    }

    /// Allocate a zero batch for parameter set `P`.
    pub(crate) fn allocate_of<P: DirectParams>(rows: usize) -> Self {
        assert!(rows != 0, "a batch must contain at least one row");
        let padded_rows = rows.next_power_of_two();
        let coefficient_cells = padded_rows * P::DEGREE;
        Self {
            live_rows: rows,
            padded_rows,
            h: vec![0; coefficient_cells],
            t: vec![0; coefficient_cells],
            s1: vec![0; coefficient_cells],
            s2: vec![0; coefficient_cells],
            v: vec![0; coefficient_cells],
            slack: vec![0; 4 * padded_rows],
        }
    }

    /// Ring degree recovered from the batch shape (512 or 1,024 for a valid
    /// batch; use [`Self::validate_shape`] to reject anything else).
    #[must_use]
    pub fn degree(&self) -> usize {
        self.h.len().checked_div(self.padded_rows).unwrap_or(0)
    }

    /// `log2` of the ring degree.
    #[must_use]
    pub fn logn(&self) -> usize {
        self.degree().trailing_zeros() as usize
    }

    pub(crate) fn fill_from_verified<P: DirectParams>(
        &mut self,
        verified: &[VerifiedFalconSignature<P>],
    ) -> Result<(), DirectError> {
        assert_eq!(verified.len(), self.live_rows, "verified row count changed");
        assert_eq!(self.degree(), P::DEGREE, "batch degree changed");
        let degree = P::DEGREE;
        #[cfg(feature = "parallel")]
        {
            let coefficient_chunk_cells = DIRECT_CHUNK_ROWS * degree;
            let slack_chunk_cells = DIRECT_CHUNK_ROWS * 4;
            self.h
                .par_chunks_mut(coefficient_chunk_cells)
                .zip(self.t.par_chunks_mut(coefficient_chunk_cells))
                .zip(self.s1.par_chunks_mut(coefficient_chunk_cells))
                .zip(self.s2.par_chunks_mut(coefficient_chunk_cells))
                .zip(self.v.par_chunks_mut(coefficient_chunk_cells))
                .zip(self.slack.par_chunks_mut(slack_chunk_cells))
                .zip(verified.par_chunks(DIRECT_CHUNK_ROWS))
                .enumerate()
                .try_for_each(
                    |(chunk_index, ((((((h, t), s1), s2), v), slack), verified))| {
                        tracing::info_span!(
                            "v1.prove.witness.compact_direct.chunk",
                            chunk = chunk_index,
                            records = verified.len()
                        )
                        .in_scope(|| {
                            for (row_index, value) in verified.iter().enumerate() {
                                let start = row_index * degree;
                                let end = start + degree;
                                let slack_start = row_index * 4;
                                write_compact_verified_row(
                                    &mut h[start..end],
                                    &mut t[start..end],
                                    &mut s1[start..end],
                                    &mut s2[start..end],
                                    &mut v[start..end],
                                    &mut slack[slack_start..slack_start + 4],
                                    value,
                                )?;
                            }
                            Ok::<(), DirectError>(())
                        })
                    },
                )?;
        }
        #[cfg(not(feature = "parallel"))]
        for (row_index, value) in verified.iter().enumerate() {
            let start = row_index * degree;
            let end = start + degree;
            let slack_start = row_index * 4;
            write_compact_verified_row(
                &mut self.h[start..end],
                &mut self.t[start..end],
                &mut self.s1[start..end],
                &mut self.s2[start..end],
                &mut self.v[start..end],
                &mut self.slack[slack_start..slack_start + 4],
                value,
            )?;
        }
        Ok(())
    }

    /// Consume and checked-narrow the shared validation representation.
    pub fn from_direct(batch: DirectBatch) -> Result<Self, DirectError> {
        let degree = supported_degree(batch.degree())?;
        let coefficient_cells = batch
            .padded_rows
            .checked_mul(degree)
            .ok_or(DirectError::IntegerOverflow)?;
        let valid_shape = batch.live_rows != 0
            && batch.live_rows <= batch.padded_rows
            && batch.padded_rows == batch.live_rows.next_power_of_two()
            && batch.h.len() == coefficient_cells
            && batch.t.len() == coefficient_cells
            && batch.s1.len() == coefficient_cells
            && batch.s2.len() == coefficient_cells
            && batch.v.len() == coefficient_cells
            && batch.slack.len() == coefficient_cells;
        if !valid_shape {
            return Err(DirectError::InvalidLength);
        }

        let narrow_i16 = |source: Vec<i64>| {
            source
                .into_iter()
                .map(|value| i16::try_from(value).map_err(|_| DirectError::IntegerOverflow))
                .collect::<Result<Vec<_>, _>>()
        };
        let narrow_i32 = |source: Vec<i64>| {
            source
                .into_iter()
                .map(|value| i32::try_from(value).map_err(|_| DirectError::IntegerOverflow))
                .collect::<Result<Vec<_>, _>>()
        };
        let mut slack = Vec::with_capacity(4 * batch.padded_rows);
        for record in 0..batch.padded_rows {
            let start = record * degree;
            for &value in &batch.slack[start..start + 4] {
                slack.push(i16::try_from(value).map_err(|_| DirectError::IntegerOverflow)?);
            }
        }
        Ok(Self {
            live_rows: batch.live_rows,
            padded_rows: batch.padded_rows,
            h: narrow_i16(batch.h)?,
            t: narrow_i16(batch.t)?,
            s1: narrow_i16(batch.s1)?,
            s2: narrow_i16(batch.s2)?,
            v: narrow_i32(batch.v)?,
            slack,
        })
    }

    /// Build the compact owner from exact direct rows.
    #[must_use]
    pub fn from_rows<P: DirectParams>(rows: &[DirectRow<P>]) -> Self {
        let degree = P::DEGREE;
        let mut batch = Self::allocate_of::<P>(rows.len());
        for (record, row) in rows.iter().enumerate() {
            let start = record * degree;
            let end = start + degree;
            batch.h[start..end].copy_from_slice(row.h.as_ref().as_ref());
            batch.t[start..end].copy_from_slice(row.t.as_ref().as_ref());
            batch.s1[start..end].copy_from_slice(row.s1.as_ref().as_ref());
            batch.s2[start..end].copy_from_slice(row.s2.as_ref().as_ref());
            for (destination, &source) in
                batch.v[start..end].iter_mut().zip(row.v.as_ref().as_ref())
            {
                *destination = i32::try_from(source)
                    .expect("validated DirectRow quotient always fits its compact carrier");
            }
            batch.slack[4 * record..4 * record + 4].copy_from_slice(&row.slack);
        }
        batch
    }

    /// Bytes retained by the coefficient vectors, excluding allocator slack.
    #[must_use]
    pub fn retained_coefficient_bytes(&self) -> usize {
        (self.h.len() + self.t.len() + self.s1.len() + self.s2.len() + self.slack.len())
            * size_of::<i16>()
            + self.v.len() * size_of::<i32>()
    }
}

#[allow(clippy::too_many_arguments)]
fn write_compact_verified_row<P: DirectParams>(
    h: &mut [i16],
    t: &mut [i16],
    s1: &mut [i16],
    s2: &mut [i16],
    v: &mut [i32],
    slack: &mut [i16],
    value: &VerifiedFalconSignature<P>,
) -> Result<(), DirectError> {
    let auxiliaries = CompactDirectAuxiliaries::derive(value)?;
    h.copy_from_slice(value.h_centered().as_ref());
    t.copy_from_slice(value.t_centered().as_ref());
    s1.copy_from_slice(value.s1().as_ref());
    s2.copy_from_slice(value.s2().as_ref());
    for (destination, &source) in v.iter_mut().zip(&auxiliaries.v) {
        *destination = source;
    }
    for (destination, source) in slack.iter_mut().zip(auxiliaries.slack) {
        *destination = source;
    }
    Ok(())
}

#[derive(Debug, Eq, PartialEq)]
struct CompactDirectAuxiliaries {
    v: Vec<i32>,
    slack: [i16; 4],
}

impl CompactDirectAuxiliaries {
    fn derive<P: DirectParams>(value: &VerifiedFalconSignature<P>) -> Result<Self, DirectError> {
        let norm = value.norm_squared();
        if norm > P::BETA_SQUARED {
            return Err(DirectError::NormExceeded(norm));
        }
        let product_mod_q = value.product_mod_q();
        let product = P::convolution_negacyclic_i16_from_mod_q(
            value.h_centered().as_ref(),
            value.s2().as_ref(),
            product_mod_q.as_ref(),
        );
        Self::derive_from_product(value, norm, &product)
    }

    fn derive_from_product<P: DirectParams>(
        value: &VerifiedFalconSignature<P>,
        norm: u64,
        product: &[i64],
    ) -> Result<Self, DirectError> {
        let degree = P::DEGREE;
        let mut wide_v = vec![0_i64; degree];
        for index in 0..degree {
            let numerator = product[index]
                .checked_add(i64::from(value.s1().as_ref()[index]))
                .and_then(|coefficient| {
                    coefficient.checked_sub(i64::from(value.t_centered().as_ref()[index]))
                })
                .ok_or(DirectError::IntegerOverflow)?;
            if numerator % FALCON_Q != 0 {
                return Err(DirectError::NonIntegralLift(index));
            }
            wide_v[index] = numerator / FALCON_Q;
        }
        let slack = P::BETA_SQUARED - norm;
        let slack_u64 =
            four_squares(slack).ok_or(DirectError::CappedFourSquaresUnavailable(slack))?;
        let mut slack = [0_i16; 4];
        for (destination, source) in slack.iter_mut().zip(slack_u64) {
            *destination = i16::try_from(source).map_err(|_| DirectError::IntegerOverflow)?;
        }
        super::validation::validate_row_semantics::<P>(
            value.t_centered().as_ref(),
            value.s1().as_ref(),
            value.s2().as_ref(),
            &wide_v,
            slack,
            product,
        )?;
        let v = wide_v
            .into_iter()
            .map(|source| i32::try_from(source).map_err(|_| DirectError::IntegerOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { v, slack })
    }
}

#[cfg(test)]
mod tests {
    use aerie_falcon_native::{witness::verify_signatures_native, Falcon1024};

    use super::*;
    #[cfg(feature = "honest-vectors")]
    use crate::direct::ExactConvolution;
    use crate::fixtures::constructed_statement;
    #[cfg(feature = "honest-vectors")]
    use crate::honest::honest_statement;
    use crate::{DirectRow, FALCON_DEGREE};

    #[cfg(feature = "honest-vectors")]
    #[test]
    fn reused_residue_auxiliaries_match_the_full_crt_oracle() {
        let statement = honest_statement(3, 0x47cc_8d25).unwrap();
        let verified = verify_signatures_native(statement.records()).unwrap();
        for value in &verified {
            let actual = CompactDirectAuxiliaries::derive(value).unwrap();
            let product = Falcon512::convolution_negacyclic_i16(
                value.h_centered().as_ref(),
                value.s2().as_ref(),
            );
            let expected = CompactDirectAuxiliaries::derive_from_product(
                value,
                value.norm_squared(),
                &product,
            )
            .unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn writer_matches_the_row_materialization_path() {
        let statement = constructed_statement(3).unwrap();
        let verified = verify_signatures_native(statement.records()).unwrap();
        let rows = verified
            .iter()
            .map(DirectRow::from_verified)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let expected = CompactDirectBatch::from_direct(DirectBatch::from_rows(&rows)).unwrap();
        let mut actual = CompactDirectBatch::allocate(verified.len());
        actual.fill_from_verified(&verified).unwrap();

        assert_eq!(actual, expected);
        assert_eq!(actual.slack.len(), 4 * actual.padded_rows);
        assert_eq!(actual.degree(), 512);
    }

    #[test]
    fn writer_crosses_the_parallel_chunk_boundary() {
        const LIVE_ROWS: usize = 257;

        let source = constructed_statement(2).unwrap();
        let pair = verify_signatures_native(source.records()).unwrap();
        let mut verified = vec![pair[0].clone(); LIVE_ROWS];
        verified[LIVE_ROWS - 1] = pair[1].clone();
        let final_row = DirectRow::from_verified(&pair[1]).unwrap();
        let final_compact = CompactDirectBatch::from_rows(std::slice::from_ref(&final_row));

        let mut direct = DirectBatch::allocate(LIVE_ROWS);
        direct.fill_from_verified(&verified).unwrap();
        let expected = CompactDirectBatch::from_direct(direct).unwrap();
        let mut actual = CompactDirectBatch::allocate(LIVE_ROWS);
        actual.fill_from_verified(&verified).unwrap();

        assert_eq!(actual, expected);
        assert_eq!((actual.live_rows, actual.padded_rows), (257, 512));
        let final_start = (LIVE_ROWS - 1) * FALCON_DEGREE;
        assert_eq!(
            actual.h[final_start..final_start + FALCON_DEGREE],
            final_compact.h
        );
        assert_eq!(
            actual.v[final_start..final_start + FALCON_DEGREE],
            final_compact.v
        );
        assert_eq!(
            actual.slack[(LIVE_ROWS - 1) * 4..LIVE_ROWS * 4],
            final_compact.slack
        );
        assert!(actual.h[LIVE_ROWS * FALCON_DEGREE..]
            .iter()
            .all(|&cell| cell == 0));
        assert!(actual.t[LIVE_ROWS * FALCON_DEGREE..]
            .iter()
            .all(|&cell| cell == 0));
        assert!(actual.s1[LIVE_ROWS * FALCON_DEGREE..]
            .iter()
            .all(|&cell| cell == 0));
        assert!(actual.s2[LIVE_ROWS * FALCON_DEGREE..]
            .iter()
            .all(|&cell| cell == 0));
        assert!(actual.v[LIVE_ROWS * FALCON_DEGREE..]
            .iter()
            .all(|&cell| cell == 0));
        assert!(actual.slack[LIVE_ROWS * 4..].iter().all(|&cell| cell == 0));
    }

    #[test]
    fn conversion_rejects_values_outside_signed_carriers() {
        let row = DirectRow::<Falcon512>::synthetic(19).unwrap();

        let mut oversized_native = DirectBatch::from_rows(std::slice::from_ref(&row));
        oversized_native.h[0] = i64::from(i16::MAX) + 1;
        assert_eq!(
            CompactDirectBatch::from_direct(oversized_native),
            Err(DirectError::IntegerOverflow)
        );

        let mut oversized_quotient = DirectBatch::from_rows(std::slice::from_ref(&row));
        oversized_quotient.v[0] = i64::from(i32::MAX) + 1;
        assert_eq!(
            CompactDirectBatch::from_direct(oversized_quotient),
            Err(DirectError::IntegerOverflow)
        );
    }

    #[test]
    fn falcon_1024_rows_round_trip_through_both_batch_forms() {
        let rows = (0..3)
            .map(|seed| DirectRow::<Falcon1024>::synthetic(seed + 5).unwrap())
            .collect::<Vec<_>>();
        let compact = CompactDirectBatch::from_rows(&rows);
        assert_eq!(compact.degree(), 1024);
        assert_eq!(compact.logn(), 10);
        assert_eq!((compact.live_rows, compact.padded_rows), (3, 4));
        assert_eq!(compact.slack.len(), 16);
        let via_direct = CompactDirectBatch::from_direct(DirectBatch::from_rows(&rows)).unwrap();
        assert_eq!(compact, via_direct);
        compact.validate_semantics().unwrap();
        assert!(compact.h[3 * 1024..].iter().all(|&cell| cell == 0));
    }

    #[test]
    fn retained_bytes_match_the_compact_layout() {
        const TARGET_ROWS: usize = 1 << 14;
        const TARGET_BYTES: usize = (4 * FALCON_DEGREE + 4) * TARGET_ROWS * size_of::<i16>()
            + FALCON_DEGREE * TARGET_ROWS * size_of::<i32>();

        let rows = (0..3)
            .map(|seed| DirectRow::<Falcon512>::synthetic(seed).unwrap())
            .collect::<Vec<_>>();
        let batch = CompactDirectBatch::from_rows(&rows);
        let expected = (4 * FALCON_DEGREE + 4) * batch.padded_rows * size_of::<i16>()
            + FALCON_DEGREE * batch.padded_rows * size_of::<i32>();
        assert_eq!(batch.retained_coefficient_bytes(), expected);
        assert_eq!(TARGET_BYTES, 100_794_368);
    }
}
