use core::mem::size_of;

use crate::native::witness::VerifiedFalconSignature;
use aerie_falcon_native::params::FALCON_Q;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use super::DIRECT_CHUNK_ROWS;
use super::{four_squares, DirectError, DirectParams, DirectRow};

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
        let product = super::convolution_negacyclic_i16_from_mod_q::<P>(
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
    use aerie_falcon_native::{Falcon1024, Falcon512, FalconParams};

    use super::*;
    use crate::fixtures::constructed_aggregation;
    #[cfg(feature = "honest-vectors")]
    use crate::honest::{honest_aggregation, SaltMode};
    use crate::native::{record::DirectRecordRef, witness::verify_signatures_native};
    use crate::{AggregationStatement, AggregationWitness, DirectRow};

    fn verify_aggregation<P: DirectParams>(
        statement: &AggregationStatement<P>,
        witness: &AggregationWitness<P>,
    ) -> Vec<VerifiedFalconSignature<P>> {
        let records = statement
            .public_keys()
            .iter()
            .zip(statement.messages())
            .zip(witness.signatures())
            .map(|((public_key, message), signature)| {
                DirectRecordRef::new(public_key, signature, message)
            })
            .collect::<Vec<_>>();
        verify_signatures_native(&records).unwrap()
    }

    fn assert_written_row<P: DirectParams>(
        batch: &CompactDirectBatch,
        record: usize,
        value: &VerifiedFalconSignature<P>,
    ) {
        let start = record * P::DEGREE;
        let end = start + P::DEGREE;
        assert_eq!(&batch.h[start..end], value.h_centered().as_ref());
        assert_eq!(&batch.t[start..end], value.t_centered().as_ref());
        assert_eq!(&batch.s1[start..end], value.s1().as_ref());
        assert_eq!(&batch.s2[start..end], value.s2().as_ref());
        let mut product = vec![0_i64; P::DEGREE];
        for (i, &h) in value.h_centered().as_ref().iter().enumerate() {
            for (j, &s2) in value.s2().as_ref().iter().enumerate() {
                let term = i64::from(h) * i64::from(s2);
                if i + j < P::DEGREE {
                    product[i + j] += term;
                } else {
                    product[i + j - P::DEGREE] -= term;
                }
            }
        }
        for (index, &coefficient) in product.iter().enumerate() {
            let numerator = coefficient + i64::from(value.s1().as_ref()[index])
                - i64::from(value.t_centered().as_ref()[index]);
            assert_eq!(numerator % FALCON_Q, 0);
            assert_eq!(i64::from(batch.v[start + index]), numerator / FALCON_Q);
        }
        let completed_norm = value.norm_squared()
            + batch.slack[record * 4..record * 4 + 4]
                .iter()
                .map(|&cell| i64::from(cell).unsigned_abs().pow(2))
                .sum::<u64>();
        assert_eq!(completed_norm, P::BETA_SQUARED);
    }

    #[cfg(feature = "honest-vectors")]
    #[test]
    fn reused_residue_auxiliaries_satisfy_the_integer_definition() {
        let (statement, witness) =
            honest_aggregation::<Falcon512>(3, 0x47cc_8d25, SaltMode::PerRecord).unwrap();
        let verified = verify_aggregation(&statement, &witness);
        let mut batch = CompactDirectBatch::allocate_of::<Falcon512>(verified.len());
        batch.fill_from_verified(&verified).unwrap();
        for (index, value) in verified.iter().enumerate() {
            assert_written_row(&batch, index, value);
        }
    }

    #[test]
    fn writer_satisfies_the_integer_definition_at_both_degrees() {
        fn check<P: DirectParams>() {
            let (statement, witness) = constructed_aggregation::<P>(3).unwrap();
            let verified = verify_aggregation(&statement, &witness);
            let mut batch = CompactDirectBatch::allocate_of::<P>(verified.len());
            batch.fill_from_verified(&verified).unwrap();
            for (index, value) in verified.iter().enumerate() {
                assert_written_row(&batch, index, value);
            }
            assert_eq!(batch.slack.len(), 4 * batch.padded_rows);
            assert_eq!(batch.degree(), P::DEGREE);
            batch.validate_semantics().unwrap();
        }
        check::<Falcon512>();
        check::<Falcon1024>();
    }

    #[test]
    fn writer_crosses_the_parallel_chunk_boundary() {
        const LIVE_ROWS: usize = 257;

        let (statement, witness) = constructed_aggregation::<Falcon512>(2).unwrap();
        let pair = verify_aggregation(&statement, &witness);
        let mut verified = vec![pair[0].clone(); LIVE_ROWS];
        verified[LIVE_ROWS - 1] = pair[1].clone();
        let mut actual = CompactDirectBatch::allocate_of::<Falcon512>(LIVE_ROWS);
        actual.fill_from_verified(&verified).unwrap();

        assert_eq!((actual.live_rows, actual.padded_rows), (257, 512));
        assert_written_row(&actual, 0, &pair[0]);
        assert_written_row(&actual, LIVE_ROWS - 1, &pair[1]);
        for record in 1..LIVE_ROWS - 1 {
            let start = record * Falcon512::DEGREE;
            let end = start + Falcon512::DEGREE;
            for column in [&actual.h, &actual.t, &actual.s1, &actual.s2] {
                assert_eq!(&column[start..end], &column[..Falcon512::DEGREE]);
            }
            assert_eq!(&actual.v[start..end], &actual.v[..Falcon512::DEGREE]);
            assert_eq!(
                &actual.slack[record * 4..record * 4 + 4],
                &actual.slack[..4]
            );
        }
        for column in [&actual.h, &actual.t, &actual.s1, &actual.s2] {
            assert!(column[LIVE_ROWS * Falcon512::DEGREE..]
                .iter()
                .all(|&cell| cell == 0));
        }
        assert!(actual.v[LIVE_ROWS * Falcon512::DEGREE..]
            .iter()
            .all(|&cell| cell == 0));
        assert!(actual.slack[LIVE_ROWS * 4..].iter().all(|&cell| cell == 0));
    }

    #[test]
    fn falcon_1024_rows_have_valid_compact_semantics() {
        let rows = (0..3)
            .map(|seed| DirectRow::<Falcon1024>::synthetic(seed + 5).unwrap())
            .collect::<Vec<_>>();
        let compact = CompactDirectBatch::from_rows(&rows);
        assert_eq!(compact.degree(), 1024);
        assert_eq!((compact.live_rows, compact.padded_rows), (3, 4));
        assert_eq!(compact.slack.len(), 16);
        compact.validate_semantics().unwrap();
        assert!(compact.h[3 * 1024..].iter().all(|&cell| cell == 0));
    }

    #[test]
    fn retained_bytes_match_the_compact_layout() {
        const TARGET_ROWS: usize = 1 << 14;
        const TARGET_BYTES: usize = (4 * Falcon512::DEGREE + 4) * TARGET_ROWS * size_of::<i16>()
            + Falcon512::DEGREE * TARGET_ROWS * size_of::<i32>();

        let rows = (0..3)
            .map(|seed| DirectRow::<Falcon512>::synthetic(seed).unwrap())
            .collect::<Vec<_>>();
        let batch = CompactDirectBatch::from_rows(&rows);
        let expected = (4 * Falcon512::DEGREE + 4) * batch.padded_rows * size_of::<i16>()
            + Falcon512::DEGREE * batch.padded_rows * size_of::<i32>();
        assert_eq!(batch.retained_coefficient_bytes(), expected);
        assert_eq!(TARGET_BYTES, 100_794_368);
    }
}
