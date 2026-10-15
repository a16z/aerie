//! Public statement inputs shared by the PCS prover and verifier.

use super::ring::{PublicRingKeys, PublicTargets};
use super::FalconDegree;

/// Public statement data the verifier derives independently of the proof.
#[derive(Clone, Debug)]
pub struct PublicInputs<'a, C = i64> {
    /// The Falcon parameter set, independent of the public input representations.
    pub degree: FalconDegree,
    /// The LIVE record count `N`; the padded domain `N_sig*` is derived.
    pub records: usize,
    /// Salt transport mode, bound independently of the public-record digest.
    pub salt_mode: super::wire::SaltMode,
    /// The 32-byte public-record binding digest.
    pub public_records: &'a [u8],
    /// Public ring keys as a padded coefficient table or canonical bytes.
    pub h: PublicRingKeys<'a, C>,
    /// Public `HashToPoint` targets as a padded table or live source tuples.
    pub t: PublicTargets<'a, C>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expanded_table_degree_is_explicit_at_the_minimum_shape() {
        let table = vec![0_i64; super::super::MIN_PADDED_COEFFICIENTS];
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            let inputs = PublicInputs {
                degree,
                records: 1,
                salt_mode: super::super::wire::SaltMode::PerRecord,
                public_records: &[0; 32],
                h: PublicRingKeys::Expanded(&table),
                t: PublicTargets::Expanded(&table),
            };
            assert_eq!(inputs.degree, degree);
            let records = degree.padded_records(inputs.records).unwrap();
            assert_eq!(
                inputs
                    .h
                    .validate_shape_of(degree.degree(), inputs.records, records),
                Ok(())
            );
            assert_eq!(
                inputs
                    .t
                    .validate_shape_of(degree.degree(), inputs.records, records),
                Ok(())
            );
        }
    }

    #[test]
    fn expanded_padding_is_distinct_from_dimension_mismatches() {
        let mut table = vec![0_i64; 2 * 512];
        table[512] = 1;
        assert_eq!(
            PublicRingKeys::Expanded(&table).validate_shape_of(512, 1, 2),
            Err(crate::StatementMismatch::NonZeroPadding)
        );
        assert_eq!(
            PublicTargets::Expanded(&table).validate_shape_of(512, 1, 2),
            Err(crate::StatementMismatch::NonZeroPadding)
        );
        assert_eq!(
            PublicRingKeys::Expanded(&table).validate_shape_of(512, 1, 4),
            Err(crate::StatementMismatch::PublicInputShape)
        );
        assert_eq!(
            PublicTargets::Expanded(&table).validate_shape_of(512, 1, 4),
            Err(crate::StatementMismatch::PublicInputShape)
        );
    }
}
