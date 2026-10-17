//! Source-validated Falcon witness preparation and private input parsing.

use crate::native::{
    record::DirectRecordRef,
    witness::{verify_signatures_native, VerifiedFalconSignature},
};
use crate::{CompactDirectBatch, DirectParams};
use aerie_falcon_native::{source::CanonicalSignatureBytes, FalconParams};
use aerie_falcon_statement::production::{length_u32, public_canonical_encoding_len, WireReader};
pub use aerie_falcon_statement::{
    AggregationRecord, AggregationStatement, ProductionError, MAX_PRODUCTION_BATCH_SIZE,
};

/// Private signatures corresponding positionally to an
/// [`AggregationStatement`] of the same parameter set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AggregationWitness<P: FalconParams> {
    signatures: Vec<CanonicalSignatureBytes<P>>,
}

impl<P: FalconParams> AggregationWitness<P> {
    pub fn new(signatures: Vec<CanonicalSignatureBytes<P>>) -> Result<Self, ProductionError> {
        if signatures.is_empty() || signatures.len() > MAX_PRODUCTION_BATCH_SIZE {
            return Err(ProductionError::InvalidBatchLength(signatures.len()));
        }
        Ok(Self { signatures })
    }

    #[must_use]
    pub fn signatures(&self) -> &[CanonicalSignatureBytes<P>] {
        &self.signatures
    }

    /// Versioned private-witness encoding used by test vectors and prover
    /// input files. Signatures have the parameter set's fixed padded wire
    /// length.
    pub fn canonical_encoding(&self) -> Result<Vec<u8>, ProductionError> {
        let mut output = Vec::with_capacity(6 + self.signatures.len() * P::SIGNATURE_BYTES);
        output.extend_from_slice(&1_u16.to_le_bytes());
        output.extend_from_slice(
            &length_u32("signature count", self.signatures.len())?.to_le_bytes(),
        );
        for signature in &self.signatures {
            output.extend_from_slice(signature.as_bytes().as_ref());
        }
        Ok(output)
    }

    /// Parse a version-1 private witness with a bounded signature count and
    /// exact input consumption.
    pub fn parse_canonical(bytes: &[u8]) -> Result<Self, ProductionError> {
        parse_aggregate_witness::<P>(bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreprocessingMetrics {
    pub records: usize,
    pub canonical_statement_bytes: usize,
    pub record_materialize_ns: u128,
    pub native_verify_ns: u128,
    pub direct_witness_ns: u128,
    pub pack_ns: u128,
}

/// Source-validated compact witness for a transport that supplies its own
/// transcript binding.
///
/// The Falcon transport binds canonical keys, public salts, and messages
/// directly, so this carrier contains only the validated witness and metrics.
#[derive(Clone, Debug)]
pub struct FalconPreparedWitness {
    witness: CompactDirectBatch,
    metrics: PreprocessingMetrics,
}

struct VerifiedAggregateWitness<P: FalconParams> {
    canonical_statement_bytes: usize,
    record_materialize_ns: u128,
    native_verify_ns: u128,
    verified: Vec<VerifiedFalconSignature<P>>,
}

impl<P: DirectParams> VerifiedAggregateWitness<P> {
    fn finalize(self) -> Result<FalconPreparedWitness, ProductionError> {
        let Self {
            canonical_statement_bytes,
            record_materialize_ns,
            native_verify_ns,
            verified,
        } = self;

        let mut pack_ns: u128 = 0;
        let mut witness = crate::staged!(
            "v1.prove.witness.pack",
            pack_ns,
            CompactDirectBatch::allocate_of::<P>(verified.len())
        );
        let mut direct_witness_ns: u128 = 0;
        crate::staged!(
            "v1.prove.witness.direct",
            direct_witness_ns,
            witness.fill_from_verified::<P>(&verified)?
        );
        let records = verified.len();
        drop(verified);

        Ok(FalconPreparedWitness {
            witness,
            metrics: PreprocessingMetrics {
                records,
                canonical_statement_bytes,
                record_materialize_ns,
                native_verify_ns,
                direct_witness_ns,
                pack_ns,
            },
        })
    }
}

impl FalconPreparedWitness {
    /// Validate public records and private signatures and build only the
    /// compact witness needed by the Falcon transport.
    ///
    /// Native Falcon verification and witness range checks are performed
    /// before the compact witness is returned.
    pub fn from_aggregate<P: DirectParams>(
        statement: &AggregationStatement<P>,
        witness: &AggregationWitness<P>,
    ) -> Result<Self, ProductionError> {
        let preprocess_span = tracing::info_span!(
            "v1.prove.witness.falcon",
            layout = "compact-v1",
            binding = "transport"
        )
        .entered();
        let prepared = verify_aggregate_witness::<P>(statement, witness)?.finalize()?;
        drop(preprocess_span);
        Ok(prepared)
    }

    #[must_use]
    pub const fn witness(&self) -> &CompactDirectBatch {
        &self.witness
    }

    #[must_use]
    pub fn into_witness(self) -> CompactDirectBatch {
        self.witness
    }

    #[must_use]
    pub const fn metrics(&self) -> PreprocessingMetrics {
        self.metrics
    }
}

fn verify_aggregate_witness<P: FalconParams>(
    statement: &AggregationStatement<P>,
    witness: &AggregationWitness<P>,
) -> Result<VerifiedAggregateWitness<P>, ProductionError> {
    let mut verified = verify_aggregate_records(statement, witness)?;
    verified.canonical_statement_bytes = public_canonical_encoding_len::<P>(statement.messages())?;
    Ok(verified)
}

fn verify_aggregate_records<P: FalconParams>(
    statement: &AggregationStatement<P>,
    witness: &AggregationWitness<P>,
) -> Result<VerifiedAggregateWitness<P>, ProductionError> {
    validate_aggregate_shapes(statement, witness)?;
    let mut record_materialize_ns: u128 = 0;
    let records = crate::staged!("v1.prove.witness.records", record_materialize_ns, {
        statement
            .public_keys()
            .iter()
            .zip(statement.messages())
            .zip(&witness.signatures)
            .map(|((public_key, message), signature)| {
                DirectRecordRef::<P>::new(public_key, signature, message)
            })
            .collect::<Vec<_>>()
    });
    let mut native_verify_ns: u128 = 0;
    let verified = crate::staged!(
        "v1.prove.witness.native_verify",
        native_verify_ns,
        verify_signatures_native::<P>(&records)?
    );
    Ok(VerifiedAggregateWitness {
        canonical_statement_bytes: 0,
        record_materialize_ns,
        native_verify_ns,
        verified,
    })
}

fn validate_aggregate_shapes<P: FalconParams>(
    statement: &AggregationStatement<P>,
    witness: &AggregationWitness<P>,
) -> Result<(), ProductionError> {
    if statement.len() == witness.signatures.len() {
        Ok(())
    } else {
        Err(ProductionError::WitnessLengthMismatch {
            statement: statement.len(),
            witness: witness.signatures.len(),
        })
    }
}

fn parse_aggregate_witness<P: FalconParams>(
    bytes: &[u8],
) -> Result<AggregationWitness<P>, ProductionError> {
    let mut reader = WireReader::new(bytes);
    if reader.u16()? != 1 {
        return Err(ProductionError::MalformedWire(
            "unsupported aggregate-witness version".to_owned(),
        ));
    }
    let count = reader.count("signature count", MAX_PRODUCTION_BATCH_SIZE)?;
    if count == 0 {
        return Err(ProductionError::InvalidBatchLength(0));
    }
    // Signatures have fixed wire sizes. Validate the complete table before
    // reserving it; the bounded count and sealed parameter sets fit usize.
    reader.ensure_remaining(count * P::SIGNATURE_BYTES)?;
    let mut signatures = Vec::with_capacity(count);
    for _ in 0..count {
        signatures.push(CanonicalSignatureBytes::<P>::parse(
            reader.take(P::SIGNATURE_BYTES)?,
        )?);
    }
    reader.finish()?;
    AggregationWitness::new(signatures)
}

#[cfg(test)]
mod tests {
    use aerie_falcon_native::{
        params::FALCON_SALT_BYTES,
        source::{CanonicalPublicKeyBytes, CanonicalSignatureBytes},
        Falcon512,
    };

    use super::*;
    use crate::fixtures::constructed_aggregation;
    use crate::native::{record::DirectRecord, target::HashToPointTarget};

    #[test]
    fn public_statement_encoding_binds_message_data() {
        let (first, _) = statement(0);
        let (second, _) = statement(1);
        assert_ne!(
            first.canonical_encoding().unwrap(),
            second.canonical_encoding().unwrap()
        );
        assert_eq!(first.len(), 1);
    }

    #[test]
    fn statement_columns_move_messages_and_preserve_record_order() {
        fn check<P: FalconParams>() {
            let (original, _) = constructed_aggregation::<P>(3).unwrap();
            let expected_encoding = original.canonical_encoding().unwrap();
            let records = original.to_records();
            let message_pointers = records
                .iter()
                .map(|record| record.message().as_ptr())
                .collect::<Vec<_>>();
            let statement = AggregationStatement::new(records).unwrap();
            assert_eq!(statement.len(), 3);
            assert!(!statement.is_empty());
            assert_eq!(statement.public_keys().len(), statement.messages().len());
            for (message, pointer) in statement.messages().iter().zip(message_pointers) {
                assert_eq!(
                    message.as_ptr(),
                    pointer,
                    "constructor must move message allocations"
                );
            }
            assert_eq!(statement, original);
            assert_eq!(statement.canonical_encoding().unwrap(), expected_encoding);
            assert_eq!(
                AggregationStatement::new(statement.to_records()).unwrap(),
                statement
            );
        }
        check::<Falcon512>();
        check::<aerie_falcon_native::Falcon1024>();
    }

    #[test]
    fn prepared_falcon_witness_reports_current_metrics() {
        let (public, witness) = constructed_aggregation::<Falcon512>(3).unwrap();
        let encoded = public.canonical_encoding().unwrap();

        let prepared = FalconPreparedWitness::from_aggregate(&public, &witness).unwrap();

        assert_eq!(prepared.metrics().records, public.len());
        assert_eq!(prepared.metrics().canonical_statement_bytes, encoded.len());
        assert_eq!(prepared.witness().live_rows, public.len());
        assert_eq!(prepared.witness().padded_rows, 4);
    }

    #[test]
    fn aggregate_statement_excludes_signatures_and_salts() {
        let (first_public, first_witness) = statement_with_salt(7, 0);
        let (second_public, second_witness) = statement_with_salt(7, 1);

        assert_eq!(first_public, second_public);
        assert_ne!(first_witness, second_witness);
        let first_prepared =
            FalconPreparedWitness::from_aggregate(&first_public, &first_witness).unwrap();
        let second_prepared =
            FalconPreparedWitness::from_aggregate(&second_public, &second_witness).unwrap();
        assert_ne!(first_prepared.witness().t, second_prepared.witness().t);
        assert_eq!(
            first_public.canonical_encoding().unwrap().len(),
            6 + 1 + 4 + Falcon512::PUBLIC_KEY_BYTES + 4 + 1
        );
    }

    #[test]
    fn aggregate_witness_length_must_match_public_records() {
        let (public, witness) = statement(9);
        let signature = witness.signatures()[0].clone();
        let too_many = AggregationWitness::new(vec![signature.clone(), signature]).unwrap();
        assert!(matches!(
            FalconPreparedWitness::from_aggregate(&public, &too_many),
            Err(ProductionError::WitnessLengthMismatch {
                statement: 1,
                witness: 2
            })
        ));
    }

    #[test]
    fn aggregate_statement_and_witness_wire_roundtrip() {
        fn check<P: FalconParams>() {
            for count in [1, 3] {
                let (public, witness) = constructed_aggregation::<P>(count).unwrap();
                let public_bytes = public.canonical_encoding().unwrap();
                let witness_bytes = witness.canonical_encoding().unwrap();
                assert_eq!(&public_bytes[..2], &1_u16.to_le_bytes());
                assert_eq!(&public_bytes[2..6], &(count as u32).to_le_bytes());
                assert_eq!(
                    AggregationStatement::<P>::parse_canonical(&public_bytes).unwrap(),
                    public
                );
                assert_eq!(
                    AggregationWitness::<P>::parse_canonical(&witness_bytes).unwrap(),
                    witness
                );
                assert!(AggregationWitness::<P>::parse_canonical(
                    &witness_bytes[..witness_bytes.len() - 1]
                )
                .is_err());
            }
        }
        check::<Falcon512>();
        check::<aerie_falcon_native::Falcon1024>();
    }

    #[test]
    fn public_statement_reconstructs_the_prepared_witness_batch() {
        let (public, witness) = constructed_aggregation::<Falcon512>(3).unwrap();
        let prepared = FalconPreparedWitness::from_aggregate(&public, &witness).unwrap();

        let batch = prepared.witness();
        assert_eq!(batch.live_rows, 3);
        assert_eq!(batch.padded_rows, 4);
        assert_eq!(batch.degree(), Falcon512::DEGREE);
        assert_eq!(batch.h.len(), batch.padded_rows * Falcon512::DEGREE);
        for (row, public_key) in public.public_keys().iter().enumerate() {
            let decoded = public_key.decode();
            let start = row * Falcon512::DEGREE;
            let end = start + Falcon512::DEGREE;
            assert_eq!(&batch.h[start..end], decoded.h_centered().as_ref());
        }
        assert!(batch.h[batch.live_rows * Falcon512::DEGREE..]
            .iter()
            .all(|&coefficient| coefficient == 0));
    }

    #[test]
    fn aggregate_wire_parser_rejects_malformed_inputs() {
        let (public, witness) = statement(9);
        let public_bytes = public.canonical_encoding().unwrap();
        let witness_bytes = witness.canonical_encoding().unwrap();

        for length in 0..public_bytes.len() {
            assert!(
                AggregationStatement::<Falcon512>::parse_canonical(&public_bytes[..length])
                    .is_err()
            );
        }
        for length in 0..witness_bytes.len() {
            assert!(
                AggregationWitness::<Falcon512>::parse_canonical(&witness_bytes[..length]).is_err()
            );
        }

        let mut trailing = public_bytes.clone();
        trailing.push(0);
        assert!(AggregationStatement::<Falcon512>::parse_canonical(&trailing).is_err());
        let mut trailing = witness_bytes.clone();
        trailing.push(0);
        assert!(AggregationWitness::<Falcon512>::parse_canonical(&trailing).is_err());

        let mut bad_version = public_bytes.clone();
        bad_version[0] = 2;
        assert!(AggregationStatement::<Falcon512>::parse_canonical(&bad_version).is_err());

        let count_offset = 2;
        let mut oversized_count = public_bytes.clone();
        oversized_count[count_offset..count_offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(AggregationStatement::<Falcon512>::parse_canonical(&oversized_count).is_err());

        let record_offset = count_offset + 4;
        let mut bad_record_tag = public_bytes.clone();
        bad_record_tag[record_offset] = 2;
        assert!(AggregationStatement::<Falcon512>::parse_canonical(&bad_record_tag).is_err());

        let public_key_offset = record_offset + 1 + 4;
        let mut bad_public_key = public_bytes.clone();
        bad_public_key[public_key_offset] = 0;
        assert!(AggregationStatement::<Falcon512>::parse_canonical(&bad_public_key).is_err());

        let mut bad_signature = witness_bytes;
        bad_signature[6] = 0;
        assert!(AggregationWitness::<Falcon512>::parse_canonical(&bad_signature).is_err());
    }

    fn statement(
        index: u8,
    ) -> (
        AggregationStatement<Falcon512>,
        AggregationWitness<Falcon512>,
    ) {
        statement_with_salt(index, index)
    }

    fn statement_with_salt(
        message_index: u8,
        salt_index: u8,
    ) -> (
        AggregationStatement<Falcon512>,
        AggregationWitness<Falcon512>,
    ) {
        const H: i16 = 111;
        let mut public_coefficients = [0_u16; Falcon512::DEGREE];
        public_coefficients[0] = H as u16;
        let public_key =
            CanonicalPublicKeyBytes::<Falcon512>::parse(&encode_public(&public_coefficients))
                .unwrap();
        let mut salt = [0_u8; FALCON_SALT_BYTES];
        salt[0] = salt_index;
        let message = vec![message_index];
        let zero_signature = CanonicalSignatureBytes::<Falcon512>::parse(&encode_signature(
            &salt,
            &[0; Falcon512::DEGREE],
        ))
        .unwrap();
        let hash_record =
            DirectRecord::<Falcon512>::new(public_key.clone(), zero_signature, message.clone());
        let target = HashToPointTarget::<Falcon512>::derive(&hash_record);
        let s2 = target.t_centered().map(|coefficient| {
            let coefficient = i32::from(coefficient);
            let rounded = if coefficient >= 0 {
                (coefficient + i32::from(H) / 2) / i32::from(H)
            } else {
                (coefficient - i32::from(H) / 2) / i32::from(H)
            };
            rounded as i16
        });
        let signature =
            CanonicalSignatureBytes::<Falcon512>::parse(&encode_signature(&salt, &s2)).unwrap();
        (
            AggregationStatement::new(vec![AggregationRecord::new(public_key, message).unwrap()])
                .unwrap(),
            AggregationWitness::new(vec![signature]).unwrap(),
        )
    }

    fn encode_public(coefficients: &[u16; Falcon512::DEGREE]) -> [u8; Falcon512::PUBLIC_KEY_BYTES] {
        let mut output = [0_u8; Falcon512::PUBLIC_KEY_BYTES];
        output[0] = 0x09;
        let mut bit = 0;
        for &coefficient in coefficients {
            write_bits(&mut output[1..], &mut bit, coefficient, 14);
        }
        output
    }

    fn encode_signature(
        salt: &[u8; FALCON_SALT_BYTES],
        coefficients: &[i16; Falcon512::DEGREE],
    ) -> [u8; Falcon512::SIGNATURE_BYTES] {
        let mut output = [0_u8; Falcon512::SIGNATURE_BYTES];
        output[0] = 0x39;
        output[1..=FALCON_SALT_BYTES].copy_from_slice(salt);
        let mut bit = 0;
        for &coefficient in coefficients {
            write_bit(
                &mut output[1 + FALCON_SALT_BYTES..],
                &mut bit,
                coefficient < 0,
            );
            let magnitude = coefficient.unsigned_abs();
            write_bits(
                &mut output[1 + FALCON_SALT_BYTES..],
                &mut bit,
                magnitude & 127,
                7,
            );
            for _ in 0..magnitude >> 7 {
                write_bit(&mut output[1 + FALCON_SALT_BYTES..], &mut bit, false);
            }
            write_bit(&mut output[1 + FALCON_SALT_BYTES..], &mut bit, true);
        }
        output
    }

    fn write_bits(output: &mut [u8], bit: &mut usize, value: u16, count: usize) {
        for shift in (0..count).rev() {
            write_bit(output, bit, ((value >> shift) & 1) != 0);
        }
    }

    fn write_bit(output: &mut [u8], bit: &mut usize, value: bool) {
        if value {
            output[*bit / 8] |= 1 << (7 - *bit % 8);
        }
        *bit += 1;
    }
}
