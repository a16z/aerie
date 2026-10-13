//! Canonical Falcon record ingestion for Aerie.
//!
//! Canonical byte parsing remains verifier-local preprocessing, while the
//! dense witness is derived only from source-bound records accepted by the
//! native Aerie verifier.

use crate::{CompactDirectBatch, DirectError, DirectParams, DirectPublicBatch};
use aerie_falcon_native::{
    record::DirectRecord,
    source::{CanonicalPublicKeyBytes, CanonicalSignatureBytes},
    witness::{verify_signatures_native, VerifiedFalconSignature},
    Falcon512, FalconParams,
};

/// Maximum canonical production batch. This matches Falcon's largest
/// generated runtime schedule (`2^16` records).
pub const MAX_PRODUCTION_BATCH_SIZE: usize = 1 << 16;
pub const MAX_MESSAGE_BYTES: usize = 1 << 24;

/// Canonical Aerie direct statement for all supported Falcon batch sizes.
///
/// Its byte encoding follows the v0 direct-record framing. The higher batch
/// cap affects only the outer descriptor, not any FN-DSA record encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionStatement {
    records: Vec<DirectRecord<Falcon512>>,
}

impl ProductionStatement {
    pub fn new(records: Vec<DirectRecord<Falcon512>>) -> Result<Self, ProductionError> {
        if records.is_empty() || records.len() > MAX_PRODUCTION_BATCH_SIZE {
            return Err(ProductionError::InvalidBatchLength(records.len()));
        }
        Ok(Self { records })
    }

    #[must_use]
    pub fn records(&self) -> &[DirectRecord<Falcon512>] {
        &self.records
    }

    pub fn canonical_encoding(&self) -> Result<Vec<u8>, ProductionError> {
        canonical_encoding(&self.records)
    }
}

/// One public aggregate-verification tuple. The FN-DSA signature is supplied
/// separately by [`AggregationWitness`]; its public salt is extracted into the
/// proof transport so the verifier can recompute `HashToPoint`. The canonical
/// FN-DSA framing fixes raw message mode and the empty context, so this record
/// contains key and message bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AggregationRecord<P: FalconParams> {
    public_key: CanonicalPublicKeyBytes<P>,
    message: Vec<u8>,
}

impl<P: FalconParams> AggregationRecord<P> {
    pub fn new(
        public_key: CanonicalPublicKeyBytes<P>,
        message: Vec<u8>,
    ) -> Result<Self, ProductionError> {
        enforce_limit("raw message", message.len(), MAX_MESSAGE_BYTES)?;
        Ok(Self {
            public_key,
            message,
        })
    }

    #[must_use]
    pub const fn public_key(&self) -> &CanonicalPublicKeyBytes<P> {
        &self.public_key
    }

    #[must_use]
    pub fn message(&self) -> &[u8] {
        &self.message
    }
}

/// Public input to aggregate verification: keys and messages only. The
/// parameter set fixes the canonical key length inside the encoding; the
/// encoding version and framing are shared by both parameter sets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AggregationStatement<P: FalconParams> {
    records: Vec<AggregationRecord<P>>,
}

impl<P: FalconParams> AggregationStatement<P> {
    pub fn new(records: Vec<AggregationRecord<P>>) -> Result<Self, ProductionError> {
        validate_batch(records.len())?;
        Ok(Self { records })
    }

    #[must_use]
    pub fn records(&self) -> &[AggregationRecord<P>] {
        &self.records
    }

    pub fn canonical_encoding(&self) -> Result<Vec<u8>, ProductionError> {
        public_canonical_encoding(&self.records)
    }

    /// Parse the version-1 public statement with allocation limits and exact
    /// input consumption.
    pub fn parse_canonical(bytes: &[u8]) -> Result<Self, ProductionError> {
        parse_public_statement::<P>(bytes)
    }

    /// Reconstruct the verifier-side public dense batch from the statement
    /// alone: every canonical public key decoded into its centered
    /// polynomial, rows zero-padded to the power-of-two count. This matches
    /// the prover-side [`crate::DirectBatch::public_batch`] layout exactly and
    /// consumes no signature data.
    #[must_use]
    pub fn public_batch(&self) -> DirectPublicBatch {
        let live_rows = self.records.len();
        let padded_rows = live_rows.next_power_of_two();
        let degree = P::DEGREE;
        let mut h = vec![0_i64; padded_rows * degree];
        for (row, record) in self.records.iter().enumerate() {
            let decoded = record.public_key.decode();
            for (dst, src) in h[row * degree..(row + 1) * degree]
                .iter_mut()
                .zip(decoded.h_centered().as_ref())
            {
                *dst = i64::from(*src);
            }
        }
        DirectPublicBatch {
            live_rows,
            padded_rows,
            h,
        }
    }
}

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

/// Split a source-validation statement into the production public statement
/// and private signature witness without changing any per-record semantics.
pub fn split_source_statement(
    source: &ProductionStatement,
) -> Result<
    (
        AggregationStatement<Falcon512>,
        AggregationWitness<Falcon512>,
    ),
    ProductionError,
> {
    let records = source
        .records()
        .iter()
        .map(|record| {
            AggregationRecord::new(record.public_key().clone(), record.message().to_vec())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let signatures = source
        .records()
        .iter()
        .map(|record| record.signature().clone())
        .collect();
    Ok((
        AggregationStatement::new(records)?,
        AggregationWitness::new(signatures)?,
    ))
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

struct FinalizedWitness<W> {
    witness: W,
    metrics: PreprocessingMetrics,
}

impl<P: FalconParams> VerifiedAggregateWitness<P> {
    fn finalize<W>(
        self,
        allocate: impl FnOnce(usize) -> W,
        fill: impl FnOnce(&mut W, &[VerifiedFalconSignature<P>]) -> Result<(), DirectError>,
    ) -> Result<FinalizedWitness<W>, ProductionError> {
        let Self {
            canonical_statement_bytes,
            record_materialize_ns,
            native_verify_ns,
            verified,
        } = self;

        let mut pack_ns: u128 = 0;
        let mut witness =
            crate::staged!("v1.prove.witness.pack", pack_ns, allocate(verified.len()));
        let mut direct_witness_ns: u128 = 0;
        crate::staged!(
            "v1.prove.witness.direct",
            direct_witness_ns,
            fill(&mut witness, &verified)?
        );
        let records = verified.len();
        drop(verified);

        Ok(FinalizedWitness {
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
    pub fn from_aggregate(
        statement: &AggregationStatement<Falcon512>,
        witness: &AggregationWitness<Falcon512>,
    ) -> Result<Self, ProductionError> {
        Self::from_aggregate_of::<Falcon512>(statement, witness)
    }

    /// [`Self::from_aggregate`] for parameter set `P`.
    pub fn from_aggregate_of<P: DirectParams>(
        statement: &AggregationStatement<P>,
        witness: &AggregationWitness<P>,
    ) -> Result<Self, ProductionError> {
        let preprocess_span = tracing::info_span!(
            "v1.prove.witness.falcon",
            layout = "compact-v1",
            binding = "transport"
        )
        .entered();
        let FinalizedWitness {
            witness: compact,
            metrics,
        } = verify_aggregate_witness::<P>(statement, witness)?.finalize(
            CompactDirectBatch::allocate_of::<P>,
            CompactDirectBatch::fill_from_verified::<P>,
        )?;
        drop(preprocess_span);
        Ok(Self {
            witness: compact,
            metrics,
        })
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
    verified.canonical_statement_bytes = public_canonical_encoding_len(statement.records())?;
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
            .records
            .iter()
            .zip(&witness.signatures)
            .map(|(record, signature)| {
                DirectRecord::<P>::new(
                    record.public_key.clone(),
                    signature.clone(),
                    record.message.clone(),
                )
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
    if statement.records.len() == witness.signatures.len() {
        Ok(())
    } else {
        Err(ProductionError::WitnessLengthMismatch {
            statement: statement.records.len(),
            witness: witness.signatures.len(),
        })
    }
}

#[cfg(test)]
fn verify_records(
    records: &[DirectRecord<Falcon512>],
) -> Result<
    Vec<aerie_falcon_native::witness::VerifiedFalconSignature<Falcon512>>,
    aerie_falcon_native::AerieError,
> {
    verify_signatures_native(records)
}

#[derive(Debug)]
pub enum ProductionError {
    Aerie(aerie_falcon_native::AerieError),
    Direct(DirectError),
    InvalidBatchLength(usize),
    WitnessLengthMismatch {
        statement: usize,
        witness: usize,
    },
    MalformedWire(String),
    LengthLimit {
        field: &'static str,
        length: usize,
        maximum: usize,
    },
    IntegerOverflow(&'static str),
}

impl core::fmt::Display for ProductionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Aerie(error) => write!(f, "canonical FN-DSA input: {error}"),
            Self::Direct(error) => write!(f, "direct Falcon witness: {error}"),
            Self::InvalidBatchLength(length) => {
                write!(f, "production batch has invalid length {length}")
            }
            Self::WitnessLengthMismatch { statement, witness } => write!(
                f,
                "aggregate statement has {statement} records but witness has {witness} signatures"
            ),
            Self::MalformedWire(message) => write!(f, "malformed aggregate wire input: {message}"),
            Self::LengthLimit {
                field,
                length,
                maximum,
            } => write!(f, "{field} has {length} bytes/items; maximum is {maximum}"),
            Self::IntegerOverflow(name) => write!(f, "{name} length exceeds u32"),
        }
    }
}

fn parse_public_statement<P: FalconParams>(
    bytes: &[u8],
) -> Result<AggregationStatement<P>, ProductionError> {
    let mut reader = WireReader::new(bytes);
    if reader.u16()? != 1 {
        return Err(ProductionError::MalformedWire(
            "unsupported public-statement version".to_owned(),
        ));
    }
    let count = reader.count("record count", MAX_PRODUCTION_BATCH_SIZE)?;
    if count == 0 {
        return Err(ProductionError::InvalidBatchLength(0));
    }
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        if reader.u8()? != 1 {
            return Err(ProductionError::MalformedWire(
                "unsupported aggregate-record version".to_owned(),
            ));
        }
        let public_key = reader.length_prefixed("public key", P::PUBLIC_KEY_BYTES)?;
        if public_key.len() != P::PUBLIC_KEY_BYTES {
            return Err(ProductionError::MalformedWire(
                "public key has noncanonical length".to_owned(),
            ));
        }
        let public_key = CanonicalPublicKeyBytes::<P>::parse(&public_key)?;
        let message = reader.length_prefixed("raw message", MAX_MESSAGE_BYTES)?;
        records.push(AggregationRecord::new(public_key, message)?);
    }
    reader.finish()?;
    AggregationStatement::new(records)
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
    let mut signatures = Vec::with_capacity(count);
    for _ in 0..count {
        signatures.push(CanonicalSignatureBytes::<P>::parse(
            reader.take(P::SIGNATURE_BYTES)?,
        )?);
    }
    reader.finish()?;
    AggregationWitness::new(signatures)
}

struct WireReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> WireReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], ProductionError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| ProductionError::MalformedWire("wire offset overflow".to_owned()))?;
        let value = self.bytes.get(self.position..end).ok_or_else(|| {
            ProductionError::MalformedWire("truncated aggregate input".to_owned())
        })?;
        self.position = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, ProductionError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ProductionError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("two-byte slice"),
        ))
    }

    fn u32(&mut self) -> Result<u32, ProductionError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four-byte slice"),
        ))
    }

    fn count(&mut self, field: &'static str, maximum: usize) -> Result<usize, ProductionError> {
        let count = self.u32()? as usize;
        if count > maximum {
            return Err(ProductionError::LengthLimit {
                field,
                length: count,
                maximum,
            });
        }
        Ok(count)
    }

    fn length_prefixed(
        &mut self,
        field: &'static str,
        maximum: usize,
    ) -> Result<Vec<u8>, ProductionError> {
        let length = self.count(field, maximum)?;
        Ok(self.take(length)?.to_vec())
    }

    fn finish(self) -> Result<(), ProductionError> {
        if self.position == self.bytes.len() {
            Ok(())
        } else {
            Err(ProductionError::MalformedWire(
                "trailing bytes after aggregate input".to_owned(),
            ))
        }
    }
}

fn canonical_encoding(records: &[DirectRecord<Falcon512>]) -> Result<Vec<u8>, ProductionError> {
    let mut output = Vec::new();
    output.extend_from_slice(&0_u16.to_le_bytes());
    output.extend_from_slice(&length_u32("record count", records.len())?.to_le_bytes());
    for record in records {
        output.push(1);
        append_bytes(&mut output, "public key", record.public_key().as_bytes())?;
        append_bytes(&mut output, "signature", record.signature().as_bytes())?;
        append_bytes(&mut output, "raw message", record.message())?;
    }
    Ok(output)
}

fn public_canonical_encoding<P: FalconParams>(
    records: &[AggregationRecord<P>],
) -> Result<Vec<u8>, ProductionError> {
    let mut output = Vec::new();
    output.extend_from_slice(&1_u16.to_le_bytes());
    output.extend_from_slice(&length_u32("record count", records.len())?.to_le_bytes());
    for record in records {
        output.push(1);
        append_bytes(
            &mut output,
            "public key",
            record.public_key.as_bytes().as_ref(),
        )?;
        append_bytes(&mut output, "raw message", &record.message)?;
    }
    Ok(output)
}

fn public_canonical_encoding_len<P: FalconParams>(
    records: &[AggregationRecord<P>],
) -> Result<usize, ProductionError> {
    length_u32("record count", records.len())?;
    let add = |length: usize, extra: usize| {
        length
            .checked_add(extra)
            .ok_or(ProductionError::IntegerOverflow(
                "public statement encoding",
            ))
    };
    let mut length = add(2, 4)?;
    for record in records {
        length_u32("raw message", record.message.len())?;
        length = add(length, 1)?;
        length = add(length, 4)?;
        length = add(length, P::PUBLIC_KEY_BYTES)?;
        length = add(length, 4)?;
        length = add(length, record.message.len())?;
    }
    Ok(length)
}

fn validate_batch(records: usize) -> Result<(), ProductionError> {
    if records == 0 || records > MAX_PRODUCTION_BATCH_SIZE {
        return Err(ProductionError::InvalidBatchLength(records));
    }
    Ok(())
}

fn enforce_limit(
    field: &'static str,
    length: usize,
    maximum: usize,
) -> Result<(), ProductionError> {
    if length > maximum {
        Err(ProductionError::LengthLimit {
            field,
            length,
            maximum,
        })
    } else {
        Ok(())
    }
}

fn append_bytes(
    output: &mut Vec<u8>,
    name: &'static str,
    value: &[u8],
) -> Result<(), ProductionError> {
    output.extend_from_slice(&length_u32(name, value.len())?.to_le_bytes());
    output.extend_from_slice(value);
    Ok(())
}

fn length_u32(name: &'static str, length: usize) -> Result<u32, ProductionError> {
    u32::try_from(length).map_err(|_| ProductionError::IntegerOverflow(name))
}

impl std::error::Error for ProductionError {}

impl From<aerie_falcon_native::AerieError> for ProductionError {
    fn from(error: aerie_falcon_native::AerieError) -> Self {
        Self::Aerie(error)
    }
}

impl From<DirectError> for ProductionError {
    fn from(error: DirectError) -> Self {
        Self::Direct(error)
    }
}

#[cfg(test)]
mod tests {
    use aerie_falcon_native::{
        preprocess::HashToPointTarget,
        record::DirectRecord,
        source::{
            CanonicalPublicKeyBytes, CanonicalSignatureBytes, FALCON_512_PUBLIC_KEY_BYTES,
            FALCON_512_SALT_BYTES, FALCON_512_SIGNATURE_BYTES,
        },
    };

    use super::*;
    use crate::{DirectBatch, DirectRow, FALCON_DEGREE, FALCON_Q};

    #[test]
    fn cataloged_power_of_two_batch_lengths_are_accepted() {
        for records in [1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16] {
            assert!(validate_batch(records).is_ok());
        }
        assert!(matches!(
            validate_batch((1 << 16) + 1),
            Err(ProductionError::InvalidBatchLength(65_537))
        ));
    }

    #[test]
    fn public_statement_encoding_binds_message_data() {
        let first = statement(0);
        let second = statement(1);
        let (first, _) = split_source_statement(&first).unwrap();
        let (second, _) = split_source_statement(&second).unwrap();
        assert_ne!(
            first.canonical_encoding().unwrap(),
            second.canonical_encoding().unwrap()
        );
        assert_eq!(first.public_batch().live_rows, 1);
    }

    #[test]
    fn fused_direct_writer_matches_the_row_materialization_path() {
        let source = ProductionStatement::new(
            (0..3)
                .flat_map(|index| statement(index).records().to_vec())
                .collect(),
        )
        .unwrap();
        let verified = verify_records(source.records()).unwrap();
        let rows = verified
            .iter()
            .map(DirectRow::from_verified)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let expected = DirectBatch::from_rows(&rows);
        let mut actual = DirectBatch::allocate(verified.len());
        actual.fill_from_verified(&verified).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn prepared_falcon_witness_reports_current_metrics() {
        let records = (0..3)
            .flat_map(|index| statement(index).records().to_vec())
            .collect::<Vec<_>>();
        let source = ProductionStatement::new(records).unwrap();
        let (public, witness) = split_source_statement(&source).unwrap();
        let encoded = public.canonical_encoding().unwrap();

        let prepared = FalconPreparedWitness::from_aggregate(&public, &witness).unwrap();

        assert_eq!(prepared.metrics().records, public.records().len());
        assert_eq!(prepared.metrics().canonical_statement_bytes, encoded.len());
        assert_eq!(prepared.witness().live_rows, public.records().len());
        assert_eq!(prepared.witness().padded_rows, 4);
    }

    #[test]
    fn aggregate_statement_excludes_signatures_and_salts() {
        let first_source = statement_with_salt(7, 0);
        let second_source = statement_with_salt(7, 1);
        let (first_public, first_witness) = split_source_statement(&first_source).unwrap();
        let (second_public, second_witness) = split_source_statement(&second_source).unwrap();

        assert_eq!(first_public, second_public);
        assert_ne!(first_witness, second_witness);
        let first_prepared =
            FalconPreparedWitness::from_aggregate(&first_public, &first_witness).unwrap();
        let second_prepared =
            FalconPreparedWitness::from_aggregate(&second_public, &second_witness).unwrap();
        assert_ne!(first_prepared.witness().t, second_prepared.witness().t);
        assert!(
            first_public.canonical_encoding().unwrap().len()
                < first_source.canonical_encoding().unwrap().len()
        );
    }

    #[test]
    fn aggregate_witness_length_must_match_public_records() {
        let source = statement(9);
        let (public, witness) = split_source_statement(&source).unwrap();
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
        let source = statement(7);
        let (public, witness) = split_source_statement(&source).unwrap();
        let public_bytes = public.canonical_encoding().unwrap();
        let witness_bytes = witness.canonical_encoding().unwrap();
        assert_eq!(&public_bytes[..2], &1_u16.to_le_bytes());
        assert_eq!(&public_bytes[2..6], &1_u32.to_le_bytes());
        assert_eq!(
            AggregationStatement::<Falcon512>::parse_canonical(&public_bytes).unwrap(),
            public
        );
        assert_eq!(
            AggregationWitness::<Falcon512>::parse_canonical(&witness_bytes).unwrap(),
            witness
        );
    }

    #[test]
    fn public_statement_reconstructs_the_prepared_witness_batch() {
        let records = (0..3)
            .flat_map(|index| statement(index).records().to_vec())
            .collect::<Vec<_>>();
        let source = ProductionStatement::new(records).unwrap();
        let (public, witness) = split_source_statement(&source).unwrap();
        let prepared = FalconPreparedWitness::from_aggregate(&public, &witness).unwrap();

        let reconstructed = public.public_batch();
        assert_eq!(reconstructed.live_rows, 3);
        assert_eq!(reconstructed.padded_rows, 4);
        let compact_h = prepared
            .witness()
            .h
            .iter()
            .map(|&coefficient| i64::from(coefficient))
            .collect::<Vec<_>>();
        assert_eq!(reconstructed.h, compact_h);
    }

    #[test]
    fn aggregate_wire_parser_rejects_malformed_inputs() {
        let source = statement(9);
        let (public, witness) = split_source_statement(&source).unwrap();
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

    fn statement(index: u8) -> ProductionStatement {
        statement_with_salt(index, index)
    }

    fn statement_with_salt(message_index: u8, salt_index: u8) -> ProductionStatement {
        const H: i16 = 111;
        let mut public_coefficients = [0_u16; FALCON_DEGREE];
        public_coefficients[0] = H as u16;
        let public_key =
            CanonicalPublicKeyBytes::<Falcon512>::parse(&encode_public(&public_coefficients))
                .unwrap();
        let mut salt = [0_u8; FALCON_512_SALT_BYTES];
        salt[0] = salt_index;
        let message = vec![message_index];
        let zero_signature = CanonicalSignatureBytes::<Falcon512>::parse(&encode_signature(
            &salt,
            &[0; FALCON_DEGREE],
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
        let record = DirectRecord::<Falcon512>::new(public_key, signature, message);
        ProductionStatement::new(vec![record]).unwrap()
    }

    fn encode_public(coefficients: &[u16; FALCON_DEGREE]) -> [u8; FALCON_512_PUBLIC_KEY_BYTES] {
        let mut output = [0_u8; FALCON_512_PUBLIC_KEY_BYTES];
        output[0] = 0x09;
        let mut bit = 0;
        for &coefficient in coefficients {
            write_bits(&mut output[1..], &mut bit, coefficient, 14);
        }
        output
    }

    fn encode_signature(
        salt: &[u8; FALCON_512_SALT_BYTES],
        coefficients: &[i16; FALCON_DEGREE],
    ) -> [u8; FALCON_512_SIGNATURE_BYTES] {
        let mut output = [0_u8; FALCON_512_SIGNATURE_BYTES];
        output[0] = 0x39;
        output[1..=FALCON_512_SALT_BYTES].copy_from_slice(salt);
        let mut bit = 0;
        for &coefficient in coefficients {
            write_bit(
                &mut output[1 + FALCON_512_SALT_BYTES..],
                &mut bit,
                coefficient < 0,
            );
            let magnitude = coefficient.unsigned_abs();
            write_bits(
                &mut output[1 + FALCON_512_SALT_BYTES..],
                &mut bit,
                magnitude & 127,
                7,
            );
            for _ in 0..magnitude >> 7 {
                write_bit(&mut output[1 + FALCON_512_SALT_BYTES..], &mut bit, false);
            }
            write_bit(&mut output[1 + FALCON_512_SALT_BYTES..], &mut bit, true);
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

    #[test]
    fn constants_match_falcon_modulus() {
        assert_eq!(FALCON_Q, 12_289);
    }

    #[test]
    fn production_encoding_is_records_only() {
        let production = statement(7);
        let encoding = production.canonical_encoding().unwrap();
        assert_eq!(&encoding[..2], &0_u16.to_le_bytes());
        assert_eq!(&encoding[2..6], &1_u32.to_le_bytes());
        assert_eq!(encoding[6], 1);
    }
}
