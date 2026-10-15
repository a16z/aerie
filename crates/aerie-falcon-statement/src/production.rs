//! Canonical Falcon record ingestion for Aerie.
//!
//! Canonical byte parsing remains verifier-local preprocessing, while the
//! dense witness is derived only from source-bound records accepted by the
//! native Aerie verifier.

use crate::DirectError;
use aerie_falcon_native::{source::CanonicalPublicKeyBytes, FalconParams};

/// Maximum canonical production batch. This matches Falcon's largest
/// generated runtime schedule (`2^16` records).
pub const MAX_PRODUCTION_BATCH_SIZE: usize = 1 << 16;
pub const MAX_MESSAGE_BYTES: usize = 1 << 24;

/// One public aggregate-verification tuple. The FN-DSA signature is supplied
/// separately by the private witness; its public salt is extracted into the
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
    public_keys: Vec<CanonicalPublicKeyBytes<P>>,
    messages: Vec<Vec<u8>>,
}

impl<P: FalconParams> AggregationStatement<P> {
    pub fn new(records: Vec<AggregationRecord<P>>) -> Result<Self, ProductionError> {
        validate_batch(records.len())?;
        let (public_keys, messages) = records
            .into_iter()
            .map(|record| (record.public_key, record.message))
            .unzip();
        Ok(Self {
            public_keys,
            messages,
        })
    }

    #[must_use]
    pub fn public_keys(&self) -> &[CanonicalPublicKeyBytes<P>] {
        &self.public_keys
    }

    #[must_use]
    pub fn messages(&self) -> &[Vec<u8>] {
        &self.messages
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.public_keys.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.public_keys.is_empty()
    }

    /// Clone the ordered records for callers that need to edit or subset them.
    #[must_use]
    pub fn to_records(&self) -> Vec<AggregationRecord<P>> {
        self.public_keys
            .iter()
            .zip(&self.messages)
            .map(|(public_key, message)| AggregationRecord {
                public_key: public_key.clone(),
                message: message.clone(),
            })
            .collect()
    }

    pub fn canonical_encoding(&self) -> Result<Vec<u8>, ProductionError> {
        public_canonical_encoding(&self.public_keys, &self.messages)
    }

    /// Parse the version-1 public statement with allocation limits and exact
    /// input consumption.
    pub fn parse_canonical(bytes: &[u8]) -> Result<Self, ProductionError> {
        parse_public_statement::<P>(bytes)
    }
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

#[doc(hidden)]
pub struct WireReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> WireReader<'a> {
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    pub fn take(&mut self, length: usize) -> Result<&'a [u8], ProductionError> {
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

    pub fn u16(&mut self) -> Result<u16, ProductionError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("two-byte slice"),
        ))
    }

    fn u32(&mut self) -> Result<u32, ProductionError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four-byte slice"),
        ))
    }

    pub fn count(&mut self, field: &'static str, maximum: usize) -> Result<usize, ProductionError> {
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

    pub fn finish(self) -> Result<(), ProductionError> {
        if self.position == self.bytes.len() {
            Ok(())
        } else {
            Err(ProductionError::MalformedWire(
                "trailing bytes after aggregate input".to_owned(),
            ))
        }
    }
}

fn public_canonical_encoding<P: FalconParams>(
    public_keys: &[CanonicalPublicKeyBytes<P>],
    messages: &[Vec<u8>],
) -> Result<Vec<u8>, ProductionError> {
    let mut output = Vec::new();
    output.extend_from_slice(&1_u16.to_le_bytes());
    output.extend_from_slice(&length_u32("record count", public_keys.len())?.to_le_bytes());
    for (public_key, message) in public_keys.iter().zip(messages) {
        output.push(1);
        append_bytes(&mut output, "public key", public_key.as_bytes().as_ref())?;
        append_bytes(&mut output, "raw message", message)?;
    }
    Ok(output)
}

#[doc(hidden)]
pub fn public_canonical_encoding_len<P: FalconParams>(
    messages: &[Vec<u8>],
) -> Result<usize, ProductionError> {
    length_u32("record count", messages.len())?;
    let add = |length: usize, extra: usize| {
        length
            .checked_add(extra)
            .ok_or(ProductionError::IntegerOverflow(
                "public statement encoding",
            ))
    };
    let mut length = add(2, 4)?;
    for message in messages {
        length_u32("raw message", message.len())?;
        length = add(length, 1)?;
        length = add(length, 4)?;
        length = add(length, P::PUBLIC_KEY_BYTES)?;
        length = add(length, 4)?;
        length = add(length, message.len())?;
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

#[doc(hidden)]
pub fn length_u32(name: &'static str, length: usize) -> Result<u32, ProductionError> {
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
    use super::*;

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
}
