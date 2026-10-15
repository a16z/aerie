//! Canonical wire codec for version 1 aggregate proofs.
//!
//! The encoding is strictly shaped by the public statement: every table
//! length, round count, and claim count is derived from the record count
//! and the parameter set, per Section 10.2. The decoder takes the
//! verifier's parameter set and rejects an unknown version, an unknown or
//! mismatched parameter-set byte (both checked before the record count is
//! read), an invalid record count, a wrong round
//! or coefficient count, and any trailing data.
//!
//! The 27-byte PCS frame (`aerie/falcon/v1p`) carries transported salts, the
//! Akita commitments, clear projection outputs, the PIOP frames, and the
//! batched opening. The common-salt PCS frame (`aerie/falcon/v1c`) carries
//! one shared salt and the same proof material. The projection outputs `(y, z)` sit directly after the
//! commitments, their transcript position, as one self-delimiting Rice
//! section (`projection::codec`, `specs/falcon.md` Section 10.4): at most
//! 1,471 bytes for any decodable input, at most 904 for outputs that pass
//! the predicates, and about 610 to 770 bytes on honest proofs. The
//! transcript absorbs the canonical Rice section bytes, including parameters
//! and final zero padding, without event headers. Registry metadata is local;
//! the statement shape and canonical codecs determine payload boundaries.

use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Validate};
use akita_types::Commitment;
use jolt_field::Zero as _;

use crate::{AggregationError, MalformedProof, StatementMismatch, UnsupportedBatch};

use crate::substrate::{BaseField, OpeningField as E};

use super::grinding::PiopGrindingPlan;
use super::manifest::CommitmentManifest;
use super::pcs::{commitment_coefficient_counts_of, AggregateProof};
use super::stage::StageShapes;
use super::{FalconDegree, MAX_RECORDS};

mod piop;

use piop::{read_piop_frames, write_piop_frames, PiopFramesRef};

/// Frame magic of the real-commitment wire proof carrying the transport and
/// PCS payloads.
const MAGIC_PCS: &[u8; 16] = b"aerie/falcon/v1p";
/// Frame magic for PCS proofs with one shared salt.
const MAGIC_PCS_COMMON: &[u8; 16] = b"aerie/falcon/v1c";
/// Known frame kinds are rejected explicitly when passed to another codec.
const KNOWN_MAGICS: &[&[u8; 16]] = &[MAGIC_PCS, MAGIC_PCS_COMMON];
/// Upper bound on the PCS section's length prefix. Akita's batched opening
/// is a Spongefish argument string whose exact length the pinned Akita does
/// not expose as a function of the statement, so the decoder bounds it
/// instead of deriving it. Honest proofs at the supported shapes are about
/// 63 KiB; the bound leaves headroom for other record counts while keeping
/// a hostile prefix from reaching Akita's parser.
pub const MAX_PCS_PROOF_BYTES: usize = 1 << 20;
/// Per-record salt length of the transport frame.
pub const TRANSPORT_SALT_BYTES: usize = aerie_falcon_native::params::FALCON_SALT_BYTES;

/// Salt transport and public-record binding mode.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SaltMode {
    #[default]
    PerRecord,
    Common,
}

impl SaltMode {
    fn magic(self) -> &'static [u8; 16] {
        match self {
            Self::PerRecord => MAGIC_PCS,
            Self::Common => MAGIC_PCS_COMMON,
        }
    }

    #[doc(hidden)]
    pub fn validate<'a>(
        self,
        salts: impl IntoIterator<Item = &'a [u8; TRANSPORT_SALT_BYTES]>,
    ) -> Result<(), AggregationError> {
        if self == Self::Common {
            let mut salts = salts.into_iter();
            let first = salts.next();
            if first.is_none() || salts.any(|salt| Some(salt) != first) {
                return Err(AggregationError::UnsupportedBatch(
                    UnsupportedBatch::SaltMode,
                ));
            }
        }
        Ok(())
    }
}

/// Transported salts, retaining one array for a common-salt frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SaltFrame {
    Common([u8; TRANSPORT_SALT_BYTES]),
    PerRecord(Vec<[u8; TRANSPORT_SALT_BYTES]>),
}

impl SaltFrame {
    /// Borrow the logical record salts without expanding common storage.
    #[must_use]
    pub fn view(&self) -> SaltView<'_> {
        match self {
            Self::Common(salt) => SaltView::Common(salt),
            Self::PerRecord(salts) => SaltView::PerRecord(salts),
        }
    }
}

/// Borrowed salts for logical records in a transport frame.
#[derive(Clone, Copy, Debug)]
pub enum SaltView<'a> {
    Common(&'a [u8; TRANSPORT_SALT_BYTES]),
    PerRecord(&'a [[u8; TRANSPORT_SALT_BYTES]]),
}

impl<'a> SaltView<'a> {
    /// The transport and binding mode encoded by this view.
    #[must_use]
    pub fn mode(self) -> SaltMode {
        match self {
            Self::Common(_) => SaltMode::Common,
            Self::PerRecord(_) => SaltMode::PerRecord,
        }
    }

    /// Salt for a logical record. The caller validates the record count.
    #[must_use]
    pub fn record(self, index: usize) -> &'a [u8; TRANSPORT_SALT_BYTES] {
        match self {
            Self::Common(salt) => salt,
            Self::PerRecord(salts) => &salts[index],
        }
    }

    #[cfg(any(test, feature = "internals"))]
    pub(super) fn with_mode(
        salts: &'a [[u8; TRANSPORT_SALT_BYTES]],
        mode: SaltMode,
    ) -> Result<Self, AggregationError> {
        mode.validate(salts)?;
        Ok(match mode {
            SaltMode::Common => Self::Common(&salts[0]),
            SaltMode::PerRecord => Self::PerRecord(salts),
        })
    }
}

/// Sumcheck round and terminal counts per stage, derived from the parameter
/// set and the live record count.
#[doc(hidden)]
pub struct WireShape {
    pub rounds: usize,
    pub stages: StageShapes,
    group_count: usize,
    pub grinding: PiopGrindingPlan,
}
#[doc(hidden)]
pub fn stage_shapes(degree: FalconDegree, records: usize) -> Result<WireShape, AggregationError> {
    let manifest = CommitmentManifest::for_batch_of(degree, records).map_err(|_| {
        AggregationError::MalformedProof(MalformedProof::InvalidRecordCount { records })
    })?;
    let grinding =
        PiopGrindingPlan::for_shape(degree, manifest.padded_records()).map_err(|_| {
            AggregationError::UnsupportedBatch(UnsupportedBatch::Shape {
                degree: degree.degree(),
                records,
            })
        })?;
    Ok(WireShape {
        rounds: grinding.rounds,
        stages: StageShapes::for_degree(degree),
        group_count: manifest.group_count(),
        grinding,
    })
}

#[doc(hidden)]
pub fn write_value(
    buffer: &mut Vec<u8>,
    value: &impl AkitaSerialize,
) -> Result<(), AggregationError> {
    value
        .serialize_with_mode(&mut *buffer, Compress::Yes)
        .map_err(AggregationError::backend)?;
    Ok(())
}

fn write_ext_values(buffer: &mut Vec<u8>, values: &[E]) -> Result<(), AggregationError> {
    for value in values {
        write_value(buffer, value)?;
    }
    Ok(())
}

fn decode_field_error(error: akita_serialization::SerializationError) -> AggregationError {
    use akita_serialization::SerializationError;
    let kind = match error {
        SerializationError::IoError(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
            MalformedProof::Truncated
        }
        SerializationError::IoError(_) | SerializationError::UnexpectedData => {
            MalformedProof::NonCanonicalEncoding
        }
        SerializationError::InvalidData(_) => MalformedProof::NonCanonicalFieldElement,
        SerializationError::LengthLimitExceeded { .. } => MalformedProof::NonCanonicalLength,
    };
    AggregationError::MalformedProof(kind)
}

fn decode_commitment_error(error: akita_serialization::SerializationError) -> AggregationError {
    match error {
        akita_serialization::SerializationError::InvalidData(_) => {
            AggregationError::MalformedProof(MalformedProof::CommitmentShape)
        }
        error => decode_field_error(error),
    }
}

fn read_value<T: AkitaDeserialize<Context = ()>>(
    reader: &mut &[u8],
) -> Result<T, AggregationError> {
    T::deserialize_with_mode(&mut *reader, Compress::Yes, Validate::Yes, &())
        .map_err(decode_field_error)
}

#[doc(hidden)]
pub fn read_ext_array<const N: usize>(reader: &mut &[u8]) -> Result<[E; N], AggregationError> {
    let mut values = [E::zero(); N];
    for value in &mut values {
        *value = read_value(reader)?;
    }
    Ok(values)
}

/// Decoded statement header, transport frame, and real-commitment proof.
///
/// The transport frame carries the `HashToPoint` salts, retaining a single
/// array for common-salt frames.
/// Public keys and messages are statement inputs the verifier already
/// holds; they are bound through the public-record digest, never
/// transported.
#[derive(Clone, Debug)]
pub struct DecodedAggregateProof {
    /// Header mode for internals callers; the salt view derives its mode.
    #[cfg(any(test, feature = "internals"))]
    pub salt_mode: SaltMode,
    #[cfg(any(test, feature = "internals"))]
    pub degree: FalconDegree,
    pub records: usize,
    /// Transported `HashToPoint` salts, live rows only.
    pub salts: SaltFrame,
    pub proof: AggregateProof,
}

#[doc(hidden)]
pub fn read_shaped<T: AkitaDeserialize>(
    reader: &mut &[u8],
    context: &T::Context,
) -> Result<T, AggregationError> {
    T::deserialize_with_mode(&mut *reader, Compress::Yes, Validate::Yes, context)
        .map_err(decode_commitment_error)
}

/// Byte offset of the little-endian `u16` protocol version.
pub const HEADER_VERSION_OFFSET: usize = 16;
/// Byte offset of the parameter-set byte (`log2 n`: 9 or 10).
pub const HEADER_PARAMETER_SET_OFFSET: usize = HEADER_VERSION_OFFSET + 2;
/// Byte offset of the little-endian `u64` live record count.
pub const HEADER_RECORDS_OFFSET: usize = HEADER_PARAMETER_SET_OFFSET + 1;
/// Header length; the frame body starts here.
pub const HEADER_BYTES: usize = HEADER_RECORDS_OFFSET + 8;

/// Header: `magic (16) || u16 LE version || u8 parameter set || u64 LE
/// records`, 27 bytes. Both parameter
/// sets share the magic and the version; the parameter-set byte is `log2 n`.
fn write_header(buffer: &mut Vec<u8>, magic: &[u8; 16], degree: FalconDegree, records: usize) {
    buffer.extend_from_slice(magic);
    buffer.extend_from_slice(&degree.identity_bytes());
    buffer.extend_from_slice(&(records as u64).to_le_bytes());
}

/// Read and check the header against the verifier's parameter set. The
/// magic, the version, and the parameter-set byte are checked, in that
/// order, before the record count or anything after it is parsed.
#[doc(hidden)]
pub fn read_header(
    reader: &mut &[u8],
    magic: &[u8; 16],
    expected: FalconDegree,
) -> Result<usize, AggregationError> {
    if reader.len() < HEADER_BYTES {
        return Err(AggregationError::MalformedProof(MalformedProof::Truncated));
    }
    if &reader[..magic.len()] != magic {
        let other_mode = KNOWN_MAGICS
            .iter()
            .any(|known| *known != magic && reader[..magic.len()] == known[..]);
        return Err(AggregationError::MalformedProof(if other_mode {
            MalformedProof::FrameKind
        } else {
            MalformedProof::Magic
        }));
    }
    let header = &reader[..HEADER_BYTES];
    let version = u16::from_le_bytes(
        header[HEADER_VERSION_OFFSET..HEADER_PARAMETER_SET_OFFSET]
            .try_into()
            .expect("checked length"),
    );
    if version != crate::WIRE_VERSION {
        return Err(AggregationError::MalformedProof(
            MalformedProof::UnknownVersion { version },
        ));
    }
    let parameter_set = FalconDegree::from_parameter_set(header[HEADER_PARAMETER_SET_OFFSET])
        .map_err(|_| {
            AggregationError::MalformedProof(MalformedProof::UnknownParameterSet {
                parameter_set: header[HEADER_PARAMETER_SET_OFFSET],
            })
        })?;
    if parameter_set != expected {
        return Err(AggregationError::StatementMismatch(
            StatementMismatch::ParameterSet {
                expected: expected.parameter_set(),
                actual: parameter_set.parameter_set(),
            },
        ));
    }
    let records = u64::from_le_bytes(
        header[HEADER_RECORDS_OFFSET..HEADER_BYTES]
            .try_into()
            .expect("checked length"),
    );
    let records = usize::try_from(records)
        .map_err(|_| AggregationError::MalformedProof(MalformedProof::LengthOverflow))?;
    if records == 0 || records > MAX_RECORDS {
        return Err(AggregationError::MalformedProof(
            MalformedProof::InvalidRecordCount { records },
        ));
    }
    *reader = &reader[HEADER_BYTES..];
    Ok(records)
}

const _: () =
    assert!(HEADER_VERSION_OFFSET == MAGIC_PCS.len() && MAGIC_PCS.len() == MAGIC_PCS_COMMON.len());
const _: () = assert!(HEADER_BYTES == 27);

/// The transport frame is one common salt or `records` fixed-width salts,
/// with no length words. Messages are not transported; the verifier
/// supplies them alongside the public keys and the public-record digest
/// binds both.
#[cfg(test)]
fn write_transport_frame(
    buffer: &mut Vec<u8>,
    mode: SaltMode,
    records: usize,
    salts: &[[u8; TRANSPORT_SALT_BYTES]],
) -> Result<(), AggregationError> {
    validate_salt_count(records, salts.len())?;
    write_transport_view(buffer, records, SaltView::with_mode(salts, mode)?)
}

fn validate_salt_count(records: usize, actual: usize) -> Result<(), AggregationError> {
    if actual != records {
        return Err(StatementMismatch::SaltCount {
            expected: records,
            actual,
        }
        .into());
    }
    Ok(())
}

fn write_transport_view(
    buffer: &mut Vec<u8>,
    records: usize,
    salts: SaltView<'_>,
) -> Result<(), AggregationError> {
    match salts {
        SaltView::Common(salt) => {
            if records == 0 {
                return Err(UnsupportedBatch::SaltMode.into());
            }
            buffer.extend_from_slice(salt);
        }
        SaltView::PerRecord(salts) => {
            validate_salt_count(records, salts.len())?;
            for salt in salts {
                buffer.extend_from_slice(salt);
            }
        }
    }
    Ok(())
}

#[doc(hidden)]
pub fn read_transport_frame(
    reader: &mut &[u8],
    mode: SaltMode,
    records: usize,
) -> Result<SaltFrame, AggregationError> {
    let transported = if mode == SaltMode::Common { 1 } else { records };
    let transported_bytes =
        transported
            .checked_mul(TRANSPORT_SALT_BYTES)
            .ok_or(AggregationError::MalformedProof(
                MalformedProof::LengthOverflow,
            ))?;
    if reader.len() < transported_bytes {
        return Err(AggregationError::MalformedProof(MalformedProof::Truncated));
    }
    let mut read_salt = || {
        let mut salt = [0_u8; TRANSPORT_SALT_BYTES];
        salt.copy_from_slice(&reader[..TRANSPORT_SALT_BYTES]);
        *reader = &reader[TRANSPORT_SALT_BYTES..];
        salt
    };
    Ok(match mode {
        SaltMode::Common => SaltFrame::Common(read_salt()),
        SaltMode::PerRecord => {
            let mut salts = Vec::with_capacity(records);
            for _ in 0..records {
                salts.push(read_salt());
            }
            SaltFrame::PerRecord(salts)
        }
    })
}

/// Encode a real-commitment proof into canonical, statement-shaped bytes:
/// the transport frame and the commitment payloads carry no length prefixes,
/// and the decoder derives their sizes from the record count and the
/// generated catalogs. The final PCS section is `u64 LE length || Akita
/// argument string`; the length must be nonzero, at most
/// [`MAX_PCS_PROOF_BYTES`], and exactly the rest of the input. The wire proof carries the salts and the proof
/// material only; keys and messages are verifier inputs. The header's
/// parameter-set byte carries `degree`.
#[cfg(any(test, feature = "internals"))]
pub fn encode_pcs_proof(
    degree: FalconDegree,
    records: usize,
    salts: &[[u8; TRANSPORT_SALT_BYTES]],
    proof: &AggregateProof,
) -> Result<Vec<u8>, AggregationError> {
    encode_pcs_proof_with_mode(degree, records, salts, proof, SaltMode::PerRecord)
}

/// Encode a PCS proof with an explicit salt mode. The proof must have been
/// produced with the corresponding public-record binding.
#[cfg(any(test, feature = "internals"))]
pub fn encode_pcs_proof_with_mode(
    degree: FalconDegree,
    records: usize,
    salts: &[[u8; TRANSPORT_SALT_BYTES]],
    proof: &AggregateProof,
    mode: SaltMode,
) -> Result<Vec<u8>, AggregationError> {
    validate_salt_count(records, salts.len())?;
    encode_pcs_proof_from_view(degree, records, SaltView::with_mode(salts, mode)?, proof)
}

/// Re-encode a decoded salt frame without expanding common salts.
#[cfg(any(test, feature = "internals"))]
pub fn encode_pcs_proof_from_salt_frame(
    degree: FalconDegree,
    records: usize,
    salts: &SaltFrame,
    proof: &AggregateProof,
) -> Result<Vec<u8>, AggregationError> {
    encode_pcs_proof_from_view(degree, records, salts.view(), proof)
}

#[doc(hidden)]
pub fn encode_pcs_proof_from_view(
    degree: FalconDegree,
    records: usize,
    salts: SaltView<'_>,
    proof: &AggregateProof,
) -> Result<Vec<u8>, AggregationError> {
    let mut buffer = Vec::new();
    write_header(&mut buffer, salts.mode().magic(), degree, records);
    write_transport_view(&mut buffer, records, salts)?;
    write_proof_body(&mut buffer, degree, records, proof)?;
    Ok(buffer)
}

/// Decode a real-commitment proof of the expected parameter set; reject a
/// header naming another parameter set and anything noncanonical. The
/// parameter set is checked before the record count is parsed.
pub fn decode_pcs_proof(
    degree: FalconDegree,
    bytes: &[u8],
) -> Result<DecodedAggregateProof, AggregationError> {
    let (salt_mode, records) = peek_pcs_proof_header(degree, bytes)?;
    let mut reader = &bytes[HEADER_BYTES..];
    let salts = read_transport_frame(&mut reader, salt_mode, records)?;
    let proof = read_proof_body(&mut reader, degree, records)?;
    Ok(DecodedAggregateProof {
        #[cfg(any(test, feature = "internals"))]
        salt_mode,
        #[cfg(any(test, feature = "internals"))]
        degree,
        records,
        salts,
        proof,
    })
}

/// Check the fixed-width PCS header without decoding or allocating for the
/// proof body. The returned record count has already been bounded.
#[doc(hidden)]
pub fn peek_pcs_proof_header(
    degree: FalconDegree,
    bytes: &[u8],
) -> Result<(SaltMode, usize), AggregationError> {
    let salt_mode = if bytes.starts_with(MAGIC_PCS_COMMON) {
        SaltMode::Common
    } else {
        SaltMode::PerRecord
    };
    let mut reader = bytes;
    let records = read_header(&mut reader, salt_mode.magic(), degree)?;
    Ok((salt_mode, records))
}

/// The proof body: the statement-shaped group commitments and PIOP frames,
/// then the length-prefixed batched PCS argument string.
fn write_proof_body(
    buffer: &mut Vec<u8>,
    degree: FalconDegree,
    records: usize,
    proof: &AggregateProof,
) -> Result<(), AggregationError> {
    let shapes = stage_shapes(degree, records)?;
    if proof.commitments.len() != shapes.group_count {
        return Err(AggregationError::MalformedProof(
            MalformedProof::CommitmentShape,
        ));
    }
    for commitment in &proof.commitments {
        write_value(buffer, commitment)?;
    }
    write_piop_frames(
        buffer,
        &shapes,
        PiopFramesRef {
            outputs: &proof.projection_outputs,
            stage1: &proof.stage1,
            stage2: &proof.stage2,
            stage3: &proof.stage3,
            work: &proof.piop_work,
        },
    )?;
    if proof.pcs.is_empty() || proof.pcs.len() > MAX_PCS_PROOF_BYTES {
        return Err(AggregationError::MalformedProof(
            MalformedProof::NonCanonicalLength,
        ));
    }
    buffer.extend_from_slice(&(proof.pcs.len() as u64).to_le_bytes());
    buffer.extend_from_slice(&proof.pcs);
    Ok(())
}

/// Read the proof body to the end of the input. The PCS length prefix must
/// be nonzero, within [`MAX_PCS_PROOF_BYTES`], and equal to the remaining
/// input, so trailing data and truncation are both rejected.
fn read_proof_body(
    reader: &mut &[u8],
    degree: FalconDegree,
    records: usize,
) -> Result<AggregateProof, AggregationError> {
    let coefficient_counts = commitment_coefficient_counts_of(degree, records)?;
    let shapes = stage_shapes(degree, records)?;
    if coefficient_counts.len() != shapes.group_count {
        return Err(AggregationError::MalformedProof(
            MalformedProof::CommitmentShape,
        ));
    }
    let mut commitments = Vec::with_capacity(shapes.group_count);
    for count in &coefficient_counts {
        commitments.push(read_shaped::<Commitment<BaseField>>(reader, count)?);
    }
    let frames = read_piop_frames(reader, &shapes)?;
    let pcs = read_pcs_section(reader)?;
    Ok(AggregateProof {
        commitments,
        projection_outputs: frames.outputs,
        stage1: frames.stage1,
        stage2: frames.stage2,
        stage3: frames.stage3,
        piop_work: frames.work,
        pcs,
    })
}

/// Read the length-prefixed PCS section, which must end the input.
fn read_pcs_section(reader: &mut &[u8]) -> Result<Vec<u8>, AggregationError> {
    if reader.len() < 8 {
        return Err(AggregationError::MalformedProof(MalformedProof::Truncated));
    }
    let length = u64::from_le_bytes(reader[..8].try_into().expect("checked length"));
    *reader = &reader[8..];
    let length = usize::try_from(length)
        .ok()
        .filter(|length| (1..=MAX_PCS_PROOF_BYTES).contains(length))
        .ok_or(AggregationError::MalformedProof(
            MalformedProof::NonCanonicalLength,
        ))?;
    match reader.len().cmp(&length) {
        core::cmp::Ordering::Less => {
            return Err(AggregationError::MalformedProof(MalformedProof::Truncated))
        }
        core::cmp::Ordering::Greater => {
            return Err(AggregationError::MalformedProof(
                MalformedProof::TrailingBytes,
            ))
        }
        core::cmp::Ordering::Equal => {}
    }
    let pcs = reader.to_vec();
    *reader = &[];
    Ok(pcs)
}

/// Narrow codec hooks used by producer-owned wire regression tests.
#[cfg(feature = "internals")]
#[doc(hidden)]
pub mod test_support {
    use super::{piop, AggregationError};
    pub fn take_work_nonce<'a>(
        reader: &mut &'a [u8],
        difficulty: u32,
    ) -> Result<&'a [u8], AggregationError> {
        piop::take_work_nonce(reader, difficulty)
    }
    pub const MAGIC_PCS: &[u8; 16] = super::MAGIC_PCS;
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::pcs;
    use crate::protocol::ring::PublicTargets;
    use aerie_falcon_native::{Falcon1024, Falcon512, FalconParams};

    const DEGREES: [FalconDegree; 2] = [FalconDegree::Falcon512, FalconDegree::Falcon1024];

    fn other(degree: FalconDegree) -> FalconDegree {
        match degree {
            FalconDegree::Falcon512 => FalconDegree::Falcon1024,
            FalconDegree::Falcon1024 => FalconDegree::Falcon512,
        }
    }

    #[test]
    fn common_salt_transport_is_exactly_one_salt() {
        let salts = vec![[19; TRANSPORT_SALT_BYTES]; 5];
        let mut bytes = Vec::new();
        write_transport_frame(&mut bytes, SaltMode::Common, 5, &salts).unwrap();
        assert_eq!(bytes, salts[0]);
        let mut reader = bytes.as_slice();
        assert_eq!(
            read_transport_frame(&mut reader, SaltMode::Common, 5).unwrap(),
            SaltFrame::Common(salts[0])
        );
        assert!(reader.is_empty());
        assert!(matches!(
            read_transport_frame(&mut &bytes[..39], SaltMode::Common, 5),
            Err(AggregationError::MalformedProof(MalformedProof::Truncated))
        ));
        assert!(matches!(
            write_transport_frame(&mut Vec::new(), SaltMode::Common, 0, &[]),
            Err(AggregationError::UnsupportedBatch(
                UnsupportedBatch::SaltMode
            ))
        ));
        assert!(matches!(
            write_transport_frame(&mut Vec::new(), SaltMode::Common, 4, &salts),
            Err(AggregationError::StatementMismatch(
                StatementMismatch::SaltCount {
                    expected: 4,
                    actual: 5
                }
            ))
        ));
        let mut unequal = salts;
        unequal[4][39] ^= 1;
        assert!(matches!(
            write_transport_frame(&mut Vec::new(), SaltMode::Common, 5, &unequal),
            Err(AggregationError::UnsupportedBatch(
                UnsupportedBatch::SaltMode
            ))
        ));
    }

    #[test]
    fn header_layout_and_record_bounds_are_pinned() {
        for degree in DEGREES {
            let mut header = Vec::new();
            write_header(&mut header, MAGIC_PCS, degree, 0x0102);
            assert_eq!(header.len(), HEADER_BYTES);
            assert_eq!(&header[..HEADER_VERSION_OFFSET], MAGIC_PCS);
            assert_eq!(
                header[HEADER_VERSION_OFFSET..HEADER_PARAMETER_SET_OFFSET],
                crate::WIRE_VERSION.to_le_bytes()
            );
            assert_eq!(header[HEADER_PARAMETER_SET_OFFSET], degree.logn() as u8);
            assert_eq!(
                header[HEADER_RECORDS_OFFSET..HEADER_BYTES],
                0x0102_u64.to_le_bytes()
            );

            let mut maximum = Vec::new();
            write_header(&mut maximum, MAGIC_PCS, degree, MAX_RECORDS);
            let records = read_header(&mut maximum.as_slice(), MAGIC_PCS, degree)
                .expect("maximum record header");
            assert_eq!(records, MAX_RECORDS);
            let mut above_maximum = Vec::new();
            write_header(&mut above_maximum, MAGIC_PCS, degree, MAX_RECORDS + 1);
            let result = read_header(&mut above_maximum.as_slice(), MAGIC_PCS, degree);
            let rejected = matches!(
                result,
                Err(AggregationError::MalformedProof(
                    MalformedProof::InvalidRecordCount { records }
                )) if records == MAX_RECORDS + 1
            );
            assert!(rejected, "{result:?}");
        }
    }

    /// The committed fuzz seeds decode under their own parameter set only and
    /// re-encode byte for byte in their own salt mode.
    #[test]
    fn committed_fuzz_seeds_roundtrip_in_their_salt_mode() {
        let seeds: [(FalconDegree, SaltMode, &[u8]); 4] = [
            (
                FalconDegree::Falcon512,
                SaltMode::PerRecord,
                include_bytes!("../../fuzz/seeds/wire_pcs_proof/falcon512-v1p"),
            ),
            (
                FalconDegree::Falcon512,
                SaltMode::Common,
                include_bytes!("../../fuzz/seeds/wire_pcs_proof/falcon512-v1c"),
            ),
            (
                FalconDegree::Falcon1024,
                SaltMode::PerRecord,
                include_bytes!("../../fuzz/seeds/wire_pcs_proof/falcon1024-v1p"),
            ),
            (
                FalconDegree::Falcon1024,
                SaltMode::Common,
                include_bytes!("../../fuzz/seeds/wire_pcs_proof/falcon1024-v1c"),
            ),
        ];
        for (degree, mode, bytes) in seeds {
            let other = match degree {
                FalconDegree::Falcon512 => FalconDegree::Falcon1024,
                FalconDegree::Falcon1024 => FalconDegree::Falcon512,
            };
            assert!(matches!(
                decode_pcs_proof(other, bytes),
                Err(AggregationError::StatementMismatch(
                    StatementMismatch::ParameterSet { .. }
                ))
            ));
            let decoded = decode_pcs_proof(degree, bytes).expect("seed decodes");
            assert_eq!(decoded.salt_mode, mode);
            let reencoded = encode_pcs_proof_from_salt_frame(
                decoded.degree,
                decoded.records,
                &decoded.salts,
                &decoded.proof,
            )
            .expect("seed re-encodes");
            assert_eq!(reencoded.as_slice(), bytes);
        }
    }

    /// Validated commitments serialize to exactly their consumed wire bytes.
    /// The pinned `Commitment` -> `RingVec` -> `Fp64` codec uses fixed LE64 fields;
    /// dependency changes must re-audit this canonicality premise.
    #[test]
    fn commitment_wire_bytes_are_canonical() {
        let seeds: [(FalconDegree, SaltMode, &[u8]); 4] = [
            (
                FalconDegree::Falcon512,
                SaltMode::PerRecord,
                include_bytes!("../../fuzz/seeds/wire_pcs_proof/falcon512-v1p"),
            ),
            (
                FalconDegree::Falcon512,
                SaltMode::Common,
                include_bytes!("../../fuzz/seeds/wire_pcs_proof/falcon512-v1c"),
            ),
            (
                FalconDegree::Falcon1024,
                SaltMode::PerRecord,
                include_bytes!("../../fuzz/seeds/wire_pcs_proof/falcon1024-v1p"),
            ),
            (
                FalconDegree::Falcon1024,
                SaltMode::Common,
                include_bytes!("../../fuzz/seeds/wire_pcs_proof/falcon1024-v1c"),
            ),
        ];
        for (degree, mode, bytes) in seeds {
            let mut reader = bytes;
            let records = read_header(&mut reader, mode.magic(), degree).expect("seed header");
            read_transport_frame(&mut reader, mode, records).expect("seed salts");
            let counts = commitment_coefficient_counts_of(degree, records)
                .expect("statement-shaped coefficient counts");
            assert!(!counts.is_empty());
            for (group, count) in counts.into_iter().enumerate() {
                let original = reader;
                // read_shaped uses Compress::Yes and Validate::Yes.
                let commitment: Commitment<BaseField> =
                    read_shaped(&mut reader, &count).expect("validated seed commitment");
                let consumed = &original[..original.len() - reader.len()];
                assert!(count > 0);
                assert_eq!(consumed.len(), count * 8);
                let mut encoded = Vec::new();
                write_value(&mut encoded, &commitment).expect("commitment serialization");
                assert_eq!(encoded, consumed, "{degree:?} {mode:?} group {group}");

                for invalid in [crate::substrate::FIELD_MODULUS, u64::MAX] {
                    let mut mutated = consumed.to_vec();
                    mutated[..8].copy_from_slice(&invalid.to_le_bytes());
                    let result =
                        read_shaped::<Commitment<BaseField>>(&mut mutated.as_slice(), &count);
                    assert!(
                        matches!(
                            result,
                            Err(AggregationError::MalformedProof(
                                MalformedProof::CommitmentShape
                            ))
                        ),
                        "{degree:?} {mode:?} group {group}: coefficient {invalid} must reject"
                    );
                }
            }
        }
    }

    #[test]
    fn common_salt_storage_is_constant_and_targets_match() {
        use crate::protocol::public_records_binding::{
            public_records_binding_from_view, public_records_binding_with_mode,
        };
        use crate::protocol::ring::target_evaluation;
        use aerie_falcon_native::preprocess::{
            hash_to_point_prehashed_batch_visit_of, HashedVerifyingKey, PrehashedHashToPointBatch,
        };
        use aerie_falcon_native::source::CanonicalPublicKeyBytes;

        let salt = [23; TRANSPORT_SALT_BYTES];
        for degree in DEGREES {
            for records in [1, 129, MAX_RECORDS] {
                let mut bytes = Vec::new();
                write_header(&mut bytes, MAGIC_PCS_COMMON, degree, records);
                bytes.extend_from_slice(&salt);
                let (mode, count) = peek_pcs_proof_header(degree, &bytes).unwrap();
                let mut reader = &bytes[HEADER_BYTES..];
                let frame = read_transport_frame(&mut reader, mode, count).unwrap();
                assert!(matches!(frame, SaltFrame::Common(actual) if actual == salt));
                assert!(reader.is_empty());
                assert_eq!(frame.view().record(records - 1), &salt);
                let mut reencoded = Vec::new();
                write_transport_view(&mut reencoded, records, frame.view()).unwrap();
                assert_eq!(reencoded, salt);
            }
        }

        fn check<P: crate::protocol::AggregationParams>() {
            use jolt_field::Ring as _;
            let mut key_bytes = vec![0; P::PUBLIC_KEY_BYTES];
            key_bytes[0] = P::DEGREE_ID.logn() as u8;
            let key = CanonicalPublicKeyBytes::<P>::parse(&key_bytes).unwrap();
            let salt = [23; TRANSPORT_SALT_BYTES];
            for records in [1, 127, 128, 129] {
                let frame =
                    read_transport_frame(&mut salt.as_slice(), SaltMode::Common, records).unwrap();
                let SaltView::Common(shared) = frame.view() else {
                    panic!("common frame retains one salt");
                };
                let repeated = vec![salt; records];
                let hpks = vec![HashedVerifyingKey::derive(&key); records];
                let messages = (0..records)
                    .map(|record| vec![record as u8; record % 37])
                    .collect::<Vec<_>>();
                let common =
                    PrehashedHashToPointBatch::new_common(&hpks, shared, &messages).unwrap();
                let reference =
                    PrehashedHashToPointBatch::new(&hpks, &repeated, &messages).unwrap();
                let padded = P::DEGREE_ID.padded_records(records).unwrap();
                let point = (0..P::DEGREE_ID.logn() + padded.trailing_zeros() as usize)
                    .map(|index| E::from_u64(index as u64 * 17 + 3))
                    .collect::<Vec<_>>();
                let mut expanded = vec![0; padded * P::DEGREE];
                let mut visited = vec![false; records];
                hash_to_point_prehashed_batch_visit_of::<P, _>(reference, |record, row| {
                    assert!(!core::mem::replace(&mut visited[record], true));
                    expanded[record * P::DEGREE..][..P::DEGREE].copy_from_slice(row.as_ref());
                });
                assert!(visited.iter().all(|&seen| seen));
                let expected = target_evaluation(
                    P::DEGREE_ID,
                    padded,
                    records,
                    PublicTargets::Expanded(&expanded),
                    &point,
                );
                let expected_binding =
                    public_records_binding_with_mode(&hpks, &repeated, &messages, SaltMode::Common)
                        .unwrap();
                let compute = || {
                    assert_eq!(
                        public_records_binding_from_view(&hpks, frame.view(), &messages).unwrap(),
                        expected_binding,
                    );
                    assert_eq!(
                        target_evaluation(
                            P::DEGREE_ID,
                            padded,
                            records,
                            PublicTargets::<i16>::SourceBound(common),
                            &point,
                        ),
                        expected
                    );
                    assert_eq!(
                        target_evaluation(
                            P::DEGREE_ID,
                            padded,
                            records,
                            PublicTargets::<i16>::SourceBound(reference),
                            &point,
                        ),
                        expected
                    );
                };
                compute();
                #[cfg(feature = "parallel")]
                for workers in [1, 2, 4] {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(workers)
                        .build()
                        .unwrap()
                        .install(compute);
                }
            }
        }
        check::<Falcon512>();
        check::<Falcon1024>();
    }

    #[test]
    fn truncated_maximum_salt_frame_rejects_before_materializing_salts() {
        let mut bytes = Vec::new();
        write_header(&mut bytes, MAGIC_PCS, FalconDegree::Falcon512, MAX_RECORDS);
        let error = decode_pcs_proof(FalconDegree::Falcon512, &bytes)
            .expect_err("header-only maximum batch must reject");
        assert!(matches!(
            error,
            AggregationError::MalformedProof(MalformedProof::Truncated)
        ));
    }

    /// The parameter-set byte is checked against the verifier's parameter set
    /// before the record count or proof body is parsed.
    #[test]
    fn header_rejects_a_foreign_or_unknown_parameter_set_first() {
        for degree in DEGREES {
            for magic in KNOWN_MAGICS {
                let mut header = Vec::new();
                write_header(&mut header, magic, degree, 8);
                header[HEADER_RECORDS_OFFSET..HEADER_BYTES].copy_from_slice(&0_u64.to_le_bytes());
                let error = read_header(&mut header.as_slice(), magic, other(degree))
                    .expect_err("foreign parameter set");
                let expected = other(degree).parameter_set();
                let actual = degree.parameter_set();
                let mismatch = matches!(
                    error,
                    AggregationError::StatementMismatch(StatementMismatch::ParameterSet {
                        expected: found_expected,
                        actual: found_actual,
                    }) if (found_expected, found_actual) == (expected, actual)
                );
                assert!(mismatch, "{error:?}");
                for unknown in [0_u8, 8, 11, 0xff] {
                    let mut unknown_set = header.clone();
                    unknown_set[HEADER_PARAMETER_SET_OFFSET] = unknown;
                    let error = read_header(&mut unknown_set.as_slice(), magic, degree)
                        .expect_err("unknown parameter set");
                    let malformed = matches!(
                        error,
                        AggregationError::MalformedProof(
                            MalformedProof::UnknownParameterSet { parameter_set }
                        ) if parameter_set == unknown
                    );
                    assert!(malformed, "{error:?}");
                }
                for version in [0_u16, 2, 40, 42, 0x0100] {
                    let mut wrong_version = header.clone();
                    wrong_version[HEADER_VERSION_OFFSET..HEADER_PARAMETER_SET_OFFSET]
                        .copy_from_slice(&version.to_le_bytes());
                    let error = read_header(&mut wrong_version.as_slice(), magic, degree)
                        .expect_err("unsupported version");
                    let malformed = matches!(
                        error,
                        AggregationError::MalformedProof(MalformedProof::UnknownVersion {
                            version: actual
                        }) if actual == version
                    );
                    assert!(malformed, "{error:?}");
                }
            }
        }
    }

    #[test]
    fn malformed_and_mismatched_headers_have_precise_kinds() {
        let bytes = include_bytes!("../../fuzz/seeds/wire_pcs_proof/falcon512-v1p");
        let mut trailing = bytes.to_vec();
        trailing.push(0);
        let mut wrong_version = bytes.to_vec();
        wrong_version[HEADER_VERSION_OFFSET] = 2;
        let mut wrong_records = bytes.to_vec();
        wrong_records[HEADER_RECORDS_OFFSET..HEADER_BYTES].copy_from_slice(&9_u64.to_le_bytes());
        let mut wrong_parameter_set = bytes.to_vec();
        wrong_parameter_set[HEADER_PARAMETER_SET_OFFSET] = 10;
        let records = peek_pcs_proof_header(FalconDegree::Falcon512, bytes)
            .unwrap()
            .1;
        let cases = [
            (&[][..], Err(MalformedProof::Truncated)),
            (&bytes[..bytes.len() - 1], Err(MalformedProof::Truncated)),
            (trailing.as_slice(), Err(MalformedProof::TrailingBytes)),
            (
                wrong_version.as_slice(),
                Err(MalformedProof::UnknownVersion { version: 2 }),
            ),
            (
                wrong_records.as_slice(),
                Ok(StatementMismatch::RecordCount {
                    expected: records,
                    actual: 9,
                }),
            ),
            (
                wrong_parameter_set.as_slice(),
                Ok(StatementMismatch::ParameterSet {
                    expected: 9,
                    actual: 10,
                }),
            ),
        ];
        let setup = pcs::FalconVerifierSetup::<Falcon512>::new(records).unwrap();
        let mut key_bytes = vec![0; Falcon512::PUBLIC_KEY_BYTES];
        key_bytes[0] = Falcon512::PUBLIC_KEY_HEADER;
        let key =
            aerie_falcon_native::source::CanonicalPublicKeyBytes::<Falcon512>::parse(&key_bytes)
                .unwrap();
        let statement = aerie_falcon_statement::AggregationStatement::new(
            (0..records)
                .map(|_| {
                    aerie_falcon_statement::AggregationRecord::new(key.clone(), Vec::new()).unwrap()
                })
                .collect(),
        )
        .unwrap();
        for (input, expected) in cases {
            let result = if matches!(expected, Ok(StatementMismatch::RecordCount { .. })) {
                crate::protocol::transport::verify_aggregate(&setup, &statement, input)
            } else {
                decode_pcs_proof(FalconDegree::Falcon512, input).map(|_| ())
            };
            let rejected = match expected {
                Err(expected) => matches!(
                    result,
                    Err(AggregationError::MalformedProof(actual)) if actual == expected
                ),
                Ok(expected) => matches!(
                    result,
                    Err(AggregationError::StatementMismatch(actual)) if actual == expected
                ),
            };
            assert!(rejected, "{result:?}");
        }
    }

    #[test]
    fn stage_wire_rounds_match_the_coefficient_domain() {
        for (degree, logn) in [(FalconDegree::Falcon512, 9), (FalconDegree::Falcon1024, 10)] {
            for records in [8_usize, 16, 1 << 14, 1 << 16] {
                let shapes = stage_shapes(degree, records).expect("wire shapes");
                assert_eq!(
                    shapes.rounds,
                    logn + records.trailing_zeros() as usize,
                    "{degree:?} at {records}"
                );
            }
        }
    }
}
