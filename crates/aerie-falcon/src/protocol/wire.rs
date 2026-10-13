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
//! and final zero padding.

use akita_serialization::{AkitaDeserialize, AkitaSerialize, Compress, Validate};
use akita_transcript::{NonceAtom, GRINDING_NONCE_SLACK_BITS};
use akita_types::Commitment;
use jolt_field::Zero as _;
use spongefish::NargDeserialize;

use crate::substrate::{BaseField, OpeningField as E};

use super::grinding::PiopGrindingPlan;
use super::manifest::CommitmentManifest;
use super::pcs::{commitment_coefficient_counts_of, AggregateProof};
use super::projection::ProjectionOutputs;
use super::{stage1::Stage1Proof, stage2::Stage2Proof, stage3::Stage3Proof};
use super::{FalconDegree, MAX_RECORDS};

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
pub const TRANSPORT_SALT_BYTES: usize = aerie_falcon_native::source::FALCON_512_SALT_BYTES;

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

    pub(super) fn validate<'a>(
        self,
        salts: impl IntoIterator<Item = &'a [u8; TRANSPORT_SALT_BYTES]>,
    ) -> Result<(), WireError> {
        if self == Self::Common {
            let mut salts = salts.into_iter();
            let first = salts.next();
            if first.is_none() || salts.any(|salt| Some(salt) != first) {
                return Err(WireError::Malformed(
                    "common-salt mode requires equal nonempty salts",
                ));
            }
        }
        Ok(())
    }
}

/// Wire codec failures. Every variant is a hard rejection.
#[derive(Debug)]
pub enum WireError {
    Serialization(akita_serialization::SerializationError),
    Malformed(&'static str),
    /// Statement-derived shape or catalog resolution failed.
    Statement(String),
}

impl core::fmt::Display for WireError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Serialization(error) => write!(f, "wire serialization: {error}"),
            Self::Malformed(reason) => write!(f, "malformed wire: {reason}"),
            Self::Statement(reason) => write!(f, "wire statement: {reason}"),
        }
    }
}

impl std::error::Error for WireError {}

impl From<akita_serialization::SerializationError> for WireError {
    fn from(error: akita_serialization::SerializationError) -> Self {
        Self::Serialization(error)
    }
}

/// Sumcheck round and terminal counts per stage, derived from the parameter
/// set and the live record count.
struct StageShapes {
    rounds: usize,
    basis16_sources: usize,
    group_count: usize,
    grinding: PiopGrindingPlan,
}
fn stage_shapes(degree: FalconDegree, records: usize) -> Result<StageShapes, WireError> {
    let manifest = CommitmentManifest::for_batch_of(degree, records)
        .map_err(|_| WireError::Malformed("invalid record count"))?;
    let grinding = PiopGrindingPlan::for_shape(degree, manifest.padded_records())
        .map_err(WireError::Malformed)?;
    Ok(StageShapes {
        rounds: grinding.rounds,
        basis16_sources: 2 + usize::from(degree == FalconDegree::Falcon1024),
        group_count: manifest.group_count(),
        grinding,
    })
}

fn write_value(buffer: &mut Vec<u8>, value: &impl AkitaSerialize) -> Result<(), WireError> {
    value.serialize_with_mode(&mut *buffer, Compress::Yes)?;
    Ok(())
}

fn write_ext_values(buffer: &mut Vec<u8>, values: &[E]) -> Result<(), WireError> {
    for value in values {
        write_value(buffer, value)?;
    }
    Ok(())
}

fn read_value<T: AkitaDeserialize<Context = ()>>(reader: &mut &[u8]) -> Result<T, WireError> {
    T::deserialize_with_mode(&mut *reader, Compress::Yes, Validate::Yes, &())
        .map_err(WireError::from)
}

fn read_ext_array<const N: usize>(reader: &mut &[u8]) -> Result<[E; N], WireError> {
    let mut values = [E::zero(); N];
    for value in &mut values {
        *value = read_value(reader)?;
    }
    Ok(values)
}

/// Decoded statement header, transport frame, and real-commitment proof.
///
/// The transport frame carries the `HashToPoint` salts; common-salt frames
/// expand their single salt to the live record count on decoding.
/// Public keys and messages are statement inputs the verifier already
/// holds; they are bound through the public-record digest, never
/// transported.
#[derive(Clone, Debug)]
pub struct DecodedAggregateProof {
    pub salt_mode: SaltMode,
    #[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
    pub degree: FalconDegree,
    pub records: usize,
    /// Per-record `HashToPoint` salts, live rows only.
    pub salts: Vec<[u8; TRANSPORT_SALT_BYTES]>,
    pub proof: AggregateProof,
}

fn read_shaped<T: AkitaDeserialize>(
    reader: &mut &[u8],
    context: &T::Context,
) -> Result<T, WireError> {
    T::deserialize_with_mode(&mut *reader, Compress::Yes, Validate::Yes, context)
        .map_err(WireError::from)
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
fn read_header(
    reader: &mut &[u8],
    magic: &[u8; 16],
    expected: FalconDegree,
) -> Result<usize, WireError> {
    if reader.len() < HEADER_BYTES {
        return Err(WireError::Malformed("truncated header"));
    }
    if &reader[..magic.len()] != magic {
        let other_mode = KNOWN_MAGICS
            .iter()
            .any(|known| *known != magic && reader[..magic.len()] == known[..]);
        return Err(WireError::Malformed(if other_mode {
            "wire mode mismatch"
        } else {
            "missing magic"
        }));
    }
    let header = &reader[..HEADER_BYTES];
    let version = u16::from_le_bytes(
        header[HEADER_VERSION_OFFSET..HEADER_PARAMETER_SET_OFFSET]
            .try_into()
            .expect("checked length"),
    );
    if version != crate::WIRE_VERSION {
        return Err(WireError::Malformed("unsupported version"));
    }
    let parameter_set = FalconDegree::from_parameter_set(header[HEADER_PARAMETER_SET_OFFSET])
        .map_err(|_| WireError::Malformed("unsupported parameter set"))?;
    if parameter_set != expected {
        return Err(WireError::Malformed("parameter set mismatch"));
    }
    let records = u64::from_le_bytes(
        header[HEADER_RECORDS_OFFSET..HEADER_BYTES]
            .try_into()
            .expect("checked length"),
    );
    let records =
        usize::try_from(records).map_err(|_| WireError::Malformed("record count overflow"))?;
    if records == 0 || records > MAX_RECORDS {
        return Err(WireError::Malformed("invalid record count"));
    }
    *reader = &reader[HEADER_BYTES..];
    Ok(records)
}

const _: () =
    assert!(HEADER_VERSION_OFFSET == MAGIC_PCS.len() && MAGIC_PCS.len() == MAGIC_PCS_COMMON.len());
const _: () = assert!(HEADER_BYTES == 27);

/// Borrowed PIOP frames in normative transcript order.
#[derive(Clone, Copy)]
struct PiopFramesRef<'a> {
    outputs: &'a ProjectionOutputs,
    stage1: &'a Stage1Proof,
    stage2: &'a Stage2Proof,
    stage3: &'a Stage3Proof,
    work: &'a [u8],
}
fn write_piop_frames(
    buffer: &mut Vec<u8>,
    shapes: &StageShapes,
    f: PiopFramesRef<'_>,
) -> Result<(), WireError> {
    if f.stage1.basis16_claims.len() != shapes.basis16_sources {
        return Err(WireError::Malformed("unexpected basis16 claim count"));
    }
    f.outputs
        .encode_wire(buffer)
        .map_err(WireError::Malformed)?;
    let mut work = f.work;
    write_work_nonce(buffer, &mut work, shapes.grinding.mu)?;
    write_work_nonce(buffer, &mut work, shapes.grinding.tau)?;
    write_work_nonce(buffer, &mut work, shapes.grinding.gamma[0])?;
    if f.stage1.rounds.len() != shapes.rounds {
        return Err(WireError::Malformed("unexpected stage round count"));
    }
    for round in &f.stage1.rounds {
        write_ext_values(buffer, round)?;
        write_work_nonce(buffer, &mut work, shapes.grinding.sumcheck[0])?;
    }
    for pair in &f.stage1.basis16_claims {
        write_ext_values(buffer, pair)?;
    }
    write_ext_values(buffer, &f.stage1.projection_claims)?;
    write_work_nonce(buffer, &mut work, shapes.grinding.gamma[1])?;
    if f.stage2.rounds.len() != shapes.rounds {
        return Err(WireError::Malformed("unexpected stage round count"));
    }
    for round in &f.stage2.rounds {
        write_ext_values(buffer, round)?;
        write_work_nonce(buffer, &mut work, shapes.grinding.sumcheck[1])?;
    }
    write_ext_values(buffer, &f.stage2.folded_digits)?;
    write_work_nonce(buffer, &mut work, shapes.grinding.gamma[2])?;
    if f.stage3.rounds.len() != shapes.rounds {
        return Err(WireError::Malformed("unexpected stage round count"));
    }
    for round in &f.stage3.rounds {
        write_ext_values(buffer, round)?;
        write_work_nonce(buffer, &mut work, shapes.grinding.sumcheck[2])?;
    }
    write_ext_values(buffer, &f.stage3.terminal_evaluations)?;
    if !work.is_empty() {
        return Err(WireError::Malformed("trailing PIOP work bytes"));
    }
    Ok(())
}
struct PiopFrames {
    outputs: ProjectionOutputs,
    stage1: Stage1Proof,
    stage2: Stage2Proof,
    stage3: Stage3Proof,
    work: Vec<u8>,
}
fn read_rounds<const N: usize>(reader: &mut &[u8], count: usize) -> Result<Vec<[E; N]>, WireError> {
    (0..count).map(|_| read_ext_array(reader)).collect()
}
fn read_piop_frames(reader: &mut &[u8], shapes: &StageShapes) -> Result<PiopFrames, WireError> {
    let outputs = ProjectionOutputs::decode_wire(reader).map_err(WireError::Malformed)?;
    let mut work = Vec::new();
    read_work_nonce(reader, &mut work, shapes.grinding.mu)?;
    read_work_nonce(reader, &mut work, shapes.grinding.tau)?;
    read_work_nonce(reader, &mut work, shapes.grinding.gamma[0])?;
    let mut stage1_rounds = Vec::with_capacity(shapes.rounds);
    for _ in 0..shapes.rounds {
        stage1_rounds.push(read_ext_array(reader)?);
        read_work_nonce(reader, &mut work, shapes.grinding.sumcheck[0])?;
    }
    let stage1 = Stage1Proof {
        rounds: stage1_rounds,
        basis16_claims: read_rounds(reader, shapes.basis16_sources)?,
        projection_claims: read_ext_array(reader)?,
    };
    read_work_nonce(reader, &mut work, shapes.grinding.gamma[1])?;
    let mut stage2_rounds = Vec::with_capacity(shapes.rounds);
    for _ in 0..shapes.rounds {
        stage2_rounds.push(read_ext_array(reader)?);
        read_work_nonce(reader, &mut work, shapes.grinding.sumcheck[1])?;
    }
    let stage2 = Stage2Proof {
        rounds: stage2_rounds,
        folded_digits: read_ext_array(reader)?,
    };
    read_work_nonce(reader, &mut work, shapes.grinding.gamma[2])?;
    let mut stage3_rounds = Vec::with_capacity(shapes.rounds);
    for _ in 0..shapes.rounds {
        stage3_rounds.push(read_ext_array(reader)?);
        read_work_nonce(reader, &mut work, shapes.grinding.sumcheck[2])?;
    }
    let stage3 = Stage3Proof {
        rounds: stage3_rounds,
        terminal_evaluations: read_ext_array(reader)?,
    };
    Ok(PiopFrames {
        outputs,
        stage1,
        stage2,
        stage3,
        work,
    })
}

fn take_work_nonce<'a>(reader: &mut &'a [u8], difficulty: u32) -> Result<&'a [u8], WireError> {
    if difficulty == 0 {
        return Ok(&[]);
    }
    let difficulty = u8::try_from(difficulty)
        .map_err(|_| WireError::Malformed("PIOP work difficulty out of range"))?;
    let nonce_bits = difficulty
        .checked_add(GRINDING_NONCE_SLACK_BITS)
        .ok_or(WireError::Malformed("PIOP nonce width overflow"))?;
    let original = *reader;
    let nonce = NonceAtom::deserialize_from_narg(reader)
        .map_err(|_| WireError::Malformed("invalid PIOP nonce encoding"))?;
    if u64::from(nonce.into_inner()) >= (1_u64 << nonce_bits) {
        return Err(WireError::Malformed("PIOP nonce out of range"));
    }
    Ok(&original[..original.len() - reader.len()])
}

fn write_work_nonce(
    buffer: &mut Vec<u8>,
    work: &mut &[u8],
    difficulty: u32,
) -> Result<(), WireError> {
    buffer.extend_from_slice(take_work_nonce(work, difficulty)?);
    Ok(())
}

fn read_work_nonce(
    reader: &mut &[u8],
    work: &mut Vec<u8>,
    difficulty: u32,
) -> Result<(), WireError> {
    work.extend_from_slice(take_work_nonce(reader, difficulty)?);
    Ok(())
}

/// The transport frame is one common salt or `records` fixed-width salts,
/// with no length words. Messages are not transported; the verifier
/// supplies them alongside the public keys and the public-record digest
/// binds both.
fn write_transport_frame(
    buffer: &mut Vec<u8>,
    mode: SaltMode,
    records: usize,
    salts: &[[u8; TRANSPORT_SALT_BYTES]],
) -> Result<(), WireError> {
    if salts.len() != records {
        return Err(WireError::Malformed("transport frame length mismatch"));
    }
    mode.validate(salts)?;
    let transported = if mode == SaltMode::Common {
        &salts[..1]
    } else {
        salts
    };
    for salt in transported {
        buffer.extend_from_slice(salt);
    }
    Ok(())
}

fn read_transport_frame(
    reader: &mut &[u8],
    mode: SaltMode,
    records: usize,
) -> Result<Vec<[u8; TRANSPORT_SALT_BYTES]>, WireError> {
    let transported = if mode == SaltMode::Common { 1 } else { records };
    let transported_bytes = transported
        .checked_mul(TRANSPORT_SALT_BYTES)
        .ok_or(WireError::Malformed("transport frame length overflow"))?;
    if reader.len() < transported_bytes {
        return Err(WireError::Malformed("truncated transport frame"));
    }
    let mut salts = Vec::with_capacity(records);
    for _ in 0..transported {
        let mut salt = [0_u8; TRANSPORT_SALT_BYTES];
        salt.copy_from_slice(&reader[..TRANSPORT_SALT_BYTES]);
        *reader = &reader[TRANSPORT_SALT_BYTES..];
        salts.push(salt);
    }
    if mode == SaltMode::Common {
        salts.resize(records, salts[0]);
    }
    Ok(salts)
}

/// Encode a real-commitment proof into canonical, statement-shaped bytes:
/// the transport frame and the commitment payloads carry no length prefixes,
/// and the decoder derives their sizes from the record count and the
/// generated catalogs. The final PCS section is `u64 LE length || Akita
/// argument string`; the length must be nonzero, at most
/// [`MAX_PCS_PROOF_BYTES`], and exactly the rest of the input. The wire proof carries the salts and the proof
/// material only; keys and messages are verifier inputs. The header's
/// parameter-set byte carries `degree`.
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn encode_pcs_proof(
    degree: FalconDegree,
    records: usize,
    salts: &[[u8; TRANSPORT_SALT_BYTES]],
    proof: &AggregateProof,
) -> Result<Vec<u8>, WireError> {
    encode_pcs_proof_with_mode(degree, records, salts, proof, SaltMode::PerRecord)
}

/// Encode a PCS proof with an explicit salt mode. The proof must have been
/// produced with the corresponding public-record binding.
pub fn encode_pcs_proof_with_mode(
    degree: FalconDegree,
    records: usize,
    salts: &[[u8; TRANSPORT_SALT_BYTES]],
    proof: &AggregateProof,
    mode: SaltMode,
) -> Result<Vec<u8>, WireError> {
    let mut buffer = Vec::new();
    write_header(&mut buffer, mode.magic(), degree, records);
    write_transport_frame(&mut buffer, mode, records, salts)?;
    write_proof_body(&mut buffer, degree, records, proof)?;
    Ok(buffer)
}

/// Decode a real-commitment proof of the expected parameter set; reject a
/// header naming another parameter set and anything noncanonical. The
/// parameter set is checked before the record count is parsed.
pub fn decode_pcs_proof(
    degree: FalconDegree,
    bytes: &[u8],
) -> Result<DecodedAggregateProof, WireError> {
    let (salt_mode, records) = peek_pcs_proof_header(degree, bytes)?;
    let mut reader = &bytes[HEADER_BYTES..];
    let salts = read_transport_frame(&mut reader, salt_mode, records)?;
    let proof = read_proof_body(&mut reader, degree, records)?;
    Ok(DecodedAggregateProof {
        salt_mode,
        #[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
        degree,
        records,
        salts,
        proof,
    })
}

/// Check the fixed-width PCS header without decoding or allocating for the
/// proof body. The returned record count has already been bounded.
pub(super) fn peek_pcs_proof_header(
    degree: FalconDegree,
    bytes: &[u8],
) -> Result<(SaltMode, usize), WireError> {
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
) -> Result<(), WireError> {
    let shapes = stage_shapes(degree, records)?;
    if proof.commitments.len() != shapes.group_count {
        return Err(WireError::Malformed("unexpected commitment count"));
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
        return Err(WireError::Malformed("PCS proof length out of range"));
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
) -> Result<AggregateProof, WireError> {
    let coefficient_counts = commitment_coefficient_counts_of(degree, records)
        .map_err(|error| WireError::Statement(error.to_string()))?;
    let shapes = stage_shapes(degree, records)?;
    if coefficient_counts.len() != shapes.group_count {
        return Err(WireError::Malformed("unexpected commitment count"));
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
fn read_pcs_section(reader: &mut &[u8]) -> Result<Vec<u8>, WireError> {
    if reader.len() < 8 {
        return Err(WireError::Malformed("truncated PCS length"));
    }
    let length = u64::from_le_bytes(reader[..8].try_into().expect("checked length"));
    *reader = &reader[8..];
    let length = usize::try_from(length)
        .ok()
        .filter(|length| (1..=MAX_PCS_PROOF_BYTES).contains(length))
        .ok_or(WireError::Malformed("PCS proof length out of range"))?;
    match reader.len().cmp(&length) {
        core::cmp::Ordering::Less => return Err(WireError::Malformed("truncated PCS proof")),
        core::cmp::Ordering::Greater => return Err(WireError::Malformed("trailing data")),
        core::cmp::Ordering::Equal => {}
    }
    let pcs = reader.to_vec();
    *reader = &[];
    Ok(pcs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::pcs;
    use crate::protocol::public_inputs::PublicInputs;
    use crate::protocol::ring::{PublicRingKeys, PublicTargets};
    use crate::protocol::{Falcon512, FalconWitness};
    use aerie_falcon_native::Falcon1024;
    use aerie_falcon_statement::DirectRow;
    use jolt_field::Ring as _;
    use std::ops::Range;
    use std::sync::OnceLock;

    fn on_deep_stack<T: Send>(body: impl FnOnce() -> T + Send) -> T {
        std::thread::scope(|scope| {
            let handle = std::thread::Builder::new()
                .stack_size(512 * 1024 * 1024)
                .spawn_scoped(scope, move || {
                    #[cfg(feature = "parallel")]
                    {
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(1)
                            .stack_size(512 * 1024 * 1024)
                            .build()
                            .expect("proof thread pool");
                        pool.install(body)
                    }
                    #[cfg(not(feature = "parallel"))]
                    body()
                })
                .expect("spawn PCS verifier");
            handle.join().expect("join PCS verifier")
        })
    }

    fn fixture(records: usize) -> FalconWitness {
        let rows = (0..records)
            .map(|seed| {
                DirectRow::<Falcon512>::synthetic(seed as u64 + 121).expect("synthetic row")
            })
            .collect::<Vec<_>>();
        let mut batch = FalconWitness::from_rows(&rows);
        crate::protocol::pad_witness(&mut batch).unwrap();
        batch
    }

    fn fixture_1024(records: usize) -> FalconWitness {
        let rows = (0..records)
            .map(|seed| {
                DirectRow::<Falcon1024>::synthetic(seed as u64 + 0x0102_4c00)
                    .expect("synthetic row")
            })
            .collect::<Vec<_>>();
        let mut batch = FalconWitness::from_rows(&rows);
        crate::protocol::pad_witness(&mut batch).unwrap();
        batch
    }

    type Encoded = (
        FalconWitness,
        AggregateProof,
        Vec<[u8; TRANSPORT_SALT_BYTES]>,
        Vec<u8>,
    );

    fn prove_and_encode(batch: FalconWitness, degree: FalconDegree, records: usize) -> Encoded {
        let proof_result = on_deep_stack(|| {
            pcs::prove(&batch, b"wire test records               ")
                .expect("production PCS prover")
                .0
        });
        let salts = vec![[7_u8; TRANSPORT_SALT_BYTES]; records];
        let bytes =
            encode_pcs_proof(degree, records, &salts, &proof_result).expect("encode PCS proof");
        (batch, proof_result, salts, bytes)
    }

    fn honest_512() -> &'static Encoded {
        static HONEST: OnceLock<Encoded> = OnceLock::new();
        HONEST.get_or_init(|| prove_and_encode(fixture(8), FalconDegree::Falcon512, 8))
    }

    fn honest_1024() -> &'static Encoded {
        static HONEST: OnceLock<Encoded> = OnceLock::new();
        HONEST.get_or_init(|| prove_and_encode(fixture_1024(4), FalconDegree::Falcon1024, 4))
    }

    fn public<'a>(
        batch: &'a FalconWitness,
        records: usize,
        digest: &'a [u8],
    ) -> PublicInputs<'a, i16> {
        let degree = FalconDegree::from_degree(batch.degree()).unwrap();
        PublicInputs {
            records,
            salt_mode: crate::protocol::wire::SaltMode::PerRecord,
            public_records: digest,
            h: PublicRingKeys::ExpandedOf(degree, &batch.h),
            t: PublicTargets::Expanded(&batch.t),
        }
    }

    const DEGREES: [FalconDegree; 2] = [FalconDegree::Falcon512, FalconDegree::Falcon1024];

    fn other(degree: FalconDegree) -> FalconDegree {
        match degree {
            FalconDegree::Falcon512 => FalconDegree::Falcon1024,
            FalconDegree::Falcon1024 => FalconDegree::Falcon512,
        }
    }

    fn nonce_ranges(bytes: &[u8]) -> Vec<Range<usize>> {
        let mut reader = bytes;
        let records = read_header(&mut reader, MAGIC_PCS, FalconDegree::Falcon512).unwrap();
        read_transport_frame(&mut reader, SaltMode::PerRecord, records).unwrap();
        for count in commitment_coefficient_counts_of(FalconDegree::Falcon512, records).unwrap() {
            let _: Commitment<BaseField> = read_shaped(&mut reader, &count).unwrap();
        }
        ProjectionOutputs::decode_wire(&mut reader).unwrap();
        let shapes = stage_shapes(FalconDegree::Falcon512, records).unwrap();
        let mut ranges = Vec::new();
        let mut record_nonce = |reader: &mut &[u8], difficulty| {
            let start = bytes.len() - reader.len();
            let nonce = take_work_nonce(reader, difficulty).unwrap();
            ranges.push(start..start + nonce.len());
        };
        record_nonce(&mut reader, shapes.grinding.mu);
        record_nonce(&mut reader, shapes.grinding.tau);
        record_nonce(&mut reader, shapes.grinding.gamma[0]);
        for _ in 0..shapes.rounds {
            let _: [E; 3] = read_ext_array(&mut reader).unwrap();
            record_nonce(&mut reader, shapes.grinding.sumcheck[0]);
        }
        for _ in 0..shapes.basis16_sources {
            let _: [E; 2] = read_ext_array(&mut reader).unwrap();
        }
        let _: [E; 3] = read_ext_array(&mut reader).unwrap();
        record_nonce(&mut reader, shapes.grinding.gamma[1]);
        for _ in 0..shapes.rounds {
            let _: [E; 4] = read_ext_array(&mut reader).unwrap();
            record_nonce(&mut reader, shapes.grinding.sumcheck[1]);
        }
        let _: [E; 6] = read_ext_array(&mut reader).unwrap();
        record_nonce(&mut reader, shapes.grinding.gamma[2]);
        for _ in 0..shapes.rounds {
            let _: [E; 3] = read_ext_array(&mut reader).unwrap();
            record_nonce(&mut reader, shapes.grinding.sumcheck[2]);
        }
        let _: [E; 9] = read_ext_array(&mut reader).unwrap();
        assert_eq!(ranges.len(), 5 + 3 * shapes.rounds);
        ranges
    }

    fn replace_range(bytes: &[u8], range: Range<usize>, replacement: &[u8]) -> Vec<u8> {
        let mut changed = Vec::with_capacity(bytes.len() - range.len() + replacement.len());
        changed.extend_from_slice(&bytes[..range.start]);
        changed.extend_from_slice(replacement);
        changed.extend_from_slice(&bytes[range.end..]);
        changed
    }

    fn proof_rejects_512(bytes: &[u8]) -> bool {
        let Ok(decoded) = decode_pcs_proof(FalconDegree::Falcon512, bytes) else {
            return true;
        };
        let (batch, _, _, _) = honest_512();
        on_deep_stack(|| {
            let verifier = pcs::FalconVerifierSetup::<Falcon512>::new(8).unwrap();
            pcs::verify(
                &public(batch, 8, b"wire test records               "),
                &verifier,
                &decoded.proof,
            )
        })
        .is_err()
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
            salts
        );
        assert!(reader.is_empty());
        assert!(read_transport_frame(&mut &bytes[..39], SaltMode::Common, 5).is_err());
        assert!(write_transport_frame(&mut Vec::new(), SaltMode::Common, 0, &[]).is_err());
        assert!(write_transport_frame(&mut Vec::new(), SaltMode::Common, 4, &salts).is_err());
        let mut unequal = salts;
        unequal[4][39] ^= 1;
        assert!(write_transport_frame(&mut Vec::new(), SaltMode::Common, 5, &unequal).is_err());
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
            assert!(read_header(&mut above_maximum.as_slice(), MAGIC_PCS, degree).is_err());
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
            assert!(decode_pcs_proof(other, bytes).is_err());
            let decoded = decode_pcs_proof(degree, bytes).expect("seed decodes");
            assert_eq!(decoded.salt_mode, mode);
            let reencoded = encode_pcs_proof_with_mode(
                decoded.degree,
                decoded.records,
                &decoded.salts,
                &decoded.proof,
                decoded.salt_mode,
            )
            .expect("seed re-encodes");
            assert_eq!(reencoded.as_slice(), bytes);
        }
    }

    #[test]
    fn truncated_maximum_salt_frame_rejects_before_materializing_salts() {
        let mut bytes = Vec::new();
        write_header(&mut bytes, MAGIC_PCS, FalconDegree::Falcon512, MAX_RECORDS);
        let error = decode_pcs_proof(FalconDegree::Falcon512, &bytes)
            .expect_err("header-only maximum batch must reject");
        assert!(matches!(
            error,
            WireError::Malformed("truncated transport frame")
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
                assert!(matches!(
                    error,
                    WireError::Malformed("parameter set mismatch")
                ));
                for unknown in [0_u8, 8, 11, 0xff] {
                    let mut unknown_set = header.clone();
                    unknown_set[HEADER_PARAMETER_SET_OFFSET] = unknown;
                    let error = read_header(&mut unknown_set.as_slice(), magic, degree)
                        .expect_err("unknown parameter set");
                    assert!(matches!(
                        error,
                        WireError::Malformed("unsupported parameter set")
                    ));
                }
                for version in [0_u16, 2, 40, 42, 0x0100] {
                    let mut wrong_version = header.clone();
                    wrong_version[HEADER_VERSION_OFFSET..HEADER_PARAMETER_SET_OFFSET]
                        .copy_from_slice(&version.to_le_bytes());
                    let error = read_header(&mut wrong_version.as_slice(), magic, degree)
                        .expect_err("unsupported version");
                    assert!(matches!(error, WireError::Malformed("unsupported version")));
                }
            }
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

    #[test]
    fn native_v1_wire_golden_vectors() {
        let actual = [honest_512(), honest_1024()].map(|fixture| {
            crate::protocol::instance::digest(b"aerie/test/v1/wire-golden", &fixture.3)
        });
        assert_eq!(
            actual,
            [
                [
                    225, 60, 85, 165, 45, 232, 60, 93, 176, 163, 112, 124, 60, 128, 199, 128, 170,
                    169, 59, 151, 174, 1, 29, 19, 61, 81, 136, 155, 146, 15, 139, 42
                ],
                [
                    175, 135, 142, 101, 20, 107, 137, 127, 11, 36, 179, 203, 8, 109, 252, 220, 131,
                    217, 137, 170, 26, 32, 144, 72, 39, 229, 177, 154, 27, 39, 250, 67
                ]
            ],
        );
    }

    #[test]
    fn production_projection_outputs_are_rice_encoded_after_commitments() {
        let (_, proof, salts, bytes) = honest_512();
        let commitment_bytes = proof
            .commitments
            .iter()
            .map(|commitment| {
                let mut encoded = Vec::new();
                write_value(&mut encoded, commitment).expect("commitment encoding");
                encoded.len()
            })
            .sum::<usize>();
        let start = HEADER_BYTES + salts.len() * TRANSPORT_SALT_BYTES + commitment_bytes;
        let mut section = Vec::new();
        proof
            .projection_outputs
            .encode_wire(&mut section)
            .expect("outputs encode");
        assert_eq!(&bytes[start..start + section.len()], section.as_slice());
        assert!(section.len() < 904);
        let mut noncanonical = bytes.clone();
        noncanonical[start] = 64;
        assert!(decode_pcs_proof(FalconDegree::Falcon512, &noncanonical).is_err());
        for (offset, delta) in [(0, 1), (0, -1), (1, 1), (1, -1)] {
            let mut noncanonical = bytes.clone();
            let slot = &mut noncanonical[start + offset];
            *slot = slot.wrapping_add_signed(delta);
            assert!(decode_pcs_proof(FalconDegree::Falcon512, &noncanonical).is_err());
        }
        let mut off_lattice = proof.clone();
        off_lattice.projection_outputs.z[0] += BaseField::from_u64(1);
        assert!(encode_pcs_proof(FalconDegree::Falcon512, 8, salts, &off_lattice).is_err());
    }

    #[test]
    fn pcs_codec_roundtrips_verifies_and_rejects_malformed_frames() {
        let (batch, proof, salts, bytes) = honest_512();
        let decoded = decode_pcs_proof(FalconDegree::Falcon512, bytes).expect("decode");
        assert_eq!(decoded.records, 8);
        assert_eq!(decoded.salts, *salts);
        assert!(decode_pcs_proof(FalconDegree::Falcon1024, bytes).is_err());
        on_deep_stack(|| {
            let verifier_setup =
                pcs::FalconVerifierSetup::<Falcon512>::new(8).expect("verifier setup");
            pcs::verify(
                &public(batch, 8, b"wire test records               "),
                &verifier_setup,
                &decoded.proof,
            )
        })
        .expect("decoded production proof verifies");
        let reencoded =
            encode_pcs_proof(FalconDegree::Falcon512, 8, &decoded.salts, &decoded.proof)
                .expect("re-encode");
        assert_eq!(reencoded, *bytes);

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(decode_pcs_proof(FalconDegree::Falcon512, &trailing).is_err());
        assert!(decode_pcs_proof(FalconDegree::Falcon512, &bytes[..bytes.len() - 1]).is_err());
        let commitment_start = HEADER_BYTES + salts.len() * TRANSPORT_SALT_BYTES;
        assert!(
            decode_pcs_proof(FalconDegree::Falcon512, &bytes[..commitment_start + 100]).is_err(),
            "a truncated commitment payload must reject"
        );
        let prefix = bytes.len() - proof.pcs.len() - 8;
        assert_eq!(
            &bytes[prefix..prefix + 8],
            &(proof.pcs.len() as u64).to_le_bytes()
        );
        for length in [
            0,
            proof.pcs.len() as u64 - 1,
            proof.pcs.len() as u64 + 1,
            MAX_PCS_PROOF_BYTES as u64 + 1,
            u64::MAX,
        ] {
            let mut mutated = bytes.clone();
            mutated[prefix..prefix + 8].copy_from_slice(&length.to_le_bytes());
            assert!(decode_pcs_proof(FalconDegree::Falcon512, &mutated).is_err());
        }
        let mut flipped = bytes.clone();
        let index = flipped.len() / 2;
        flipped[index] ^= 1;
        let rejected = match decode_pcs_proof(FalconDegree::Falcon512, &flipped) {
            Err(_) => true,
            Ok(mutated) => on_deep_stack(|| {
                let verifier_setup =
                    pcs::FalconVerifierSetup::<Falcon512>::new(8).expect("verifier setup");
                pcs::verify(
                    &public(batch, 8, b"wire test records               "),
                    &verifier_setup,
                    &mutated.proof,
                )
                .is_err()
            }),
        };
        assert!(rejected, "a mid-frame bit flip must be rejected");
    }

    #[test]
    fn proof_rejects_every_invalid_piop_nonce_form() {
        let (_, _, _, bytes) = honest_512();
        let ranges = nonce_ranges(bytes);
        let first = ranges[0].clone();
        let second = ranges
            .iter()
            .skip(1)
            .find(|range| bytes[(*range).clone()] != bytes[first.clone()])
            .expect("fixture has distinct work nonces")
            .clone();

        assert!(
            proof_rejects_512(&replace_range(bytes, first.clone(), &[])),
            "omitted"
        );

        let mut extra = bytes.clone();
        extra.insert(first.start, 0);
        assert!(proof_rejects_512(&extra), "extra");

        let mut reordered = Vec::with_capacity(bytes.len());
        reordered.extend_from_slice(&bytes[..first.start]);
        reordered.extend_from_slice(&bytes[second.clone()]);
        reordered.extend_from_slice(&bytes[first.end..second.start]);
        reordered.extend_from_slice(&bytes[first.clone()]);
        reordered.extend_from_slice(&bytes[second.end..]);
        assert!(proof_rejects_512(&reordered), "reordered");

        let mut truncated = bytes[..first.start].to_vec();
        truncated.push(0x80);
        assert!(proof_rejects_512(&truncated), "truncated");

        assert!(
            proof_rejects_512(&replace_range(bytes, first.clone(), &[0x80, 0x00])),
            "non-canonical"
        );
        assert!(
            proof_rejects_512(&replace_range(bytes, first.clone(), &[0x80, 0x10])),
            "out of range"
        );

        let mut failed_predicate = bytes.clone();
        failed_predicate[first.start] ^= 1;
        assert!(proof_rejects_512(&failed_predicate), "failed predicate");

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(proof_rejects_512(&trailing), "trailing bytes");
    }

    fn assert_rejects_noncanonical_frames(degree: FalconDegree, bytes: &[u8]) {
        let decode = |bytes: &[u8]| decode_pcs_proof(degree, bytes);
        assert!(decode(bytes).is_ok(), "{degree:?}: honest frame");
        let mut trailing = bytes.to_vec();
        trailing.push(0);
        assert!(decode(&trailing).is_err(), "trailing data");
        assert!(decode(&bytes[..bytes.len() - 1]).is_err(), "truncation");
        let mut wrong_version = bytes.to_vec();
        wrong_version[HEADER_VERSION_OFFSET..HEADER_PARAMETER_SET_OFFSET]
            .copy_from_slice(&40_u16.to_le_bytes());
        assert!(decode(&wrong_version).is_err(), "version");
        assert!(
            decode_pcs_proof(other(degree), bytes).is_err(),
            "cross-parameter-set decode"
        );
        let mut flipped_set = bytes.to_vec();
        flipped_set[HEADER_PARAMETER_SET_OFFSET] = other(degree).parameter_set();
        assert!(decode(&flipped_set).is_err(), "flipped parameter set");
        assert!(
            decode_pcs_proof(other(degree), &flipped_set).is_err(),
            "relabeled body"
        );
        let mut wrong_records = bytes.to_vec();
        wrong_records[HEADER_RECORDS_OFFSET..HEADER_BYTES].copy_from_slice(&0_u64.to_le_bytes());
        assert!(decode(&wrong_records).is_err(), "record count");
        let mut wrong_shape = bytes.to_vec();
        wrong_shape[HEADER_RECORDS_OFFSET..HEADER_BYTES].copy_from_slice(&16_u64.to_le_bytes());
        assert!(decode(&wrong_shape).is_err(), "group layout");
    }

    #[test]
    fn production_codecs_reject_noncanonical_frames_for_both_degrees() {
        assert_rejects_noncanonical_frames(FalconDegree::Falcon512, &honest_512().3);
        assert_rejects_noncanonical_frames(FalconDegree::Falcon1024, &honest_1024().3);
        assert!(decode_pcs_proof(FalconDegree::Falcon1024, &honest_512().3).is_err());
    }

    #[test]
    fn falcon_1024_production_codec_roundtrips_and_verifies() {
        let (batch, _, salts, bytes) = honest_1024();
        assert_eq!(bytes[HEADER_PARAMETER_SET_OFFSET], 10);
        let decoded = decode_pcs_proof(FalconDegree::Falcon1024, bytes).expect("decode");
        assert_eq!(decoded.degree, FalconDegree::Falcon1024);
        assert_eq!(decoded.records, 4);
        assert_eq!(decoded.salts, *salts);
        on_deep_stack(|| {
            let verifier_setup =
                pcs::FalconVerifierSetup::<Falcon1024>::new(4).expect("verifier setup");
            pcs::verify(
                &public(batch, 4, b"wire test records               "),
                &verifier_setup,
                &decoded.proof,
            )
        })
        .expect("decoded Falcon-1024 production proof verifies");
        let reencoded =
            encode_pcs_proof(FalconDegree::Falcon1024, 4, &decoded.salts, &decoded.proof)
                .expect("re-encode");
        assert_eq!(reencoded, *bytes);
    }
}
