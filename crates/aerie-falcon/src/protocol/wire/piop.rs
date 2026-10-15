//! Statement-shaped PIOP frames in normative transcript order.

use akita_transcript::NonceAtom;
use spongefish::NargDeserialize;

use crate::{AggregationError, MalformedProof};

use super::super::projection::ProjectionOutputs;
use super::super::{stage1::Stage1Proof, stage2::Stage2Proof, stage3::Stage3Proof};
use super::{read_ext_array, write_ext_values, WireShape, E};

/// Borrowed PIOP frames in normative transcript order.
#[derive(Clone, Copy)]
pub(super) struct PiopFramesRef<'a> {
    pub(super) outputs: &'a ProjectionOutputs,
    pub(super) stage1: &'a Stage1Proof,
    pub(super) stage2: &'a Stage2Proof,
    pub(super) stage3: &'a Stage3Proof,
    pub(super) work: &'a [u8],
}
pub(super) fn write_piop_frames(
    buffer: &mut Vec<u8>,
    shapes: &WireShape,
    f: PiopFramesRef<'_>,
) -> Result<(), AggregationError> {
    if f.stage1.basis16_claims.len() != shapes.stages.basis16_sources {
        return Err(AggregationError::MalformedProof(MalformedProof::StageShape));
    }
    f.outputs
        .encode_wire(buffer)
        .map_err(|_| AggregationError::MalformedProof(MalformedProof::ProjectionEncoding))?;
    let mut work = f.work;
    write_work_nonce(buffer, &mut work, shapes.grinding.mu)?;
    write_work_nonce(buffer, &mut work, shapes.grinding.tau)?;
    write_work_nonce(buffer, &mut work, shapes.grinding.gamma[0])?;
    if f.stage1.rounds.len() != shapes.rounds {
        return Err(AggregationError::MalformedProof(MalformedProof::StageShape));
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
        return Err(AggregationError::MalformedProof(MalformedProof::StageShape));
    }
    for round in &f.stage2.rounds {
        write_ext_values(buffer, round)?;
        write_work_nonce(buffer, &mut work, shapes.grinding.sumcheck[1])?;
    }
    write_ext_values(buffer, &f.stage2.folded_digits)?;
    write_work_nonce(buffer, &mut work, shapes.grinding.gamma[2])?;
    if f.stage3.rounds.len() != shapes.rounds {
        return Err(AggregationError::MalformedProof(MalformedProof::StageShape));
    }
    for round in &f.stage3.rounds {
        write_ext_values(buffer, round)?;
        write_work_nonce(buffer, &mut work, shapes.grinding.sumcheck[2])?;
    }
    write_ext_values(buffer, &f.stage3.terminal_evaluations)?;
    if !work.is_empty() {
        return Err(AggregationError::MalformedProof(
            MalformedProof::TrailingBytes,
        ));
    }
    Ok(())
}
pub(super) struct PiopFrames {
    pub(super) outputs: ProjectionOutputs,
    pub(super) stage1: Stage1Proof,
    pub(super) stage2: Stage2Proof,
    pub(super) stage3: Stage3Proof,
    pub(super) work: Vec<u8>,
}
fn read_rounds<const N: usize>(
    reader: &mut &[u8],
    count: usize,
) -> Result<Vec<[E; N]>, AggregationError> {
    (0..count).map(|_| read_ext_array(reader)).collect()
}
pub(super) fn read_piop_frames(
    reader: &mut &[u8],
    shapes: &WireShape,
) -> Result<PiopFrames, AggregationError> {
    let outputs = ProjectionOutputs::decode_wire(reader)
        .map_err(|_| AggregationError::MalformedProof(MalformedProof::ProjectionEncoding))?;
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
        basis16_claims: read_rounds(reader, shapes.stages.basis16_sources)?,
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

pub(super) fn take_work_nonce<'a>(
    reader: &mut &'a [u8],
    difficulty: u32,
) -> Result<&'a [u8], AggregationError> {
    if difficulty == 0 {
        return Ok(&[]);
    }
    let difficulty = u8::try_from(difficulty)
        .map_err(|_| AggregationError::MalformedProof(MalformedProof::NonceEncoding))?;
    let nonce_limit = super::super::transcript::work_nonce_limit(difficulty)
        .map_err(|_| AggregationError::MalformedProof(MalformedProof::NonceEncoding))?;
    let original = *reader;
    let nonce = NonceAtom::deserialize_from_narg(reader)
        .map_err(|_| AggregationError::MalformedProof(MalformedProof::NonceEncoding))?;
    if u64::from(nonce.into_inner()) >= nonce_limit {
        return Err(AggregationError::MalformedProof(
            MalformedProof::NonceEncoding,
        ));
    }
    Ok(&original[..original.len() - reader.len()])
}

fn write_work_nonce(
    buffer: &mut Vec<u8>,
    work: &mut &[u8],
    difficulty: u32,
) -> Result<(), AggregationError> {
    buffer.extend_from_slice(take_work_nonce(work, difficulty)?);
    Ok(())
}

fn read_work_nonce(
    reader: &mut &[u8],
    work: &mut Vec<u8>,
    difficulty: u32,
) -> Result<(), AggregationError> {
    work.extend_from_slice(take_work_nonce(reader, difficulty)?);
    Ok(())
}
