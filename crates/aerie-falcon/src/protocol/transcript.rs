//! Native Spongefish transcript and the version-1 PIOP event registry.
//!
//! Registry metadata checks local order and shape without being absorbed.
//! The bound shape, fixed-width fields, canonical Rice section and canonical
//! nonces determine payload boundaries. Spongefish streams adjacent absorbs
//! and binds consumed squeeze lengths on the next absorb.
//! Wire version 1 is frozen: incompatible transcript or encoding changes
//! require a new wire version. Proof interchange requires the same pinned
//! implementation and profile.

use akita_transcript::{
    grinding_predicate_accepts, nonce_max_bytes, NonceAtom, ProtocolContextRecord,
    ProtocolMessageKind, ProtocolSiteId, TranscriptSponge, GRINDING_NONCE_SLACK_BITS,
};
use jolt_field::{CanonicalEncoding, ExtField};
use spongefish::{
    DomainSeparator, DuplexSpongeInterface, Encoding, NargDeserialize, VerifierState,
    WithoutInstance,
};
use std::num::NonZeroU8;

use super::{
    grinding::PiopGrindingPlan,
    instance::{BINDING_PREFIX_BYTES, INSTANCE_BYTES, MANIFEST_SUFFIX_RANGE},
    manifest::{CommitmentManifest, GroupRole},
    stage::StageShapes,
    FalconDegree,
};
use crate::substrate::{BaseField, OpeningField};

#[doc(hidden)]
pub const INSTANCE_LABEL: &str = "aerie/falcon/v1/instance";
const PROJECTION_SEED_SITE: &str = "aerie/falcon/v1/projection/seed";
const PCS_SESSION_SITE: &str = "aerie/falcon/v1/pcs/session";
#[doc(hidden)]
pub const PROOF_KIND: u8 = 1;
const PUBLIC_KIND: u8 = 0;
const CHALLENGE_KIND: u8 = 2;
const EXPORT_KIND: u8 = 4;

const PROJECTION_OUTPUT_SITE: &str = "aerie/falcon/v1/projection/output";
#[doc(hidden)]
pub const MU_SITE: &str = "aerie/falcon/v1/mu";
#[doc(hidden)]
pub const TAU_SITE: &str = "aerie/falcon/v1/tau";
#[doc(hidden)]
pub const STAGE1_GAMMA_SITE: &str = "aerie/falcon/v1/stage1/gamma";
#[doc(hidden)]
pub const STAGE1_TERMINAL_SITE: &str = "aerie/falcon/v1/stage1/terminal";
#[doc(hidden)]
pub const STAGE2_GAMMA_SITE: &str = "aerie/falcon/v1/stage2/gamma";
#[doc(hidden)]
pub const STAGE2_TERMINAL_SITE: &str = "aerie/falcon/v1/stage2/terminal";
#[doc(hidden)]
pub const STAGE3_GAMMA_SITE: &str = "aerie/falcon/v1/stage3/gamma";
#[doc(hidden)]
pub const STAGE3_TERMINAL_SITE: &str = "aerie/falcon/v1/stage3/terminal";

/// Spongefish protocol tag for Aerie's PIOP stream, zero-padded to 64 bytes.
pub const PIOP_PROTOCOL_TAG: [u8; 64] = protocol_tag();

const fn protocol_tag() -> [u8; 64] {
    let source = b"aerie/falcon/v1/piop/blake2b";
    let mut tag = [0; 64];
    let mut index = 0;
    while index < source.len() {
        tag[index] = source[index];
        index += 1;
    }
    tag
}

/// Transcript errors indicate a local protocol replay mismatch or an
/// impossible native sampling failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TranscriptOrderError {
    /// The instance shape did not match the supplied canonical manifest.
    InstanceShape,
    /// The next event does not match the shared registry.
    EventOrder { index: usize },
    /// The event's atom count or canonical payload size is incorrect.
    EventShape,
    /// A proof-of-work nonce fails the scheduled predicate.
    InvalidWork { difficulty: u8 },
    /// A proof-of-work nonce is not canonically encoded or is out of range.
    NonceEncoding,
    /// The work stream contains bytes after the scheduled nonces.
    TrailingBytes,
    /// The work schedule cannot be represented by the native nonce format.
    WorkSchedule,
    /// The bounded prover nonce search exhausted its candidates.
    WorkSearch { difficulty: u8 },
    /// A committed prover nonce differs from its searched predicate.
    WorkReplay { difficulty: u8 },
    /// The pinned sampler rejected a field encoding configuration.
    /// A required suffix of the registry was not consumed.
    Incomplete { index: usize },
}

impl core::fmt::Display for TranscriptOrderError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InstanceShape => {
                formatter.write_str("transcript instance does not match manifest")
            }
            Self::EventOrder { index } => write!(
                formatter,
                "transcript event {index} does not match the registry"
            ),
            Self::EventShape => formatter.write_str("invalid transcript event shape"),
            Self::InvalidWork { difficulty } => write!(
                formatter,
                "proof-of-work nonce fails difficulty {difficulty}"
            ),
            Self::NonceEncoding => formatter.write_str("invalid proof-of-work nonce encoding"),
            Self::TrailingBytes => formatter.write_str("trailing proof-of-work bytes"),
            Self::WorkSchedule => formatter.write_str("invalid proof-of-work schedule"),
            Self::WorkSearch { difficulty } => write!(
                formatter,
                "proof-of-work search exhausted at difficulty {difficulty}"
            ),
            Self::WorkReplay { difficulty } => write!(
                formatter,
                "proof-of-work replay differs at difficulty {difficulty}"
            ),
            Self::Incomplete { index } => {
                write!(formatter, "transcript ended before event {index}")
            }
        }
    }
}

impl std::error::Error for TranscriptOrderError {}

/// Expected byte size for a registry payload. The Rice section is encoded by
/// the canonical Section 10.4 codec and its length is therefore payload-derived.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PayloadSize {
    Exact(usize),
    RiceSection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RegistryEvent {
    site: String,
    kind: u8,
    atom_count: usize,
    payload_size: PayloadSize,
    /// The shape-derived work-plan lookup attached to each field challenge.
    difficulty: Option<u32>,
}

#[derive(Clone, Debug)]
struct EventCursor {
    events: Vec<RegistryEvent>,
    next: usize,
}

impl EventCursor {
    fn new(
        manifest: &CommitmentManifest,
        record: &[u8; INSTANCE_BYTES],
    ) -> Result<Self, TranscriptOrderError> {
        let binding = manifest.binding_bytes();
        if binding.len() != BINDING_PREFIX_BYTES + MANIFEST_SUFFIX_RANGE.len()
            || record[..BINDING_PREFIX_BYTES] != binding[..BINDING_PREFIX_BYTES]
            || record[MANIFEST_SUFFIX_RANGE] != binding[BINDING_PREFIX_BYTES..]
        {
            return Err(TranscriptOrderError::InstanceShape);
        }

        let degree = match (
            record[2],
            u32::from_le_bytes(record[7..11].try_into().unwrap()),
        ) {
            (9, 512) => FalconDegree::Falcon512,
            (10, 1024) => FalconDegree::Falcon1024,
            _ => return Err(TranscriptOrderError::InstanceShape),
        };
        let padded_rows = u64::from_le_bytes(record[19..27].try_into().unwrap());
        let Ok(padded_rows) = usize::try_from(padded_rows) else {
            return Err(TranscriptOrderError::InstanceShape);
        };
        if padded_rows == 0 || !padded_rows.is_power_of_two() {
            return Err(TranscriptOrderError::InstanceShape);
        }
        let plan = PiopGrindingPlan::for_shape(degree, padded_rows)
            .map_err(|_| TranscriptOrderError::InstanceShape)?;
        let rounds = plan.rounds;
        let [stage1, stage2, stage3] = StageShapes::for_degree(degree).stages;

        let mut events = Vec::with_capacity(21 + 6 * rounds);
        push_absorb(&mut events, INSTANCE_LABEL, PUBLIC_KIND, 1, INSTANCE_BYTES);
        for site in [
            "aerie/falcon/v1/commit/s1",
            "aerie/falcon/v1/commit/slack",
            "aerie/falcon/v1/commit/s2-sources",
            "aerie/falcon/v1/commit/budget",
        ] {
            push_absorb(&mut events, site, PROOF_KIND, 16, 128);
        }
        push_challenge(&mut events, PROJECTION_SEED_SITE, 1, 32, 0);
        events.push(RegistryEvent {
            site: PROJECTION_OUTPUT_SITE.to_owned(),
            kind: PROOF_KIND,
            atom_count: 1,
            payload_size: PayloadSize::RiceSection,
            difficulty: None,
        });
        push_challenge(&mut events, MU_SITE, 8, 128, plan.mu);
        push_challenge(&mut events, TAU_SITE, rounds, 16 * rounds, plan.tau);
        push_challenge(
            &mut events,
            STAGE1_GAMMA_SITE,
            stage1.gamma_count,
            16 * stage1.gamma_count,
            plan.gamma[0],
        );
        for round in 0..rounds {
            push_round(
                &mut events,
                "stage1",
                round,
                stage1.round_message_width,
                16 * stage1.round_message_width,
                plan.sumcheck[0],
            );
        }
        push_absorb(
            &mut events,
            STAGE1_TERMINAL_SITE,
            PROOF_KIND,
            stage1.terminal_count,
            16 * stage1.terminal_count,
        );
        push_challenge(
            &mut events,
            STAGE2_GAMMA_SITE,
            stage2.gamma_count,
            16 * stage2.gamma_count,
            plan.gamma[1],
        );
        for round in 0..rounds {
            push_round(
                &mut events,
                "stage2",
                round,
                stage2.round_message_width,
                16 * stage2.round_message_width,
                plan.sumcheck[1],
            );
        }
        push_absorb(
            &mut events,
            STAGE2_TERMINAL_SITE,
            PROOF_KIND,
            stage2.terminal_count,
            16 * stage2.terminal_count,
        );
        push_challenge(
            &mut events,
            STAGE3_GAMMA_SITE,
            stage3.gamma_count,
            16 * stage3.gamma_count,
            plan.gamma[2],
        );
        for round in 0..rounds {
            push_round(
                &mut events,
                "stage3",
                round,
                stage3.round_message_width,
                16 * stage3.round_message_width,
                plan.sumcheck[2],
            );
        }
        push_absorb(
            &mut events,
            STAGE3_TERMINAL_SITE,
            PROOF_KIND,
            stage3.terminal_count,
            16 * stage3.terminal_count,
        );
        events.push(RegistryEvent {
            site: PCS_SESSION_SITE.to_owned(),
            kind: EXPORT_KIND,
            atom_count: 1,
            payload_size: PayloadSize::Exact(32),
            difficulty: None,
        });

        Ok(Self { events, next: 0 })
    }

    fn take(
        &mut self,
        site: &str,
        kind: u8,
        atom_count: usize,
        payload_bytes: usize,
    ) -> Result<RegistryEvent, TranscriptOrderError> {
        let Some(expected) = self.events.get(self.next).cloned() else {
            return Err(TranscriptOrderError::Incomplete { index: self.next });
        };
        let shape_matches = expected.site == site
            && expected.kind == kind
            && expected.atom_count == atom_count
            && match expected.payload_size {
                PayloadSize::Exact(bytes) => bytes == payload_bytes,
                PayloadSize::RiceSection => payload_bytes >= 2,
            };
        if !shape_matches {
            return Err(TranscriptOrderError::EventOrder { index: self.next });
        }
        self.next += 1;
        Ok(expected)
    }

    fn require_next(&self, index: usize) -> Result<(), TranscriptOrderError> {
        if self.next != index {
            return Err(TranscriptOrderError::Incomplete { index: self.next });
        }
        Ok(())
    }

    fn complete(&self) -> Result<(), TranscriptOrderError> {
        self.require_next(self.events.len())
    }
}

fn push_absorb(events: &mut Vec<RegistryEvent>, site: &str, kind: u8, count: usize, bytes: usize) {
    events.push(RegistryEvent {
        site: site.to_owned(),
        kind,
        atom_count: count,
        payload_size: PayloadSize::Exact(bytes),
        difficulty: None,
    });
}

fn push_challenge(
    events: &mut Vec<RegistryEvent>,
    site: &str,
    count: usize,
    bytes: usize,
    difficulty: u32,
) {
    events.push(RegistryEvent {
        site: site.to_owned(),
        kind: CHALLENGE_KIND,
        atom_count: count,
        payload_size: PayloadSize::Exact(bytes),
        difficulty: Some(difficulty),
    });
}

fn push_round(
    events: &mut Vec<RegistryEvent>,
    stage: &str,
    round: usize,
    count: usize,
    bytes: usize,
    difficulty: u32,
) {
    let message_site = format!("aerie/falcon/v1/{stage}/round/{round}/message");
    let challenge_site = format!("aerie/falcon/v1/{stage}/round/{round}/challenge");
    push_absorb(events, &message_site, PROOF_KIND, count, bytes);
    push_challenge(events, &challenge_site, 1, 16, difficulty);
}

#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct LengthPrefixedBytes<'a>(pub &'a [u8]);

impl Encoding<[u8]> for LengthPrefixedBytes<'_> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut bytes = Vec::with_capacity(8 + self.0.len());
        bytes.extend_from_slice(&(self.0.len() as u64).to_le_bytes());
        bytes.extend_from_slice(self.0);
        bytes
    }
}

/// Native transcript adapter implemented independently by each role.
#[doc(hidden)]
pub trait NativeState {
    fn public_message(&mut self, bytes: &[u8]);
    // Rejection-samples eight squeezed bytes, as Akita's spongefish sampler
    // does. Akita's `transcript-keccak` sampler also absorbs an attempt count
    // and is not reproduced here; that feature must stay off.
    fn sample_base(&mut self) -> Result<BaseField, TranscriptOrderError> {
        sample_base_by(|bytes| self.squeeze(bytes))
    }
    fn sponge_clone(&self) -> TranscriptSponge;
    fn squeeze(&mut self, bytes: &mut [u8]);
    fn acquire_work_nonce(
        &mut self,
        difficulty: u8,
        nonce_bits: u8,
        nonce_limit: u64,
    ) -> Result<(NonceAtom, Option<[u8; 32]>), TranscriptOrderError>;
    fn work_context(&mut self, _record: ProtocolContextRecord) {}
    fn emit_work_nonce(&mut self, nonce: NonceAtom);
    fn apply_work(&mut self, difficulty: u8) -> Result<(), TranscriptOrderError> {
        let (nonce_bits, nonce_limit) = work_nonce_range(difficulty)?;
        let nonzero_difficulty =
            NonZeroU8::new(difficulty).ok_or(TranscriptOrderError::WorkSchedule)?;
        let (nonce, preview) = self.acquire_work_nonce(difficulty, nonce_bits, nonce_limit)?;
        let (nonce_record, predicate_record) = work_context_records(difficulty, nonce_bits);
        self.work_context(nonce_record);
        self.emit_work_nonce(nonce);
        self.work_context(predicate_record);
        let mut predicate = [0; 32];
        self.squeeze(&mut predicate);
        if preview.is_some_and(|preview| predicate != preview) {
            return Err(TranscriptOrderError::WorkReplay { difficulty });
        }
        if !grinding_predicate_accepts(&predicate, nonzero_difficulty) {
            return Err(TranscriptOrderError::InvalidWork { difficulty });
        }
        Ok(())
    }
    fn check_work_eof(&self) -> Result<(), TranscriptOrderError>;
}

#[doc(hidden)]
pub struct NativeVerifier {
    state: VerifierState<'static, TranscriptSponge>,
    work: Vec<u8>,
    next: usize,
}

#[cfg(any(test, feature = "internals"))]
impl NativeVerifier {
    /// Wrap an independent verifier state for nonce-validation tests.
    #[doc(hidden)]
    #[must_use]
    pub fn from_state(state: VerifierState<'static, TranscriptSponge>, work: &[u8]) -> Self {
        Self {
            state,
            work: work.to_vec(),
            next: 0,
        }
    }
}
impl NativeState for NativeVerifier {
    fn public_message(&mut self, bytes: &[u8]) {
        self.state.public_message(bytes);
    }

    fn sponge_clone(&self) -> TranscriptSponge {
        self.state.duplex_sponge_state.clone()
    }

    fn squeeze(&mut self, bytes: &mut [u8]) {
        self.state.duplex_sponge_state.squeeze(bytes);
    }

    fn acquire_work_nonce(
        &mut self,
        _difficulty: u8,
        _nonce_bits: u8,
        nonce_limit: u64,
    ) -> Result<(NonceAtom, Option<[u8; 32]>), TranscriptOrderError> {
        let mut cursor = self
            .work
            .get(self.next..)
            .ok_or(TranscriptOrderError::NonceEncoding)?;
        let initial_len = cursor.len();
        let nonce = NonceAtom::deserialize_from_narg(&mut cursor)
            .map_err(|_| TranscriptOrderError::NonceEncoding)?;
        let value = nonce.into_inner();
        if u64::from(value) >= nonce_limit {
            return Err(TranscriptOrderError::NonceEncoding);
        }
        self.next += initial_len - cursor.len();
        Ok((nonce, None))
    }

    fn emit_work_nonce(&mut self, nonce: NonceAtom) {
        self.state.public_message(&nonce);
    }

    fn check_work_eof(&self) -> Result<(), TranscriptOrderError> {
        if self.next == self.work.len() {
            Ok(())
        } else {
            Err(TranscriptOrderError::TrailingBytes)
        }
    }
}

pub(crate) fn work_nonce_limit(difficulty: u8) -> Result<u64, TranscriptOrderError> {
    work_nonce_range(difficulty).map(|(_, limit)| limit)
}

fn work_nonce_range(difficulty: u8) -> Result<(u8, u64), TranscriptOrderError> {
    let nonce_bits = difficulty
        .checked_add(GRINDING_NONCE_SLACK_BITS)
        .ok_or(TranscriptOrderError::WorkSchedule)?;
    let nonce_limit = 1_u64
        .checked_shl(u32::from(nonce_bits))
        .ok_or(TranscriptOrderError::WorkSchedule)?;
    Ok((nonce_bits, nonce_limit))
}

#[doc(hidden)]
#[must_use]
pub fn work_context_records(
    difficulty: u8,
    nonce_bits: u8,
) -> (ProtocolContextRecord, ProtocolContextRecord) {
    let site = ProtocolSiteId {
        family: u32::from_le_bytes(*b"PIOP"),
        detail: u32::from(difficulty),
        ..ProtocolSiteId::default()
    };
    (
        ProtocolContextRecord::new(
            site.to_bytes(),
            ProtocolMessageKind::GrindingNonce as u32,
            1,
            nonce_max_bytes(nonce_bits) as u64,
            0,
        ),
        ProtocolContextRecord::new(
            site.to_bytes(),
            ProtocolMessageKind::GrindingPredicate as u32,
            0,
            0,
            32,
        ),
    )
}

/// Shared registry cursor, framing, sampler and work-validation state machine.
#[doc(hidden)]
pub struct TranscriptCore<S: NativeState> {
    state: S,
    cursor: EventCursor,
}

impl<S: NativeState> TranscriptCore<S> {
    pub fn new(
        manifest: &CommitmentManifest,
        record: &[u8; INSTANCE_BYTES],
        registry_record: &[u8; INSTANCE_BYTES],
        state: S,
    ) -> Result<Self, TranscriptOrderError> {
        let cursor = EventCursor::new(manifest, registry_record)?;
        let mut this = Self { state, cursor };
        this.absorb_record(record)?;
        Ok(this)
    }

    fn absorb_record(&mut self, record: &[u8; INSTANCE_BYTES]) -> Result<(), TranscriptOrderError> {
        self.absorb(PUBLIC_KIND, INSTANCE_LABEL, 1, record)
    }

    pub fn absorb(
        &mut self,
        kind: u8,
        site: &str,
        atom_count: usize,
        payload: &[u8],
    ) -> Result<(), TranscriptOrderError> {
        self.cursor.take(site, kind, atom_count, payload.len())?;
        // Kind-1 bytes are also carried by AggregateProof's canonical outer
        // serializer. No second Spongefish argument buffer is introduced.
        self.state.public_message(payload);
        Ok(())
    }

    pub fn challenge_extensions(
        &mut self,
        site: &str,
        count: usize,
    ) -> Result<Vec<OpeningField>, TranscriptOrderError> {
        let payload_bytes = 16usize
            .checked_mul(count)
            .ok_or(TranscriptOrderError::EventShape)?;
        let event = self
            .cursor
            .take(site, CHALLENGE_KIND, count, payload_bytes)?;
        let difficulty = event.difficulty.ok_or(TranscriptOrderError::EventShape)?;
        if difficulty != 0 {
            let difficulty =
                u8::try_from(difficulty).map_err(|_| TranscriptOrderError::WorkSchedule)?;
            self.state.apply_work(difficulty)?;
        }

        (0..count)
            .map(|_| {
                let c0 = self.state.sample_base()?;
                let c1 = self.state.sample_base()?;
                Ok(OpeningField::from_base_slice(&[c0, c1]))
            })
            .collect()
    }

    pub fn projection_seed(&mut self) -> Result<[u8; 32], TranscriptOrderError> {
        let event = self
            .cursor
            .take(PROJECTION_SEED_SITE, CHALLENGE_KIND, 1, 32)?;
        if event.difficulty != Some(0) {
            return Err(TranscriptOrderError::EventShape);
        }
        let mut seed = [0; 32];
        self.state.squeeze(&mut seed);
        Ok(seed)
    }

    pub fn pcs_session(&mut self) -> Result<[u8; 32], TranscriptOrderError> {
        self.cursor.take(PCS_SESSION_SITE, EXPORT_KIND, 1, 32)?;
        // The terminal registry position follows the stage-three payload.
        // Export only a clone for the separately domain-separated native PCS.
        let mut sponge = self.state.sponge_clone();
        let mut session = [0; 32];
        sponge.squeeze(&mut session);
        Ok(session)
    }

    /// Confirm the expected registry boundary without exposing its cursor.
    pub fn require_next(&self, index: usize) -> Result<(), TranscriptOrderError> {
        self.cursor.require_next(index)
    }
    /// Confirm all registered events were consumed.
    pub fn ensure_complete(&self) -> Result<(), TranscriptOrderError> {
        self.cursor.complete()
    }
    /// Borrow the role-owned channel, without a second adapter buffer.
    pub const fn state(&self) -> &S {
        &self.state
    }
    /// Mutably borrow the role-owned channel.
    pub fn state_mut(&mut self) -> &mut S {
        &mut self.state
    }
}

/// Verifier-owned native PIOP transcript.
pub struct VerifierProtocolTranscript {
    core: TranscriptCore<NativeVerifier>,
}

fn sample_base_by(mut squeeze: impl FnMut(&mut [u8])) -> Result<BaseField, TranscriptOrderError> {
    loop {
        let mut candidate = [0; 8];
        squeeze(&mut candidate);
        if let Some(value) = BaseField::from_bytes_le_checked(&candidate) {
            return Ok(value);
        }
    }
}

impl VerifierProtocolTranscript {
    /// Initialize a verifier PIOP stream from the independently derived
    /// canonical instance record and absorb registry event 0.
    pub fn new(
        manifest: &CommitmentManifest,
        record: &[u8; INSTANCE_BYTES],
        work: &[u8],
    ) -> Result<Self, TranscriptOrderError> {
        let digest = crate::protocol::instance::digest(INSTANCE_LABEL.as_bytes(), record);
        let session = LengthPrefixedBytes(crate::PROTOCOL_DOMAIN);
        let instance = LengthPrefixedBytes(&digest);
        let domain = DomainSeparator::<WithoutInstance>::new(PIOP_PROTOCOL_TAG)
            .session(session)
            .instance(&instance);
        let state = NativeVerifier {
            state: domain.to_verifier(TranscriptSponge::default(), &[]),
            work: work.to_vec(),
            next: 0,
        };
        Ok(Self {
            core: TranscriptCore::new(manifest, record, record, state)?,
        })
    }

    /// Absorb the next canonical group commitment.
    pub fn absorb_group_commitment(
        &mut self,
        role: GroupRole,
        payload: &[u8],
    ) -> Result<(), TranscriptOrderError> {
        self.core
            .absorb(PROOF_KIND, commitment_site(role), 16, payload)
    }

    /// Sample the 32-byte clear-projection seed after all four commitments.
    pub fn derive_projection_seed(&mut self) -> Result<[u8; 32], TranscriptOrderError> {
        self.core.projection_seed()
    }

    /// Keep the native stream while confirming the registry reached the first
    /// stage event after the complete commitment prefix.
    pub fn finish(self) -> Result<Self, TranscriptOrderError> {
        self.core.cursor.require_next(6)?;
        Ok(self)
    }

    /// Export the PCS session after all stage-three terminal bytes.
    pub fn pcs_session(&mut self) -> Result<[u8; 32], TranscriptOrderError> {
        self.core.pcs_session()
    }

    /// Confirm that every registry event, including the PCS export, ran.
    pub fn ensure_complete(&self) -> Result<(), TranscriptOrderError> {
        self.core.cursor.complete()?;
        self.core.state.check_work_eof()
    }
}

#[doc(hidden)]
#[must_use]
pub fn commitment_site(role: GroupRole) -> &'static str {
    match role {
        GroupRole::S1 => "aerie/falcon/v1/commit/s1",
        GroupRole::Slack => "aerie/falcon/v1/commit/slack",
        GroupRole::S2Sources => "aerie/falcon/v1/commit/s2-sources",
        GroupRole::Budget => "aerie/falcon/v1/commit/budget",
    }
}

/// Challenge operation shared with isolated algebra unit tests. Production
/// protocol stages receive [`PioTranscript`], whose implementations validate
/// this call against the full ordered registry.
pub trait ChallengeSource {
    /// Draw a vector of independent extension elements under one kind-2 event.
    fn challenge_extensions(
        &mut self,
        site: &str,
        count: usize,
    ) -> Result<Vec<OpeningField>, TranscriptOrderError>;
}

/// Shared stage interface. The ordered registry cursor rejects a missing,
/// extra, or out-of-order event before it reaches the sponge.
pub trait PioTranscript: ChallengeSource {
    /// Absorb one canonical kind-1 PIOP payload as one byte-string message.
    fn absorb_proof(
        &mut self,
        site: &str,
        atom_count: usize,
        payload: &[u8],
    ) -> Result<(), TranscriptOrderError>;

    /// Absorb an extension-element vector as one payload event.
    fn absorb_extensions(
        &mut self,
        site: &str,
        values: &[OpeningField],
    ) -> Result<(), TranscriptOrderError> {
        let payload = encode_extensions(values);
        self.absorb_proof(site, values.len(), &payload)
    }
}

impl ChallengeSource for VerifierProtocolTranscript {
    fn challenge_extensions(
        &mut self,
        site: &str,
        count: usize,
    ) -> Result<Vec<OpeningField>, TranscriptOrderError> {
        self.core.challenge_extensions(site, count)
    }
}

impl PioTranscript for VerifierProtocolTranscript {
    fn absorb_proof(
        &mut self,
        site: &str,
        atom_count: usize,
        payload: &[u8],
    ) -> Result<(), TranscriptOrderError> {
        self.core.absorb(PROOF_KIND, site, atom_count, payload)
    }
}

/// Canonical extension atom encoding: LE64(c0) || LE64(c1) per element.
#[must_use]
pub fn encode_extensions(values: &[OpeningField]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 * values.len());
    for value in values {
        for coefficient in value.to_base_vec() {
            bytes.extend_from_slice(&coefficient.to_canonical_u64().to_le_bytes());
        }
    }
    bytes
}

/// Standalone transcript used by isolated arithmetic unit tests. It absorbs
/// headerless payloads but has no production registry, so it is never used
/// by proving or verification entry points.
#[cfg(any(test, feature = "internals"))]
pub struct TestTranscript {
    state: NativeVerifier,
}

#[cfg(any(test, feature = "internals"))]
impl TestTranscript {
    #[must_use]
    pub fn new(label: &[u8]) -> Self {
        let session = LengthPrefixedBytes(b"aerie/falcon/test");
        let instance = LengthPrefixedBytes(label);
        let domain = DomainSeparator::<WithoutInstance>::new(PIOP_PROTOCOL_TAG)
            .session(session)
            .instance(&instance);
        Self {
            state: NativeVerifier {
                state: domain.to_verifier(TranscriptSponge::default(), &[]),
                work: Vec::new(),
                next: 0,
            },
        }
    }
}

#[cfg(any(test, feature = "internals"))]
impl ChallengeSource for TestTranscript {
    fn challenge_extensions(
        &mut self,
        _site: &str,
        count: usize,
    ) -> Result<Vec<OpeningField>, TranscriptOrderError> {
        (0..count)
            .map(|_| {
                let c0 = self.state.sample_base()?;
                let c1 = self.state.sample_base()?;
                Ok(OpeningField::from_base_slice(&[c0, c1]))
            })
            .collect()
    }
}

#[cfg(any(test, feature = "internals"))]
impl PioTranscript for TestTranscript {
    fn absorb_proof(
        &mut self,
        _site: &str,
        _atom_count: usize,
        payload: &[u8],
    ) -> Result<(), TranscriptOrderError> {
        self.state.public_message(payload);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::FalconDegree;
    use jolt_field::Ring as _;

    struct InjectedState {
        candidates: [u64; 3],
        next: usize,
        consumed_bytes: usize,
    }

    impl NativeState for InjectedState {
        fn public_message(&mut self, _bytes: &[u8]) {}

        fn sponge_clone(&self) -> TranscriptSponge {
            TranscriptSponge::default()
        }

        fn squeeze(&mut self, bytes: &mut [u8]) {
            bytes.copy_from_slice(&self.candidates[self.next].to_le_bytes());
            self.next += 1;
            self.consumed_bytes += 8;
        }

        fn acquire_work_nonce(
            &mut self,
            _difficulty: u8,
            _nonce_bits: u8,
            _nonce_limit: u64,
        ) -> Result<(NonceAtom, Option<[u8; 32]>), TranscriptOrderError> {
            unreachable!("the injected sampler fixture has zero difficulty")
        }

        fn emit_work_nonce(&mut self, _nonce: NonceAtom) {
            unreachable!("the injected sampler fixture has zero difficulty")
        }

        fn check_work_eof(&self) -> Result<(), TranscriptOrderError> {
            Ok(())
        }
    }

    #[test]
    fn exact_sampler_rejects_eight_bytes_and_does_not_reuse_between_limbs() {
        let modulus = u64::MAX - 58;
        let state = InjectedState {
            candidates: [modulus, 7, 11],
            next: 0,
            consumed_bytes: 0,
        };
        let cursor = EventCursor {
            events: vec![RegistryEvent {
                site: "injected/challenge".to_owned(),
                kind: CHALLENGE_KIND,
                atom_count: 1,
                payload_size: PayloadSize::Exact(16),
                difficulty: Some(0),
            }],
            next: 0,
        };
        let mut core = TranscriptCore { state, cursor };
        let value = core.challenge_extensions("injected/challenge", 1).unwrap()[0];
        let limbs = value.to_base_vec();
        assert_eq!(limbs, [BaseField::from_u64(7), BaseField::from_u64(11)]);
        assert_eq!(core.state.consumed_bytes, 24);
    }

    #[test]
    fn event_table_has_the_expected_order_and_shape_for_both_degrees() {
        for (degree, live, expected_rounds) in [
            (FalconDegree::Falcon512, 8, 12),
            (FalconDegree::Falcon1024, 4, 12),
        ] {
            let manifest = CommitmentManifest::for_batch_of(degree, live).unwrap();
            let record = test_record(&manifest);
            let cursor = EventCursor::new(&manifest, &record).unwrap();
            let plan = PiopGrindingPlan::for_shape(degree, manifest.padded_records()).unwrap();
            assert_eq!(cursor.events[0].site, INSTANCE_LABEL);
            let mut expected = vec![(INSTANCE_LABEL.to_owned(), 0, 1, INSTANCE_BYTES)];
            for site in [
                "aerie/falcon/v1/commit/s1",
                "aerie/falcon/v1/commit/slack",
                "aerie/falcon/v1/commit/s2-sources",
                "aerie/falcon/v1/commit/budget",
            ] {
                expected.push((site.to_owned(), 1, 16, 128));
            }
            expected.extend([
                (PROJECTION_SEED_SITE.to_owned(), 2, 1, 32),
                (PROJECTION_OUTPUT_SITE.to_owned(), 1, 1, usize::MAX),
                (MU_SITE.to_owned(), 2, 8, 128),
                (
                    TAU_SITE.to_owned(),
                    2,
                    expected_rounds,
                    16 * expected_rounds,
                ),
            ]);
            let basis_count = StageShapes::for_degree(degree).basis16_sources;
            expected.push((
                STAGE1_GAMMA_SITE.to_owned(),
                2,
                basis_count + 1,
                16 * (basis_count + 1),
            ));
            push_expected_rounds(&mut expected, "stage1", expected_rounds, 3, 48);
            expected.push((
                STAGE1_TERMINAL_SITE.to_owned(),
                1,
                2 * basis_count + 3,
                16 * (2 * basis_count + 3),
            ));
            let stage2_count = if degree == FalconDegree::Falcon512 {
                8
            } else {
                9
            };
            expected.push((
                STAGE2_GAMMA_SITE.to_owned(),
                2,
                stage2_count,
                16 * stage2_count,
            ));
            push_expected_rounds(&mut expected, "stage2", expected_rounds, 4, 64);
            expected.push((STAGE2_TERMINAL_SITE.to_owned(), 1, 6, 96));
            expected.push((STAGE3_GAMMA_SITE.to_owned(), 2, 12, 192));
            push_expected_rounds(&mut expected, "stage3", expected_rounds, 3, 48);
            expected.push((STAGE3_TERMINAL_SITE.to_owned(), 1, 9, 144));
            expected.push((PCS_SESSION_SITE.to_owned(), 4, 1, 32));

            assert_eq!(cursor.events.len(), expected.len());
            let mut challenge_difficulties = vec![0, plan.mu, plan.tau, plan.gamma[0]];
            challenge_difficulties.extend(std::iter::repeat_n(plan.sumcheck[0], expected_rounds));
            challenge_difficulties.push(plan.gamma[1]);
            challenge_difficulties.extend(std::iter::repeat_n(plan.sumcheck[1], expected_rounds));
            challenge_difficulties.push(plan.gamma[2]);
            challenge_difficulties.extend(std::iter::repeat_n(plan.sumcheck[2], expected_rounds));
            let mut challenge_difficulties = challenge_difficulties.into_iter();
            for (index, (event, (site, kind, count, bytes))) in
                cursor.events.iter().zip(expected).enumerate()
            {
                assert_eq!(event.site, site, "site at registry event {index}");
                assert_eq!(event.kind, kind, "kind at registry event {index}");
                assert_eq!(
                    event.atom_count, count,
                    "atom count at registry event {index}"
                );
                if bytes == usize::MAX {
                    assert_eq!(event.payload_size, PayloadSize::RiceSection);
                } else {
                    assert_eq!(event.payload_size, PayloadSize::Exact(bytes));
                }
                assert_eq!(
                    event.difficulty,
                    (kind == CHALLENGE_KIND).then(|| challenge_difficulties.next().unwrap())
                );
            }
            assert_eq!(challenge_difficulties.next(), None);
        }
    }

    fn push_expected_rounds(
        expected: &mut Vec<(String, u8, usize, usize)>,
        stage: &str,
        rounds: usize,
        count: usize,
        bytes: usize,
    ) {
        for round in 0..rounds {
            expected.push((
                format!("aerie/falcon/v1/{stage}/round/{round}/message"),
                1,
                count,
                bytes,
            ));
            expected.push((
                format!("aerie/falcon/v1/{stage}/round/{round}/challenge"),
                2,
                1,
                16,
            ));
        }
    }

    fn test_record(manifest: &CommitmentManifest) -> [u8; INSTANCE_BYTES] {
        let binding = manifest.binding_bytes();
        let mut record = [0; INSTANCE_BYTES];
        record[..27].copy_from_slice(&binding[..27]);
        record[60..302].copy_from_slice(&binding[27..]);
        record
    }
}
