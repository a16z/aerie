//! Native Spongefish transcript and the current version-2 PIOP event registry.
//!
//! Version 1 freezes at the first public release. Until then, this registry may
//! change without a wire-version bump.

use akita_error::AkitaError;
use akita_transcript::{
    commit_grinding_nonce, grinding_predicate_accepts, nonce_max_bytes, prover_field_challenge,
    search_grinding_nonce, NonceAtom, ProtocolContextRecord, ProtocolMessageKind, ProtocolSiteId,
    ProverChannel, TranscriptSponge, GRINDING_NONCE_SLACK_BITS,
};
use jolt_field::{CanonicalEncoding, ExtField};
use spongefish::{
    DomainSeparator, DuplexSpongeInterface, Encoding, NargDeserialize, VerifierState,
    WithoutInstance,
};
use std::num::NonZeroU8;

use super::{
    grinding::PiopGrindingPlan,
    instance::INSTANCE_BYTES,
    manifest::{CommitmentManifest, GroupRole, GROUP_COUNT},
    FalconDegree,
};
use crate::substrate::{BaseField, OpeningField};

const INSTANCE_LABEL: &str = "aerie/falcon/v2/instance";
const PROJECTION_SEED_SITE: &str = "aerie/falcon/v2/projection/seed";
const PCS_SESSION_SITE: &str = "aerie/falcon/v2/pcs/session";
const PROOF_KIND: u8 = 1;
const PUBLIC_KIND: u8 = 0;
const CHALLENGE_KIND: u8 = 2;
const WORK_KIND: u8 = 3;
const EXPORT_KIND: u8 = 4;

const PROJECTION_OUTPUT_SITE: &str = "aerie/falcon/v2/projection/output";
const MU_SITE: &str = "aerie/falcon/v2/mu";
const TAU_SITE: &str = "aerie/falcon/v2/tau";
const STAGE1_GAMMA_SITE: &str = "aerie/falcon/v2/stage1/gamma";
const STAGE1_TERMINAL_SITE: &str = "aerie/falcon/v2/stage1/terminal";
const STAGE2_GAMMA_SITE: &str = "aerie/falcon/v2/stage2/gamma";
const STAGE2_TERMINAL_SITE: &str = "aerie/falcon/v2/stage2/terminal";
const STAGE3_GAMMA_SITE: &str = "aerie/falcon/v2/stage3/gamma";
const STAGE3_TERMINAL_SITE: &str = "aerie/falcon/v2/stage3/terminal";

/// Spongefish protocol tag for Aerie's PIOP stream, zero-padded to 64 bytes.
pub const PIOP_PROTOCOL_TAG: [u8; 64] = protocol_tag();

const fn protocol_tag() -> [u8; 64] {
    let source = b"aerie/falcon/v2/piop/blake2b";
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
    EventOrder {
        index: usize,
        expected: String,
        got: String,
    },
    /// The event's atom count or canonical payload size is incorrect.
    EventShape { site: String },
    /// A scheduled proof-of-work transition failed.
    InvalidWork { site: String },
    /// The pinned sampler rejected a field encoding configuration.
    Sampler,
    /// A required suffix of the registry was not consumed.
    Incomplete { next: String },
}

impl core::fmt::Display for TranscriptOrderError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InstanceShape => {
                formatter.write_str("transcript instance does not match manifest")
            }
            Self::EventOrder {
                index,
                expected,
                got,
            } => write!(
                formatter,
                "PIOP event {index} is {got}, registry requires {expected}"
            ),
            Self::EventShape { site } => write!(formatter, "invalid PIOP event shape at {site}"),
            Self::InvalidWork { site } => write!(formatter, "invalid PIOP work at {site}"),
            Self::Sampler => formatter.write_str("native exact field sampler failed"),
            Self::Incomplete { next } => write!(formatter, "PIOP transcript ended before {next}"),
        }
    }
}

impl std::error::Error for TranscriptOrderError {}

impl From<TranscriptOrderError> for AkitaError {
    fn from(error: TranscriptOrderError) -> Self {
        Self::InvalidInput(error.to_string())
    }
}

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
        if binding.len() != 27 + 198 + 11 * GROUP_COUNT
            || record[..27] != binding[..27]
            || record[60..302] != binding[27..]
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
        let degree_value = degree.degree();
        let basis_count = 2 + usize::from(degree == FalconDegree::Falcon1024);
        let stage1_combiner_count = basis_count + 1;
        let stage2_combiner_count = if degree_value == 512 { 8 } else { 9 };

        let mut events = Vec::with_capacity(21 + 6 * rounds);
        push_absorb(&mut events, INSTANCE_LABEL, PUBLIC_KIND, 1, INSTANCE_BYTES);
        for site in [
            "aerie/falcon/v2/commit/s1",
            "aerie/falcon/v2/commit/slack",
            "aerie/falcon/v2/commit/s2-sources",
            "aerie/falcon/v2/commit/budget",
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
            stage1_combiner_count,
            16 * stage1_combiner_count,
            plan.gamma[0],
        );
        for round in 0..rounds {
            push_round(&mut events, "stage1", round, 3, 48, plan.sumcheck[0]);
        }
        push_absorb(
            &mut events,
            STAGE1_TERMINAL_SITE,
            PROOF_KIND,
            2 * basis_count + 3,
            16 * (2 * basis_count + 3),
        );
        push_challenge(
            &mut events,
            STAGE2_GAMMA_SITE,
            stage2_combiner_count,
            16 * stage2_combiner_count,
            plan.gamma[1],
        );
        for round in 0..rounds {
            push_round(&mut events, "stage2", round, 4, 64, plan.sumcheck[1]);
        }
        push_absorb(&mut events, STAGE2_TERMINAL_SITE, PROOF_KIND, 6, 96);
        push_challenge(&mut events, STAGE3_GAMMA_SITE, 12, 192, plan.gamma[2]);
        for round in 0..rounds {
            push_round(&mut events, "stage3", round, 3, 48, plan.sumcheck[2]);
        }
        push_absorb(&mut events, STAGE3_TERMINAL_SITE, PROOF_KIND, 9, 144);
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
            return Err(TranscriptOrderError::Incomplete {
                next: "no further event".to_owned(),
            });
        };
        let shape_matches = expected.site == site
            && expected.kind == kind
            && expected.atom_count == atom_count
            && match expected.payload_size {
                PayloadSize::Exact(bytes) => bytes == payload_bytes,
                PayloadSize::RiceSection => payload_bytes >= 2,
            };
        if !shape_matches {
            return Err(TranscriptOrderError::EventOrder {
                index: self.next,
                expected: format!(
                    "{} (kind {}, count {}, bytes {:?})",
                    expected.site, expected.kind, expected.atom_count, expected.payload_size
                ),
                got: format!("{site} (kind {kind}, count {atom_count}, bytes {payload_bytes})"),
            });
        }
        self.next += 1;
        Ok(expected)
    }

    fn require_next(&self, index: usize) -> Result<(), TranscriptOrderError> {
        if self.next != index {
            let next = self
                .events
                .get(self.next)
                .map_or_else(|| "no further event".to_owned(), |event| event.site.clone());
            return Err(TranscriptOrderError::Incomplete { next });
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
    let message_site = format!("aerie/falcon/v2/{stage}/round/{round}/message");
    let challenge_site = format!("aerie/falcon/v2/{stage}/round/{round}/challenge");
    push_absorb(events, &message_site, PROOF_KIND, count, bytes);
    push_challenge(events, &challenge_site, 1, 16, difficulty);
}

/// Encode one derived event header. Payload bytes themselves are absorbed as
/// one separate Spongefish public message by the registry cursor.
#[must_use]
pub fn event_header(kind: u8, site: &[u8], atom_count: u64, payload_bytes: u64) -> Vec<u8> {
    let mut header = Vec::with_capacity(25 + site.len());
    header.push(kind);
    header.extend_from_slice(&(site.len() as u64).to_le_bytes());
    header.extend_from_slice(site);
    header.extend_from_slice(&atom_count.to_le_bytes());
    header.extend_from_slice(&payload_bytes.to_le_bytes());
    header
}

#[derive(Clone, Copy)]
struct LengthPrefixedBytes<'a>(&'a [u8]);

impl Encoding<[u8]> for LengthPrefixedBytes<'_> {
    fn encode(&self) -> impl AsRef<[u8]> {
        let mut bytes = Vec::with_capacity(8 + self.0.len());
        bytes.extend_from_slice(&(self.0.len() as u64).to_le_bytes());
        bytes.extend_from_slice(self.0);
        bytes
    }
}

trait NativeState {
    fn public_message(&mut self, bytes: &[u8]);
    fn sample_base(&mut self) -> Result<BaseField, TranscriptOrderError>;
    fn sponge_clone(&self) -> TranscriptSponge;
    fn squeeze(&mut self, bytes: &mut [u8]);
    fn apply_work(&mut self, site: &str, difficulty: u8) -> Result<(), TranscriptOrderError>;
    fn check_work_eof(&self) -> Result<(), TranscriptOrderError>;
}

impl NativeState for ProverChannel {
    fn public_message(&mut self, bytes: &[u8]) {
        self.public_message(bytes);
    }

    fn sample_base(&mut self) -> Result<BaseField, TranscriptOrderError> {
        prover_field_challenge(self).map_err(|_| TranscriptOrderError::Sampler)
    }

    fn sponge_clone(&self) -> TranscriptSponge {
        self.duplex_sponge_state.clone()
    }

    fn squeeze(&mut self, bytes: &mut [u8]) {
        self.duplex_sponge_state.squeeze(bytes);
    }

    fn apply_work(&mut self, site: &str, difficulty: u8) -> Result<(), TranscriptOrderError> {
        let nonce_bits = difficulty
            .checked_add(GRINDING_NONCE_SLACK_BITS)
            .ok_or_else(|| TranscriptOrderError::InvalidWork {
                site: site.to_owned(),
            })?;
        let (nonce, preview) =
            search_grinding_nonce(self, difficulty, nonce_bits).ok_or_else(|| {
                TranscriptOrderError::InvalidWork {
                    site: site.to_owned(),
                }
            })?;
        let (nonce_record, predicate_record) = work_context_records(difficulty, nonce_bits);
        let predicate = commit_grinding_nonce(self, nonce_record, nonce, predicate_record);
        if predicate != preview {
            return Err(TranscriptOrderError::InvalidWork {
                site: site.to_owned(),
            });
        }
        Ok(())
    }

    fn check_work_eof(&self) -> Result<(), TranscriptOrderError> {
        Ok(())
    }
}

struct NativeVerifier {
    state: VerifierState<'static, TranscriptSponge>,
    work: Vec<u8>,
    next: usize,
}

impl NativeState for NativeVerifier {
    fn public_message(&mut self, bytes: &[u8]) {
        self.state.public_message(bytes);
    }

    fn sample_base(&mut self) -> Result<BaseField, TranscriptOrderError> {
        // Akita exposes this exact loop through its verifier wrapper, but that
        // wrapper's constructor is private and bound to the PCS domain. Apply
        // the same canonical 8-byte rejection rule to the Aerie-domain state.
        sample_base_by(|bytes| {
            let candidate: [u8; 8] = self.state.verifier_message();
            bytes.copy_from_slice(&candidate);
        })
    }

    fn sponge_clone(&self) -> TranscriptSponge {
        self.state.duplex_sponge_state.clone()
    }

    fn squeeze(&mut self, bytes: &mut [u8]) {
        self.state.duplex_sponge_state.squeeze(bytes);
    }

    fn apply_work(&mut self, site: &str, difficulty: u8) -> Result<(), TranscriptOrderError> {
        let nonce_bits = difficulty
            .checked_add(GRINDING_NONCE_SLACK_BITS)
            .ok_or_else(|| TranscriptOrderError::InvalidWork {
                site: site.to_owned(),
            })?;
        let mut cursor =
            self.work
                .get(self.next..)
                .ok_or_else(|| TranscriptOrderError::InvalidWork {
                    site: site.to_owned(),
                })?;
        let initial_len = cursor.len();
        let nonce = NonceAtom::deserialize_from_narg(&mut cursor).map_err(|_| {
            TranscriptOrderError::InvalidWork {
                site: site.to_owned(),
            }
        })?;
        let value = nonce.into_inner();
        if u64::from(value) >= (1_u64 << nonce_bits) {
            return Err(TranscriptOrderError::InvalidWork {
                site: site.to_owned(),
            });
        }
        self.next += initial_len - cursor.len();
        self.state.public_message(&nonce);
        let predicate: [u8; 32] = self.state.verifier_message();
        let difficulty =
            NonZeroU8::new(difficulty).ok_or_else(|| TranscriptOrderError::InvalidWork {
                site: site.to_owned(),
            })?;
        if !grinding_predicate_accepts(&predicate, difficulty) {
            return Err(TranscriptOrderError::InvalidWork {
                site: site.to_owned(),
            });
        }
        Ok(())
    }

    fn check_work_eof(&self) -> Result<(), TranscriptOrderError> {
        if self.next == self.work.len() {
            Ok(())
        } else {
            Err(TranscriptOrderError::InvalidWork {
                site: "trailing PIOP work bytes".to_owned(),
            })
        }
    }
}

fn work_context_records(
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

struct TranscriptCore<S: NativeState> {
    state: S,
    cursor: EventCursor,
    #[cfg(test)]
    next_site_override: Option<Vec<u8>>,
}

impl<S: NativeState> TranscriptCore<S> {
    fn new(
        manifest: &CommitmentManifest,
        record: &[u8; INSTANCE_BYTES],
        registry_record: &[u8; INSTANCE_BYTES],
        state: S,
    ) -> Result<Self, TranscriptOrderError> {
        let cursor = EventCursor::new(manifest, registry_record)?;
        let mut this = Self {
            state,
            cursor,
            #[cfg(test)]
            next_site_override: None,
        };
        this.absorb_record(record)?;
        Ok(this)
    }

    fn absorb_record(&mut self, record: &[u8; INSTANCE_BYTES]) -> Result<(), TranscriptOrderError> {
        self.absorb(PUBLIC_KIND, INSTANCE_LABEL, 1, record)
    }

    fn absorb(
        &mut self,
        kind: u8,
        site: &str,
        atom_count: usize,
        payload: &[u8],
    ) -> Result<(), TranscriptOrderError> {
        self.cursor.take(site, kind, atom_count, payload.len())?;
        #[cfg(test)]
        let site = self
            .next_site_override
            .take()
            .unwrap_or_else(|| site.as_bytes().to_vec());
        #[cfg(not(test))]
        let site = site.as_bytes().to_vec();
        let header = event_header(kind, &site, atom_count as u64, payload.len() as u64);
        self.state.public_message(&header);
        // Kind-1 bytes are also carried by AggregateProof's canonical outer
        // serializer. No second Spongefish argument buffer is introduced.
        self.state.public_message(payload);
        Ok(())
    }

    fn challenge_extensions(
        &mut self,
        site: &str,
        count: usize,
    ) -> Result<Vec<OpeningField>, TranscriptOrderError> {
        let payload_bytes =
            16usize
                .checked_mul(count)
                .ok_or_else(|| TranscriptOrderError::EventShape {
                    site: site.to_owned(),
                })?;
        let event = self
            .cursor
            .take(site, CHALLENGE_KIND, count, payload_bytes)?;
        let difficulty = event
            .difficulty
            .ok_or_else(|| TranscriptOrderError::EventShape {
                site: site.to_owned(),
            })?;
        let header = event_header(
            CHALLENGE_KIND,
            site.as_bytes(),
            count as u64,
            payload_bytes as u64,
        );
        self.state.public_message(&header);
        if difficulty != 0 {
            let difficulty =
                u8::try_from(difficulty).map_err(|_| TranscriptOrderError::InvalidWork {
                    site: site.to_owned(),
                })?;
            let nonce_bits = difficulty
                .checked_add(GRINDING_NONCE_SLACK_BITS)
                .ok_or_else(|| TranscriptOrderError::InvalidWork {
                    site: site.to_owned(),
                })?;
            let work_site = format!("{site}/work");
            let work_header = event_header(
                WORK_KIND,
                work_site.as_bytes(),
                1,
                nonce_max_bytes(nonce_bits) as u64,
            );
            self.state.public_message(&work_header);
            self.state.apply_work(&work_site, difficulty)?;
        }

        (0..count)
            .map(|_| {
                let c0 = self.state.sample_base()?;
                let c1 = self.state.sample_base()?;
                Ok(OpeningField::from_base_slice(&[c0, c1]))
            })
            .collect()
    }

    fn projection_seed(&mut self) -> Result<[u8; 32], TranscriptOrderError> {
        let event = self
            .cursor
            .take(PROJECTION_SEED_SITE, CHALLENGE_KIND, 1, 32)?;
        if event.difficulty != Some(0) {
            return Err(TranscriptOrderError::EventShape {
                site: PROJECTION_SEED_SITE.to_owned(),
            });
        }
        let header = event_header(CHALLENGE_KIND, PROJECTION_SEED_SITE.as_bytes(), 1, 32);
        self.state.public_message(&header);
        let mut seed = [0; 32];
        self.state.squeeze(&mut seed);
        Ok(seed)
    }

    fn pcs_session(&mut self) -> Result<[u8; 32], TranscriptOrderError> {
        let event = self.cursor.take(PCS_SESSION_SITE, EXPORT_KIND, 1, 32)?;
        if event.kind != EXPORT_KIND {
            return Err(TranscriptOrderError::EventShape {
                site: PCS_SESSION_SITE.to_owned(),
            });
        }
        let header = event_header(EXPORT_KIND, PCS_SESSION_SITE.as_bytes(), 1, 32);
        let mut sponge = self.state.sponge_clone();
        sponge.absorb(&header);
        let mut session = [0; 32];
        sponge.squeeze(&mut session);
        Ok(session)
    }
}

/// Prover-owned native PIOP transcript. Prover and verifier use this same
/// registry but maintain independent Spongefish states.
pub struct ProtocolTranscript {
    core: TranscriptCore<ProverChannel>,
}

/// Verifier-owned native PIOP transcript.
pub struct VerifierProtocolTranscript {
    core: TranscriptCore<NativeVerifier>,
}

impl ProtocolTranscript {
    /// Initialize a prover PIOP stream from the canonical instance record and
    /// absorb registry event 0.
    pub fn new(
        manifest: &CommitmentManifest,
        record: &[u8; INSTANCE_BYTES],
    ) -> Result<Self, TranscriptOrderError> {
        Self::new_inner(manifest, record, record)
    }

    #[cfg(test)]
    pub(crate) fn new_with_test_record(
        manifest: &CommitmentManifest,
        record: &[u8; INSTANCE_BYTES],
        registry_record: &[u8; INSTANCE_BYTES],
    ) -> Result<Self, TranscriptOrderError> {
        Self::new_inner(manifest, record, registry_record)
    }

    fn new_inner(
        manifest: &CommitmentManifest,
        record: &[u8; INSTANCE_BYTES],
        registry_record: &[u8; INSTANCE_BYTES],
    ) -> Result<Self, TranscriptOrderError> {
        let digest = crate::protocol::instance::digest(INSTANCE_LABEL.as_bytes(), record);
        let session = LengthPrefixedBytes(crate::PROTOCOL_DOMAIN);
        let instance = LengthPrefixedBytes(&digest);
        let domain = DomainSeparator::<WithoutInstance>::new(PIOP_PROTOCOL_TAG)
            .session(session)
            .instance(&instance);
        let state = domain.to_prover(TranscriptSponge::default());
        Ok(Self {
            core: TranscriptCore::new(manifest, record, registry_record, state)?,
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

    /// Return the inline canonical PIOP work argument bytes.
    #[must_use]
    pub fn work_bytes(&self) -> &[u8] {
        self.core.state.narg_string()
    }

    #[cfg(test)]
    pub(crate) fn override_next_site_for_test(&mut self, site: &[u8]) {
        self.core.next_site_override = Some(site.to_vec());
    }

    #[cfg(test)]
    pub(crate) fn alter_export_prefix_for_test(&mut self) {
        self.core
            .state
            .public_message(b"test-only altered PIOP prefix");
    }
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

fn commitment_site(role: GroupRole) -> &'static str {
    match role {
        GroupRole::S1 => "aerie/falcon/v2/commit/s1",
        GroupRole::Slack => "aerie/falcon/v2/commit/slack",
        GroupRole::S2Sources => "aerie/falcon/v2/commit/s2-sources",
        GroupRole::Budget => "aerie/falcon/v2/commit/budget",
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

impl ChallengeSource for ProtocolTranscript {
    fn challenge_extensions(
        &mut self,
        site: &str,
        count: usize,
    ) -> Result<Vec<OpeningField>, TranscriptOrderError> {
        self.core.challenge_extensions(site, count)
    }
}

impl PioTranscript for ProtocolTranscript {
    fn absorb_proof(
        &mut self,
        site: &str,
        atom_count: usize,
        payload: &[u8],
    ) -> Result<(), TranscriptOrderError> {
        self.core.absorb(PROOF_KIND, site, atom_count, payload)
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
pub(crate) fn encode_extensions(values: &[OpeningField]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 * values.len());
    for value in values {
        for coefficient in value.to_base_vec() {
            bytes.extend_from_slice(&coefficient.to_canonical_u64().to_le_bytes());
        }
    }
    bytes
}

/// Standalone transcript used by isolated arithmetic unit tests. It emits
/// framed byte messages but has no production registry, so it is never used
/// by proving or verification entry points.
#[cfg(test)]
pub(crate) struct TestTranscript {
    state: ProverChannel,
}

#[cfg(test)]
impl TestTranscript {
    pub(crate) fn new(label: &[u8]) -> Self {
        let session = LengthPrefixedBytes(b"aerie/falcon/test");
        let instance = LengthPrefixedBytes(label);
        let domain = DomainSeparator::<WithoutInstance>::new(PIOP_PROTOCOL_TAG)
            .session(session)
            .instance(&instance);
        Self {
            state: domain.to_prover(TranscriptSponge::default()),
        }
    }
}

#[cfg(test)]
impl ChallengeSource for TestTranscript {
    fn challenge_extensions(
        &mut self,
        site: &str,
        count: usize,
    ) -> Result<Vec<OpeningField>, TranscriptOrderError> {
        let header = event_header(
            CHALLENGE_KIND,
            site.as_bytes(),
            count as u64,
            (16 * count) as u64,
        );
        self.state.public_message(&header);
        (0..count)
            .map(|_| {
                let c0 = prover_field_challenge(&mut self.state)
                    .map_err(|_| TranscriptOrderError::Sampler)?;
                let c1 = prover_field_challenge(&mut self.state)
                    .map_err(|_| TranscriptOrderError::Sampler)?;
                Ok(OpeningField::from_base_slice(&[c0, c1]))
            })
            .collect()
    }
}

#[cfg(test)]
impl PioTranscript for TestTranscript {
    fn absorb_proof(
        &mut self,
        site: &str,
        atom_count: usize,
        payload: &[u8],
    ) -> Result<(), TranscriptOrderError> {
        let header = event_header(
            PROOF_KIND,
            site.as_bytes(),
            atom_count as u64,
            payload.len() as u64,
        );
        self.state.public_message(&header);
        self.state.public_message(payload);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{projection::ProjectionOutputs, FalconDegree};
    use jolt_field::Ring as _;

    struct InjectedState {
        candidates: [u64; 3],
        next: usize,
        consumed_bytes: usize,
    }

    impl NativeState for InjectedState {
        fn public_message(&mut self, _bytes: &[u8]) {}

        fn sample_base(&mut self) -> Result<BaseField, TranscriptOrderError> {
            sample_base_by(|bytes| {
                bytes.copy_from_slice(&self.candidates[self.next].to_le_bytes());
                self.next += 1;
                self.consumed_bytes += 8;
            })
        }

        fn sponge_clone(&self) -> TranscriptSponge {
            TranscriptSponge::default()
        }

        fn squeeze(&mut self, _bytes: &mut [u8]) {
            unreachable!("the injected sampler test only samples field limbs")
        }

        fn apply_work(&mut self, _site: &str, _difficulty: u8) -> Result<(), TranscriptOrderError> {
            unreachable!("the injected sampler fixture has zero difficulty")
        }

        fn check_work_eof(&self) -> Result<(), TranscriptOrderError> {
            Ok(())
        }
    }

    fn suffix<T: PioTranscript>(transcript: &mut T, degree: FalconDegree, records: usize) {
        let outputs = ProjectionOutputs {
            y: [BaseField::from_u64(0); crate::protocol::projection::NORM_TERMINAL_ROWS],
            z: [BaseField::from_u64(0); crate::protocol::projection::RESIDUAL_WIDTH],
        };
        crate::protocol::projection::absorb_outputs(&outputs, transcript).unwrap();
        let rounds = degree.logn() + records.trailing_zeros() as usize;
        let basis_count = 2 + usize::from(degree == FalconDegree::Falcon1024);
        transcript.challenge_extensions(MU_SITE, 8).unwrap();
        transcript.challenge_extensions(TAU_SITE, rounds).unwrap();
        transcript
            .challenge_extensions(STAGE1_GAMMA_SITE, basis_count + 1)
            .unwrap();
        for stage_round in 0..rounds {
            transcript
                .absorb_extensions(
                    &format!("aerie/falcon/v2/stage1/round/{stage_round}/message"),
                    &[OpeningField::from_u64(0); 3],
                )
                .unwrap();
            transcript
                .challenge_extensions(
                    &format!("aerie/falcon/v2/stage1/round/{stage_round}/challenge"),
                    1,
                )
                .unwrap();
        }
        transcript
            .absorb_extensions(
                STAGE1_TERMINAL_SITE,
                &vec![OpeningField::from_u64(0); 2 * basis_count + 3],
            )
            .unwrap();
        let stage2_count = if degree == FalconDegree::Falcon512 {
            8
        } else {
            9
        };
        transcript
            .challenge_extensions(STAGE2_GAMMA_SITE, stage2_count)
            .unwrap();
        for stage_round in 0..rounds {
            transcript
                .absorb_extensions(
                    &format!("aerie/falcon/v2/stage2/round/{stage_round}/message"),
                    &[OpeningField::from_u64(0); 4],
                )
                .unwrap();
            transcript
                .challenge_extensions(
                    &format!("aerie/falcon/v2/stage2/round/{stage_round}/challenge"),
                    1,
                )
                .unwrap();
        }
        transcript
            .absorb_extensions(STAGE2_TERMINAL_SITE, &[OpeningField::from_u64(0); 6])
            .unwrap();
        transcript
            .challenge_extensions(STAGE3_GAMMA_SITE, 12)
            .unwrap();
        for stage_round in 0..rounds {
            transcript
                .absorb_extensions(
                    &format!("aerie/falcon/v2/stage3/round/{stage_round}/message"),
                    &[OpeningField::from_u64(0); 3],
                )
                .unwrap();
            transcript
                .challenge_extensions(
                    &format!("aerie/falcon/v2/stage3/round/{stage_round}/challenge"),
                    1,
                )
                .unwrap();
        }
        transcript
            .absorb_extensions(STAGE3_TERMINAL_SITE, &[OpeningField::from_u64(0); 9])
            .unwrap();
    }

    fn prover_prefix(
        degree: FalconDegree,
        live_rows: usize,
    ) -> (
        ProtocolTranscript,
        CommitmentManifest,
        [u8; INSTANCE_BYTES],
        usize,
    ) {
        let manifest = CommitmentManifest::for_batch_of(degree, live_rows).unwrap();
        let records = manifest.padded_records();
        let record = test_record(&manifest);
        let mut prover = ProtocolTranscript::new(&manifest, &record).unwrap();
        for role in [
            GroupRole::S1,
            GroupRole::Slack,
            GroupRole::S2Sources,
            GroupRole::Budget,
        ] {
            let payload =
                encode_extensions(&[OpeningField::from_u64(u64::from(role.transcript_id())); 8]);
            prover.absorb_group_commitment(role, &payload).unwrap();
        }
        prover.derive_projection_seed().unwrap();
        (prover.finish().unwrap(), manifest, record, records)
    }

    fn verifier_prefix(
        manifest: &CommitmentManifest,
        record: &[u8; INSTANCE_BYTES],
        work: &[u8],
    ) -> VerifierProtocolTranscript {
        let mut verifier = VerifierProtocolTranscript::new(manifest, record, work).unwrap();
        for role in [
            GroupRole::S1,
            GroupRole::Slack,
            GroupRole::S2Sources,
            GroupRole::Budget,
        ] {
            let payload =
                encode_extensions(&[OpeningField::from_u64(u64::from(role.transcript_id())); 8]);
            verifier.absorb_group_commitment(role, &payload).unwrap();
        }
        verifier.derive_projection_seed().unwrap();
        verifier.finish().unwrap()
    }

    #[test]
    fn event_headers_cover_each_registry_kind() {
        for kind in 0..=4 {
            let site = b"x";
            let expected = [
                kind, 1, 0, 0, 0, 0, 0, 0, 0, b'x', 2, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0,
            ];
            assert_eq!(event_header(kind, site, 2, 16), expected);
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
        let mut core = TranscriptCore {
            state,
            cursor,
            next_site_override: None,
        };
        let value = core.challenge_extensions("injected/challenge", 1).unwrap()[0];
        let limbs = value.to_base_vec();
        assert_eq!(limbs, [BaseField::from_u64(7), BaseField::from_u64(11)]);
        assert_eq!(core.state.consumed_bytes, 24);
    }

    #[test]
    fn found_nonce_satisfies_the_predicate_and_a_flipped_byte_rejects() {
        let difficulty = 4_u8;
        let nonce_bits = difficulty + GRINDING_NONCE_SLACK_BITS;
        let session = LengthPrefixedBytes(b"aerie/falcon/test");
        let instance = LengthPrefixedBytes(b"grinding-transition");
        let domain = DomainSeparator::<WithoutInstance>::new(PIOP_PROTOCOL_TAG)
            .session(session)
            .instance(&instance);
        let mut prover = domain.to_prover(TranscriptSponge::default());
        prover.public_message(b"test work prefix");
        let (nonce, predicate) =
            search_grinding_nonce(&prover, difficulty, nonce_bits).expect("bounded work search");
        assert!(grinding_predicate_accepts(
            &predicate,
            NonZeroU8::new(difficulty).unwrap()
        ));
        let (nonce_record, predicate_record) = work_context_records(difficulty, nonce_bits);
        assert_eq!(
            commit_grinding_nonce(&mut prover, nonce_record, nonce, predicate_record),
            predicate
        );
        let work = prover.narg_string().to_vec();

        let check = |work: &[u8]| {
            let session = LengthPrefixedBytes(b"aerie/falcon/test");
            let instance = LengthPrefixedBytes(b"grinding-transition");
            let domain = DomainSeparator::<WithoutInstance>::new(PIOP_PROTOCOL_TAG)
                .session(session)
                .instance(&instance);
            let mut verifier = NativeVerifier {
                state: domain.to_verifier(TranscriptSponge::default(), &[]),
                work: work.to_vec(),
                next: 0,
            };
            verifier.public_message(b"test work prefix");
            verifier.apply_work("test/work", difficulty)
        };
        check(&work).expect("found nonce verifies");

        let mut rejected_flip = None;
        'outer: for index in 0..work.len() {
            for bit in 0..7 {
                let mut changed = work.clone();
                changed[index] ^= 1 << bit;
                let mut cursor = changed.as_slice();
                let Ok(candidate) = NonceAtom::deserialize_from_narg(&mut cursor) else {
                    continue;
                };
                if !cursor.is_empty() || u64::from(candidate.into_inner()) >= (1_u64 << nonce_bits)
                {
                    continue;
                }
                if check(&changed).is_err() {
                    rejected_flip = Some(changed);
                    break 'outer;
                }
            }
        }
        assert!(
            rejected_flip.is_some(),
            "a canonical in-range byte flip rejects"
        );
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
                "aerie/falcon/v2/commit/s1",
                "aerie/falcon/v2/commit/slack",
                "aerie/falcon/v2/commit/s2-sources",
                "aerie/falcon/v2/commit/budget",
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
            let basis_count = 2 + usize::from(degree == FalconDegree::Falcon1024);
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

    #[test]
    fn complete_registry_prover_and_verifier_match_golden_pcs_sessions() {
        for (degree, live_rows) in [(FalconDegree::Falcon512, 4), (FalconDegree::Falcon1024, 2)] {
            let (mut prover, manifest, record, records) = prover_prefix(degree, live_rows);
            suffix(&mut prover, degree, records);
            let mut verifier = verifier_prefix(&manifest, &record, prover.work_bytes());
            suffix(&mut verifier, degree, records);
            let prover_session = prover.pcs_session().unwrap();
            let verifier_session = verifier.pcs_session().unwrap();
            assert_eq!(prover_session, verifier_session);
            prover.ensure_complete().unwrap();
            verifier.ensure_complete().unwrap();
            let golden = match degree {
                FalconDegree::Falcon512 => [
                    65, 235, 118, 240, 220, 229, 11, 237, 59, 114, 13, 75, 109, 211, 236, 11, 60,
                    253, 53, 233, 80, 112, 128, 133, 133, 128, 176, 87, 166, 234, 148, 235,
                ],
                FalconDegree::Falcon1024 => [
                    37, 26, 139, 58, 32, 250, 100, 207, 160, 132, 99, 14, 147, 37, 171, 19, 42,
                    167, 24, 36, 50, 236, 255, 147, 70, 250, 125, 136, 175, 138, 86, 25,
                ],
            };
            assert_eq!(prover_session, golden);
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
                format!("aerie/falcon/v2/{stage}/round/{round}/message"),
                1,
                count,
                bytes,
            ));
            expected.push((
                format!("aerie/falcon/v2/{stage}/round/{round}/challenge"),
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
