//! Role-local prover channel adapter over the shared transcript state machine.
use crate::protocol::{
    instance::INSTANCE_BYTES,
    manifest::{CommitmentManifest, GroupRole},
};
use crate::substrate::OpeningField;
pub use aerie_falcon::protocol::transcript::*;
use akita_transcript::{
    prover_context, search_grinding_nonce, NonceAtom, ProtocolContextRecord, ProverChannel,
    TranscriptSponge,
};
use spongefish::{DomainSeparator, DuplexSpongeInterface, WithoutInstance};
struct NativeProver(ProverChannel);
impl NativeState for NativeProver {
    fn public_message(&mut self, bytes: &[u8]) {
        self.0.public_message(bytes);
    }

    fn sponge_clone(&self) -> TranscriptSponge {
        self.0.duplex_sponge_state.clone()
    }

    fn squeeze(&mut self, bytes: &mut [u8]) {
        self.0.duplex_sponge_state.squeeze(bytes);
    }

    fn acquire_work_nonce(
        &mut self,
        difficulty: u8,
        nonce_bits: u8,
        _nonce_limit: u64,
    ) -> Result<(NonceAtom, Option<[u8; 32]>), TranscriptOrderError> {
        let (nonce, preview) = search_grinding_nonce(&self.0, difficulty, nonce_bits)
            .ok_or(TranscriptOrderError::WorkSearch { difficulty })?;
        Ok((NonceAtom::new(nonce), Some(preview)))
    }

    fn work_context(&mut self, record: ProtocolContextRecord) {
        prover_context(&mut self.0, record);
    }

    fn emit_work_nonce(&mut self, nonce: NonceAtom) {
        self.0.prover_message(&nonce);
    }

    fn check_work_eof(&self) -> Result<(), TranscriptOrderError> {
        Ok(())
    }
}

/// Prover-owned native PIOP transcript. Prover and verifier use this same
/// registry but maintain independent Spongefish states.
pub struct ProtocolTranscript {
    core: TranscriptCore<NativeProver>,
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

    #[cfg(any(test, feature = "internals"))]
    pub fn new_with_test_record(
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
            core: TranscriptCore::new(manifest, record, registry_record, NativeProver(state))?,
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
        self.core.require_next(6)?;
        Ok(self)
    }

    /// Export the PCS session after all stage-three terminal bytes.
    pub fn pcs_session(&mut self) -> Result<[u8; 32], TranscriptOrderError> {
        self.core.pcs_session()
    }

    /// Confirm that every registry event, including the PCS export, ran.
    pub fn ensure_complete(&self) -> Result<(), TranscriptOrderError> {
        self.core.ensure_complete()?;
        self.core.state().check_work_eof()
    }

    /// Return the inline canonical PIOP work argument bytes.
    #[must_use]
    pub fn work_bytes(&self) -> &[u8] {
        self.core.state().0.narg_string()
    }

    #[cfg(any(test, feature = "internals"))]
    pub fn alter_export_prefix_for_test(&mut self) {
        self.core
            .state_mut()
            .public_message(b"test-only altered PIOP prefix");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{projection::ProjectionOutputs, stage::StageShapes, FalconDegree};
    use crate::substrate::BaseField;
    use akita_transcript::{
        commit_grinding_nonce, grinding_predicate_accepts, GRINDING_NONCE_SLACK_BITS,
    };
    use jolt_field::Ring as _;
    use spongefish::{Encoding, NargDeserialize};
    use std::num::NonZeroU8;

    fn suffix<T: PioTranscript>(transcript: &mut T, degree: FalconDegree, records: usize) {
        let outputs = ProjectionOutputs {
            y: [BaseField::from_u64(0); crate::protocol::projection::NORM_TERMINAL_ROWS],
            z: [BaseField::from_u64(0); crate::protocol::projection::RESIDUAL_WIDTH],
        };
        crate::protocol::projection::absorb_outputs(&outputs, transcript).unwrap();
        let rounds = degree.logn() + records.trailing_zeros() as usize;
        let basis_count = StageShapes::for_degree(degree).basis16_sources;
        transcript.challenge_extensions(MU_SITE, 8).unwrap();
        transcript.challenge_extensions(TAU_SITE, rounds).unwrap();
        transcript
            .challenge_extensions(STAGE1_GAMMA_SITE, basis_count + 1)
            .unwrap();
        for stage_round in 0..rounds {
            transcript
                .absorb_extensions(
                    &format!("aerie/falcon/v1/stage1/round/{stage_round}/message"),
                    &[OpeningField::from_u64(0); 3],
                )
                .unwrap();
            transcript
                .challenge_extensions(
                    &format!("aerie/falcon/v1/stage1/round/{stage_round}/challenge"),
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
                    &format!("aerie/falcon/v1/stage2/round/{stage_round}/message"),
                    &[OpeningField::from_u64(0); 4],
                )
                .unwrap();
            transcript
                .challenge_extensions(
                    &format!("aerie/falcon/v1/stage2/round/{stage_round}/challenge"),
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
                    &format!("aerie/falcon/v1/stage3/round/{stage_round}/message"),
                    &[OpeningField::from_u64(0); 3],
                )
                .unwrap();
            transcript
                .challenge_extensions(
                    &format!("aerie/falcon/v1/stage3/round/{stage_round}/challenge"),
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
            let mut verifier = NativeVerifier::from_state(
                domain.to_verifier(TranscriptSponge::default(), &[]),
                work,
            );
            verifier.public_message(b"test work prefix");
            verifier.apply_work(difficulty)?;
            verifier.check_work_eof()
        };
        check(&work).expect("found nonce verifies");
        let out_of_range = NonceAtom::new(1_u32 << nonce_bits)
            .encode()
            .as_ref()
            .to_vec();
        let mut trailing = work.clone();
        trailing.push(0);
        for malformed in [
            &[][..],
            &[0x80][..],
            &[0x80, 0x00][..],
            out_of_range.as_slice(),
        ] {
            assert_eq!(check(malformed), Err(TranscriptOrderError::NonceEncoding));
        }
        assert_eq!(check(&trailing), Err(TranscriptOrderError::TrailingBytes));

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
                if matches!(
                    check(&changed),
                    Err(TranscriptOrderError::InvalidWork { difficulty: 4 })
                ) {
                    rejected_flip = Some(changed);
                    break 'outer;
                }
            }
        }
        let rejected_flip = rejected_flip.expect("a canonical in-range byte flip rejects");
        assert_eq!(
            check(&rejected_flip),
            Err(TranscriptOrderError::InvalidWork { difficulty })
        );
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
                    137, 18, 158, 134, 80, 23, 214, 161, 234, 232, 92, 211, 220, 66, 253, 24, 44,
                    10, 5, 169, 101, 18, 127, 21, 167, 219, 153, 48, 162, 110, 70, 151,
                ],
                FalconDegree::Falcon1024 => [
                    28, 174, 37, 182, 50, 38, 240, 249, 132, 154, 104, 79, 218, 27, 143, 63, 38,
                    237, 147, 186, 192, 225, 132, 247, 53, 30, 151, 94, 9, 67, 85, 237,
                ],
            };
            assert_eq!(prover_session, golden);
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
