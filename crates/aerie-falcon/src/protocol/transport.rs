//! Transport binding: canonical record bytes and verifier-local targets.
//!
//! The statement is bound to its source records through the canonical
//! public-record tree digest over the record count, hashed public
//! keys, salts, and messages, absorbed before commitments in the Section 10.1
//! transcript. The wire proof transports only what the verifier does not
//! already hold: the `HashToPoint` salts and the proof material. Common-salt
//! mode carries one salt and binds its mode through the digest customization.
//! Public keys and messages are statement inputs supplied to the verifier.
//! The verifier streams both public tables from these inputs: stage three
//! unpacks each canonical key's `h` row and samples each `HashToPoint`
//! target row `t` exactly once, contracting it at the stage point without
//! materializing a padded table. Neither table is absorbed into the
//! transcript; the public-record digest binds their sources.
//! Nothing target-shaped is ever trusted from the prover.

use aerie_falcon_native::{
    preprocess::PrehashedHashToPointBatch, source::CanonicalPublicKeyBytes, FalconParams,
};
use aerie_falcon_statement::{
    AggregationStatement, AggregationWitness, FalconPreparedWitness, PreprocessingMetrics,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::pcs::{
    self, FalconAggregateSetup, FalconVerifierSetup, PcsStageMetrics, VerifiedOpening,
};
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
use super::public_records_binding::public_records_binding_of;
use super::public_records_binding::{hashed_keys, public_records_binding_with_mode};
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
use super::wire::TRANSPORT_SALT_BYTES;
use super::wire::{self, SaltMode};
use super::AggregationParams;
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
use super::FalconWitness;

#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub use super::public_records_binding::public_records_binding;
#[cfg(any(feature = "bench-internals", feature = "adversarial-internals"))]
pub use super::public_records_binding::PUBLIC_RECORDS_BINDING_BLOCK_BYTES;

use crate::AggregationError as Error;

/// Complete raw-record-to-proof timing for the transport-bound prover.
#[derive(Clone, Copy, Debug)]
pub struct AggregateProveMetrics {
    pub total_ns: u128,
    pub preprocessing: PreprocessingMetrics,
    pub transport_materialize_ns: u128,
    pub transport_binding_ns: u128,
    pub encode_ns: u128,
    pub protocol: PcsStageMetrics,
}

/// Complete public-input-and-proof-to-acceptance timing for the verifier.
#[derive(Clone, Copy, Debug, Default)]
pub struct AggregateVerifyMetrics {
    pub total_ns: u128,
    pub decode_ns: u128,
    /// Derivation of the fixed-width `hpk` values needed before stage one.
    pub hashed_keys_ns: u128,
    /// End-to-end public-target work: `hpk` derivation plus the streamed
    /// target evaluation.
    pub hash_to_point_ns: u128,
    /// Streamed `HashToPoint` sampling and contraction of every target row at
    /// the stage point.
    pub target_evaluation_ns: u128,
    pub binding_ns: u128,
    /// Stage verification excluding target evaluation.
    pub stage_verify_ns: u128,
    pub pcs_verify_ns: u128,
}

/// The zero-extended public ring-key table decoded from canonical bytes.
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
fn derived_ring_keys<P: FalconParams>(
    public_keys: &[CanonicalPublicKeyBytes<P>],
    padded: usize,
) -> Vec<i16> {
    let degree = P::DEGREE;
    let mut table = vec![0_i16; padded * degree];
    #[cfg(feature = "parallel")]
    table[..public_keys.len() * degree]
        .par_chunks_mut(degree)
        .zip(public_keys.par_iter())
        .for_each(|(row, public_key)| {
            copy_centered_key_row(row, public_key);
        });
    #[cfg(not(feature = "parallel"))]
    for (record, public_key) in public_keys.iter().enumerate() {
        copy_centered_key_row(
            &mut table[record * degree..(record + 1) * degree],
            public_key,
        );
    }
    table
}

#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
fn copy_centered_key_row<P: FalconParams>(
    row: &mut [i16],
    public_key: &CanonicalPublicKeyBytes<P>,
) {
    debug_assert_eq!(row.len(), P::DEGREE);
    for (destination, block) in row
        .chunks_exact_mut(4)
        .zip(public_key.centered_coefficient_blocks())
    {
        destination.copy_from_slice(&block);
    }
}

/// Prove an aggregate of parameter set `P` directly from canonical keys,
/// signatures, and messages. The wire header names `P`.
///
/// The timer starts before source-bound Falcon witness derivation and ends
/// after wire encoding. Reusable [`FalconAggregateSetup`] construction is excluded.
pub fn prove_aggregate<P: AggregationParams>(
    setup: &FalconAggregateSetup<P>,
    statement: &AggregationStatement<P>,
    witness: &AggregationWitness<P>,
) -> Result<(Vec<u8>, AggregateProveMetrics), Error> {
    prove_aggregate_with_mode(setup, statement, witness, SaltMode::PerRecord)
}

/// Prove with an explicit salt transport mode, bound before the first
/// challenge. Common-salt mode requires every signature to use the same salt.
pub fn prove_aggregate_with_mode<P: AggregationParams>(
    setup: &FalconAggregateSetup<P>,
    statement: &AggregationStatement<P>,
    witness: &AggregationWitness<P>,
    mode: SaltMode,
) -> Result<(Vec<u8>, AggregateProveMetrics), Error> {
    // Reject a statement/witness length mismatch before the early commit
    // decodes and commits anything; ingestion repeats the check.
    if statement.records().len() != witness.signatures().len() {
        return Err(
            aerie_falcon_statement::ProductionError::WitnessLengthMismatch {
                statement: statement.records().len(),
                witness: witness.signatures().len(),
            }
            .into(),
        );
    }
    let total_start = std::time::Instant::now();
    mode.validate(
        witness
            .signatures()
            .iter()
            .map(aerie_falcon_native::source::CanonicalSignatureBytes::salt),
    )?;
    let total_span = tracing::info_span!("v1.prove.total").entered();
    // The two `s2`-only groups commit while the signatures are verified
    // natively: those Akita commits are memory-bound and do not scale across
    // the pool, the ingestion is compute-bound and does. The transcript
    // absorbs the payloads in canonical slot order later, so the proof bytes
    // are those of the sequential prover.
    let padded_rows = P::DEGREE_ID.padded_records(witness.signatures().len())?;
    let precommit_early = || -> Result<(pcs::S2Precommit, Vec<i16>), Error> {
        let s2 = tracing::info_span!("v1.prove.witness.s2_decode")
            .in_scope(|| decode_s2_padded::<P>(witness.signatures(), padded_rows));
        pcs::precommit_s2_groups(setup, padded_rows, &s2).map(|precommit| (precommit, s2))
    };
    #[cfg(feature = "parallel")]
    let (production, precommit) = rayon::join(
        || FalconPreparedWitness::from_aggregate_of::<P>(statement, witness),
        precommit_early,
    );
    #[cfg(not(feature = "parallel"))]
    let (production, precommit) = (
        FalconPreparedWitness::from_aggregate_of::<P>(statement, witness),
        precommit_early(),
    );
    let production = production?;
    let preprocessing = production.metrics();
    let mut batch = production.into_witness();
    super::pad_witness(&mut batch)?;
    let (precommit, early_s2) = precommit?;
    // The early groups were committed from the signature bytes; the witness
    // `s2` comes from the same bytes through native verification, in the same
    // record order. Divergence here means the committed tables no longer
    // describe the proved witness.
    debug_assert_eq!(
        early_s2.as_slice(),
        batch.s2.as_slice(),
        "early s2 commitments must match the verified witness"
    );
    drop(early_s2);
    let mut transport_materialize_ns = 0;
    let (public_keys, messages, salts) = aerie_falcon_statement::staged!(
        "v1.prove.input.transport_materialize",
        transport_materialize_ns,
        {
            let public_keys = statement
                .records()
                .iter()
                .map(|record| record.public_key().clone())
                .collect::<Vec<_>>();
            let messages = statement
                .records()
                .iter()
                .map(|record| record.message().to_vec())
                .collect::<Vec<_>>();
            let salts = witness
                .signatures()
                .iter()
                .map(|signature| *signature.salt())
                .collect::<Vec<_>>();
            (public_keys, messages, salts)
        }
    );
    let mut transport_binding_ns = 0;
    let binding = aerie_falcon_statement::staged!(
        "v1.prove.input.transport_binding",
        transport_binding_ns,
        public_records_binding_with_mode(&hashed_keys(&public_keys), &salts, &messages, mode)?
    );
    let (proof, protocol) =
        pcs::prove_prepared_with_precommit(&batch, &binding, mode, setup, precommit)?;
    let mut encode_ns = 0;
    let proof_bytes = aerie_falcon_statement::staged!(
        "v1.prove.output.encode",
        encode_ns,
        wire::encode_pcs_proof_with_mode(
            P::DEGREE_ID,
            statement.records().len(),
            &salts,
            &proof,
            mode
        )
    )?;
    let metrics = AggregateProveMetrics {
        total_ns: total_start.elapsed().as_nanos(),
        preprocessing,
        transport_materialize_ns,
        transport_binding_ns,
        encode_ns,
        protocol,
    };
    drop(total_span);
    Ok((proof_bytes, metrics))
}

/// The padded record-major `s2` table decoded from canonical signature
/// bytes: the same coefficients native verification writes into the compact
/// witness, available before verification runs.
fn decode_s2_padded<P: FalconParams>(
    signatures: &[aerie_falcon_native::source::CanonicalSignatureBytes<P>],
    padded_rows: usize,
) -> Vec<i16> {
    let degree = P::DEGREE;
    let mut s2 = vec![0_i16; padded_rows * degree];
    let live = &mut s2[..signatures.len() * degree];
    #[cfg(feature = "parallel")]
    live.par_chunks_exact_mut(degree)
        .zip(signatures.par_iter())
        .for_each(|(row, signature)| row.copy_from_slice(signature.decode().s2().as_ref()));
    #[cfg(not(feature = "parallel"))]
    live.chunks_exact_mut(degree)
        .zip(signatures.iter())
        .for_each(|(row, signature)| row.copy_from_slice(signature.decode().s2().as_ref()));
    s2
}

/// Prove one batch of parameter set `P` and serialize the complete
/// transport-bound proof.
///
/// Fail-closed consistency: the batch must have degree `P::DEGREE`, the
/// batch's public keys must decode to the batch's ring-key rows, and the
/// transport shapes must match the live count; a target mismatch is left to
/// verification, which recomputes every target.
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn prove_batch<P: AggregationParams>(
    batch: &FalconWitness,
    public_keys: &[CanonicalPublicKeyBytes<P>],
    salts: &[[u8; TRANSPORT_SALT_BYTES]],
    messages: &[Vec<u8>],
) -> Result<(Vec<u8>, PcsStageMetrics), Error> {
    if batch.degree() != P::DEGREE {
        return Err("batch degree does not match the parameter set".into());
    }
    let records = batch.live_rows;
    if public_keys.len() != records || salts.len() != records || messages.len() != records {
        return Err("transport shapes do not match the live batch".into());
    }
    let derived_h = derived_ring_keys(public_keys, batch.padded_rows);
    if derived_h.iter().copied().ne(batch.h.iter().copied()) {
        return Err("batch ring keys do not match the canonical public keys".into());
    }
    let binding = public_records_binding_of::<P>(public_keys, salts, messages)?;
    let (proof, metrics) = pcs::prove(batch, &binding)?;
    let proof_bytes = wire::encode_pcs_proof(P::DEGREE_ID, records, salts, &proof)?;
    Ok((proof_bytes, metrics))
}

/// Verify one aggregate proof of parameter set `P` FROM WIRE BYTES: no
/// prover-side data. The arguments follow the usual order: the public
/// verifier setup, the public statement (ordered canonical public-key bytes
/// and ordered raw messages), then the proof bytes. The wire header must name
/// `P` and the setup's record count. The salts come from the proof bytes, and
/// every ring key and target is derived locally. A message list that differs
/// from the one the prover bound changes the public-record digest and the
/// recomputed targets, so it rejects.
pub fn verify_aggregate<P: AggregationParams>(
    verifier_setup: &FalconVerifierSetup<P>,
    public_keys: &[CanonicalPublicKeyBytes<P>],
    messages: &[Vec<u8>],
    proof: &[u8],
) -> Result<VerifiedOpening, Error> {
    verify_aggregate_measured(verifier_setup, public_keys, messages, proof)
        .map(|(verified, _)| verified)
}

/// [`verify_aggregate`] with complete online timing.
pub fn verify_aggregate_measured<P: AggregationParams>(
    verifier_setup: &FalconVerifierSetup<P>,
    public_keys: &[CanonicalPublicKeyBytes<P>],
    messages: &[Vec<u8>],
    proof: &[u8],
) -> Result<(VerifiedOpening, AggregateVerifyMetrics), Error> {
    let total_start = std::time::Instant::now();
    let total_span = tracing::info_span!("v1.verify.total").entered();
    // Reject fixed-width shape mismatches before decoding or allocating for
    // any attacker-supplied proof body.
    let (_, records) = wire::peek_pcs_proof_header(P::DEGREE_ID, proof)?;
    if records != verifier_setup.records() {
        return Err("verifier setup record count does not match the proof".into());
    }
    if public_keys.len() != records {
        return Err("public-key count does not match the proof".into());
    }
    if messages.len() != records {
        return Err("message count does not match the proof".into());
    }
    let mut decode_ns = 0;
    let decoded = aerie_falcon_statement::staged!(
        "v1.verify.input.decode",
        decode_ns,
        // The header's parameter-set byte must name `P`; the decoder checks
        // it before parsing anything after it.
        wire::decode_pcs_proof(P::DEGREE_ID, proof)?
    );
    debug_assert_eq!(decoded.records, records);
    let mut hashed_keys_ns = 0;
    let hashed_keys = aerie_falcon_statement::staged!(
        "v1.verify.input.hashed_keys",
        hashed_keys_ns,
        hashed_keys(public_keys)
    );
    let mut binding_ns = 0;
    let binding = aerie_falcon_statement::staged!(
        "v1.verify.input.binding",
        binding_ns,
        public_records_binding_with_mode(
            &hashed_keys,
            &decoded.salts,
            messages,
            decoded.salt_mode
        )?
    );
    // The digest above binds the hashed keys, salts and messages that these
    // views read; the tables they stream are never absorbed themselves.
    let target_inputs = PrehashedHashToPointBatch::new(&hashed_keys, &decoded.salts, messages)?;
    let public: super::public_inputs::PublicInputs<'_, i16> = super::public_inputs::PublicInputs {
        records,
        salt_mode: decoded.salt_mode,
        public_records: &binding,
        h: P::ring_keys(public_keys),
        t: super::ring::PublicTargets::SourceBound(target_inputs),
    };
    let verified = pcs::verify(&public, verifier_setup, &decoded.proof)?;
    let hash_to_point_ns = hashed_keys_ns
        .checked_add(verified.target_evaluation_ns)
        .expect("verifier target timing total fits u128");
    let metrics = AggregateVerifyMetrics {
        total_ns: total_start.elapsed().as_nanos(),
        decode_ns,
        hashed_keys_ns,
        hash_to_point_ns,
        target_evaluation_ns: verified.target_evaluation_ns,
        binding_ns,
        stage_verify_ns: verified.stage_verify_ns,
        pcs_verify_ns: verified.pcs_verify_ns,
    };
    drop(total_span);
    Ok((verified, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "parallel")]
    use crate::protocol::ring::{target_evaluation, PublicTargets};
    use crate::protocol::FalconDegree;
    #[cfg(feature = "parallel")]
    use crate::substrate::OpeningField;
    use aerie_falcon_native::source::CanonicalPublicKeyBytes;
    use aerie_falcon_native::Falcon512;
    use aerie_falcon_statement::fixtures::constructed_statement;
    #[cfg(feature = "parallel")]
    use aerie_falcon_statement::FALCON_DEGREE;
    use aerie_falcon_statement::{split_source_statement, FalconPreparedWitness};
    use jolt_field::Ring as _;
    use shake::{
        digest::{ExtendableOutput, Update, XofReader},
        Shake256,
    };

    struct Fixture {
        batch: FalconWitness,
        public_keys: Vec<CanonicalPublicKeyBytes<aerie_falcon_native::Falcon512>>,
        salts: Vec<[u8; TRANSPORT_SALT_BYTES]>,
        messages: Vec<Vec<u8>>,
    }

    /// A syntactically valid canonical public key that differs from the
    /// constructed fixture's shared key: header byte plus all-zero
    /// coefficients.
    fn zero_public_key() -> CanonicalPublicKeyBytes<aerie_falcon_native::Falcon512> {
        let mut bytes = [0_u8; aerie_falcon_native::source::FALCON_512_PUBLIC_KEY_BYTES];
        bytes[0] = 0x09;
        CanonicalPublicKeyBytes::parse(&bytes).expect("canonical zero key")
    }

    fn zero_public_key_of<P: FalconParams>() -> CanonicalPublicKeyBytes<P> {
        let mut bytes = vec![0_u8; P::PUBLIC_KEY_BYTES];
        bytes[0] = P::PUBLIC_KEY_HEADER;
        CanonicalPublicKeyBytes::parse(&bytes).expect("canonical zero key")
    }

    /// The wire verifier's streamed targets agree with a padded table of
    /// scalar native `HashToPoint` rows at both degrees.
    fn source_bound_targets_match_native_scalar_of<P: AggregationParams>() {
        let live_rows = 3;
        let padded_rows = P::DEGREE_ID.padded_records(live_rows).unwrap();
        let key = zero_public_key_of::<P>();
        let public_keys = vec![key; live_rows];
        let salts = (0..live_rows)
            .map(|record| {
                core::array::from_fn(|index| record.wrapping_mul(31).wrapping_add(index) as u8)
            })
            .collect::<Vec<_>>();
        let messages = (0..live_rows)
            .map(|record| vec![record as u8; record * 19])
            .collect::<Vec<_>>();
        let hpks = hashed_keys(&public_keys);
        let source = PrehashedHashToPointBatch::new(&hpks, &salts, &messages).unwrap();
        let mut expanded = public_keys
            .iter()
            .zip(&salts)
            .zip(&messages)
            .flat_map(|((key, salt), message)| {
                aerie_falcon_native::preprocess::hash_to_point_public_raw(key, salt, message)
                    .as_ref()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        expanded.resize(padded_rows * P::DEGREE, 0);
        let variables = P::DEGREE_ID.logn() + padded_rows.trailing_zeros() as usize;
        let point = (0..variables)
            .map(|index| {
                crate::substrate::OpeningField::from_u64(
                    (index as u64).wrapping_mul(0x9e37_79b9).wrapping_add(7),
                )
            })
            .collect::<Vec<_>>();
        let expected = crate::protocol::ring::target_evaluation(
            P::DEGREE_ID,
            padded_rows,
            live_rows,
            crate::protocol::ring::PublicTargets::Expanded(&expanded),
            &point,
        );
        let actual = crate::protocol::ring::target_evaluation(
            P::DEGREE_ID,
            padded_rows,
            live_rows,
            crate::protocol::ring::PublicTargets::<i16>::SourceBound(source),
            &point,
        );
        assert_eq!(actual, expected, "{}", P::NAME);
    }

    /// Hash-derived constructed records: relation-positive rows with
    /// genuine per-record `HashToPoint` targets, salts, and messages.
    /// The constructed fixture deliberately shares ONE public key across
    /// records, so key-substitution tests swap in `zero_public_key`.
    fn fixture(records: usize) -> Fixture {
        let source = constructed_statement(records).expect("constructed statement");
        let (statement, witness) = split_source_statement(&source).expect("split statement");
        let production =
            FalconPreparedWitness::from_aggregate(&statement, &witness).expect("production batch");
        Fixture {
            batch: production.into_witness(),
            public_keys: statement
                .records()
                .iter()
                .map(|record| record.public_key().clone())
                .collect(),
            salts: witness
                .signatures()
                .iter()
                .map(|signature| *signature.salt())
                .collect(),
            messages: statement
                .records()
                .iter()
                .map(|record| record.message().to_vec())
                .collect(),
        }
    }

    #[test]
    fn common_salt_roundtrip_binds_mode_and_salt() {
        std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| {
                let source = constructed_statement(1).unwrap();
                let (statement, witness) = split_source_statement(&source).unwrap();
                let records = 5;
                let statement = AggregationStatement::<Falcon512>::new(vec![
                    statement.records()[0]
                        .clone();
                    records
                ])
                .unwrap();
                let witness = AggregationWitness::<Falcon512>::new(vec![
                    witness.signatures()[0]
                        .clone();
                    records
                ])
                .unwrap();
                let keys = vec![statement.records()[0].public_key().clone(); records];
                let messages = vec![statement.records()[0].message().to_vec(); records];
                let setup = FalconAggregateSetup::<Falcon512>::new(records).unwrap();
                for mode in [SaltMode::Common, SaltMode::PerRecord] {
                    let (bytes, _) =
                        prove_aggregate_with_mode(&setup, &statement, &witness, mode).unwrap();
                    verify_aggregate(setup.verifier(), &keys, &messages, &bytes).unwrap();
                    let decoded = wire::decode_pcs_proof(FalconDegree::Falcon512, &bytes).unwrap();
                    assert_eq!(decoded.salt_mode, mode);
                    assert_eq!(
                        decoded.salts,
                        vec![*witness.signatures()[0].salt(); records]
                    );
                    let reencoded = wire::encode_pcs_proof_with_mode(
                        decoded.degree,
                        records,
                        &decoded.salts,
                        &decoded.proof,
                        mode,
                    )
                    .unwrap();
                    assert_eq!(bytes, reencoded);
                    if mode == SaltMode::PerRecord {
                        assert_eq!(
                            bytes,
                            wire::encode_pcs_proof(
                                decoded.degree,
                                records,
                                &decoded.salts,
                                &decoded.proof,
                            )
                            .unwrap()
                        );
                    }
                    let other = if mode == SaltMode::Common {
                        SaltMode::PerRecord
                    } else {
                        SaltMode::Common
                    };
                    let relabelled = wire::encode_pcs_proof_with_mode(
                        decoded.degree,
                        records,
                        &decoded.salts,
                        &decoded.proof,
                        other,
                    )
                    .unwrap();
                    let expected_delta = (records - 1) * TRANSPORT_SALT_BYTES;
                    assert_eq!(bytes.len().abs_diff(relabelled.len()), expected_delta);
                    assert!(wire::decode_pcs_proof(decoded.degree, &relabelled).is_ok());
                    assert!(
                        verify_aggregate(setup.verifier(), &keys, &messages, &relabelled).is_err()
                    );
                    let mut tampered = bytes.clone();
                    tampered[wire::HEADER_BYTES] ^= 1;
                    assert!(
                        verify_aggregate(setup.verifier(), &keys, &messages, &tampered).is_err()
                    );
                    let mut unequal = decoded.salts.clone();
                    unequal[records - 1][0] ^= 1;
                    assert!(matches!(
                        wire::encode_pcs_proof_with_mode(
                            decoded.degree,
                            records,
                            &unequal,
                            &decoded.proof,
                            SaltMode::Common,
                        ),
                        Err(wire::WireError::Malformed(
                            "common-salt mode requires equal nonempty salts"
                        ))
                    ));
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn verifier_rejects_setup_record_mismatch_before_body_decode() {
        let verifier_setup = FalconVerifierSetup::<Falcon512>::new(1).unwrap();
        let mut header_only = b"aerie/falcon/v2p".to_vec();
        header_only.extend_from_slice(&FalconDegree::Falcon512.identity_bytes());
        header_only.extend_from_slice(&2_u64.to_le_bytes());
        assert_eq!(header_only.len(), wire::HEADER_BYTES);

        let error = verify_aggregate(&verifier_setup, &[], &[], &header_only)
            .expect_err("setup mismatch must reject before the missing body is decoded");
        assert_eq!(
            error.to_string(),
            "verifier setup record count does not match the proof"
        );
    }

    #[test]
    fn transport_proof_roundtrips_with_streamed_targets() {
        let outcome = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| -> Result<(), String> {
                // A full batch and an arbitrary live batch on the same
                // padded domain, both against the minimum-batch catalog row.
                for records in [1_usize, 2, 5, 8] {
                    let fixture = fixture(records);
                    let (proof_bytes, metrics) = prove_batch(
                        &fixture.batch,
                        &fixture.public_keys,
                        &fixture.salts,
                        &fixture.messages,
                    )
                    .map_err(|error| error.to_string())?;
                    assert_eq!(metrics.records, records);
                    let verifier_setup = FalconVerifierSetup::<Falcon512>::new(records)
                        .map_err(|error| error.to_string())?;
                    verify_aggregate(
                        &verifier_setup,
                        &fixture.public_keys,
                        &fixture.messages,
                        &proof_bytes,
                    )
                    .map_err(|error| error.to_string())?;

                    // The transport frame is exactly the salts: the first
                    // commitment byte follows the last salt.
                    let salt_offset = wire::HEADER_BYTES;
                    assert_eq!(
                        &proof_bytes[salt_offset..salt_offset + TRANSPORT_SALT_BYTES],
                        &fixture.salts[0]
                    );

                    // A tampered transported salt changes the recomputed
                    // target and must reject.
                    let mut tampered = proof_bytes.clone();
                    tampered[salt_offset] ^= 1;
                    assert!(
                        verify_aggregate(
                            &verifier_setup,
                            &fixture.public_keys,
                            &fixture.messages,
                            &tampered
                        )
                        .is_err(),
                        "tampered salt at {records} records"
                    );

                    // A message that differs from the one the prover bound
                    // changes the binding digest and the recomputed target.
                    let mut altered = fixture.messages.clone();
                    altered[0][0] ^= 1;
                    assert!(
                        verify_aggregate(
                            &verifier_setup,
                            &fixture.public_keys,
                            &altered,
                            &proof_bytes
                        )
                        .is_err(),
                        "altered message at {records} records"
                    );

                    // Two messages swapped between records likewise.
                    if records > 1 {
                        let mut swapped = fixture.messages.clone();
                        swapped.swap(0, 1);
                        assert!(
                            verify_aggregate(
                                &verifier_setup,
                                &fixture.public_keys,
                                &swapped,
                                &proof_bytes
                            )
                            .is_err(),
                            "swapped messages at {records} records"
                        );
                    }

                    // A substituted public key changes the streamed ring
                    // keys, the recomputed targets, and the binding digest.
                    let mut substituted = fixture.public_keys.clone();
                    substituted[0] = zero_public_key();
                    assert!(
                        verify_aggregate(
                            &verifier_setup,
                            &substituted,
                            &fixture.messages,
                            &proof_bytes
                        )
                        .is_err(),
                        "substituted public key at {records} records"
                    );

                    // A wrong key or message count is rejected before any
                    // derivation.
                    assert!(verify_aggregate(
                        &verifier_setup,
                        &fixture.public_keys[..records - 1],
                        &fixture.messages,
                        &proof_bytes
                    )
                    .is_err());
                    assert!(verify_aggregate(
                        &verifier_setup,
                        &fixture.public_keys,
                        &fixture.messages[..records - 1],
                        &proof_bytes
                    )
                    .is_err());
                }
                Ok(())
            })
            .expect("spawn transport test")
            .join()
            .expect("join transport test");
        outcome.expect("transport roundtrip");
    }

    #[test]
    fn streamed_targets_match_scalar_hash_to_point_for_both_degrees() {
        source_bound_targets_match_native_scalar_of::<Falcon512>();
        source_bound_targets_match_native_scalar_of::<aerie_falcon_native::Falcon1024>();
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn streamed_target_matches_expanded_across_chunks_and_worker_counts() {
        let live_rows = 129_usize;
        let padded_rows = FalconDegree::Falcon512.padded_records(live_rows).unwrap();
        let public_key = zero_public_key();
        let salts = (0..live_rows)
            .map(|record| {
                core::array::from_fn(|index| record.wrapping_mul(31).wrapping_add(index) as u8)
            })
            .collect::<Vec<_>>();
        let messages = (0..live_rows)
            .map(|record| vec![record as u8; record % 37])
            .collect::<Vec<_>>();
        let hpks = vec![
            aerie_falcon_native::preprocess::HashedVerifyingKey::derive(&public_key);
            live_rows
        ];
        let source = PrehashedHashToPointBatch::new(&hpks, &salts, &messages).unwrap();
        let inputs = salts
            .iter()
            .zip(&messages)
            .map(|(salt, message)| {
                aerie_falcon_native::preprocess::HashToPointInput::new(&public_key, salt, message)
            })
            .collect::<Vec<_>>();
        let mut expanded = aerie_falcon_native::preprocess::hash_to_point_public_raw_batch(&inputs)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        expanded.resize(padded_rows * FALCON_DEGREE, 0);
        let row_vars = padded_rows.trailing_zeros() as usize;
        let s1_point: Vec<_> = (0..crate::protocol::ring::COEFFICIENT_VARS + row_vars)
            .map(|index| OpeningField::from_u64((index as u64).wrapping_mul(17).wrapping_add(3)))
            .collect();
        let expected = target_evaluation(
            FalconDegree::Falcon512,
            padded_rows,
            live_rows,
            PublicTargets::Expanded(&expanded),
            &s1_point,
        );
        for workers in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            let actual = pool.install(|| {
                target_evaluation(
                    FalconDegree::Falcon512,
                    padded_rows,
                    live_rows,
                    PublicTargets::<i16>::SourceBound(source),
                    &s1_point,
                )
            });
            assert_eq!(actual, expected, "{workers} Rayon workers");
        }
    }

    /// Falcon-1024 end to end on real FN-DSA-1024 keys and signatures:
    /// source-bound ingestion, four real commitments, the wire with
    /// parameter-set byte `10`, verifier-local `HashToPoint` at degree
    /// 1024 from canonical Falcon-1024 key bytes; cross-degree proofs and
    /// keys reject in both directions; a changed salt or message rejects.
    #[test]
    fn single_falcon_1024_roundtrips() {
        std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| {
                let (statement, witness) =
                    aerie_falcon_statement::honest::honest_aggregation_1024(1, 0x0010_2401)
                        .unwrap();
                let setup =
                    FalconAggregateSetup::<aerie_falcon_native::Falcon1024>::new(1).unwrap();
                let (bytes, _) = prove_aggregate(&setup, &statement, &witness).unwrap();
                let keys = statement
                    .records()
                    .iter()
                    .map(|r| r.public_key().clone())
                    .collect::<Vec<_>>();
                let messages = statement
                    .records()
                    .iter()
                    .map(|r| r.message().to_vec())
                    .collect::<Vec<_>>();
                verify_aggregate(setup.verifier(), &keys, &messages, &bytes).unwrap();
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn falcon_1024_aggregate_roundtrips_on_honest_fn_dsa_signatures() {
        use aerie_falcon_native::Falcon1024;
        use aerie_falcon_statement::honest::honest_aggregation_1024;
        let outcome = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| -> Result<(), String> {
                let records = 8_usize;
                let (statement, witness) =
                    honest_aggregation_1024(records, 0x1024_0a11).map_err(|e| e.to_string())?;
                let setup =
                    FalconAggregateSetup::<Falcon1024>::new(records).map_err(|e| e.to_string())?;
                let (bytes, metrics) =
                    prove_aggregate(&setup, &statement, &witness).map_err(|e| e.to_string())?;
                assert_eq!(metrics.preprocessing.records, records);
                let decoded = wire::decode_pcs_proof(FalconDegree::Falcon1024, &bytes)
                    .map_err(|e| e.to_string())?;
                assert_eq!(decoded.degree, FalconDegree::Falcon1024);
                assert_eq!(decoded.proof.commitments.len(), 4);
                let salts = witness
                    .signatures()
                    .iter()
                    .map(|signature| *signature.salt())
                    .collect::<Vec<_>>();
                assert_eq!(decoded.salts, salts);

                let public_keys = statement
                    .records()
                    .iter()
                    .map(|record| record.public_key().clone())
                    .collect::<Vec<_>>();
                let messages = statement
                    .records()
                    .iter()
                    .map(|record| record.message().to_vec())
                    .collect::<Vec<_>>();
                let verified = verify_aggregate(setup.verifier(), &public_keys, &messages, &bytes)
                    .map_err(|e| e.to_string())?;
                assert_eq!(verified.ledger.groups.len(), 4);

                // Cross-parameter-set rejection in both directions. Each
                // mismatched pair stops at the header's parameter-set byte,
                // before the record count or any proof material is read.
                let source_512 = constructed_statement(records).map_err(|e| e.to_string())?;
                let (statement_512, witness_512) =
                    aerie_falcon_statement::split_source_statement(&source_512)
                        .map_err(|e| e.to_string())?;
                let keys_512 = statement_512
                    .records()
                    .iter()
                    .map(|record| record.public_key().clone())
                    .collect::<Vec<_>>();
                let messages_512 = statement_512
                    .records()
                    .iter()
                    .map(|record| record.message().to_vec())
                    .collect::<Vec<_>>();
                let setup_512 =
                    FalconAggregateSetup::<Falcon512>::new(records).map_err(|e| e.to_string())?;
                let (bytes_512, _) = prove_aggregate(&setup_512, &statement_512, &witness_512)
                    .map_err(|e| e.to_string())?;
                assert_eq!(bytes_512[wire::HEADER_PARAMETER_SET_OFFSET], 9);
                assert_eq!(bytes[wire::HEADER_PARAMETER_SET_OFFSET], 10);
                let rejects_on_parameter_set = |result: Result<VerifiedOpening, Error>| {
                    result
                        .err()
                        .is_some_and(|error| error.to_string().contains("parameter set mismatch"))
                };
                // A Falcon-1024 proof under the Falcon-512 verifier, and the
                // converse.
                assert!(rejects_on_parameter_set(verify_aggregate(
                    setup_512.verifier(),
                    &keys_512,
                    &messages_512,
                    &bytes,
                )));
                assert!(rejects_on_parameter_set(verify_aggregate(
                    setup.verifier(),
                    &public_keys,
                    &messages,
                    &bytes_512,
                )));
                // A flipped parameter-set byte: the relabeled frame is refused
                // by the verifier it names (the body has the other shape) and
                // by the one it came from (the byte no longer matches).
                let mut relabeled_1024 = bytes.clone();
                relabeled_1024[wire::HEADER_PARAMETER_SET_OFFSET] = 9;
                assert!(rejects_on_parameter_set(verify_aggregate(
                    setup.verifier(),
                    &public_keys,
                    &messages,
                    &relabeled_1024,
                )));
                assert!(verify_aggregate(
                    setup_512.verifier(),
                    &keys_512,
                    &messages_512,
                    &relabeled_1024,
                )
                .is_err());
                let mut relabeled_512 = bytes_512.clone();
                relabeled_512[wire::HEADER_PARAMETER_SET_OFFSET] = 10;
                assert!(rejects_on_parameter_set(verify_aggregate(
                    setup_512.verifier(),
                    &keys_512,
                    &messages_512,
                    &relabeled_512,
                )));
                assert!(verify_aggregate(
                    setup.verifier(),
                    &public_keys,
                    &messages,
                    &relabeled_512,
                )
                .is_err());
                // Proving a Falcon-1024 statement under a Falcon-512 setup
                // does not type-check: `FalconAggregateSetup<P>` fixes `P`.

                // Tampering: one salt byte, one message byte.
                let header = wire::HEADER_BYTES;
                let mut tampered = bytes.clone();
                tampered[header + 3] ^= 1;
                assert!(
                    verify_aggregate(setup.verifier(), &public_keys, &messages, &tampered).is_err()
                );
                let mut other_messages = messages.clone();
                other_messages[2].push(0x5a);
                assert!(
                    verify_aggregate(setup.verifier(), &public_keys, &other_messages, &bytes)
                        .is_err()
                );
                Ok(())
            })
            .expect("spawn")
            .join()
            .expect("join");
        outcome.expect("Falcon-1024 aggregate round trip");
    }

    /// Falcon-1024 at a live count that pads to `N_sig* = 16`, on real
    /// FN-DSA-1024 signatures, against the generated
    /// `falcon-v1-batch-32-budget1-final` row (the Falcon-1024 shapes at
    /// `N_sig* = 16` equal the Falcon-512 shapes at `N_sig* = 32`): the aggregate verifies and a
    /// changed message rejects.
    #[test]
    fn falcon_1024_aggregate_roundtrips_at_a_batch_padded_to_16() {
        use crate::protocol::FalconDegree;
        use aerie_falcon_native::Falcon1024;
        use aerie_falcon_statement::honest::honest_aggregation_1024;
        let outcome = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| -> Result<(), String> {
                let records = 12_usize;
                let (statement, witness) =
                    honest_aggregation_1024(records, 0x1024_0a16).map_err(|e| e.to_string())?;
                let setup =
                    FalconAggregateSetup::<Falcon1024>::new(records).map_err(|e| e.to_string())?;
                let (bytes, metrics) =
                    prove_aggregate(&setup, &statement, &witness).map_err(|e| e.to_string())?;
                assert_eq!(metrics.preprocessing.records, records);
                let decoded = wire::decode_pcs_proof(FalconDegree::Falcon1024, &bytes)
                    .map_err(|e| e.to_string())?;
                assert_eq!(decoded.degree, FalconDegree::Falcon1024);
                assert_eq!(decoded.proof.commitments.len(), 4);

                let public_keys = statement
                    .records()
                    .iter()
                    .map(|record| record.public_key().clone())
                    .collect::<Vec<_>>();
                let messages = statement
                    .records()
                    .iter()
                    .map(|record| record.message().to_vec())
                    .collect::<Vec<_>>();
                let verified = verify_aggregate(setup.verifier(), &public_keys, &messages, &bytes)
                    .map_err(|e| e.to_string())?;
                assert_eq!(verified.ledger.groups.len(), 4);

                let mut other_messages = messages.clone();
                other_messages[records - 1].push(0x5a);
                assert!(
                    verify_aggregate(setup.verifier(), &public_keys, &other_messages, &bytes,)
                        .is_err()
                );
                Ok(())
            })
            .expect("spawn")
            .join()
            .expect("join");
        outcome.expect("Falcon-1024 aggregate round trip at 12 records");
    }

    #[test]
    fn aggregate_api_accounts_for_source_witness_through_wire_verification() {
        let outcome = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| -> Result<(), String> {
                let source = constructed_statement(8).map_err(|error| error.to_string())?;
                let (statement, witness) =
                    split_source_statement(&source).map_err(|error| error.to_string())?;
                let statement = &statement;
                let setup = FalconAggregateSetup::<Falcon512>::new(statement.records().len())
                    .map_err(|error| error.to_string())?;
                let (proof_bytes, prove_metrics) = prove_aggregate(&setup, statement, &witness)
                    .map_err(|error| error.to_string())?;
                let public_keys = statement
                    .records()
                    .iter()
                    .map(|record| record.public_key().clone())
                    .collect::<Vec<_>>();
                let messages = statement
                    .records()
                    .iter()
                    .map(|record| record.message().to_vec())
                    .collect::<Vec<_>>();
                let (_, verify_metrics) = verify_aggregate_measured(
                    setup.verifier(),
                    &public_keys,
                    &messages,
                    &proof_bytes,
                )
                .map_err(|error| error.to_string())?;

                assert_eq!(prove_metrics.protocol.records, statement.records().len());
                assert!(prove_metrics.total_ns >= prove_metrics.protocol.commit_ns);
                assert!(verify_metrics.total_ns >= verify_metrics.pcs_verify_ns);
                Ok(())
            })
            .expect("spawn aggregate API test")
            .join()
            .expect("join aggregate API test");
        outcome.expect("aggregate API roundtrip");
    }

    #[test]
    fn prepared_witness_preserves_transport_proof_and_wire_bytes() {
        let outcome = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| -> Result<(), String> {
                let source = constructed_statement(8).map_err(|error| error.to_string())?;
                let (statement, witness) =
                    split_source_statement(&source).map_err(|error| error.to_string())?;
                let statement = &statement;
                let prepared = FalconPreparedWitness::from_aggregate(statement, &witness)
                    .map_err(|error| error.to_string())?;
                assert_eq!(
                    prepared.metrics().canonical_statement_bytes,
                    statement
                        .canonical_encoding()
                        .map_err(|error| error.to_string())?
                        .len()
                );

                let public_keys = statement
                    .records()
                    .iter()
                    .map(|record| record.public_key().clone())
                    .collect::<Vec<_>>();
                let messages = statement
                    .records()
                    .iter()
                    .map(|record| record.message().to_vec())
                    .collect::<Vec<_>>();
                let salts = witness
                    .signatures()
                    .iter()
                    .map(|signature| *signature.salt())
                    .collect::<Vec<_>>();
                let binding = public_records_binding(&public_keys, &salts, &messages)
                    .map_err(|error| error.to_string())?;
                let setup = FalconAggregateSetup::<Falcon512>::new(statement.records().len())
                    .map_err(|error| error.to_string())?;
                let (aggregate_bytes, _) = prove_aggregate(&setup, statement, &witness)
                    .map_err(|error| error.to_string())?;
                let (prepared_proof, _) = pcs::prove_prepared(prepared.witness(), &binding, &setup)
                    .map_err(|error| error.to_string())?;
                let prepared_proof_bytes = wire::encode_pcs_proof(
                    FalconDegree::Falcon512,
                    statement.records().len(),
                    &salts,
                    &prepared_proof,
                )
                .map_err(|error| error.to_string())?;
                assert_eq!(aggregate_bytes, prepared_proof_bytes);
                Ok(())
            })
            .expect("spawn parity test")
            .join()
            .expect("join parity test");
        outcome.expect("prepared witness parity");
    }

    /// Byte-identity regression gate for prover optimizations: the
    /// proof bytes at fixed constructed inputs are deterministic, so their
    /// digest must not move unless the wire deliberately changes.
    #[test]
    fn proof_bytes_are_pinned() {
        let digest = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| -> Result<String, String> {
                let fixture = fixture(8);
                let (proof_bytes, _) = prove_batch(
                    &fixture.batch,
                    &fixture.public_keys,
                    &fixture.salts,
                    &fixture.messages,
                )
                .map_err(|error| error.to_string())?;
                let verifier_setup =
                    FalconVerifierSetup::<Falcon512>::new(8).map_err(|error| error.to_string())?;
                verify_aggregate(
                    &verifier_setup,
                    &fixture.public_keys,
                    &fixture.messages,
                    &proof_bytes,
                )
                .map_err(|error| error.to_string())?;
                eprintln!("proof bytes: Falcon-512, N=8: {}", proof_bytes.len());
                Ok(proof_digest(&proof_bytes))
            })
            .expect("spawn digest gate")
            .join()
            .expect("join digest gate")
            .expect("proof digest");
        // Pins the complete version 2 PCS frame for this fixture; any change
        // to the transcript, codec, or Akita pin moves it.
        assert_eq!(
            digest, "d224b1372141d04ab2ba16faf3d9163a",
            "version 2 proof bytes moved"
        );
    }

    /// The 128-bit SHAKE256 digest of a proof, in hex.
    fn proof_digest(proof_bytes: &[u8]) -> String {
        use std::fmt::Write;
        let mut hasher = Shake256::default();
        hasher.update(proof_bytes);
        let mut digest = [0_u8; 16];
        hasher.finalize_xof().read(&mut digest);
        digest.iter().fold(String::new(), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
    }

    /// The Falcon-1024 byte-identity gate at the original smallest supported batch
    /// (three records, padded to four), on the seeded honest FN-DSA-1024
    /// fixture. The digest moves with the transcript, the codec, the Akita
    /// pin, or the pinned `fn-dsa` signer that generates the fixture.
    #[test]
    fn falcon_1024_proof_bytes_are_pinned() {
        use aerie_falcon_native::Falcon1024;
        use aerie_falcon_statement::honest::honest_aggregation_1024;
        let digest = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| -> Result<String, String> {
                let records = 3_usize;
                assert!(FalconAggregateSetup::<Falcon1024>::supports(records));
                assert!(FalconAggregateSetup::<Falcon1024>::supports(records - 1));
                let (statement, witness) =
                    honest_aggregation_1024(records, 0x1024_d16e).map_err(|e| e.to_string())?;
                let setup =
                    FalconAggregateSetup::<Falcon1024>::new(records).map_err(|e| e.to_string())?;
                let (proof_bytes, _) =
                    prove_aggregate(&setup, &statement, &witness).map_err(|e| e.to_string())?;
                let (public_keys, messages): (Vec<_>, Vec<_>) = statement
                    .records()
                    .iter()
                    .map(|record| (record.public_key().clone(), record.message().to_vec()))
                    .unzip();
                verify_aggregate(setup.verifier(), &public_keys, &messages, &proof_bytes)
                    .map_err(|e| e.to_string())?;
                eprintln!(
                    "proof bytes: Falcon-1024, N=3 (padded 4): {}",
                    proof_bytes.len()
                );
                Ok(proof_digest(&proof_bytes))
            })
            .expect("spawn digest gate")
            .join()
            .expect("join digest gate")
            .expect("proof digest");
        assert_eq!(
            digest, "3b536a53c0ef1fd8618e835eb59ba2f0",
            "version 2 Falcon-1024 proof bytes moved"
        );
    }

    /// Byte-identity gates at batches large enough to reach the parallel
    /// and bucketed prover kernels (the range-source quartet buckets need a
    /// `2^21`-cell coefficient domain). Run with `--ignored` in a release
    /// profile; the small pinned gates above run on every test pass.
    #[test]
    #[ignore = "large batch; run explicitly in a release profile"]
    fn large_batch_proof_bytes_are_pinned() {
        let digest = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| -> Result<String, String> {
                let fixture = fixture(4096);
                let (proof_bytes, _) = prove_batch(
                    &fixture.batch,
                    &fixture.public_keys,
                    &fixture.salts,
                    &fixture.messages,
                )
                .map_err(|error| error.to_string())?;
                let verifier_setup = FalconVerifierSetup::<Falcon512>::new(4096)
                    .map_err(|error| error.to_string())?;
                verify_aggregate(
                    &verifier_setup,
                    &fixture.public_keys,
                    &fixture.messages,
                    &proof_bytes,
                )
                .map_err(|error| error.to_string())?;
                eprintln!("proof bytes: Falcon-512, N=4096: {}", proof_bytes.len());
                Ok(proof_digest(&proof_bytes))
            })
            .expect("spawn digest gate")
            .join()
            .expect("join digest gate")
            .expect("proof digest");
        assert_eq!(
            digest, "286911fc3dda02ca14359de0c9cf9ab8",
            "version 2 proof bytes moved at 4096 records"
        );
    }

    /// The Falcon-1024 large-batch byte-identity gate; see
    /// [`large_batch_proof_bytes_are_pinned`].
    #[test]
    #[ignore = "large batch; run explicitly in a release profile"]
    fn falcon_1024_large_batch_proof_bytes_are_pinned() {
        use aerie_falcon_native::Falcon1024;
        use aerie_falcon_statement::honest::honest_aggregation_1024;
        let digest = std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| -> Result<String, String> {
                let records = 2048_usize;
                let (statement, witness) =
                    honest_aggregation_1024(records, 0x1024_d16e).map_err(|e| e.to_string())?;
                let setup =
                    FalconAggregateSetup::<Falcon1024>::new(records).map_err(|e| e.to_string())?;
                let (proof_bytes, _) =
                    prove_aggregate(&setup, &statement, &witness).map_err(|e| e.to_string())?;
                let (public_keys, messages): (Vec<_>, Vec<_>) = statement
                    .records()
                    .iter()
                    .map(|record| (record.public_key().clone(), record.message().to_vec()))
                    .unzip();
                verify_aggregate(setup.verifier(), &public_keys, &messages, &proof_bytes)
                    .map_err(|e| e.to_string())?;
                eprintln!("proof bytes: Falcon-1024, N=2048: {}", proof_bytes.len());
                Ok(proof_digest(&proof_bytes))
            })
            .expect("spawn digest gate")
            .join()
            .expect("join digest gate")
            .expect("proof digest");
        assert_eq!(
            digest, "99a89ce83dd44d1ba09ac5e8d94b9ac7",
            "version 2 Falcon-1024 proof bytes moved at 2048 records"
        );
    }

    /// Regenerate the committed wire-fuzzer seeds from deterministic honest
    /// FN-DSA vectors. Each proof is verified before it replaces its seed.
    #[test]
    #[ignore = "writes crates/aerie-falcon/fuzz/seeds/wire_pcs_proof"]
    fn generate_wire_fuzz_seeds() {
        use aerie_falcon_native::Falcon1024;
        use aerie_falcon_statement::honest::{
            honest_aggregation_1024_common_salt, honest_aggregation_common_salt,
        };

        std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(|| {
                let output = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("fuzz/seeds/wire_pcs_proof");
                std::fs::create_dir_all(&output).expect("create fuzz seed directory");

                macro_rules! generate_for {
                    ($params:ty, $fixture:path, $seed:expr, $name:literal) => {{
                        let (statement, witness) =
                            $fixture(1, $seed).expect("build deterministic honest vector");
                        let setup = FalconAggregateSetup::<$params>::new(1)
                            .expect("build minimum-record setup");
                        let (public_keys, messages): (Vec<_>, Vec<_>) = statement
                            .records()
                            .iter()
                            .map(|record| (record.public_key().clone(), record.message().to_vec()))
                            .unzip();
                        for (mode, suffix) in
                            [(SaltMode::PerRecord, "v2p"), (SaltMode::Common, "v2c")]
                        {
                            let (proof, _) =
                                prove_aggregate_with_mode(&setup, &statement, &witness, mode)
                                    .expect("prove fuzz seed");
                            verify_aggregate(setup.verifier(), &public_keys, &messages, &proof)
                                .expect("verify fuzz seed");
                            let decoded = wire::decode_pcs_proof(<$params>::DEGREE_ID, &proof)
                                .expect("decode fuzz seed");
                            assert_eq!(decoded.records, 1);
                            assert_eq!(decoded.salt_mode, mode);
                            std::fs::write(output.join(format!("{}-{suffix}", $name)), proof)
                                .expect("write fuzz seed");
                        }
                    }};
                }

                generate_for!(
                    Falcon512,
                    honest_aggregation_common_salt,
                    0x0512_f022_u64,
                    "falcon512"
                );
                generate_for!(
                    Falcon1024,
                    honest_aggregation_1024_common_salt,
                    0x1024_f022_u64,
                    "falcon1024"
                );
            })
            .expect("spawn seed generator")
            .join()
            .expect("join seed generator");
    }

    #[test]
    fn prove_batch_rejects_mismatched_ring_keys() {
        let fixture = fixture(8);
        // The batch with a substituted key: the fail-closed decode
        // comparison must reject before proving.
        let mut mismatched = fixture.public_keys.clone();
        mismatched[0] = zero_public_key();
        let result = prove_batch(
            &fixture.batch,
            &mismatched,
            &fixture.salts,
            &fixture.messages,
        );
        assert!(result.is_err());
    }
}
