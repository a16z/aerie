//! Transport-bound aggregate proving over borrowed statement and witness inputs.
#[cfg(test)]
use super::pcs::VerifiedOpening;
use super::pcs::{self, FalconAggregateSetup, PcsStageMetrics};
#[cfg(any(test, feature = "internals"))]
pub use super::public_records_binding::public_records_binding;
use super::public_records_binding::{hashed_keys, public_records_binding_from_view};
use super::wire::{self, SaltFrame, SaltMode};
use super::AggregationParams;
#[cfg(any(test, feature = "internals"))]
use super::{
    public_records_binding::public_records_binding_of, wire::TRANSPORT_SALT_BYTES, FalconWitness,
};
#[cfg(any(test, feature = "internals"))]
use crate::StatementMismatch;
use crate::{AggregationError as Error, UnsupportedBatch};
#[cfg(any(test, feature = "internals"))]
use aerie_falcon::protocol::transport::derived_ring_keys;
#[cfg(any(test, feature = "internals"))]
pub use aerie_falcon::protocol::transport::{verify_aggregate, AggregateVerifyMetrics};
#[cfg(any(test, feature = "internals"))]
pub use aerie_falcon::protocol::transport::{verify_aggregate_impl, verify_aggregate_measured};
#[cfg(any(test, feature = "internals"))]
use aerie_falcon_native::source::CanonicalPublicKeyBytes;
use aerie_falcon_native::FalconParams;
use aerie_falcon_statement::AggregationStatement;
use aerie_falcon_witness::{AggregationWitness, FalconPreparedWitness, PreprocessingMetrics};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
/// Complete raw-record-to-proof timing for the transport-bound prover.
#[cfg_attr(
    not(feature = "internals"),
    allow(dead_code, reason = "read only by the measured entry points")
)]
#[derive(Clone, Copy, Debug)]
pub struct AggregateProveMetrics {
    pub total_ns: u128,
    pub preprocessing: PreprocessingMetrics,
    pub transport_materialize_ns: u128,
    pub transport_binding_ns: u128,
    pub encode_ns: u128,
    pub protocol: PcsStageMetrics,
}

fn prepare_witness<P: AggregationParams>(
    statement: &AggregationStatement<P>,
    witness: &AggregationWitness<P>,
) -> Result<FalconPreparedWitness, Error> {
    FalconPreparedWitness::from_aggregate(statement, witness).map_err(|error| {
        if matches!(
            error,
            aerie_falcon_statement::ProductionError::Aerie(_)
                | aerie_falcon_statement::ProductionError::Direct(_)
        ) {
            // Error-path index recovery costs up to one extra native preparation per record.
            // If no record fails alone, return the original error without an index.
            for (index, ((public_key, message), signature)) in statement
                .public_keys()
                .iter()
                .zip(statement.messages())
                .zip(witness.signatures())
                .enumerate()
            {
                let Ok(record) = aerie_falcon_statement::AggregationRecord::new(
                    public_key.clone(),
                    message.clone(),
                ) else {
                    continue;
                };
                let Ok(statement) = AggregationStatement::new(vec![record]) else {
                    continue;
                };
                let Ok(witness) = AggregationWitness::new(vec![signature.clone()]) else {
                    continue;
                };
                if let Err(error) = FalconPreparedWitness::from_aggregate(&statement, &witness) {
                    return Error::InvalidStatement(
                        crate::StatementError::new(error).at_record(index),
                    );
                }
            }
        }
        error.into()
    })
}

/// Prove an aggregate directly from its ordered public records and signatures.
#[inline]
pub fn prove_aggregate<P: AggregationParams>(
    setup: &FalconAggregateSetup<P>,
    statement: &AggregationStatement<P>,
    witness: &AggregationWitness<P>,
) -> Result<Vec<u8>, Error> {
    prove_aggregate_with_mode(setup, statement, witness, SaltMode::PerRecord)
}

/// Prove with an explicit salt transport mode, bound before the first
/// challenge. Common-salt mode requires every signature to use the same salt.
#[inline]
pub fn prove_aggregate_with_mode<P: AggregationParams>(
    setup: &FalconAggregateSetup<P>,
    statement: &AggregationStatement<P>,
    witness: &AggregationWitness<P>,
    mode: SaltMode,
) -> Result<Vec<u8>, Error> {
    prove_aggregate_impl(setup, statement, witness, mode).map(|(proof, _)| proof)
}

/// Prove with complete raw-record-to-wire timing, excluding reusable setup.
#[cfg(any(test, feature = "internals"))]
#[inline]
pub fn prove_aggregate_measured_with_mode<P: AggregationParams>(
    setup: &FalconAggregateSetup<P>,
    statement: &AggregationStatement<P>,
    witness: &AggregationWitness<P>,
    mode: SaltMode,
) -> Result<(Vec<u8>, AggregateProveMetrics), Error> {
    prove_aggregate_impl(setup, statement, witness, mode)
}

#[inline]
fn prove_aggregate_impl<P: AggregationParams>(
    setup: &FalconAggregateSetup<P>,
    statement: &AggregationStatement<P>,
    witness: &AggregationWitness<P>,
    mode: SaltMode,
) -> Result<(Vec<u8>, AggregateProveMetrics), Error> {
    // Reject a statement/witness length mismatch before the early commit
    // decodes and commits anything; ingestion repeats the check.
    if statement.len() != witness.signatures().len() {
        return Err(
            aerie_falcon_statement::ProductionError::WitnessLengthMismatch {
                statement: statement.len(),
                witness: witness.signatures().len(),
            }
            .into(),
        );
    }
    // The early s2 commitment uses the setup's manifest, so reject a live-count
    // mismatch before decoding signatures or preparing either witness path.
    if setup.records() != statement.len() {
        return Err(UnsupportedBatch::SetupSize {
            expected: setup.records(),
            actual: statement.len(),
        }
        .into());
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
    let (production, precommit) =
        rayon::join(|| prepare_witness::<P>(statement, witness), precommit_early);
    #[cfg(not(feature = "parallel"))]
    let (production, precommit) = (prepare_witness::<P>(statement, witness), precommit_early());
    let production = production?;
    let preprocessing = production.metrics();
    let mut batch = production.into_witness();
    super::pad_witness(&mut batch)?;
    let (precommit, early_s2) = precommit?;
    // The early groups were committed from the signature bytes; the witness
    // `s2` comes from the same bytes through native verification, in the same
    // record order. Divergence here means the committed tables do not
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
            let public_keys = statement.public_keys();
            let messages = statement.messages();
            let salts = match mode {
                SaltMode::Common => SaltFrame::Common(*witness.signatures()[0].salt()),
                SaltMode::PerRecord => SaltFrame::PerRecord(
                    witness
                        .signatures()
                        .iter()
                        .map(|signature| *signature.salt())
                        .collect(),
                ),
            };
            (public_keys, messages, salts)
        }
    );
    let mut transport_binding_ns = 0;
    let binding = aerie_falcon_statement::staged!(
        "v1.prove.input.transport_binding",
        transport_binding_ns,
        public_records_binding_from_view(&hashed_keys(public_keys), salts.view(), messages)?
    );
    let (proof, protocol) =
        pcs::prove_prepared_with_precommit(&batch, &binding, mode, setup, precommit)?;
    let mut encode_ns = 0;
    let proof_bytes = aerie_falcon_statement::staged!(
        "v1.prove.output.encode",
        encode_ns,
        wire::encode_pcs_proof_from_view(P::DEGREE_ID, statement.len(), salts.view(), &proof)
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

/// Prove one batch of parameter set `P` and serialize its transport-bound proof.
///
/// Fail-closed consistency: the batch must have degree `P::DEGREE`, the
/// batch's public keys must decode to the batch's ring-key rows, and the
/// transport shapes must match the live count; a target mismatch is left to
/// verification, which recomputes every target.
#[cfg(any(test, feature = "internals"))]
pub fn prove_batch<P: AggregationParams>(
    batch: &FalconWitness,
    public_keys: &[CanonicalPublicKeyBytes<P>],
    salts: &[[u8; TRANSPORT_SALT_BYTES]],
    messages: &[Vec<u8>],
) -> Result<(Vec<u8>, PcsStageMetrics), Error> {
    if batch.degree() != P::DEGREE {
        return Err(StatementMismatch::ParameterSet {
            expected: P::DEGREE_ID.parameter_set(),
            actual: super::FalconDegree::from_degree(batch.degree())?.parameter_set(),
        }
        .into());
    }
    let records = batch.live_rows;
    if public_keys.len() != records || salts.len() != records || messages.len() != records {
        return Err(StatementMismatch::PublicInputShape.into());
    }
    let derived_h = derived_ring_keys(public_keys, batch.padded_rows);
    if derived_h.iter().copied().ne(batch.h.iter().copied()) {
        return Err(Error::internal(
            "batch ring keys differ from canonical public keys",
        ));
    }
    let binding = public_records_binding_of::<P, _>(public_keys, salts, messages)?;
    let (proof, metrics) = pcs::prove(batch, &binding)?;
    let proof_bytes = wire::encode_pcs_proof(P::DEGREE_ID, records, salts, &proof)?;
    Ok((proof_bytes, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::FalconDegree;
    use crate::FalconVerifierSetup;
    use crate::MalformedProof;
    use aerie_falcon_native::source::CanonicalPublicKeyBytes;
    use aerie_falcon_native::{Falcon1024, Falcon512};
    use aerie_falcon_witness::fixtures::constructed_aggregation;
    use aerie_falcon_witness::honest::{
        honest_aggregation, HonestParams, SaltMode as FixtureSaltMode,
    };
    use aerie_falcon_witness::FalconPreparedWitness;
    use shake::{
        digest::{ExtendableOutput, Update, XofReader},
        Shake256,
    };

    use std::fmt::Write;

    use DigestFixture::Constructed;
    use DigestPath::{Aggregate, Batch};
    use SaltMode::{Common, PerRecord};

    // Tampering tests edit keys and messages directly; like the public
    // statement type, the lists always pair one key with one message.
    fn verify_lists_measured<P: AggregationParams>(
        setup: &FalconVerifierSetup<P>,
        keys: &[CanonicalPublicKeyBytes<P>],
        messages: &[Vec<u8>],
        proof: &[u8],
    ) -> Result<(VerifiedOpening, AggregateVerifyMetrics), Error> {
        assert_eq!(keys.len(), messages.len(), "one message per key");
        verify_aggregate_impl(setup, keys, messages, proof)
    }

    fn verify_lists<P: AggregationParams>(
        setup: &FalconVerifierSetup<P>,
        keys: &[CanonicalPublicKeyBytes<P>],
        messages: &[Vec<u8>],
        proof: &[u8],
    ) -> Result<VerifiedOpening, Error> {
        verify_lists_measured(setup, keys, messages, proof).map(|(verified, _)| verified)
    }

    fn prove_measured<P: AggregationParams>(
        setup: &FalconAggregateSetup<P>,
        statement: &AggregationStatement<P>,
        witness: &AggregationWitness<P>,
    ) -> Result<(Vec<u8>, AggregateProveMetrics), Error> {
        prove_aggregate_measured_with_mode(setup, statement, witness, SaltMode::PerRecord)
    }

    #[test]
    fn aggregate_prover_rejects_a_different_setup_record_count() {
        fn check<P: AggregationParams>() {
            let setup = FalconAggregateSetup::<P>::new(8).unwrap();
            // Five records share the setup's padded shape; nine require a
            // larger one. Both must report the live-count mismatch.
            for records in [5, 9] {
                let (statement, witness) = constructed_aggregation::<P>(records).unwrap();
                let result = prove_aggregate(&setup, &statement, &witness);
                assert!(
                    matches!(
                        result,
                        Err(Error::UnsupportedBatch(UnsupportedBatch::SetupSize {
                            expected: 8,
                            actual,
                        })) if actual == records
                    ),
                    "{} with {records} records: {result:?}",
                    P::NAME
                );
            }
        }
        check::<Falcon512>();
        check::<Falcon1024>();
    }

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
        let mut bytes = [0_u8; Falcon512::PUBLIC_KEY_BYTES];
        bytes[0] = 0x09;
        CanonicalPublicKeyBytes::parse(&bytes).expect("canonical zero key")
    }

    fn zero_public_key_of<P: FalconParams>() -> CanonicalPublicKeyBytes<P> {
        let mut bytes = vec![0_u8; P::PUBLIC_KEY_BYTES];
        bytes[0] = P::PUBLIC_KEY_HEADER;
        CanonicalPublicKeyBytes::parse(&bytes).expect("canonical zero key")
    }

    /// Hash-derived constructed records: relation-positive rows with
    /// genuine per-record `HashToPoint` targets, salts, and messages.
    /// The constructed fixture deliberately shares ONE public key across
    /// records, so key-substitution tests swap in `zero_public_key`.
    fn fixture(records: usize) -> Fixture {
        let (statement, witness) =
            constructed_aggregation::<Falcon512>(records).expect("constructed statement");
        let production =
            FalconPreparedWitness::from_aggregate(&statement, &witness).expect("production batch");
        Fixture {
            batch: production.into_witness(),
            public_keys: statement.public_keys().to_vec(),
            salts: witness
                .signatures()
                .iter()
                .map(|signature| *signature.salt())
                .collect(),
            messages: statement.messages().to_vec(),
        }
    }

    fn d3_public_api_of<P: AggregationParams + HonestParams>() {
        for mode in [SaltMode::PerRecord, SaltMode::Common] {
            let (statement, witness) = honest_aggregation::<P>(
                5,
                0xd3_0512,
                match mode {
                    SaltMode::PerRecord => FixtureSaltMode::PerRecord,
                    SaltMode::Common => FixtureSaltMode::Common,
                },
            )
            .unwrap();
            let setup = FalconAggregateSetup::<P>::new(statement.len()).unwrap();
            let bytes = match mode {
                SaltMode::PerRecord => prove_aggregate(&setup, &statement, &witness),
                SaltMode::Common => prove_aggregate_with_mode(&setup, &statement, &witness, mode),
            }
            .unwrap();
            let result: Result<(), Error> = verify_aggregate(setup.verifier(), &statement, &bytes);
            result.unwrap();
            let (measured_bytes, _) =
                prove_aggregate_measured_with_mode(&setup, &statement, &witness, mode).unwrap();
            assert_eq!(bytes, measured_bytes);
            verify_aggregate_measured(setup.verifier(), &statement, &bytes).unwrap();
            let keys = statement.public_keys();
            let messages = statement.messages();
            let cloned = statement.to_records();
            for (index, record) in cloned.iter().enumerate() {
                assert!(core::ptr::eq(
                    &raw const keys[index],
                    &raw const statement.public_keys()[index]
                ));
                assert!(core::ptr::eq(
                    messages[index].as_slice(),
                    statement.messages()[index].as_slice()
                ));
                assert_eq!(&keys[index], record.public_key());
                assert_eq!(messages[index], record.message());
            }
            let mut changed = statement.to_records();
            changed[0] = aerie_falcon_statement::AggregationRecord::new(
                changed[0].public_key().clone(),
                [changed[0].message(), &[0x5a]].concat(),
            )
            .unwrap();
            let changed_message = AggregationStatement::new(changed).unwrap();
            let mut changed = statement.to_records();
            changed[0] = aerie_falcon_statement::AggregationRecord::new(
                zero_public_key_of::<P>(),
                changed[0].message().to_vec(),
            )
            .unwrap();
            let changed_key = AggregationStatement::new(changed).unwrap();
            let mut changed = statement.to_records();
            changed.swap(0, 1);
            let changed_order = AggregationStatement::new(changed).unwrap();
            let shorter = AggregationStatement::new(statement.to_records()[..4].to_vec()).unwrap();
            for changed in [&changed_message, &changed_key, &changed_order, &shorter] {
                let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    verify_aggregate(setup.verifier(), changed, &bytes)
                }));
                assert!(
                    rejected.is_ok(),
                    "altered statement must not panic: {}",
                    P::NAME
                );
                assert!(
                    rejected.unwrap().is_err(),
                    "altered statement must reject: {}",
                    P::NAME
                );
            }
        }
    }

    #[test]
    fn d3_public_api_borrows_statement_and_rejects_mutations() {
        d3_public_api_of::<Falcon512>();
        d3_public_api_of::<Falcon1024>();
    }

    #[test]
    fn common_salt_roundtrip_binds_mode_and_salt() {
        {
            let (statement, witness) = constructed_aggregation::<Falcon512>(1).unwrap();
            let records = 5;
            let statement = AggregationStatement::<Falcon512>::new(vec![
                statement.to_records()[0]
                    .clone();
                records
            ])
            .unwrap();
            let witness = AggregationWitness::<Falcon512>::new(vec![
                witness.signatures()[0].clone();
                records
            ])
            .unwrap();
            let setup = FalconAggregateSetup::<Falcon512>::new(records).unwrap();
            for mode in [SaltMode::Common, SaltMode::PerRecord] {
                let (bytes, _) =
                    prove_aggregate_measured_with_mode(&setup, &statement, &witness, mode).unwrap();
                verify_aggregate(setup.verifier(), &statement, &bytes).unwrap();
                let decoded = wire::decode_pcs_proof(FalconDegree::Falcon512, &bytes).unwrap();
                assert_eq!(decoded.salt_mode, mode);
                let repeated = vec![*witness.signatures()[0].salt(); records];
                for (record, salt) in repeated.iter().enumerate() {
                    assert_eq!(decoded.salts.view().record(record), salt);
                }
                let reencoded = wire::encode_pcs_proof_from_salt_frame(
                    decoded.degree,
                    records,
                    &decoded.salts,
                    &decoded.proof,
                )
                .unwrap();
                assert_eq!(bytes, reencoded);
                if mode == SaltMode::PerRecord {
                    assert_eq!(
                        bytes,
                        wire::encode_pcs_proof(decoded.degree, records, &repeated, &decoded.proof)
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
                    &repeated,
                    &decoded.proof,
                    other,
                )
                .unwrap();
                let expected_delta = (records - 1) * TRANSPORT_SALT_BYTES;
                assert_eq!(bytes.len().abs_diff(relabelled.len()), expected_delta);
                assert!(wire::decode_pcs_proof(decoded.degree, &relabelled).is_ok());
                assert!(matches!(
                    verify_aggregate(setup.verifier(), &statement, &relabelled),
                    Err(Error::ProofRejected { .. })
                ));
                let mut tampered = bytes.clone();
                tampered[wire::HEADER_BYTES] ^= 1;
                assert!(matches!(
                    verify_aggregate(setup.verifier(), &statement, &tampered),
                    Err(Error::ProofRejected { .. })
                ));
                let mut unequal = repeated;
                unequal[records - 1][0] ^= 1;
                assert!(matches!(
                    wire::encode_pcs_proof_with_mode(
                        decoded.degree,
                        records,
                        &unequal,
                        &decoded.proof,
                        SaltMode::Common,
                    ),
                    Err(Error::UnsupportedBatch(UnsupportedBatch::SaltMode))
                ));
            }
        };
    }

    #[test]
    fn transport_proof_roundtrips_with_streamed_targets() {
        let outcome = (|| -> Result<(), String> {
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
                verify_lists(
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
                    matches!(
                        verify_lists(
                            &verifier_setup,
                            &fixture.public_keys,
                            &fixture.messages,
                            &tampered
                        ),
                        Err(Error::ProofRejected { .. })
                    ),
                    "tampered salt at {records} records"
                );

                // A message that differs from the one the prover bound
                // changes the binding digest and the recomputed target.
                let mut altered = fixture.messages.clone();
                altered[0][0] ^= 1;
                assert!(
                    matches!(
                        verify_lists(
                            &verifier_setup,
                            &fixture.public_keys,
                            &altered,
                            &proof_bytes
                        ),
                        Err(Error::ProofRejected { .. })
                    ),
                    "altered message at {records} records"
                );

                // Two messages swapped between records likewise.
                if records > 1 {
                    let mut swapped = fixture.messages.clone();
                    swapped.swap(0, 1);
                    assert!(
                        matches!(
                            verify_lists(
                                &verifier_setup,
                                &fixture.public_keys,
                                &swapped,
                                &proof_bytes
                            ),
                            Err(Error::ProofRejected { .. })
                        ),
                        "swapped messages at {records} records"
                    );
                }

                // A substituted public key changes the streamed ring
                // keys, the recomputed targets, and the binding digest.
                let mut substituted = fixture.public_keys.clone();
                substituted[0] = zero_public_key();
                assert!(
                    matches!(
                        verify_lists(
                            &verifier_setup,
                            &substituted,
                            &fixture.messages,
                            &proof_bytes
                        ),
                        Err(Error::ProofRejected { .. })
                    ),
                    "substituted public key at {records} records"
                );

                // A statement with the wrong record count is rejected before
                // any derivation.
                let result = verify_lists(
                    &verifier_setup,
                    &fixture.public_keys[..records - 1],
                    &fixture.messages[..records - 1],
                    &proof_bytes,
                );
                let wrong_count = matches!(
                    result,
                    Err(Error::StatementMismatch(StatementMismatch::RecordCount {
                        expected,
                        actual,
                    })) if expected == records - 1 && actual == records
                );
                assert!(wrong_count, "{result:?}");
            }
            Ok(())
        })();
        outcome.expect("transport roundtrip");
    }

    /// Falcon-1024 end to end on real FN-DSA-1024 keys and signatures:
    /// source-bound ingestion, four real commitments, the wire with
    /// parameter-set byte `10`, verifier-local `HashToPoint` at degree
    /// 1024 from canonical Falcon-1024 key bytes; cross-degree proofs and
    /// keys reject in both directions; a changed salt or message rejects.
    #[test]
    fn single_falcon_1024_roundtrips() {
        {
            let (statement, witness) =
                aerie_falcon_witness::honest::honest_aggregation::<Falcon1024>(
                    1,
                    0x0010_2401,
                    aerie_falcon_witness::honest::SaltMode::PerRecord,
                )
                .unwrap();
            let setup = FalconAggregateSetup::<aerie_falcon_native::Falcon1024>::new(1).unwrap();
            let bytes = prove_aggregate(&setup, &statement, &witness).unwrap();
            verify_aggregate(setup.verifier(), &statement, &bytes).unwrap();
        };
    }

    #[test]
    fn falcon_1024_aggregate_roundtrips_on_honest_fn_dsa_signatures() {
        let outcome = (|| -> Result<(), String> {
            let records = 8_usize;
            let (statement, witness) = honest_aggregation::<Falcon1024>(
                records,
                0x1024_0a11,
                aerie_falcon_witness::honest::SaltMode::PerRecord,
            )
            .map_err(|e| e.to_string())?;
            let setup =
                FalconAggregateSetup::<Falcon1024>::new(records).map_err(|e| e.to_string())?;
            let (bytes, metrics) =
                prove_measured(&setup, &statement, &witness).map_err(|e| e.to_string())?;
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
            assert_eq!(decoded.salts, SaltFrame::PerRecord(salts));

            let public_keys = statement.public_keys().to_vec();
            let messages = statement.messages().to_vec();
            let verified = verify_lists(setup.verifier(), &public_keys, &messages, &bytes)
                .map_err(|e| e.to_string())?;
            assert_eq!(verified.ledger.groups.len(), 4);

            // Cross-parameter-set rejection in both directions. Each
            // mismatched pair stops at the header's parameter-set byte,
            // before the record count or any proof material is read.
            let (statement_512, witness_512) =
                constructed_aggregation::<Falcon512>(records).map_err(|e| e.to_string())?;
            let keys_512 = statement_512.public_keys().to_vec();
            let messages_512 = statement_512.messages().to_vec();
            let setup_512 =
                FalconAggregateSetup::<Falcon512>::new(records).map_err(|e| e.to_string())?;
            let (bytes_512, _) = prove_measured(&setup_512, &statement_512, &witness_512)
                .map_err(|e| e.to_string())?;
            assert_eq!(bytes_512[wire::HEADER_PARAMETER_SET_OFFSET], 9);
            assert_eq!(bytes[wire::HEADER_PARAMETER_SET_OFFSET], 10);
            let rejects_on_parameter_set = |result: Result<VerifiedOpening, Error>| {
                matches!(
                    result,
                    Err(Error::StatementMismatch(
                        StatementMismatch::ParameterSet { .. }
                    ))
                )
            };
            // A Falcon-1024 proof under the Falcon-512 verifier, and the
            // converse.
            assert!(rejects_on_parameter_set(verify_lists(
                setup_512.verifier(),
                &keys_512,
                &messages_512,
                &bytes,
            )));
            assert!(rejects_on_parameter_set(verify_lists(
                setup.verifier(),
                &public_keys,
                &messages,
                &bytes_512,
            )));
            // A flipped parameter-set byte: the relabeled frame is refused
            // by the verifier it names (the body has the other shape) and
            // by the one it came from (the byte differs).
            let mut relabeled_1024 = bytes.clone();
            relabeled_1024[wire::HEADER_PARAMETER_SET_OFFSET] = 9;
            assert!(rejects_on_parameter_set(verify_lists(
                setup.verifier(),
                &public_keys,
                &messages,
                &relabeled_1024,
            )));
            let error = verify_lists(
                setup_512.verifier(),
                &keys_512,
                &messages_512,
                &relabeled_1024,
            )
            .expect_err("relabeled body");
            assert!(matches!(
                error,
                Error::MalformedProof(MalformedProof::NonceEncoding)
            ));
            let mut relabeled_512 = bytes_512.clone();
            relabeled_512[wire::HEADER_PARAMETER_SET_OFFSET] = 10;
            assert!(rejects_on_parameter_set(verify_lists(
                setup_512.verifier(),
                &keys_512,
                &messages_512,
                &relabeled_512,
            )));
            let error = verify_lists(setup.verifier(), &public_keys, &messages, &relabeled_512)
                .expect_err("relabeled body");
            assert!(matches!(
                error,
                Error::MalformedProof(MalformedProof::NonceEncoding)
            ));
            // Proving a Falcon-1024 statement under a Falcon-512 setup
            // does not type-check: `FalconAggregateSetup<P>` fixes `P`.

            // Tampering: one salt byte, one message byte.
            let header = wire::HEADER_BYTES;
            let mut tampered = bytes.clone();
            tampered[header + 3] ^= 1;
            assert!(matches!(
                verify_lists(setup.verifier(), &public_keys, &messages, &tampered),
                Err(Error::ProofRejected { .. })
            ));
            let mut other_messages = messages.clone();
            other_messages[2].push(0x5a);
            assert!(matches!(
                verify_lists(setup.verifier(), &public_keys, &other_messages, &bytes),
                Err(Error::ProofRejected { .. })
            ));
            Ok(())
        })();
        outcome.expect("Falcon-1024 aggregate round trip");
    }

    /// Falcon-1024 at a live count that pads to `N_sig* = 16`, on real
    /// FN-DSA-1024 signatures, against the generated
    /// `falcon-v1-batch-32-budget1-final` row (the Falcon-1024 shapes at
    /// `N_sig* = 16` equal the Falcon-512 shapes at `N_sig* = 32`): the aggregate verifies and a
    /// changed message rejects.
    #[test]
    fn falcon_1024_aggregate_roundtrips_at_a_batch_padded_to_16() {
        let outcome = (|| -> Result<(), String> {
            let records = 12_usize;
            let (statement, witness) = honest_aggregation::<Falcon1024>(
                records,
                0x1024_0a16,
                aerie_falcon_witness::honest::SaltMode::PerRecord,
            )
            .map_err(|e| e.to_string())?;
            let setup =
                FalconAggregateSetup::<Falcon1024>::new(records).map_err(|e| e.to_string())?;
            let (bytes, metrics) =
                prove_measured(&setup, &statement, &witness).map_err(|e| e.to_string())?;
            assert_eq!(metrics.preprocessing.records, records);
            let decoded = wire::decode_pcs_proof(FalconDegree::Falcon1024, &bytes)
                .map_err(|e| e.to_string())?;
            assert_eq!(decoded.degree, FalconDegree::Falcon1024);
            assert_eq!(decoded.proof.commitments.len(), 4);

            let public_keys = statement.public_keys().to_vec();
            let messages = statement.messages().to_vec();
            let verified = verify_lists(setup.verifier(), &public_keys, &messages, &bytes)
                .map_err(|e| e.to_string())?;
            assert_eq!(verified.ledger.groups.len(), 4);

            let mut other_messages = messages.clone();
            other_messages[records - 1].push(0x5a);
            assert!(matches!(
                verify_lists(setup.verifier(), &public_keys, &other_messages, &bytes,),
                Err(Error::ProofRejected { .. })
            ));
            Ok(())
        })();
        outcome.expect("Falcon-1024 aggregate round trip at 12 records");
    }

    #[test]
    fn aggregate_api_accounts_for_source_witness_through_wire_verification() {
        let outcome = (|| -> Result<(), String> {
            let (statement, witness) =
                constructed_aggregation::<Falcon512>(8).map_err(|error| error.to_string())?;
            let statement = &statement;
            let setup = FalconAggregateSetup::<Falcon512>::new(statement.len())
                .map_err(|error| error.to_string())?;
            let (proof_bytes, prove_metrics) =
                prove_measured(&setup, statement, &witness).map_err(|error| error.to_string())?;
            let (_, verify_metrics) =
                verify_aggregate_measured(setup.verifier(), statement, &proof_bytes)
                    .map_err(|error| error.to_string())?;

            assert_eq!(prove_metrics.protocol.records, statement.len());
            assert!(prove_metrics.total_ns >= prove_metrics.protocol.commit_ns);
            assert!(verify_metrics.total_ns >= verify_metrics.pcs_verify_ns);
            Ok(())
        })();
        outcome.expect("aggregate API roundtrip");
    }

    #[test]
    fn prepared_witness_preserves_transport_proof_and_wire_bytes() {
        let outcome = (|| -> Result<(), String> {
            let (statement, witness) =
                constructed_aggregation::<Falcon512>(8).map_err(|error| error.to_string())?;
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

            let public_keys = statement.public_keys().to_vec();
            let messages = statement.messages().to_vec();
            let salts = witness
                .signatures()
                .iter()
                .map(|signature| *signature.salt())
                .collect::<Vec<_>>();
            let binding = public_records_binding(&public_keys, &salts, &messages)
                .map_err(|error| error.to_string())?;
            let setup = FalconAggregateSetup::<Falcon512>::new(statement.len())
                .map_err(|error| error.to_string())?;
            let (aggregate_bytes, _) =
                prove_measured(&setup, statement, &witness).map_err(|error| error.to_string())?;
            let (prepared_proof, _) = pcs::prove_prepared(prepared.witness(), &binding, &setup)
                .map_err(|error| error.to_string())?;
            let prepared_proof_bytes = wire::encode_pcs_proof(
                FalconDegree::Falcon512,
                statement.len(),
                &salts,
                &prepared_proof,
            )
            .map_err(|error| error.to_string())?;
            assert_eq!(aggregate_bytes, prepared_proof_bytes);
            Ok(())
        })();
        outcome.expect("prepared witness parity");
    }

    /// The 128-bit SHAKE256 digest of a proof, in hex.
    fn proof_digest(proof_bytes: &[u8]) -> String {
        let mut hasher = Shake256::default();
        hasher.update(proof_bytes);
        let mut digest = [0_u8; 16];
        hasher.finalize_xof().read(&mut digest);
        digest.iter().fold(String::new(), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
    }

    #[derive(Clone, Copy)]
    enum DigestFixture {
        Constructed,
        Honest(u64),
    }

    #[derive(Clone, Copy)]
    enum DigestPath {
        Batch,
        Aggregate,
    }

    fn assert_digest<P: AggregationParams + HonestParams>(
        records: usize,
        fixture: DigestFixture,
        salt_mode: SaltMode,
        path: DigestPath,
        expected: &str,
    ) {
        let (statement, witness) = match fixture {
            DigestFixture::Constructed => constructed_aggregation::<P>(records),
            DigestFixture::Honest(seed) => honest_aggregation::<P>(
                records,
                seed,
                match salt_mode {
                    SaltMode::PerRecord => FixtureSaltMode::PerRecord,
                    SaltMode::Common => FixtureSaltMode::Common,
                },
            ),
        }
        .expect("digest fixture");
        assert!(FalconAggregateSetup::<P>::supports(records));
        let public_keys = statement.public_keys().to_vec();
        let messages = statement.messages().to_vec();
        let proof_bytes = match path {
            DigestPath::Batch => {
                assert_eq!(salt_mode, SaltMode::PerRecord);
                let batch = FalconPreparedWitness::from_aggregate(&statement, &witness)
                    .expect("digest batch")
                    .into_witness();
                let salts: Vec<_> = witness
                    .signatures()
                    .iter()
                    .map(|signature| *signature.salt())
                    .collect();
                let (proof_bytes, _) =
                    prove_batch(&batch, &public_keys, &salts, &messages).expect("digest proof");
                let setup = FalconVerifierSetup::<P>::new(records).expect("digest verifier setup");
                verify_lists(&setup, &public_keys, &messages, &proof_bytes)
                    .expect("digest verification");
                proof_bytes
            }
            DigestPath::Aggregate => {
                let setup = FalconAggregateSetup::<P>::new(records).expect("digest setup");
                let (proof_bytes, _) =
                    prove_aggregate_measured_with_mode(&setup, &statement, &witness, salt_mode)
                        .expect("digest proof");
                verify_lists(setup.verifier(), &public_keys, &messages, &proof_bytes)
                    .expect("digest verification");
                proof_bytes
            }
        };
        let digest = proof_digest(&proof_bytes);
        eprintln!(
            "proof bytes: {}, {salt_mode:?}, N={records}: {}, digest={digest}",
            P::NAME,
            proof_bytes.len()
        );
        assert_eq!(digest, expected, "version 1 proof bytes moved");
    }

    const HONEST_512: DigestFixture = DigestFixture::Honest(0x0512_d16e);
    const HONEST_1024: DigestFixture = DigestFixture::Honest(0x1024_d16e);

    const PIN_512: &str = "d9917aa254074917bf6dd24009edbd34";
    const PIN_1024: &str = "e21ae4a69bd8b9179e3e2cda62d3e8cc";
    const PIN_512_LARGE: &str = "29c171d45842b1efa00e30947f022180";
    const PIN_1024_LARGE: &str = "50a1accf22b78d4f7b96f1e443ea3759";
    const PIN_512_COMMON: &str = "c43c657243f2245c7d6e03387819b0e5";
    const PIN_1024_COMMON: &str = "1d2a69d997f41f413751b498ce7a3fea";
    const PIN_512_COMMON_LARGE: &str = "bc1e87a3b4064e3068ea582589d345e1";
    const PIN_1024_COMMON_LARGE: &str = "004f71882d7b838fd48b43d54211c6d4";

    /// Byte-identity gate for the complete frame at fixed constructed inputs.
    #[test]
    fn proof_bytes_are_pinned() {
        assert_digest::<Falcon512>(8, Constructed, PerRecord, Batch, PIN_512);
    }

    /// Byte-identity gate for the seeded honest Falcon-1024 fixture.
    #[test]
    fn falcon_1024_proof_bytes_are_pinned() {
        let records = 3;
        assert!(FalconAggregateSetup::<Falcon1024>::supports(records - 1));
        assert_digest::<Falcon1024>(records, HONEST_1024, PerRecord, Aggregate, PIN_1024);
    }

    /// Byte-identity gate covering parallel and bucketed prover kernels.
    #[test]
    #[ignore = "large batch; run explicitly in a release profile"]
    fn large_batch_proof_bytes_are_pinned() {
        assert_digest::<Falcon512>(4096, Constructed, PerRecord, Batch, PIN_512_LARGE);
    }

    /// Falcon-1024 byte-identity gate covering large-batch prover kernels.
    #[test]
    #[ignore = "large batch; run explicitly in a release profile"]
    fn falcon_1024_large_batch_proof_bytes_are_pinned() {
        assert_digest::<Falcon1024>(2048, HONEST_1024, PerRecord, Aggregate, PIN_1024_LARGE);
    }

    /// Common-salt byte-identity gate with distinct honest Falcon-512 rows.
    #[test]
    fn common_salt_proof_bytes_are_pinned() {
        assert_digest::<Falcon512>(8, HONEST_512, Common, Aggregate, PIN_512_COMMON);
    }

    /// Common-salt byte-identity gate with distinct honest Falcon-1024 rows.
    #[test]
    fn falcon_1024_common_salt_proof_bytes_are_pinned() {
        assert_digest::<Falcon1024>(3, HONEST_1024, Common, Aggregate, PIN_1024_COMMON);
    }

    /// Common-salt byte-identity gate covering large Falcon-512 batches.
    #[test]
    #[ignore = "large batch; run explicitly in a release profile"]
    fn common_salt_large_batch_proof_bytes_are_pinned() {
        assert_digest::<Falcon512>(4096, HONEST_512, Common, Aggregate, PIN_512_COMMON_LARGE);
    }

    /// Common-salt byte-identity gate covering large Falcon-1024 batches.
    #[test]
    #[ignore = "large batch; run explicitly in a release profile"]
    fn falcon_1024_common_salt_large_batch_proof_bytes_are_pinned() {
        assert_digest::<Falcon1024>(2048, HONEST_1024, Common, Aggregate, PIN_1024_COMMON_LARGE);
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
        assert!(matches!(result, Err(Error::Backend(_))));
    }
}
