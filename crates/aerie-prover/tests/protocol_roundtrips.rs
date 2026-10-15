//! Producer-backed regression tests for common protocol owners.
#![cfg(feature = "internals")]
use aerie_prover as protocol;
use aerie_prover::{prove_aggregate, FalconAggregateSetup, RejectionStage};
mod error {
    use aerie_prover::*;
    mod tests {
        use super::*;
        use std::error::Error as _;

        #[test]
        fn opaque_errors_preserve_sources_and_stable_display() {
            let backend = AggregationError::backend(std::io::Error::other("commitment failed"));
            assert!(matches!(backend, AggregationError::Backend(_)));
            assert_eq!(backend.to_string(), "aggregation backend failed");
            let source = backend.source().unwrap();
            assert_eq!(source.to_string(), "commitment failed");
            assert_eq!(
                source.downcast_ref::<std::io::Error>().unwrap().kind(),
                std::io::ErrorKind::Other
            );

            let statement = AggregationError::from(
                aerie_falcon_statement::ProductionError::IntegerOverflow("statement payload"),
            );
            assert!(matches!(statement, AggregationError::InvalidStatement(_)));
            assert_eq!(
                statement.to_string(),
                "invalid statement: statement payload length exceeds u32"
            );
            let indexed = AggregationError::from(
                crate::protocol::source_tables::SourceTableError::Encodability {
                    record: 3,
                    source: StatementError::new(aerie_falcon_statement::ProductionError::Aerie(
                        aerie_falcon_native::AerieError::InvalidRecord("invalid key".into()),
                    )),
                },
            );
            assert_eq!(
                indexed.to_string(),
                "invalid statement: record 3: canonical FN-DSA input: invalid record: invalid key"
            );
            assert!(AggregationError::from(MalformedProof::Truncated)
                .source()
                .is_none());
            assert!(AggregationError::ProjectionRejected.source().is_none());
            let mut messages = vec![statement.to_string()];
            let mut source = statement.source();
            while let Some(cause) = source {
                messages.push(cause.to_string());
                source = cause.source();
            }
            assert_eq!(
                messages,
                ["invalid statement: statement payload length exceeds u32"]
            );
            let AggregationError::InvalidStatement(statement) = statement else {
                unreachable!("the production error is an invalid statement")
            };
            assert_eq!(statement.record(), None);
            assert!(statement.source().is_none());
        }

        #[test]
        fn ingestion_error_identifies_one_invalid_record_among_valid_records() {
            use aerie_falcon_native::Falcon512;
            use aerie_falcon_statement::{AggregationRecord, AggregationStatement};
            use aerie_falcon_witness::fixtures::constructed_aggregation;

            let (statement, witness) = constructed_aggregation::<Falcon512>(3).unwrap();
            let mut records = statement.to_records();
            let mut message = records[1].message().to_vec();
            message.push(0x5a);
            records[1] = AggregationRecord::new(records[1].public_key().clone(), message).unwrap();
            let statement = AggregationStatement::new(records).unwrap();
            let setup = crate::FalconAggregateSetup::<Falcon512>::new(3).unwrap();
            let error = crate::prove_aggregate(&setup, &statement, &witness).unwrap_err();
            let AggregationError::InvalidStatement(error) = error else {
                panic!("expected an invalid statement, got {error:?}")
            };
            assert_eq!(error.record(), Some(1));
            let message = error.to_string();
            assert_eq!(message.matches("record 1:").count(), 1, "{message}");
            assert!(error.source().is_none(), "{error:?}");
        }
    }
}
mod openings {
    use aerie_prover::openings::*;
    mod tests {
        use super::*;
        use crate::protocol::manifest::{CommitmentManifest, GroupRole};
        use crate::protocol::pcs::{self, FalconVerifierSetup};
        use crate::protocol::public_inputs::PublicInputs;
        use crate::protocol::ring::{PublicRingKeys, PublicTargets};
        use crate::protocol::source_tables::{slack_group, SourceTables};
        use crate::protocol::AggregationParams;
        use crate::protocol::{FalconDegree, FalconWitness};
        use aerie_falcon::protocol::algebra::{evaluate_mle, lift_table};
        use aerie_falcon_native::{Falcon1024, Falcon512};
        use aerie_falcon_witness::DirectRow;

        fn with_single_thread_pool<T: Send>(body: impl FnOnce() -> T + Send) -> T {
            #[cfg(feature = "parallel")]
            {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .expect("proof thread pool");
                pool.install(body)
            }
            #[cfg(not(feature = "parallel"))]
            body()
        }

        /// Native polynomial tables in canonical group and polynomial order.
        fn group_tables(batch: &FalconWitness, tables: &SourceTables) -> [Vec<Vec<i64>>; 4] {
            [
                vec![batch.s1.iter().copied().map(i64::from).collect()],
                vec![slack_group(batch).iter().copied().map(i64::from).collect()],
                (0..3)
                    .map(|source| tables.coefficient_digits(source).map(i64::from).collect())
                    .chain(std::iter::once(
                        tables.sign_digits().map(i64::from).collect(),
                    ))
                    .collect(),
                (0..3)
                    .map(|index| {
                        tables
                            .budget(index)
                            .iter()
                            .copied()
                            .map(i64::from)
                            .collect()
                    })
                    .collect(),
            ]
        }

        fn verify_opening<P: AggregationParams>(
            batch: &FalconWitness,
            degree: FalconDegree,
            records: usize,
        ) -> OpeningLedger {
            let proof = with_single_thread_pool(|| {
                pcs::prove(batch, b"opening ledger fixture          ")
                    .expect("production PCS prover")
                    .0
            });
            with_single_thread_pool(|| {
                let verifier_setup =
                    FalconVerifierSetup::<P>::new(records).expect("verifier setup");
                let public = PublicInputs {
                    degree,
                    records,
                    salt_mode: crate::protocol::wire::SaltMode::PerRecord,
                    public_records: b"opening ledger fixture          ",
                    h: PublicRingKeys::Expanded(&batch.h),
                    t: PublicTargets::Expanded(&batch.t),
                };
                pcs::verify(&public, &verifier_setup, &proof)
                    .expect("production proof verifies")
                    .ledger
            })
        }

        #[test]
        fn every_group_claim_is_the_true_native_mle_at_its_shared_point() {
            for (degree, records) in [
                (FalconDegree::Falcon512, 3_usize),
                (FalconDegree::Falcon512, 8),
                (FalconDegree::Falcon1024, 3),
                (FalconDegree::Falcon1024, 4),
            ] {
                let batch = if degree == FalconDegree::Falcon512 {
                    let rows = (0..records)
                        .map(|seed| {
                            DirectRow::<Falcon512>::synthetic(seed as u64 + 101)
                                .expect("synthetic row")
                        })
                        .collect::<Vec<_>>();
                    let mut batch = FalconWitness::from_rows(&rows);
                    crate::protocol::pad_witness(&mut batch).unwrap();
                    batch
                } else {
                    let rows = (0..records)
                        .map(|seed| {
                            DirectRow::<Falcon1024>::synthetic(seed as u64 + 101)
                                .expect("synthetic row")
                        })
                        .collect::<Vec<_>>();
                    let mut batch = FalconWitness::from_rows(&rows);
                    crate::protocol::pad_witness(&mut batch).unwrap();
                    batch
                };
                let tables = SourceTables::build(&batch).expect("source tables");
                let ledger = if degree == FalconDegree::Falcon512 {
                    verify_opening::<Falcon512>(&batch, degree, records)
                } else {
                    verify_opening::<Falcon1024>(&batch, degree, records)
                };
                let manifest = CommitmentManifest::for_batch_of(degree, records).expect("manifest");
                assert_eq!(ledger.groups.len(), manifest.group_count());
                for (((opening, polynomials), descriptor), expected_role) in ledger
                    .groups
                    .iter()
                    .zip(group_tables(&batch, &tables))
                    .zip(manifest.groups())
                    .zip([
                        GroupRole::S1,
                        GroupRole::Slack,
                        GroupRole::S2Sources,
                        GroupRole::Budget,
                    ])
                {
                    assert_eq!(opening.role, expected_role, "canonical group order");
                    assert_eq!(opening.role, descriptor.role, "manifest role");
                    assert_eq!(polynomials.len(), descriptor.logical_polynomials);
                    assert_eq!(opening.values.len(), descriptor.logical_polynomials);
                    assert_eq!(
                        opening.point.len(),
                        descriptor.logical_cells.trailing_zeros() as usize,
                        "{:?} per-polynomial point arity",
                        descriptor.role
                    );
                    for (polynomial, &value) in polynomials.iter().zip(&opening.values) {
                        assert_eq!(polynomial.len(), descriptor.logical_cells);
                        assert_eq!(
                            value,
                            evaluate_mle(&lift_table(polynomial), &opening.point),
                            "{:?} must authenticate every native polynomial at records {records}",
                            descriptor.role
                        );
                    }
                }
            }
        }
    }
}

mod wire {
    use aerie_falcon::protocol::pcs::commitment_coefficient_counts_of;
    use aerie_falcon::protocol::wire::test_support::{take_work_nonce, MAGIC_PCS};
    use aerie_falcon::protocol::wire::*;
    use aerie_falcon_native::Falcon1024;
    use aerie_falcon_witness::DirectRow;
    use aerie_prover::pcs::{self, AggregateProof};
    use aerie_prover::projection::ProjectionOutputs;
    use aerie_prover::ring::{PublicRingKeys, PublicTargets};
    use aerie_prover::PublicInputs;
    use aerie_prover::{
        AggregationError, BaseField, FalconDegree, MalformedProof, OpeningField as E,
        StatementMismatch,
    };
    use aerie_prover::{Falcon512, FalconWitness};
    use akita_types::Commitment;
    use jolt_field::Ring as _;
    use std::ops::Range;
    use std::sync::OnceLock;
    type Encoded = (
        FalconWitness,
        AggregateProof,
        Vec<[u8; TRANSPORT_SALT_BYTES]>,
        Vec<u8>,
    );
    fn with_single_thread_pool<T: Send>(body: impl FnOnce() -> T + Send) -> T {
        #[cfg(feature = "parallel")]
        {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("proof thread pool");
            pool.install(body)
        }
        #[cfg(not(feature = "parallel"))]
        body()
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

    fn prove_and_encode(batch: FalconWitness, degree: FalconDegree, records: usize) -> Encoded {
        let proof_result = with_single_thread_pool(|| {
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
            degree,
            records,
            salt_mode: crate::protocol::wire::SaltMode::PerRecord,
            public_records: digest,
            h: PublicRingKeys::Expanded(&batch.h),
            t: PublicTargets::Expanded(&batch.t),
        }
    }

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
        for _ in 0..shapes.stages.basis16_sources {
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

    fn rejection_512(bytes: &[u8]) -> AggregationError {
        let decoded = match decode_pcs_proof(FalconDegree::Falcon512, bytes) {
            Ok(decoded) => decoded,
            Err(error) => return error,
        };
        let (batch, _, _, _) = honest_512();
        with_single_thread_pool(|| {
            let verifier = pcs::FalconVerifierSetup::<Falcon512>::new(8).unwrap();
            pcs::verify(
                &public(batch, 8, b"wire test records               "),
                &verifier,
                &decoded.proof,
            )
        })
        .expect_err("mutated proof rejects")
    }

    fn assert_rejects_noncanonical_frames(degree: FalconDegree, bytes: &[u8]) {
        let decode = |bytes: &[u8]| decode_pcs_proof(degree, bytes);
        assert!(decode(bytes).is_ok(), "{degree:?}: honest frame");
        let mut trailing = bytes.to_vec();
        trailing.push(0);
        assert!(
            matches!(
                decode(&trailing),
                Err(AggregationError::MalformedProof(
                    MalformedProof::TrailingBytes
                ))
            ),
            "trailing data"
        );
        assert!(
            matches!(
                decode(&bytes[..bytes.len() - 1]),
                Err(AggregationError::MalformedProof(MalformedProof::Truncated))
            ),
            "truncation"
        );
        let mut wrong_version = bytes.to_vec();
        wrong_version[HEADER_VERSION_OFFSET..HEADER_PARAMETER_SET_OFFSET]
            .copy_from_slice(&40_u16.to_le_bytes());
        assert!(
            matches!(
                decode(&wrong_version),
                Err(AggregationError::MalformedProof(
                    MalformedProof::UnknownVersion { version: 40 }
                ))
            ),
            "version"
        );
        assert!(
            matches!(
                decode_pcs_proof(other(degree), bytes),
                Err(AggregationError::StatementMismatch(
                    StatementMismatch::ParameterSet { .. }
                ))
            ),
            "cross-parameter-set decode"
        );
        let mut flipped_set = bytes.to_vec();
        flipped_set[HEADER_PARAMETER_SET_OFFSET] = other(degree).parameter_set();
        assert!(
            matches!(
                decode(&flipped_set),
                Err(AggregationError::StatementMismatch(
                    StatementMismatch::ParameterSet { .. }
                ))
            ),
            "flipped parameter set"
        );
        let error = decode_pcs_proof(other(degree), &flipped_set).expect_err("relabeled body");
        assert!(matches!(
            error,
            AggregationError::MalformedProof(MalformedProof::NonceEncoding)
        ));
        let mut wrong_records = bytes.to_vec();
        wrong_records[HEADER_RECORDS_OFFSET..HEADER_BYTES].copy_from_slice(&0_u64.to_le_bytes());
        assert!(
            matches!(
                decode(&wrong_records),
                Err(AggregationError::MalformedProof(
                    MalformedProof::InvalidRecordCount { records: 0 }
                ))
            ),
            "record count"
        );
        let mut wrong_shape = bytes.to_vec();
        wrong_shape[HEADER_RECORDS_OFFSET..HEADER_BYTES].copy_from_slice(&16_u64.to_le_bytes());
        let error = decode(&wrong_shape).expect_err("group layout");
        assert!(matches!(
            error,
            AggregationError::MalformedProof(MalformedProof::ProjectionEncoding)
        ));
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
                    186, 32, 222, 50, 55, 2, 73, 120, 76, 70, 168, 11, 51, 109, 250, 46, 223, 52,
                    111, 192, 108, 217, 48, 128, 156, 243, 67, 98, 205, 2, 253, 178
                ],
                [
                    104, 164, 99, 134, 120, 101, 223, 99, 235, 133, 244, 202, 124, 133, 180, 208,
                    65, 168, 39, 142, 199, 208, 155, 14, 206, 155, 140, 80, 211, 21, 191, 103
                ]
            ]
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
        assert!(matches!(
            decode_pcs_proof(FalconDegree::Falcon512, &noncanonical),
            Err(AggregationError::MalformedProof(
                MalformedProof::ProjectionEncoding
            ))
        ));
        for (offset, delta) in [(0, 1), (0, -1), (1, 1), (1, -1)] {
            let mut forged = section.clone();
            let slot = &mut forged[offset];
            *slot = slot.wrapping_add_signed(delta);
            let parameters = [forged[0], forged[1]];
            if parameters.iter().all(|&k| k <= 63) {
                // Preserve the output vector while forcing a nonoptimal parameter.
                forged.truncate(2);
                let mut bits = 0_usize;
                let mut push_bit = |bit: bool| {
                    if bits.is_multiple_of(8) {
                        forged.push(0);
                    }
                    if bit {
                        *forged.last_mut().expect("pushed byte") |= 1 << (bits % 8);
                    }
                    bits += 1;
                };
                for (values, scale, k) in [
                    (proof.projection_outputs.y.as_slice(), 1, parameters[0]),
                    (
                        proof.projection_outputs.z.as_slice(),
                        i128::from(crate::protocol::projection::FALCON_Q),
                        parameters[1],
                    ),
                ] {
                    for &value in values {
                        let value = crate::protocol::projection::centered(value) / scale;
                        let magnitude =
                            u64::try_from(value.unsigned_abs()).expect("centered value");
                        let code = 2 * magnitude - u64::from(value < 0);
                        for _ in 0..(code >> k) {
                            push_bit(true);
                        }
                        push_bit(false);
                        for bit in 0..k {
                            push_bit((code >> bit) & 1 != 0);
                        }
                    }
                }
            }
            let mut noncanonical = bytes[..start].to_vec();
            noncanonical.extend_from_slice(&forged);
            noncanonical.extend_from_slice(&bytes[start + section.len()..]);
            let error = decode_pcs_proof(FalconDegree::Falcon512, &noncanonical)
                .expect_err("noncanonical projection section");
            assert!(
                matches!(
                    &error,
                    AggregationError::MalformedProof(MalformedProof::ProjectionEncoding)
                ),
                "Rice parameter mutation ({offset}, {delta}) rejected as {error:?}"
            );
        }
        let mut off_lattice = proof.clone();
        off_lattice.projection_outputs.z[0] += BaseField::from_u64(1);
        assert!(matches!(
            encode_pcs_proof(FalconDegree::Falcon512, 8, salts, &off_lattice),
            Err(AggregationError::MalformedProof(
                MalformedProof::ProjectionEncoding
            ))
        ));
    }

    #[test]
    fn pcs_codec_roundtrips_verifies_and_rejects_malformed_frames() {
        let (batch, proof, salts, bytes) = honest_512();
        let decoded = decode_pcs_proof(FalconDegree::Falcon512, bytes).expect("decode");
        assert_eq!(decoded.records, 8);
        assert_eq!(decoded.salts, SaltFrame::PerRecord(salts.clone()));
        assert!(matches!(
            decode_pcs_proof(FalconDegree::Falcon1024, bytes),
            Err(AggregationError::StatementMismatch(
                StatementMismatch::ParameterSet {
                    expected: 10,
                    actual: 9
                }
            ))
        ));
        with_single_thread_pool(|| {
            let verifier_setup =
                pcs::FalconVerifierSetup::<Falcon512>::new(8).expect("verifier setup");
            pcs::verify(
                &public(batch, 8, b"wire test records               "),
                &verifier_setup,
                &decoded.proof,
            )
        })
        .expect("decoded production proof verifies");
        let reencoded = encode_pcs_proof_from_salt_frame(
            FalconDegree::Falcon512,
            8,
            &decoded.salts,
            &decoded.proof,
        )
        .expect("re-encode");
        assert_eq!(reencoded, *bytes);

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(matches!(
            decode_pcs_proof(FalconDegree::Falcon512, &trailing),
            Err(AggregationError::MalformedProof(
                MalformedProof::TrailingBytes
            ))
        ));
        assert!(matches!(
            decode_pcs_proof(FalconDegree::Falcon512, &bytes[..bytes.len() - 1]),
            Err(AggregationError::MalformedProof(MalformedProof::Truncated))
        ));
        let commitment_start = HEADER_BYTES + salts.len() * TRANSPORT_SALT_BYTES;
        assert!(
            matches!(
                decode_pcs_proof(FalconDegree::Falcon512, &bytes[..commitment_start + 100]),
                Err(AggregationError::MalformedProof(MalformedProof::Truncated))
            ),
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
            let expected = if length == 0 || length > MAX_PCS_PROOF_BYTES as u64 {
                MalformedProof::NonCanonicalLength
            } else if length < proof.pcs.len() as u64 {
                MalformedProof::TrailingBytes
            } else {
                MalformedProof::Truncated
            };
            let result = decode_pcs_proof(FalconDegree::Falcon512, &mutated);
            let rejected = matches!(
                result,
                Err(AggregationError::MalformedProof(kind)) if kind == expected
            );
            assert!(rejected, "{result:?}");
        }
        let mut flipped = bytes.clone();
        let index = flipped.len() / 2;
        flipped[index] ^= 1;
        let mutated = decode_pcs_proof(FalconDegree::Falcon512, &flipped)
            .expect("canonical midpoint mutation");
        let error = with_single_thread_pool(|| {
            let verifier_setup =
                pcs::FalconVerifierSetup::<Falcon512>::new(8).expect("verifier setup");
            pcs::verify(
                &public(batch, 8, b"wire test records               "),
                &verifier_setup,
                &mutated.proof,
            )
        })
        .expect_err("mid-frame bit flip rejects");
        assert!(matches!(
            error,
            AggregationError::ProofRejected {
                stage: crate::RejectionStage::PcsOpening
            }
        ));
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
            matches!(
                rejection_512(&replace_range(bytes, first.clone(), &[])),
                AggregationError::MalformedProof(MalformedProof::NonceEncoding)
            ),
            "omitted"
        );

        let mut extra = bytes.clone();
        extra.insert(first.start, 0);
        assert!(
            matches!(
                rejection_512(&extra),
                AggregationError::MalformedProof(MalformedProof::NonceEncoding)
            ),
            "extra"
        );

        let mut reordered = Vec::with_capacity(bytes.len());
        reordered.extend_from_slice(&bytes[..first.start]);
        reordered.extend_from_slice(&bytes[second.clone()]);
        reordered.extend_from_slice(&bytes[first.end..second.start]);
        reordered.extend_from_slice(&bytes[first.clone()]);
        reordered.extend_from_slice(&bytes[second.end..]);
        assert!(
            matches!(
                rejection_512(&reordered),
                AggregationError::ProofRejected {
                    stage: crate::RejectionStage::Grinding
                }
            ),
            "reordered"
        );

        let mut truncated = bytes[..first.start].to_vec();
        truncated.push(0x80);
        assert!(
            matches!(
                rejection_512(&truncated),
                AggregationError::MalformedProof(MalformedProof::NonceEncoding)
            ),
            "truncated"
        );

        assert!(
            matches!(
                rejection_512(&replace_range(bytes, first.clone(), &[0x80, 0x00])),
                AggregationError::MalformedProof(MalformedProof::NonceEncoding)
            ),
            "non-canonical"
        );
        assert!(
            matches!(
                rejection_512(&replace_range(bytes, first.clone(), &[0x80, 0x10])),
                AggregationError::MalformedProof(MalformedProof::NonceEncoding)
            ),
            "out of range"
        );

        let mut failed_predicate = bytes.clone();
        failed_predicate[first.start] ^= 1;
        assert!(
            matches!(
                rejection_512(&failed_predicate),
                AggregationError::ProofRejected {
                    stage: crate::RejectionStage::Grinding
                }
            ),
            "failed predicate"
        );

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(
            matches!(
                rejection_512(&trailing),
                AggregationError::MalformedProof(MalformedProof::TrailingBytes)
            ),
            "trailing bytes"
        );
    }

    #[test]
    fn production_codecs_reject_noncanonical_frames_for_both_degrees() {
        assert_rejects_noncanonical_frames(FalconDegree::Falcon512, &honest_512().3);
        assert_rejects_noncanonical_frames(FalconDegree::Falcon1024, &honest_1024().3);
        assert!(matches!(
            decode_pcs_proof(FalconDegree::Falcon1024, &honest_512().3),
            Err(AggregationError::StatementMismatch(
                StatementMismatch::ParameterSet {
                    expected: 10,
                    actual: 9
                }
            ))
        ));
    }

    #[test]
    fn falcon_1024_production_codec_roundtrips_and_verifies() {
        let (batch, _, salts, bytes) = honest_1024();
        assert_eq!(bytes[HEADER_PARAMETER_SET_OFFSET], 10);
        let decoded = decode_pcs_proof(FalconDegree::Falcon1024, bytes).expect("decode");
        assert_eq!(decoded.degree, FalconDegree::Falcon1024);
        assert_eq!(decoded.records, 4);
        assert_eq!(decoded.salts, SaltFrame::PerRecord(salts.clone()));
        with_single_thread_pool(|| {
            let verifier_setup =
                pcs::FalconVerifierSetup::<Falcon1024>::new(4).expect("verifier setup");
            pcs::verify(
                &public(batch, 4, b"wire test records               "),
                &verifier_setup,
                &decoded.proof,
            )
        })
        .expect("decoded Falcon-1024 production proof verifies");
        let reencoded = encode_pcs_proof_from_salt_frame(
            FalconDegree::Falcon1024,
            4,
            &decoded.salts,
            &decoded.proof,
        )
        .expect("re-encode");
        assert_eq!(reencoded, *bytes);
    }
}
