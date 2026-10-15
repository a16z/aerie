//! Producer commitment execution, setup, and instrumentation.

use super::manifest::GROUP_COUNT;
use crate::substrate::{BaseField, OpeningField};
use crate::AggregationError as Error;
use akita_pcs::CommitmentHandle;

mod commit;
mod group;
mod opening;
mod prover;
mod setup;

use aerie_falcon::protocol::pcs::serialized_commitment;
use aerie_falcon::protocol::pcs::verifier_group_profiles_of;
#[cfg(any(test, feature = "internals"))]
pub use aerie_falcon::protocol::pcs::{verify, VerifiedOpening};
pub use aerie_falcon::protocol::pcs::{AggregateProof, FalconVerifierSetup};
pub use commit::{precommit_s2_groups, S2Precommit};
pub use prover::prove_prepared_with_precommit;
#[cfg(any(test, feature = "internals"))]
pub use prover::{prove, prove_prepared};
#[cfg(feature = "internals")]
pub use prover::{prove_prepared_with_witness_tamper, prove_with_group_tamper};
pub use setup::FalconAggregateSetup;
pub(crate) use setup::ProverCore;

#[cfg(test)]
use super::ring::{PublicRingKeys, PublicTargets};
#[cfg(test)]
use super::{public_inputs::PublicInputs, wire::SaltMode, FalconDegree, FalconWitness};
#[cfg(test)]
use aerie_falcon::protocol::pcs::verify_opening;
#[cfg(test)]
use prover::{prove_prepared_impl, ProverHooks};

/// Prover-side stage instrumentation (Section 12). Nanosecond fields
/// follow the `aerie_falcon_statement::staged!` convention; timings measured in managed
/// sessions are not comparable to host numbers.
#[cfg_attr(
    not(feature = "internals"),
    allow(dead_code, reason = "read only by the measured entry points")
)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PcsStageMetrics {
    pub records: usize,
    pub padded_records: usize,
    pub encodability_ns: u128,
    pub group_materialization_ns: u128,
    pub setup_ns: u128,
    pub commit_ns: u128,
    pub projection_ns: u128,
    pub stage1_ns: u128,
    pub stage2_ns: u128,
    pub stage3_ns: u128,
    pub pcs_prove_ns: u128,
    /// Public-target evaluation in the configured Rayon pool.
    pub target_evaluation_ns: u128,
    /// Protocol-block verification excluding public-target evaluation.
    pub stage_verify_ns: u128,
    pub pcs_verify_ns: u128,
    /// The same verifier replayed inside a dedicated one-worker Rayon pool.
    pub target_evaluation_single_thread_ns: u128,
    pub stage_verify_single_thread_ns: u128,
    pub pcs_verify_single_thread_ns: u128,
    /// Per-group commit wall time in canonical manifest order (Section
    /// 26.4). Groups commit inside a parallel fan-out, so entries overlap
    /// and do not sum to `commit_ns`.
    pub commit_group_ns: [u128; GROUP_COUNT],
    pub commitment_bytes: usize,
    pub pcs_proof_bytes: usize,
}

/// A retained commitment owned by the producer backend.
type GroupHandle = CommitmentHandle<BaseField, OpeningField>;

/// Prove and verify in one call, merging the stage timings.
#[cfg(test)]
#[tracing::instrument(skip_all, name = "v1.roundtrip")]
pub fn pcs_roundtrip(
    batch: &FalconWitness,
    public_records: &[u8],
) -> Result<PcsStageMetrics, Error> {
    let mut padded_batch = batch.clone();
    super::pad_witness(&mut padded_batch)?;
    let batch = &padded_batch;
    let setup_start = std::time::Instant::now();
    let degree = FalconDegree::from_degree(batch.degree())?;
    let (setup, verifier) = tracing::info_span!("v1.setup").in_scope(|| {
        let setup = ProverCore::new(degree, batch.live_rows)?;
        let verifier = setup.verifier()?;
        Ok::<_, Error>((setup, verifier))
    })?;
    let setup_ns = setup_start.elapsed().as_nanos();
    let (proof, mut metrics) = prove_prepared_impl(
        batch,
        public_records,
        SaltMode::PerRecord,
        &setup,
        #[cfg(any(test, feature = "internals"))]
        ProverHooks::default(),
        None,
    )?;
    metrics.setup_ns = setup_ns;
    let public = PublicInputs {
        degree,
        records: batch.live_rows,
        salt_mode: crate::protocol::wire::SaltMode::PerRecord,
        public_records,
        h: PublicRingKeys::Expanded(&batch.h),
        t: PublicTargets::Expanded(&batch.t),
    };
    let verified = tracing::info_span!("v1.verify").in_scope(|| {
        let verified = tracing::info_span!("v1.verify.configured_pool")
            .in_scope(|| verify_opening(&public, &verifier, &proof))?;
        Ok::<_, Error>(verified)
    })?;
    metrics.target_evaluation_ns = verified.target_evaluation_ns;
    metrics.stage_verify_ns = verified.stage_verify_ns;
    metrics.pcs_verify_ns = verified.pcs_verify_ns;

    #[cfg(feature = "parallel")]
    let single_thread_pool =
        tracing::info_span!("v1.verify.single_thread.setup").in_scope(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .thread_name(|_| "aerie-v1-verifier-single".to_owned())
                .build()
                .map_err(Error::backend)
        })?;
    #[cfg(feature = "parallel")]
    let verified_single = tracing::info_span!("v1.verify.single_thread")
        .in_scope(|| single_thread_pool.install(|| verify_opening(&public, &verifier, &proof)))?;
    #[cfg(not(feature = "parallel"))]
    let verified_single = tracing::info_span!("v1.verify.single_thread")
        .in_scope(|| verify_opening(&public, &verifier, &proof))?;
    if verified.ledger != verified_single.ledger {
        return Err(Error::internal(
            "verifier thread configurations produced different ledgers",
        ));
    }
    metrics.target_evaluation_single_thread_ns = verified_single.target_evaluation_ns;
    metrics.stage_verify_single_thread_ns = verified_single.stage_verify_ns;
    metrics.pcs_verify_single_thread_ns = verified_single.pcs_verify_ns;
    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::group::{group_polynomial, SignedGroup};
    #[cfg(feature = "parallel")]
    use super::prover::STAGE3_PREPARE_OVERLAP_MIN_ROWS;
    use super::*;
    use crate::protocol::kernels::{DigitSource, SignSource};
    use crate::protocol::manifest::GroupRole;
    use crate::substrate::{Width14Config, Width4Config};
    use crate::{
        AggregationParams, MalformedProof, RejectionStage, StatementMismatch, UnsupportedBatch,
    };
    use aerie_falcon_native::{Falcon1024, Falcon512};
    use aerie_falcon_witness::DirectRow;
    use akita_params::{PolynomialGroupLayout, ScheduleLookupKey};

    #[test]
    fn supported_record_counts_follow_the_generated_catalog() {
        let small_counts: Vec<_> = (0..=16)
            .filter(|&records| FalconAggregateSetup::<Falcon512>::supports(records))
            .collect();
        assert_eq!(small_counts, (1..=16).collect::<Vec<_>>());
        for exponent in 3..=16 {
            let records = 1 << exponent;
            assert!(
                FalconAggregateSetup::<Falcon512>::supports(records),
                "{records}"
            );
        }
        for records in [(1 << 16) + 1, usize::MAX] {
            assert!(
                !FalconAggregateSetup::<Falcon512>::supports(records),
                "{records}"
            );
        }
    }

    fn fixture(records: usize) -> FalconWitness {
        let rows = (0..records)
            .map(|seed| {
                DirectRow::<aerie_falcon_native::Falcon512>::synthetic(seed as u64 + 151)
                    .expect("synthetic row")
            })
            .collect::<Vec<_>>();
        FalconWitness::from_rows(&rows)
    }

    fn fixture_1024(records: usize) -> FalconWitness {
        let rows = (0..records)
            .map(|seed| DirectRow::<Falcon1024>::synthetic(seed as u64 + 0x0102_4151).expect("row"))
            .collect::<Vec<_>>();
        FalconWitness::from_rows(&rows)
    }

    #[test]
    fn explicit_degree_rejects_mismatched_representations() {
        use aerie_falcon_native::source::CanonicalPublicKeyBytes;
        use aerie_falcon_native::FalconParams;

        fn check<P: AggregationParams>(
            wrong_keys: PublicRingKeys<'_, i16>,
            wrong_degree: FalconDegree,
        ) {
            let row = DirectRow::<P>::synthetic(151).expect("synthetic row");
            let mut batch = FalconWitness::from_rows(&[row]);
            super::super::pad_witness(&mut batch).expect("minimum padding");
            assert_eq!(batch.h.len(), super::super::MIN_PADDED_COEFFICIENTS);
            assert_eq!(batch.t.len(), super::super::MIN_PADDED_COEFFICIENTS);
            let setup = FalconAggregateSetup::<P>::new(1).expect("setup");
            let (proof, _) = prove_prepared(&batch, &[0x51; 32], &setup).expect("proof");
            let public = PublicInputs {
                degree: P::DEGREE_ID,
                records: 1,
                salt_mode: SaltMode::PerRecord,
                public_records: &[0x51; 32],
                h: PublicRingKeys::Expanded(&batch.h),
                t: PublicTargets::Expanded(&batch.t),
            };
            verify(&public, setup.verifier(), &proof)
                .expect("explicit degree accepts minimum-padded tables");
            for h in [
                wrong_keys,
                PublicRingKeys::Expanded(&batch.h[..batch.h.len() - 1]),
            ] {
                assert!(matches!(
                    verify(&PublicInputs { h, ..public }, setup.verifier(), &proof),
                    Err(Error::StatementMismatch(
                        StatementMismatch::PublicInputShape
                    ))
                ));
            }
            assert!(matches!(
                verify(
                    &PublicInputs {
                        t: PublicTargets::Expanded(&batch.t[..batch.t.len() - 1]),
                        ..public
                    },
                    setup.verifier(),
                    &proof,
                ),
                Err(Error::StatementMismatch(
                    StatementMismatch::PublicInputShape
                ))
            ));
            assert!(matches!(
                verify(
                    &PublicInputs { degree: wrong_degree, ..public },
                    setup.verifier(),
                    &proof,
                ),
                Err(Error::StatementMismatch(StatementMismatch::ParameterSet {
                    expected,
                    actual,
                })) if expected == P::PARAMETER_SET && actual == wrong_degree.parameter_set()
            ));
        }

        let mut encoded_512 = vec![0; Falcon512::PUBLIC_KEY_BYTES];
        encoded_512[0] = Falcon512::PUBLIC_KEY_HEADER;
        let keys_512 = [CanonicalPublicKeyBytes::<Falcon512>::parse(&encoded_512).unwrap()];
        let mut encoded_1024 = vec![0; Falcon1024::PUBLIC_KEY_BYTES];
        encoded_1024[0] = Falcon1024::PUBLIC_KEY_HEADER;
        let keys_1024 = [CanonicalPublicKeyBytes::<Falcon1024>::parse(&encoded_1024).unwrap()];
        check::<Falcon512>(
            PublicRingKeys::Canonical1024(&keys_1024),
            FalconDegree::Falcon1024,
        );
        check::<Falcon1024>(
            PublicRingKeys::Canonical512(&keys_512),
            FalconDegree::Falcon512,
        );
    }

    fn assert_group_profiles(degree: FalconDegree, records: usize, vars: [usize; 4]) {
        let (precommits, final_group, _) =
            verifier_group_profiles_of(degree, records).expect("cataloged group shapes");
        let expected_precommits = [
            PolynomialGroupLayout::new(vars[0], 1),
            PolynomialGroupLayout::new(vars[1], 1),
            PolynomialGroupLayout::new(vars[2], 4),
        ];
        assert_eq!(precommits.len(), expected_precommits.len());
        for (profile, expected) in precommits.iter().zip(expected_precommits) {
            assert_eq!(profile.group, expected, "precommitted native group shape");
        }
        assert_eq!(
            final_group.group,
            PolynomialGroupLayout::new(vars[3], 3),
            "final native budget group shape"
        );
    }

    /// Native group layouts are resolved at every padded Falcon-1024 batch.
    #[test]
    fn falcon_1024_catalog_covers_the_production_batches() {
        let small_counts: Vec<_> = (0..=16)
            .filter(|&records| FalconAggregateSetup::<Falcon1024>::supports(records))
            .collect();
        assert_eq!(small_counts, (1..=16).collect::<Vec<_>>());
        for records in [32_usize, 64, 128, 256, 512, 1024, 2048] {
            assert!(
                FalconAggregateSetup::<Falcon1024>::supports(records),
                "{records}"
            );
        }
        let cases = [
            (16, [14, 13, 14, 13]),
            (1 << 12, [22, 14, 22, 13]),
            (1 << 13, [23, 15, 23, 13]),
            (1 << 14, [24, 16, 24, 14]),
            (1 << 15, [25, 17, 25, 15]),
            (1 << 16, [26, 18, 26, 16]),
        ];
        for (records, expected_vars) in cases {
            assert_group_profiles(FalconDegree::Falcon1024, records, expected_vars);
        }
        // The Falcon-512 rows do not answer for Falcon-1024 shapes.
        let (precommits_512, _, _) =
            verifier_group_profiles_of(FalconDegree::Falcon512, 1 << 14).expect("Falcon-512 row");
        assert_eq!(precommits_512.len(), 3);
    }

    /// End-to-end Falcon-1024: four real commitments, the blocks and one
    /// multi-group opening against the generated
    /// minimum-batch budget-final row (the Falcon-1024 shapes at
    /// `N_sig* = 4` equal the Falcon-512 shapes at `N_sig* = 8`).
    #[test]
    fn falcon_1024_pcs_roundtrip_at_the_minimum_batch() {
        let metrics = {
            let batch = fixture_1024(3);
            pcs_roundtrip(&batch, &[0x10; 32])
        }
        .expect("Falcon-1024 PCS roundtrip");
        assert_eq!(metrics.records, 3);
        assert_eq!(metrics.padded_records, 4);
        assert!(
            metrics.commit_group_ns[3] > 0,
            "the final group was committed"
        );
        assert!(metrics.commitment_bytes > 0);
        assert!(metrics.pcs_proof_bytes > 0);
    }

    /// A Falcon-1024 proof must carry four commitments in manifest order: a
    /// swapped or missing final-group commitment rejects, and a Falcon-512
    /// setup refuses a Falcon-1024 witness.
    #[test]
    fn falcon_1024_pcs_verify_rejects_reordered_commitments() {
        {
            let batch = fixture_1024(8);
            let setup = FalconAggregateSetup::<Falcon1024>::new(8).expect("setup");
            assert_eq!(setup.records(), 8);
            let (proof, _) = prove_prepared(&batch, b"falcon-1024 records             ", &setup)
                .expect("Falcon-1024 prover");
            assert_eq!(proof.commitments.len(), 4);
            let public = PublicInputs {
                degree: FalconDegree::from_degree(batch.degree()).expect("parameter set"),
                records: 8,
                salt_mode: crate::protocol::wire::SaltMode::PerRecord,
                public_records: b"falcon-1024 records             ",
                h: PublicRingKeys::Expanded(&batch.h),
                t: PublicTargets::Expanded(&batch.t),
            };
            assert_eq!(public.degree, FalconDegree::Falcon1024);
            let verified = verify(&public, setup.verifier(), &proof).expect("verifies");
            assert_eq!(verified.ledger.groups.len(), 4);
            assert_eq!(verified.ledger.groups[3].role, GroupRole::Budget);

            let mut swapped = proof.clone();
            swapped.commitments.swap(2, 3);
            let swapped_result = verify(&public, setup.verifier(), &swapped);
            assert!(
                matches!(
                    swapped_result,
                    Err(Error::ProofRejected {
                        stage: RejectionStage::Grinding
                    })
                ),
                "swapped Falcon-1024 commitments: {swapped_result:?}"
            );
            let mut truncated = proof.clone();
            truncated.commitments.pop();
            assert!(matches!(
                verify(&public, setup.verifier(), &truncated),
                Err(Error::MalformedProof(MalformedProof::CommitmentShape))
            ));

            let wrong_setup = FalconAggregateSetup::<Falcon512>::new(8).expect("Falcon-512 setup");
            assert!(matches!(
                verify(&public, wrong_setup.verifier(), &proof),
                Err(Error::StatementMismatch(StatementMismatch::ParameterSet {
                    expected: 9,
                    actual: 10
                }))
            ));
            assert!(matches!(
                prove_prepared(&batch, b"falcon-1024 records             ", &wrong_setup),
                Err(Error::StatementMismatch(StatementMismatch::ParameterSet {
                    expected: 9,
                    actual: 10
                }))
            ));
        };
    }

    fn public(batch: &FalconWitness, records: usize) -> PublicInputs<'_, i16> {
        PublicInputs {
            degree: FalconDegree::from_degree(batch.degree()).expect("parameter set"),
            records,
            salt_mode: crate::protocol::wire::SaltMode::PerRecord,
            public_records: b"pcs roundtrip records           ",
            h: PublicRingKeys::Expanded(&batch.h),
            t: PublicTargets::Expanded(&batch.t),
        }
    }

    #[test]
    fn malformed_stage_shapes_reject_before_ring_evaluation() {
        use crate::protocol::{stage1, stage2, stage3, transcript::TestTranscript};
        use jolt_field::Zero as _;
        use std::panic::{catch_unwind, AssertUnwindSafe};

        let batch = fixture(8);
        let setup = FalconAggregateSetup::<Falcon512>::new(8).expect("setup");
        let public = public(&batch, 8);
        let (proof, _) = prove_prepared(&batch, public.public_records, &setup).expect("proof");
        for stage in 0..3 {
            let mut truncated = proof.clone();
            match stage {
                0 => {
                    truncated.stage1.rounds.pop();
                }
                1 => {
                    truncated.stage2.rounds.pop();
                }
                _ => {
                    truncated.stage3.rounds.pop();
                }
            }
            let result = catch_unwind(AssertUnwindSafe(|| {
                verify(&public, setup.verifier(), &truncated)
            }))
            .expect("truncated stage rounds must not panic");
            assert!(
                matches!(
                    result,
                    Err(Error::MalformedProof(MalformedProof::StageShape))
                ),
                "stage {stage}: {result:?}"
            );
        }
        for malformed in [
            PublicInputs {
                h: PublicRingKeys::Expanded(&batch.h[..batch.h.len() - 1]),
                ..public.clone()
            },
            PublicInputs {
                t: PublicTargets::Expanded(&batch.t[..batch.t.len() - 1]),
                ..public.clone()
            },
        ] {
            let result = catch_unwind(AssertUnwindSafe(|| {
                verify(&malformed, setup.verifier(), &proof)
            }))
            .expect("malformed internal public shape must not panic");
            assert!(
                matches!(
                    result,
                    Err(Error::StatementMismatch(
                        StatementMismatch::PublicInputShape
                    ))
                ),
                "public shape: {result:?}"
            );
        }

        // Stage points are derived outputs rather than wire fields. Exercise
        // their validation at the stage boundary that consumes them.
        let degree = public.degree;
        let records = batch.padded_rows;
        let rounds = degree.logn() + records.trailing_zeros() as usize;
        let first = stage1::Stage1Output {
            point: vec![OpeningField::zero(); rounds],
            basis16_claims: vec![[OpeningField::zero(); 2]; 2],
            projection_point: vec![OpeningField::zero(); rounds],
            slack_projection_point: vec![OpeningField::zero(); rounds - degree.logn() + 2],
            projection_claims: [OpeningField::zero(); 3],
        };
        let second = stage2::Stage2Output {
            point: vec![OpeningField::zero(); rounds],
            folded_digits: [OpeningField::zero(); 6],
        };
        for point in 0..4 {
            let mut first = first.clone();
            let mut second = second.clone();
            match point {
                0 => {
                    first.point.pop();
                }
                1 => {
                    first.projection_point.pop();
                }
                2 => {
                    first.slack_projection_point.pop();
                }
                _ => {
                    second.point.pop();
                }
            }
            let mut target_evaluation_ns = 0;
            let result = catch_unwind(AssertUnwindSafe(|| {
                stage3::verify(
                    records,
                    &public,
                    &first,
                    &second,
                    &proof.stage3,
                    &mut TestTranscript::new(b"malformed PCS stage point"),
                    &mut target_evaluation_ns,
                )
            }))
            .expect("truncated stage points must not panic");
            assert!(
                matches!(
                    result,
                    Err(Error::MalformedProof(MalformedProof::StageShape))
                ),
                "point {point}: {result:?}"
            );
            assert_eq!(
                target_evaluation_ns, 0,
                "ring targets must not be evaluated"
            );
        }
        let mut truncated = proof.stage3.clone();
        truncated.rounds.pop();
        let mut target_evaluation_ns = 0;
        let result = catch_unwind(AssertUnwindSafe(|| {
            stage3::verify(
                records,
                &public,
                &first,
                &second,
                &truncated,
                &mut TestTranscript::new(b"malformed PCS stage rounds"),
                &mut target_evaluation_ns,
            )
        }))
        .expect("truncated stage-three rounds must not panic");
        assert!(matches!(
            result,
            Err(Error::MalformedProof(MalformedProof::StageShape))
        ));
        assert_eq!(
            target_evaluation_ns, 0,
            "ring targets must not be evaluated"
        );
    }

    #[test]
    fn compact_group_materialization_matches_the_semantic_i64_table() {
        let compact = [i16::MIN, -8192, -1, 0, 1, 8191, 16_384, i16::MAX];
        let wide = compact.map(i64::from);
        assert!(
            group_polynomial(&compact).expect("compact polynomial")
                == group_polynomial(&wide).expect("wide polynomial")
        );
    }

    #[test]
    fn native_groups_keep_polynomials_independent_and_split_tamper_deltas() {
        let first = [-8_i8, -1, 0, 7];
        let second = [7_i8, 0, -1, -8];
        let third = [-4_i8, 1, 2, 3];
        let fourth = [3_i8, 2, 1, -4];
        let mut group = SignedGroup::S2Sources {
            digits: [&first, &second, &third].map(|cells| DigitSource::Direct(cells)),
            signs: SignSource::Direct(&fourth),
        };
        assert_eq!(group.logical_cells(), first.len());
        assert_eq!(group.polynomial_count(), 4);
        assert_eq!(
            group.polynomials().expect("four source polynomials"),
            vec![
                group_polynomial(&first).expect("a0"),
                group_polynomial(&second).expect("a1"),
                group_polynomial(&third).expect("a2"),
                group_polynomial(&fourth).expect("sigma"),
            ]
        );
        let deltas = group
            .tamper(GroupRole::S2Sources, &|role, cells| {
                assert_eq!(role, GroupRole::S2Sources);
                cells[0] -= 1;
                *cells.last_mut().expect("nonempty group") += 2;
            })
            .expect("fixed-shape adversarial edit");
        assert_eq!(deltas, vec![vec![(0, -1)], vec![], vec![], vec![(3, 2)]]);
        let expected = vec![
            vec![-9, -1, 0, 7],
            vec![7, 0, -1, -8],
            vec![-4, 1, 2, 3],
            vec![3, 2, 1, -2],
        ];
        match &group {
            #[cfg(any(test, feature = "internals"))]
            SignedGroup::OwnedI64(polys) => assert_eq!(polys, &expected),
            _ => panic!("tampering must retain polynomial-major group shape"),
        }

        let budget = SignedGroup::MultiI8Three([&first, &second, &third]);
        assert_eq!(budget.polynomial_count(), 3);
        assert_eq!(budget.logical_cells(), first.len());
        assert_eq!(
            budget
                .polynomials()
                .expect("three native budget polynomials"),
            [&first, &second, &third]
                .into_iter()
                .map(|cells| group_polynomial(cells).expect("semantic narrow polynomial"))
                .collect::<Vec<_>>()
        );
    }

    /// Every cataloged production power-of-two batch has the four native
    /// group layouts in the runtime profile derivation path.
    #[test]
    fn catalog_covers_production_power_of_two_batches() {
        let cases = [
            (1 << 12, [21, 14, 21, 13]),
            (1 << 13, [22, 15, 22, 13]),
            (1 << 14, [23, 16, 23, 14]),
            (1 << 15, [24, 17, 24, 15]),
            (1 << 16, [25, 18, 25, 16]),
        ];
        for (records, expected_vars) in cases {
            let expected_layouts = [
                PolynomialGroupLayout::new(expected_vars[0], 1),
                PolynomialGroupLayout::new(expected_vars[1], 1),
                PolynomialGroupLayout::new(expected_vars[2], 4),
            ];
            let precommits = expected_layouts
                .iter()
                .enumerate()
                .map(|(index, &layout)| match index {
                    0 | 1 => Width14Config::commitment_profile(layout),
                    _ => Width4Config::commitment_profile(layout),
                })
                .collect::<Result<Vec<_>, _>>()
                .expect("cataloged precommit profiles");
            Width4Config::catalog()
                .expect("trusted test catalog")
                .resolve_key(&ScheduleLookupKey {
                    final_group: PolynomialGroupLayout::new(expected_vars[3], 3),
                    precommitteds: precommits,
                })
                .expect("cataloged requested batch");
            assert_group_profiles(FalconDegree::Falcon512, records, expected_vars);
        }
    }

    /// End-to-end: four real commitments, the blocks, and one multi-group
    /// opening against the generated minimum-batch budget-final row. Requires the
    /// regenerated schedule catalogs.
    #[test]
    fn pcs_roundtrip_at_the_minimum_batch() {
        let metrics = {
            let batch = fixture(8);
            pcs_roundtrip(&batch, b"pcs roundtrip records           ")
        }
        .expect("PCS roundtrip");
        assert_eq!(metrics.records, 8);
        assert_eq!(metrics.padded_records, 8);
        assert!(metrics.commitment_bytes > 0);
        assert!(metrics.pcs_proof_bytes > 0);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn stage3_prepare_overlap_preserves_proof_bytes() {
        {
            // 129 is the smallest live batch whose canonical padded domain
            // reaches the 256-row overlap gate.
            let mut batch = fixture(129);
            crate::protocol::pad_witness(&mut batch).expect("pad overlap fixture");
            assert_eq!(batch.padded_rows, STAGE3_PREPARE_OVERLAP_MIN_ROWS);
            let setup =
                FalconAggregateSetup::<Falcon512>::new(batch.live_rows).expect("overlap setup");
            let prove_in_pool = |threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("overlap proof pool")
                    .install(|| {
                        prove_prepared(&batch, b"stage3 prepare overlap proof    ", &setup)
                            .expect("overlap proof")
                            .0
                    })
            };
            let single = prove_in_pool(1);
            let parallel = prove_in_pool(4);
            let salts = vec![[0x5a; crate::protocol::wire::TRANSPORT_SALT_BYTES]; batch.live_rows];
            let encode = |proof| {
                crate::protocol::wire::encode_pcs_proof(
                    FalconDegree::Falcon512,
                    batch.live_rows,
                    &salts,
                    proof,
                )
                .expect("encode overlap proof")
            };
            assert_eq!(encode(&single), encode(&parallel));
        };
    }

    #[test]
    fn native_transcript_rejects_proofs_bound_to_another_instance_or_session() {
        {
            let run = || {
                for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
                    let batch = if degree == FalconDegree::Falcon512 {
                        fixture(8)
                    } else {
                        fixture_1024(4)
                    };
                    let mut setup = ProverCore::new(degree, batch.live_rows).unwrap();
                    let verifier = setup.verifier().unwrap();
                    let public = PublicInputs {
                        degree,
                        records: batch.live_rows,
                        salt_mode: SaltMode::PerRecord,
                        public_records: &[0x91; 32],
                        h: PublicRingKeys::Expanded(&batch.h),
                        t: PublicTargets::Expanded(&batch.t),
                    };
                    for offset in [
                        None,
                        Some(2),
                        Some(27),
                        Some(28),
                        Some(60),
                        Some(302),
                        Some(334),
                        Some(366),
                    ] {
                        setup.instance_mutation = offset;
                        let (proof, _) = prove_prepared_impl(
                            &batch,
                            public.public_records,
                            SaltMode::PerRecord,
                            &setup,
                            #[cfg(any(test, feature = "internals"))]
                            ProverHooks::default(),
                            None,
                        )
                        .unwrap();
                        let result = verify_opening(&public, &verifier, &proof);
                        match offset {
                            None => assert!(result.is_ok()),
                            Some(_) => assert!(
                                matches!(result, Err(Error::ProofRejected { .. })),
                                "instance field {offset:?}: {result:?}"
                            ),
                        }
                    }
                    setup.instance_mutation = None;
                    setup.session_mutation = true;
                    let (proof, _) = prove_prepared_impl(
                        &batch,
                        public.public_records,
                        SaltMode::PerRecord,
                        &setup,
                        #[cfg(any(test, feature = "internals"))]
                        ProverHooks::default(),
                        None,
                    )
                    .unwrap();
                    assert!(
                        matches!(
                            verify_opening(&public, &verifier, &proof),
                            Err(Error::ProofRejected {
                                stage: RejectionStage::PcsOpening
                            })
                        ),
                        "same PIOP with changed PCS session must reach and fail PCS verification"
                    );
                }
            };
            #[cfg(feature = "parallel")]
            {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap();
                pool.install(run);
            }
            #[cfg(not(feature = "parallel"))]
            run();
        };
    }

    /// The split halves are wire-style: the verifier reconstructs every
    /// profile from the catalogs and rejects swapped or mismatched
    /// commitment payloads and wrong statements.
    #[test]
    fn split_verify_rejects_tampered_statements_and_payloads() {
        let outcome = (|| -> Result<(), Error> {
            let batch = fixture(8);
            let (proof, _) = prove(&batch, b"pcs roundtrip records           ")?;
            let verifier_setup = FalconVerifierSetup::<Falcon512>::new(8)?;
            verify(&public(&batch, 8), &verifier_setup, &proof)?;

            // Swapped commitment payloads change the bound transcript.
            let mut swapped = proof.clone();
            swapped.commitments.swap(0, 1);
            let swapped_result = verify(&public(&batch, 8), &verifier_setup, &swapped);
            assert!(
                matches!(
                    swapped_result,
                    Err(Error::ProofRejected {
                        stage: RejectionStage::Grinding
                    })
                ),
                "swapped Falcon-512 commitments: {swapped_result:?}"
            );

            // A wrong public live count is refused by the eight-record
            // setup, and under a seven-record setup (same padded domain)
            // it rebinds the manifest.
            let mut wrong_live = public(&batch, 8);
            wrong_live.records = 7;
            assert!(matches!(
                verify(&wrong_live, &verifier_setup, &proof),
                Err(Error::UnsupportedBatch(UnsupportedBatch::SetupSize {
                    expected: 8,
                    actual: 7
                }))
            ));
            let seven_setup = FalconVerifierSetup::<Falcon512>::new(7)?;
            let wrong_live_result = verify(&wrong_live, &seven_setup, &proof);
            assert!(
                matches!(
                    wrong_live_result,
                    Err(Error::StatementMismatch(
                        crate::StatementMismatch::NonZeroPadding
                    ))
                ),
                "rebound live record count: {wrong_live_result:?}"
            );

            // Tampered public targets break the ring scalar.
            let mut tampered_targets = batch.t.clone();
            tampered_targets[3] += 1;
            let mut wrong_targets = public(&batch, 8);
            wrong_targets.t = PublicTargets::Expanded(&tampered_targets);
            assert!(matches!(
                verify(&wrong_targets, &verifier_setup, &proof),
                Err(Error::ProofRejected {
                    stage: RejectionStage::Stage3
                })
            ));
            Ok(())
        })();
        outcome.expect("split verification checks");
    }

    #[test]
    fn prepared_setup_rejects_a_different_live_record_count() {
        let batch = fixture(5);
        let setup = FalconAggregateSetup::<Falcon512>::new(8).expect("setup for eight records");
        let result = prove_prepared(&batch, b"setup mismatch                  ", &setup);
        assert!(matches!(
            result,
            Err(Error::UnsupportedBatch(UnsupportedBatch::SetupSize {
                expected: 8,
                actual: 5
            }))
        ));
    }

    /// The production batch against the `falcon-512-batch-16384-budget-final`
    /// catalog row. Ignored by default: minutes of unoptimized serial
    /// sumcheck work and a multi-gigabyte peak. Timings measured in managed
    /// sessions are not comparable to host numbers.
    #[test]
    #[ignore = "target-batch roundtrip; slow and memory heavy"]
    fn pcs_roundtrip_at_the_target_batch() {
        let metrics = {
            let batch = fixture(1 << 14);
            pcs_roundtrip(&batch, b"pcs target roundtrip records    ")
        }
        .expect("PCS roundtrip at the target batch");
        assert_eq!(metrics.records, 1 << 14);
        println!(
            "target batch: commitment bytes {}, pcs proof bytes {}",
            metrics.commitment_bytes, metrics.pcs_proof_bytes
        );
    }
}
