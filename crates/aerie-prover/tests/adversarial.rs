//! Adversarial tests for the three-stage Falcon protocol.
//!
//! Mutations cover committed sources, projection outputs, stage claims and rounds,
//! group descriptors, transport framing, and Akita openings at both degrees.

use aerie_falcon_native::FalconParams;

use jolt_field::{One, Ring, Zero};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use aerie_falcon_witness::DirectRow;
use aerie_prover::manifest::{CommitmentManifest, GroupRole, GROUP_COUNT};
use aerie_prover::projection::{
    self, centered, norm_cap, ProjectionOutputs, FALCON_Q, FIELD_MODULUS, NORM_TERMINAL_ROWS,
};
use aerie_prover::source_tables::SourceTables;
use aerie_prover::transcript::{ProtocolTranscript, TranscriptOrderError};
use aerie_prover::{
    pcs, wire, AggregationError, AggregationParams, BaseField, Falcon512, FalconWitness,
    MalformedProof, OpeningField, PublicInputs, RejectionStage, UnsupportedBatch,
};
use akita_serialization::{AkitaSerialize, Compress};

const ROWS: usize = 8;
const PUBLIC_RECORDS: &[u8] = b"adversarial records             ";

// Wire header layout: 16-byte magic, u16 version, u8 parameter set, u64
// record count.
use aerie_prover::FalconDegree;

/// The four groups in canonical wire order.
const ROLES: [GroupRole; GROUP_COUNT] = [
    GroupRole::S1,
    GroupRole::Slack,
    GroupRole::S2Sources,
    GroupRole::Budget,
];

fn fixture(records: usize) -> FalconWitness {
    let rows = (0..records)
        .map(|seed| DirectRow::<Falcon512>::synthetic(seed as u64 + 263).expect("synthetic row"))
        .collect::<Vec<_>>();
    let mut batch = FalconWitness::from_rows(&rows);
    aerie_prover::pad_witness(&mut batch).expect("canonical padding");
    batch
}

fn public(batch: &FalconWitness) -> PublicInputs<'_, i16> {
    PublicInputs {
        degree: FalconDegree::from_degree(batch.degree()).expect("parameter set"),
        records: batch.live_rows,
        salt_mode: aerie_prover::wire::SaltMode::PerRecord,
        public_records: PUBLIC_RECORDS,
        h: aerie_prover::ring::PublicRingKeys::Expanded(&batch.h),
        t: aerie_prover::ring::PublicTargets::Expanded(&batch.t),
    }
}

/// Cache the minimum shape and build an exact-count setup for other fixtures.
/// Every attack uses the same setup shape as its honest control.
enum TestSetup<P: AggregationParams> {
    Cached(MutexGuard<'static, pcs::FalconAggregateSetup<P>>),
    Owned(Box<pcs::FalconAggregateSetup<P>>),
}

impl<P: AggregationParams> std::ops::Deref for TestSetup<P> {
    type Target = pcs::FalconAggregateSetup<P>;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Cached(setup) => setup,
            Self::Owned(setup) => setup,
        }
    }
}

fn setup_512(records: usize) -> TestSetup<Falcon512> {
    if records != ROWS {
        return TestSetup::Owned(Box::new(
            pcs::FalconAggregateSetup::<Falcon512>::new(records).expect("fixture setup"),
        ));
    }
    static SETUP: OnceLock<Mutex<pcs::FalconAggregateSetup<Falcon512>>> = OnceLock::new();
    TestSetup::Cached(
        SETUP
            .get_or_init(|| Mutex::new(pcs::FalconAggregateSetup::<Falcon512>::new(ROWS).unwrap()))
            .lock()
            .unwrap(),
    )
}

fn setup_1024(records: usize) -> TestSetup<aerie_prover::Falcon1024> {
    if records != 4 {
        return TestSetup::Owned(Box::new(
            pcs::FalconAggregateSetup::<aerie_prover::Falcon1024>::new(records)
                .expect("fixture setup"),
        ));
    }
    static SETUP: OnceLock<Mutex<pcs::FalconAggregateSetup<aerie_prover::Falcon1024>>> =
        OnceLock::new();
    TestSetup::Cached(
        SETUP
            .get_or_init(|| {
                Mutex::new(pcs::FalconAggregateSetup::<aerie_prover::Falcon1024>::new(4).unwrap())
            })
            .lock()
            .unwrap(),
    )
}

/// One shared honest production PCS proof for the finished-proof mutations.
fn honest() -> &'static (FalconWitness, pcs::AggregateProof) {
    static HONEST: OnceLock<(FalconWitness, pcs::AggregateProof)> = OnceLock::new();
    HONEST.get_or_init(|| {
        with_single_thread_pool_value("honest Falcon-512 PCS proof", || {
            let batch = fixture(ROWS);
            let setup = setup_512(batch.live_rows);
            let (proof, _) = pcs::prove_prepared(&batch, PUBLIC_RECORDS, &setup)
                .expect("honest production PCS prover");
            assert_eq!(proof.commitments.len(), GROUP_COUNT);
            pcs::verify(&public(&batch), setup.verifier(), &proof).expect("honest proof verifies");
            (batch, proof)
        })
    })
}

fn assert_proof_rejected(
    tamper: impl FnOnce(&mut pcs::AggregateProof),
    expected: ExpectedRejection,
    label: &str,
) {
    let (batch, proof) = honest();
    let mut tampered = proof.clone();
    tamper(&mut tampered);
    let error = verify_proof(batch, &tampered).expect_err(label);
    assert_rejection(&error, expected, label);
}

/// All source commitments precede the challenge reused by the range links,
/// sign, norm, and budget relations. Replacing its transcript prefix cannot
/// reuse the original stage-three proof.
fn assert_reused_point_is_bound(batch: &FalconWitness, proof: &pcs::AggregateProof) {
    let mut altered = proof.clone();
    let middle = altered.stage2.rounds.len() / 2;
    altered.stage2.rounds[middle][1] += one();
    let error = verify_proof(batch, &altered).expect_err("reused stage-two point");
    assert!(matches!(
        error,
        AggregationError::ProofRejected {
            stage: RejectionStage::Grinding
        }
    ));
    let degree = FalconDegree::from_degree(batch.degree()).expect("parameter set");
    let salts = vec![[0_u8; wire::TRANSPORT_SALT_BYTES]; batch.live_rows];
    let mut commitment_offset = wire::HEADER_BYTES + batch.live_rows * wire::TRANSPORT_SALT_BYTES;
    let encoded = wire::encode_pcs_proof(degree, batch.live_rows, &salts, proof)
        .expect("honest production wire frame");
    for group in 0..GROUP_COUNT {
        let mut bytes = Vec::new();
        proof.commitments[group]
            .serialize_with_mode(&mut bytes, Compress::Yes)
            .expect("commitment serialization");
        assert!(bytes.len() > 16, "commitment payload has a mutation byte");
        let mut found_decodable_tamper = false;
        for payload_offset in 0..bytes.len().min(8) {
            for bit in 0..8 {
                let mut mutated = encoded.clone();
                mutated[commitment_offset + payload_offset] ^= 1 << bit;
                let Ok(decoded) = wire::decode_pcs_proof(degree, &mutated) else {
                    continue;
                };
                assert!(
                    matches!(
                        verify_proof(batch, &decoded.proof),
                        Err(AggregationError::ProofRejected {
                            stage: RejectionStage::Grinding
                        })
                    ),
                    "a canonical mutation to commitment group {group} must reject"
                );
                found_decodable_tamper = true;
                break;
            }
            if found_decodable_tamper {
                break;
            }
        }
        assert!(
            found_decodable_tamper,
            "commitment group {group} needs a decodable one-bit mutation"
        );
        commitment_offset += bytes.len();
    }
}

fn commitment_bytes(commitment: &akita_types::Commitment<BaseField>) -> Vec<u8> {
    let mut bytes = Vec::new();
    commitment
        .serialize_with_mode(&mut bytes, Compress::Yes)
        .expect("commitment serialization");
    bytes
}

fn verify_proof(
    batch: &FalconWitness,
    proof: &pcs::AggregateProof,
) -> Result<pcs::VerifiedOpening, aerie_prover::AggregationError> {
    with_single_thread_pool_value("PCS verifier", || {
        if batch.degree() == 512 {
            let setup = setup_512(batch.live_rows);
            pcs::verify(&public(batch), setup.verifier(), proof)
        } else {
            let setup = setup_1024(batch.live_rows);
            pcs::verify(&public(batch), setup.verifier(), proof)
        }
    })
}

#[test]
fn reused_stage_two_point_is_bound_to_all_prior_commitments() {
    let (batch, proof) = honest();
    assert_reused_point_is_bound(batch, proof);
}

#[derive(Clone, Copy, Debug)]
enum ExpectedRejection {
    Projection,
    Backend,
    InvalidStatement,
    Malformed(MalformedProof),
    Rejected(RejectionStage),
    /// A tampered message changes the next challenge's grinding input. The
    /// stale nonce still meets a `d`-bit difficulty with probability `2^-d`,
    /// and the given stage's check then rejects.
    GrindingOr(RejectionStage),
}

fn assert_rejection(error: &AggregationError, expected: ExpectedRejection, label: &str) {
    match expected {
        ExpectedRejection::Projection => assert!(
            matches!(error, AggregationError::ProjectionRejected),
            "{label}: {error:?}"
        ),
        ExpectedRejection::Backend => assert!(
            matches!(error, AggregationError::Backend(_)),
            "{label}: {error:?}"
        ),
        ExpectedRejection::InvalidStatement => assert!(
            matches!(error, AggregationError::InvalidStatement(_)),
            "{label}: {error:?}"
        ),
        ExpectedRejection::Malformed(expected) => assert!(
            matches!(error, AggregationError::MalformedProof(kind) if *kind == expected),
            "{label}: {error:?}"
        ),
        ExpectedRejection::Rejected(expected) => assert!(
            matches!(error, AggregationError::ProofRejected { stage } if *stage == expected),
            "{label}: {error:?}"
        ),
        ExpectedRejection::GrindingOr(expected) => assert!(
            matches!(
                error,
                AggregationError::ProofRejected { stage }
                    if *stage == RejectionStage::Grinding || *stage == expected
            ),
            "{label}: {error:?}"
        ),
    }
}

/// Cheating prover on tampered committed witness data or projection
/// outputs: the prover's own guard may error, otherwise the produced proof
/// must fail verification.
fn assert_witness_tamper_rejected(
    tamper_tables: impl Fn(&mut SourceTables) + Send + Sync,
    tamper_outputs: impl Fn(&[u8], &mut ProjectionOutputs) + Send + Sync,
    expected: ExpectedRejection,
    label: &str,
) {
    assert_witness_tamper_rejected_on(
        &fixture(ROWS),
        tamper_tables,
        tamper_outputs,
        expected,
        label,
    );
}

/// [`assert_witness_tamper_rejected`] on a given batch. The semantic oracle
/// flags a defect when the tampered tables fail their semantic checks, the
/// batch violates the relation, or the emitted outputs differ from the
/// honest projection under the transcript's seed.
fn assert_witness_tamper_rejected_on(
    batch: &FalconWitness,
    tamper_tables: impl Fn(&mut SourceTables) + Send + Sync,
    tamper_outputs: impl Fn(&[u8], &mut ProjectionOutputs) + Send + Sync,
    expected: ExpectedRejection,
    label: &str,
) {
    let defect = AtomicBool::new(batch.validate_semantics().is_err());
    let table_hook = |tables: &mut SourceTables| {
        tamper_tables(tables);
        defect.fetch_or(tables.validate_semantics(batch).is_err(), Ordering::Relaxed);
    };
    let output_hook = |seed: &[u8], outputs: &mut ProjectionOutputs| {
        tamper_outputs(seed, outputs);
        let honest = projection::project(batch, seed);
        defect.fetch_or(
            honest.map_or(true, |projected| projected.outputs() != outputs),
            Ordering::Relaxed,
        );
    };
    let result = with_single_thread_pool_value("witness tamper PCS proof", || {
        if batch.degree() == 512 {
            let setup = setup_512(batch.live_rows);
            pcs::prove_prepared_with_witness_tamper(
                batch,
                PUBLIC_RECORDS,
                &setup,
                table_hook,
                output_hook,
            )
        } else {
            let setup = setup_1024(batch.live_rows);
            pcs::prove_prepared_with_witness_tamper(
                batch,
                PUBLIC_RECORDS,
                &setup,
                table_hook,
                output_hook,
            )
        }
    });
    let error = match result {
        Err(error) => error,
        Ok((proof, _)) => verify_proof(batch, &proof).expect_err(label),
    };
    assert_rejection(&error, expected, label);
    assert!(
        defect.load(Ordering::Relaxed),
        "{label}: the semantic oracle must identify a witness defect"
    );
}

/// A cheating prover whose `(y, z)` pass both predicates: the prover must
/// emit a proof, the verifier's predicate check must pass on it, and the
/// verifier must still reject, so the rejection is stage 1's (or a later
/// stage's), never the predicates'.
fn assert_predicate_passing_outputs_rejected(
    batch: &FalconWitness,
    tamper_outputs: impl Fn(&[u8], &mut ProjectionOutputs) + Send + Sync,
    label: &str,
) {
    let degree = FalconDegree::from_degree(batch.degree()).expect("parameter set");
    let no_table_tamper = |_: &mut SourceTables| {};
    let result = with_single_thread_pool_value("predicate-passing output tamper", || {
        if batch.degree() == 512 {
            let setup = setup_512(batch.live_rows);
            pcs::prove_prepared_with_witness_tamper(
                batch,
                PUBLIC_RECORDS,
                &setup,
                no_table_tamper,
                tamper_outputs,
            )
        } else {
            let setup = setup_1024(batch.live_rows);
            pcs::prove_prepared_with_witness_tamper(
                batch,
                PUBLIC_RECORDS,
                &setup,
                no_table_tamper,
                tamper_outputs,
            )
        }
    });
    let (proof, _) = result
        .unwrap_or_else(|error| panic!("{label}: the cheating prover must emit a proof: {error}"));
    assert!(
        projection::check_outputs(degree, &proof.projection_outputs),
        "{label}: the cheating outputs must pass the predicates"
    );
    let error = verify_proof(batch, &proof).expect_err(label);
    assert!(
        matches!(
            error,
            AggregationError::ProofRejected {
                stage: RejectionStage::Stage1
            }
        ),
        "{label}: {error:?}"
    );
}

/// Cheating prover on a tampered batch witness column, honest public data.
fn assert_cheating_batch_rejected(batch: &FalconWitness, expected: ExpectedRejection, label: &str) {
    assert!(
        batch.validate_semantics().is_err(),
        "{label}: the batch must violate the relation"
    );
    let result = with_single_thread_pool_value("malformed batch PCS proof", || {
        if batch.degree() == 512 {
            let setup = setup_512(batch.live_rows);
            pcs::prove_prepared(batch, PUBLIC_RECORDS, &setup)
        } else {
            let setup = setup_1024(batch.live_rows);
            pcs::prove_prepared(batch, PUBLIC_RECORDS, &setup)
        }
    });
    let error = match result {
        Err(error) => error,
        Ok((proof, _)) => verify_proof(batch, &proof).expect_err(label),
    };
    assert_rejection(&error, expected, label);
}

/// Port the reference padded-row soundness assertion to the production PCS
/// path. An honest witness at the same live record count first proves and
/// verifies with this exact setup, so a setup-shape mismatch cannot make the
/// tampered witness appear rejected.
fn assert_padded_rows_rejected_512() {
    let honest_batch = fixture(3);
    with_single_thread_pool_value("three-record padded-row control", || {
        let setup = pcs::FalconAggregateSetup::<Falcon512>::new(3).expect("three-record setup");
        let (proof, _) = pcs::prove_prepared(&honest_batch, PUBLIC_RECORDS, &setup)
            .expect("honest three-record prover");
        pcs::verify(&public(&honest_batch), setup.verifier(), &proof)
            .expect("honest three-record proof verifies");

        for (label, batch) in [
            ("nonzero padded s1 row", {
                let mut batch = fixture(3);
                batch.s1[3 * Falcon512::DEGREE + 2] = 7;
                batch
            }),
            ("nonzero padded slack row", {
                let mut batch = fixture(3);
                batch.slack[3 * 4] = 5;
                batch
            }),
        ] {
            assert!(
                batch.validate_semantics().is_err(),
                "{label}: invalid witness"
            );
            match pcs::prove_prepared(&batch, PUBLIC_RECORDS, &setup) {
                Err(error) => assert!(
                    matches!(error, AggregationError::ProjectionRejected),
                    "{label}: preserve the projection rejection reason, got {error}"
                ),
                Ok((proof, _)) => match pcs::verify(&public(&batch), setup.verifier(), &proof) {
                    Err(error) => {
                        let stage = if label == "nonzero padded s1 row" {
                            RejectionStage::Stage1
                        } else {
                            RejectionStage::Stage3
                        };
                        assert_rejection(&error, ExpectedRejection::Rejected(stage), label);
                    }
                    Ok(_) => panic!("{label}: production PCS verifier accepted the witness"),
                },
            }
        }
    });
}

/// Commit group cells that differ (cell 3 of `role` raised by one) from the
/// honest witness the PIOP proves, with the opening proving the committed
/// polynomials' true evaluations. The prover must emit a proof, and the
/// PCS verifier must reject it at the multi-group opening: the PIOP replays
/// honestly, so only the zero-extension tie between the ledger claim and
/// the committed physical polynomial can fail, and a rejection anywhere
/// else is the wrong reason.
fn assert_group_tamper_rejected_at_opening<P: AggregationParams>(
    batch: &FalconWitness,
    verifier_setup: &pcs::FalconVerifierSetup<P>,
    role: GroupRole,
) -> Result<(), String> {
    let (proof, _) = pcs::prove_with_group_tamper(batch, PUBLIC_RECORDS, &|group, cells| {
        if group == role {
            cells[3] += 1;
        }
    })
    .map_err(|error| format!("the {role:?} group-tamper prover must emit a proof: {error}"))?;
    match pcs::verify(&public(batch), verifier_setup, &proof) {
        Ok(_) => Err(format!(
            "the PCS verifier accepted a tampered {role:?} group"
        )),
        Err(AggregationError::ProofRejected {
            stage: RejectionStage::PcsOpening,
        }) => Ok(()),
        Err(error) => Err(format!(
            "the tampered {role:?} group must be rejected at the opening, not by: {error}"
        )),
    }
}

fn with_single_thread_pool(label: &'static str, body: impl FnOnce() -> Result<(), String> + Send) {
    with_single_thread_pool_value(label, body).unwrap_or_else(|error| panic!("{label}: {error}"));
}

fn with_single_thread_pool_value<T: Send>(
    label: &'static str,
    body: impl FnOnce() -> T + Send,
) -> T {
    #[cfg(feature = "parallel")]
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap_or_else(|error| panic!("build {label} pool: {error}"));
        pool.install(body)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = label;
        body()
    }
}

fn one() -> OpeningField {
    OpeningField::one()
}

fn isqrt(value: u128) -> u128 {
    let mut root = (value as f64).sqrt() as u128;
    while root * root > value {
        root -= 1;
    }
    while (root + 1) * (root + 1) <= value {
        root += 1;
    }
    root
}

/// A `y` with `sum ctr_p(y_j)^2 = target` exactly, alternating signs.
fn y_with_energy(target: u128) -> [BaseField; NORM_TERMINAL_ROWS] {
    let mut rest = target;
    let mut y = [BaseField::zero(); NORM_TERMINAL_ROWS];
    let mut index = 0;
    while rest > 0 {
        let root = isqrt(rest);
        rest -= root * root;
        let signed = i128::try_from(root).expect("root below 2^64");
        y[index] = BaseField::from_i128(if index % 2 == 0 { signed } else { -signed });
        index += 1;
    }
    let energy: u128 = y
        .iter()
        .map(|&value| centered(value).unsigned_abs().pow(2))
        .sum();
    assert_eq!(energy, target);
    y
}

// Bullets 1 and 2 (the `a_1` case): balanced radix-16 and radix-8
// coefficient digits outside their alphabets.
#[test]
fn rejects_a0_and_a1_cells_outside_their_alphabets() {
    assert_witness_tamper_rejected(
        |tables| tables.a0_mut()[5] = 4,
        |_, _| {},
        ExpectedRejection::Backend,
        "a0 above 3",
    );
    assert_witness_tamper_rejected(
        |tables| tables.a0_mut()[5] = -5,
        |_, _| {},
        ExpectedRejection::Backend,
        "a0 below -4",
    );
    assert_witness_tamper_rejected(
        |tables| tables.a1_mut()[7] = 8,
        |_, _| {},
        ExpectedRejection::Backend,
        "a1 above 7",
    );
    assert_witness_tamper_rejected(
        |tables| tables.a1_mut()[7] = -9,
        |_, _| {},
        ExpectedRejection::Backend,
        "a1 below -8",
    );
}

// Bullet 2 (the budget case): balanced radix-8 row digits outside
// `[-4, 3]`.
#[test]
fn rejects_budget_cells_outside_radix8() {
    assert_witness_tamper_rejected(
        |tables| tables.budget_mut(0)[3] = 4,
        |_, _| {},
        ExpectedRejection::Backend,
        "budget u0 above 3",
    );
    assert_witness_tamper_rejected(
        |tables| tables.budget_mut(2)[5] = -5,
        |_, _| {},
        ExpectedRejection::Backend,
        "budget u2 below -4",
    );
}

// Bullet 3: `a2` outside the balanced radix-16 alphabet.
#[test]
fn rejects_high_cells_outside_radix16() {
    assert_witness_tamper_rejected(
        |tables| tables.a2_mut()[13] = 8,
        |_, _| {},
        ExpectedRejection::Backend,
        "a2 above 7",
    );
    assert_witness_tamper_rejected(
        |tables| tables.a2_mut()[13] = -9,
        |_, _| {},
        ExpectedRejection::Backend,
        "a2 below -8",
    );
}

// Bullet 4: sign cells outside `{-1, 1}`. Zero is the case a bit-valued
// sign would admit, so it is the one that matters.
#[test]
fn rejects_sign_cells_outside_plus_minus_one() {
    assert_witness_tamper_rejected(
        |tables| tables.sigma_mut()[21] = 0,
        |_, _| {},
        ExpectedRejection::Rejected(RejectionStage::Stage3),
        "sign of 0",
    );
    assert_witness_tamper_rejected(
        |tables| tables.sigma_mut()[21] = 2,
        |_, _| {},
        ExpectedRejection::Rejected(RejectionStage::Stage3),
        "sign of 2",
    );
}

// Bullet 5: an extracted `S_2` coefficient outside `[-2047, 2047]`. The
// honest encoder is the prover-side guard; within committed tables the
// bound is exactly the alphabet identities (a maximal in-alphabet cell
// tuple decodes to 2047), so the cell-level route is bullets 1 to 3.
#[test]
fn rejects_extracted_s2_coefficients_outside_the_falcon_limit() {
    let mut batch = fixture(ROWS);
    batch.s2[9] = 2048;
    assert_cheating_batch_rejected(
        &batch,
        ExpectedRejection::InvalidStatement,
        "s2 coefficient of 2048",
    );
    let mut batch = fixture(ROWS);
    batch.s2[9] = -2048;
    assert_cheating_batch_rejected(
        &batch,
        ExpectedRejection::InvalidStatement,
        "s2 coefficient of -2048",
    );
}

// Bullet 6: an encodability high-part total above 392, both as a raw batch
// whose slack would be negative and as in-alphabet budget digits that
// misdeclare the slack of a valid row.
#[test]
fn rejects_an_encodability_high_part_total_above_392() {
    let mut batch = fixture(ROWS);
    // Row 0 becomes 512 coefficients of high part 8: total 4096 > 392,
    // while every coefficient stays inside the Falcon limit.
    for cell in &mut batch.s2[..Falcon512::DEGREE] {
        *cell = 1024;
    }
    assert_cheating_batch_rejected(
        &batch,
        ExpectedRejection::InvalidStatement,
        "high-part total above 392",
    );
    // A shifted (still in-alphabet) budget digit breaks the exact
    // `high_sum + slack = 392` tie for an otherwise valid row.
    assert_witness_tamper_rejected(
        |tables| {
            let cell = &mut tables.budget_mut(0)[2];
            *cell = if *cell == 3 { 2 } else { *cell + 1 };
        },
        |_, _| {},
        ExpectedRejection::Rejected(RejectionStage::Stage3),
        "misdeclared budget slack",
    );
}

// Bullet 7: norm outputs whose centered energy is exactly `C_S` pass the
// predicate and are rejected by stage 1; energy `C_S + 1` fails the
// predicate. Bullets 8 and 9 share the helper and run in their own tests.
#[test]
fn rejects_norm_outputs_one_above_the_cap() {
    let batch = fixture(ROWS);
    let degree = FalconDegree::Falcon512;
    let cap = norm_cap(degree);
    for (label, energy, passes) in [
        ("y energy C_S", cap, true),
        ("y energy C_S + 1", cap + 1, false),
    ] {
        let y = y_with_energy(energy);
        let probe = ProjectionOutputs {
            y,
            z: [BaseField::zero(); projection::RESIDUAL_WIDTH],
        };
        assert_eq!(projection::check_outputs(degree, &probe), passes, "{label}");
        if passes {
            assert_predicate_passing_outputs_rejected(
                &batch,
                move |_, outputs| outputs.y = y,
                label,
            );
        } else {
            assert_witness_tamper_rejected_on(
                &batch,
                |_| {},
                move |_, outputs| outputs.y = y,
                ExpectedRejection::Rejected(RejectionStage::Stage1),
                label,
            );
        }
    }
    // The same `y` on a finished proof: the predicate check runs first.
    assert_proof_rejected(
        |proof| proof.projection_outputs.y = y_with_energy(cap + 1),
        ExpectedRejection::Rejected(RejectionStage::Stage1),
        "finished proof with y energy C_S + 1",
    );
    assert_proof_rejected(
        |proof| proof.projection_outputs.y[17] += BaseField::one(),
        ExpectedRejection::Rejected(RejectionStage::Grinding),
        "finished proof with a predicate-passing y edit",
    );
}

// Bullet 8: `z_j = +-1 (mod q)` fails the residual predicate; `z_j + q`
// passes it and is rejected by stage 1.
#[test]
fn rejects_residual_outputs_off_a_multiple_of_q() {
    let q = BaseField::from_u64(FALCON_Q);
    let batch = fixture(ROWS);
    assert_predicate_passing_outputs_rejected(&batch, |_, outputs| outputs.z[4] += q, "z_j + q");
    assert_witness_tamper_rejected_on(
        &batch,
        |_| {},
        |_, outputs| outputs.z[4] += BaseField::one(),
        ExpectedRejection::Malformed(MalformedProof::ProjectionEncoding),
        "z_j + 1",
    );
    assert_witness_tamper_rejected_on(
        &batch,
        |_| {},
        |_, outputs| outputs.z[30] -= BaseField::one(),
        ExpectedRejection::Malformed(MalformedProof::ProjectionEncoding),
        "z_j - 1",
    );
    assert_proof_rejected(
        |proof| proof.projection_outputs.z[4] += q,
        ExpectedRejection::Rejected(RejectionStage::Grinding),
        "finished proof with a predicate-passing z edit",
    );
}

// Bullet 9: the smallest canonical multiple of `q` above `(p - 1)/2`. As a
// canonical `u64` it is a multiple of `q`; its centered value is not, so a
// verifier reducing in `F_p` instead of on centered integers would accept.
// The companion `-q` (canonical `p - q`) passes the predicate and is
// rejected by stage 1.
#[test]
fn rejects_an_output_at_the_centered_boundary() {
    let degree = FalconDegree::Falcon512;
    let half = (FIELD_MODULUS - 1) / 2;
    let canonical = (half / FALCON_Q + 1) * FALCON_Q;
    assert!(canonical > half && canonical.is_multiple_of(FALCON_Q));
    let boundary = BaseField::from_u64(canonical);
    assert!(centered(boundary) % i128::from(FALCON_Q) != 0);
    let minus_q = BaseField::from_u64(FIELD_MODULUS - FALCON_Q);
    assert_eq!(centered(minus_q), -i128::from(FALCON_Q));
    let (_, honest_proof) = honest();
    let mut probe = honest_proof.projection_outputs.clone();
    probe.z[0] = boundary;
    assert!(!projection::check_outputs(degree, &probe));
    probe.z[0] = minus_q;
    assert!(projection::check_outputs(degree, &probe));
    assert_witness_tamper_rejected(
        |_| {},
        move |_, outputs| outputs.z[0] = boundary,
        ExpectedRejection::Malformed(MalformedProof::ProjectionEncoding),
        "cheating prover with the ctr_p boundary z_0",
    );
    assert_predicate_passing_outputs_rejected(
        &fixture(ROWS),
        move |_, outputs| outputs.z[0] = minus_q,
        "cheating prover with z_0 = -q",
    );
    assert_proof_rejected(
        |proof| proof.projection_outputs.z[0] = boundary,
        ExpectedRejection::Rejected(RejectionStage::Stage1),
        "ctr_p boundary z_0",
    );
    assert_proof_rejected(
        |proof| proof.projection_outputs.z[0] = minus_q,
        ExpectedRejection::Rejected(RejectionStage::Grinding),
        "z_0 = -q",
    );
}

// Bullet 10: a source with nonzero residual `delta = S_1 + h S_2^* - t` modulo
// `q` and a valid norm. The honest projection of such a batch fails the
// residual predicate (the prover's own guard); a cheating prover emitting
// the honest batch's `z` (or zeros) passes the predicate and is rejected by
// stage 1.
#[test]
fn rejects_a_source_with_a_nonzero_residual() {
    let honest_z = honest().1.projection_outputs.z;
    let mut batch = fixture(ROWS);
    batch.s1[5 * Falcon512::DEGREE + 11] += 1;
    assert_cheating_batch_rejected(&batch, ExpectedRejection::Projection, "Delta mod q nonzero");
    assert_predicate_passing_outputs_rejected(
        &batch,
        move |_, outputs| outputs.z = honest_z,
        "Delta mod q nonzero with the honest z",
    );
    assert_predicate_passing_outputs_rejected(
        &batch,
        |_, outputs| outputs.z = [BaseField::zero(); projection::RESIDUAL_WIDTH],
        "Delta mod q nonzero with z = 0",
    );
}

// Bullet 11: a false `w_P`, alone and moved together with `s_{1,P}` so the
// ring claim `v_ring = C(r_Pi) + v_Delta - v_S1` stage three consumes is unchanged.
// Stage 1's terminal identity must reject both.
#[test]
fn rejects_a_false_delta_terminal() {
    assert_proof_rejected(
        |proof| proof.stage1.projection_claims[2] += one(),
        ExpectedRejection::Rejected(RejectionStage::Stage1),
        "false w_P",
    );
    assert_proof_rejected(
        |proof| {
            proof.stage1.projection_claims[0] += one();
            proof.stage1.projection_claims[2] += one();
        },
        ExpectedRejection::Rejected(RejectionStage::Stage1),
        "false residual claim preserving the ring difference",
    );
}

// Stage 1 tampers: every terminal claim, the first, a middle and the last
// round in each compressed coefficient, and a dropped round.
#[test]
fn rejects_stage1_round_and_terminal_tampers() {
    for slot in 0..3 {
        assert_proof_rejected(
            |proof| proof.stage1.projection_claims[slot] += one(),
            ExpectedRejection::Rejected(RejectionStage::Stage1),
            &format!("P terminal {slot}"),
        );
    }
    let rounds = honest().1.stage1.rounds.len();
    for round in [0, rounds / 2, rounds - 1] {
        for coefficient in 0..3 {
            assert_proof_rejected(
                |proof| {
                    proof.stage1.rounds[round][coefficient] += one();
                },
                ExpectedRejection::GrindingOr(RejectionStage::Stage1),
                &format!("P round {round} coefficient {coefficient}"),
            );
        }
    }
    assert_proof_rejected(
        |proof| {
            proof.stage1.rounds.pop();
        },
        ExpectedRejection::Malformed(MalformedProof::StageShape),
        "P missing its last round",
    );
}

// Bullet 12: the projection seed cannot be sampled before every commitment
// is absorbed, and a transcript cannot be finished without it.
#[test]
fn rejects_a_projection_seed_sampled_before_a_commitment() {
    let (batch, proof) = honest();
    let manifest = CommitmentManifest::for_batch(ROWS).expect("manifest");
    let instance = setup_512(ROWS)
        .verifier()
        .transcript_instance(&public(batch))
        .unwrap();
    for absorbed in 0..GROUP_COUNT {
        let mut transcript = ProtocolTranscript::new(&manifest, &instance).unwrap();
        for (role, payload) in ROLES.iter().zip(&proof.commitments).take(absorbed) {
            let payload = commitment_bytes(payload);
            transcript
                .absorb_group_commitment(*role, &payload)
                .expect("canonical order");
        }
        assert!(matches!(
            transcript.derive_projection_seed(),
            Err(TranscriptOrderError::EventOrder { .. })
        ));
        assert!(matches!(
            transcript.finish(),
            Err(TranscriptOrderError::Incomplete { .. })
        ));
    }
}

/// The seed a transcript over the manifest for `live_records` derives from
/// the honest proof's commitments.
fn seed_under(live_records: usize) -> Vec<u8> {
    let (batch, proof) = honest();
    let manifest = CommitmentManifest::for_batch(live_records).expect("manifest");
    let mut instance = setup_512(ROWS)
        .verifier()
        .transcript_instance(&public(batch))
        .unwrap();
    instance[11..19].copy_from_slice(&(live_records as u64).to_le_bytes());
    let mut transcript = ProtocolTranscript::new(&manifest, &instance).unwrap();
    for (role, payload) in ROLES.iter().zip(&proof.commitments) {
        let payload = commitment_bytes(payload);
        transcript
            .absorb_group_commitment(*role, &payload)
            .expect("canonical order");
    }
    transcript.derive_projection_seed().expect("seed").to_vec()
}

// Bullet 13 (source order): outputs of the same records projected in a
// different order (the two eight-record halves exchanged, keys and targets
// with them so every row stays relation-positive). They pass both
// predicates; stage 1 links them to the committed order and rejects.
#[test]
fn rejects_a_changed_source_order() {
    let mut permuted = fixture(ROWS);
    let half = (ROWS / 2) * Falcon512::DEGREE;
    for column in [
        &mut permuted.h,
        &mut permuted.t,
        &mut permuted.s1,
        &mut permuted.s2,
    ] {
        column.rotate_left(half);
    }
    permuted.v.rotate_left(half);
    permuted.slack.rotate_left((ROWS / 2) * 4);
    permuted
        .validate_semantics()
        .expect("the permuted batch stays relation-positive");
    assert_predicate_passing_outputs_rejected(
        &fixture(ROWS),
        |seed, outputs| {
            *outputs = projection::project(&permuted, seed)
                .expect("permuted projection passes the predicates")
                .outputs()
                .clone();
        },
        "changed source order",
    );
}

// Bullet 13 (binding): outputs projected under a seed the transcript did
// not derive: an unrelated seed, and the seed of a transcript whose profile
// binding names another live record count over the same padded domain.
// Both pass the predicates and fail stage 1. The honest seed recomputed
// from the proof's commitments matches the one the prover saw, so the
// wrong-binding seed differs only in the bound profile.
#[test]
fn rejects_outputs_under_a_wrong_seed_or_profile_binding() {
    let batch = fixture(ROWS);
    let honest_seed = seed_under(ROWS);
    let rebound = seed_under(ROWS - 1);
    assert_ne!(honest_seed, rebound);
    let seen = Mutex::new(Vec::new());
    for (label, seed) in [
        ("unrelated seed", vec![7_u8; 32]),
        ("wrong profile binding", rebound),
    ] {
        assert_predicate_passing_outputs_rejected(
            &batch,
            |transcript_seed, outputs| {
                seen.lock().unwrap().push(transcript_seed.to_vec());
                *outputs = projection::project(&batch, &seed)
                    .expect("honest batch under another seed")
                    .outputs()
                    .clone();
            },
            label,
        );
    }
    assert!(seen.lock().unwrap().iter().all(|seed| *seed == honest_seed));
    // The verifier side of the same binding: the honest proof checked
    // against another live count re-derives the seed and rejects.
    let (batch, proof) = honest();
    let mut rebound_public = public(batch);
    rebound_public.records = ROWS - 1;
    let result = with_single_thread_pool_value("live-count rebinding", || {
        let setup = setup_512(batch.live_rows);
        pcs::verify(&rebound_public, setup.verifier(), proof)
    });
    let rejected = matches!(
        &result,
        Err(AggregationError::UnsupportedBatch(UnsupportedBatch::SetupSize {
            expected: ROWS,
            actual,
        })) if *actual == ROWS - 1
    );
    assert!(rejected, "live-count rebinding: {result:?}");
}

#[test]
fn production_pcs_rejects_changed_public_digest_keys_and_target_sign() {
    let (batch, proof) = honest();

    let wrong_digest = PublicInputs {
        degree: FalconDegree::from_degree(batch.degree()).expect("parameter set"),
        records: ROWS,
        salt_mode: aerie_prover::wire::SaltMode::PerRecord,
        public_records: b"different public records        ",
        h: aerie_prover::ring::PublicRingKeys::Expanded(&batch.h),
        t: aerie_prover::ring::PublicTargets::Expanded(&batch.t),
    };
    assert!(
        matches!(
            with_single_thread_pool_value("changed digest verification", || {
                let setup = setup_512(batch.live_rows);
                pcs::verify(&wrong_digest, setup.verifier(), proof)
            }),
            Err(AggregationError::ProofRejected {
                stage: RejectionStage::Grinding
            })
        ),
        "changing the public-record digest must reject the production proof"
    );

    let mut changed_h = batch.h.clone();
    changed_h[3] += 1;
    let wrong_keys = PublicInputs {
        degree: FalconDegree::from_degree(batch.degree()).expect("parameter set"),
        records: ROWS,
        salt_mode: aerie_prover::wire::SaltMode::PerRecord,
        public_records: PUBLIC_RECORDS,
        h: aerie_prover::ring::PublicRingKeys::Expanded(&changed_h),
        t: aerie_prover::ring::PublicTargets::Expanded(&batch.t),
    };
    assert!(
        matches!(
            with_single_thread_pool_value("changed key verification", || {
                let setup = setup_512(batch.live_rows);
                pcs::verify(&wrong_keys, setup.verifier(), proof)
            }),
            Err(AggregationError::ProofRejected {
                stage: RejectionStage::Stage3
            })
        ),
        "changing a public ring-key coefficient must reject"
    );

    let negated_t = batch.t.iter().map(|&value| -value).collect::<Vec<_>>();
    let wrong_targets = PublicInputs {
        degree: FalconDegree::from_degree(batch.degree()).expect("parameter set"),
        records: ROWS,
        salt_mode: aerie_prover::wire::SaltMode::PerRecord,
        public_records: PUBLIC_RECORDS,
        h: aerie_prover::ring::PublicRingKeys::Expanded(&batch.h),
        t: aerie_prover::ring::PublicTargets::Expanded(&negated_t),
    };
    assert!(
        matches!(
            with_single_thread_pool_value("negated target verification", || {
                let setup = setup_512(batch.live_rows);
                pcs::verify(&wrong_targets, setup.verifier(), proof)
            }),
            Err(AggregationError::ProofRejected {
                stage: RejectionStage::Stage3
            })
        ),
        "negating public HashToPoint targets must reject"
    );
}

#[cfg(feature = "honest-vectors")]
#[test]
fn production_pcs_accepts_honest_fndsa_rows_and_pins_target_sign() {
    with_single_thread_pool("honest Falcon-512 vector proof", || {
        let rows = (0..4)
            .map(|index| {
                aerie_falcon_witness::honest::honest_direct_row::<Falcon512>(
                    0xd1ff_5eed,
                    index as u64,
                )
                .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut batch = FalconWitness::from_rows(&rows);
        aerie_prover::pad_witness(&mut batch).map_err(|error| error.to_string())?;
        if !batch.v.iter().any(|&value| value != 0) {
            return Err("honest vectors must include a nonzero quotient".into());
        }
        let setup =
            pcs::FalconAggregateSetup::<Falcon512>::new(4).map_err(|error| error.to_string())?;
        let (proof, _) = pcs::prove_prepared(&batch, b"honest records digest           ", &setup)
            .map_err(|error| error.to_string())?;
        let public = PublicInputs {
            degree: FalconDegree::from_degree(batch.degree()).expect("parameter set"),
            records: 4,
            salt_mode: aerie_prover::wire::SaltMode::PerRecord,
            public_records: b"honest records digest           ",
            h: aerie_prover::ring::PublicRingKeys::Expanded(&batch.h),
            t: aerie_prover::ring::PublicTargets::Expanded(&batch.t),
        };
        pcs::verify(&public, setup.verifier(), &proof)
            .map_err(|error| format!("honest FN-DSA proof: {error}"))?;

        let negated_targets = batch.t.iter().map(|&target| -target).collect::<Vec<_>>();
        let wrong_public = PublicInputs {
            t: aerie_prover::ring::PublicTargets::Expanded(&negated_targets),
            ..public
        };
        assert!(matches!(
            pcs::verify(&wrong_public, setup.verifier(), &proof),
            Err(AggregationError::ProofRejected {
                stage: RejectionStage::Stage3
            })
        ));

        let mut negated_witness = batch.clone();
        for target in &mut negated_witness.t {
            *target = -*target;
        }
        match pcs::prove_prepared(
            &negated_witness,
            b"honest records digest           ",
            &setup,
        ) {
            Err(error) => assert!(matches!(error, AggregationError::ProjectionRejected)),
            Ok((negated_proof, _)) => assert!(matches!(
                pcs::verify(&public, setup.verifier(), &negated_proof),
                Err(AggregationError::ProofRejected {
                    stage: RejectionStage::Stage3
                })
            )),
        }
        Ok(())
    });
}

#[test]
fn rejects_each_stage_claim_and_each_round_coefficient() {
    let (_, proof) = honest();
    for source in 0..proof.stage1.basis16_claims.len() {
        for half in 0..2 {
            assert_proof_rejected(
                |p| p.stage1.basis16_claims[source][half] += one(),
                ExpectedRejection::Rejected(
                    if proof.stage1.basis16_claims[source][1 - half].is_zero() {
                        RejectionStage::Grinding
                    } else {
                        RejectionStage::Stage1
                    },
                ),
                &format!("basis16 child claim {source} {half}"),
            );
        }
    }
    for index in 0..6 {
        assert_proof_rejected(
            |p| p.stage2.folded_digits[index] += one(),
            ExpectedRejection::Rejected(RejectionStage::Stage2),
            &format!("folded digit claim {index}"),
        );
    }
    for index in 0..9 {
        assert_proof_rejected(
            |p| p.stage3.terminal_evaluations[index] += one(),
            ExpectedRejection::Rejected(RejectionStage::Stage3),
            &format!("committed terminal evaluation {index}"),
        );
    }
    for round in [
        0,
        proof.stage2.rounds.len() / 2,
        proof.stage2.rounds.len() - 1,
    ] {
        for coefficient in 0..4 {
            assert_proof_rejected(
                |p| p.stage2.rounds[round][coefficient] += one(),
                ExpectedRejection::GrindingOr(RejectionStage::Stage2),
                &format!("range leaf round {round} {coefficient}"),
            );
        }
        for coefficient in 0..3 {
            assert_proof_rejected(
                |p| p.stage3.rounds[round][coefficient] += one(),
                ExpectedRejection::GrindingOr(RejectionStage::Stage3),
                &format!("stage three round {round} {coefficient}"),
            );
        }
    }
    assert_proof_rejected(
        |p| {
            p.stage2.rounds.pop();
        },
        ExpectedRejection::Malformed(MalformedProof::StageShape),
        "missing stage two round",
    );
    assert_proof_rejected(
        |p| {
            p.stage3.rounds.pop();
        },
        ExpectedRejection::Malformed(MalformedProof::StageShape),
        "missing stage three round",
    );
    assert_proof_rejected(
        |p| p.stage1.basis16_claims.swap(0, 1),
        ExpectedRejection::Rejected(RejectionStage::Stage1),
        "reordered basis16 claims",
    );
}

#[test]
fn rejects_inconsistent_source_reduction_claims() {
    for source in [0, 1] {
        assert_proof_rejected(
            |p| p.stage1.projection_claims[source] += one(),
            ExpectedRejection::Rejected(RejectionStage::Stage1),
            &format!("source reduction input {source}"),
        );
        assert_proof_rejected(
            |p| p.stage3.terminal_evaluations[source] += one(),
            ExpectedRejection::Rejected(RejectionStage::Stage3),
            &format!("source reduction output {source}"),
        );
    }
}

// Bullet 17: a transcript reorder: commitment payloads absorbed in swapped
// slots, among the Width14 groups, among the Width4 groups, and across the
// two families.
#[test]
fn rejects_a_transcript_reorder() {
    assert_proof_rejected(
        |proof| proof.commitments.swap(0, 1),
        ExpectedRejection::Rejected(RejectionStage::Grinding),
        "S1 and slack swap",
    );
    assert_proof_rejected(
        |proof| proof.commitments.swap(2, 3),
        ExpectedRejection::Rejected(RejectionStage::Grinding),
        "S2Sources and final Budget swap",
    );
    assert_proof_rejected(
        |proof| proof.commitments.swap(1, 2),
        ExpectedRejection::Rejected(RejectionStage::Grinding),
        "reorder across the group families",
    );
}

// PCS frame tests live in `protocol::wire`, where they exercise the PCS
// codec and verifier.

// Bullet 19: a nonzero semantic padded row in an arbitrary live batch
// (three live rows padded to four), in the `S_1` witness and in the
// slack table.
#[test]
fn rejects_a_nonzero_semantic_padded_row() {
    assert_padded_rows_rejected_512();
}

// The semantic witness hook shares the production PCS path: with no edits it
// must still produce an honestly verifiable proof at the supported minimum
// batch shape. The resulting aggregate proof also provides the regression base
// for the joint mutation of all nine stage-three claims.
#[test]
fn pcs_semantic_witness_hook_roundtrips_and_rejects_joint_terminal_mutation() {
    with_single_thread_pool("semantic-hook roundtrip", || {
        let batch = fixture(8);
        let setup =
            pcs::FalconAggregateSetup::<Falcon512>::new(8).map_err(|error| error.to_string())?;
        let (proof, _) = pcs::prove_prepared_with_witness_tamper(
            &batch,
            PUBLIC_RECORDS,
            &setup,
            |_| {},
            |_, _| {},
        )
        .map_err(|error| error.to_string())?;
        pcs::verify(&public(&batch), setup.verifier(), &proof)
            .map_err(|error| format!("no-op semantic witness hook must verify: {error}"))?;
        // A joint mutation must fail the complete grouped opening.
        let mut tampered = proof;
        let deltas = [-4005_i64, -3607, 10014, 3749, 5954, 123, -188, 4458, -15752];
        for (terminal, delta) in tampered.stage3.terminal_evaluations.iter_mut().zip(deltas) {
            for _ in 0..delta.unsigned_abs() {
                if delta >= 0 {
                    *terminal += OpeningField::one();
                } else {
                    *terminal -= OpeningField::one();
                }
            }
        }
        let error = pcs::verify(&public(&batch), setup.verifier(), &tampered)
            .expect_err("joint terminal mutation");
        assert!(matches!(
            error,
            AggregationError::ProofRejected {
                stage: RejectionStage::Stage3
            }
        ));
        Ok(())
    });
}

// Bullet 20: an invalid small-shape zero-extension tie, and in general a
// group whose committed cells differ from the logical table the PIOP
// proves. Each of the four groups (both Width14 groups and both
// Width4 groups, including the final `Budget` group and the below-floor
// shapes at this batch) is tampered in turn; the Equation (6.5) tie between
// the ledger claim and the committed physical polynomial fails at the
// multi-group opening, and only there. Runs the real PCS at the minimum
// batch.
#[test]
fn pcs_rejects_a_tampered_group_at_the_opening() {
    with_single_thread_pool("Falcon-512 group tamper", || {
        let batch = fixture(8);
        let verifier_setup =
            pcs::FalconVerifierSetup::<Falcon512>::new(8).map_err(|error| error.to_string())?;
        for role in ROLES {
            assert_group_tamper_rejected_at_opening(&batch, &verifier_setup, role)?;
        }
        Ok(())
    });
}

#[test]
fn production_pcs_accepts_arbitrary_live_counts_and_rejects_rebinding() {
    with_single_thread_pool("arbitrary live-count PCS binding", || {
        for records in [3_usize, 5, 11] {
            let batch = fixture(records);
            let setup = pcs::FalconAggregateSetup::<Falcon512>::new(records)
                .map_err(|error| error.to_string())?;
            assert_eq!(batch.live_rows, records);
            let (proof, _) = pcs::prove_prepared(&batch, PUBLIC_RECORDS, &setup)
                .map_err(|error| error.to_string())?;
            pcs::verify(&public(&batch), setup.verifier(), &proof)
                .map_err(|error| format!("honest {records}-record proof: {error}"))?;

            // Counts 3, 5, and 11 share their padded domains with the next
            // live count. The proof still binds the exact public N.
            let rebound_records = records + 1;
            let rebound_setup = pcs::FalconVerifierSetup::<Falcon512>::new(rebound_records)
                .map_err(|error| error.to_string())?;
            let mut rebound_public = public(&batch);
            rebound_public.records = rebound_records;
            let error = pcs::verify(&rebound_public, &rebound_setup, &proof)
                .expect_err("rebound live count");
            assert!(matches!(
                error,
                AggregationError::ProofRejected {
                    stage: RejectionStage::Grinding
                }
            ));
        }
        Ok(())
    });
}

/// The Falcon-1024 half of the suite: the conditions whose attack does not
/// depend on Falcon-512 constants, on the synthetic relation-positive
/// Falcon-1024 fixture. See the module documentation for the cases that
/// stay Falcon-512-only and why.
mod falcon_1024 {
    use super::*;
    use aerie_falcon_witness::DirectRow;
    use aerie_prover::Falcon1024;

    /// Four live records fill the minimum padded Falcon-1024 domain.
    const ROWS_1024: usize = 4;
    const DEGREE_1024: usize = 1024;

    fn fixture_1024(records: usize) -> FalconWitness {
        let rows = (0..records)
            .map(|seed| {
                DirectRow::<Falcon1024>::synthetic(seed as u64 + 0x0102_4263)
                    .expect("synthetic Falcon-1024 row")
            })
            .collect::<Vec<_>>();
        let mut batch = FalconWitness::from_rows(&rows);
        aerie_prover::pad_witness(&mut batch).expect("canonical padding");
        batch
    }

    fn honest_1024() -> &'static (FalconWitness, pcs::AggregateProof) {
        static HONEST: OnceLock<(FalconWitness, pcs::AggregateProof)> = OnceLock::new();
        HONEST.get_or_init(|| {
            with_single_thread_pool_value("honest Falcon-1024 PCS proof", || {
                let batch = fixture_1024(ROWS_1024);
                assert_eq!(batch.degree(), DEGREE_1024);
                let setup = setup_1024(batch.live_rows);
                let (proof, _) = pcs::prove_prepared(&batch, PUBLIC_RECORDS, &setup)
                    .expect("honest Falcon-1024 production PCS prover");
                assert_eq!(proof.commitments.len(), GROUP_COUNT);
                pcs::verify(&public(&batch), setup.verifier(), &proof)
                    .expect("honest proof verifies");
                (batch, proof)
            })
        })
    }

    fn assert_proof_rejected_1024(
        tamper: impl FnOnce(&mut pcs::AggregateProof),
        expected: ExpectedRejection,
        label: &str,
    ) {
        let (batch, proof) = honest_1024();
        let mut tampered = proof.clone();
        tamper(&mut tampered);
        let error = verify_proof(batch, &tampered).expect_err(label);
        assert_rejection(&error, expected, label);
    }

    // Finished-proof tampering across each stage and commitment group.
    #[test]
    fn rejects_finished_proof_tampers() {
        for index in 0..3 {
            assert_proof_rejected_1024(
                |proof| proof.stage1.projection_claims[index] += one(),
                ExpectedRejection::Rejected(RejectionStage::Stage1),
                &format!("projection terminal {index}"),
            );
            assert_proof_rejected_1024(
                |proof| proof.stage3.terminal_evaluations[index] += one(),
                ExpectedRejection::Rejected(RejectionStage::Stage3),
                &format!("stage-three terminal {index}"),
            );
        }
        let (_, honest_proof) = honest_1024();
        let (batch, _) = honest_1024();
        assert_reused_point_is_bound(batch, honest_proof);
        for round in [
            0,
            honest_proof.stage3.rounds.len() / 2,
            honest_proof.stage3.rounds.len() - 1,
        ] {
            for coefficient in 0..3 {
                assert_proof_rejected_1024(
                    |proof| proof.stage3.rounds[round][coefficient] += one(),
                    ExpectedRejection::GrindingOr(RejectionStage::Stage3),
                    &format!("stage-three round coefficient {round} {coefficient}"),
                );
            }
            for coefficient in 0..4 {
                assert_proof_rejected_1024(
                    |proof| proof.stage2.rounds[round][coefficient] += one(),
                    ExpectedRejection::GrindingOr(RejectionStage::Stage2),
                    &format!("stage-two round coefficient {round} {coefficient}"),
                );
            }
        }
        for source in 0..honest_proof.stage1.basis16_claims.len() {
            for child in 0..2 {
                assert_proof_rejected_1024(
                    |proof| proof.stage1.basis16_claims[source][child] += one(),
                    ExpectedRejection::Rejected(
                        if honest_proof.stage1.basis16_claims[source][1 - child].is_zero() {
                            RejectionStage::Grinding
                        } else {
                            RejectionStage::Stage1
                        },
                    ),
                    &format!("basis16 child {source} {child}"),
                );
            }
        }
        for source in 0..6 {
            assert_proof_rejected_1024(
                |proof| proof.stage2.folded_digits[source] += one(),
                ExpectedRejection::Rejected(RejectionStage::Stage2),
                &format!("folded range image {source}"),
            );
        }
        let rounds = honest_proof.stage1.rounds.len();
        for round in [0, rounds - 1] {
            assert_proof_rejected_1024(
                |proof| {
                    proof.stage1.rounds[round][0] += one();
                },
                ExpectedRejection::GrindingOr(RejectionStage::Stage1),
                &format!("stage-one round {round}"),
            );
        }
        for index in 0..honest_proof.stage3.terminal_evaluations.len() {
            assert_proof_rejected_1024(
                |proof| proof.stage3.terminal_evaluations[index] += one(),
                ExpectedRejection::Rejected(RejectionStage::Stage3),
                &format!("stage-three terminal {index}"),
            );
        }
        assert_proof_rejected_1024(
            |proof| {
                proof.stage1.projection_claims[0] += one();
                proof.stage1.projection_claims[2] += one();
            },
            ExpectedRejection::Rejected(RejectionStage::Stage1),
            "false residual claim preserving the ring difference",
        );
        assert_proof_rejected_1024(
            |proof| proof.stage2.folded_digits[0] += one(),
            ExpectedRejection::Rejected(RejectionStage::Stage2),
            "folded digit claim",
        );
        assert_proof_rejected_1024(
            |proof| proof.commitments.swap(0, 1),
            ExpectedRejection::Rejected(RejectionStage::Grinding),
            "S1 and slack swap",
        );
        assert_proof_rejected_1024(
            |proof| proof.commitments.swap(2, 3),
            ExpectedRejection::Rejected(RejectionStage::Grinding),
            "S2Sources and final Budget swap",
        );
        assert_proof_rejected_1024(
            |proof| {
                proof.commitments.pop();
            },
            ExpectedRejection::Malformed(MalformedProof::CommitmentShape),
            "missing final Budget",
        );
    }

    // Bullets 7 to 9 against the Falcon-1024 cap: `y` at `C_S` and
    // `C_S + 1`, `z_j` off by `q` and by `+-1`, and the `ctr_p` boundary.
    #[test]
    fn rejects_cheating_projection_outputs() {
        let batch = fixture_1024(ROWS_1024);
        assert_cheating_outputs_rejected_1024(&batch);
    }

    fn assert_cheating_outputs_rejected_1024(batch: &FalconWitness) {
        let degree = FalconDegree::Falcon1024;
        let cap = norm_cap(degree);
        let at_cap = y_with_energy(cap);
        assert_predicate_passing_outputs_rejected(
            batch,
            move |_, outputs| outputs.y = at_cap,
            "y energy C_S",
        );
        let above = y_with_energy(cap + 1);
        assert_witness_tamper_rejected_on(
            batch,
            |_| {},
            move |_, outputs| outputs.y = above,
            ExpectedRejection::Rejected(RejectionStage::Stage1),
            "y energy C_S + 1",
        );
        let q = BaseField::from_u64(FALCON_Q);
        let half = (FIELD_MODULUS - 1) / 2;
        let boundary = BaseField::from_u64((half / FALCON_Q + 1) * FALCON_Q);
        assert_predicate_passing_outputs_rejected(batch, |_, outputs| outputs.z[4] += q, "z_j + q");
        let cases: [(&str, &(dyn Fn(&mut ProjectionOutputs) + Send + Sync)); 3] = [
            ("z_j + 1", &|outputs| outputs.z[4] += BaseField::one()),
            ("z_j - 1", &|outputs| outputs.z[30] -= BaseField::one()),
            ("ctr_p boundary z_0", &|outputs| outputs.z[0] = boundary),
        ];
        for (label, tamper) in cases {
            assert_witness_tamper_rejected_on(
                batch,
                |_| {},
                |_, outputs| tamper(outputs),
                ExpectedRejection::Malformed(MalformedProof::ProjectionEncoding),
                label,
            );
        }
    }

    // Bullets 1 to 4 on the Falcon-1024 alphabets: the `s2` decoder digits
    // and the `a2`/`sigma` cells share the Falcon-512 alphabets; the
    // third budget digit is radix 16 (`[8, 8, 16]`).
    #[test]
    fn rejects_witness_cells_outside_their_alphabets() {
        let batch = fixture_1024(ROWS_1024);
        let cases: [(
            &str,
            ExpectedRejection,
            &(dyn Fn(&mut SourceTables) + Send + Sync),
        ); 10] = [
            ("a0 above 3", ExpectedRejection::Backend, &|tables| {
                tables.a0_mut()[5] = 4;
            }),
            ("a0 below -4", ExpectedRejection::Backend, &|tables| {
                tables.a0_mut()[5] = -5;
            }),
            ("a1 above 7", ExpectedRejection::Backend, &|tables| {
                tables.a1_mut()[7] = 8;
            }),
            ("a1 below -8", ExpectedRejection::Backend, &|tables| {
                tables.a1_mut()[7] = -9;
            }),
            ("budget u0 above 3", ExpectedRejection::Backend, &|tables| {
                tables.budget_mut(0)[3] = 4;
            }),
            ("budget u2 above 7", ExpectedRejection::Backend, &|tables| {
                tables.budget_mut(2)[2] = 8;
            }),
            (
                "budget u2 below -8",
                ExpectedRejection::Backend,
                &|tables| tables.budget_mut(2)[2] = -9,
            ),
            ("a2 above 7", ExpectedRejection::Backend, &|tables| {
                tables.a2_mut()[13] = 8;
            }),
            (
                "sign of 0",
                ExpectedRejection::Rejected(RejectionStage::Stage3),
                &|tables| tables.sigma_mut()[21] = 0,
            ),
            (
                "sign of 2",
                ExpectedRejection::Rejected(RejectionStage::Stage3),
                &|tables| tables.sigma_mut()[21] = 2,
            ),
        ];
        for (label, expected, tamper) in cases {
            assert_witness_tamper_rejected_on(&batch, tamper, |_, _| {}, expected, label);
        }
    }

    // Bullet 19: a nonzero semantic padded row (three live rows padded to
    // four) in the `S_1` witness and in the slack table.
    #[test]
    fn rejects_a_nonzero_semantic_padded_row() {
        let honest_batch = fixture_1024(3);
        with_single_thread_pool_value("Falcon-1024 padded-row control", || {
            let setup = pcs::FalconAggregateSetup::<Falcon1024>::new(3)
                .expect("Falcon-1024 three-record setup");
            let (proof, _) = pcs::prove_prepared(&honest_batch, PUBLIC_RECORDS, &setup)
                .expect("honest Falcon-1024 three-record prover");
            pcs::verify(&public(&honest_batch), setup.verifier(), &proof)
                .expect("honest Falcon-1024 three-record proof verifies");

            for (label, batch) in [
                ("nonzero padded s1 row", {
                    let mut batch = fixture_1024(3);
                    batch.s1[3 * DEGREE_1024 + 2] = 7;
                    batch
                }),
                ("nonzero padded slack row", {
                    let mut batch = fixture_1024(3);
                    batch.slack[3 * 4] = 5;
                    batch
                }),
            ] {
                assert!(
                    batch.validate_semantics().is_err(),
                    "{label}: invalid witness"
                );
                match pcs::prove_prepared(&batch, PUBLIC_RECORDS, &setup) {
                    Err(error) => assert!(
                        matches!(error, AggregationError::ProjectionRejected),
                        "Falcon-1024 {label}: preserve the projection rejection reason, got {error}"
                    ),
                    Ok((proof, _)) => {
                        match pcs::verify(&public(&batch), setup.verifier(), &proof) {
                            Err(error) => {
                                let stage = if label == "nonzero padded s1 row" {
                                    RejectionStage::Stage1
                                } else {
                                    RejectionStage::Stage3
                                };
                                assert_rejection(&error, ExpectedRejection::Rejected(stage), label);
                            }
                            Ok(_) => panic!(
                                "Falcon-1024 {label}: production PCS verifier accepted the witness"
                            ),
                        }
                    }
                }
            }
        });
    }

    // Bullet 20 and the PCS opening: a tampered cell in each of the four
    // groups (commitments that do not match the ledger claims, each
    // rejected at the multi-group opening), then one-bit flips of an honest
    // Falcon-1024 PCS frame in the first commitment payload, the middle of
    // the frame, and the last byte of the batched opening proof.
    #[test]
    fn pcs_rejects_group_and_opening_tampers() {
        with_single_thread_pool("Falcon-1024 PCS tamper", || {
            let batch = fixture_1024(ROWS_1024);
            let setup = pcs::FalconAggregateSetup::<Falcon1024>::new(ROWS_1024)
                .map_err(|error| error.to_string())?;
            let public = public(&batch);
            for role in ROLES {
                assert_group_tamper_rejected_at_opening(&batch, setup.verifier(), role)?;
            }

            let (proof, _) = pcs::prove_prepared(&batch, PUBLIC_RECORDS, &setup)
                .map_err(|error| error.to_string())?;
            pcs::verify(&public, setup.verifier(), &proof)
                .map_err(|error| format!("honest Falcon-1024 PCS proof: {error}"))?;
            let salts = vec![[0_u8; wire::TRANSPORT_SALT_BYTES]; ROWS_1024];
            let bytes = wire::encode_pcs_proof(FalconDegree::Falcon1024, ROWS_1024, &salts, &proof)
                .map_err(|error| error.to_string())?;
            // No length prefixes: the first commitment payload starts
            // right after the salts.
            let first_commitment = wire::HEADER_BYTES + ROWS_1024 * wire::TRANSPORT_SALT_BYTES;
            for (offset, label) in [
                (first_commitment + 16, "commitment region"),
                (bytes.len() - 1, "opening proof tail"),
                (bytes.len() / 2, "frame middle"),
            ] {
                let mut tampered = bytes.clone();
                tampered[offset] ^= 1;
                // A single flipped bit in these positions keeps the frame
                // canonical, so the rejection is the verifier's.
                let decoded = wire::decode_pcs_proof(FalconDegree::Falcon1024, &tampered)
                    .map_err(|error| format!("{label}: flipped frame must decode: {error}"))?;
                let error =
                    pcs::verify(&public, setup.verifier(), &decoded.proof).expect_err(label);
                let expected = if label == "commitment region" {
                    RejectionStage::Grinding
                } else {
                    RejectionStage::PcsOpening
                };
                assert!(
                    matches!(error, AggregationError::ProofRejected { stage } if stage == expected),
                    "{label}: {error:?}"
                );
            }
            Ok(())
        });
    }
}
