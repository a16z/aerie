use aerie_falcon_statement::{DirectParams, DirectRow};
use jolt_field::{One as _, Ring as _};

use crate::protocol::manifest::{CommitmentManifest, GroupRole};
use crate::protocol::projection;
use crate::protocol::transcript::{
    ChallengeSource, ProtocolTranscript, VerifierProtocolTranscript,
};
use crate::protocol::{source_tables::SourceTables, FalconDegree, FalconWitness};
use crate::substrate::{BaseField as F, OpeningField as E};
use aerie_falcon_native::{Falcon1024, Falcon512};

use super::*;

fn batch<P: DirectParams>(live_rows: usize) -> FalconWitness {
    let rows = (0..live_rows)
        .map(|index| DirectRow::<P>::synthetic(0x5eed_0000 + index as u64).expect("row"))
        .collect::<Vec<_>>();
    {
        let mut batch = FalconWitness::from_rows(&rows);
        crate::protocol::pad_witness(&mut batch).unwrap();
        batch
    }
}

fn roundtrip(degree: FalconDegree, batch: &FalconWitness) {
    let tables = SourceTables::build(batch).expect("source tables");
    let (mut prover, _, seed) = transcript_pair(degree, batch.live_rows);
    let projected = projection::project(batch, &seed).expect("projection");
    let (proof, output) = prove(batch, &tables, &projected, &mut prover).expect("stage one prove");
    let stage2_gammas = if degree == FalconDegree::Falcon512 {
        8
    } else {
        9
    };
    let prover_stage2_gammas = prover
        .challenge_extensions("aerie/falcon/v1/stage2/gamma", stage2_gammas)
        .unwrap();
    let mut verifier = verifier_for_work(degree, batch.live_rows, prover.work_bytes());

    assert_eq!(
        proof.rounds.len(),
        degree.logn() + batch.padded_rows.trailing_zeros() as usize
    );
    assert_eq!(
        proof.basis16_claims.len(),
        2 + usize::from(degree == FalconDegree::Falcon1024)
    );

    let verified = verify(
        degree,
        batch.padded_rows,
        batch.live_rows,
        &seed,
        projected.outputs(),
        &proof,
        &mut verifier,
    )
    .expect("stage one verify");
    assert_eq!(verified, output);
    assert_eq!(
        prover_stage2_gammas,
        verifier
            .challenge_extensions("aerie/falcon/v1/stage2/gamma", stage2_gammas)
            .unwrap()
    );

    let (_tau, point) = replay_points(
        degree,
        batch.live_rows,
        batch.padded_rows,
        projected.outputs(),
        &proof,
    );
    assert_eq!(point, output.point);

    let decode = |source: crate::protocol::kernels::DigitSource<'_>| {
        let mut digits = vec![0; source.len()];
        source.decode_into(0, &mut digits);
        digits
    };
    let basis_sources: Vec<Vec<i8>> = if degree == FalconDegree::Falcon512 {
        vec![decode(tables.a1_view()), decode(tables.a2_view())]
    } else {
        vec![
            decode(tables.a1_view()),
            decode(tables.a2_view()),
            tables.budget(2).to_vec(),
        ]
    };
    for (index, source) in basis_sources.iter().enumerate() {
        let source_point = if index == 2 {
            &point[degree.logn()..]
        } else {
            &point[..]
        };
        let weights = super::super::algebra::eq_table(source_point);
        for lane in 0..2 {
            let expected = weights
                .iter()
                .zip(source)
                .fold(E::zero(), |sum, (&weight, &digit)| {
                    sum + weight * basis_factor(digit)[lane]
                });
            assert_eq!(proof.basis16_claims[index][lane], expected);
        }
    }

    let projection_weights = super::super::algebra::eq_table(&output.projection_point);
    let expected_s1 = projection_weights
        .iter()
        .zip(&batch.s1)
        .fold(E::zero(), |sum, (&weight, &value)| {
            sum + weight * E::from_i64(i64::from(value))
        });
    assert_eq!(proof.projection_claims[0], expected_s1);

    let slack_weights = super::super::algebra::eq_table(&output.slack_projection_point);
    let expected_slack = slack_weights
        .iter()
        .zip(&batch.slack)
        .fold(E::zero(), |sum, (&weight, &value)| {
            sum + weight * E::from_i64(i64::from(value))
        });
    assert_eq!(proof.projection_claims[1], expected_slack);

    let mut delta = Vec::with_capacity(batch.padded_rows * degree.degree());
    for record in 0..batch.padded_rows {
        for coefficient in 0..degree.degree() {
            delta.push(match &projected.delta.wide[record] {
                Some(values) => values[coefficient],
                None => i64::from(projected.delta.narrow[record * degree.degree() + coefficient]),
            });
        }
    }
    let expected_delta = projection_weights
        .iter()
        .zip(delta)
        .fold(E::zero(), |sum, (&weight, value)| {
            sum + weight * E::from_i64(value)
        });
    assert_eq!(proof.projection_claims[2], expected_delta);
}

fn basis_factor(digit: i8) -> [E; 2] {
    let digit = E::from_i64(i64::from(digit));
    let y = digit * (digit + E::one());
    [
        y * (y - E::from_i64(2)) * (y - E::from_i64(6)) * (y - E::from_i64(12)),
        (y - E::from_i64(20))
            * (y - E::from_i64(30))
            * (y - E::from_i64(42))
            * (y - E::from_i64(56)),
    ]
}

fn replay_points(
    degree: FalconDegree,
    live_rows: usize,
    records: usize,
    outputs: &projection::ProjectionOutputs,
    proof: &Stage1Proof,
) -> (Vec<E>, Vec<E>) {
    let (mut transcript, _, _) = transcript_pair(degree, live_rows);
    projection::absorb_outputs(outputs, &mut transcript).expect("output absorption");
    let basis_count = 2 + usize::from(degree == FalconDegree::Falcon1024);
    let _mu = transcript
        .challenge_extensions("aerie/falcon/v1/mu", 8)
        .expect("mu");
    let rounds = degree.logn() + records.trailing_zeros() as usize;
    let tau = transcript
        .challenge_extensions("aerie/falcon/v1/tau", rounds)
        .expect("tau");
    let _gammas = transcript
        .challenge_extensions("aerie/falcon/v1/stage1/gamma", basis_count + 1)
        .expect("gammas");
    let mut point = Vec::with_capacity(rounds);
    for (round, message) in proof.rounds.iter().enumerate() {
        point.push(
            super::super::stage::append_round(&mut transcript, "stage1", round, *message)
                .expect("stage-one challenge"),
        );
    }
    (tau, point)
}

fn transcript_pair(
    degree: FalconDegree,
    live_rows: usize,
) -> (ProtocolTranscript, VerifierProtocolTranscript, [u8; 32]) {
    let manifest = CommitmentManifest::for_batch_of(degree, live_rows).expect("manifest");
    let record = test_record(&manifest);
    let mut prover = ProtocolTranscript::new(&manifest, &record).expect("prover transcript");
    let mut verifier =
        VerifierProtocolTranscript::new(&manifest, &record, &[]).expect("verifier transcript");
    let roles = [
        GroupRole::S1,
        GroupRole::Slack,
        GroupRole::S2Sources,
        GroupRole::Budget,
    ];
    for (role_index, role) in roles.into_iter().enumerate() {
        let mut payload = [0; 128];
        for atom in 0..16 {
            payload[8 * atom] = (role_index + 1) as u8;
        }
        prover.absorb_group_commitment(role, &payload).unwrap();
        verifier.absorb_group_commitment(role, &payload).unwrap();
    }
    let seed = prover.derive_projection_seed().unwrap();
    assert_eq!(seed, verifier.derive_projection_seed().unwrap());
    (prover.finish().unwrap(), verifier.finish().unwrap(), seed)
}

fn verifier_for_work(
    degree: FalconDegree,
    live_rows: usize,
    work: &[u8],
) -> VerifierProtocolTranscript {
    let manifest = CommitmentManifest::for_batch_of(degree, live_rows).expect("manifest");
    let record = test_record(&manifest);
    let mut verifier =
        VerifierProtocolTranscript::new(&manifest, &record, work).expect("verifier transcript");
    for (role_index, role) in [
        GroupRole::S1,
        GroupRole::Slack,
        GroupRole::S2Sources,
        GroupRole::Budget,
    ]
    .into_iter()
    .enumerate()
    {
        let mut payload = [0; 128];
        for atom in 0..16 {
            payload[8 * atom] = (role_index + 1) as u8;
        }
        verifier.absorb_group_commitment(role, &payload).unwrap();
    }
    verifier.derive_projection_seed().unwrap();
    verifier.finish().unwrap()
}

fn test_record(manifest: &CommitmentManifest) -> [u8; 398] {
    let binding = manifest.binding_bytes();
    let mut record = [0; 398];
    record[..27].copy_from_slice(&binding[..27]);
    record[60..302].copy_from_slice(&binding[27..]);
    record
}

#[test]
fn verify_rejects_oversized_live_rows() {
    let outputs = ProjectionOutputs {
        y: [F::zero(); projection::NORM_TERMINAL_ROWS],
        z: [F::zero(); projection::RESIDUAL_WIDTH],
    };
    let proof = Stage1Proof {
        rounds: Vec::new(),
        basis16_claims: Vec::new(),
        projection_claims: [E::zero(); 3],
    };
    let (_, mut transcript, _) = transcript_pair(FalconDegree::Falcon512, 4);
    let result = verify(
        FalconDegree::Falcon512,
        1,
        usize::MAX,
        &[0; 32],
        &outputs,
        &proof,
        &mut transcript,
    );
    assert!(matches!(result, Err(AkitaError::InvalidInput(message))
        if message == "stage-one batch shape is not canonically padded"));
}

#[test]
fn batches_basis16_sources_and_projection_for_both_degrees() {
    roundtrip(FalconDegree::Falcon512, &batch::<Falcon512>(5));
    roundtrip(FalconDegree::Falcon1024, &batch::<Falcon1024>(3));
}

#[test]
fn basis_children_and_projection_claims_match_direct_mles_at_rest_boundaries() {
    // R = 0 at the minimum admitted N*, and R = 2 at a larger shape, for
    // both parameter sets. `roundtrip` compares every child and committed
    // source claim to a direct Boolean-cube MLE.
    roundtrip(FalconDegree::Falcon512, &batch::<Falcon512>(4));
    roundtrip(FalconDegree::Falcon512, &batch::<Falcon512>(15));
    roundtrip(FalconDegree::Falcon1024, &batch::<Falcon1024>(2));
    roundtrip(FalconDegree::Falcon1024, &batch::<Falcon1024>(7));
}

#[test]
fn projection_points_follow_the_two_local_variable_orders() {
    for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
        let low_bits = 12 - (degree.logn() + 1);
        for rest_bits in [0, 1, 3] {
            let records: usize = 1 << (low_bits + rest_bits);
            let rounds = degree.logn() + records.trailing_zeros() as usize;
            let stage_point = (0..rounds)
                .map(|coordinate| E::from_u64((coordinate + 1) as u64))
                .collect::<Vec<_>>();
            let (coefficient, slack) = reindex_projection_points(degree, &stage_point);
            let inactive = degree.logn() - 2;

            let mut expected_coefficient =
                stage_point[rest_bits..rest_bits + degree.logn()].to_vec();
            expected_coefficient.extend_from_slice(&stage_point[rest_bits + degree.logn()..]);
            expected_coefficient.extend_from_slice(&stage_point[..rest_bits]);
            assert_eq!(coefficient, expected_coefficient);

            let active = &stage_point[inactive..];
            let mut expected_slack = active[rest_bits..rest_bits + 2].to_vec();
            expected_slack.extend_from_slice(&active[rest_bits + 2..]);
            expected_slack.extend_from_slice(&active[..rest_bits]);
            assert_eq!(slack, expected_slack);
            assert_ne!(slack, coefficient);
        }
    }
}

#[test]
fn projection_lambda_is_the_eight_coordinate_monomial_batch() {
    let (mut actual_transcript, _, _) = transcript_pair(FalconDegree::Falcon512, 4);
    let (mut reference_transcript, _, _) = transcript_pair(FalconDegree::Falcon512, 4);
    let outputs = projection::ProjectionOutputs {
        y: [F::zero(); projection::NORM_TERMINAL_ROWS],
        z: [F::zero(); projection::RESIDUAL_WIDTH],
    };
    projection::absorb_outputs(&outputs, &mut actual_transcript).unwrap();
    projection::absorb_outputs(&outputs, &mut reference_transcript).unwrap();
    let mu = reference_transcript
        .challenge_extensions("aerie/falcon/v1/mu", 8)
        .unwrap();
    let actual = super::sample_lambda(&mut actual_transcript).unwrap();

    assert_eq!(actual.len(), OUTPUT_COUNT);
    assert_eq!(actual[0], E::one());
    for (output, &weight) in actual.iter().enumerate() {
        let expected = (0..8).fold(E::one(), |product, bit| {
            if (output >> bit) & 1 == 1 {
                product * mu[bit]
            } else {
                product
            }
        });
        assert_eq!(weight, expected, "projection output index {output}");
    }
    assert_eq!(
        actual_transcript
            .challenge_extensions("aerie/falcon/v1/tau", 12)
            .unwrap(),
        reference_transcript
            .challenge_extensions("aerie/falcon/v1/tau", 12)
            .unwrap()
    );
}
