//! Clear-projection timings: core derivation plus the forward projection,
//! the stage one prover, and the stage one verifier (predicates, output
//! absorption, public-weight evaluation and the sumcheck replay).
//!
//! `cargo bench -p aerie-falcon --features bench-internals --bench projection -- [records] [iterations] [512|1024]`

use std::{hint::black_box, time::Instant};

use aerie_falcon::manifest::{CommitmentManifest, GroupRole};
use aerie_falcon::source_tables::SourceTables;
use aerie_falcon::transcript::{ProtocolTranscript, VerifierProtocolTranscript};
use aerie_falcon::{projection, stage1, FalconDegree, FalconWitness};
use aerie_falcon_native::{Falcon1024, Falcon512};
use aerie_falcon_statement::DirectRow;

/// Distinct synthetic rows; larger batches repeat them. The projection's
/// cost does not depend on the row values.
const DISTINCT_ROWS: usize = 64;
const INSTANCE_RECORD_BYTES: usize = 398;
const DUMMY_COMMITMENT_BYTES: [u8; 128] = [0; 128];

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        f64::midpoint(samples[middle - 1], samples[middle])
    } else {
        samples[middle]
    }
}

fn batch(degree: FalconDegree, records: usize) -> FalconWitness {
    let distinct = DISTINCT_ROWS.min(records);
    match degree {
        FalconDegree::Falcon512 => {
            let rows = (0..distinct)
                .map(|seed| DirectRow::<Falcon512>::synthetic(seed as u64).expect("row"))
                .collect::<Vec<_>>();
            let tiled = (0..records).map(|index| rows[index % distinct].clone());
            FalconWitness::from_rows(&tiled.collect::<Vec<_>>())
        }
        FalconDegree::Falcon1024 => {
            let rows = (0..distinct)
                .map(|seed| DirectRow::<Falcon1024>::synthetic(seed as u64).expect("row"))
                .collect::<Vec<_>>();
            let tiled = (0..records).map(|index| rows[index % distinct].clone());
            FalconWitness::from_rows(&tiled.collect::<Vec<_>>())
        }
    }
}

/// Build a shape-valid synthetic record for an isolated stage-one benchmark.
fn benchmark_instance(
    degree: FalconDegree,
    records: usize,
) -> (CommitmentManifest, [u8; INSTANCE_RECORD_BYTES]) {
    let manifest = CommitmentManifest::for_batch_of(degree, records).expect("manifest");
    let binding = manifest.binding_bytes();
    let mut record = [0; INSTANCE_RECORD_BYTES];
    record[..27].copy_from_slice(&binding[..27]);
    record[60..302].copy_from_slice(&binding[27..]);
    (manifest, record)
}

/// Prepare the prover PIOP stream at the stage-one boundary.
///
/// This isolated benchmark uses zero commitment payloads. The verifier
/// rebuilds the same prefix with its own native transcript state.
fn prover_transcript(
    manifest: &CommitmentManifest,
    record: &[u8; INSTANCE_RECORD_BYTES],
) -> (ProtocolTranscript, [u8; 32]) {
    let mut transcript = ProtocolTranscript::new(manifest, record).expect("prover transcript");
    for role in [
        GroupRole::S1,
        GroupRole::Slack,
        GroupRole::S2Sources,
        GroupRole::Budget,
    ] {
        transcript
            .absorb_group_commitment(role, &DUMMY_COMMITMENT_BYTES)
            .expect("commitment prefix");
    }
    let seed = transcript
        .derive_projection_seed()
        .expect("projection seed");
    (transcript.finish().expect("stage-one prefix"), seed)
}

fn verifier_transcript(
    manifest: &CommitmentManifest,
    record: &[u8; INSTANCE_RECORD_BYTES],
    work: &[u8],
) -> (VerifierProtocolTranscript, [u8; 32]) {
    let mut transcript =
        VerifierProtocolTranscript::new(manifest, record, work).expect("verifier transcript");
    for role in [
        GroupRole::S1,
        GroupRole::Slack,
        GroupRole::S2Sources,
        GroupRole::Budget,
    ] {
        transcript
            .absorb_group_commitment(role, &DUMMY_COMMITMENT_BYTES)
            .expect("commitment prefix");
    }
    let seed = transcript
        .derive_projection_seed()
        .expect("projection seed");
    (transcript.finish().expect("stage-one prefix"), seed)
}

fn main() {
    let mut arguments = std::env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with("--"));
    let records: usize = arguments.next().map_or(16_384, |value| {
        value.parse().expect("records must be an integer")
    });
    let iterations: usize = arguments.next().map_or(5, |value| {
        value.parse().expect("iterations must be an integer")
    });
    let degree = arguments.next().map_or(FalconDegree::Falcon512, |value| {
        FalconDegree::from_degree(value.parse().expect("degree must be 512 or 1024"))
            .expect("degree must be 512 or 1024")
    });
    assert!(records.is_power_of_two());
    assert!(iterations > 0);
    let batch = batch(degree, records);
    let tables = SourceTables::build(&batch).expect("source tables");

    let mut project_samples = Vec::with_capacity(iterations);
    let mut prove_samples = Vec::with_capacity(iterations);
    let mut verify_samples = Vec::with_capacity(iterations);
    let (manifest, record) = benchmark_instance(degree, records);
    let (_, seed) = prover_transcript(&manifest, &record);
    for _ in 0..iterations {
        let (mut prover, prover_seed) = prover_transcript(&manifest, &record);
        assert_eq!(seed, prover_seed, "projection seed for prover prefix");
        let started = Instant::now();
        let projected = projection::project(black_box(&batch), &seed).expect("projection");
        project_samples.push(started.elapsed().as_secs_f64() * 1_000.0);

        let started = Instant::now();
        let (proof, _) =
            stage1::prove(&batch, &tables, &projected, &mut prover).expect("stage one prover");
        prove_samples.push(started.elapsed().as_secs_f64() * 1_000.0);

        let (mut verifier, verifier_seed) =
            verifier_transcript(&manifest, &record, prover.work_bytes());
        assert_eq!(seed, verifier_seed, "prover and verifier projection seed");
        let started = Instant::now();
        black_box(
            stage1::verify(
                degree,
                records,
                records,
                &seed,
                projected.outputs(),
                &proof,
                &mut verifier,
            )
            .expect("stage one verifier"),
        );
        verify_samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }

    println!("degree\trecords\titerations\tproject_ms\tprove_stage1_ms\tverify_stage1_ms");
    println!(
        "{}\t{records}\t{iterations}\t{:.3}\t{:.3}\t{:.3}",
        degree.degree(),
        median_ms(&mut project_samples),
        median_ms(&mut prove_samples),
        median_ms(&mut verify_samples),
    );
}
