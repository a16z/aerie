//! Synthetic microbenchmark for verifier-local Falcon `HashToPoint` only.
//!
//! This does not generate or verify signatures and is not an aggregate-proof
//! benchmark. It measures the buffered scalar public API and the
//! backend-width batch API. `MODE=verifier`
//! measures the composition verifier transport uses: parallel hashed-key
//! derivation, then streamed prehashed targets over the canonical Rayon grain.
use std::{array, env, hint::black_box, time::Instant};

use aerie_falcon_native::{
    params::FALCON_SALT_BYTES,
    preprocess::{
        hash_to_point_parallel_grain, hash_to_point_prehashed_batch_visit_of,
        hash_to_point_public_raw, hash_to_point_public_raw_batch,
        hashed_verifying_keys_batch_parallel, HashToPointInput, PrehashedHashToPointBatch,
    },
    source::CanonicalPublicKeyBytes,
    Falcon512, FalconParams,
};
use aerie_keccak::{Backend, BackendClosure};
use hybrid_array::typenum::Unsigned;
use rayon::prelude::*;
const COMPARISON_GRAIN: usize = 64;

struct Width<'a>(&'a mut usize);

impl BackendClosure for Width<'_> {
    fn call_once<B: Backend>(self) {
        *self.0 = B::ParSize1600::USIZE;
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn measure(samples: usize, mut operation: impl FnMut() -> i64) -> (f64, i64) {
    let mut times = Vec::with_capacity(samples);
    let mut sink = 0;
    for _ in 0..samples {
        let start = Instant::now();
        sink ^= black_box(operation());
        times.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    (median(times), sink)
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Scenario {
    Compare,
    Verifier,
}

impl Scenario {
    fn from_env() -> Self {
        match env::var("MODE").as_deref() {
            Ok("verifier") => Self::Verifier,
            Ok("compare") | Err(_) => Self::Compare,
            Ok(mode) => panic!("MODE must be compare or verifier, got {mode}"),
        }
    }
}

/// Sum of every target's first coefficient, derived as verifier transport
/// derives it: hash every verifying key once, then stream the targets from the
/// prehashed source tuple in canonical-grain chunks.
fn verifier_first_coefficient_sum<P: FalconParams>(
    public_keys: &[CanonicalPublicKeyBytes<P>],
    salts: &[[u8; FALCON_SALT_BYTES]],
    messages: &[Vec<u8>],
) -> i64 {
    let hashed_keys = hashed_verifying_keys_batch_parallel(public_keys);
    let batch =
        PrehashedHashToPointBatch::new(&hashed_keys, salts, messages).expect("aligned batch");
    let grain = hash_to_point_parallel_grain(batch.len());
    (0..batch.len())
        .into_par_iter()
        .step_by(grain)
        .map(|start| {
            let mut sum = 0_i64;
            let end = (start + grain).min(batch.len());
            hash_to_point_prehashed_batch_visit_of::<P, _>(batch.slice(start..end), |_, row| {
                sum += i64::from(row.as_ref()[0]);
            });
            sum
        })
        .sum()
}

fn measure_verifier<P: FalconParams>(
    samples: usize,
    pool: &rayon::ThreadPool,
    public_keys: &[CanonicalPublicKeyBytes<P>],
    salts: &[[u8; FALCON_SALT_BYTES]],
    messages: &[Vec<u8>],
) -> (f64, i64) {
    measure(samples, || {
        pool.install(|| verifier_first_coefficient_sum(public_keys, salts, messages))
    })
}

#[allow(clippy::too_many_lines)]
fn main() {
    let records: usize = env::var("RECORDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16_384);
    let samples = env::var("SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(7);
    let threads = env::var("THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let scenario = Scenario::from_env();

    let mut encoded = [0_u8; Falcon512::PUBLIC_KEY_BYTES];
    encoded[0] = Falcon512::PUBLIC_KEY_HEADER;
    let public_key = CanonicalPublicKeyBytes::<Falcon512>::parse(&encoded).expect("canonical key");
    let salts = (0..records)
        .map(|record| {
            array::from_fn(|index| (record.wrapping_mul(131).wrapping_add(index) & 0xff) as u8)
        })
        .collect::<Vec<[u8; FALCON_SALT_BYTES]>>();
    let messages = (0..records)
        .map(|record| format!("Aerie Falcon HashToPoint benchmark record {record}").into_bytes())
        .collect::<Vec<_>>();
    let inputs = salts
        .iter()
        .zip(&messages)
        .map(|(salt, message)| HashToPointInput::new(&public_key, salt, message))
        .collect::<Vec<_>>();

    let mut width = 0;
    aerie_keccak::with_backend(Width(&mut width));
    let public_keys = vec![public_key.clone(); records];
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    if scenario == Scenario::Verifier {
        let (milliseconds, sink) =
            measure_verifier(samples, &pool, &public_keys, &salts, &messages);
        let expected_sum = inputs
            .iter()
            .map(|input| {
                i64::from(
                    hash_to_point_public_raw(input.public_key(), input.salt(), input.message())[0],
                )
            })
            .sum::<i64>();
        // `measure` XORs the per-sample sums, so an odd sample count leaves one.
        assert_eq!(if samples % 2 == 1 { expected_sum } else { 0 }, sink);
        println!(
            "arch={} backend={} width={width} records={records} threads={threads} samples={samples}",
            env::consts::ARCH,
            aerie_keccak::backend_name(),
        );
        println!("verifier_prehashed_ms={milliseconds:.3}");
        return;
    }
    let (buffered_ms, buffered_sink) = measure(samples, || {
        inputs
            .iter()
            .map(|input| {
                i64::from(
                    hash_to_point_public_raw(input.public_key(), input.salt(), input.message())[0],
                )
            })
            .sum()
    });
    let (batch_ms, batch_sink) = measure(samples, || {
        hash_to_point_public_raw_batch(&inputs)
            .iter()
            .map(|target| i64::from(target[0]))
            .sum()
    });
    let (buffered_rayon_ms, buffered_rayon_sink) = measure(samples, || {
        pool.install(|| {
            inputs
                .par_iter()
                .map(|input| {
                    i64::from(
                        hash_to_point_public_raw(input.public_key(), input.salt(), input.message())
                            [0],
                    )
                })
                .sum()
        })
    });
    let (rayon_ms, rayon_sink) = measure(samples, || {
        pool.install(|| {
            inputs
                .par_chunks(COMPARISON_GRAIN)
                .map(hash_to_point_public_raw_batch)
                .map(|targets| {
                    targets
                        .iter()
                        .map(|target| i64::from(target[0]))
                        .sum::<i64>()
                })
                .sum()
        })
    });
    let (verifier_ms, verifier_sink) =
        measure_verifier(samples, &pool, &public_keys, &salts, &messages);
    assert_eq!(buffered_sink, batch_sink);
    assert_eq!(buffered_sink, buffered_rayon_sink);
    assert_eq!(buffered_sink, rayon_sink);
    assert_eq!(buffered_sink, verifier_sink);

    println!(
        "arch={} backend={} width={width} records={records} threads={threads} samples={samples}",
        env::consts::ARCH,
        aerie_keccak::backend_name(),
    );
    println!("buffered_scalar_ms={buffered_ms:.3}");
    println!("backend_batch_ms={batch_ms:.3}");
    println!("buffered_rayon_ms={buffered_rayon_ms:.3}");
    println!("rayon_batch_ms={rayon_ms:.3}");
    println!("verifier_prehashed_ms={verifier_ms:.3}");
    println!("sink={buffered_sink}");
}
