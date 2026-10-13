//! Synthetic microbenchmark for verifier-local Falcon `HashToPoint` only.
//!
//! This does not generate or verify signatures and is not an aggregate-proof
//! benchmark. It compares the pre-optimization two-byte SHAKE reader, the
//! buffered scalar public API, and the backend-width batch API. `MODE=verifier`
//! measures the composition verifier transport uses: parallel hashed-key
//! derivation, then streamed prehashed targets over the canonical Rayon grain.
use std::{array, env, hint::black_box, time::Instant};

use aerie_falcon_native::{
    params::{FALCON_DEGREE, FALCON_Q},
    preprocess::{
        hash_to_point_parallel_grain, hash_to_point_prehashed_batch_visit_of,
        hash_to_point_public_raw, hash_to_point_public_raw_batch,
        hashed_verifying_keys_batch_parallel, HashToPointInput, PrehashedHashToPointBatch,
    },
    source::{CanonicalPublicKeyBytes, FALCON_512_PUBLIC_KEY_BYTES, FALCON_512_SALT_BYTES},
    Falcon512,
};
use aerie_keccak::{Backend, BackendClosure};
use hybrid_array::typenum::Unsigned;
use rayon::prelude::*;
use shake::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

const ACCEPT_BOUND: u16 = 61_445;
const COMPARISON_GRAIN: usize = 64;

struct Width<'a>(&'a mut usize);

impl BackendClosure for Width<'_> {
    fn call_once<B: Backend>(self) {
        *self.0 = B::ParSize1600::USIZE;
    }
}

fn legacy_hash_to_point(
    public_key: &CanonicalPublicKeyBytes<Falcon512>,
    salt: &[u8; FALCON_512_SALT_BYTES],
    message: &[u8],
) -> [i16; FALCON_DEGREE] {
    let mut hasher = Shake256::default();
    hasher.update(public_key.as_bytes());
    let mut reader = hasher.finalize_xof();
    let mut hpk = [0_u8; 64];
    reader.read(&mut hpk);

    let mut hasher = Shake256::default();
    hasher.update(salt);
    hasher.update(&hpk);
    hasher.update(&[0, 0]);
    hasher.update(message);
    let mut reader = hasher.finalize_xof();
    let mut output = [0_i16; FALCON_DEGREE];
    let mut count = 0;
    while count < FALCON_DEGREE {
        let mut encoded = [0_u8; 2];
        reader.read(&mut encoded);
        let value = u16::from_be_bytes(encoded);
        if value < ACCEPT_BOUND {
            let residue = value % FALCON_Q as u16;
            output[count] = if residue > (FALCON_Q as u16 - 1) / 2 {
                residue.cast_signed() - FALCON_Q as i16
            } else {
                residue.cast_signed()
            };
            count += 1;
        }
    }
    output
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
fn verifier_first_coefficient_sum(
    public_keys: &[CanonicalPublicKeyBytes<Falcon512>],
    salts: &[[u8; FALCON_512_SALT_BYTES]],
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
            hash_to_point_prehashed_batch_visit_of::<Falcon512, _>(
                batch.slice(start..end),
                |_, row| sum += i64::from(row[0]),
            );
            sum
        })
        .sum()
}

fn measure_verifier(
    samples: usize,
    pool: &rayon::ThreadPool,
    public_keys: &[CanonicalPublicKeyBytes<Falcon512>],
    salts: &[[u8; FALCON_512_SALT_BYTES]],
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

    let mut encoded = [0_u8; FALCON_512_PUBLIC_KEY_BYTES];
    encoded[0] = 0x09;
    let public_key = CanonicalPublicKeyBytes::<Falcon512>::parse(&encoded).expect("canonical key");
    let salts = (0..records)
        .map(|record| {
            array::from_fn(|index| (record.wrapping_mul(131).wrapping_add(index) & 0xff) as u8)
        })
        .collect::<Vec<[u8; FALCON_512_SALT_BYTES]>>();
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
    let expected = legacy_hash_to_point(&public_key, &salts[0], &messages[0]);
    assert_eq!(
        expected,
        hash_to_point_public_raw(&public_key, &salts[0], &messages[0])
    );
    assert_eq!(expected, hash_to_point_public_raw_batch(&inputs[..1])[0]);
    for (actual, input) in hash_to_point_public_raw_batch(&inputs[..inputs.len().min(257)])
        .iter()
        .zip(&inputs)
    {
        assert_eq!(
            actual,
            &legacy_hash_to_point(input.public_key(), input.salt(), input.message())
        );
    }

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
                    legacy_hash_to_point(input.public_key(), input.salt(), input.message())[0],
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
    let (legacy_ms, legacy_sink) = measure(samples, || {
        inputs
            .iter()
            .map(|input| {
                i64::from(
                    legacy_hash_to_point(input.public_key(), input.salt(), input.message())[0],
                )
            })
            .sum()
    });
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
    let (legacy_rayon_ms, legacy_rayon_sink) = measure(samples, || {
        pool.install(|| {
            inputs
                .par_iter()
                .map(|input| {
                    i64::from(
                        legacy_hash_to_point(input.public_key(), input.salt(), input.message())[0],
                    )
                })
                .sum()
        })
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
    assert_eq!(legacy_sink, buffered_sink);
    assert_eq!(legacy_sink, batch_sink);
    assert_eq!(legacy_sink, legacy_rayon_sink);
    assert_eq!(legacy_sink, buffered_rayon_sink);
    assert_eq!(legacy_sink, rayon_sink);
    assert_eq!(legacy_sink, verifier_sink);

    println!(
        "arch={} backend={} width={width} records={records} threads={threads} samples={samples}",
        env::consts::ARCH,
        aerie_keccak::backend_name(),
    );
    println!("legacy_two_byte_ms={legacy_ms:.3}");
    println!(
        "buffered_scalar_ms={buffered_ms:.3} speedup={:.3}",
        legacy_ms / buffered_ms
    );
    println!(
        "backend_batch_ms={batch_ms:.3} speedup={:.3}",
        legacy_ms / batch_ms
    );
    println!(
        "legacy_rayon_ms={legacy_rayon_ms:.3} speedup={:.3}",
        legacy_ms / legacy_rayon_ms
    );
    println!(
        "buffered_rayon_ms={buffered_rayon_ms:.3} speedup={:.3}",
        legacy_rayon_ms / buffered_rayon_ms
    );
    println!(
        "rayon_batch_ms={rayon_ms:.3} speedup_vs_legacy_rayon={:.3}",
        legacy_rayon_ms / rayon_ms
    );
    println!(
        "verifier_prehashed_ms={verifier_ms:.3} speedup_vs_legacy_rayon={:.3}",
        legacy_rayon_ms / verifier_ms
    );
    println!("sink={legacy_sink}");
}
