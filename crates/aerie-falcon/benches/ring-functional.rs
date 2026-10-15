use aerie_falcon_native::{Falcon512, FalconParams};
use std::{hint::black_box, time::Instant};

use aerie_falcon::{
    ring::{functional_evaluation, functional_evaluation_canonical_for_bench},
    OpeningField as E,
};
use aerie_falcon_native::params::FALCON_Q;
use aerie_falcon_native::source::CanonicalPublicKeyBytes;
use aerie_simd::SmallSignedDotPlan;
use jolt_field::{Ring as _, Zero as _};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const COEFFICIENT_VARS: usize = 9;

fn canonical_key(record: usize) -> CanonicalPublicKeyBytes<aerie_falcon_native::Falcon512> {
    let mut encoded = [0_u8; Falcon512::PUBLIC_KEY_BYTES];
    encoded[0] = 0x09;
    for (block, destination) in encoded[1..].chunks_exact_mut(7).enumerate() {
        let residues: [u16; 4] = std::array::from_fn(|lane| {
            u16::try_from((record * 4_099 + (4 * block + lane) * 7_919 + 1_237) % FALCON_Q as usize)
                .expect("residue fits u16")
        });
        let word = (u64::from(residues[0]) << 42)
            | (u64::from(residues[1]) << 28)
            | (u64::from(residues[2]) << 14)
            | u64::from(residues[3]);
        for (index, byte) in destination.iter_mut().enumerate() {
            *byte = (word >> (8 * (6 - index))) as u8;
        }
    }
    CanonicalPublicKeyBytes::parse(&encoded).expect("benchmark key is canonical")
}

fn expanded_keys(
    public_keys: &[CanonicalPublicKeyBytes<aerie_falcon_native::Falcon512>],
    records: usize,
) -> Vec<i16> {
    let mut expanded = vec![0_i16; records * Falcon512::DEGREE];
    #[cfg(feature = "parallel")]
    expanded
        .par_chunks_mut(Falcon512::DEGREE)
        .zip(public_keys.par_iter())
        .for_each(|(destination, public_key)| {
            destination.copy_from_slice(public_key.decode().h_centered());
        });
    #[cfg(not(feature = "parallel"))]
    for (destination, public_key) in expanded
        .chunks_exact_mut(Falcon512::DEGREE)
        .zip(public_keys)
    {
        destination.copy_from_slice(public_key.decode().h_centered());
    }
    expanded
}

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        f64::midpoint(samples[middle - 1], samples[middle])
    } else {
        samples[middle]
    }
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let records: usize = arguments.next().map_or(4_096, |value| {
        value.parse().expect("records must be an integer")
    });
    let iterations: usize = arguments.next().map_or(21, |value| {
        value.parse().expect("iterations must be an integer")
    });
    let backend = arguments.next().unwrap_or_else(|| "auto".to_owned());
    let dot = match backend.as_str() {
        "auto" => SmallSignedDotPlan::best_available(),
        "avx2" => SmallSignedDotPlan::avx2(),
        "avx512" => SmallSignedDotPlan::avx512(),
        _ => panic!("backend must be auto, avx2, or avx512"),
    }
    .expect("requested SIMD backend is unavailable");
    assert!(records.is_power_of_two());
    assert!(iterations > 0);

    let public_keys = (0..records).map(canonical_key).collect::<Vec<_>>();
    let row_vars = records.trailing_zeros() as usize;
    let point = |offset: i64| {
        (0..COEFFICIENT_VARS + row_vars)
            .map(|index| E::from_i64(offset + i64::try_from(index).expect("index fits i64") * 17))
            .collect::<Vec<_>>()
    };
    let s1_point = point(3);
    let n_point = point(11);

    let expanded = expanded_keys(&public_keys, records);
    let expected = functional_evaluation(&expanded, records, &s1_point, &n_point);
    for plan in [None, Some(dot)] {
        assert_eq!(
            functional_evaluation_canonical_for_bench(
                &public_keys,
                records,
                &s1_point,
                &n_point,
                plan,
            ),
            expected
        );
    }
    black_box(E::zero());

    let mut expanded_samples = Vec::with_capacity(iterations);
    let mut field_samples = Vec::with_capacity(iterations);
    let mut simd_samples = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let measure_expanded = || {
            let started = Instant::now();
            let expanded = expanded_keys(black_box(&public_keys), records);
            black_box(functional_evaluation(
                black_box(&expanded),
                records,
                black_box(&s1_point),
                black_box(&n_point),
            ));
            started.elapsed().as_secs_f64() * 1_000.0
        };
        let measure_canonical = |plan| {
            let started = Instant::now();
            black_box(functional_evaluation_canonical_for_bench(
                black_box(&public_keys),
                records,
                black_box(&s1_point),
                black_box(&n_point),
                black_box(plan),
            ));
            started.elapsed().as_secs_f64() * 1_000.0
        };
        match iteration % 3 {
            0 => {
                expanded_samples.push(measure_expanded());
                field_samples.push(measure_canonical(None));
                simd_samples.push(measure_canonical(Some(dot)));
            }
            1 => {
                field_samples.push(measure_canonical(None));
                simd_samples.push(measure_canonical(Some(dot)));
                expanded_samples.push(measure_expanded());
            }
            _ => {
                simd_samples.push(measure_canonical(Some(dot)));
                expanded_samples.push(measure_expanded());
                field_samples.push(measure_canonical(None));
            }
        }
    }

    let expanded_ms = median_ms(&mut expanded_samples);
    let field_ms = median_ms(&mut field_samples);
    let simd_ms = median_ms(&mut simd_samples);
    println!(
        "records\titerations\tbackend\texpanded_ms\tfield_ms\tsimd_ms\tsimd_speedup\texpanded_bytes"
    );
    println!(
        "{records}\t{iterations}\t{backend}\t{expanded_ms:.3}\t{field_ms:.3}\t{simd_ms:.3}\t{:.3}\t{}",
        field_ms / simd_ms,
        records * Falcon512::DEGREE * size_of::<i16>()
    );
}
