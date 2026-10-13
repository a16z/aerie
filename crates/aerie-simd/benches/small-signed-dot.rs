use std::{hint::black_box, time::Instant};

use aerie_simd::{Backend, KeyUnpackPlan, SmallSignedDotPlan};

const TERMS: usize = 512;
const KEY_COEFFICIENTS: usize = 1024;
const KEY_BYTES: usize = KEY_COEFFICIENTS * 14 / 8;

fn mix(index: usize) -> u64 {
    u64::try_from(index)
        .expect("term index fits u64")
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .rotate_left(u32::try_from(index % 64).expect("rotation fits u32"))
}

fn signed_digits(value: u64) -> (i32, i32) {
    let wide = i128::from(value);
    let low = i32::from_le_bytes(value.to_le_bytes()[..4].try_into().expect("four bytes"));
    let high = i32::try_from((wide - i128::from(low)) >> 32).expect("high digit fits i32");
    (low, high)
}

fn dot_fixture() -> ([[i32; TERMS]; 4], [i16; TERMS]) {
    let values0: [u64; TERMS] = core::array::from_fn(mix);
    let values1 = values0.map(|value| value.rotate_left(29) ^ 0xa5a5_5a5a_1234_5678);
    let digits = [
        values0.map(|value| signed_digits(value >> 1).0),
        values0.map(|value| signed_digits(value >> 1).1),
        values1.map(|value| signed_digits(value >> 1).0),
        values1.map(|value| signed_digits(value >> 1).1),
    ];
    let factors = core::array::from_fn(|index| {
        i16::try_from(mix(index + TERMS) % 12_289).expect("residue fits i16") - 6_144
    });
    (digits, factors)
}

fn time(label: &str, iterations: usize, mut run: impl FnMut()) {
    for _ in 0..1_000 {
        run();
    }
    let started = Instant::now();
    for _ in 0..iterations {
        run();
    }
    let iterations = f64::from(u32::try_from(iterations).expect("iteration count fits u32"));
    let ns = started.elapsed().as_secs_f64() * 1e9 / iterations;
    println!("{label}\t{ns:.2}");
}

fn main() {
    let iterations = std::env::args()
        .nth(1)
        .map_or(200_000, |value| value.parse().expect("integer iterations"));
    println!("# selected backend: {}", Backend::selected().name());

    let (digits, factors) = dot_fixture();
    let references = [&digits[0], &digits[1], &digits[2], &digits[3]];
    println!("dot_kernel\tns_per_{TERMS}_term_dot");
    let dot_plans = [
        Some(SmallSignedDotPlan::scalar()),
        SmallSignedDotPlan::neon(),
        SmallSignedDotPlan::avx2(),
        SmallSignedDotPlan::avx512(),
    ];
    for plan in dot_plans.into_iter().flatten() {
        time(plan.name(), iterations, || {
            black_box(plan.dot(black_box(references), black_box(&factors)));
        });
    }

    let payload: Vec<u8> = (0..KEY_BYTES)
        .map(|index| mix(index).to_le_bytes()[0])
        .collect();
    let mut out = vec![0_i16; KEY_COEFFICIENTS];
    println!("unpack_kernel\tns_per_{KEY_COEFFICIENTS}_coefficient_key");
    let unpack_plans = [
        Some(KeyUnpackPlan::scalar()),
        KeyUnpackPlan::neon(),
        KeyUnpackPlan::avx2(),
        KeyUnpackPlan::avx512(),
    ];
    for plan in unpack_plans.into_iter().flatten() {
        time(plan.name(), iterations, || {
            plan.unpack_centered_14(black_box(&payload), black_box(&mut out));
        });
    }
}
