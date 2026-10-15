//! Usage: ring-table [records=16384] [iterations=15] [degree=512] [live=records].
use aerie_prover::{ring::functional_kernel_table_for_bench, BaseField as F, OpeningField as E};
use jolt_field::Ring;
use std::{hint::black_box, time::Instant};

fn main() {
    let arguments = std::env::args()
        .skip(1)
        .filter(|argument| argument != "--bench")
        .map(|argument| {
            argument
                .parse::<usize>()
                .expect("arguments must be integers")
        })
        .collect::<Vec<_>>();
    assert!(arguments.len() <= 4);
    let records = arguments.first().copied().unwrap_or(16_384);
    let iterations = arguments.get(1).copied().unwrap_or(15);
    let degree = arguments.get(2).copied().unwrap_or(512);
    let live = arguments.get(3).copied().unwrap_or(records);
    assert!(records > 0 && iterations > 0 && live <= records);
    assert!(matches!(degree, 512 | 1024));
    let cells = records.checked_mul(degree).expect("input size overflow");
    let h = (0..cells)
        .map(|i| ((i * 7_919 + i / degree * 4_099 + 1_237) % 12_289) as i16 - 6_144)
        .collect::<Vec<_>>();
    let sigma = (0..cells / 64)
        .map(|i| (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0xa5a5_5a5a_f0f0_0f0f)
        .collect::<Vec<_>>();
    let magnitude = (0..cells)
        .map(|i| ((i * 37 + 19) % 2048) as i16)
        .collect::<Vec<_>>();
    let mut state = 0x2147_2578_93ad_72df_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        F::from_u64(state)
    };
    let point = (0..degree.trailing_zeros())
        .map(|_| E::new(next(), next()))
        .collect::<Vec<_>>();
    let row_scale = (0..records)
        .map(|_| E::new(next(), next()))
        .collect::<Vec<_>>();
    #[cfg(feature = "parallel")]
    let threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let threads = 1;
    let mut samples = Vec::with_capacity(iterations);
    let mut expected = None;
    for _ in 0..iterations {
        let started = Instant::now();
        let (table, coefficients) = black_box(functional_kernel_table_for_bench(
            black_box(&h),
            records,
            live,
            black_box(&point),
            black_box(&sigma),
            black_box(&magnitude),
            black_box(&row_scale),
        ));
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        // Hash canonical limbs in a fixed byte order, outside the timed region.
        let checksum =
            table
                .iter()
                .chain(&coefficients)
                .fold(0xcbf2_9ce4_8422_2325_u64, |mut hash, value| {
                    for limb in [value.c0(), value.c1()] {
                        for byte in limb.to_canonical_u64().to_le_bytes() {
                            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
                        }
                    }
                    hash
                });
        assert_eq!(*expected.get_or_insert(checksum), checksum);
    }
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    let median = if samples.len().is_multiple_of(2) {
        f64::midpoint(samples[middle - 1], samples[middle])
    } else {
        samples[middle]
    };
    println!("records\tdegree\tlive\titerations\tthreads\tmin_ms\tmedian_ms\tchecksum");
    println!(
        "{records}\t{degree}\t{live}\t{iterations}\t{threads}\t{:.6}\t{median:.6}\t{:016x}",
        samples[0],
        expected.unwrap()
    );
}
