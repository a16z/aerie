use jolt_field::{Ring as _, Zero as _};

use crate::substrate::BaseField as F;

use super::super::FalconDegree;
use super::cores::{
    core_xof_prefix, decode_ternary, decode_uniform, CoreContext, CoreId, Leg, ProjectionCores,
};
use super::profile::{CoreDistribution, Geometry};
use super::*;

const SEED: [u8; 32] = [0x5a; 32];
const DEGREES: [FalconDegree; 2] = [FalconDegree::Falcon512, FalconDegree::Falcon1024];

#[test]
fn core_rows_match_independent_customized_keccak() {
    let prefix = b"core row reference fixture";
    let row = 17_usize;
    let mut input = prefix.to_vec();
    input.extend_from_slice(&(row as u64).to_le_bytes());
    let reference =
        aerie_keccak::customized_chaining_values(&[&input], b"", b"aerie/falcon/v1/projection");
    let bytes = reference[0].as_bytes();
    // Lengths off a multiple of four end inside a byte.
    for len in [256, 255, 254, 253, 43] {
        let mut norm = vec![0; len];
        super::cores::fill_core_rows(
            &super::cores::core_stream(prefix),
            row,
            CoreDistribution::TernaryQuarterHalfQuarter,
            len,
            &mut norm,
        );
        for (col, value) in norm.iter().enumerate() {
            let pair = (bytes[col / 4] >> (2 * (col % 4))) & 3;
            assert_eq!(
                *value,
                match pair {
                    0 => -1,
                    3 => 1,
                    _ => 0,
                },
                "len {len}, col {col}"
            );
        }
    }
    let mut residual = [0; 32];
    super::cores::fill_core_rows(
        &super::cores::core_stream(prefix),
        row,
        CoreDistribution::UniformSymmetric { bound: 12 },
        residual.len(),
        &mut residual,
    );
    let expected: Vec<_> = bytes
        .iter()
        .filter(|&&b| b < 250)
        .map(|&b| i8::try_from(b % 25).unwrap() - 12)
        .take(32)
        .collect();
    assert_eq!(residual.as_slice(), expected);
}

#[test]
fn batched_core_rows_match_single_rows() {
    let stream = super::cores::core_stream(b"batched core row fixture");
    let distributions = [
        CoreDistribution::TernaryQuarterHalfQuarter,
        CoreDistribution::UniformSymmetric { bound: 12 },
    ];
    for distribution in distributions {
        for cols in [1, 43, 545, 2048] {
            for rows in [1, 2, 3, 9] {
                let mut batched = vec![0; rows * cols];
                super::cores::fill_core_rows(&stream, 5, distribution, cols, &mut batched);
                for (offset, row) in batched.chunks(cols).enumerate() {
                    let mut single = vec![0; cols];
                    super::cores::fill_core_rows(
                        &stream,
                        5 + offset,
                        distribution,
                        cols,
                        &mut single,
                    );
                    assert_eq!(
                        row, single,
                        "{distribution:?}, cols {cols}, row {offset}/{rows}"
                    );
                }
            }
        }
    }
}

// ------------------------------------------------------------- profile

#[test]
fn projection_profile_geometry_and_binding() {
    for (degree, k) in [(FalconDegree::Falcon512, 2), (FalconDegree::Falcon1024, 1)] {
        let minimum = 1 << k;
        assert!(ProjectionProfile::for_batch(degree, minimum / 2).is_err());
        assert!(ProjectionProfile::for_batch(degree, 3).is_err());
        assert!(ProjectionProfile::for_batch(degree, 1 << 17).is_err());
        for record_bits in k..=16 {
            let records = 1 << record_bits;
            let geometry = Geometry::new(degree, records).expect("admitted geometry");
            let rest = record_bits - k;
            let b2 = rest.min(4);
            let b3 = (rest - b2).min(4);
            let b4 = (rest - b2 - b3).min(4);
            assert_eq!(geometry.low_record_bits, k, "{degree:?}, N*={records}");
            assert_eq!(geometry.rest_bits, rest, "{degree:?}, N*={records}");
            assert_eq!(
                geometry.norm_bits,
                [12, b2, b3, b4, rest - b2 - b3 - b4],
                "{degree:?}, N*={records}"
            );
            assert_eq!(
                geometry.residual_bits,
                [11, rest / 2, rest - rest / 2],
                "{degree:?}, N*={records}"
            );
            let profile = ProjectionProfile::for_batch(degree, records).expect("profile");
            assert_eq!(profile.norm.radices, geometry.norm_radices());
            assert_eq!(profile.residual.radices, geometry.residual_radices());
        }
        let geometry = Geometry::new(degree, 1 << 16).expect("geometry");
        assert_eq!(geometry.low_record_bits, k);
        assert_eq!(geometry.rest_bits, 16 - k);
        let rest = 16 - k;
        let b2 = rest.min(4);
        let b3 = (rest - b2).min(4);
        let b4 = (rest - b2 - b3).min(4);
        assert_eq!(geometry.norm_bits, [12, b2, b3, b4, rest - b2 - b3 - b4]);
        assert_eq!(geometry.residual_bits, [11, rest / 2, rest - rest / 2]);
        let profile = ProjectionProfile::for_batch(degree, 1 << 16).expect("profile");
        assert_eq!(profile.norm.cap, norm_cap(degree));
        assert_eq!(profile.residual.modulus, FALCON_Q);
        let bytes = profile.binding_bytes();
        // label, 1 + 4 + 1 + 5 LE64, ternary tag, order, LE128 cap,
        // 3 + 3 LE64, uniform tag and bound, LE64 modulus.
        assert_eq!(
            bytes.len(),
            PROFILE_BINDING_LABEL.len() + 88 + 1 + 1 + 16 + 48 + 2 + 8
        );
        assert!(bytes.starts_with(PROFILE_BINDING_LABEL));
        let other = ProjectionProfile::for_batch(degree, 1 << 15).expect("profile");
        assert_ne!(bytes, other.binding_bytes());
    }
    assert_ne!(
        ProjectionProfile::for_batch(FalconDegree::Falcon512, 8)
            .expect("profile")
            .binding_bytes(),
        ProjectionProfile::for_batch(FalconDegree::Falcon1024, 8)
            .expect("profile")
            .binding_bytes()
    );
}

// ------------------------------------------------------ core derivation

fn from_hex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&hex[index..index + 2], 16).expect("hex"))
        .collect()
}

/// Expected values computed independently with Python's
/// `PyCryptodome` `Crypto.Hash.cSHAKE256` over the documented framing, seed `0x00..0x1f`,
/// Falcon-512, `N_sig = 5`, `N_sig* = 8` (radices `[4096, 2, 1, 1, 1]` and
/// `[2048, 1, 2]`).
#[test]
fn projection_core_derivation_known_answer() {
    let seed: Vec<u8> = (0..32).collect();
    let geometry = Geometry::new(FalconDegree::Falcon512, 8).expect("geometry");
    assert_eq!(geometry.norm_radices(), [4096, 2, 1, 1, 1]);
    assert_eq!(geometry.residual_radices(), [2048, 1, 2]);
    let context = CoreContext {
        seed: &seed,
        degree: FalconDegree::Falcon512,
        live_rows: 5,
        records: 8,
    };
    let prefix = core_xof_prefix(
        &context,
        CoreId {
            leg: Leg::Norm,
            group: 0,
            layer: 1,
        },
        43,
        4096,
        &geometry.norm_radices(),
    );
    assert_eq!(
        prefix,
        from_hex(concat!(
            "61657269652f66616c636f6e2f76312f70726f6a656374696f6e2f636f72652d7631",
            "010009",
            "0500000000000000",
            "0800000000000000",
            "000001",
            "2b00000000000000",
            "0010000000000000",
            "05",
            "0010000000000000",
            "0200000000000000",
            "0100000000000000",
            "0100000000000000",
            "0100000000000000",
            "2000000000000000",
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f",
        ))
    );

    let cores = ProjectionCores::derive(&seed, &geometry, 5);
    let head = |row: &[i8]| row[..16].to_vec();
    assert_eq!(
        head(cores.norm(0, 1).row(0)),
        [0, 0, 1, 0, 1, 0, 0, 0, -1, -1, 0, 0, 0, -1, -1, 0]
    );
    assert_eq!(
        head(cores.norm(0, 1).row(42)),
        [-1, -1, 1, 0, -1, 0, 0, 0, 0, 0, 1, 0, 1, -1, 0, -1]
    );
    assert_eq!(cores.norm(3, 2).cols, 86);
    assert_eq!(
        head(cores.norm(3, 2).row(7)),
        [1, 1, 1, 0, 0, 0, 0, -1, 0, -1, 1, 1, 0, 0, 1, 0]
    );
    assert_eq!(cores.norm_terminal.cols, 172);
    assert_eq!(
        head(cores.norm_terminal.row(155)),
        [0, 1, 0, 1, -1, 0, 0, -1, 1, -1, 0, 1, 1, 0, 1, 0]
    );
    assert_eq!(
        head(cores.residual(1).row(0)),
        [-2, -12, 3, 6, 5, -11, 1, -7, -4, -2, 2, 5, 1, 10, -9, -3]
    );
    assert_eq!(cores.residual(3).cols, 62);
    assert_eq!(
        head(cores.residual(3).row(30)),
        [-1, -11, 2, -12, -7, 10, -7, -4, -3, 9, -9, -2, 6, 8, 3, -8]
    );

    // Every framing field separates streams.
    let other_seed = [1_u8; 32];
    let base = CoreId {
        leg: Leg::Norm,
        group: 0,
        layer: 1,
    };
    let radices = geometry.norm_radices();
    let variants = [
        core_xof_prefix(
            &CoreContext {
                live_rows: 6,
                ..context
            },
            base,
            43,
            4096,
            &radices,
        ),
        core_xof_prefix(
            &CoreContext {
                records: 16,
                ..context
            },
            base,
            43,
            4096,
            &radices,
        ),
        core_xof_prefix(
            &CoreContext {
                degree: FalconDegree::Falcon1024,
                ..context
            },
            base,
            43,
            4096,
            &radices,
        ),
        core_xof_prefix(
            &CoreContext {
                seed: &other_seed,
                ..context
            },
            base,
            43,
            4096,
            &radices,
        ),
        core_xof_prefix(&context, CoreId { group: 1, ..base }, 43, 4096, &radices),
        core_xof_prefix(&context, CoreId { layer: 2, ..base }, 43, 4096, &radices),
        core_xof_prefix(
            &context,
            CoreId {
                leg: Leg::Residual,
                ..base
            },
            43,
            4096,
            &radices,
        ),
        core_xof_prefix(&context, base, 42, 4096, &radices),
        core_xof_prefix(&context, base, 43, 4096, &[4096, 1, 2, 1, 1]),
    ];
    for variant in &variants {
        assert_ne!(variant, &prefix);
    }
    // Derivation is deterministic and independent of the thread count.
    assert_eq!(cores, ProjectionCores::derive(&seed, &geometry, 5));
}

#[test]
fn projection_sampler_bias() {
    // Uniform: every accepted byte class has exactly 10 preimages.
    let mut counts = [0_usize; 25];
    let mut rejected = 0;
    for byte in 0..=u8::MAX {
        match decode_uniform(byte) {
            Some(value) => counts[usize::try_from(value + 12).expect("in range")] += 1,
            None => rejected += 1,
        }
    }
    assert_eq!(counts, [10; 25]);
    assert_eq!(rejected, 6);

    // Ternary: per slot, 64 / 128 / 64 of the 256 bytes.
    for slot in 0..4 {
        let mut counts = [0_usize; 3];
        for byte in 0..=u8::MAX {
            let bits = (byte >> (2 * slot)) & 0b11;
            let expected = match bits {
                0b00 => -1,
                0b11 => 1,
                0b01 | 0b10 => 0,
                _ => unreachable!(),
            };
            let actual = decode_ternary(byte, slot);
            assert_eq!(actual, expected, "byte {byte:#04x}, slot {slot}");
            counts[usize::try_from(actual + 1).expect("in range")] += 1;
        }
        assert_eq!(counts, [64, 128, 64]);
    }

    // Empirical frequencies of derived cores (fixed seed, so deterministic).
    let geometry = Geometry::new(FalconDegree::Falcon512, 8).expect("geometry");
    let cores = ProjectionCores::derive(&SEED, &geometry, 8);
    let entries = &cores.residual(1).entries;
    let mut counts = [0_f64; 25];
    for &entry in entries {
        counts[usize::try_from(entry + 12).expect("in range")] += 1.0;
    }
    let expected = entries.len() as f64 / 25.0;
    let chi_square: f64 = counts
        .iter()
        .map(|&count| (count - expected).powi(2) / expected)
        .sum();
    // 24 degrees of freedom: P[chi^2 > 60] < 1e-4.
    assert!(chi_square < 60.0, "uniform chi-square {chi_square}");

    let mut counts = [0_f64; 3];
    let mut total = 0.0;
    for group in 0..NORM_GROUPS {
        for &entry in &cores.norm(group, 1).entries {
            counts[usize::try_from(entry + 1).expect("in range")] += 1.0;
            total += 1.0;
        }
    }
    let expected = [total / 4.0, total / 2.0, total / 4.0];
    let chi_square: f64 = counts
        .iter()
        .zip(expected)
        .map(|(&count, expected)| (count - expected).powi(2) / expected)
        .sum();
    // 2 degrees of freedom: P[chi^2 > 20] < 1e-4.
    assert!(chi_square < 20.0, "ternary chi-square {chi_square}");
}

// ----------------------------------------------------------- predicates

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

/// A `y` with `sum ctr_p(y_j)^2 = target`, alternating signs.
fn y_with_norm(target: u128) -> [F; NORM_TERMINAL_ROWS] {
    let mut rest = target;
    let mut y = [F::zero(); NORM_TERMINAL_ROWS];
    let mut index = 0;
    while rest > 0 {
        let root = isqrt(rest);
        rest -= root * root;
        let signed = i128::try_from(root).expect("small");
        y[index] = F::from_i128(if index % 2 == 0 { signed } else { -signed });
        index += 1;
    }
    y
}

#[test]
fn projection_norm_predicate_boundary() {
    for degree in DEGREES {
        let cap = norm_cap(degree);
        let at_cap = y_with_norm(cap);
        let sum: u128 = at_cap
            .iter()
            .map(|&value| centered(value).unsigned_abs().pow(2))
            .sum();
        assert_eq!(sum, cap);
        assert!(norm_predicate(&at_cap, cap));
        assert!(!norm_predicate(&y_with_norm(cap + 1), cap));
        let outputs = ProjectionOutputs {
            y: at_cap,
            z: [F::zero(); RESIDUAL_WIDTH],
        };
        assert!(check_outputs(degree, &outputs));
        let outputs = ProjectionOutputs {
            y: y_with_norm(cap + 1),
            ..outputs
        };
        assert!(!check_outputs(degree, &outputs));
    }
    // Early exit: 156 squares of (p - 1)/2 would overflow u128; with
    // overflow checks on, a missing early exit panics.
    let half = F::from_u64((FIELD_MODULUS - 1) / 2);
    assert!(!norm_predicate(
        &[half; NORM_TERMINAL_ROWS],
        NORM_CAP_FALCON_1024
    ));
    assert!(!norm_predicate(&[half; NORM_TERMINAL_ROWS], u128::MAX >> 1));
}

#[test]
fn projection_centered_and_residual_boundary() {
    let p = FIELD_MODULUS;
    let half = (p - 1) / 2;
    assert_eq!(centered(F::zero()), 0);
    assert_eq!(centered(F::from_u64(p - 1)), -1);
    assert_eq!(centered(F::from_u64(half)), i128::from(half));
    assert_eq!(centered(F::from_u64(half + 1)), -i128::from(half));

    let q = FALCON_Q;
    let mut z = [F::zero(); RESIDUAL_WIDTH];
    assert!(residual_predicate(&z, q));
    let check = |value: F| {
        let mut z = [F::zero(); RESIDUAL_WIDTH];
        z[RESIDUAL_WIDTH - 1] = value;
        residual_predicate(&z, q)
    };
    assert!(check(F::from_u64(5 * q)));
    assert!(check(F::from_u64(p - 5 * q)));
    assert!(!check(F::from_u64(5 * q + 1)));
    assert!(!check(F::from_u64(5 * q - 1)));
    assert!(!check(F::from_u64(p - 5 * q + 1)));
    assert!(!check(F::from_u64(p - 5 * q - 1)));
    // The largest centered multiple of q, on both sides of zero.
    let largest = half / q * q;
    assert!(check(F::from_u64(largest)));
    assert!(check(F::from_u64(p - largest)));
    // A canonical multiple of q above (p - 1)/2 is a negative centered
    // value that q does not divide.
    let above = largest + q;
    assert!(above > half && above.is_multiple_of(q));
    assert!(!check(F::from_u64(above)));
    z[0] = F::from_u64(q + 1);
    assert!(!residual_predicate(&z, q));
}
