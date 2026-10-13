use aerie_falcon_native::{Falcon1024, Falcon512};
use aerie_falcon_statement::{DirectParams, DirectRow};
use jolt_field::{Ring as _, Zero as _};

use crate::substrate::BaseField as F;

use super::super::{FalconDegree, FalconWitness};
use super::cores::{
    core_xof_prefix, decode_ternary, decode_uniform, CoreContext, CoreId, Leg, ProjectionCores,
};
use super::profile::{Geometry, FIRST_CHANNELS};
use super::*;

const SEED: [u8; 32] = [0x5a; 32];
const DEGREES: [FalconDegree; 2] = [FalconDegree::Falcon512, FalconDegree::Falcon1024];

// ---------------------------------------------------------------- fixtures

fn rows_of<P: DirectParams>(live_rows: usize) -> FalconWitness {
    let build = |index: usize| DirectRow::<P>::synthetic(0x5eed_0000 + index as u64).expect("row");
    #[cfg(feature = "parallel")]
    let rows: Vec<DirectRow<P>> = {
        use rayon::prelude::*;
        (0..live_rows).into_par_iter().map(build).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<DirectRow<P>> = (0..live_rows).map(build).collect();
    FalconWitness::from_rows(&rows)
}

/// An honest synthetic batch with nonzero slack cells on the live rows.
///
/// Synthetic rows have `Delta = 0` exactly (their `h (*) s_2^*` never leaves
/// `(-q/2, q/2)`), so `t` is shifted by `+-q` on a pattern of live cells to
/// make `Delta` a nonzero multiple of `q`, as it is for real Falcon keys.
fn honest_batch(degree: FalconDegree, live_rows: usize) -> FalconWitness {
    let mut batch = match degree {
        FalconDegree::Falcon512 => rows_of::<Falcon512>(live_rows),
        FalconDegree::Falcon1024 => rows_of::<Falcon1024>(live_rows),
    };
    let q = i16::try_from(FALCON_Q).expect("q fits i16");
    let live_cells = degree.degree() * live_rows;
    for (cell, slot) in batch.t[..live_cells].iter_mut().enumerate() {
        match cell % 3 {
            0 if *slot < 0 => *slot += q,
            0 => *slot -= q,
            _ => {}
        }
    }
    for (cell, slot) in batch.slack[..4 * live_rows].iter_mut().enumerate() {
        if *slot == 0 {
            *slot = i16::try_from((cell * 37 + 11) % 401).expect("small") - 200;
        }
    }
    batch
}

// --------------------------------------------------------- naive sources

/// Schoolbook negacyclic `h (*) s_2` of one record.
fn naive_negacyclic(h: &[i16], s2: &[i16]) -> Vec<i64> {
    let n = h.len();
    let mut out = vec![0_i64; n];
    for (a, &left) in h.iter().enumerate() {
        if left == 0 {
            continue;
        }
        for (b, &right) in s2.iter().enumerate() {
            let product = i64::from(left) * i64::from(right);
            if a + b < n {
                out[a + b] += product;
            } else {
                out[a + b - n] -= product;
            }
        }
    }
    out
}

/// Flat `Delta`, index `j + n i`.
fn naive_residual(batch: &FalconWitness, n: usize) -> Vec<i64> {
    let mut out = Vec::with_capacity(batch.s1.len());
    for record in 0..batch.padded_rows {
        let range = record * n..(record + 1) * n;
        let product = naive_negacyclic(&batch.h[range.clone()], &batch.s2[range.clone()]);
        for ((&s1, &product), &t) in batch.s1[range.clone()]
            .iter()
            .zip(&product)
            .zip(&batch.t[range])
        {
            out.push(i64::from(s1) + product - i64::from(t));
        }
    }
    out
}

/// Flat norm source `S`, index `j + n a + 2n i`.
fn naive_norm_source(batch: &FalconWitness, n: usize) -> Vec<i64> {
    let mut out = vec![0_i64; 2 * n * batch.padded_rows];
    for record in 0..batch.padded_rows {
        for j in 0..n {
            out[2 * n * record + j] = i64::from(batch.s1[record * n + j]);
        }
        for slot in 0..4 {
            out[2 * n * record + n + slot] = i64::from(batch.slack[4 * record + slot]);
        }
    }
    out
}

/// First-core table for the coefficient source, evaluated directly from
/// `s_1` and `Delta` without the norm slack lane.
fn naive_first_without_slack(
    batch: &FalconWitness,
    geometry: &Geometry,
    cores: &ProjectionCores,
) -> Vec<i64> {
    let n = geometry.n;
    let low_records = geometry.low_records();
    let delta = naive_residual(batch, n);
    let mut out = vec![0_i64; geometry.rest_size() * FIRST_CHANNELS];
    for high in 0..geometry.rest_size() {
        let channels = &mut out[high * FIRST_CHANNELS..(high + 1) * FIRST_CHANNELS];
        for group in 0..NORM_GROUPS {
            let core = cores.norm(group, 1);
            for channel in 0..NORM_PREFIX_WIDTH {
                let row = core.row(channel);
                let mut sum = 0_i64;
                for low in 0..low_records {
                    let record = high * low_records + low;
                    for coefficient in 0..n {
                        sum += i64::from(row[2 * n * low + coefficient])
                            * i64::from(batch.s1[record * n + coefficient]);
                    }
                }
                channels[group * NORM_PREFIX_WIDTH + channel] = sum;
            }
        }
        let residual = cores.residual(1);
        for channel in 0..RESIDUAL_WIDTH {
            let row = residual.row(channel);
            let mut sum = 0_i64;
            for low in 0..low_records {
                let record = high * low_records + low;
                for coefficient in 0..n {
                    sum += i64::from(row[n * low + coefficient]) * delta[record * n + coefficient];
                }
            }
            channels[NORM_CHANNELS + channel] = sum;
        }
    }
    out
}

// ------------------------------------------------ column-wise projection

/// Digits of a source index: the first-block index, then one radix digit
/// per later core.
fn digits(mut index: usize, radices: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(radices.len());
    for &radix in radices {
        out.push(index % radix);
        index /= radix;
    }
    assert_eq!(index, 0, "index out of range");
    out
}

/// Column `x` of the dense norm operator, straight from the definition.
fn norm_column(cores: &ProjectionCores, geometry: &Geometry, x: usize) -> Vec<i128> {
    let digit = digits(x, &geometry.norm_radices());
    let mut concatenated = vec![0_i128; NORM_CHANNELS];
    for group in 0..NORM_GROUPS {
        let first = cores.norm(group, 1);
        let mut vector: Vec<i128> = (0..NORM_PREFIX_WIDTH)
            .map(|c| i128::from(first.get(c, digit[0])))
            .collect();
        for layer in 2..=NORM_PREFIX_LAYERS {
            let core = cores.norm(group, layer);
            vector = (0..NORM_PREFIX_WIDTH)
                .map(|out| {
                    (0..NORM_PREFIX_WIDTH)
                        .map(|c| {
                            i128::from(core.get(out, c + NORM_PREFIX_WIDTH * digit[layer - 1]))
                                * vector[c]
                        })
                        .sum()
                })
                .collect();
        }
        concatenated[NORM_PREFIX_WIDTH * group..][..NORM_PREFIX_WIDTH].copy_from_slice(&vector);
    }
    (0..NORM_TERMINAL_ROWS)
        .map(|out| {
            (0..NORM_CHANNELS)
                .map(|c| {
                    i128::from(cores.norm_terminal.get(out, c + NORM_CHANNELS * digit[4]))
                        * concatenated[c]
                })
                .sum()
        })
        .collect()
}

/// Column `w` of the dense residual operator.
fn residual_column(cores: &ProjectionCores, geometry: &Geometry, w: usize) -> Vec<i128> {
    let digit = digits(w, &geometry.residual_radices());
    let mut vector: Vec<i128> = (0..RESIDUAL_WIDTH)
        .map(|c| i128::from(cores.residual(1).get(c, digit[0])))
        .collect();
    for layer in 2..=RESIDUAL_LAYERS {
        let core = cores.residual(layer);
        vector = (0..RESIDUAL_WIDTH)
            .map(|out| {
                (0..RESIDUAL_WIDTH)
                    .map(|c| {
                        i128::from(core.get(out, c + RESIDUAL_WIDTH * digit[layer - 1])) * vector[c]
                    })
                    .sum()
            })
            .collect();
    }
    vector
}

/// Dense operators `[column][row]`.
fn dense_operators(
    cores: &ProjectionCores,
    geometry: &Geometry,
) -> (Vec<Vec<i128>>, Vec<Vec<i128>>) {
    let norm_columns = 2 * geometry.n * geometry.records;
    let residual_columns = geometry.n * geometry.records;
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (
            (0..norm_columns)
                .into_par_iter()
                .map(|x| norm_column(cores, geometry, x))
                .collect(),
            (0..residual_columns)
                .into_par_iter()
                .map(|w| residual_column(cores, geometry, w))
                .collect(),
        )
    }
    #[cfg(not(feature = "parallel"))]
    (
        (0..norm_columns)
            .map(|x| norm_column(cores, geometry, x))
            .collect(),
        (0..residual_columns)
            .map(|w| residual_column(cores, geometry, w))
            .collect(),
    )
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
/// `hashlib.shake_256` over the documented framing, seed `0x00..0x1f`,
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
        [0, 1, 0, 1, 0, 0, 1, 1, 0, -1, 1, 0, 1, -1, 1, 0]
    );
    assert_eq!(
        head(cores.norm(0, 1).row(42)),
        [0, 1, 0, 0, 0, 0, 0, -1, 0, -1, -1, 0, 0, 0, 1, 0]
    );
    assert_eq!(cores.norm(3, 2).cols, 86);
    assert_eq!(
        head(cores.norm(3, 2).row(7)),
        [0, -1, -1, 0, -1, 0, 1, -1, -1, 0, 0, -1, 1, -1, 0, 0]
    );
    assert_eq!(cores.norm_terminal.cols, 172);
    assert_eq!(
        head(cores.norm_terminal.row(155)),
        [1, -1, 1, 0, 1, 0, 0, 0, 0, -1, -1, 0, -1, 1, -1, 0]
    );
    assert_eq!(
        head(cores.residual(1).row(0)),
        [4, -1, 6, 2, -5, 5, 8, -3, 12, -9, 7, 1, -12, -9, 7, 1]
    );
    assert_eq!(cores.residual(3).cols, 62);
    assert_eq!(
        head(cores.residual(3).row(30)),
        [2, 9, 6, 1, 10, 7, 5, 6, -2, 3, 11, -11, 6, -9, -3, -1]
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
        assert!(check_outputs(degree, &outputs).is_ok());
        let outputs = ProjectionOutputs {
            y: y_with_norm(cap + 1),
            ..outputs
        };
        assert!(check_outputs(degree, &outputs).is_err());
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

// -------------------------------------------------------------- forward

#[test]
fn projection_sources_match_definition() {
    for degree in DEGREES {
        let n = degree.degree();
        let batch = honest_batch(degree, 3);
        let naive = naive_residual(&batch, n);
        let mut record = vec![0_i64; n];
        for index in 0..batch.padded_rows {
            sources::delta_record(&batch, degree, index, &mut record);
            assert_eq!(record, naive[index * n..(index + 1) * n]);
        }
        // Honest rows: Delta = q v, so q divides every cell.
        assert!(naive
            .iter()
            .all(|&value| value.rem_euclid(i64::try_from(FALCON_Q).expect("q")) == 0));
        assert!(naive.iter().any(|&value| value != 0));
    }
}

fn forward_matches_dense(degree: FalconDegree, live_rows: usize) {
    let n = degree.degree();
    let batch = honest_batch(degree, live_rows);
    let geometry = Geometry::new(degree, batch.padded_rows).expect("geometry");
    let cores = ProjectionCores::derive(&SEED, &geometry, live_rows);
    let (norm, residual) = dense_operators(&cores, &geometry);
    let source = naive_norm_source(&batch, n);
    let w = naive_residual(&batch, n);
    let y: Vec<i128> = (0..NORM_TERMINAL_ROWS)
        .map(|row| {
            norm.iter()
                .zip(&source)
                .map(|(column, &value)| column[row] * i128::from(value))
                .sum()
        })
        .collect();
    let z: Vec<i128> = (0..RESIDUAL_WIDTH)
        .map(|row| {
            residual
                .iter()
                .zip(&w)
                .map(|(column, &value)| column[row] * i128::from(value))
                .sum()
        })
        .collect();
    let expected_first_without_slack = naive_first_without_slack(&batch, &geometry, &cores);
    let projection = forward::forward(&batch, &geometry, cores);
    let expected = ProjectionOutputs {
        y: core::array::from_fn(|index| F::from_i128(y[index])),
        z: core::array::from_fn(|index| F::from_i128(z[index])),
    };
    assert_eq!(projection.first_without_slack, expected_first_without_slack);
    assert_eq!(projection.outputs, expected);
    assert!(z.iter().all(|&value| value % i128::from(FALCON_Q) == 0));
    assert!(z.iter().any(|&value| value != 0));
    assert!(check_outputs(degree, &expected).is_ok());
}

#[test]
fn projection_forward_matches_dense_falcon_512() {
    // N = 5, N* = 8: R = 1.
    forward_matches_dense(FalconDegree::Falcon512, 5);
}

#[test]
fn projection_forward_matches_dense_falcon_1024() {
    // N = 3, N* = 4: R = 1.
    forward_matches_dense(FalconDegree::Falcon1024, 3);
}
