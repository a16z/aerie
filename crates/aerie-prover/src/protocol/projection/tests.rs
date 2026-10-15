use aerie_falcon_native::{Falcon1024, Falcon512};
use aerie_falcon_statement::DirectParams;
use aerie_falcon_witness::DirectRow;
use jolt_field::Ring as _;

use crate::substrate::BaseField as F;

use super::super::{FalconDegree, FalconWitness};
use super::cores::ProjectionCores;
use super::profile::{Geometry, FIRST_CHANNELS};
use super::*;

const DEGREES: [FalconDegree; 2] = [FalconDegree::Falcon512, FalconDegree::Falcon1024];
const SEED: [u8; 32] = [0x5a; 32];

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
                    i128::from(cores.norm_terminal().get(out, c + NORM_CHANNELS * digit[4]))
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
    assert!(check_outputs(degree, &expected));
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
