//! Exact limb-split negacyclic convolution for the stage-three ring table.

use crate::substrate::{BaseField as F, OpeningField as E};
use akita_algebra::{
    ntt::{
        butterfly::{forward_ntt, inverse_ntt},
        prime::MontCoeff,
        tables::Q32_PRIMES,
    },
    CrtNttParamSet, CyclotomicCrtNtt,
};
use akita_sumcheck::reduce_signed_accum;
use jolt_field::{One, Ring, Unreduced, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::protocol::kernels::SignSource;

type SmallAccum = <E as Unreduced>::SmallProduct;
type Product = <E as Unreduced>::Product;

#[derive(Clone, Copy)]
struct Round0Input<'a> {
    sigma: SignSource<'a>,
    magnitude: &'a [i16],
    row_scale: &'a [E],
}

#[derive(Clone, Copy)]
struct Round0Accumulator {
    positive: [SmallAccum; 3],
    negative: [SmallAccum; 3],
}

impl Round0Accumulator {
    fn zero() -> Self {
        Self {
            positive: [SmallAccum::zero(); 3],
            negative: [SmallAccum::zero(); 3],
        }
    }

    #[inline]
    fn add_signed(&mut self, lane: usize, weight: E, scalar: i64) {
        let product = weight.mul_u64_unreduced(scalar.unsigned_abs());
        if scalar < 0 {
            self.negative[lane] += product;
        } else {
            self.positive[lane] += product;
        }
    }

    #[inline]
    fn push_pair(&mut self, weights: [E; 2], sigma: [i8; 2], magnitude: [i16; 2]) {
        const { assert!(E::SUM_IS_EXACT) };
        let [w0, w1] = weights;
        let [s0, s1] = sigma.map(i64::from);
        let [m0, m1] = magnitude.map(i64::from);
        let dw = w1 - w0;
        let ds = s1 - s0;
        let dm = m1 - m0;
        self.add_signed(0, w0, s0 * m0);
        self.add_signed(1, dw, ds * m0 + s0 * dm);
        self.add_signed(1, w0, ds * dm);
        self.add_signed(2, dw, ds * dm);
    }

    fn finish(self, row_scale: E) -> [Product; 3] {
        std::array::from_fn(|lane| {
            row_scale.mul_unreduced(reduce_signed_accum::<E>(
                self.positive[lane],
                self.negative[lane],
            ))
        })
    }
}

fn eq_weights(point: &[E]) -> Vec<E> {
    let mut weights = vec![E::zero(); 1 << point.len()];
    weights[0] = E::one();
    for (bit, &r) in point.iter().enumerate() {
        for i in 0..1 << bit {
            let high = weights[i] * r;
            weights[i | 1 << bit] = high;
            weights[i] -= high;
        }
    }
    weights
}

const fn pow_mod(mut x: i64, mut exp: i64, p: i64) -> i64 {
    let mut y = 1;
    while exp != 0 {
        if exp & 1 != 0 {
            y = y * x % p;
        }
        x = x * x % p;
        exp >>= 1;
    }
    y
}

/// Residues of one polynomial modulo both primes.
type Residues<const D: usize> = CyclotomicCrtNtt<i32, 2, D>;

struct RowScratch<const D: usize> {
    adjoint: Box<Residues<D>>,
    products: [Box<Residues<D>>; 4],
}

impl<const D: usize> RowScratch<D> {
    fn new() -> Self {
        Self {
            adjoint: Box::new(Residues::zero()),
            products: std::array::from_fn(|_| Box::new(Residues::zero())),
        }
    }
}

fn table_impl<const D: usize>(
    h: &[i16],
    records: usize,
    live: usize,
    point: &[E],
    round0: Round0Input<'_>,
) -> (Vec<E>, [E; 3]) {
    assert_eq!(h.len(), records * D);
    assert_eq!(1 << point.len(), D);
    assert!(live <= records);
    assert_eq!(round0.sigma.len(), records * D);
    assert_eq!(round0.magnitude.len(), records * D);
    assert_eq!(round0.row_scale.len(), records);

    let params = CrtNttParamSet::<i32, 2, D>::new(Q32_PRIMES);
    let plan = params.kernel_plan();
    let tw = &params.twiddles;
    let weights = eq_weights(point);
    let common: [Box<Residues<D>>; 4] = std::array::from_fn(|limb| {
        let mut common = Box::new(Residues::zero());
        for (pi, &prime) in Q32_PRIMES.iter().enumerate() {
            let values = &mut common.limbs[pi];
            for (i, cell) in values.iter_mut().enumerate() {
                let value = if limb / 2 == 0 {
                    weights[i].c0()
                } else {
                    weights[i].c1()
                };
                let value = ((value.to_canonical_u64() >> (32 * (limb % 2))) & 0xffff_ffff) as i64;
                *cell = prime.from_canonical((value % i64::from(prime.p)) as i32);
            }
            forward_ntt(values, prime, &tw[pi], plan);
        }
        common
    });
    const P0: i64 = Q32_PRIMES[0].p as i64;
    const P1: i64 = Q32_PRIMES[1].p as i64;
    const INVERSE: i64 = pow_mod(P0 % P1, P1 - 2, P1);
    const MODULUS: i64 = P0 * P1;
    const BIAS: i128 = 1 << 90;
    let bias = F::from_u128(BIAS.cast_unsigned());
    // At most D signed i16 terms contribute to each 32-bit limb convolution.
    // For D <= 1024 this is below 2^57, so two Q32 primes uniquely recover the
    // centered integer because their product exceeds twice that bound.
    assert!(2_i128 * D as i128 * 32_768 * i128::from(u32::MAX) < i128::from(MODULUS));
    // The residues may be any representatives in `(-p, p)`.
    let centered = |r0: MontCoeff<i32>, r1: MontCoeff<i32>| {
        let (r0, r1) = (i64::from(r0.raw()), i64::from(r1.raw()));
        let digit = ((r1 - r0) * INVERSE).rem_euclid(P1);
        let value = r0 + P0 * digit;
        i128::from(if value > MODULUS / 2 {
            value - MODULUS
        } else {
            value
        })
    };
    let fill = |scratch: &mut RowScratch<D>, record: usize, out: &mut [E]| -> [Product; 3] {
        if record >= live {
            return [Product::zero(); 3];
        }
        let out: &mut [E; D] = out.try_into().expect("row has D cells");
        let row = &h[record * D..(record + 1) * D];
        let RowScratch { adjoint, products } = scratch;
        // The signed coefficients enter the transform without a Montgomery
        // factor, so each product with the Montgomery-form `common` is a plain
        // residue, and so is its inverse transform.
        for (pi, &prime) in Q32_PRIMES.iter().enumerate() {
            let values = &mut adjoint.limbs[pi];
            values[0] = MontCoeff::from_raw(i32::from(row[0]));
            for (cell, &value) in values[1..].iter_mut().zip(row[1..].iter().rev()) {
                *cell = MontCoeff::from_raw(-i32::from(value));
            }
            forward_ntt(values, prime, &tw[pi], plan);
        }
        for (product, common) in products.iter_mut().zip(&common) {
            // Akita exposes only the accumulating pointwise product.
            product.limbs.as_flattened_mut().fill(MontCoeff::default());
            product.add_assign_pointwise_mul(adjoint, common, &params);
            for (pi, &prime) in Q32_PRIMES.iter().enumerate() {
                inverse_ntt(&mut product.limbs[pi], prime, &tw[pi], plan);
            }
        }
        let limb = |limb: usize, i: usize| {
            let [r0, r1] = &products[limb].limbs;
            centered(r0[i], r1[i])
        };
        // A limb pair is below 2^90 in magnitude; the bias keeps its reduction
        // on the nonnegative path.
        let coordinate = |low: i128, high: i128| {
            F::from_u128((low + (high << 32) + BIAS).cast_unsigned()) - bias
        };
        let mut round0_accumulator = Round0Accumulator::zero();
        let mut previous_weight = E::zero();
        let row_start = record * D;
        for (i, value) in out.iter_mut().enumerate() {
            let current = E::new(
                coordinate(limb(0, i), limb(1, i)),
                coordinate(limb(2, i), limb(3, i)),
            );
            *value = current;
            if i & 1 == 0 {
                previous_weight = current;
            } else {
                let low = row_start + i - 1;
                let high = low + 1;
                round0_accumulator.push_pair(
                    [previous_weight, current],
                    [round0.sigma.get(low), round0.sigma.get(high)],
                    [round0.magnitude[low], round0.magnitude[high]],
                );
            }
        }
        round0_accumulator.finish(round0.row_scale[record])
    };

    // Zero the table on the worker threads, which then first touch its pages.
    #[cfg(feature = "parallel")]
    let mut output: Vec<E> = (0..records * D)
        .into_par_iter()
        .map(|_| E::zero())
        .collect();
    #[cfg(not(feature = "parallel"))]
    let mut output = vec![E::zero(); records * D];
    let add = |mut left: [Product; 3], right: [Product; 3]| {
        for lane in 0..3 {
            left[lane] += right[lane];
        }
        left
    };
    #[cfg(feature = "parallel")]
    let products = output
        .par_chunks_mut(D)
        .enumerate()
        .with_min_len(16)
        .map_init(RowScratch::new, |scratch, (record, row)| {
            fill(scratch, record, row)
        })
        .reduce(|| [Product::zero(); 3], add);
    #[cfg(not(feature = "parallel"))]
    let products = {
        let mut scratch = RowScratch::new();
        output
            .chunks_mut(D)
            .enumerate()
            .map(|(record, row)| fill(&mut scratch, record, row))
            .fold([Product::zero(); 3], add)
    };
    (output, products.map(E::reduce_product))
}

pub(super) fn table_with_round0<const D: usize>(
    h: &[i16],
    records: usize,
    live: usize,
    point: &[E],
    sigma: SignSource<'_>,
    magnitude: &[i16],
    row_scale: &[E],
) -> (Vec<E>, [E; 3]) {
    table_impl::<D>(
        h,
        records,
        live,
        point,
        Round0Input {
            sigma,
            magnitude,
            row_scale,
        },
    )
}

#[cfg(test)]
fn point(bits: usize) -> Vec<E> {
    let mut state = 0x2147_2578_93ad_72df_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        F::from_u64(state)
    };
    (0..bits).map(|_| E::new(next(), next())).collect()
}

#[cfg(test)]
fn coefficient_boundary_point() -> Vec<E> {
    let mut state = 0x941a_531b_721d_86bf_u64;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        F::from_u64(value ^ (value >> 31))
    };
    (0..9).map(|_| E::new(next(), next())).collect()
}

#[cfg(test)]
fn direct_eq_weights(point: &[E]) -> Vec<E> {
    (0..1 << point.len())
        .map(|index| {
            point
                .iter()
                .enumerate()
                .fold(E::one(), |weight, (bit, &coordinate)| {
                    if index & (1 << bit) == 0 {
                        weight * (E::one() - coordinate)
                    } else {
                        weight * coordinate
                    }
                })
        })
        .collect()
}

#[cfg(test)]
fn direct_kernel_row(h: &[i16], weights: &[E]) -> Vec<E> {
    let degree = h.len();
    (0..degree)
        .map(|j| {
            weights
                .iter()
                .enumerate()
                .fold(E::zero(), |sum, (u, &weight)| {
                    let coefficient = if u >= j {
                        i64::from(h[u - j])
                    } else {
                        -i64::from(h[degree + u - j])
                    };
                    sum + weight * E::from_i64(coefficient)
                })
        })
        .collect()
}

#[cfg(test)]
#[test]
fn exact_for_every_prefix_and_coefficient_boundary() {
    let point = coefficient_boundary_point();
    let weights = direct_eq_weights(&point);
    for records in [1, 2, 8] {
        let sigma = vec![1_i8; records * 512];
        let magnitude = vec![0_i16; records * 512];
        let row_scale = vec![E::zero(); records];
        let cases: Vec<Vec<i16>> = vec![
            vec![0; records * 512],
            vec![6_144; records * 512],
            (0..records * 512)
                .map(|i| if i % 2 == 0 { -6_144 } else { 6_144 })
                .collect(),
            (0..records * 512)
                .map(|i| ((i * 7_919 + 71) % 12_289) as i16 - 6_144)
                .collect(),
            (0..records * 512)
                .map(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX })
                .collect(),
        ];
        for h in cases {
            for live in [0, records / 2, records] {
                let (actual, _) = table_with_round0::<512>(
                    &h,
                    records,
                    live,
                    &point,
                    SignSource::Direct(&sigma),
                    &magnitude,
                    &row_scale,
                );
                for record in 0..records {
                    let start = record * 512;
                    let expected = if record < live {
                        direct_kernel_row(&h[start..start + 512], &weights)
                    } else {
                        vec![E::zero(); 512]
                    };
                    assert_eq!(
                        &actual[start..start + 512],
                        expected.as_slice(),
                        "records={records} live={live} row={record}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
#[test]
fn degenerate_points_and_each_basis_coefficient() {
    let degree: usize = 512;
    let records = degree;
    let sigma = vec![1_i8; records * degree];
    let magnitude = vec![0_i16; records * degree];
    let row_scale = vec![E::zero(); records];
    for value in [E::zero(), E::one()] {
        let point = vec![value; degree.trailing_zeros() as usize];
        let support = if value.is_zero() { 0 } else { degree - 1 };
        let mut h = vec![0_i16; records * degree];
        for row in 0..degree {
            h[row * degree + row] = if row % 2 == 0 { 6_144 } else { -6_144 };
        }
        let (actual, _) = table_with_round0::<512>(
            &h,
            records,
            records,
            &point,
            SignSource::Direct(&sigma),
            &magnitude,
            &row_scale,
        );
        for row in 0..records {
            for j in 0..degree {
                let coefficient = if support >= j {
                    i64::from(h[row * degree + support - j])
                } else {
                    -i64::from(h[row * degree + degree + support - j])
                };
                assert_eq!(
                    actual[row * degree + j],
                    E::from_i64(coefficient),
                    "point={value:?} row={row} coefficient={j}"
                );
            }
        }
    }
}

#[cfg(test)]
#[test]
fn ntt_table_and_fused_round0_match_direct_formulas() {
    fn check<const D: usize>() {
        let records = 2;
        let live_rows = [0, 1, records];
        let points = [
            vec![E::zero(); D.trailing_zeros() as usize],
            vec![E::one(); D.trailing_zeros() as usize],
            point(D.trailing_zeros() as usize),
        ];
        let h: Vec<i16> = (0..records * D)
            .map(|i| match i % 5 {
                0 => i16::MIN,
                1 => i16::MAX,
                _ => ((i * 7919) % 12289) as i16 - 6144,
            })
            .collect();
        let sigma: Vec<i8> = (0..records * D)
            .map(|i| match i % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();
        let magnitude: Vec<i16> = (0..records * D)
            .map(|i| match i % 5 {
                0 => i16::MIN,
                1 => i16::MAX,
                _ => i16::try_from((i * 37 + 19) % 2048).unwrap(),
            })
            .collect();
        let row_scale = (0..records)
            .map(|i| E::from_i64(i64::try_from(i).unwrap() + 5))
            .collect::<Vec<_>>();

        for point in points {
            let eq = eq_weights(&point);
            for live in live_rows {
                let (table, coefficients) = table_with_round0::<D>(
                    &h,
                    records,
                    live,
                    &point,
                    SignSource::Direct(&sigma),
                    &magnitude,
                    &row_scale,
                );
                for row in 0..records {
                    let start = row * D;
                    if row >= live {
                        assert!(table[start..start + D].iter().all(Zero::is_zero));
                        continue;
                    }
                    for j in 0..D {
                        let expected = (0..D).fold(E::zero(), |sum, u| {
                            let coefficient = if u >= j {
                                i64::from(h[start + u - j])
                            } else {
                                -i64::from(h[start + D + u - j])
                            };
                            sum + eq[u] * E::from_i64(coefficient)
                        });
                        assert_eq!(table[start + j], expected, "D={D},row={row},j={j}");
                    }
                }

                let mut expected = [E::zero(); 3];
                for (row, &scale) in row_scale.iter().enumerate().take(live) {
                    let start = row * D;
                    for pair in 0..D / 2 {
                        let low = start + 2 * pair;
                        let high = low + 1;
                        let w0 = table[low];
                        let dw = table[high] - w0;
                        let s0 = i64::from(sigma[low]);
                        let ds = i64::from(sigma[high]) - s0;
                        let m0 = i64::from(magnitude[low]);
                        let dm = i64::from(magnitude[high]) - m0;
                        expected[0] += scale * w0 * E::from_i64(s0 * m0);
                        expected[1] += scale
                            * (dw * E::from_i64(ds * m0 + s0 * dm) + w0 * E::from_i64(ds * dm));
                        expected[2] += scale * dw * E::from_i64(ds * dm);
                    }
                }
                assert_eq!(
                    coefficients, expected,
                    "fused coefficients D={D}, live={live}"
                );
            }
        }
    }
    check::<512>();
    check::<1024>();
}
