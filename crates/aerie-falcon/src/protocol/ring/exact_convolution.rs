//! Exact limb-split negacyclic convolution for the stage-three ring table.

use crate::substrate::{BaseField as F, OpeningField as E};
use akita_algebra::ntt::{
    butterfly::{forward_ntt, inverse_ntt, NttTwiddles},
    prime::MontCoeff,
    tables::Q32_PRIMES,
    NttKernelPlan,
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

fn pow_mod(mut x: i64, mut exp: i64, p: i64) -> i64 {
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

fn table_impl<const D: usize, const FUSE_ROUND0: bool>(
    h: &[i16],
    records: usize,
    live: usize,
    point: &[E],
    round0: Option<Round0Input<'_>>,
) -> (Vec<E>, Option<[E; 3]>) {
    assert_eq!(h.len(), records * D);
    assert_eq!(1 << point.len(), D);
    assert!(live <= records);
    if FUSE_ROUND0 {
        let inputs = round0.expect("fused ring round needs compact sources");
        assert_eq!(inputs.sigma.len(), records * D);
        assert_eq!(inputs.magnitude.len(), records * D);
        assert_eq!(inputs.row_scale.len(), records);
    }

    let plan = NttKernelPlan::detect::<i32>();
    let weights = eq_weights(point);
    let tw = Q32_PRIMES.map(NttTwiddles::<i32, D>::compute);
    let common: [[[MontCoeff<i32>; D]; 4]; 2] = std::array::from_fn(|prime_index| {
        let prime = Q32_PRIMES[prime_index];
        std::array::from_fn(|limb| {
            let mut values = std::array::from_fn(|i| {
                let value = if limb / 2 == 0 {
                    weights[i].c0()
                } else {
                    weights[i].c1()
                };
                let value = ((value.to_canonical_u64() >> (32 * (limb % 2))) & 0xffff_ffff) as i64;
                prime.from_canonical((value % i64::from(prime.p)) as i32)
            });
            forward_ntt(&mut values, prime, &tw[prime_index], plan);
            values
        })
    });
    let [p0, p1] = Q32_PRIMES.map(|prime| i64::from(prime.p));
    let inverse = pow_mod(p0 % p1, p1 - 2, p1);
    let modulus = p0 * p1;
    // At most D signed i16 terms contribute to each 32-bit limb convolution.
    // For D <= 1024 this is below 2^57, so two Q32 primes uniquely recover the
    // centered integer because their product exceeds twice that bound.
    assert!(2_i128 * D as i128 * 32_768 * i128::from(u32::MAX) < i128::from(modulus));
    let centered = |r0: i64, r1: i64| {
        let digit = ((r1 - r0) * inverse).rem_euclid(p1);
        let value = r0 + p0 * digit;
        i128::from(if value > modulus / 2 {
            value - modulus
        } else {
            value
        })
    };
    let fill = |record: usize, out: &mut [E]| -> [Product; 3] {
        if record >= live {
            return [Product::zero(); 3];
        }
        let row = &h[record * D..(record + 1) * D];
        let mut residues = [[[0_i32; D]; 4]; 2];
        for (pi, &prime) in Q32_PRIMES.iter().enumerate() {
            let mut adjoint = std::array::from_fn(|i| {
                let value = if i == 0 {
                    i32::from(row[0])
                } else {
                    -i32::from(row[D - i])
                };
                prime.from_canonical(value.rem_euclid(prime.p))
            });
            forward_ntt(&mut adjoint, prime, &tw[pi], plan);
            for limb in 0..4 {
                let mut product =
                    std::array::from_fn(|i| prime.mul(adjoint[i], common[pi][limb][i]));
                inverse_ntt(&mut product, prime, &tw[pi], plan);
                for (cell, value) in residues[pi][limb].iter_mut().zip(product) {
                    *cell = prime.to_canonical(value);
                }
            }
        }
        let round0_inputs = if FUSE_ROUND0 {
            Some(round0.expect("fused ring round needs compact sources"))
        } else {
            None
        };
        let mut round0_accumulator = Round0Accumulator::zero();
        let mut previous_weight = E::zero();
        let row_start = record * D;
        for (i, value) in out.iter_mut().enumerate() {
            let limbs: [i128; 4] = std::array::from_fn(|limb| {
                centered(
                    i64::from(residues[0][limb][i]),
                    i64::from(residues[1][limb][i]),
                )
            });
            let current = E::new(
                F::from_i128(limbs[0] + (limbs[1] << 32)),
                F::from_i128(limbs[2] + (limbs[3] << 32)),
            );
            *value = current;
            if FUSE_ROUND0 {
                let inputs = round0_inputs.expect("round-0 sources were validated");
                if i & 1 == 0 {
                    previous_weight = current;
                } else {
                    let low = row_start + i - 1;
                    let high = low + 1;
                    round0_accumulator.push_pair(
                        [previous_weight, current],
                        [inputs.sigma.get(low), inputs.sigma.get(high)],
                        [inputs.magnitude[low], inputs.magnitude[high]],
                    );
                }
            }
        }
        if FUSE_ROUND0 {
            round0_accumulator.finish(round0.expect("fused ring round inputs").row_scale[record])
        } else {
            [Product::zero(); 3]
        }
    };

    let (output, coefficients) = if FUSE_ROUND0 {
        #[cfg(feature = "parallel")]
        {
            let (rows, row_products): (Vec<[E; D]>, Vec<[Product; 3]>) = (0..records)
                .into_par_iter()
                .map(|record| {
                    let mut row = [E::zero(); D];
                    let products = fill(record, &mut row);
                    (row, products)
                })
                .unzip();
            let products = row_products.into_par_iter().reduce(
                || [Product::zero(); 3],
                |mut left, right| {
                    for lane in 0..3 {
                        left[lane] += right[lane];
                    }
                    left
                },
            );
            (rows.into_flattened(), Some(products.map(E::reduce_product)))
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut rows = Vec::<[E; D]>::with_capacity(records);
            let mut products = [Product::zero(); 3];
            for record in 0..records {
                let mut row = [E::zero(); D];
                let row_products = fill(record, &mut row);
                rows.push(row);
                for lane in 0..3 {
                    products[lane] += row_products[lane];
                }
            }
            (rows.into_flattened(), Some(products.map(E::reduce_product)))
        }
    } else {
        #[cfg(feature = "parallel")]
        {
            let rows: Vec<[E; D]> = (0..records)
                .into_par_iter()
                .map(|record| {
                    let mut row = [E::zero(); D];
                    fill(record, &mut row);
                    row
                })
                .collect();
            (rows.into_flattened(), None)
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut rows = Vec::<[E; D]>::with_capacity(records);
            for record in 0..records {
                let mut row = [E::zero(); D];
                fill(record, &mut row);
                rows.push(row);
            }
            (rows.into_flattened(), None)
        }
    };
    (output, coefficients)
}

#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub(super) fn table<const D: usize>(h: &[i16], records: usize, live: usize, point: &[E]) -> Vec<E> {
    table_impl::<D, false>(h, records, live, point, None).0
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
    let (table, coefficients) = table_impl::<D, true>(
        h,
        records,
        live,
        point,
        Some(Round0Input {
            sigma,
            magnitude,
            row_scale,
        }),
    );
    (
        table,
        coefficients.expect("round-0 accumulation was requested"),
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
                let actual = table::<512>(&h, records, live, &point);
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
    for value in [E::zero(), E::one()] {
        let point = vec![value; degree.trailing_zeros() as usize];
        let support = if value.is_zero() { 0 } else { degree - 1 };
        let mut h = vec![0_i16; records * degree];
        for row in 0..degree {
            h[row * degree + row] = if row % 2 == 0 { 6_144 } else { -6_144 };
        }
        let actual = table::<512>(&h, records, records, &point);
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
