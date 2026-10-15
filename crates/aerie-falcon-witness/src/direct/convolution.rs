//! Exact integer convolutions for both Falcon degrees.
//!
//! Every product is reconstructed from two NTT residues by the Chinese
//! remainder theorem: the wide prime `Q32_PRIMES[0] = 1,073,692,673`
//! (`2^14 | p - 1`, so it carries cyclic transforms through length 16,384)
//! and Falcon's own `q = 12,289`. The centered CRT interval covers
//! `n * 2^30` for both degrees, which bounds every `i16 * i16` convolution.

use std::sync::OnceLock;

use aerie_falcon_native::{Falcon1024, Falcon512, FalconParams};
use akita_algebra::{
    ntt::{
        butterfly::{forward_ntt, forward_ntt_cyclic, inverse_ntt, inverse_ntt_cyclic},
        tables::{I16_TAIL_PRIME, Q32_PRIMES},
    },
    CrtNttParamSet, I16TailParams, MontCoeff,
};

use super::FALCON_Q;

const CRT_P0: i64 = Q32_PRIMES[0].p as i64;
const CRT_P1: i64 = I16_TAIL_PRIME.p as i64;
const CRT_P0_INVERSE_MOD_P1: i64 = 1_353;
const CRT_MODULUS: i64 = CRT_P0 * CRT_P1;
const CRT_HALF_MODULUS: i64 = CRT_MODULUS / 2;

#[allow(clippy::cast_possible_wrap, reason = "degree is 512 or 1,024")]
const fn universal_i16_convolution_abs_bound(degree: usize) -> i64 {
    degree as i64 * (1_i64 << 15) * (1_i64 << 15)
}

const _: () = assert!(2 * universal_i16_convolution_abs_bound(1024) < CRT_MODULUS);
const _: () = assert!(CRT_P1 == FALCON_Q);
const _: () = assert!((CRT_P0 - 1) % 4096 == 0, "wide prime lacks 4,096th roots");

/// Exact integer convolutions for one parameter set.
trait ExactConvolution: FalconParams {
    /// Exact ordinary convolution of two degree-`n` polynomials: `2n` cells,
    /// the last of which is zero.
    fn convolution_i16(a: &[i16], b: &[i16]) -> Vec<i64>;
    /// Exact negacyclic convolution in `Z[X] / (X^n + 1)`.
    fn convolution_negacyclic_i16(a: &[i16], b: &[i16]) -> Vec<i64>;
    /// Exact negacyclic convolution reusing a product already verified
    /// modulo Falcon's `q`, which is also the CRT tail prime.
    fn convolution_negacyclic_i16_from_mod_q(
        a: &[i16],
        b: &[i16],
        product_mod_q: &[i16],
    ) -> Vec<i64>;
}

/// Exact convolution selected by the sealed parameter set's constant degree.
#[doc(hidden)]
#[must_use]
#[inline]
pub fn convolution_i16<P: FalconParams>(a: &[i16], b: &[i16]) -> Vec<i64> {
    match P::DEGREE {
        Falcon512::DEGREE => <Falcon512 as ExactConvolution>::convolution_i16(a, b),
        Falcon1024::DEGREE => <Falcon1024 as ExactConvolution>::convolution_i16(a, b),
        _ => unreachable!("sealed Falcon parameter degree"),
    }
}

/// Exact convolution selected by the sealed parameter set's constant degree.
#[doc(hidden)]
#[must_use]
#[inline]
pub fn convolution_negacyclic_i16<P: FalconParams>(a: &[i16], b: &[i16]) -> Vec<i64> {
    match P::DEGREE {
        Falcon512::DEGREE => <Falcon512 as ExactConvolution>::convolution_negacyclic_i16(a, b),
        Falcon1024::DEGREE => <Falcon1024 as ExactConvolution>::convolution_negacyclic_i16(a, b),
        _ => unreachable!("sealed Falcon parameter degree"),
    }
}

/// Exact convolution selected by the sealed parameter set's constant degree.
#[doc(hidden)]
#[must_use]
#[inline]
pub fn convolution_negacyclic_i16_from_mod_q<P: FalconParams>(
    a: &[i16],
    b: &[i16],
    product_mod_q: &[i16],
) -> Vec<i64> {
    match P::DEGREE {
        Falcon512::DEGREE => {
            <Falcon512 as ExactConvolution>::convolution_negacyclic_i16_from_mod_q(
                a,
                b,
                product_mod_q,
            )
        }
        Falcon1024::DEGREE => {
            <Falcon1024 as ExactConvolution>::convolution_negacyclic_i16_from_mod_q(
                a,
                b,
                product_mod_q,
            )
        }
        _ => unreachable!("sealed Falcon parameter degree"),
    }
}

fn reconstruct(wide_residue: i32, tail_residue: i16, bound: i64) -> i64 {
    let residue0 = i64::from(wide_residue);
    let residue1 = i64::from(tail_residue);
    let digit1 =
        ((residue1 - residue0).rem_euclid(CRT_P1) * CRT_P0_INVERSE_MOD_P1).rem_euclid(CRT_P1);
    let reconstructed = residue0 + CRT_P0 * digit1;
    let centered = if reconstructed > CRT_HALF_MODULUS {
        reconstructed - CRT_MODULUS
    } else {
        reconstructed
    };
    debug_assert!(centered.unsigned_abs() <= bound as u64);
    let _ = bound;
    centered
}

fn sparse_negacyclic_i16(a: &[i16], b: &[i16], degree: usize) -> Option<Vec<i64>> {
    let nonzero_a = a.iter().filter(|&&coefficient| coefficient != 0).count();
    if nonzero_a > 32 {
        return None;
    }
    let mut output = vec![0_i64; degree];
    for (left_index, &left) in a
        .iter()
        .enumerate()
        .filter(|(_, coefficient)| **coefficient != 0)
    {
        for (right_index, &right) in b.iter().enumerate() {
            let product = i64::from(left) * i64::from(right);
            let index = left_index + right_index;
            if index < degree {
                output[index] += product;
            } else {
                output[index - degree] -= product;
            }
        }
    }
    Some(output)
}

fn sparse_ordinary_i16(a: &[i16], b: &[i16], degree: usize) -> Option<Vec<i64>> {
    let nonzero_a = a.iter().filter(|&&coefficient| coefficient != 0).count();
    if nonzero_a > 32 {
        return None;
    }
    let mut output = vec![0_i64; 2 * degree];
    for (left_index, &left) in a
        .iter()
        .enumerate()
        .filter(|(_, coefficient)| **coefficient != 0)
    {
        for (right_index, &right) in b.iter().enumerate() {
            output[left_index + right_index] += i64::from(left) * i64::from(right);
        }
    }
    Some(output)
}

macro_rules! exact_convolution_impl {
    ($params:ty, $degree:literal, $double:literal, $negacyclic_cell:ident, $ordinary_cell:ident) => {
        static $negacyclic_cell: OnceLock<I16TailParams<1, $degree>> = OnceLock::new();
        static $ordinary_cell: OnceLock<I16TailParams<1, $double>> = OnceLock::new();

        impl ExactConvolution for $params {
            fn convolution_i16(a: &[i16], b: &[i16]) -> Vec<i64> {
                assert_eq!(a.len(), $degree);
                assert_eq!(b.len(), $degree);
                if let Some(output) = sparse_ordinary_i16(a, b, $degree) {
                    return output;
                }
                let params = $ordinary_cell.get_or_init(new_params::<$double>);
                let wide = {
                    let prime = params.wide.primes[0];
                    let twiddles = &params.wide.twiddles[0];
                    let mut lhs = [MontCoeff::default(); $double];
                    let mut rhs = [MontCoeff::default(); $double];
                    load_wide(&mut lhs, a, prime);
                    load_wide(&mut rhs, b, prime);
                    forward_ntt_cyclic(&mut lhs, prime, twiddles, params.wide.kernel_plan());
                    forward_ntt_cyclic(&mut rhs, prime, twiddles, params.wide.kernel_plan());
                    for (left, right) in lhs.iter_mut().zip(rhs) {
                        *left = prime.reduce_range(prime.mul(*left, right));
                    }
                    inverse_ntt_cyclic(&mut lhs, prime, twiddles, params.wide.kernel_plan());
                    lhs.map(|coefficient| prime.to_canonical(coefficient))
                };
                let tail = {
                    let prime = params.tail.primes[0];
                    let twiddles = &params.tail.twiddles[0];
                    let mut lhs = [MontCoeff::default(); $double];
                    let mut rhs = [MontCoeff::default(); $double];
                    load_tail(&mut lhs, a, prime);
                    load_tail(&mut rhs, b, prime);
                    forward_ntt_cyclic(&mut lhs, prime, twiddles, params.tail.kernel_plan());
                    forward_ntt_cyclic(&mut rhs, prime, twiddles, params.tail.kernel_plan());
                    for (left, right) in lhs.iter_mut().zip(rhs) {
                        *left = prime.reduce_range(prime.mul(*left, right));
                    }
                    inverse_ntt_cyclic(&mut lhs, prime, twiddles, params.tail.kernel_plan());
                    lhs.map(|coefficient| prime.to_canonical(coefficient))
                };
                let bound = universal_i16_convolution_abs_bound($degree);
                let mut output = vec![0_i64; $double];
                for index in 0..$double - 1 {
                    output[index] = reconstruct(wide[index], tail[index], bound);
                }
                output
            }

            fn convolution_negacyclic_i16(a: &[i16], b: &[i16]) -> Vec<i64> {
                assert_eq!(a.len(), $degree);
                assert_eq!(b.len(), $degree);
                if let Some(output) = sparse_negacyclic_i16(a, b, $degree) {
                    return output;
                }
                let params = $negacyclic_cell.get_or_init(new_params::<$degree>);
                let prime = params.tail.primes[0];
                let twiddles = &params.tail.twiddles[0];
                let mut lhs = [MontCoeff::default(); $degree];
                let mut rhs = [MontCoeff::default(); $degree];
                load_tail(&mut lhs, a, prime);
                load_tail(&mut rhs, b, prime);
                forward_ntt(&mut lhs, prime, twiddles, params.tail.kernel_plan());
                forward_ntt(&mut rhs, prime, twiddles, params.tail.kernel_plan());
                for (left, right) in lhs.iter_mut().zip(rhs) {
                    *left = prime.reduce_range(prime.mul(*left, right));
                }
                inverse_ntt(&mut lhs, prime, twiddles, params.tail.kernel_plan());
                let tail = lhs.map(|coefficient| prime.to_canonical(coefficient));
                dense_negacyclic_from_tail::<$degree>(a, b, &tail, params)
            }

            fn convolution_negacyclic_i16_from_mod_q(
                a: &[i16],
                b: &[i16],
                product_mod_q: &[i16],
            ) -> Vec<i64> {
                assert_eq!(a.len(), $degree);
                assert_eq!(b.len(), $degree);
                assert_eq!(product_mod_q.len(), $degree);
                if let Some(output) = sparse_negacyclic_i16(a, b, $degree) {
                    debug_assert!(output.iter().zip(product_mod_q).all(|(&value, &residue)| {
                        value.rem_euclid(CRT_P1) == i64::from(residue).rem_euclid(CRT_P1)
                    }));
                    return output;
                }
                let params = $negacyclic_cell.get_or_init(new_params::<$degree>);
                dense_negacyclic_from_tail::<$degree>(a, b, product_mod_q, params)
            }
        }
    };
}

exact_convolution_impl!(Falcon512, 512, 1024, NEGACYCLIC_512, ORDINARY_512);
exact_convolution_impl!(Falcon1024, 1024, 2048, NEGACYCLIC_1024, ORDINARY_1024);

fn new_params<const D: usize>() -> I16TailParams<1, D> {
    assert_eq!(
        (CRT_P0 * CRT_P0_INVERSE_MOD_P1).rem_euclid(CRT_P1),
        1,
        "pinned exact-convolution CRT inverse changed"
    );
    assert_eq!(
        i64::from(I16_TAIL_PRIME.p),
        FALCON_Q,
        "exact negacyclic CRT tail prime no longer matches Falcon q"
    );
    I16TailParams::new(
        CrtNttParamSet::new([Q32_PRIMES[0]]),
        CrtNttParamSet::new([I16_TAIL_PRIME]),
    )
}

fn load_wide<const D: usize>(
    destination: &mut [MontCoeff<i32>; D],
    source: &[i16],
    prime: akita_algebra::NttPrime<i32>,
) {
    for (destination, &coefficient) in destination.iter_mut().zip(source) {
        let canonical = i64::from(coefficient).rem_euclid(i64::from(prime.p)) as i32;
        *destination = prime.from_canonical(canonical);
    }
}

fn load_tail<const D: usize>(
    destination: &mut [MontCoeff<i16>; D],
    source: &[i16],
    prime: akita_algebra::NttPrime<i16>,
) {
    for (destination, &coefficient) in destination.iter_mut().zip(source) {
        let canonical = i64::from(coefficient).rem_euclid(i64::from(prime.p)) as i16;
        *destination = prime.from_canonical(canonical);
    }
}

fn dense_negacyclic_from_tail<const D: usize>(
    a: &[i16],
    b: &[i16],
    tail_residues: &[i16],
    params: &I16TailParams<1, D>,
) -> Vec<i64> {
    let prime = params.wide.primes[0];
    let twiddles = &params.wide.twiddles[0];
    let mut lhs = [MontCoeff::default(); D];
    let mut rhs = [MontCoeff::default(); D];
    load_wide(&mut lhs, a, prime);
    load_wide(&mut rhs, b, prime);
    forward_ntt(&mut lhs, prime, twiddles, params.wide.kernel_plan());
    forward_ntt(&mut rhs, prime, twiddles, params.wide.kernel_plan());
    for (left, right) in lhs.iter_mut().zip(rhs) {
        *left = prime.reduce_range(prime.mul(*left, right));
    }
    inverse_ntt(&mut lhs, prime, twiddles, params.wide.kernel_plan());
    let bound = universal_i16_convolution_abs_bound(D);
    lhs.iter()
        .zip(tail_residues)
        .map(|(&wide, &tail)| reconstruct(prime.to_canonical(wide), tail, bound))
        .collect()
}

pub(super) fn prepare() {
    let _ = ORDINARY_512.get_or_init(new_params::<1024>);
    let _ = NEGACYCLIC_512.get_or_init(new_params::<512>);
    let _ = ORDINARY_1024.get_or_init(new_params::<2048>);
    let _ = NEGACYCLIC_1024.get_or_init(new_params::<1024>);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn negacyclic_from_ordinary(product: &[i64], degree: usize) -> Vec<i64> {
        (0..degree)
            .map(|index| product[index] - product[index + degree])
            .collect()
    }

    fn residues_mod_q(product: &[i64]) -> Vec<i16> {
        product
            .iter()
            .map(|coefficient| coefficient.rem_euclid(CRT_P1) as i16)
            .collect()
    }

    fn extremes(degree: usize) -> (Vec<i16>, Vec<i16>) {
        let left = (0..degree)
            .map(|index| match index % 4 {
                0 => i16::MIN,
                1 => i16::MAX,
                2 => -1,
                _ => 1,
            })
            .collect();
        let right = (0..degree)
            .map(|index| match index % 5 {
                0 => i16::MAX,
                1 => i16::MIN,
                2 => 17,
                3 => -31,
                _ => 1,
            })
            .collect();
        (left, right)
    }

    fn schoolbook(left: &[i16], right: &[i16]) -> Vec<i64> {
        let mut want = vec![0_i64; 2 * left.len()];
        for (left_index, &left) in left.iter().enumerate() {
            for (right_index, &right) in right.iter().enumerate() {
                want[left_index + right_index] += i64::from(left) * i64::from(right);
            }
        }
        want
    }

    fn check_degree<P: FalconParams>() {
        let degree = P::DEGREE;
        let (left, right) = extremes(degree);
        let got = convolution_i16::<P>(&left, &right);
        let want = schoolbook(&left, &right);
        assert_eq!(
            got, want,
            "ordinary CRT-NTT at i16 extremes, degree {degree}"
        );
        let negacyclic = negacyclic_from_ordinary(&want, degree);
        assert_eq!(convolution_negacyclic_i16::<P>(&left, &right), negacyclic);
        assert_eq!(
            convolution_negacyclic_i16_from_mod_q::<P>(&left, &right, &residues_mod_q(&negacyclic)),
            negacyclic,
        );

        // Sparse path.
        let mut sparse_left = vec![0_i16; degree];
        sparse_left[0] = -7;
        sparse_left[degree / 2 + 1] = 11;
        sparse_left[degree - 1] = -13;
        let right = (0..degree)
            .map(|index| i16::try_from(index % 31).expect("remainder fits i16") - 15)
            .collect::<Vec<_>>();
        let ordinary = convolution_i16::<P>(&sparse_left, &right);
        assert_eq!(ordinary, schoolbook(&sparse_left, &right));
        let want = negacyclic_from_ordinary(&ordinary, degree);
        assert_eq!(convolution_negacyclic_i16::<P>(&sparse_left, &right), want);

        // Dense random inputs through the reused-residue path.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..4 {
            let mut next = || {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut value = state;
                value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                value ^ (value >> 31)
            };
            let left = (0..degree)
                .map(|_| (next() as i16).wrapping_rem(12_289))
                .collect::<Vec<_>>();
            let right = (0..degree)
                .map(|_| (next() as i16).wrapping_rem(8_192))
                .collect::<Vec<_>>();
            let full = convolution_negacyclic_i16::<P>(&left, &right);
            assert_eq!(
                full,
                negacyclic_from_ordinary(&schoolbook(&left, &right), degree)
            );
            let reused =
                convolution_negacyclic_i16_from_mod_q::<P>(&left, &right, &residues_mod_q(&full));
            assert_eq!(reused, full);
        }
    }

    #[test]
    fn falcon_512_convolutions_match_schoolbook() {
        check_degree::<Falcon512>();
    }

    #[test]
    fn falcon_1024_convolutions_match_schoolbook() {
        check_degree::<Falcon1024>();
    }
}
