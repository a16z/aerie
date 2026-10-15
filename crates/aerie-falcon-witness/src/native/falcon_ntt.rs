//! Negacyclic multiplication in `Z_q[X] / (X^n + 1)` for both Falcon degrees.
//!
//! One transform plan per degree is built on first use. Transforms run in
//! place on inline arrays. The Falcon modulus
//! `q - 1 = 2^12 * 3` supplies primitive `2n`-th roots of unity for both
//! `n = 512` and `n = 1024`.

use std::sync::OnceLock;

use super::params::{filled, FalconParams, FALCON_Q};

static PLAN_512: OnceLock<NegacyclicPlan> = OnceLock::new();
static PLAN_1024: OnceLock<NegacyclicPlan> = OnceLock::new();

/// Bit-reversed powers of a primitive `2n`-th root `psi`, as used by the
/// merged negacyclic transforms of Longa and Naehrig (CANS 2016). The forward
/// transform maps natural order to bit-reversed order and the inverse maps
/// back, so neither needs a twist or a permutation pass.
struct NegacyclicPlan {
    degree: usize,
    stages: usize,
    forward_roots: Vec<u32>,
    inverse_roots: Vec<u32>,
    degree_inverse: u32,
}

impl NegacyclicPlan {
    fn new(degree: usize) -> Self {
        let modulus = FALCON_Q as u32;
        let degree_u32 = u32::try_from(degree).expect("Falcon degree fits u32");
        assert!(degree.is_power_of_two() && (modulus - 1).is_multiple_of(2 * degree_u32));
        let exponent = (modulus - 1) / (2 * degree_u32);
        let psi = (2..modulus)
            .map(|candidate| pow_mod(candidate, exponent))
            .find(|root| pow_mod(*root, degree_u32) == modulus - 1)
            .expect("Falcon modulus has a primitive 2n-th root");
        let psi_inverse = pow_mod(psi, modulus - 2);
        let stages = degree.trailing_zeros() as usize;
        Self {
            degree,
            stages,
            forward_roots: bit_reversed_powers(psi, degree, stages),
            inverse_roots: bit_reversed_powers(psi_inverse, degree, stages),
            degree_inverse: pow_mod(degree_u32, modulus - 2),
        }
    }

    /// Cooley-Tukey butterflies from natural order to bit-reversed order.
    fn forward(&self, values: &mut [u32]) {
        debug_assert_eq!(values.len(), self.degree);
        let mut half = self.degree;
        let mut blocks = 1;
        while blocks < self.degree {
            half /= 2;
            for (block, root) in values
                .chunks_exact_mut(2 * half)
                .zip(&self.forward_roots[blocks..2 * blocks])
            {
                let (low, high) = block.split_at_mut(half);
                for (even, odd) in low.iter_mut().zip(high) {
                    let product = mul_mod(*odd, *root);
                    let original = *even;
                    *even = add_mod(original, product);
                    *odd = sub_mod(original, product);
                }
            }
            blocks *= 2;
        }
    }

    /// Gentleman-Sande butterflies from bit-reversed order to natural order,
    /// without the final `1/n` scaling.
    fn inverse_unscaled(&self, values: &mut [u32]) {
        debug_assert_eq!(values.len(), self.degree);
        let mut half = 1;
        let mut blocks = self.degree / 2;
        while blocks >= 1 {
            for (block, root) in values
                .chunks_exact_mut(2 * half)
                .zip(&self.inverse_roots[blocks..2 * blocks])
            {
                let (low, high) = block.split_at_mut(half);
                for (even, odd) in low.iter_mut().zip(high) {
                    let (left, right) = (*even, *odd);
                    *even = add_mod(left, right);
                    *odd = mul_mod(sub_mod(left, right), *root);
                }
            }
            half *= 2;
            blocks /= 2;
        }
    }
}

fn plan<P: FalconParams>() -> &'static NegacyclicPlan {
    let cell = match P::LOGN {
        9 => &PLAN_512,
        10 => &PLAN_1024,
        _ => unreachable!("FalconParams is sealed to logn 9 and 10"),
    };
    let plan = cell.get_or_init(|| NegacyclicPlan::new(P::DEGREE));
    debug_assert_eq!(plan.stages, P::LOGN as usize);
    plan
}

/// Multiply centered Falcon polynomials in `Z_q[X] / (X^n + 1)`.
pub(crate) fn negacyclic_product<P: FalconParams>(
    left: &P::Array<i16>,
    right: &P::Array<i16>,
) -> P::Array<i16> {
    let plan = plan::<P>();
    let mut left_ntt = canonical_residues::<P>(left);
    let mut right_ntt = canonical_residues::<P>(right);
    plan.forward(left_ntt.as_mut());
    plan.forward(right_ntt.as_mut());
    for (left, right) in left_ntt.as_mut().iter_mut().zip(right_ntt.as_ref()) {
        *left = mul_mod(*left, *right);
    }
    plan.inverse_unscaled(left_ntt.as_mut());
    let mut output = filled::<P, i16>(0);
    for (slot, residue) in output.as_mut().iter_mut().zip(left_ntt.as_ref()) {
        *slot = center_residue(mul_mod(*residue, plan.degree_inverse));
    }
    output
}

fn canonical_residues<P: FalconParams>(values: &P::Array<i16>) -> P::Array<u32> {
    let mut residues = filled::<P, u32>(0);
    for (residue, value) in residues.as_mut().iter_mut().zip(values.as_ref()) {
        *residue = canonical_residue(*value);
    }
    residues
}

/// `root^bitrev(k)` for `k` in `0..degree`, with `stages`-bit reversal.
fn bit_reversed_powers(root: u32, degree: usize, stages: usize) -> Vec<u32> {
    let mut powers = vec![0; degree];
    let mut current = 1;
    for index in 0..degree {
        powers[index.reverse_bits() >> (usize::BITS as usize - stages)] = current;
        current = mul_mod(current, root);
    }
    powers
}

fn pow_mod(mut base: u32, mut exponent: u32) -> u32 {
    let mut result = 1;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = mul_mod(result, base);
        }
        base = mul_mod(base, base);
        exponent >>= 1;
    }
    result
}

fn mul_mod(left: u32, right: u32) -> u32 {
    // Every caller supplies canonical residues. Thus each operand is below
    // q = 12_289, and the largest product, 12_288^2, fits in 28 bits.
    debug_assert!(left < FALCON_Q as u32);
    debug_assert!(right < FALCON_Q as u32);
    (left * right) % FALCON_Q as u32
}

fn add_mod(left: u32, right: u32) -> u32 {
    let sum = left + right;
    if sum >= FALCON_Q as u32 {
        sum - FALCON_Q as u32
    } else {
        sum
    }
}

fn sub_mod(left: u32, right: u32) -> u32 {
    if left >= right {
        left - right
    } else {
        left + FALCON_Q as u32 - right
    }
}

fn canonical_residue(value: i16) -> u32 {
    // Adding 3q makes every i16 nonnegative before the reduction.
    (i32::from(value) + 3 * FALCON_Q as i32).cast_unsigned() % FALCON_Q as u32
}

#[inline]
fn center_residue(residue: u32) -> i16 {
    // Inverse scaling and the integer reference both produce canonical
    // residues. Since q < i16::MAX, narrowing before centering is exact.
    debug_assert!(residue < FALCON_Q as u32);
    let residue = residue as i16;
    if residue > 6_144 {
        residue - FALCON_Q as i16
    } else {
        residue
    }
}

#[cfg(test)]
mod tests {
    use super::super::params::{Falcon1024, Falcon512};
    use super::*;

    fn schoolbook<P: FalconParams>(left: &P::Array<i16>, right: &P::Array<i16>) -> P::Array<i16> {
        let mut product = vec![0_i64; P::DEGREE];
        for (left_index, left_value) in left.as_ref().iter().enumerate() {
            for (right_index, right_value) in right.as_ref().iter().enumerate() {
                let degree = left_index + right_index;
                let term = i64::from(*left_value) * i64::from(*right_value);
                if degree < P::DEGREE {
                    product[degree] += term;
                } else {
                    product[degree - P::DEGREE] -= term;
                }
            }
        }
        let mut output = filled::<P, i16>(0);
        for (slot, value) in output.as_mut().iter_mut().zip(product) {
            *slot = center_residue(value.rem_euclid(FALCON_Q) as u32);
        }
        output
    }

    fn ntt_matches_basis_wraparound_for<P: FalconParams>() {
        let mut left = filled::<P, i16>(0);
        left.as_mut()[P::DEGREE - 1] = 1;
        let mut right = filled::<P, i16>(0);
        right.as_mut()[1] = 2;
        let product = negacyclic_product::<P>(&left, &right);
        assert_eq!(product.as_ref()[0], -2);
        assert_eq!(product, schoolbook::<P>(&left, &right));
    }

    #[test]
    fn ntt_matches_basis_wraparound() {
        ntt_matches_basis_wraparound_for::<Falcon512>();
        ntt_matches_basis_wraparound_for::<Falcon1024>();
    }

    fn ntt_matches_deterministic_edge_vector_for<P: FalconParams>() {
        let mut left = filled::<P, i16>(0);
        for (index, slot) in left.as_mut().iter_mut().enumerate() {
            *slot = match index % 5 {
                0 => -6_144,
                1 => 6_144,
                2 => -1,
                3 => 1,
                _ => 0,
            };
        }
        let mut right = filled::<P, i16>(0);
        for (index, slot) in right.as_mut().iter_mut().enumerate() {
            *slot = match index % 7 {
                0 => -2_047,
                1 => 2_047,
                2 => -128,
                3 => 127,
                _ => 0,
            };
        }
        assert_eq!(
            negacyclic_product::<P>(&left, &right),
            schoolbook::<P>(&left, &right)
        );
    }

    #[test]
    fn ntt_matches_deterministic_edge_vector() {
        ntt_matches_deterministic_edge_vector_for::<Falcon512>();
        ntt_matches_deterministic_edge_vector_for::<Falcon1024>();
    }

    fn ntt_matches_random_full_range_vectors_for<P: FalconParams>() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..4 {
            let mut left = filled::<P, i16>(0);
            for slot in left.as_mut() {
                *slot = (next() % 12_289) as i16 - 6_144;
            }
            let mut right = filled::<P, i16>(0);
            for slot in right.as_mut() {
                *slot = (next() % 4_095) as i16 - 2_047;
            }
            assert_eq!(
                negacyclic_product::<P>(&left, &right),
                schoolbook::<P>(&left, &right)
            );
        }
    }

    #[test]
    fn ntt_matches_random_full_range_vectors() {
        ntt_matches_random_full_range_vectors_for::<Falcon512>();
        ntt_matches_random_full_range_vectors_for::<Falcon1024>();
    }

    #[test]
    fn mul_mod_matches_u64_reference_at_boundaries_and_random_reduced_pairs() {
        let modulus = FALCON_Q as u32;
        let boundary_values = [0, 1, 2, modulus / 2, modulus - 2, modulus - 1];
        for left in boundary_values {
            for right in boundary_values {
                let expected = (u64::from(left) * u64::from(right) % u64::from(modulus)) as u32;
                assert_eq!(mul_mod(left, right), expected, "{left} * {right} mod q");
            }
        }

        let mut state = 0x9e37_79b9_u32;
        for _ in 0..4_096 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let left = state % modulus;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let right = state % modulus;
            let expected = (u64::from(left) * u64::from(right) % u64::from(modulus)) as u32;
            assert_eq!(mul_mod(left, right), expected, "{left} * {right} mod q");
        }
    }

    #[test]
    fn canonical_residues_have_unique_centered_representatives() {
        for residue in 0..FALCON_Q as u32 {
            let centered = center_residue(residue);
            assert!((-6_144..=6_144).contains(&centered));
            assert_eq!(
                (i64::from(centered) - i64::from(residue)).rem_euclid(FALCON_Q),
                0
            );
        }
    }

    #[test]
    fn plans_use_primitive_roots_of_the_right_order() {
        for (degree, plan) in [(512, plan::<Falcon512>()), (1024, plan::<Falcon1024>())] {
            let psi = plan.forward_roots[degree / 2];
            assert_eq!(pow_mod(psi, degree as u32), FALCON_Q as u32 - 1);
            assert_eq!(pow_mod(psi, 2 * degree as u32), 1);
            assert_eq!(plan.degree, degree);
        }
    }
}
