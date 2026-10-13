//! Negacyclic multiplication in `Z_q[X] / (X^n + 1)` for both Falcon degrees.
//!
//! One transform plan per degree is built on first use. The Falcon modulus
//! `q - 1 = 2^12 * 3` supplies primitive `2n`-th roots of unity for both
//! `n = 512` and `n = 1024`.

use std::sync::OnceLock;

use crate::params::{filled, FalconParams, FALCON_Q};

static PLAN_512: OnceLock<NegacyclicPlan> = OnceLock::new();
static PLAN_1024: OnceLock<NegacyclicPlan> = OnceLock::new();

struct NegacyclicPlan {
    degree: usize,
    stages: usize,
    bit_reversal: Vec<u16>,
    forward_twiddles: Vec<Vec<u32>>,
    inverse_twiddles: Vec<Vec<u32>>,
    twist: Vec<u32>,
    inverse_scale_twist: Vec<u32>,
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
        let omega = mul_mod(psi, psi);
        let omega_inverse = pow_mod(omega, modulus - 2);
        let degree_inverse = pow_mod(degree_u32, modulus - 2);
        let stages = degree.trailing_zeros() as usize;
        Self {
            degree,
            stages,
            bit_reversal: bit_reversal_table(degree, stages),
            forward_twiddles: stage_twiddles(omega, degree, stages),
            inverse_twiddles: stage_twiddles(omega_inverse, degree, stages),
            twist: powers(psi, degree),
            inverse_scale_twist: powers(psi_inverse, degree)
                .into_iter()
                .map(|value| mul_mod(value, degree_inverse))
                .collect(),
        }
    }

    fn forward(&self, values: &[i16]) -> Vec<u32> {
        debug_assert_eq!(values.len(), self.degree);
        let mut transformed = (0..self.degree)
            .map(|index| {
                let source = usize::from(self.bit_reversal[index]);
                mul_mod(canonical_residue(values[source]), self.twist[source])
            })
            .collect::<Vec<_>>();
        cyclic_ntt_ordered(&mut transformed, &self.forward_twiddles);
        transformed
    }

    fn inverse(&self, values: &mut [u32]) {
        bit_reverse(values, &self.bit_reversal);
        cyclic_ntt_ordered(values, &self.inverse_twiddles);
        for (value, scale_twist) in values.iter_mut().zip(&self.inverse_scale_twist) {
            *value = mul_mod(*value, *scale_twist);
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
    let left_ntt = plan.forward(left.as_ref());
    let right_ntt = plan.forward(right.as_ref());
    let mut product = left_ntt
        .iter()
        .zip(&right_ntt)
        .map(|(left, right)| mul_mod(*left, *right))
        .collect::<Vec<_>>();
    plan.inverse(&mut product);
    let mut output = filled::<P, i16>(0);
    for (slot, residue) in output.as_mut().iter_mut().zip(product) {
        *slot = center_residue(residue);
    }
    output
}

/// Iterative cyclic NTT from bit-reversed input to natural-order output.
fn cyclic_ntt_ordered(values: &mut [u32], twiddles: &[Vec<u32>]) {
    let mut length = 2;
    for stage_twiddles in twiddles {
        let mut twiddle_index = 0;
        for block in values.chunks_exact_mut(length) {
            let (left, right) = block.split_at_mut(length / 2);
            for (even, odd) in left.iter_mut().zip(right) {
                let product = mul_mod(*odd, stage_twiddles[twiddle_index]);
                let original = *even;
                *even = add_mod(original, product);
                *odd = sub_mod(original, product);
                twiddle_index += 1;
            }
        }
        length *= 2;
    }
}

fn bit_reverse(values: &mut [u32], table: &[u16]) {
    for (index, &reverse) in table.iter().enumerate().skip(1) {
        let reverse = usize::from(reverse);
        if index < reverse {
            values.swap(index, reverse);
        }
    }
}

fn bit_reversal_table(degree: usize, stages: usize) -> Vec<u16> {
    (0..degree)
        .map(|index| {
            u16::try_from(index.reverse_bits() >> (usize::BITS as usize - stages))
                .expect("Falcon bit-reversal index fits in u16")
        })
        .collect()
}

fn stage_twiddles(root: u32, degree: usize, stages: usize) -> Vec<Vec<u32>> {
    let mut length = 2;
    (0..stages)
        .map(|_| {
            let stage_root = pow_mod(root, (degree / length) as u32);
            let mut twiddles = Vec::with_capacity(degree / 2);
            for _ in 0..degree / length {
                let mut twiddle = 1;
                for _ in 0..length / 2 {
                    twiddles.push(twiddle);
                    twiddle = mul_mod(twiddle, stage_root);
                }
            }
            length *= 2;
            twiddles
        })
        .collect()
}

fn powers(base: u32, count: usize) -> Vec<u32> {
    let mut current = 1;
    (0..count)
        .map(|_| {
            let output = current;
            current = mul_mod(current, base);
            output
        })
        .collect()
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
    i32::from(value).rem_euclid(FALCON_Q as i32) as u32
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
    use super::*;
    use crate::params::{Falcon1024, Falcon512};

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
            let psi = plan.twist[1];
            assert_eq!(pow_mod(psi, degree as u32), FALCON_Q as u32 - 1);
            assert_eq!(pow_mod(psi, 2 * degree as u32), 1);
            assert_eq!(plan.degree, degree);
        }
    }
}
