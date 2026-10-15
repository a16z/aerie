//! `AArch64` Keccak-f1600 times-three backend: two states in `FEAT_SHA3`
//! vector lanes and a third in general-purpose registers.
//!
//! The two-state SHA3 kernel is limited by vector issue and leaves the
//! integer pipes idle, so a scalar state runs beside the vector pair in the
//! same round loop. Two interleaved vector pairs are slower per state than
//! one, so the vector side stays at two states.

#![allow(clippy::inline_always)]

use hybrid_array::{typenum::U3, Array};
use keccak::{Backend, Fn1600, ParFn1600, ParState1600, State1600};
use std::arch::aarch64::{
    uint64x2_t, vbcaxq_u64, vcombine_u64, vcreate_u64, vdupq_n_u64, veor3q_u64, veorq_u64,
    vgetq_lane_u64, vrax1q_u64, vxarq_u64,
};

const ROUNDS: usize = 24;
const RC: [u64; ROUNDS] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];
/// Rotation of lane `x + 5 y`.
const RHO: [u32; 25] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];
/// `SOURCE[j]` is the lane that rho-pi moves to position `j`.
const SOURCE: [usize; 25] = {
    let mut source = [0; 25];
    let mut lane = 0;
    while lane < 25 {
        let (x, y) = (lane % 5, lane / 5);
        source[y + 5 * ((2 * x + 3 * y) % 5)] = lane;
        lane += 1;
    }
    source
};

pub(crate) struct HybridBackend;

impl Backend for HybridBackend {
    type ParSize1600 = U3;

    fn get_p1600<const ROUND_COUNT: usize>() -> Fn1600 {
        let mut permutation = None;
        keccak::Keccak::new().with_p1600::<ROUND_COUNT>(|function| permutation = Some(function));
        permutation.expect("Keccak backend must supply p1600")
    }

    fn get_par_p1600<const ROUND_COUNT: usize>() -> ParFn1600<Self> {
        if ROUND_COUNT == ROUNDS {
            permute3
        } else {
            |states| {
                let permutation = Self::get_p1600::<ROUND_COUNT>();
                states.iter_mut().for_each(permutation);
            }
        }
    }
}

pub(crate) fn supported() -> bool {
    std::arch::is_aarch64_feature_detected!("sha3")
}

fn permute3(states: &mut ParState1600<HybridBackend>) {
    debug_assert!(supported());
    let Array([first, second, third]) = states;
    // SAFETY: `with_backend` selects this backend only after `supported()`.
    unsafe { permute3_sha3(first, second, third) }
}

#[target_feature(enable = "sha3")]
unsafe fn permute3_sha3(first: &mut State1600, second: &mut State1600, third: &mut State1600) {
    let mut vector: [uint64x2_t; 25] =
        std::array::from_fn(|i| vcombine_u64(vcreate_u64(first[i]), vcreate_u64(second[i])));
    let mut scalar = *third;
    for &rc in &RC {
        let parity: [uint64x2_t; 5] = std::array::from_fn(|x| {
            veor3q_u64(
                vector[x],
                vector[x + 5],
                veor3q_u64(vector[x + 10], vector[x + 15], vector[x + 20]),
            )
        });
        let theta: [uint64x2_t; 5] =
            std::array::from_fn(|x| vrax1q_u64(parity[(x + 4) % 5], parity[(x + 1) % 5]));
        // Rho and pi: `rotated[j]` is lane `SOURCE[j]` after theta and its rotation.
        let rotated: [uint64x2_t; 25] = [
            vxarq_u64::<0>(vector[0], theta[0]),
            vxarq_u64::<20>(vector[6], theta[1]),
            vxarq_u64::<21>(vector[12], theta[2]),
            vxarq_u64::<43>(vector[18], theta[3]),
            vxarq_u64::<50>(vector[24], theta[4]),
            vxarq_u64::<36>(vector[3], theta[3]),
            vxarq_u64::<44>(vector[9], theta[4]),
            vxarq_u64::<61>(vector[10], theta[0]),
            vxarq_u64::<19>(vector[16], theta[1]),
            vxarq_u64::<3>(vector[22], theta[2]),
            vxarq_u64::<63>(vector[1], theta[1]),
            vxarq_u64::<58>(vector[7], theta[2]),
            vxarq_u64::<39>(vector[13], theta[3]),
            vxarq_u64::<56>(vector[19], theta[4]),
            vxarq_u64::<46>(vector[20], theta[0]),
            vxarq_u64::<37>(vector[4], theta[4]),
            vxarq_u64::<28>(vector[5], theta[0]),
            vxarq_u64::<54>(vector[11], theta[1]),
            vxarq_u64::<49>(vector[17], theta[2]),
            vxarq_u64::<8>(vector[23], theta[3]),
            vxarq_u64::<2>(vector[2], theta[2]),
            vxarq_u64::<9>(vector[8], theta[3]),
            vxarq_u64::<25>(vector[14], theta[4]),
            vxarq_u64::<23>(vector[15], theta[0]),
            vxarq_u64::<62>(vector[21], theta[1]),
        ];
        // The scalar round has no data dependence on the vector lanes, so the
        // core issues both in the same window.
        scalar_round(&mut scalar, rc);
        vector = std::array::from_fn(|i| {
            let (x, row) = (i % 5, i - i % 5);
            vbcaxq_u64(
                rotated[i],
                rotated[row + (x + 2) % 5],
                rotated[row + (x + 1) % 5],
            )
        });
        vector[0] = veorq_u64(vector[0], vdupq_n_u64(rc));
    }
    for i in 0..25 {
        first[i] = vgetq_lane_u64::<0>(vector[i]);
        second[i] = vgetq_lane_u64::<1>(vector[i]);
    }
    *third = scalar;
}

/// One scalar round, chi applied row by row so few rotated lanes are live.
#[inline(always)]
fn scalar_round(a: &mut State1600, rc: u64) {
    let c: [u64; 5] = std::array::from_fn(|x| a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20]);
    let d: [u64; 5] = std::array::from_fn(|x| c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1));
    let lanes = *a;
    for row in (0..25).step_by(5) {
        let b: [u64; 5] = std::array::from_fn(|x| {
            let lane = SOURCE[row + x];
            (lanes[lane] ^ d[lane % 5]).rotate_left(RHO[lane])
        });
        for x in 0..5 {
            a[row + x] = b[x] ^ (!b[(x + 1) % 5] & b[(x + 2) % 5]);
        }
    }
    a[0] ^= rc;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permute3_matches_the_reference_permutation() {
        if !supported() {
            return;
        }
        let reference = HybridBackend::get_p1600::<ROUNDS>();
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..64 {
            let states: [State1600; 3] = std::array::from_fn(|_| {
                std::array::from_fn(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed
                })
            });
            let mut expected = states;
            expected.iter_mut().for_each(reference);
            let mut actual = Array(states);
            permute3(&mut actual);
            assert_eq!(actual.0, expected);
        }
    }
}
