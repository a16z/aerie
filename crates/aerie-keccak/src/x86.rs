//! x86-64 Keccak-f1600 times-four and times-eight backends.
//!
//! The round function is adapted from `RustCrypto`'s MIT/Apache-2.0 `keccak`
//! implementation. Its portable-SIMD lane operations are replaced by stable
//! `std::arch` intrinsics. States are transposed into vectors before the 24
//! rounds and transposed back afterwards.

#![allow(clippy::inline_always)]

use hybrid_array::{typenum::U4, typenum::U8, Array};
use keccak::{Backend, Fn1600, ParFn1600, ParState1600, State1600};
use std::{
    arch::x86_64::{
        __m256i, __m512i, _mm256_andnot_si256, _mm256_or_si256, _mm256_set1_epi64x,
        _mm256_setzero_si256, _mm256_sllv_epi64, _mm256_srlv_epi64, _mm256_xor_si256,
        _mm512_andnot_si512, _mm512_or_si512, _mm512_set1_epi64, _mm512_setzero_si512,
        _mm512_sllv_epi64, _mm512_srlv_epi64, _mm512_xor_si512,
    },
    array,
};

const ROUNDS: usize = 24;
const RHO: [u32; ROUNDS] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];
const PI: [usize; ROUNDS] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];
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

pub(crate) struct Avx2Backend;

impl Backend for Avx2Backend {
    type ParSize1600 = U4;

    fn get_p1600<const ROUND_COUNT: usize>() -> Fn1600 {
        platform_p1600::<ROUND_COUNT>()
    }

    fn get_par_p1600<const ROUND_COUNT: usize>() -> ParFn1600<Self> {
        if ROUND_COUNT == ROUNDS {
            permute4
        } else {
            |states| {
                let permutation = Self::get_p1600::<ROUND_COUNT>();
                states.iter_mut().for_each(permutation);
            }
        }
    }
}

pub(crate) struct Avx512Backend;

impl Backend for Avx512Backend {
    type ParSize1600 = U8;

    fn get_p1600<const ROUND_COUNT: usize>() -> Fn1600 {
        platform_p1600::<ROUND_COUNT>()
    }

    fn get_par_p1600<const ROUND_COUNT: usize>() -> ParFn1600<Self> {
        if ROUND_COUNT == ROUNDS {
            permute8
        } else {
            |states| {
                let permutation = Self::get_p1600::<ROUND_COUNT>();
                states.iter_mut().for_each(permutation);
            }
        }
    }
}

fn platform_p1600<const ROUND_COUNT: usize>() -> Fn1600 {
    let mut permutation = None;
    keccak::Keccak::new().with_p1600::<ROUND_COUNT>(|function| permutation = Some(function));
    permutation.expect("Keccak backend must supply p1600")
}

trait VectorLane: Copy {
    // SAFETY: the caller must enable the ISA required by the lane implementation.
    unsafe fn zero() -> Self;
    // SAFETY: the caller must enable the ISA required by the lane implementation.
    unsafe fn splat(value: u64) -> Self;
    // SAFETY: the caller must enable the ISA required by the lane implementation.
    unsafe fn xor(self, other: Self) -> Self;
    // SAFETY: the caller must enable the ISA required by the lane implementation.
    unsafe fn and_not(self, other: Self) -> Self;
    // SAFETY: the caller must enable the ISA required by the lane implementation.
    // For rotation, amount must be in 0..=64.
    unsafe fn rotate_left(self, amount: u32) -> Self;
}

macro_rules! unroll5 {
    ($index:ident, $body:block) => {{
        const $index: usize = 0;
        $body
    }
    {
        const $index: usize = 1;
        $body
    }
    {
        const $index: usize = 2;
        $body
    }
    {
        const $index: usize = 3;
        $body
    }
    {
        const $index: usize = 4;
        $body
    }};
}

macro_rules! unroll24 {
    ($index:ident, $body:block) => {{
        const $index: usize = 0;
        $body
    }
    {
        const $index: usize = 1;
        $body
    }
    {
        const $index: usize = 2;
        $body
    }
    {
        const $index: usize = 3;
        $body
    }
    {
        const $index: usize = 4;
        $body
    }
    {
        const $index: usize = 5;
        $body
    }
    {
        const $index: usize = 6;
        $body
    }
    {
        const $index: usize = 7;
        $body
    }
    {
        const $index: usize = 8;
        $body
    }
    {
        const $index: usize = 9;
        $body
    }
    {
        const $index: usize = 10;
        $body
    }
    {
        const $index: usize = 11;
        $body
    }
    {
        const $index: usize = 12;
        $body
    }
    {
        const $index: usize = 13;
        $body
    }
    {
        const $index: usize = 14;
        $body
    }
    {
        const $index: usize = 15;
        $body
    }
    {
        const $index: usize = 16;
        $body
    }
    {
        const $index: usize = 17;
        $body
    }
    {
        const $index: usize = 18;
        $body
    }
    {
        const $index: usize = 19;
        $body
    }
    {
        const $index: usize = 20;
        $body
    }
    {
        const $index: usize = 21;
        $body
    }
    {
        const $index: usize = 22;
        $body
    }
    {
        const $index: usize = 23;
        $body
    }};
}

// SAFETY: the caller must enable the ISA required by L (AVX2 or AVX-512F).
#[allow(non_upper_case_globals, unused_assignments)]
unsafe fn keccak_f<L: VectorLane>(state: &mut [L; 25]) {
    for &round_constant in &RC {
        // SAFETY: the caller guarantees the ISA required by L.
        let mut scratch = [unsafe { L::zero() }; 5];

        unroll5!(x, {
            unroll5!(y, {
                // SAFETY: the caller guarantees L's ISA; xor uses registers only.
                scratch[x] = unsafe { scratch[x].xor(state[5 * y + x]) };
            });
        });
        unroll5!(x, {
            // SAFETY: L's ISA is guaranteed by the caller; rotation by 1 is valid.
            let delta = unsafe { scratch[(x + 4) % 5].xor(scratch[(x + 1) % 5].rotate_left(1)) };
            unroll5!(y, {
                // SAFETY: the caller guarantees L's ISA; xor uses registers only.
                state[5 * y + x] = unsafe { state[5 * y + x].xor(delta) };
            });
        });

        let mut last = state[1];
        unroll24!(index, {
            let next = state[PI[index]];
            // SAFETY: L's ISA is guaranteed; all RHO entries are in 1..64.
            state[PI[index]] = unsafe { last.rotate_left(RHO[index]) };
            last = next;
        });

        unroll5!(y, {
            let offset = 5 * y;
            scratch.copy_from_slice(&state[offset..offset + 5]);
            unroll5!(x, {
                // SAFETY: L's ISA is guaranteed; these operations use registers only.
                state[offset + x] =
                    unsafe { scratch[x].xor(scratch[(x + 1) % 5].and_not(scratch[(x + 2) % 5])) };
            });
        });

        // SAFETY: the caller guarantees L's ISA for splat and xor.
        state[0] = unsafe { state[0].xor(L::splat(round_constant)) };
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
struct Avx2Lane(__m256i);

impl VectorLane for Avx2Lane {
    // SAFETY: the VectorLane caller must enable AVX2.
    #[inline(always)]
    unsafe fn zero() -> Self {
        // SAFETY: the caller guarantees AVX2; this intrinsic uses registers only.
        Self(unsafe { _mm256_setzero_si256() })
    }

    // SAFETY: the VectorLane caller must enable AVX2.
    #[inline(always)]
    unsafe fn splat(value: u64) -> Self {
        let value = i64::from_ne_bytes(value.to_ne_bytes());
        // SAFETY: the caller guarantees AVX2; this intrinsic uses registers only.
        Self(unsafe { _mm256_set1_epi64x(value) })
    }

    // SAFETY: the VectorLane caller must enable AVX2.
    #[inline(always)]
    unsafe fn xor(self, other: Self) -> Self {
        // SAFETY: the caller guarantees AVX2; this intrinsic uses registers only.
        Self(unsafe { _mm256_xor_si256(self.0, other.0) })
    }

    // SAFETY: the VectorLane caller must enable AVX2.
    #[inline(always)]
    unsafe fn and_not(self, other: Self) -> Self {
        // SAFETY: the caller guarantees AVX2; this intrinsic uses registers only.
        Self(unsafe { _mm256_andnot_si256(self.0, other.0) })
    }

    // SAFETY: the VectorLane caller must enable AVX2.
    #[inline(always)]
    unsafe fn rotate_left(self, amount: u32) -> Self {
        // SAFETY: the caller guarantees AVX2; all five intrinsics use
        // registers only. keccak_f supplies rotation amounts in 1..64.
        unsafe {
            let left_count = _mm256_set1_epi64x(i64::from(amount));
            let right_count = _mm256_set1_epi64x(i64::from(64 - amount));
            let left = _mm256_sllv_epi64(self.0, left_count);
            let right = _mm256_srlv_epi64(self.0, right_count);
            Self(_mm256_or_si256(left, right))
        }
    }
}

// SAFETY: the caller must enable AVX2; state dimensions are fixed by the type.
#[target_feature(enable = "avx2")]
unsafe fn permute4_inner(states: &mut [State1600; 4]) {
    let mut vector_state: [Avx2Lane; 25] = array::from_fn(|word| {
        let lanes: [u64; 4] = array::from_fn(|lane| states[lane][word]);
        // SAFETY: `[u64; 4]` and `__m256i` are equal-size plain bit containers.
        Avx2Lane(unsafe { std::mem::transmute::<[u64; 4], __m256i>(lanes) })
    });
    // SAFETY: this function enables AVX2, the ISA required by its lane type.
    unsafe { keccak_f(&mut vector_state) };
    for (word, vector) in vector_state.iter().enumerate() {
        // SAFETY: `__m256i` and `[u64; 4]` are equal-size plain bit containers.
        let lanes = unsafe { std::mem::transmute::<__m256i, [u64; 4]>(vector.0) };
        for lane in 0..4 {
            states[lane][word] = lanes[lane];
        }
    }
}

fn permute4(Array(states): &mut ParState1600<Avx2Backend>) {
    debug_assert!(std::is_x86_feature_detected!("avx2"));
    // SAFETY: `Avx2Backend` is selected only after runtime AVX2 detection.
    unsafe { permute4_inner(states) };
}

#[derive(Clone, Copy)]
#[repr(transparent)]
struct Avx512Lane(__m512i);

impl VectorLane for Avx512Lane {
    // SAFETY: the VectorLane caller must enable AVX-512F.
    #[inline(always)]
    unsafe fn zero() -> Self {
        // SAFETY: the caller guarantees AVX-512F; this intrinsic uses registers only.
        Self(unsafe { _mm512_setzero_si512() })
    }

    // SAFETY: the VectorLane caller must enable AVX-512F.
    #[inline(always)]
    unsafe fn splat(value: u64) -> Self {
        let value = i64::from_ne_bytes(value.to_ne_bytes());
        // SAFETY: the caller guarantees AVX-512F; this intrinsic uses registers only.
        Self(unsafe { _mm512_set1_epi64(value) })
    }

    // SAFETY: the VectorLane caller must enable AVX-512F.
    #[inline(always)]
    unsafe fn xor(self, other: Self) -> Self {
        // SAFETY: the caller guarantees AVX-512F; this intrinsic uses registers only.
        Self(unsafe { _mm512_xor_si512(self.0, other.0) })
    }

    // SAFETY: the VectorLane caller must enable AVX-512F.
    #[inline(always)]
    unsafe fn and_not(self, other: Self) -> Self {
        // SAFETY: the caller guarantees AVX-512F; this intrinsic uses registers only.
        Self(unsafe { _mm512_andnot_si512(self.0, other.0) })
    }

    // SAFETY: the VectorLane caller must enable AVX-512F.
    #[inline(always)]
    unsafe fn rotate_left(self, amount: u32) -> Self {
        // SAFETY: the caller guarantees AVX-512F; all five intrinsics use
        // registers only. keccak_f supplies rotation amounts in 1..64.
        unsafe {
            let left_count = _mm512_set1_epi64(i64::from(amount));
            let right_count = _mm512_set1_epi64(i64::from(64 - amount));
            let left = _mm512_sllv_epi64(self.0, left_count);
            let right = _mm512_srlv_epi64(self.0, right_count);
            Self(_mm512_or_si512(left, right))
        }
    }
}

// SAFETY: the caller must enable AVX512F; state dimensions are fixed by the type.
#[target_feature(enable = "avx512f")]
unsafe fn permute8_inner(states: &mut [State1600; 8]) {
    let mut vector_state: [Avx512Lane; 25] = array::from_fn(|word| {
        let lanes: [u64; 8] = array::from_fn(|lane| states[lane][word]);
        // SAFETY: `[u64; 8]` and `__m512i` are equal-size plain bit containers.
        Avx512Lane(unsafe { std::mem::transmute::<[u64; 8], __m512i>(lanes) })
    });
    // SAFETY: this function enables AVX512F, the ISA required by its lane type.
    unsafe { keccak_f(&mut vector_state) };
    for (word, vector) in vector_state.iter().enumerate() {
        // SAFETY: `__m512i` and `[u64; 8]` are equal-size plain bit containers.
        let lanes = unsafe { std::mem::transmute::<__m512i, [u64; 8]>(vector.0) };
        for lane in 0..8 {
            states[lane][word] = lanes[lane];
        }
    }
}

fn permute8(Array(states): &mut ParState1600<Avx512Backend>) {
    debug_assert!(std::is_x86_feature_detected!("avx512f"));
    // SAFETY: `Avx512Backend` is selected only after runtime AVX-512F detection.
    unsafe { permute8_inner(states) };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_states<const N: usize>() -> [State1600; N] {
        array::from_fn(|lane| {
            array::from_fn(|word| {
                let lane = u64::try_from(lane).expect("lane index fits u64");
                let word = u64::try_from(word).expect("word index fits u64");
                lane.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(word.wrapping_mul(0x0101_0101_0101_0101))
            })
        })
    }

    fn scalar_expected<const N: usize>(mut states: [State1600; N]) -> [State1600; N] {
        let permutation = platform_p1600::<ROUNDS>();
        states.iter_mut().for_each(permutation);
        states
    }

    #[test]
    fn avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut actual = deterministic_states::<4>();
        let expected = scalar_expected(actual);
        // SAFETY: the feature check above proves AVX2 availability.
        unsafe { permute4_inner(&mut actual) };
        assert_eq!(actual, expected);
    }

    #[test]
    fn avx512_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        let mut actual = deterministic_states::<8>();
        let expected = scalar_expected(actual);
        // SAFETY: the feature check above proves AVX-512F availability.
        unsafe { permute8_inner(&mut actual) };
        assert_eq!(actual, expected);
    }
}
