use std::arch::x86_64::{
    _mm256_add_epi64, _mm256_cvtepi16_epi32, _mm256_loadu_si256, _mm256_mul_epi32,
    _mm256_setzero_si256, _mm256_srli_epi64, _mm256_storeu_si256, _mm512_add_epi64,
    _mm512_cvtepi16_epi32, _mm512_loadu_si512, _mm512_mul_epi32, _mm512_reduce_add_epi64,
    _mm512_setzero_si512, _mm512_srli_epi64, _mm_loadu_si128,
};

// `mul_epi32` (VPMULDQ) multiplies the sign-extended low 32-bit words of each
// 64-bit lane into a signed 64-bit product. Even terms use the vectors as
// loaded; odd terms shift each lane right by 32 so the odd word becomes the
// low word. The high word of each lane is ignored by the instruction, so the
// logical shift needs no sign fixup.

pub(super) fn dot_avx2(digits: [&[i32]; 4], factors: &[i16]) -> [i128; 2] {
    debug_assert!(std::is_x86_feature_detected!("avx2"));
    debug_assert!(factors.len() <= super::MAX_TERMS);
    debug_assert!(digits.iter().all(|stream| stream.len() == factors.len()));
    // SAFETY: feature detection guards the target-feature implementation and
    // the fixed-size safe plan guarantees equal input lengths.
    unsafe { dot_avx2_inner(digits, factors) }
}

// SAFETY: the caller must enable AVX2 and supply streams at least as long as
// `factors`, with at most MAX_TERMS terms, as enforced by SmallSignedDotPlan.
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2_inner(digits: [&[i32]; 4], factors: &[i16]) -> [i128; 2] {
    let mut sums = [[_mm256_setzero_si256(); 2]; 4];
    let chunks = factors.len() / 8 * 8;
    for offset in (0..chunks).step_by(8) {
        // SAFETY: offset + 8 <= chunks <= factors.len(); this unaligned load
        // reads exactly 8 i16 values. The function enables AVX2.
        let packed = unsafe { _mm_loadu_si128(factors.as_ptr().add(offset).cast()) };
        let factors_even = _mm256_cvtepi16_epi32(packed);
        let factors_odd = _mm256_srli_epi64::<32>(factors_even);
        for (sum, stream) in sums.iter_mut().zip(digits) {
            // SAFETY: stream.len() >= factors.len() >= offset + 8; loadu
            // needs no vector alignment.
            let values = unsafe { _mm256_loadu_si256(stream.as_ptr().add(offset).cast()) };
            let even = _mm256_mul_epi32(values, factors_even);
            let odd = _mm256_mul_epi32(_mm256_srli_epi64::<32>(values), factors_odd);
            // Each product is at most 2^31 * 2^15 = 2^46 in magnitude and a
            // lane receives at most MAX_TERMS / 8 = 2^13 of them, so no i64
            // lane passes 2^59.
            sum[0] = _mm256_add_epi64(sum[0], even);
            sum[1] = _mm256_add_epi64(sum[1], odd);
        }
    }
    // Every sum of lanes of a stream adds at most MAX_TERMS = 2^16 products
    // of magnitude at most 2^46, so each lies within 2^62 and the
    // in-register horizontal reduction stays exact in i64.
    let totals = sums.map(|[even, odd]| {
        let mut words = [0_i64; 4];
        // SAFETY: words holds exactly one 32-byte vector; storeu has no
        // alignment requirement, and this function enables AVX2.
        unsafe { _mm256_storeu_si256(words.as_mut_ptr().cast(), _mm256_add_epi64(even, odd)) };
        i128::from(words.into_iter().sum::<i64>())
    });
    super::finish_digit_sums(totals, digits, factors, chunks)
}

pub(super) fn dot_avx512(digits: [&[i32]; 4], factors: &[i16]) -> [i128; 2] {
    debug_assert!(std::is_x86_feature_detected!("avx512f"));
    debug_assert!(factors.len() <= super::MAX_TERMS);
    debug_assert!(digits.iter().all(|stream| stream.len() == factors.len()));
    // SAFETY: feature detection guards the target-feature implementation and
    // the fixed-size safe plan guarantees equal input lengths.
    unsafe { dot_avx512_inner(digits, factors) }
}

// SAFETY: the caller must enable AVX-512F and supply streams at least as long
// as `factors`, with at most MAX_TERMS terms, as enforced by SmallSignedDotPlan.
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512_inner(digits: [&[i32]; 4], factors: &[i16]) -> [i128; 2] {
    let mut sums = [[_mm512_setzero_si512(); 2]; 4];
    let chunks = factors.len() / 16 * 16;
    for offset in (0..chunks).step_by(16) {
        // SAFETY: offset + 16 <= chunks <= factors.len(); this unaligned load
        // reads exactly 16 i16 values. AVX-512F implies AVX2.
        let packed = unsafe { _mm256_loadu_si256(factors.as_ptr().add(offset).cast()) };
        let factors_even = _mm512_cvtepi16_epi32(packed);
        let factors_odd = _mm512_srli_epi64::<32>(factors_even);
        for (sum, stream) in sums.iter_mut().zip(digits) {
            // SAFETY: stream.len() >= factors.len() >= offset + 16; loadu
            // needs no vector alignment.
            let values = unsafe { _mm512_loadu_si512(stream.as_ptr().add(offset).cast()) };
            let even = _mm512_mul_epi32(values, factors_even);
            let odd = _mm512_mul_epi32(_mm512_srli_epi64::<32>(values), factors_odd);
            // Each product is at most 2^31 * 2^15 = 2^46 in magnitude and a
            // lane receives at most MAX_TERMS / 16 = 2^12 of them, so no i64
            // lane passes 2^58.
            sum[0] = _mm512_add_epi64(sum[0], even);
            sum[1] = _mm512_add_epi64(sum[1], odd);
        }
    }
    // Every sum of lanes of a stream adds at most MAX_TERMS = 2^16 products
    // of magnitude at most 2^46, so each lies within 2^62 and the
    // in-register horizontal reduction stays exact in i64.
    let totals =
        sums.map(|[even, odd]| i128::from(_mm512_reduce_add_epi64(_mm512_add_epi64(even, odd))));
    super::finish_digit_sums(totals, digits, factors, chunks)
}
