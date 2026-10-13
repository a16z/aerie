use std::arch::aarch64::{
    vaddq_s64, vaddvq_s64, vdupq_n_s64, vget_high_s16, vget_low_s16, vget_low_s32, vld1q_s16,
    vld1q_s32, vmlal_high_s32, vmlal_s32, vmovl_s16,
};

pub(super) fn dot_neon(digits: [&[i32]; 4], factors: &[i16]) -> [i128; 2] {
    debug_assert!(std::arch::is_aarch64_feature_detected!("neon"));
    debug_assert!(factors.len() <= super::MAX_TERMS);
    debug_assert!(digits.iter().all(|stream| stream.len() == factors.len()));
    // SAFETY: feature detection guards the target-feature implementation and
    // the fixed-size safe plan guarantees equal input lengths.
    unsafe { dot_neon_inner(digits, factors) }
}

// SAFETY: the caller must enable NEON and supply streams at least as long as
// `factors`, with at most MAX_TERMS terms, as enforced by SmallSignedDotPlan.
#[target_feature(enable = "neon")]
unsafe fn dot_neon_inner(digits: [&[i32]; 4], factors: &[i16]) -> [i128; 2] {
    // accumulators[stream][k] holds signed i64 lanes: k = 0, 1 take the low
    // four factors of each block of eight, k = 2, 3 the high four.
    let mut accumulators = [[vdupq_n_s64(0); 4]; 4];
    let chunks = factors.len() / 8 * 8;
    for offset in (0..chunks).step_by(8) {
        // SAFETY: offset + 8 <= chunks <= factors.len(); vld1q reads eight
        // i16 values with no vector-alignment requirement. NEON is enabled here.
        let packed = unsafe { vld1q_s16(factors.as_ptr().add(offset)) };
        let factors_low = vmovl_s16(vget_low_s16(packed));
        let factors_high = vmovl_s16(vget_high_s16(packed));
        for (accumulator, stream) in accumulators.iter_mut().zip(digits) {
            // SAFETY: stream.len() >= factors.len() >= offset + 8; neither
            // load requires vector alignment.
            let (digits_low, digits_high) = unsafe {
                (
                    vld1q_s32(stream.as_ptr().add(offset)),
                    vld1q_s32(stream.as_ptr().add(offset + 4)),
                )
            };
            // SMLAL widens i32 x i32 to i64 and accumulates. Each product
            // is at most 2^31 * 2^15 = 2^46 in magnitude and a lane receives
            // at most MAX_TERMS / 8 = 2^13 of them, so no lane passes 2^59.
            accumulator[0] = vmlal_s32(
                accumulator[0],
                vget_low_s32(digits_low),
                vget_low_s32(factors_low),
            );
            accumulator[1] = vmlal_high_s32(accumulator[1], digits_low, factors_low);
            accumulator[2] = vmlal_s32(
                accumulator[2],
                vget_low_s32(digits_high),
                vget_low_s32(factors_high),
            );
            accumulator[3] = vmlal_high_s32(accumulator[3], digits_high, factors_high);
        }
    }
    // Every sum of lanes of a stream adds at most MAX_TERMS = 2^16 products
    // of magnitude at most 2^46, so each lies within 2^62 and the
    // in-register horizontal reduction stays exact in i64.
    let sums = accumulators
        .map(|[a, b, c, d]| i128::from(vaddvq_s64(vaddq_s64(vaddq_s64(a, b), vaddq_s64(c, d)))));
    super::finish_digit_sums(sums, digits, factors, chunks)
}
