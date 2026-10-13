//! `AArch64` NEON implementation.

use std::arch::aarch64::{
    vaddvq_u16, vandq_u16, vcgtq_u16, vcltq_u16, vdupq_n_u16, vld1q_u16, vld1q_u8, vmulq_u16,
    vqtbl1q_u8, vreinterpretq_s16_u8, vreinterpretq_u16_u8, vreinterpretq_u8_u16, vrev16q_u8,
    vshrq_n_u16, vst1q_s16, vsubq_u16,
};

use super::{compact_centered_u16_scalar, COMPACTION_SHUFFLES, HASH_TO_POINT_BLOCK_CANDIDATES};

const MASK_WEIGHTS: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

// SAFETY: the caller must provide NEON; unless BOUNDED, output.len() must
// be at least input.len()/2. The public wrapper validates the modulus.
#[target_feature(enable = "neon")]
unsafe fn compact_centered_u16_neon_inner<const MODULUS: u16, const BOUNDED: bool>(
    input: &[u8],
    output: &mut [i16],
) -> usize {
    let vector_bytes = input.len() / 16 * 16;
    let bound = vdupq_n_u16(MODULUS * 5);
    let q = vdupq_n_u16(MODULUS);
    let q_minus_one = vdupq_n_u16(MODULUS - 1);
    let half = vdupq_n_u16((MODULUS - 1) / 2);
    // SAFETY: MASK_WEIGHTS holds eight u16 elements; vld1q reads exactly
    // those elements without vector alignment. NEON is enabled here.
    let weights = unsafe { vld1q_u16(MASK_WEIGHTS.as_ptr()) };
    let mut count = 0;
    for offset in (0..vector_bytes).step_by(16) {
        // SAFETY: offset + 16 <= vector_bytes <= input.len(); vld1q reads
        // 16 bytes with no vector-alignment requirement.
        let encoded = unsafe { vld1q_u8(input.as_ptr().add(offset)) };
        let swapped = vrev16q_u8(encoded);
        let candidates = vreinterpretq_u16_u8(swapped);
        let accepted = vcltq_u16(candidates, bound);
        let bits = vshrq_n_u16::<15>(accepted);
        let mask = usize::from(vaddvq_u16(vmulq_u16(bits, weights)));
        let accepted_count = mask.count_ones() as usize;

        let mut centered = candidates;
        for _ in 0..4 {
            centered = vsubq_u16(centered, vandq_u16(vcgtq_u16(centered, q_minus_one), q));
        }
        centered = vsubq_u16(centered, vandq_u16(vcgtq_u16(centered, half), q));

        // SAFETY: the eight 0/1 bits with weights 1..128 give mask < 256.
        // Each table row holds 16 bytes, and vld1q needs no vector alignment.
        let shuffle = unsafe { vld1q_u8(COMPACTION_SHUFFLES[mask].as_ptr()) };
        let compacted = vqtbl1q_u8(vreinterpretq_u8_u16(centered), shuffle);
        if BOUNDED {
            let remaining = output.len() - count;
            if remaining >= 8 {
                // SAFETY: remaining guarantees room for eight i16 values;
                // vst1q needs no vector alignment, and NEON is enabled here.
                unsafe {
                    vst1q_s16(
                        output.as_mut_ptr().add(count),
                        vreinterpretq_s16_u8(compacted),
                    );
                };
            } else {
                let mut lanes = [0_i16; 8];
                // SAFETY: lanes holds all eight i16 values; vst1q needs
                // no vector alignment.
                unsafe { vst1q_s16(lanes.as_mut_ptr(), vreinterpretq_s16_u8(compacted)) };
                let take = accepted_count.min(remaining);
                output[count..count + take].copy_from_slice(&lanes[..take]);
            }
            count += accepted_count.min(remaining);
            if count == output.len() {
                return count;
            }
        } else {
            // SAFETY: count <= offset/2 and offset + 16 <= input.len();
            // full output capacity leaves room for all eight i16 values.
            // vst1q imposes no vector-alignment requirement.
            unsafe {
                vst1q_s16(
                    output.as_mut_ptr().add(count),
                    vreinterpretq_s16_u8(compacted),
                );
            };
            count += accepted_count;
        }
    }
    count + compact_centered_u16_scalar::<MODULUS>(&input[vector_bytes..], &mut output[count..])
}

pub(super) fn compact_centered_u16_neon<const MODULUS: u16>(
    input: &[u8],
    output: &mut [i16],
) -> usize {
    // SAFETY: NEON is part of the supported AArch64 target baseline. The first
    // two branches provide capacity for every input candidate; the last uses
    // BOUNDED to check remaining capacity before each store. Loads are unaligned.
    unsafe {
        if output.len() >= input.len() / 2 {
            compact_centered_u16_neon_inner::<MODULUS, false>(input, output)
        } else if input.len() / 2 <= HASH_TO_POINT_BLOCK_CANDIDATES {
            let mut scratch = [0_i16; HASH_TO_POINT_BLOCK_CANDIDATES];
            let accepted = compact_centered_u16_neon_inner::<MODULUS, false>(input, &mut scratch);
            let written = accepted.min(output.len());
            output[..written].copy_from_slice(&scratch[..written]);
            written
        } else {
            compact_centered_u16_neon_inner::<MODULUS, true>(input, output)
        }
    }
}
