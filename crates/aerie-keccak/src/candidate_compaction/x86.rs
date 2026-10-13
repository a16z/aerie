//! x86-64 AVX2 and AVX-512 implementations.

use std::arch::x86_64::{
    __m128i, _mm512_cmpgt_epu16_mask, _mm512_cmplt_epu16_mask, _mm512_loadu_si512,
    _mm512_mask_compressstoreu_epi16, _mm512_mask_sub_epi16, _mm512_set1_epi16,
    _mm512_shuffle_epi8, _mm_and_si128, _mm_cmpgt_epi16, _mm_loadu_si128, _mm_movemask_epi8,
    _mm_packs_epi16, _mm_set1_epi16, _mm_setr_epi8, _mm_setzero_si128, _mm_shuffle_epi8,
    _mm_storeu_si128, _mm_sub_epi16, _mm_xor_si128,
};

use super::{
    compact_centered_u16_scalar, compact_centered_u16_scalar_runtime, COMPACTION_SHUFFLES,
    HASH_TO_POINT_BLOCK_CANDIDATES,
};

const SWAP_U16_BYTES_512: [u8; 64] = [
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13,
    12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8,
    11, 10, 13, 12, 15, 14,
];

#[target_feature(enable = "avx2")]
fn subtract_if_gt_u16(value: __m128i, threshold: __m128i, amount: __m128i) -> __m128i {
    let sign = _mm_set1_epi16(i16::MIN);
    let mask = _mm_cmpgt_epi16(_mm_xor_si128(value, sign), _mm_xor_si128(threshold, sign));
    _mm_sub_epi16(value, _mm_and_si128(mask, amount))
}

// SAFETY: the caller must enable AVX2, provide 16 readable input bytes,
// and leave space for all eight candidates at output[*count..].
#[target_feature(enable = "avx2")]
unsafe fn compact_centered_u16x8(
    input: *const u8,
    modulus: u16,
    output: &mut [i16],
    count: &mut usize,
) {
    // SAFETY: the caller provides 16 readable bytes; loadu needs no alignment.
    // AVX2 is enabled on this function.
    let encoded = unsafe { _mm_loadu_si128(input.cast()) };
    let swap = _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    let candidates = _mm_shuffle_epi8(encoded, swap);
    let sign = _mm_set1_epi16(i16::MIN);
    let signed_candidates = _mm_xor_si128(candidates, sign);
    let bound = _mm_set1_epi16((modulus * 5).cast_signed());
    let signed_bound = _mm_xor_si128(bound, sign);
    let accepted = _mm_cmpgt_epi16(signed_bound, signed_candidates);
    let packed = _mm_packs_epi16(accepted, _mm_setzero_si128());
    let mask = usize::try_from(_mm_movemask_epi8(packed)).expect("movemask is non-negative") & 0xff;
    let accepted_count = mask.count_ones() as usize;

    let q = _mm_set1_epi16(modulus.cast_signed());
    let q_minus_one = _mm_set1_epi16((modulus - 1).cast_signed());
    let half = _mm_set1_epi16(((modulus - 1) / 2).cast_signed());
    let mut centered = candidates;
    for _ in 0..4 {
        centered = subtract_if_gt_u16(centered, q_minus_one, q);
    }
    centered = subtract_if_gt_u16(centered, half, q);

    // SAFETY: mask is restricted to 0..256, and each table row has 16 bytes.
    // loadu needs no vector alignment.
    let shuffle = unsafe { _mm_loadu_si128(COMPACTION_SHUFFLES[mask].as_ptr().cast()) };
    let compacted = _mm_shuffle_epi8(centered, shuffle);
    if *count + 8 <= output.len() {
        // SAFETY: the branch checks room for all eight i16 lanes; storeu
        // needs no vector alignment, and AVX2 is enabled here.
        unsafe { _mm_storeu_si128(output.as_mut_ptr().add(*count).cast(), compacted) };
    } else {
        let mut lanes = [0_i16; 8];
        // SAFETY: lanes has exactly eight i16 elements; storeu is unaligned.
        unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast(), compacted) };
        output[*count..*count + accepted_count].copy_from_slice(&lanes[..accepted_count]);
    }
    *count += accepted_count;
}

// SAFETY: the caller must enable AVX2 and provide output.len() >= input.len()/2.
#[target_feature(enable = "avx2")]
unsafe fn compact_centered_u16_avx2_inner(input: &[u8], modulus: u16, output: &mut [i16]) -> usize {
    let vector_bytes = input.len() / 16 * 16;
    let mut count = 0;
    for offset in (0..vector_bytes).step_by(16) {
        // SAFETY: offset + 16 <= input.len() and count <= offset/2, so the
        // full-capacity output has room for eight candidates. AVX2 is enabled.
        unsafe {
            compact_centered_u16x8(input.as_ptr().add(offset), modulus, output, &mut count);
        }
    }
    count
        + compact_centered_u16_scalar_runtime(&input[vector_bytes..], modulus, &mut output[count..])
}

pub(super) fn compact_centered_u16_avx2<const MODULUS: u16>(
    input: &[u8],
    output: &mut [i16],
) -> usize {
    debug_assert!(std::is_x86_feature_detected!("avx2"));
    // SAFETY: selection checks AVX2 before this call. Each inner call has
    // capacity for every input candidate, either in output or in scratch.
    unsafe {
        if output.len() >= input.len() / 2 {
            compact_centered_u16_avx2_inner(input, MODULUS, output)
        } else if input.len() / 2 <= HASH_TO_POINT_BLOCK_CANDIDATES {
            let mut scratch = [0_i16; HASH_TO_POINT_BLOCK_CANDIDATES];
            let accepted = compact_centered_u16_avx2_inner(input, MODULUS, &mut scratch);
            let written = accepted.min(output.len());
            output[..written].copy_from_slice(&scratch[..written]);
            written
        } else {
            compact_centered_u16_scalar::<MODULUS>(input, output)
        }
    }
}

// SAFETY: the caller must enable AVX-512F/BW/VBMI2 and AVX2 (for the tail).
// Unless BOUNDED, output.len() must be at least input.len()/2.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2")]
unsafe fn compact_centered_u16_avx512_inner<const MODULUS: u16, const BOUNDED: bool>(
    input: &[u8],
    output: &mut [i16],
) -> usize {
    let vector_bytes = input.len() / 64 * 64;
    // SAFETY: the table contains 64 readable bytes; loadu needs no alignment.
    // This function enables AVX-512F/BW/VBMI2.
    let swap = unsafe { _mm512_loadu_si512(SWAP_U16_BYTES_512.as_ptr().cast()) };
    let bound = _mm512_set1_epi16((MODULUS * 5).cast_signed());
    let q = _mm512_set1_epi16(MODULUS.cast_signed());
    let q_minus_one = _mm512_set1_epi16((MODULUS - 1).cast_signed());
    let half = _mm512_set1_epi16(((MODULUS - 1) / 2).cast_signed());
    let mut count = 0;
    for offset in (0..vector_bytes).step_by(64) {
        // SAFETY: offset + 64 <= vector_bytes <= input.len(); loadu reads
        // exactly 64 bytes without an alignment requirement.
        let encoded = unsafe { _mm512_loadu_si512(input.as_ptr().add(offset).cast()) };
        let candidates = _mm512_shuffle_epi8(encoded, swap);
        let accepted = _mm512_cmplt_epu16_mask(candidates, bound);
        let accepted_count = accepted.count_ones() as usize;
        let mut centered = candidates;
        for _ in 0..4 {
            let subtract = _mm512_cmpgt_epu16_mask(centered, q_minus_one);
            centered = _mm512_mask_sub_epi16(centered, subtract, centered, q);
        }
        let subtract = _mm512_cmpgt_epu16_mask(centered, half);
        centered = _mm512_mask_sub_epi16(centered, subtract, centered, q);

        if BOUNDED {
            let remaining = output.len() - count;
            if accepted_count <= remaining {
                // SAFETY: compressstoreu writes exactly accepted_count i16
                // elements, checked to fit here, with no vector alignment.
                unsafe {
                    _mm512_mask_compressstoreu_epi16(
                        output.as_mut_ptr().add(count),
                        accepted,
                        centered,
                    );
                }
            } else {
                let mut lanes = [0_i16; 32];
                // SAFETY: at most 32 lanes are accepted; lanes holds 32 i16s.
                // compressstoreu has no vector-alignment requirement.
                unsafe { _mm512_mask_compressstoreu_epi16(lanes.as_mut_ptr(), accepted, centered) };
                output[count..].copy_from_slice(&lanes[..remaining]);
            }
            count += accepted_count.min(remaining);
            if count == output.len() {
                return count;
            }
        } else {
            // SAFETY: count <= offset/2 and output.len() >= input.len()/2;
            // offset + 64 <= input.len() leaves room for all 32 candidates.
            // compressstoreu writes only accepted lanes and is unaligned.
            unsafe {
                _mm512_mask_compressstoreu_epi16(
                    output.as_mut_ptr().add(count),
                    accepted,
                    centered,
                );
            }
            count += accepted_count;
        }
    }
    count
        + if BOUNDED {
            compact_centered_u16_scalar::<MODULUS>(&input[vector_bytes..], &mut output[count..])
        } else {
            // SAFETY: the caller also checks AVX2. Since count <= vector_bytes/2,
            // the full-capacity output still has room for every tail candidate.
            unsafe {
                compact_centered_u16_avx2_inner(
                    &input[vector_bytes..],
                    MODULUS,
                    &mut output[count..],
                )
            }
        }
}

pub(super) fn compact_centered_u16_avx512<const MODULUS: u16>(
    input: &[u8],
    output: &mut [i16],
) -> usize {
    if std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vbmi2")
        && std::is_x86_feature_detected!("avx2")
    {
        // SAFETY: the checks above provide AVX2 and AVX-512BW/VBMI2; BW also
        // implies AVX-512F. The first two branches provide full candidate
        // capacity, and the last checks remaining capacity with BOUNDED.
        unsafe {
            if output.len() >= input.len() / 2 {
                compact_centered_u16_avx512_inner::<MODULUS, false>(input, output)
            } else if input.len() / 2 <= HASH_TO_POINT_BLOCK_CANDIDATES {
                let mut scratch = [0_i16; HASH_TO_POINT_BLOCK_CANDIDATES];
                let accepted =
                    compact_centered_u16_avx512_inner::<MODULUS, false>(input, &mut scratch);
                let written = accepted.min(output.len());
                output[..written].copy_from_slice(&scratch[..written]);
                written
            } else {
                compact_centered_u16_avx512_inner::<MODULUS, true>(input, output)
            }
        }
    } else if std::is_x86_feature_detected!("avx2") {
        compact_centered_u16_avx2::<MODULUS>(input, output)
    } else {
        compact_centered_u16_scalar::<MODULUS>(input, output)
    }
}
