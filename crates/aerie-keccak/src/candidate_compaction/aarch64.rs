//! `AArch64` NEON implementation.

use std::arch::aarch64::{
    uint16x4_t, uint16x8_t, uint8x16_t, uint8x16x2_t, vadd_u16, vaddq_u16, vaddv_u16, vaddvq_u16,
    vand_u16, vandq_u16, vcge_u16, vcgeq_u16, vclt_u16, vcltq_u16, vcombine_u8, vdup_n_s16,
    vdup_n_u16, vdupq_n_s16, vdupq_n_u16, vdupq_n_u32, vget_low_u16, vld1_u16, vld1_u8, vld1q_s16,
    vld1q_u16, vld1q_u8, vmla_u16, vmlal_high_u16, vmlal_u16, vmlaq_u16, vmls_u16, vmlsq_u16,
    vqtbl1q_u8, vqtbl2q_u8, vreinterpret_u16_u8, vreinterpret_u8_u16, vreinterpretq_s16_u8,
    vreinterpretq_u16_u32, vreinterpretq_u16_u8, vreinterpretq_u8_s16, vreinterpretq_u8_u16,
    vrev16_u8, vrev16q_u8, vshl_u16, vshlq_u16, vshrn_n_u32, vst1q_s16, vuzp2q_u16,
};

use super::{compact_centered_u16_scalar, COMPACTION_SHUFFLES, HASH_TO_POINT_BLOCK_CANDIDATES};

// Lane i contributes its mask bit 2^i in the low byte and one count unit in
// the high byte, so one horizontal sum yields both the shuffle row and the
// number of accepted lanes: at most 255 + 8 * 256 fits in u16.
const MASK_AND_COUNT_WEIGHTS: [u16; 8] = [0x101, 0x102, 0x104, 0x108, 0x110, 0x120, 0x140, 0x180];

/// The centering thresholds `j * MODULUS - (MODULUS - 1) / 2` for `j = 1..=5`.
/// An accepted candidate `v < 5 MODULUS` reaches exactly `k` of them, where
/// `v - k MODULUS` is its centered residue: the intervals
/// `[j MODULUS - h, j MODULUS + h]` with `h = (MODULUS - 1) / 2` tile `0..5 MODULUS`.
const fn centering_thresholds<const MODULUS: u16>() -> [u16; 5] {
    let half = (MODULUS - 1) / 2;
    [
        MODULUS - half,
        2 * MODULUS - half,
        3 * MODULUS - half,
        4 * MODULUS - half,
        5 * MODULUS - half,
    ]
}

/// Constants `(multiplier, offset, shift)` with
/// `(v * multiplier + offset) >> shift == (v + (MODULUS - 1) / 2) / MODULUS`
/// for every `v < 5 MODULUS`, with `multiplier` in u16, the sum in u32 and
/// `shift >= 16`, or `None` when no such constants exist for small `shift`.
///
/// The quotient is constant on each centering interval, so the feasible
/// offsets form an interval bounded by the endpoints of those six intervals.
// The casts at the return are in range: `multiplier` is checked against
// `u16::MAX`, `low` is nonnegative and checked against `u32::MAX`, and
// `shift - 16` is below 16.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
const fn multiply_centering<const MODULUS: u16>() -> Option<(u16, u32, i16)> {
    let modulus = MODULUS as i64;
    if modulus == 0 {
        // Rejected by the public wrapper; the kernel is never reached.
        return None;
    }
    let half = (modulus - 1) / 2;
    let mut shift = 16;
    while shift < 32 {
        let base = (1_i64 << shift) / modulus;
        let mut multiplier = base;
        while multiplier <= base + 1 {
            if multiplier >= 1 && multiplier <= u16::MAX as i64 {
                let (mut low, mut high) = (0_i64, i64::MAX);
                let mut quotient = 0;
                while quotient <= 5 {
                    let first = if quotient * modulus > half {
                        quotient * modulus - half
                    } else {
                        0
                    };
                    let last = if quotient * modulus + half < 5 * modulus - 1 {
                        quotient * modulus + half
                    } else {
                        5 * modulus - 1
                    };
                    if first <= last {
                        let floor = (quotient << shift) - first * multiplier;
                        let ceiling = ((quotient + 1) << shift) - 1 - last * multiplier;
                        if floor > low {
                            low = floor;
                        }
                        if ceiling < high {
                            high = ceiling;
                        }
                    }
                    quotient += 1;
                }
                if low <= high && (5 * modulus - 1) * multiplier + low <= u32::MAX as i64 {
                    return Some((multiplier as u16, low as u32, shift as i16 - 16));
                }
            }
            multiplier += 1;
        }
        shift += 1;
    }
    None
}

/// Centered residues of eight big-endian candidates, and the packed accepted
/// mask plus count.
#[inline]
#[target_feature(enable = "neon")]
fn center_x8<const MODULUS: u16>(encoded: uint16x8_t) -> (uint16x8_t, usize) {
    let candidates = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(encoded)));
    let accepted = vcltq_u16(candidates, vdupq_n_u16(MODULUS * 5));
    // SAFETY: the table holds eight u16 values; vld1q needs no alignment.
    let weights = unsafe { vld1q_u16(MASK_AND_COUNT_WEIGHTS.as_ptr()) };
    let packed = usize::from(vaddvq_u16(vandq_u16(accepted, weights)));
    if let Some((multiplier, offset, shift)) = const { multiply_centering::<MODULUS>() } {
        // Rejected lanes may wrap in u32; the shuffle discards them.
        let low = vmlal_u16(
            vdupq_n_u32(offset),
            vget_low_u16(candidates),
            vdup_n_u16(multiplier),
        );
        let high = vmlal_high_u16(vdupq_n_u32(offset), candidates, vdupq_n_u16(multiplier));
        let upper = vuzp2q_u16(vreinterpretq_u16_u32(low), vreinterpretq_u16_u32(high));
        let quotient = vshlq_u16(upper, vdupq_n_s16(-shift));
        return (
            vmlsq_u16(candidates, quotient, vdupq_n_u16(MODULUS)),
            packed,
        );
    }
    let [t1, t2, t3, t4, t5] = centering_thresholds::<MODULUS>();
    // Each comparison is all ones, that is -1, when reached; their sum is -k.
    let reached = vaddq_u16(
        vaddq_u16(
            vcgeq_u16(candidates, vdupq_n_u16(t1)),
            vcgeq_u16(candidates, vdupq_n_u16(t2)),
        ),
        vaddq_u16(
            vaddq_u16(
                vcgeq_u16(candidates, vdupq_n_u16(t3)),
                vcgeq_u16(candidates, vdupq_n_u16(t4)),
            ),
            vcgeq_u16(candidates, vdupq_n_u16(t5)),
        ),
    );
    (vmlaq_u16(candidates, reached, vdupq_n_u16(MODULUS)), packed)
}

/// The four-lane form of [`center_x8`].
#[inline]
#[target_feature(enable = "neon")]
fn center_x4<const MODULUS: u16>(encoded: uint16x4_t) -> (uint16x4_t, usize) {
    let candidates = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(encoded)));
    let accepted = vclt_u16(candidates, vdup_n_u16(MODULUS * 5));
    // SAFETY: the table holds at least four u16 values; vld1 needs no alignment.
    let weights = unsafe { vld1_u16(MASK_AND_COUNT_WEIGHTS.as_ptr()) };
    let packed = usize::from(vaddv_u16(vand_u16(accepted, weights)));
    if let Some((multiplier, offset, shift)) = const { multiply_centering::<MODULUS>() } {
        let product = vmlal_u16(vdupq_n_u32(offset), candidates, vdup_n_u16(multiplier));
        let quotient = vshl_u16(vshrn_n_u32::<16>(product), vdup_n_s16(-shift));
        return (vmls_u16(candidates, quotient, vdup_n_u16(MODULUS)), packed);
    }
    let [t1, t2, t3, t4, t5] = centering_thresholds::<MODULUS>();
    let reached = vadd_u16(
        vadd_u16(
            vcge_u16(candidates, vdup_n_u16(t1)),
            vcge_u16(candidates, vdup_n_u16(t2)),
        ),
        vadd_u16(
            vadd_u16(
                vcge_u16(candidates, vdup_n_u16(t3)),
                vcge_u16(candidates, vdup_n_u16(t4)),
            ),
            vcge_u16(candidates, vdup_n_u16(t5)),
        ),
    );
    (vmla_u16(candidates, reached, vdup_n_u16(MODULUS)), packed)
}

/// The accepted centered candidates among eight read from `input`, packed
/// into the low lanes, and how many were accepted.
///
/// # Safety
///
/// NEON must be available and `input` must provide 16 readable bytes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn compacted_x8<const MODULUS: u16>(input: *const u8) -> (uint8x16_t, usize) {
    // SAFETY: the caller provides 16 readable bytes; vld1q needs no alignment.
    let encoded = vreinterpretq_u16_u8(unsafe { vld1q_u8(input) });
    let (centered, packed) = center_x8::<MODULUS>(encoded);
    // SAFETY: the low byte of `packed` is a mask below 256, and each table
    // row holds 16 bytes; vld1q needs no alignment.
    let shuffle = unsafe { vld1q_u8(COMPACTION_SHUFFLES[packed & 0xff].as_ptr()) };
    (
        vqtbl1q_u8(vreinterpretq_u8_u16(centered), shuffle),
        packed >> 8,
    )
}

/// The four-candidate form of [`compacted_x8`]; lanes four to seven are zero.
///
/// # Safety
///
/// NEON must be available and `input` must provide 8 readable bytes.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn compacted_x4<const MODULUS: u16>(input: *const u8) -> (uint8x16_t, usize) {
    // SAFETY: the caller provides 8 readable bytes; vld1 needs no alignment.
    let encoded = vreinterpret_u16_u8(unsafe { vld1_u8(input) });
    let (centered, packed) = center_x4::<MODULUS>(encoded);
    // A four-lane mask selects a row that indexes only lanes 0..4.
    // SAFETY: the mask is below 16 and each table row holds 16 bytes.
    let shuffle = unsafe { vld1q_u8(COMPACTION_SHUFFLES[packed & 0xf].as_ptr()) };
    let bytes = vreinterpret_u8_u16(centered);
    (vqtbl1q_u8(vcombine_u8(bytes, bytes), shuffle), packed >> 8)
}

/// Compact eight candidates read from `input` to the front of `output`,
/// returning how many were accepted.
///
/// # Safety
///
/// NEON must be available, `input` must provide 16 readable bytes, and
/// `output` must have room for eight values.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn compact_x8<const MODULUS: u16>(input: *const u8, output: *mut i16) -> usize {
    // SAFETY: forwarded from the caller.
    let (compacted, accepted) = unsafe { compacted_x8::<MODULUS>(input) };
    // SAFETY: the caller leaves room for eight values; vst1q needs no alignment.
    unsafe { vst1q_s16(output, vreinterpretq_s16_u8(compacted)) };
    accepted
}

/// Byte indices into `(previous, compacted)` that keep the first `keep` lanes
/// of `previous` and follow them with the first `8 - keep` compacted lanes.
#[allow(clippy::cast_possible_truncation)]
const fn tail_blends() -> [[u8; 16]; 8] {
    let mut table = [[0_u8; 16]; 8];
    let mut keep = 0;
    while keep < 8 {
        let mut byte = 0;
        while byte < 16 {
            table[keep][byte] = if byte < 2 * keep {
                byte as u8
            } else {
                (16 + byte - 2 * keep) as u8
            };
            byte += 1;
        }
        keep += 1;
    }
    table
}

static TAIL_BLENDS: [[u8; 16]; 8] = tail_blends();

/// Append `accepted` compacted lanes at `filled` in a row of `len >= 8`
/// slots, returning the new filled length, at most `len`.
///
/// Away from the row end this stores all eight lanes. Within eight slots of
/// the end it rewrites the last eight slots, keeping the filled ones, so no
/// store passes the end. Slots past the returned length may change.
///
/// # Safety
///
/// NEON must be available, `row` must point to `len >= 8` writable values,
/// and `filled < len`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn append_lanes(
    row: *mut i16,
    len: usize,
    filled: usize,
    compacted: uint8x16_t,
    accepted: usize,
) -> usize {
    if filled + 8 <= len {
        // SAFETY: slots filled..filled + 8 are inside the row.
        unsafe { vst1q_s16(row.add(filled), vreinterpretq_s16_u8(compacted)) };
        return filled + accepted;
    }
    let keep = filled + 8 - len;
    // SAFETY: len >= 8, so slots len - 8..len are inside the row; keep is
    // below 8 because filled < len; vld1q and vst1q need no alignment.
    unsafe {
        let last = row.add(len - 8);
        let previous = vreinterpretq_u8_s16(vld1q_s16(last));
        let blend = vld1q_u8(TAIL_BLENDS[keep].as_ptr());
        let merged = vqtbl2q_u8(uint8x16x2_t(previous, compacted), blend);
        vst1q_s16(last, vreinterpretq_s16_u8(merged));
    }
    (filled + accepted).min(len)
}

// SAFETY: the caller must provide NEON; unless BOUNDED, output.len() must
// be at least input.len()/2. The public wrapper validates the modulus.
#[target_feature(enable = "neon")]
unsafe fn compact_centered_u16_neon_inner<const MODULUS: u16, const BOUNDED: bool>(
    input: &[u8],
    output: &mut [i16],
) -> usize {
    let vector_bytes = input.len() / 16 * 16;
    let mut count = 0;
    for offset in (0..vector_bytes).step_by(16) {
        if BOUNDED {
            let mut lanes = [0_i16; 8];
            // SAFETY: offset + 16 <= input.len(), and lanes holds eight values.
            let accepted =
                unsafe { compact_x8::<MODULUS>(input.as_ptr().add(offset), lanes.as_mut_ptr()) };
            let take = accepted.min(output.len() - count);
            output[count..count + take].copy_from_slice(&lanes[..take]);
            count += take;
            if count == output.len() {
                return count;
            }
        } else {
            // SAFETY: offset + 16 <= input.len(), and count <= offset / 2
            // with output.len() >= input.len() / 2 leaves room for eight.
            count += unsafe {
                compact_x8::<MODULUS>(input.as_ptr().add(offset), output.as_mut_ptr().add(count))
            };
        }
    }
    count + compact_centered_u16_scalar::<MODULUS>(&input[vector_bytes..], &mut output[count..])
}

/// # Safety
///
/// The caller must detect NEON, validate paired input bytes and an odd
/// nonzero MODULUS with 5 * MODULUS fitting in u16, and provide nonempty output.
pub(super) unsafe fn compact_centered_u16_neon<const MODULUS: u16>(
    input: &[u8],
    output: &mut [i16],
) -> usize {
    // SAFETY: the caller provides NEON and validates input and MODULUS. The
    // first two branches provide capacity for every input candidate; the last
    // uses BOUNDED to check remaining capacity before each store. Loads are unaligned.
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

/// Append the accepted candidates of one SHAKE256 rate block, held as
/// little-endian state words, to `row[filled..]`; returns the new filled
/// length, at most `row.len()`.
///
/// # Safety
///
/// The caller must detect NEON, validate an odd nonzero MODULUS with
/// 5 * MODULUS fitting in u16, and provide `filled < row.len()` with
/// `row.len() >= 8`.
#[target_feature(enable = "neon")]
pub(super) unsafe fn compact_centered_rate_neon<const MODULUS: u16>(
    rate: &[u64; super::RATE_WORDS],
    row: &mut [i16],
    mut filled: usize,
) -> usize {
    debug_assert!(row.len() >= 8 && filled < row.len());
    // On little-endian AArch64 the words' memory is the rate's byte stream.
    let input = rate.as_ptr().cast::<u8>();
    let len = row.len();
    let row = row.as_mut_ptr();
    if filled + super::HASH_TO_POINT_BLOCK_CANDIDATES + 4 <= len {
        // Each step adds at most eight values, so after `k` steps at most
        // 8 k slots are filled past `filled`; every eight-lane store,
        // including the last step's padded four, ends by `filled + 72`.
        for vector in 0..8 {
            // SAFETY: vector * 16 + 16 <= 128 of the 136 bytes in `rate`,
            // and filled + 8 <= len by the bound above.
            unsafe {
                let (compacted, accepted) = compacted_x8::<MODULUS>(input.add(16 * vector));
                vst1q_s16(row.add(filled), vreinterpretq_s16_u8(compacted));
                filled += accepted;
            }
        }
        // SAFETY: bytes 128..136 are the last rate word; filled + 8 <= len.
        unsafe {
            let (compacted, accepted) = compacted_x4::<MODULUS>(input.add(128));
            vst1q_s16(row.add(filled), vreinterpretq_s16_u8(compacted));
            return filled + accepted;
        }
    }
    for vector in 0..8 {
        // SAFETY: vector * 16 + 16 <= 128 of the 136 bytes in `rate`, and
        // the caller's bounds hold for `filled < len`.
        unsafe {
            let (compacted, accepted) = compacted_x8::<MODULUS>(input.add(16 * vector));
            filled = append_lanes(row, len, filled, compacted, accepted);
        }
        if filled == len {
            return len;
        }
    }
    // SAFETY: bytes 128..136 are the last rate word; filled < len as above.
    unsafe {
        let (compacted, accepted) = compacted_x4::<MODULUS>(input.add(128));
        append_lanes(row, len, filled, compacted, accepted)
    }
}
