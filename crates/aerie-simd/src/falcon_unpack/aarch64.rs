use std::arch::aarch64::{
    uint8x16_t, vandq_u16, vbslq_u16, vcgtq_u16, vdupq_n_u16, vld1q_s32, vld1q_u8, vqtbl1q_u8,
    vreinterpretq_s16_u16, vreinterpretq_u16_u32, vreinterpretq_u32_u8, vshlq_u32, vst1q_s16,
    vsubq_u16, vuzp1q_u16,
};

// One step turns a 16-byte window (two seven-byte blocks plus two bytes of
// over-read) into eight residues. Each 32-bit lane gathers four bytes in
// big-endian order so that its residue sits at bits [s + 13, s] with
// s = 18, 12, 14, 16 for the four slots of a block.
const WINDOW_BYTES: usize = 16;
const STEP_BYTES: usize = 14;
const STEP_RESIDUES: usize = 8;
const FIRST_BLOCK: [u8; 16] = [3, 2, 1, 0, 4, 3, 2, 1, 6, 5, 4, 3, 8, 7, 6, 5];
const SECOND_BLOCK: [u8; 16] = [10, 9, 8, 7, 11, 10, 9, 8, 13, 12, 11, 10, 15, 14, 13, 12];
const RIGHT_SHIFTS: [i32; 4] = [-18, -12, -14, -16];

/// # Safety
///
/// The CPU must support NEON. `out.len()` must be a multiple of four,
/// and `payload` must contain exactly `14 * out.len()` bits. The slices must
/// provide `payload.len()` readable bytes and `out.len()` writable `i16` slots.
#[target_feature(enable = "neon")]
pub(super) unsafe fn unpack_neon(payload: &[u8], out: &mut [i16]) {
    let mut step = 0;
    while STEP_BYTES * step + WINDOW_BYTES <= payload.len() {
        // SAFETY: the loop condition keeps the 16-byte window in `payload`;
        // 14 * (step + 1) <= payload.len() means eight residues remain in
        // `out`.
        unsafe {
            unpack_step(
                payload.as_ptr().add(STEP_BYTES * step),
                out.as_mut_ptr().add(STEP_RESIDUES * step),
            );
        }
        step += 1;
    }
    // Fewer than 16 bytes, hence at most eight residues, remain.
    super::unpack_padded_tail::<WINDOW_BYTES, STEP_RESIDUES>(
        &payload[STEP_BYTES * step..],
        &mut out[STEP_RESIDUES * step..],
        |bytes, residues| {
            // SAFETY: both buffers have exactly one window or step; NEON is
            // enabled in this function.
            unsafe { unpack_step(bytes.as_ptr(), residues.as_mut_ptr()) }
        },
    );
}

// SAFETY: the caller must enable NEON, make 16 bytes readable at `bytes`, and
// make eight i16 slots writable at `out`.
#[target_feature(enable = "neon")]
unsafe fn unpack_step(bytes: *const u8, out: *mut i16) {
    // SAFETY: the caller guarantees the 16-byte window; vld1q has no
    // alignment requirement.
    let window: uint8x16_t = unsafe { vld1q_u8(bytes) };
    // SAFETY: the constant tables are 16 bytes and 4 lanes long.
    let (first_block, second_block, shifts) = unsafe {
        (
            vld1q_u8(FIRST_BLOCK.as_ptr()),
            vld1q_u8(SECOND_BLOCK.as_ptr()),
            vld1q_s32(RIGHT_SHIFTS.as_ptr()),
        )
    };
    let first = vshlq_u32(
        vreinterpretq_u32_u8(vqtbl1q_u8(window, first_block)),
        shifts,
    );
    let second = vshlq_u32(
        vreinterpretq_u32_u8(vqtbl1q_u8(window, second_block)),
        shifts,
    );
    // The low halves of the shifted lanes hold residues 0..7 in order.
    let residues = vandq_u16(
        vuzp1q_u16(vreinterpretq_u16_u32(first), vreinterpretq_u16_u32(second)),
        vdupq_n_u16(0x3fff),
    );
    let centered = vbslq_u16(
        vcgtq_u16(residues, vdupq_n_u16(super::HALF)),
        vsubq_u16(residues, vdupq_n_u16(super::Q)),
        residues,
    );
    // SAFETY: the caller guarantees eight writable slots.
    unsafe { vst1q_s16(out, vreinterpretq_s16_u16(centered)) };
}
