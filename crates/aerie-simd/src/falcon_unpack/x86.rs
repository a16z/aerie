use std::arch::x86_64::{
    __m256i, __m512i, _mm256_and_si256, _mm256_castsi128_si256, _mm256_cmpgt_epi16,
    _mm256_inserti128_si256, _mm256_packus_epi32, _mm256_set1_epi16, _mm256_set1_epi32,
    _mm256_setr_epi32, _mm256_shuffle_epi8, _mm256_srlv_epi32, _mm256_storeu_si256,
    _mm256_sub_epi16, _mm512_and_si512, _mm512_cmpgt_epi16_mask, _mm512_loadu_si512,
    _mm512_mask_storeu_epi16, _mm512_mask_sub_epi16, _mm512_maskz_loadu_epi8, _mm512_packus_epi32,
    _mm512_permutexvar_epi8, _mm512_set1_epi16, _mm512_set1_epi32, _mm512_srlv_epi32,
    _mm512_storeu_si512, _mm_loadu_si128,
};

// Each 128-bit lane turns a 16-byte window (two seven-byte blocks plus two
// bytes of over-read) into eight residues. Each 32-bit lane gathers four
// bytes in big-endian order so that its residue sits at bits [s + 13, s] with
// s = 18, 12, 14, 16 for the four slots of a block. The residues are masked
// to 14 bits before `packus_epi32`, which would otherwise saturate; the pack
// then interleaves the two blocks back into residue order within each lane.
const FIRST_BLOCK: [u8; 16] = [3, 2, 1, 0, 4, 3, 2, 1, 6, 5, 4, 3, 8, 7, 6, 5];
const SECOND_BLOCK: [u8; 16] = [10, 9, 8, 7, 11, 10, 9, 8, 13, 12, 11, 10, 15, 14, 13, 12];
const RIGHT_SHIFTS: [i32; 4] = [18, 12, 14, 16];
const LANE_STEP_BYTES: usize = 14;

const AVX2_RESIDUES: usize = 16;
// Two windows at offsets 0 and 14 span 30 bytes; the padded tail rounds up.
const AVX2_READ_BYTES: usize = 30;
const AVX2_TAIL_BYTES: usize = 32;

/// # Safety
///
/// The CPU must support AVX2. `out.len()` must be a multiple of four,
/// and `payload` must contain exactly `14 * out.len()` bits. The slices must
/// provide `payload.len()` readable bytes and `out.len()` writable `i16` slots.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn unpack_avx2(payload: &[u8], out: &mut [i16]) {
    let mut step = 0;
    while 2 * LANE_STEP_BYTES * step + AVX2_READ_BYTES <= payload.len() {
        // SAFETY: the loop condition keeps both 16-byte windows in `payload`;
        // 28 * (step + 1) <= payload.len() means sixteen residues remain in
        // `out`.
        unsafe {
            unpack_avx2_step(
                payload.as_ptr().add(2 * LANE_STEP_BYTES * step),
                out.as_mut_ptr().add(AVX2_RESIDUES * step),
            );
        }
        step += 1;
    }
    // Fewer than 30 bytes, hence at most sixteen residues, remain.
    super::unpack_padded_tail::<AVX2_TAIL_BYTES, AVX2_RESIDUES>(
        &payload[2 * LANE_STEP_BYTES * step..],
        &mut out[AVX2_RESIDUES * step..],
        |bytes, residues| {
            // SAFETY: the buffers hold 32 bytes and sixteen slots; AVX2 is
            // enabled in this function.
            unsafe { unpack_avx2_step(bytes.as_ptr(), residues.as_mut_ptr()) }
        },
    );
}

// SAFETY: the caller must enable AVX2, make 30 bytes readable at `bytes`, and
// make sixteen i16 slots writable at `out`.
#[target_feature(enable = "avx2")]
unsafe fn unpack_avx2_step(bytes: *const u8, out: *mut i16) {
    // SAFETY: the caller guarantees bytes [0, 30); each unaligned load reads
    // 16 of them.
    let window = unsafe {
        _mm256_inserti128_si256::<1>(
            _mm256_castsi128_si256(_mm_loadu_si128(bytes.cast())),
            _mm_loadu_si128(bytes.add(LANE_STEP_BYTES).cast()),
        )
    };
    let [first_block, second_block] = [FIRST_BLOCK, SECOND_BLOCK].map(|table| {
        // SAFETY: each table is 16 bytes; loadu has no alignment requirement.
        let lane = unsafe { _mm_loadu_si128(table.as_ptr().cast()) };
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lane), lane)
    });
    let shifts = _mm256_setr_epi32(
        RIGHT_SHIFTS[0],
        RIGHT_SHIFTS[1],
        RIGHT_SHIFTS[2],
        RIGHT_SHIFTS[3],
        RIGHT_SHIFTS[0],
        RIGHT_SHIFTS[1],
        RIGHT_SHIFTS[2],
        RIGHT_SHIFTS[3],
    );
    let mask = _mm256_set1_epi32(0x3fff);
    let extract = |table: __m256i| {
        _mm256_and_si256(
            _mm256_srlv_epi32(_mm256_shuffle_epi8(window, table), shifts),
            mask,
        )
    };
    let residues = _mm256_packus_epi32(extract(first_block), extract(second_block));
    let large = _mm256_cmpgt_epi16(residues, _mm256_set1_epi16(super::HALF.cast_signed()));
    let centered = _mm256_sub_epi16(
        residues,
        _mm256_and_si256(large, _mm256_set1_epi16(super::Q.cast_signed())),
    );
    // SAFETY: the caller guarantees sixteen writable slots.
    unsafe { _mm256_storeu_si256(out.cast(), centered) };
}

const AVX512_RESIDUES: usize = 32;
const AVX512_STEP_BYTES: usize = 4 * LANE_STEP_BYTES;
const AVX512_LOAD_BYTES: usize = 64;

/// # Safety
///
/// The CPU must support AVX-512F, BW and VBMI. `out.len()` must be a multiple of four,
/// and `payload` must contain exactly `14 * out.len()` bits. The slices must
/// provide `payload.len()` readable bytes and `out.len()` writable `i16` slots.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(super) unsafe fn unpack_avx512(payload: &[u8], out: &mut [i16]) {
    let mut step = 0;
    // Full-width loads and stores: masked stores are slow on some cores.
    while AVX512_STEP_BYTES * step + AVX512_LOAD_BYTES <= payload.len() {
        // SAFETY: the loop condition keeps the 64-byte load in `payload`;
        // 56 * (step + 1) <= payload.len() means 32 residues remain in `out`.
        unsafe {
            let window = _mm512_loadu_si512(payload.as_ptr().add(AVX512_STEP_BYTES * step).cast());
            _mm512_storeu_si512(
                out.as_mut_ptr().add(AVX512_RESIDUES * step).cast(),
                centered_avx512(window),
            );
        }
        step += 1;
    }
    // Fewer than 64 bytes remain: at most one full 56-byte step and a partial
    // one, both through masked loads and stores.
    let payload = &payload[AVX512_STEP_BYTES * step..];
    let out = &mut out[AVX512_RESIDUES * step..];
    for (step, chunk) in payload.chunks(AVX512_STEP_BYTES).enumerate() {
        // A chunk of b bytes (a multiple of 7) holds 4 * b / 7 residues.
        let byte_mask = u64::MAX >> (64 - chunk.len());
        let residue_count = chunk.len() / 7 * 4;
        let residue_mask = u32::MAX >> (32 - residue_count);
        // SAFETY: the mask enables only the chunk's bytes; masked-off bytes
        // are neither read nor faulted on. The chunk's residues occupy
        // out[32 * step..][..residue_count], and masked-off lanes are not
        // written.
        unsafe {
            let window = _mm512_maskz_loadu_epi8(byte_mask, chunk.as_ptr().cast());
            _mm512_mask_storeu_epi16(
                out.as_mut_ptr().add(AVX512_RESIDUES * step),
                residue_mask,
                centered_avx512(window),
            );
        }
    }
}

// Lane L of the permute reads the window at byte 14 * L of a 56-byte step, so
// both tables index at most byte 14 * 3 + 15 = 57 < 64.
const fn avx512_table(block: [u8; 16]) -> [u8; 64] {
    let mut table = [0_u8; 64];
    let mut index = 0;
    while index < 64 {
        #[allow(clippy::cast_possible_truncation)] // At most 14 * 3 = 42.
        let lane_offset = (LANE_STEP_BYTES * (index / 16)) as u8;
        table[index] = lane_offset + block[index % 16];
        index += 1;
    }
    table
}

const AVX512_TABLES: [[u8; 64]; 2] = [avx512_table(FIRST_BLOCK), avx512_table(SECOND_BLOCK)];
const AVX512_SHIFTS: [i32; 16] = {
    let mut shifts = [0; 16];
    let mut index = 0;
    while index < 16 {
        shifts[index] = RIGHT_SHIFTS[index % 4];
        index += 1;
    }
    shifts
};

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
fn centered_avx512(window: __m512i) -> __m512i {
    // SAFETY: each constant table is 64 bytes; loadu has no alignment
    // requirement.
    let (first_block, second_block, shifts) = unsafe {
        (
            _mm512_loadu_si512(AVX512_TABLES[0].as_ptr().cast()),
            _mm512_loadu_si512(AVX512_TABLES[1].as_ptr().cast()),
            _mm512_loadu_si512(AVX512_SHIFTS.as_ptr().cast()),
        )
    };
    let mask = _mm512_set1_epi32(0x3fff);
    let extract = |table: __m512i| {
        _mm512_and_si512(
            _mm512_srlv_epi32(_mm512_permutexvar_epi8(table, window), shifts),
            mask,
        )
    };
    let residues = _mm512_packus_epi32(extract(first_block), extract(second_block));
    let large = _mm512_cmpgt_epi16_mask(residues, _mm512_set1_epi16(super::HALF.cast_signed()));
    _mm512_mask_sub_epi16(
        residues,
        large,
        residues,
        _mm512_set1_epi16(super::Q.cast_signed()),
    )
}
