//! Stable ordered compaction for rejection-sampled centered `u16` values.

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use crate::SHAKE256_RATE_BYTES;
#[cfg(target_arch = "x86_64")]
use crate::{selected_backend, SelectedBackend};

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod aarch64;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod x86;

// Keeping the rare bounded final block on the stack lets each architecture's
// SIMD hot loop remain identical to its full-capacity path.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const HASH_TO_POINT_BLOCK_CANDIDATES: usize = SHAKE256_RATE_BYTES / 2;

/// Compact big-endian candidates below `5 * MODULUS` and center them modulo
/// `MODULUS`, preserving their input order.
///
/// At most `output.len()` accepted candidates are written. `MODULUS` must be
/// odd and `5 * MODULUS` must fit in `u16`.
pub fn compact_centered_be_u16_5q<const MODULUS: u16>(input: &[u8], output: &mut [i16]) -> usize {
    assert!(
        input.len().is_multiple_of(2),
        "candidate bytes must be paired"
    );
    assert!(MODULUS % 2 == 1, "centered modulus must be odd");
    assert!(
        MODULUS <= u16::MAX / 5,
        "five residue intervals must fit in u16"
    );
    if output.is_empty() {
        return 0;
    }

    #[cfg(target_arch = "x86_64")]
    match selected_backend() {
        SelectedBackend::Avx512 => {
            return x86::compact_centered_u16_avx512::<MODULUS>(input, output);
        }
        SelectedBackend::Avx2 => {
            return x86::compact_centered_u16_avx2::<MODULUS>(input, output);
        }
        SelectedBackend::Platform => {}
    }
    #[cfg(target_arch = "aarch64")]
    return aarch64::compact_centered_u16_neon::<MODULUS>(input, output);

    #[allow(unreachable_code)]
    compact_centered_u16_scalar::<MODULUS>(input, output)
}

#[inline(never)]
fn compact_centered_u16_scalar<const MODULUS: u16>(input: &[u8], output: &mut [i16]) -> usize {
    let bound = MODULUS * 5;
    let half = (MODULUS - 1) / 2;
    let mut count = 0;
    for word in input.chunks_exact(2) {
        let value = u16::from_be_bytes([word[0], word[1]]);
        if value < bound {
            let quotient = (u32::from(value) + u32::from(half)) / u32::from(MODULUS);
            let quotient = i32::try_from(quotient).expect("accepted quotient is at most five");
            let centered = i32::from(value) - quotient * i32::from(MODULUS);
            output[count] = i16::try_from(centered).expect("centered residue fits in i16");
            count += 1;
            if count == output.len() {
                break;
            }
        }
    }
    count
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn compact_centered_u16_scalar_runtime(input: &[u8], modulus: u16, output: &mut [i16]) -> usize {
    let bound = modulus * 5;
    let half = (modulus - 1) / 2;
    let mut count = 0;
    for word in input.chunks_exact(2) {
        let value = u16::from_be_bytes([word[0], word[1]]);
        if value < bound {
            let numerator = u32::from(value) + u32::from(half);
            let quotient = if modulus == 12_289 {
                // Exact reciprocal reduction for every accepted Falcon
                // numerator (value + 6_144). This is the same strength
                // reduction LLVM emits when the divisor is visible as a
                // compile-time constant.
                u32::try_from((u64::from(numerator) * 0x5_5539) >> 32)
                    .expect("Falcon quotient fits in u32")
            } else {
                numerator / u32::from(modulus)
            };
            let quotient = i32::try_from(quotient).expect("accepted quotient is at most five");
            let centered = i32::from(value) - quotient * i32::from(modulus);
            output[count] = i16::try_from(centered).expect("centered residue fits in i16");
            count += 1;
        }
    }
    count
}

#[allow(clippy::cast_possible_truncation)]
const fn compaction_shuffles() -> [[u8; 16]; 256] {
    let mut table = [[0x80_u8; 16]; 256];
    let mut mask = 0;
    while mask < table.len() {
        let mut lane = 0_usize;
        let mut output = 0_usize;
        while lane < 8 {
            if mask & (1 << lane) != 0 {
                table[mask][2 * output] = (2 * lane) as u8;
                table[mask][2 * output + 1] = (2 * lane + 1) as u8;
                output += 1;
            }
            lane += 1;
        }
        mask += 1;
    }
    table
}

static COMPACTION_SHUFFLES: [[u8; 16]; 256] = compaction_shuffles();
