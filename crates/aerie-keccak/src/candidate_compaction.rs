//! Stable ordered compaction for rejection-sampled centered `u16` values.

#[cfg(any(
    all(target_arch = "aarch64", target_endian = "little"),
    target_arch = "x86_64"
))]
use crate::SHAKE256_RATE_BYTES;
#[cfg(target_arch = "x86_64")]
use crate::{selected_backend, SelectedBackend};

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[allow(unsafe_code)]
mod aarch64;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod x86;

// Keeping the rare bounded final block on the stack lets each architecture's
// SIMD hot loop remain identical to its full-capacity path.
#[cfg(any(
    all(target_arch = "aarch64", target_endian = "little"),
    target_arch = "x86_64"
))]
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
            // SAFETY: SelectedBackend::Avx2 is selected only after AVX2
            // detection. The assertions above validate the input and modulus,
            // and the early return ensures nonempty output.
            #[allow(unsafe_code)]
            return unsafe { x86::compact_centered_u16_avx2::<MODULUS>(input, output) };
        }
        SelectedBackend::Platform => {}
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: the check above provides NEON. The assertions validate the
        // input and modulus, and the early return ensures nonempty output.
        #[allow(unsafe_code)]
        return unsafe { aarch64::compact_centered_u16_neon::<MODULUS>(input, output) };
    }

    compact_centered_u16_scalar::<MODULUS>(input, output)
}

/// Number of little-endian state words in one SHAKE256 rate block.
pub const RATE_WORDS: usize = 17;

/// [`compact_centered_be_u16_5q`] over one SHAKE256 rate block held as the
/// sponge's little-endian state words, appending to a partly filled row.
///
/// The candidates are the block's bytes in stream order, read as big-endian
/// pairs. Accepted candidates are written from `row[filled]` until the row
/// is full, and the new filled length is returned. `row[..filled]` is
/// unchanged; slots past the returned length may be overwritten.
///
/// # Panics
///
/// Panics if `filled > row.len()` or the modulus is invalid.
pub fn compact_centered_rate_5q<const MODULUS: u16>(
    rate: &[u64; RATE_WORDS],
    row: &mut [i16],
    filled: usize,
) -> usize {
    assert!(MODULUS % 2 == 1, "centered modulus must be odd");
    assert!(
        MODULUS <= u16::MAX / 5,
        "five residue intervals must fit in u16"
    );
    assert!(filled <= row.len(), "filled prefix exceeds the row");
    if filled == row.len() {
        return filled;
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    if row.len() >= 8 && std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: the check above provides NEON and a row of at least eight
        // slots; the assertions validate the modulus and filled < row.len().
        #[allow(unsafe_code)]
        return unsafe { aarch64::compact_centered_rate_neon::<MODULUS>(rate, row, filled) };
    }
    let mut bytes = [0_u8; 8 * RATE_WORDS];
    for (chunk, word) in bytes.chunks_exact_mut(8).zip(rate) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
    filled + compact_centered_be_u16_5q::<MODULUS>(&bytes, &mut row[filled..])
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

#[cfg(any(
    all(target_arch = "aarch64", target_endian = "little"),
    target_arch = "x86_64"
))]
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

#[cfg(any(
    all(target_arch = "aarch64", target_endian = "little"),
    target_arch = "x86_64"
))]
static COMPACTION_SHUFFLES: [[u8; 16]; 256] = compaction_shuffles();

#[cfg(test)]
mod tests {
    use super::{
        compact_centered_be_u16_5q, compact_centered_rate_5q, compact_centered_u16_scalar,
        RATE_WORDS,
    };

    fn reference<const MODULUS: u16>(input: &[u8], capacity: usize) -> Vec<i16> {
        input
            .chunks_exact(2)
            .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
            .filter(|&value| u32::from(value) < 5 * u32::from(MODULUS))
            .map(|value| {
                let residue = i32::from(value % MODULUS);
                let centered = if residue > i32::from(MODULUS / 2) {
                    residue - i32::from(MODULUS)
                } else {
                    residue
                };
                i16::try_from(centered).expect("centered residue fits in i16")
            })
            .take(capacity)
            .collect()
    }

    fn fixtures<const MODULUS: u16>(count: usize) -> Vec<Vec<u8>> {
        let q = u32::from(MODULUS);
        let bound = 5 * q;
        let maximum = u32::from(u16::MAX);
        let thresholds = [
            0,
            q - 1,
            q,
            q + 1,
            q / 2,
            q / 2 + 1,
            bound - 1,
            bound,
            (bound + 1).min(maximum),
            maximum,
        ];
        (0..6)
            .map(|pattern| {
                (0..count)
                    .flat_map(|index| {
                        let accepted = (u32::try_from(index).expect("fixture index fits") * 257
                            + q / 2)
                            % bound;
                        let value = match pattern {
                            0 => accepted,
                            1 => {
                                bound
                                    + u32::try_from(index).expect("fixture index fits")
                                        % (maximum - bound + 1)
                            }
                            2 if index % 17 == 3 => accepted,
                            3 => thresholds[index % thresholds.len()],
                            4 if index >= count / 8 * 8 => accepted,
                            5 if index >= count / 32 * 32 => accepted,
                            _ => maximum,
                        };
                        u16::try_from(value)
                            .expect("fixture value fits in u16")
                            .to_be_bytes()
                    })
                    .collect()
            })
            .collect()
    }

    pub(super) fn check_cases<const MODULUS: u16>(kernel: impl Fn(&[u8], &mut [i16]) -> usize) {
        const CANARY: i16 = 23_456;
        for input_count in [0_usize, 1, 7, 8, 9, 31, 32, 33, 67, 68, 69, 84, 96, 129] {
            for (pattern, input) in fixtures::<MODULUS>(input_count).iter().enumerate() {
                for capacity in [
                    0,
                    1,
                    7,
                    8,
                    31,
                    32,
                    67,
                    68,
                    input_count.saturating_sub(1),
                    input_count,
                ] {
                    let expected = reference::<MODULUS>(input, capacity);
                    let mut guarded = vec![CANARY; capacity + 4];
                    let count = kernel(input, &mut guarded[2..2 + capacity]);
                    assert_eq!(
                        count,
                        expected.len(),
                        "q={MODULUS}, n={input_count}, capacity={capacity}, pattern={pattern}"
                    );
                    assert_eq!(
                        &guarded[2..2 + count],
                        expected,
                        "q={MODULUS}, n={input_count}, capacity={capacity}, pattern={pattern}"
                    );
                    assert_eq!(&guarded[..2], &[CANARY; 2]);
                    assert_eq!(&guarded[2 + capacity..], &[CANARY; 2]);
                }
            }
        }
    }

    fn check_public_and_scalar<const MODULUS: u16>() {
        check_cases::<MODULUS>(compact_centered_be_u16_5q::<MODULUS>);
        check_cases::<MODULUS>(|input, output| {
            if output.is_empty() {
                0
            } else {
                compact_centered_u16_scalar::<MODULUS>(input, output)
            }
        });
    }

    fn check_rate<const MODULUS: u16>() {
        const CANARY: i16 = 23_456;
        for (pattern, input) in fixtures::<MODULUS>(68).iter().enumerate() {
            let words: [u64; RATE_WORDS] = core::array::from_fn(|word| {
                u64::from_le_bytes(
                    input[8 * word..8 * word + 8]
                        .try_into()
                        .expect("eight bytes"),
                )
            });
            for len in [0, 1, 7, 8, 9, 31, 32, 63, 64, 65, 67, 68, 69, 80, 140] {
                for filled in [0, 1, 3, 7, 8, 9, 30, 61, 64, 79, 139] {
                    if filled > len {
                        continue;
                    }
                    let mut expected: Vec<i16> = (-50..).take(filled).collect();
                    expected.extend(reference::<MODULUS>(input, len - filled));
                    let mut guarded = vec![CANARY; len + 4];
                    guarded[2..2 + filled].copy_from_slice(&expected[..filled]);
                    let count = compact_centered_rate_5q::<MODULUS>(
                        &words,
                        &mut guarded[2..2 + len],
                        filled,
                    );
                    let context =
                        format!("q={MODULUS}, len={len}, filled={filled}, pattern={pattern}");
                    assert_eq!(count, expected.len(), "{context}");
                    assert_eq!(&guarded[2..2 + count], expected, "{context}");
                    assert_eq!(&guarded[..2], &[CANARY; 2], "{context}");
                    assert_eq!(&guarded[2 + len..], &[CANARY; 2], "{context}");
                }
            }
        }
        // Pseudo-random blocks cover mixed masks in every vector position,
        // and successive blocks fill rows across the row end.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for len in [68, 512, 1024] {
            for _ in 0..200 {
                let mut row = vec![0_i16; len];
                let mut expected = Vec::new();
                let mut filled = 0;
                while filled < len {
                    let words: [u64; RATE_WORDS] = core::array::from_fn(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        state
                    });
                    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                    expected.extend(reference::<MODULUS>(&bytes, len - filled));
                    filled = compact_centered_rate_5q::<MODULUS>(&words, &mut row, filled);
                    assert_eq!(&row[..filled], expected, "q={MODULUS}, len={len}");
                }
            }
        }
    }

    #[test]
    fn rate_words_match_reference() {
        check_rate::<1>();
        check_rate::<3>();
        check_rate::<257>();
        check_rate::<12_289>();
        check_rate::<13_107>();
    }

    #[test]
    fn capacity_regimes_match_reference() {
        check_public_and_scalar::<1>();
        check_public_and_scalar::<3>();
        check_public_and_scalar::<257>();
        check_public_and_scalar::<12_289>();
        check_public_and_scalar::<13_107>();
    }

    #[test]
    #[should_panic(expected = "candidate bytes must be paired")]
    fn odd_byte_length_panics() {
        compact_centered_be_u16_5q::<3>(&[0], &mut []);
    }

    #[test]
    #[should_panic(expected = "centered modulus must be odd")]
    fn zero_modulus_panics() {
        compact_centered_be_u16_5q::<0>(&[], &mut []);
    }

    #[test]
    #[should_panic(expected = "centered modulus must be odd")]
    fn even_modulus_panics() {
        compact_centered_be_u16_5q::<2>(&[], &mut []);
    }

    #[test]
    #[should_panic(expected = "five residue intervals must fit in u16")]
    fn excessive_modulus_panics() {
        compact_centered_be_u16_5q::<13_109>(&[], &mut []);
    }
}
