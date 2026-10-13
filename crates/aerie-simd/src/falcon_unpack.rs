//! Unpack Falcon public-key payloads directly to centered `i16` coefficients.
//!
//! A Falcon public key stores each residue `h_i mod q` in 14 big-endian bits,
//! four residues per seven bytes. Every kernel maps a 14-bit residue `r` to
//! `r - q` when `r > (q - 1) / 2` and to `r` otherwise. The map is total on
//! 14-bit inputs, so the kernels agree even on non-canonical residues; callers
//! still reject `r >= q` when parsing a key.

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod aarch64;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod x86;

use std::sync::OnceLock;

use crate::Backend;

const Q: u16 = 12_289;
const HALF: u16 = (Q - 1) / 2;
const BLOCK_BYTES: usize = 7;
const BLOCK_RESIDUES: usize = 4;

type UnpackKernel = fn(&[u8], &mut [i16]);

/// A reusable, runtime-selected 14-bit public-key unpacker.
#[derive(Clone, Copy)]
pub struct KeyUnpackPlan {
    kernel: UnpackKernel,
    name: &'static str,
}

impl core::fmt::Debug for KeyUnpackPlan {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("KeyUnpackPlan")
            .field("kernel", &self.name)
            .finish()
    }
}

impl KeyUnpackPlan {
    /// The widest implementation at or below [`Backend::selected`] that this
    /// CPU supports, falling back to scalar. Selected once per process.
    #[must_use]
    pub fn best_available() -> Self {
        static SELECTED: OnceLock<KeyUnpackPlan> = OnceLock::new();
        *SELECTED.get_or_init(|| {
            let ceiling = Backend::selected();
            if ceiling >= Backend::Avx512 {
                if let Some(plan) = Self::avx512() {
                    return plan;
                }
            }
            if ceiling >= Backend::Avx2 {
                if let Some(plan) = Self::avx2() {
                    return plan;
                }
            }
            if ceiling >= Backend::Neon {
                if let Some(plan) = Self::neon() {
                    return plan;
                }
            }
            Self::scalar()
        })
    }

    /// The portable scalar reference implementation.
    #[must_use]
    pub fn scalar() -> Self {
        Self {
            kernel: unpack_scalar,
            name: "scalar",
        }
    }

    /// Select NEON table lookups when available.
    #[must_use]
    pub fn neon() -> Option<Self> {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Some(Self {
                kernel: aarch64::unpack_neon,
                name: "neon",
            });
        }
        None
    }

    /// Select AVX2 byte shuffles when available.
    #[must_use]
    pub fn avx2() -> Option<Self> {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            return Some(Self {
                kernel: x86::unpack_avx2,
                name: "avx2",
            });
        }
        None
    }

    /// Select AVX-512 byte permutes when the BW and VBMI subsets are
    /// available.
    #[must_use]
    pub fn avx512() -> Option<Self> {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
        {
            return Some(Self {
                kernel: x86::unpack_avx512,
                name: "avx512",
            });
        }
        None
    }

    /// Name of the selected kernel.
    #[must_use]
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// Unpack `out.len()` centered coefficients from a packed payload (the
    /// public key without its one-byte header).
    ///
    /// # Panics
    ///
    /// Panics unless `out.len()` is a multiple of four and `payload` holds
    /// exactly `14 * out.len()` bits.
    pub fn unpack_centered_14(self, payload: &[u8], out: &mut [i16]) {
        assert!(
            out.len().is_multiple_of(BLOCK_RESIDUES)
                && payload.len().checked_mul(BLOCK_RESIDUES) == out.len().checked_mul(BLOCK_BYTES)
                && out.len().checked_mul(BLOCK_BYTES).is_some(),
            "a 14-bit payload of {} bytes cannot fill {} coefficients",
            payload.len(),
            out.len()
        );
        (self.kernel)(payload, out);
    }
}

fn unpack_scalar(payload: &[u8], out: &mut [i16]) {
    for (block, slots) in payload
        .chunks_exact(BLOCK_BYTES)
        .zip(out.chunks_exact_mut(BLOCK_RESIDUES))
    {
        let word = block
            .iter()
            .fold(0_u64, |word, &byte| (word << 8) | u64::from(byte));
        for (slot, shift) in slots.iter_mut().zip([42, 28, 14, 0]) {
            #[allow(clippy::cast_possible_truncation)] // Masked to 14 bits.
            let residue = ((word >> shift) & 0x3fff) as u16;
            *slot = center(residue);
        }
    }
}

const fn center(residue: u16) -> i16 {
    if residue > HALF {
        residue.cast_signed() - Q.cast_signed()
    } else {
        residue.cast_signed()
    }
}

/// Vector kernels finish the last partial block of whole payload groups here:
/// `payload` and `out` are the unconsumed suffixes, and `WIDTH` bytes of zero
/// padding let the kernel run once with in-bounds loads.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn unpack_padded_tail<const WIDTH: usize, const RESIDUES: usize>(
    payload: &[u8],
    out: &mut [i16],
    kernel_step: impl FnOnce(&[u8; WIDTH], &mut [i16; RESIDUES]),
) {
    if out.is_empty() {
        return;
    }
    let mut bytes = [0_u8; WIDTH];
    bytes[..payload.len()].copy_from_slice(payload);
    let mut residues = [0_i16; RESIDUES];
    kernel_step(&bytes, &mut residues);
    out.copy_from_slice(&residues[..out.len()]);
}

#[cfg(test)]
// Tests truncate pseudo-random words into fixture digits on purpose.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
mod tests {
    use super::*;

    fn plans() -> Vec<KeyUnpackPlan> {
        [
            Some(KeyUnpackPlan::scalar()),
            KeyUnpackPlan::neon(),
            KeyUnpackPlan::avx2(),
            KeyUnpackPlan::avx512(),
            Some(KeyUnpackPlan::best_available()),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// Bit-by-bit packer, independent of every kernel.
    fn pack(residues: &[u16]) -> Vec<u8> {
        let mut bytes = vec![0_u8; residues.len() * 14 / 8];
        for (index, &residue) in residues.iter().enumerate() {
            assert!(residue < 1 << 14);
            for bit in 0..14 {
                if residue >> (13 - bit) & 1 == 1 {
                    let position = 14 * index + bit;
                    bytes[position / 8] |= 0x80 >> (position % 8);
                }
            }
        }
        bytes
    }

    fn reference(residues: &[u16]) -> Vec<i16> {
        residues
            .iter()
            .map(|&residue| {
                let value = i32::from(residue);
                i16::try_from(if value > 6_144 { value - 12_289 } else { value })
                    .expect("centered 14-bit residue fits i16")
            })
            .collect()
    }

    fn check_all_plans(residues: &[u16], label: &str) {
        let payload = pack(residues);
        let expected = reference(residues);
        for plan in plans() {
            // Poison the output so a skipped slot cannot pass by accident.
            let mut actual = vec![i16::MIN; residues.len()];
            plan.unpack_centered_14(&payload, &mut actual);
            assert_eq!(actual, expected, "{} on {label}", plan.name());
        }
    }

    #[test]
    fn every_kernel_unpacks_every_14_bit_residue_in_every_slot() {
        // All 2^14 residues, including the non-canonical ones at or above q,
        // rotated through the four slots of a block and the lanes of every
        // vector width.
        for rotation in 0..32 {
            let residues: Vec<u16> = (0..1_u16 << 14)
                .map(|index| index.wrapping_add(rotation * 517) & 0x3fff)
                .collect();
            check_all_plans(&residues, &format!("rotation {rotation}"));
        }
    }

    #[test]
    fn every_kernel_handles_every_block_count() {
        // Every length in blocks through several AVX-512 iterations, so every
        // vector tail and padded-step path runs.
        let edge = [0, 1, HALF - 1, HALF, HALF + 1, Q - 1, Q, 0x3fff];
        for blocks in 0..=40 {
            let residues: Vec<u16> = (0..blocks * BLOCK_RESIDUES)
                .map(|index| {
                    if index % 3 == 0 {
                        edge[(index / 3) % edge.len()]
                    } else {
                        ((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 50) as u16
                    }
                })
                .collect();
            check_all_plans(&residues, &format!("{blocks} blocks"));
        }
    }

    #[test]
    fn every_kernel_handles_falcon_key_sizes_with_extreme_residues() {
        for coefficients in [512, 1024] {
            for &residue in &[0, HALF, HALF + 1, Q - 1, 0x3fff] {
                check_all_plans(
                    &vec![residue; coefficients],
                    &format!("{residue} x{coefficients}"),
                );
            }
        }
    }

    #[test]
    fn best_available_respects_the_selected_ceiling() {
        let expected = match Backend::selected() {
            Backend::Avx512 if KeyUnpackPlan::avx512().is_none() => "avx2",
            backend => backend.name(),
        };
        assert_eq!(KeyUnpackPlan::best_available().name(), expected);
    }

    #[test]
    #[should_panic(expected = "cannot fill")]
    fn mismatched_lengths_are_rejected() {
        KeyUnpackPlan::scalar().unpack_centered_14(&[0; 7], &mut [0; 8]);
    }
}
