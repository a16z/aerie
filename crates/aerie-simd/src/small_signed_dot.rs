//! Exact inner products between signed 32-bit digit streams and `i16` factors.
//!
//! A caller splits each wide value `x` into two signed digits,
//! `x = d_lo + 2^32 d_hi` with `d_lo, d_hi` in `i32`, and passes four digit
//! streams `[x0_lo, x0_hi, x1_lo, x1_hi]`. Every kernel returns
//! `[sum_i f_i x0_i, sum_i f_i x1_i]` exactly, for any `i32` digits, any
//! `i16` factors, and at most `MAX_TERMS` terms.

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod aarch64;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod x86;

use crate::Backend;

// Each digit product has magnitude at most 2^31 * 2^15 = 2^46. With at most
// 2^16 terms, any sum of products, and so every i64 accumulator lane however
// the terms are distributed, is at most 2^62 in magnitude. The two-digit
// reconstruction `s_lo + 2^32 s_hi` is then below 2^95 and exact in i128.
pub(crate) const MAX_TERMS: usize = 1 << 16;
type DotKernel = fn([&[i32]; 4], &[i16]) -> [i128; 2];

/// A reusable, runtime-safe signed-digit dot implementation.
#[derive(Clone, Copy)]
pub struct SmallSignedDotPlan {
    kernel: DotKernel,
    name: &'static str,
}

impl core::fmt::Debug for SmallSignedDotPlan {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SmallSignedDotPlan")
            .field("kernel", &self.name)
            .finish()
    }
}

impl SmallSignedDotPlan {
    /// Select the widest vectorized implementation at or below
    /// [`Backend::selected`], or return `None` when only the scalar fallback
    /// is available.
    #[must_use]
    pub fn best_available() -> Option<Self> {
        let ceiling = Backend::selected();
        if ceiling >= Backend::Avx512 {
            if let Some(plan) = Self::avx512() {
                return Some(plan);
            }
        }
        if ceiling >= Backend::Avx2 {
            if let Some(plan) = Self::avx2() {
                return Some(plan);
            }
        }
        if ceiling >= Backend::Neon {
            return Self::neon();
        }
        None
    }

    /// Select the native platform implementation (NEON on `AArch64`, scalar
    /// elsewhere).
    #[must_use]
    pub fn platform() -> Self {
        Self::neon().unwrap_or_else(Self::scalar)
    }

    /// The portable scalar reference implementation.
    #[must_use]
    pub fn scalar() -> Self {
        Self {
            kernel: dot_scalar,
            name: "scalar",
        }
    }

    /// Select NEON when it is available.
    #[must_use]
    pub fn neon() -> Option<Self> {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Some(Self {
                kernel: aarch64::dot_neon,
                name: "neon",
            });
        }
        None
    }

    /// Select AVX2 when it is available.
    #[must_use]
    pub fn avx2() -> Option<Self> {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            return Some(Self {
                kernel: x86::dot_avx2,
                name: "avx2",
            });
        }
        None
    }

    /// Select AVX-512 when the required F subset is available.
    #[must_use]
    pub fn avx512() -> Option<Self> {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f") {
            return Some(Self {
                kernel: x86::dot_avx512,
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

    /// Compute `[sum_i f_i (d0_i + 2^32 d1_i), sum_i f_i (d2_i + 2^32 d3_i)]`
    /// exactly from fixed, equally sized streams `[d0, d1, d2, d3]`.
    #[must_use]
    pub fn dot<const N: usize>(self, digits: [&[i32; N]; 4], factors: &[i16; N]) -> [i128; 2] {
        assert!(
            N <= MAX_TERMS,
            "signed-digit inner product exceeds exact bound"
        );
        (self.kernel)(digits.map(<[i32; N]>::as_slice), factors)
    }
}

fn dot_scalar(digits: [&[i32]; 4], factors: &[i16]) -> [i128; 2] {
    finish_digit_sums([0; 4], digits, factors, 0)
}

/// Add the products of terms `start..` to per-stream sums and reconstruct
/// the two wide values. Shared by every kernel's scalar tail.
fn finish_digit_sums(
    mut sums: [i128; 4],
    digits: [&[i32]; 4],
    factors: &[i16],
    start: usize,
) -> [i128; 2] {
    for (index, &factor) in factors.iter().enumerate().skip(start) {
        for (sum, stream) in sums.iter_mut().zip(digits) {
            *sum += i128::from(stream[index]) * i128::from(factor);
        }
    }
    [sums[0] + (sums[1] << 32), sums[2] + (sums[3] << 32)]
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

    fn plans() -> Vec<SmallSignedDotPlan> {
        [
            Some(SmallSignedDotPlan::scalar()),
            SmallSignedDotPlan::neon(),
            SmallSignedDotPlan::avx2(),
            SmallSignedDotPlan::avx512(),
            SmallSignedDotPlan::best_available(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// Reference with no shared code: reconstruct each wide value first.
    fn reference(digits: [&[i32]; 4], factors: &[i16]) -> [i128; 2] {
        let mut sums = [0_i128; 2];
        for (index, &factor) in factors.iter().enumerate() {
            let x0 = i128::from(digits[0][index]) + (i128::from(digits[1][index]) << 32);
            let x1 = i128::from(digits[2][index]) + (i128::from(digits[3][index]) << 32);
            sums[0] += x0 * i128::from(factor);
            sums[1] += x1 * i128::from(factor);
        }
        sums
    }

    fn pseudo_random(seed: u64, index: usize) -> u64 {
        let mut state = seed ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        state ^= state >> 31;
        state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        state ^ (state >> 29)
    }

    const EXTREME_DIGITS: [i32; 6] = [i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX];
    const EXTREME_FACTORS: [i16; 6] = [i16::MIN, i16::MIN + 1, -1, 0, 1, i16::MAX];

    fn check_all_plans(digits: &[Vec<i32>; 4], factors: &[i16], label: &str) {
        let streams = [
            digits[0].as_slice(),
            digits[1].as_slice(),
            digits[2].as_slice(),
            digits[3].as_slice(),
        ];
        let expected = reference(streams, factors);
        for plan in plans() {
            assert_eq!(
                (plan.kernel)(streams, factors),
                expected,
                "{} on {label}",
                plan.name()
            );
        }
    }

    #[test]
    fn every_kernel_matches_reference_on_all_tails() {
        // Every length through three full AVX-512 blocks plus each tail.
        for len in 0..=64 {
            let digits = core::array::from_fn(|stream| {
                (0..len)
                    .map(|index| pseudo_random(stream as u64 + 1, index) as u32 as i32)
                    .collect()
            });
            let factors: Vec<i16> = (0..len)
                .map(|index| pseudo_random(99, index) as u16 as i16)
                .collect();
            check_all_plans(&digits, &factors, &format!("random len {len}"));
        }
    }

    #[test]
    fn every_kernel_matches_reference_on_extreme_constant_streams() {
        for len in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 517] {
            for &digit in &EXTREME_DIGITS {
                for &factor in &EXTREME_FACTORS {
                    let digits = core::array::from_fn(|stream| {
                        // Mix signs across streams so the low and high
                        // digits of one value disagree.
                        let value = if stream % 2 == 0 {
                            digit
                        } else {
                            digit.wrapping_neg()
                        };
                        vec![value; len]
                    });
                    let factors = vec![factor; len];
                    check_all_plans(&digits, &factors, &format!("{digit}*{factor} x{len}"));
                }
            }
        }
    }

    #[test]
    fn every_kernel_matches_reference_on_extreme_mixed_streams() {
        let len = 6 * 6 * 4 + 5;
        let digits = core::array::from_fn(|stream| {
            (0..len)
                .map(|index| EXTREME_DIGITS[(index / (stream + 1)) % 6])
                .collect()
        });
        let factors: Vec<i16> = (0..len).map(|index| EXTREME_FACTORS[index % 6]).collect();
        check_all_plans(&digits, &factors, "mixed extremes");
    }

    #[test]
    fn every_kernel_is_exact_at_the_public_term_bound() {
        // All products share one sign and the maximal magnitude 2^46, so
        // each stream sums to exactly 2^62 in magnitude.
        for (digit, factor) in [
            (i32::MIN, i16::MIN),
            (i32::MIN, i16::MAX),
            (i32::MAX, i16::MIN),
        ] {
            let digits: [Box<[i32; MAX_TERMS]>; 4] = core::array::from_fn(|_| {
                vec![digit; MAX_TERMS]
                    .into_boxed_slice()
                    .try_into()
                    .expect("fixed digit length")
            });
            let factors: Box<[i16; MAX_TERMS]> = vec![factor; MAX_TERMS]
                .into_boxed_slice()
                .try_into()
                .expect("fixed factor length");
            let refs = [&*digits[0], &*digits[1], &*digits[2], &*digits[3]];
            let expected = reference(refs.map(<[i32; MAX_TERMS]>::as_slice), factors.as_slice());
            let lane = i128::from(digit) * i128::from(factor) * MAX_TERMS as i128;
            assert_eq!(expected, [lane + (lane << 32); 2]);
            for plan in plans() {
                assert_eq!(plan.dot(refs, &factors), expected, "{}", plan.name());
            }
        }
    }

    #[test]
    fn best_available_respects_the_selected_ceiling() {
        let plan = SmallSignedDotPlan::best_available();
        match Backend::selected() {
            Backend::Scalar => assert!(plan.is_none()),
            backend => assert_eq!(plan.map(SmallSignedDotPlan::name), Some(backend.name())),
        }
    }
}
