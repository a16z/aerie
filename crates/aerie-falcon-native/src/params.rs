//! Falcon parameter sets.
//!
//! [`FalconParams`] fixes everything in the byte layer and the native
//! verification relation that depends on the ring degree: byte lengths,
//! header bytes, the squared verification bound, the padded-body bit budget,
//! and the budget-digit layout used by the exact encodability witness. The
//! two standardized parameter sets are [`Falcon512`] and [`Falcon1024`]. The
//! trait is sealed: the aggregation protocols only accept these two.

use core::fmt::Debug;

/// Falcon modulus, shared by every parameter set.
pub const FALCON_Q: i64 = 12_289;
/// Salt length in bytes, shared by every parameter set.
pub const FALCON_SALT_BYTES: usize = 40;

/// Bits used by every compressed Falcon coefficient: sign, seven low bits,
/// and the unary terminator.
pub const COMPRESSED_COEFFICIENT_BASE_BITS: usize = 9;

/// Fixed-length array abstraction so that degree-dependent tables can be
/// stored inline without unstable const-generic arithmetic.
pub trait FixedArray<T: Copy>:
    Copy + Eq + Debug + Send + Sync + AsRef<[T]> + AsMut<[T]> + 'static
{
    /// Number of elements.
    const LEN: usize;
    /// An array with every element equal to `value`.
    fn filled(value: T) -> Self;
    /// Copy a slice of exactly `LEN` elements.
    fn from_slice(slice: &[T]) -> Option<Self>;
}

impl<T, const N: usize> FixedArray<T> for [T; N]
where
    T: Copy + Eq + Debug + Send + Sync + 'static,
{
    const LEN: usize = N;

    fn filled(value: T) -> Self {
        [value; N]
    }

    fn from_slice(slice: &[T]) -> Option<Self> {
        slice.try_into().ok()
    }
}

/// Element bound accepted by [`FalconParams::Array`].
pub trait ArrayElement: Copy + Eq + Debug + Send + Sync + 'static {}

impl<T: Copy + Eq + Debug + Send + Sync + 'static> ArrayElement for T {}

mod sealed {
    pub trait Sealed {}
}

/// A standardized Falcon parameter set.
pub trait FalconParams:
    sealed::Sealed + Copy + Clone + Debug + Default + Eq + PartialEq + Send + Sync + 'static
{
    /// Human-readable name used in error messages, for example `Falcon-512`.
    const NAME: &'static str;
    /// Base-two logarithm of the ring degree.
    const LOGN: u32;
    /// Ring degree `n`.
    const DEGREE: usize;
    /// Canonical public-key length: one header byte plus fourteen bits per
    /// coefficient.
    const PUBLIC_KEY_BYTES: usize;
    /// Padded signature length: header, salt, and the fixed compressed body.
    const SIGNATURE_BYTES: usize;
    /// Public-key header byte, `0x00 + logn`.
    const PUBLIC_KEY_HEADER: u8;
    /// Padded-signature header byte, `0x30 + logn`.
    const PADDED_SIGNATURE_HEADER: u8;
    /// Squared verification bound `floor(beta^2)`.
    const BETA_SQUARED: u64;
    /// Bit capacity of the compressed coefficient body in a padded signature.
    const PADDED_BODY_BITS: usize;
    /// Bits available for all unary high parts after the fixed coefficient
    /// cost of nine bits per coefficient.
    const COMPRESSED_HIGH_BITS_BUDGET: usize;
    /// Radices of the three balanced budget-slack digits, least significant
    /// first.
    const BUDGET_DIGIT_RADICES: [u8; 3];
    /// Constant added when recomposing the three balanced budget digits into
    /// the unsigned slack.
    const BUDGET_CONSTANT: i16;
    /// Inline table with one element per coefficient.
    type Array<T: ArrayElement>: FixedArray<T>;
    /// Canonical public-key byte string.
    type PublicKeyBytes: FixedArray<u8>;
    /// Padded signature byte string.
    type SignatureBytes: FixedArray<u8>;
}

/// An array with one element per coefficient, every element equal to `value`.
#[must_use]
pub fn filled<P: FalconParams, T: ArrayElement>(value: T) -> P::Array<T> {
    <P::Array<T> as FixedArray<T>>::filled(value)
}

/// Falcon-512 (FN-DSA-512).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Falcon512;

impl sealed::Sealed for Falcon512 {}

impl FalconParams for Falcon512 {
    const NAME: &'static str = "Falcon-512";
    const LOGN: u32 = 9;
    const DEGREE: usize = 512;
    const PUBLIC_KEY_BYTES: usize = 897;
    const SIGNATURE_BYTES: usize = 666;
    const PUBLIC_KEY_HEADER: u8 = 0x09;
    const PADDED_SIGNATURE_HEADER: u8 = 0x39;
    const BETA_SQUARED: u64 = 34_034_726;
    const PADDED_BODY_BITS: usize = (666 - 1 - FALCON_SALT_BYTES) * 8;
    const COMPRESSED_HIGH_BITS_BUDGET: usize =
        Self::PADDED_BODY_BITS - 512 * COMPRESSED_COEFFICIENT_BASE_BITS;
    const BUDGET_DIGIT_RADICES: [u8; 3] = [8, 8, 8];
    const BUDGET_CONSTANT: i16 = 292;
    type Array<T: ArrayElement> = [T; 512];
    type PublicKeyBytes = [u8; 897];
    type SignatureBytes = [u8; 666];
}

/// Falcon-1024 (FN-DSA-1024).
///
/// Parameters follow Table 3.3 of the Falcon specification: degree 1,024,
/// `floor(beta^2) = 70,265,242`, a 1,793-byte public key, and a 1,280-byte
/// padded signature. The compressed coefficient limit and the salt length
/// are the same as for Falcon-512. See `specs/falcon.md` Section 2.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Falcon1024;

impl sealed::Sealed for Falcon1024 {}

impl FalconParams for Falcon1024 {
    const NAME: &'static str = "Falcon-1024";
    const LOGN: u32 = 10;
    const DEGREE: usize = 1024;
    const PUBLIC_KEY_BYTES: usize = 1793;
    const SIGNATURE_BYTES: usize = 1280;
    const PUBLIC_KEY_HEADER: u8 = 0x0A;
    const PADDED_SIGNATURE_HEADER: u8 = 0x3A;
    const BETA_SQUARED: u64 = 70_265_242;
    const PADDED_BODY_BITS: usize = (1280 - 1 - FALCON_SALT_BYTES) * 8;
    const COMPRESSED_HIGH_BITS_BUDGET: usize =
        Self::PADDED_BODY_BITS - 1024 * COMPRESSED_COEFFICIENT_BASE_BITS;
    const BUDGET_DIGIT_RADICES: [u8; 3] = [8, 8, 16];
    const BUDGET_CONSTANT: i16 = 548;
    type Array<T: ArrayElement> = [T; 1024];
    type PublicKeyBytes = [u8; 1793];
    type SignatureBytes = [u8; 1280];
}

/// Compile-time consistency of one parameter set.
const fn consistent<P: FalconParams>() -> bool {
    let radices = P::BUDGET_DIGIT_RADICES;
    let slack_values = radices[0] as usize * radices[1] as usize * radices[2] as usize;
    let constant = radices[0] as i16 / 2
        + radices[0] as i16 * (radices[1] as i16 / 2)
        + radices[0] as i16 * radices[1] as i16 * (radices[2] as i16 / 2);
    P::DEGREE == 1 << P::LOGN
        && P::PUBLIC_KEY_BYTES == 1 + P::DEGREE * 14 / 8
        && P::PUBLIC_KEY_HEADER == P::LOGN as u8
        && P::PADDED_SIGNATURE_HEADER == 0x30 + P::LOGN as u8
        && P::PADDED_BODY_BITS == (P::SIGNATURE_BYTES - 1 - FALCON_SALT_BYTES) * 8
        && P::COMPRESSED_HIGH_BITS_BUDGET
            == P::PADDED_BODY_BITS - P::DEGREE * COMPRESSED_COEFFICIENT_BASE_BITS
        && slack_values > P::COMPRESSED_HIGH_BITS_BUDGET
        && P::BUDGET_CONSTANT == constant
        && <P::Array<i16> as FixedArray<i16>>::LEN == P::DEGREE
        && <P::PublicKeyBytes as FixedArray<u8>>::LEN == P::PUBLIC_KEY_BYTES
        && <P::SignatureBytes as FixedArray<u8>>::LEN == P::SIGNATURE_BYTES
}

const _: () = assert!(consistent::<Falcon512>());
const _: () = assert!(consistent::<Falcon1024>());
const _: () = assert!(Falcon512::COMPRESSED_HIGH_BITS_BUDGET == 392);
const _: () = assert!(Falcon1024::COMPRESSED_HIGH_BITS_BUDGET == 696);

/// Signed selector cap for `s1`, `s2`, and four-square slack.
///
/// Shared by Falcon-512 and Falcon-1024: signed width 14 represents
/// `[-S_CAP, S_CAP - 1]`. The honest capped four-square producer emits
/// nonnegative slack components in `[0, S_CAP - 1]`.
pub const S_CAP: i64 = 8_192;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falcon_1024_constants_match_the_specification_table() {
        assert_eq!(Falcon1024::PADDED_BODY_BITS, 9_912);
        assert_eq!(Falcon1024::COMPRESSED_HIGH_BITS_BUDGET, 696);
        assert!(8_382_u64.pow(2) <= Falcon1024::BETA_SQUARED);
        assert!(8_383_u64.pow(2) > Falcon1024::BETA_SQUARED);
        assert_eq!(Falcon512::PADDED_BODY_BITS, 5_000);
        assert!(5_833_u64.pow(2) <= Falcon512::BETA_SQUARED);
        assert!(5_834_u64.pow(2) > Falcon512::BETA_SQUARED);
    }

    #[test]
    fn filled_arrays_have_the_degree_length() {
        assert_eq!(filled::<Falcon512, i16>(3).len(), 512);
        assert_eq!(filled::<Falcon1024, i16>(3).len(), 1024);
        assert!(filled::<Falcon1024, i8>(-1).iter().all(|&cell| cell == -1));
    }
}
