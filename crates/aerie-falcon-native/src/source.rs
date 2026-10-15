//! Canonical Falcon public-key and padded-signature byte strings.

use crate::{
    error::{AerieError, Result},
    params::{filled, FalconParams, FixedArray, FALCON_Q, FALCON_SALT_BYTES},
};
#[cfg(test)]
use crate::{Falcon1024, Falcon512};

/// Largest coefficient magnitude admitted by Falcon's compressed encoding.
pub const FALCON_COMPRESSED_COEFFICIENT_LIMIT: u16 = 2_047;

const PUBLIC_COEFFICIENT_BITS: usize = 14;
const PUBLIC_COEFFICIENT_BLOCK_BYTES: usize = 7;
const SIGNATURE_PAYLOAD_OFFSET: usize = 1 + FALCON_SALT_BYTES;

/// Canonical public-key bytes for parameter set `P`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalPublicKeyBytes<P: FalconParams>(P::PublicKeyBytes);

impl<P: FalconParams> CanonicalPublicKeyBytes<P> {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let encoded = P::PublicKeyBytes::from_slice(bytes).ok_or_else(|| {
            AerieError::InvalidRecord(format!(
                "{} public key has {} bytes; expected {}",
                P::NAME,
                bytes.len(),
                P::PUBLIC_KEY_BYTES
            ))
        })?;
        if encoded.as_ref()[0] != P::PUBLIC_KEY_HEADER {
            return Err(AerieError::InvalidRecord(format!(
                "{} public-key header is not {:#04x}",
                P::NAME,
                P::PUBLIC_KEY_HEADER
            )));
        }
        validate_public_coefficients(&encoded.as_ref()[1..])?;
        Ok(Self(encoded))
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &P::PublicKeyBytes {
        &self.0
    }

    fn payload(&self) -> &[u8] {
        &self.0.as_ref()[1..]
    }

    /// Decode coefficients into centered representatives in `[-6144, 6144]`.
    #[must_use]
    pub fn decode(&self) -> DecodedPublicKey<P> {
        // Write four coefficients per seven-byte block: the flattened
        // per-coefficient iterator costs the verifier's key functional about
        // half again its time at 16K records.
        let mut h_centered = filled::<P, i16>(0);
        for (slots, block) in h_centered
            .as_mut()
            .chunks_exact_mut(4)
            .zip(self.centered_coefficient_blocks())
        {
            slots.copy_from_slice(&block);
        }
        DecodedPublicKey { h_centered }
    }

    /// Stream the centered coefficients without materializing a decoded
    /// public-key object.
    ///
    /// Four 14-bit residues occupy each seven-byte block. Construction has
    /// already checked canonicality, so this iterator only unpacks and centers
    /// the residues. It is intended for verifier kernels that consume every
    /// coefficient exactly once.
    pub fn centered_coefficients(&self) -> impl Iterator<Item = i16> + '_ {
        self.centered_coefficient_blocks().flatten()
    }

    /// Stream four centered coefficients at a time in their native seven-byte
    /// packing groups.
    ///
    /// This form lets verifier kernels unroll their accumulator without the
    /// state machine required by a flattened coefficient iterator.
    pub fn centered_coefficient_blocks(&self) -> impl Iterator<Item = [i16; 4]> + '_ {
        public_residue_blocks(self.payload()).map(|residues| residues.map(center_residue))
    }

    /// Stream four coefficients at a time after translating the centered
    /// interval `[-6144, 6144]` to `[0, 12288]`.
    ///
    /// Verifier kernels can use this nonnegative representation with one
    /// shared `6144 * sum(factors)` correction, avoiding signed products.
    pub fn shifted_centered_coefficient_blocks(&self) -> impl Iterator<Item = [u16; 4]> + '_ {
        public_residue_blocks(self.payload()).map(|residues| residues.map(shift_centered_residue))
    }
}

/// Public polynomial decoded from canonical public-key bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedPublicKey<P: FalconParams> {
    h_centered: P::Array<i16>,
}

impl<P: FalconParams> DecodedPublicKey<P> {
    #[must_use]
    pub const fn h_centered(&self) -> &P::Array<i16> {
        &self.h_centered
    }
}

/// Canonical padded-signature bytes for parameter set `P`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalSignatureBytes<P: FalconParams>(P::SignatureBytes);

impl<P: FalconParams> CanonicalSignatureBytes<P> {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let encoded = P::SignatureBytes::from_slice(bytes).ok_or_else(|| {
            AerieError::InvalidRecord(format!(
                "{} signature has {} bytes; expected {}",
                P::NAME,
                bytes.len(),
                P::SIGNATURE_BYTES
            ))
        })?;
        if encoded.as_ref()[0] != P::PADDED_SIGNATURE_HEADER {
            return Err(AerieError::InvalidRecord(format!(
                "{} padded-signature header is not {:#04x}",
                P::NAME,
                P::PADDED_SIGNATURE_HEADER
            )));
        }
        decode_signature_coefficients::<P>(&encoded.as_ref()[SIGNATURE_PAYLOAD_OFFSET..])?;
        Ok(Self(encoded))
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &P::SignatureBytes {
        &self.0
    }

    /// Borrow the public salt without decompressing the `s2` body.
    #[must_use]
    pub fn salt(&self) -> &[u8; FALCON_SALT_BYTES] {
        self.0.as_ref()[1..SIGNATURE_PAYLOAD_OFFSET]
            .try_into()
            .expect("the Falcon salt range has a fixed array length")
    }

    /// Decode the salt and signed `s2` polynomial.
    #[must_use]
    pub fn decode(&self) -> DecodedSignature<P> {
        DecodedSignature {
            salt: *self.salt(),
            s2: decode_signature_coefficients::<P>(&self.0.as_ref()[SIGNATURE_PAYLOAD_OFFSET..])
                .expect("canonical signature bytes were validated at construction"),
        }
    }
}

/// Salt and `s2` decoded from one canonical signature byte string.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodedSignature<P: FalconParams> {
    salt: [u8; FALCON_SALT_BYTES],
    s2: P::Array<i16>,
}

impl<P: FalconParams> DecodedSignature<P> {
    #[must_use]
    pub const fn salt(&self) -> &[u8; FALCON_SALT_BYTES] {
        &self.salt
    }

    #[must_use]
    pub const fn s2(&self) -> &P::Array<i16> {
        &self.s2
    }
}

fn public_residue_blocks(payload: &[u8]) -> impl Iterator<Item = [u16; 4]> + '_ {
    debug_assert_eq!(payload.len() % PUBLIC_COEFFICIENT_BLOCK_BYTES, 0);
    debug_assert_eq!((payload.len() * 8) % PUBLIC_COEFFICIENT_BITS, 0);
    (0..payload.len() / PUBLIC_COEFFICIENT_BLOCK_BYTES).map(move |block| {
        // One big-endian load of the block and the byte before it; the
        // masks below drop that extra high byte. A byte-at-a-time fold
        // costs about four times as much. The first block has no
        // preceding byte.
        let word = if block == 0 {
            payload[..PUBLIC_COEFFICIENT_BLOCK_BYTES]
                .iter()
                .fold(0_u64, |word, &byte| (word << 8) | u64::from(byte))
        } else {
            let start = PUBLIC_COEFFICIENT_BLOCK_BYTES * block - 1;
            u64::from_be_bytes(
                payload[start..start + 8]
                    .try_into()
                    .expect("an eight-byte window ends at the block's last byte"),
            )
        };
        [
            ((word >> 42) & 0x3fff) as u16,
            ((word >> 28) & 0x3fff) as u16,
            ((word >> 14) & 0x3fff) as u16,
            (word & 0x3fff) as u16,
        ]
    })
}

fn public_residues(payload: &[u8]) -> impl Iterator<Item = u16> + '_ {
    public_residue_blocks(payload).flatten()
}

fn validate_public_coefficients(payload: &[u8]) -> Result<()> {
    for (index, residue) in public_residues(payload).enumerate() {
        if residue >= FALCON_Q as u16 {
            return Err(invalid_encoding(format!(
                "public-key coefficient {index} is non-canonical: {residue}"
            )));
        }
    }
    Ok(())
}

fn decode_signature_coefficients<P: FalconParams>(payload: &[u8]) -> Result<P::Array<i16>> {
    let mut coefficients = filled::<P, i16>(0);
    decode_compressed(payload, coefficients.as_mut()).map_err(|error| match error {
        CompressedError::TruncatedCoefficient => {
            invalid_encoding("truncated signature coefficient stream")
        }
        CompressedError::TruncatedUnary => {
            invalid_encoding("truncated signature unary coefficient")
        }
        CompressedError::TooLarge(index) => invalid_encoding(format!(
            "signature coefficient {index} exceeds {FALCON_COMPRESSED_COEFFICIENT_LIMIT}"
        )),
        CompressedError::NegativeZero(index) => invalid_encoding(format!(
            "signature coefficient {index} encodes negative zero"
        )),
        CompressedError::NonzeroPadding => invalid_encoding("signature padding is not all zero"),
    })?;
    Ok(coefficients)
}

enum CompressedError {
    TruncatedCoefficient,
    TruncatedUnary,
    TooLarge(usize),
    NegativeZero(usize),
    NonzeroPadding,
}

/// Decode Falcon's compressed coefficient encoding into `output`.
///
/// A bit accumulator holds fewer than eight unread bits at the start of each
/// coefficient, so the sign and seven low bits take exactly one byte.
fn decode_compressed(
    payload: &[u8],
    output: &mut [i16],
) -> core::result::Result<(), CompressedError> {
    let mut next = 0;
    let mut accumulator = 0_u32;
    let mut pending = 0_u32;
    for (index, coefficient) in output.iter_mut().enumerate() {
        let &byte = payload
            .get(next)
            .ok_or(CompressedError::TruncatedCoefficient)?;
        next += 1;
        accumulator = (accumulator << 8) | u32::from(byte);
        let sign = (accumulator >> (pending + 7)) & 1;
        let mut magnitude = (accumulator >> pending) & 0x7f;
        loop {
            if pending == 0 {
                let &byte = payload.get(next).ok_or(CompressedError::TruncatedUnary)?;
                next += 1;
                accumulator = (accumulator << 8) | u32::from(byte);
                pending = 8;
            }
            pending -= 1;
            if (accumulator >> pending) & 1 != 0 {
                break;
            }
            magnitude += 128;
            if magnitude > u32::from(FALCON_COMPRESSED_COEFFICIENT_LIMIT) {
                return Err(CompressedError::TooLarge(index));
            }
        }
        // Negative zero is rejected without branching on the sign bit, which
        // is unpredictable: `magnitude - 1` wraps to bit 31 only at zero.
        if sign & (magnitude.wrapping_sub(1) >> 31) != 0 {
            return Err(CompressedError::NegativeZero(index));
        }
        // Negate by `(m ^ -s) + s` for sign bit `s`.
        let mask = sign.wrapping_neg();
        *coefficient = ((magnitude ^ mask).wrapping_add(sign)) as i16;
    }
    if accumulator & ((1 << pending) - 1) != 0 || payload[next..].iter().any(|&byte| byte != 0) {
        return Err(CompressedError::NonzeroPadding);
    }
    Ok(())
}

const fn center_residue(residue: u16) -> i16 {
    let signed = residue.cast_signed();
    if residue > (FALCON_Q as u16 - 1) / 2 {
        signed - 12_289
    } else {
        signed
    }
}

const fn shift_centered_residue(residue: u16) -> u16 {
    if residue > (FALCON_Q as u16 - 1) / 2 {
        residue - (FALCON_Q as u16).div_ceil(2)
    } else {
        residue + (FALCON_Q as u16 - 1) / 2
    }
}

fn invalid_encoding(message: impl Into<String>) -> AerieError {
    AerieError::InvalidRecord(message.into())
}

/// Test-only canonical encoders shared by this crate's unit tests.
#[cfg(test)]
pub(crate) mod test_encoding {
    use super::*;

    pub(crate) fn encode_public<P: FalconParams>(coefficients: &[u16]) -> Vec<u8> {
        assert_eq!(coefficients.len(), P::DEGREE);
        let mut bits = Vec::with_capacity(P::DEGREE * PUBLIC_COEFFICIENT_BITS);
        for coefficient in coefficients {
            push_bits(&mut bits, *coefficient, PUBLIC_COEFFICIENT_BITS);
        }
        let mut encoded = vec![P::PUBLIC_KEY_HEADER];
        encoded.extend(pack_bits(&bits));
        encoded
    }

    pub(crate) fn encode_signature<P: FalconParams>(coefficients: &[i16]) -> Vec<u8> {
        encode_signature_with_salt::<P>(coefficients, &[0_u8; FALCON_SALT_BYTES])
    }

    pub(crate) fn encode_signature_with_salt<P: FalconParams>(
        coefficients: &[i16],
        salt: &[u8; FALCON_SALT_BYTES],
    ) -> Vec<u8> {
        assert_eq!(coefficients.len(), P::DEGREE);
        let mut bits = Vec::new();
        for coefficient in coefficients {
            let magnitude = coefficient.unsigned_abs();
            bits.push(*coefficient < 0);
            push_bits(&mut bits, magnitude & 127, 7);
            bits.extend(std::iter::repeat_n(false, usize::from(magnitude >> 7)));
            bits.push(true);
        }
        assert!(
            bits.len() <= P::PADDED_BODY_BITS,
            "test encoder must not silently truncate an oversized body"
        );
        bits.resize(P::PADDED_BODY_BITS, false);
        let mut encoded = vec![P::PADDED_SIGNATURE_HEADER];
        encoded.extend(salt);
        encoded.extend(pack_bits(&bits));
        encoded
    }

    fn push_bits(output: &mut Vec<bool>, value: u16, count: usize) {
        for shift in (0..count).rev() {
            output.push(((value >> shift) & 1) != 0);
        }
    }

    fn pack_bits(bits: &[bool]) -> Vec<u8> {
        bits.chunks(8)
            .map(|chunk| {
                let byte = chunk
                    .iter()
                    .fold(0_u8, |byte, bit| (byte << 1) | u8::from(*bit));
                byte << ((8 - chunk.len()) % 8)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{test_encoding::*, *};

    fn canonical_public_key_decodes_centered_boundaries_for<P: FalconParams>() {
        let mut coefficients = vec![0_u16; P::DEGREE];
        coefficients[0] = 6_144;
        coefficients[1] = 6_145;
        coefficients[2] = 12_288;
        let canonical = CanonicalPublicKeyBytes::<P>::parse(&encode_public::<P>(&coefficients))
            .expect("canonical public key");
        let decoded = canonical.decode();
        assert_eq!(decoded.h_centered().as_ref().len(), P::DEGREE);
        assert_eq!(&decoded.h_centered().as_ref()[..3], &[6_144, -6_144, -1]);
    }

    #[test]
    fn canonical_public_key_decodes_centered_boundaries() {
        canonical_public_key_decodes_centered_boundaries_for::<Falcon512>();
        canonical_public_key_decodes_centered_boundaries_for::<Falcon1024>();
    }

    fn streaming_matches_materialized_for<P: FalconParams>() {
        let coefficients = (0..P::DEGREE)
            .map(|index| {
                u16::try_from((index * 7_919 + 1_237) % FALCON_Q as usize)
                    .expect("residue fits u16")
            })
            .collect::<Vec<_>>();
        let canonical = CanonicalPublicKeyBytes::<P>::parse(&encode_public::<P>(&coefficients))
            .expect("canonical key");
        let streamed = canonical.centered_coefficients().collect::<Vec<_>>();
        let shifted = canonical
            .shifted_centered_coefficient_blocks()
            .flatten()
            .collect::<Vec<_>>();

        assert_eq!(streamed.len(), P::DEGREE);
        assert_eq!(
            streamed.as_slice(),
            canonical.decode().h_centered().as_ref()
        );
        assert_eq!(
            streamed,
            coefficients
                .iter()
                .map(|&residue| center_residue(residue))
                .collect::<Vec<_>>(),
            "the seven-byte block decoder must preserve the bit-stream order"
        );
        assert_eq!(
            shifted,
            streamed
                .iter()
                .map(|&coefficient| u16::try_from(i32::from(coefficient) + 6_144).unwrap())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn streaming_public_key_coefficients_match_the_materialized_decoder() {
        streaming_matches_materialized_for::<Falcon512>();
        streaming_matches_materialized_for::<Falcon1024>();
    }

    #[test]
    fn public_key_rejects_residue_q() {
        let mut coefficients = vec![0_u16; Falcon512::DEGREE];
        coefficients[0] = FALCON_Q as u16;
        assert!(
            CanonicalPublicKeyBytes::<Falcon512>::parse(&encode_public::<Falcon512>(&coefficients))
                .is_err()
        );
        let mut coefficients = vec![0_u16; Falcon1024::DEGREE];
        coefficients[1_023] = FALCON_Q as u16;
        assert!(
            CanonicalPublicKeyBytes::<Falcon1024>::parse(&encode_public::<Falcon1024>(
                &coefficients
            ))
            .is_err()
        );
    }

    fn canonical_signature_decodes_signed_boundaries_for<P: FalconParams>() {
        let mut coefficients = vec![0_i16; P::DEGREE];
        coefficients[0] = -2_047;
        coefficients[1] = 2_047;
        coefficients[2] = -128;
        coefficients[P::DEGREE - 1] = 127;
        let canonical = CanonicalSignatureBytes::<P>::parse(&encode_signature::<P>(&coefficients))
            .expect("canonical signature");
        let decoded = canonical.decode();
        assert_eq!(&decoded.s2().as_ref()[..3], &[-2_047, 2_047, -128]);
        assert_eq!(decoded.s2().as_ref()[P::DEGREE - 1], 127);
        assert_eq!(canonical.as_bytes().as_ref().len(), P::SIGNATURE_BYTES);
    }

    fn signature_digits_cross_byte_boundaries_for<P: FalconParams>() {
        for index in 0..8 {
            for value in [-2_047, -128, -1, 0, 1, 127, 128, 2_047] {
                // Each preceding zero occupies nine bits, so these eight
                // positions exercise every starting offset within a byte.
                let mut coefficients = vec![0_i16; P::DEGREE];
                coefficients[index] = value;
                let encoded = encode_signature::<P>(&coefficients);
                let canonical = CanonicalSignatureBytes::<P>::parse(&encoded)
                    .expect("canonical unaligned coefficient");
                assert_eq!(canonical.decode().s2().as_ref(), coefficients.as_slice());
            }
        }
    }

    #[test]
    fn signature_digits_cross_byte_boundaries() {
        signature_digits_cross_byte_boundaries_for::<Falcon512>();
        signature_digits_cross_byte_boundaries_for::<Falcon1024>();
    }

    #[test]
    fn canonical_signature_decodes_signed_boundaries() {
        canonical_signature_decodes_signed_boundaries_for::<Falcon512>();
        canonical_signature_decodes_signed_boundaries_for::<Falcon1024>();
    }

    fn signature_rejects_negative_zero_and_nonzero_padding_for<P: FalconParams>() {
        let coefficients = vec![0_i16; P::DEGREE];
        let mut negative_zero = encode_signature::<P>(&coefficients);
        negative_zero[SIGNATURE_PAYLOAD_OFFSET] |= 0x80;
        assert!(CanonicalSignatureBytes::<P>::parse(&negative_zero).is_err());

        let mut bad_padding = encode_signature::<P>(&coefficients);
        *bad_padding.last_mut().expect("signature byte") = 1;
        assert!(CanonicalSignatureBytes::<P>::parse(&bad_padding).is_err());
    }

    #[test]
    fn signature_rejects_negative_zero_and_nonzero_padding() {
        signature_rejects_negative_zero_and_nonzero_padding_for::<Falcon512>();
        signature_rejects_negative_zero_and_nonzero_padding_for::<Falcon1024>();
    }

    /// Bit-at-a-time decoder written directly from the Falcon compressed
    /// format, used as the reference for the accumulator decoder.
    fn reference_decode<P: FalconParams>(payload: &[u8]) -> Option<Vec<i16>> {
        let bits: Vec<bool> = payload
            .iter()
            .flat_map(|byte| (0..8).rev().map(move |shift| (byte >> shift) & 1 != 0))
            .collect();
        let mut position = 0;
        let mut coefficients = Vec::with_capacity(P::DEGREE);
        for _ in 0..P::DEGREE {
            let header = bits.get(position..position + 8)?;
            position += 8;
            let negative = header[0];
            let mut magnitude = header[1..]
                .iter()
                .fold(0_u16, |value, bit| (value << 1) | u16::from(*bit));
            while !*bits.get(position)? {
                position += 1;
                magnitude += 128;
                if magnitude > FALCON_COMPRESSED_COEFFICIENT_LIMIT {
                    return None;
                }
            }
            position += 1;
            if negative && magnitude == 0 {
                return None;
            }
            let value = magnitude.cast_signed();
            coefficients.push(if negative { -value } else { value });
        }
        bits[position..]
            .iter()
            .all(|bit| !bit)
            .then_some(coefficients)
    }

    fn signature_decoder_matches_reference_for<P: FalconParams>() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut payloads = Vec::new();
        for round in 0..64 {
            // Spread magnitudes so a few long unary runs appear without
            // overflowing the padded body.
            let coefficients: Vec<i16> = (0..P::DEGREE)
                .map(|_| {
                    let draw = next();
                    let magnitude = if draw % 64 == 0 {
                        draw >> 8 & 2_047
                    } else {
                        draw >> 8 & 255
                    };
                    let magnitude = magnitude as i16;
                    if draw & 1 << 7 != 0 {
                        -magnitude
                    } else {
                        magnitude
                    }
                })
                .collect();
            let body_bits: usize = coefficients
                .iter()
                .map(|value| 9 + usize::from(value.unsigned_abs() >> 7))
                .sum();
            if body_bits > P::PADDED_BODY_BITS {
                continue;
            }
            let encoded = encode_signature::<P>(&coefficients);
            let payload = encoded[SIGNATURE_PAYLOAD_OFFSET..].to_vec();
            payloads.push(payload.clone());
            for _ in 0..8 {
                let mut flipped = payload.clone();
                let bit = (next() as usize) % (flipped.len() * 8);
                flipped[bit / 8] ^= 0x80 >> (bit % 8);
                payloads.push(flipped);
            }
            let mut noise = payload;
            for byte in &mut noise {
                *byte = next() as u8;
            }
            if round % 4 == 0 {
                let half = noise.len() / 2;
                noise[half..].fill(0);
            }
            payloads.push(noise);
        }
        assert!(payloads.len() > 64 * 5);
        let mut accepted = 0;
        for payload in &payloads {
            let expected = reference_decode::<P>(payload);
            let decoded = decode_signature_coefficients::<P>(payload).ok();
            assert_eq!(
                decoded.as_ref().map(|values| values.as_ref().to_vec()),
                expected
            );
            accepted += usize::from(expected.is_some());
        }
        assert!(accepted >= payloads.len() / 9, "too few accepted cases");
    }

    #[test]
    fn signature_decoder_matches_reference() {
        signature_decoder_matches_reference_for::<Falcon512>();
        signature_decoder_matches_reference_for::<Falcon1024>();
    }

    #[test]
    fn falcon_1024_budget_admits_more_unary_bits_than_falcon_512() {
        // 696 high-part bits fit a Falcon-1024 body; 393 already overflow a
        // Falcon-512 body. Each coefficient of magnitude 128 costs one unary bit.
        let mut wide = vec![0_i16; Falcon1024::DEGREE];
        for cell in wide.iter_mut().take(696) {
            *cell = 128;
        }
        assert!(
            CanonicalSignatureBytes::<Falcon1024>::parse(&encode_signature::<Falcon1024>(&wide))
                .is_ok()
        );
        let mut narrow = vec![0_i16; Falcon512::DEGREE];
        for cell in narrow.iter_mut().take(392) {
            *cell = 128;
        }
        assert!(
            CanonicalSignatureBytes::<Falcon512>::parse(&encode_signature::<Falcon512>(&narrow))
                .is_ok()
        );
        narrow[392] = 128;
        let overflow = std::panic::catch_unwind(|| encode_signature::<Falcon512>(&narrow));
        assert!(
            overflow.is_err(),
            "393 unary bits exceed the Falcon-512 body"
        );
    }

    #[test]
    fn parameter_sets_reject_each_other_s_bytes() {
        let key_1024 = encode_public::<Falcon1024>(&vec![0_u16; Falcon1024::DEGREE]);
        let key_512 = encode_public::<Falcon512>(&vec![0_u16; Falcon512::DEGREE]);
        assert!(CanonicalPublicKeyBytes::<Falcon512>::parse(&key_1024).is_err());
        assert!(CanonicalPublicKeyBytes::<Falcon1024>::parse(&key_512).is_err());

        let mut wrong_header = key_1024.clone();
        wrong_header[0] = Falcon512::PUBLIC_KEY_HEADER;
        assert!(CanonicalPublicKeyBytes::<Falcon1024>::parse(&wrong_header).is_err());

        let signature_1024 = encode_signature::<Falcon1024>(&vec![0_i16; Falcon1024::DEGREE]);
        assert!(CanonicalSignatureBytes::<Falcon512>::parse(&signature_1024).is_err());
        let mut wrong_header = signature_1024;
        wrong_header[0] = Falcon512::PADDED_SIGNATURE_HEADER;
        assert!(CanonicalSignatureBytes::<Falcon1024>::parse(&wrong_header).is_err());
    }
}
