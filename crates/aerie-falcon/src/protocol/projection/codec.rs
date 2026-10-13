//! Wire codec of the projection outputs (`specs/falcon.md` Section 10.4).
//!
//! The section is `k_y || k_z || bits`: two parameter bytes, then one
//! little-endian bit stream of 187 Rice codes (the 156 norm outputs first,
//! then the 31 residual outputs), zero-padded to a byte boundary.
//!
//! - A norm output `y_j` is coded as `ctr_p(y_j)`.
//! - A residual output `z_j` is coded as `ctr_p(z_j) / q`, so a `z` that
//!   fails the residual predicate has no encoding. The quotient is at most
//!   `floor(((p - 1)/2) / q)` in magnitude, so `q` times it is again a
//!   centered representative.
//! - A signed value `v` is zigzag-mapped to `m = 2v` (`v >= 0`) or
//!   `m = -2v - 1` (`v < 0`), then Rice-coded with the leg's parameter `k`:
//!   `floor(m / 2^k)` one bits, a zero bit, and the low `k` bits of `m`,
//!   least significant first.
//! - Each leg's `k` is the smallest `k <= 63` minimizing that leg's bit
//!   length. The decoder recomputes it and rejects any other `k`, so every
//!   output vector has exactly one encoding.
//!
//! After canonical decoding and both projection checks, the transcript
//! absorbs this complete Rice section as one byte-string atom, including
//! both parameter bytes and final zero padding ([`super::absorb_outputs`]).
//! Every section the decoder accepts is at most
//! [`MAX_ENCODED_BYTES`] long, and one whose outputs pass the predicates is
//! at most 893 bytes (Falcon-512) or 904 bytes (Falcon-1024); honest
//! sections are about 610 to 770 bytes.

use crate::substrate::BaseField as F;
use jolt_field::Ring as _;

use super::predicates::{centered, FIELD_MODULUS};
use super::profile::{FALCON_Q, NORM_TERMINAL_ROWS, RESIDUAL_WIDTH};
use super::ProjectionOutputs;

/// `(p - 1) / 2`, the largest centered magnitude.
const HALF_MODULUS: u64 = (FIELD_MODULUS - 1) / 2;
/// Largest `|ctr_p(z_j) / q|` whose multiple of `q` is centered.
const RESIDUAL_QUOTIENT_BOUND: u64 = HALF_MODULUS / FALCON_Q;
/// Largest Rice parameter. At `k = 63` every zigzag value below `2^64` has
/// a quotient of at most one, so the optimum never exceeds it.
pub const MAX_RICE_PARAMETER: u8 = 63;
/// Upper bound on the section length over every decodable input: 156 norm
/// codes of at most 65 bits and 31 residual codes of at most 52 bits, plus
/// the two parameter bytes. Below the 1,496 bytes of 187 `LE64` limbs.
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub const MAX_ENCODED_BYTES: usize = 1_471;

/// Rice section failures, reported as malformed wire.
pub type CodecError = &'static str;

fn zigzag(value: i128) -> u64 {
    let magnitude = value.unsigned_abs() as u64;
    if value >= 0 {
        magnitude << 1
    } else {
        (magnitude << 1) - 1
    }
}

/// Inverse of [`zigzag`] on `m <= 2 * bound`, as a centered field element
/// scaled by `scale`; `None` above the bound.
fn unzigzag(code: u64, bound: u64, scale: u64) -> Option<F> {
    if code > 2 * bound {
        return None;
    }
    let magnitude = if code & 1 == 0 {
        code >> 1
    } else {
        (code >> 1) + 1
    };
    let scaled = magnitude * scale;
    Some(if code & 1 == 0 {
        F::from_u64(scaled)
    } else {
        F::from_u64(FIELD_MODULUS - scaled)
    })
}

/// Bit length of one leg under parameter `k`.
fn leg_bits(codes: &[u64], k: u8) -> u128 {
    codes
        .iter()
        .map(|&code| u128::from(k) + 1 + u128::from(code >> k))
        .sum()
}

/// The smallest `k` minimizing [`leg_bits`].
fn optimal_parameter(codes: &[u64]) -> u8 {
    (0..=MAX_RICE_PARAMETER)
        .min_by_key(|&k| leg_bits(codes, k))
        .expect("nonempty parameter range")
}

struct BitWriter {
    bytes: Vec<u8>,
    bits: usize,
}

impl BitWriter {
    fn push(&mut self, bit: bool) {
        if self.bits.is_multiple_of(8) {
            self.bytes.push(0);
        }
        if bit {
            *self.bytes.last_mut().expect("pushed byte") |= 1 << (self.bits % 8);
        }
        self.bits += 1;
    }

    fn push_code(&mut self, code: u64, k: u8) {
        for _ in 0..code >> k {
            self.push(true);
        }
        self.push(false);
        for bit in 0..k {
            self.push((code >> bit) & 1 == 1);
        }
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bits: usize,
}

impl BitReader<'_> {
    fn pull(&mut self) -> Result<bool, CodecError> {
        let byte = self
            .bytes
            .get(self.bits / 8)
            .ok_or("truncated projection outputs")?;
        let bit = (byte >> (self.bits % 8)) & 1 == 1;
        self.bits += 1;
        Ok(bit)
    }

    fn pull_code(&mut self, k: u8) -> Result<u64, CodecError> {
        let limit = u64::MAX >> k;
        let mut quotient: u64 = 0;
        while self.pull()? {
            if quotient == limit {
                return Err("projection output code overflows");
            }
            quotient += 1;
        }
        let mut remainder: u64 = 0;
        for bit in 0..k {
            remainder |= u64::from(self.pull()?) << bit;
        }
        Ok((quotient << k) | remainder)
    }
}

impl ProjectionOutputs {
    /// Zigzag codes of `ctr_p(y_j)` and `ctr_p(z_j) / q`; fails when a
    /// residual output is not a multiple of `q`.
    fn codes(&self) -> Result<([u64; NORM_TERMINAL_ROWS], [u64; RESIDUAL_WIDTH]), CodecError> {
        let y = self.y.map(|value| zigzag(centered(value)));
        let mut z = [0; RESIDUAL_WIDTH];
        for (code, &value) in z.iter_mut().zip(&self.z) {
            let value = centered(value);
            if value % i128::from(FALCON_Q) != 0 {
                return Err("residual output is not a multiple of q");
            }
            *code = zigzag(value / i128::from(FALCON_Q));
        }
        Ok((y, z))
    }

    /// Append the Rice section. Only outputs whose `z` passes the residual
    /// predicate are encodable; the verifier rejects every other `z`.
    pub fn encode_wire(&self, buffer: &mut Vec<u8>) -> Result<(), CodecError> {
        let (y, z) = self.codes()?;
        let (k_y, k_z) = (optimal_parameter(&y), optimal_parameter(&z));
        buffer.extend_from_slice(&[k_y, k_z]);
        let mut writer = BitWriter {
            bytes: Vec::new(),
            bits: 0,
        };
        for code in y {
            writer.push_code(code, k_y);
        }
        for code in z {
            writer.push_code(code, k_z);
        }
        buffer.extend_from_slice(&writer.bytes);
        Ok(())
    }

    /// Length of [`Self::encode_wire`]'s section.
    #[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
    pub fn encoded_len(&self) -> Result<usize, CodecError> {
        let (y, z) = self.codes()?;
        let bits = leg_bits(&y, optimal_parameter(&y)) + leg_bits(&z, optimal_parameter(&z));
        Ok(2 + usize::try_from(bits.div_ceil(8)).expect("bounded section"))
    }

    /// Read one Rice section and advance `reader` past it. Rejects a
    /// parameter above 63 or other than the leg's optimum, a value outside
    /// its centered range, nonzero padding, and truncation.
    pub fn decode_wire(reader: &mut &[u8]) -> Result<Self, CodecError> {
        let bytes: &[u8] = reader;
        let [k_y, k_z, stream @ ..] = bytes else {
            return Err("truncated projection outputs");
        };
        let (k_y, k_z) = (*k_y, *k_z);
        if k_y > MAX_RICE_PARAMETER || k_z > MAX_RICE_PARAMETER {
            return Err("projection output parameter out of range");
        }
        let mut bits = BitReader {
            bytes: stream,
            bits: 0,
        };
        let mut y_codes = [0; NORM_TERMINAL_ROWS];
        for code in &mut y_codes {
            *code = bits.pull_code(k_y)?;
        }
        let mut z_codes = [0; RESIDUAL_WIDTH];
        for code in &mut z_codes {
            *code = bits.pull_code(k_z)?;
        }
        let used = bits.bits.div_ceil(8);
        while !bits.bits.is_multiple_of(8) {
            if bits.pull()? {
                return Err("nonzero projection output padding");
            }
        }
        if k_y != optimal_parameter(&y_codes) || k_z != optimal_parameter(&z_codes) {
            return Err("noncanonical projection output parameter");
        }
        let mut y = [F::from_u64(0); NORM_TERMINAL_ROWS];
        for (value, &code) in y.iter_mut().zip(&y_codes) {
            *value = unzigzag(code, HALF_MODULUS, 1).ok_or("projection output out of range")?;
        }
        let mut z = [F::from_u64(0); RESIDUAL_WIDTH];
        for (value, &code) in z.iter_mut().zip(&z_codes) {
            *value = unzigzag(code, RESIDUAL_QUOTIENT_BOUND, FALCON_Q)
                .ok_or("projection output out of range")?;
        }
        *reader = &stream[used..];
        Ok(Self { y, z })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{check_outputs, norm_cap, NORM_CAP_FALCON_1024, NORM_CAP_FALCON_512};
    use super::*;
    use crate::protocol::FalconDegree;

    fn signed(value: i128) -> F {
        if value >= 0 {
            F::from_u64(value as u64)
        } else {
            F::from_u64(FIELD_MODULUS - value.unsigned_abs() as u64)
        }
    }

    /// A deterministic spread of magnitudes and signs around `2^bits`.
    fn outputs(y_bits: u32, u_bits: u32) -> ProjectionOutputs {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = |bits: u32| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let magnitude = i128::from((state >> 11) & ((1 << bits) - 1));
            if state >> 63 == 1 {
                -magnitude
            } else {
                magnitude
            }
        };
        ProjectionOutputs {
            y: core::array::from_fn(|_| signed(next(y_bits))),
            z: core::array::from_fn(|_| signed(next(u_bits) * i128::from(FALCON_Q))),
        }
    }

    fn encode(outputs: &ProjectionOutputs) -> Vec<u8> {
        let mut bytes = Vec::new();
        outputs.encode_wire(&mut bytes).expect("encodable");
        assert_eq!(bytes.len(), outputs.encoded_len().expect("encodable"));
        bytes
    }

    fn roundtrip(outputs: &ProjectionOutputs) -> Vec<u8> {
        let bytes = encode(outputs);
        let mut framed = bytes.clone();
        framed.extend_from_slice(b"tail");
        let mut reader = framed.as_slice();
        assert_eq!(
            &ProjectionOutputs::decode_wire(&mut reader).expect("decodes"),
            outputs
        );
        assert_eq!(reader, b"tail");
        bytes
    }

    /// Re-encode with forced parameters and the given code streams.
    fn forged(k_y: u8, k_z: u8, y: &[u64], z: &[u64]) -> Vec<u8> {
        let mut writer = BitWriter {
            bytes: Vec::new(),
            bits: 0,
        };
        for &code in y {
            writer.push_code(code, k_y);
        }
        for &code in z {
            writer.push_code(code, k_z);
        }
        [vec![k_y, k_z], writer.bytes].concat()
    }

    fn decode(bytes: &[u8]) -> Result<ProjectionOutputs, CodecError> {
        ProjectionOutputs::decode_wire(&mut &bytes[..])
    }

    #[test]
    fn roundtrips_honest_scale_and_extreme_outputs() {
        for (y_bits, u_bits) in [(1, 1), (23, 30), (29, 36), (37, 49)] {
            let bytes = roundtrip(&outputs(y_bits, u_bits));
            assert!(bytes.len() < MAX_ENCODED_BYTES);
        }
        let zero = F::from_u64(0);
        roundtrip(&ProjectionOutputs {
            y: [zero; NORM_TERMINAL_ROWS],
            z: [zero; RESIDUAL_WIDTH],
        });
        let bound = i128::from(RESIDUAL_QUOTIENT_BOUND) * i128::from(FALCON_Q);
        let half = i128::from(HALF_MODULUS);
        for sign in [1, -1] {
            let extreme = ProjectionOutputs {
                y: [signed(sign * half); NORM_TERMINAL_ROWS],
                z: [signed(sign * bound); RESIDUAL_WIDTH],
            };
            let bytes = roundtrip(&extreme);
            assert!(bytes.len() <= MAX_ENCODED_BYTES);
        }
        let alternating = ProjectionOutputs {
            y: core::array::from_fn(|j| signed(if j % 2 == 0 { half } else { -half })),
            z: core::array::from_fn(|j| signed(if j % 2 == 0 { bound } else { -bound })),
        };
        assert_eq!(roundtrip(&alternating).len(), MAX_ENCODED_BYTES);
    }

    /// The accepted-set bound of Section 10.4: `sum ctr_p(y_j)^2 <= C_S`
    /// gives `sum m_j <= 2 isqrt(156 C_S)`, so the optimal `k_y` costs at
    /// most `min_k 156 (k + 1) + floor(2 isqrt(156 C_S) / 2^k)` bits; every
    /// residual code costs at most 52 bits at `k_z = 50`.
    #[test]
    fn accepted_outputs_meet_the_section_bound() {
        for (degree, cap, bound) in [
            (FalconDegree::Falcon512, NORM_CAP_FALCON_512, 893),
            (FalconDegree::Falcon1024, NORM_CAP_FALCON_1024, 904),
        ] {
            assert_eq!(norm_cap(degree), cap);
            let rows = NORM_TERMINAL_ROWS as u128;
            let sum = 2 * (rows * cap).isqrt();
            let y_bits = (0..=MAX_RICE_PARAMETER)
                .map(|k| rows * (u128::from(k) + 1) + (sum >> k))
                .min()
                .expect("nonempty");
            let z_bits = RESIDUAL_WIDTH as u128 * 52;
            assert_eq!(2 + (y_bits + z_bits).div_ceil(8), bound);
            let flat = (cap / rows).isqrt() as i128;
            let quotient = i128::from(RESIDUAL_QUOTIENT_BOUND) * i128::from(FALCON_Q);
            let worst = ProjectionOutputs {
                y: core::array::from_fn(|j| signed(if j % 2 == 0 { flat } else { -flat })),
                z: core::array::from_fn(|j| signed(if j % 2 == 0 { quotient } else { -quotient })),
            };
            check_outputs(degree, &worst).expect("accepted");
            assert!(roundtrip(&worst).len() <= bound as usize);
        }
    }

    #[test]
    fn rejects_noncanonical_sections() {
        let honest = outputs(29, 36);
        let bytes = encode(&honest);
        let (y, z) = honest.codes().expect("encodable");
        let (k_y, k_z) = (bytes[0], bytes[1]);
        assert_eq!(decode(&forged(k_y, k_z, &y, &z)), Ok(honest));
        // Any other parameter is noncanonical, even when it decodes.
        for (a, b) in [
            (k_y + 1, k_z),
            (k_y - 1, k_z),
            (k_y, k_z + 1),
            (k_y, k_z - 1),
        ] {
            assert_eq!(
                decode(&forged(a, b, &y, &z)),
                Err("noncanonical projection output parameter")
            );
        }
        let mut wide = bytes.clone();
        wide[0] = MAX_RICE_PARAMETER + 1;
        assert!(decode(&wide).is_err());
        // Truncation anywhere, including an empty section.
        for length in 0..bytes.len() {
            assert!(decode(&bytes[..length]).is_err(), "prefix {length}");
        }
        // A set padding bit, on the samples whose stream is unaligned.
        let mut unaligned = 0;
        for (y_bits, u_bits) in [(23, 30), (26, 33), (29, 36), (31, 38), (33, 41)] {
            let sample = outputs(y_bits, u_bits);
            let (y, z) = sample.codes().expect("encodable");
            let bits = leg_bits(&y, optimal_parameter(&y)) + leg_bits(&z, optimal_parameter(&z));
            if !bits.is_multiple_of(8) {
                unaligned += 1;
                let mut padded = encode(&sample);
                *padded.last_mut().expect("nonempty") |= 1 << 7;
                assert_eq!(decode(&padded), Err("nonzero projection output padding"));
            }
        }
        assert!(unaligned > 0);
    }

    #[test]
    fn rejects_out_of_range_values() {
        let mut y = [0; NORM_TERMINAL_ROWS];
        let z = [0; RESIDUAL_WIDTH];
        // `m = p` would be the centered value `(p + 1) / 2`.
        y[0] = FIELD_MODULUS;
        let k_y = optimal_parameter(&y);
        assert_eq!(
            decode(&forged(k_y, 0, &y, &z)),
            Err("projection output out of range")
        );
        let y = [0; NORM_TERMINAL_ROWS];
        let mut z = [0; RESIDUAL_WIDTH];
        z[3] = 2 * RESIDUAL_QUOTIENT_BOUND + 1;
        let k_z = optimal_parameter(&z);
        assert_eq!(
            decode(&forged(0, k_z, &y, &z)),
            Err("projection output out of range")
        );
        // A unary run past `u64::MAX >> k`.
        let mut run = vec![MAX_RICE_PARAMETER, 0];
        run.extend_from_slice(&[0xff; 4]);
        assert_eq!(decode(&run), Err("projection output code overflows"));
    }

    #[test]
    fn residual_off_the_lattice_is_unencodable() {
        let mut outputs = outputs(29, 36);
        outputs.z[4] += F::from_u64(1);
        assert!(outputs.encode_wire(&mut Vec::new()).is_err());
        // The `ctr_p` boundary: a canonical multiple of `q` whose centered
        // value is not.
        let canonical = (HALF_MODULUS / FALCON_Q + 1) * FALCON_Q;
        outputs.z[4] = F::from_u64(canonical);
        assert!(outputs.encode_wire(&mut Vec::new()).is_err());
    }
}
