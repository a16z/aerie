//! Reproducible benchmark fixtures.
//!
//! The constructed canonical records exercise byte decoding and `HashToPoint`,
//! but their public key is deliberately weak and not produced by FN-DSA key
//! generation. They must never be labelled honest signatures.

use aerie_falcon_native::{
    preprocess::HashToPointTarget,
    record::DirectRecord,
    source::{
        CanonicalPublicKeyBytes, CanonicalSignatureBytes, FALCON_512_PUBLIC_KEY_BYTES,
        FALCON_512_SALT_BYTES, FALCON_512_SIGNATURE_BYTES,
    },
    Falcon512,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{ProductionError, ProductionStatement, FALCON_DEGREE};

const CONSTRUCTED_H: i16 = 111;

pub fn constructed_statement(batch_size: usize) -> Result<ProductionStatement, ProductionError> {
    let mut public_coefficients = [0_u16; FALCON_DEGREE];
    public_coefficients[0] = CONSTRUCTED_H as u16;
    let public_key =
        CanonicalPublicKeyBytes::<Falcon512>::parse(&encode_public_key(&public_coefficients))?;
    let records = constructed_records(batch_size, &public_key)?;
    ProductionStatement::new(records)
}

#[cfg(feature = "parallel")]
fn constructed_records(
    batch_size: usize,
    public_key: &CanonicalPublicKeyBytes<Falcon512>,
) -> Result<Vec<DirectRecord<Falcon512>>, aerie_falcon_native::AerieError> {
    (0..batch_size)
        .into_par_iter()
        .map(|index| constructed_record(index, public_key))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn constructed_records(
    batch_size: usize,
    public_key: &CanonicalPublicKeyBytes<Falcon512>,
) -> Result<Vec<DirectRecord<Falcon512>>, aerie_falcon_native::AerieError> {
    (0..batch_size)
        .map(|index| constructed_record(index, public_key))
        .collect()
}

fn constructed_record(
    record_index: usize,
    public_key: &CanonicalPublicKeyBytes<Falcon512>,
) -> Result<DirectRecord<Falcon512>, aerie_falcon_native::AerieError> {
    let salt = fixture_salt(record_index);
    let message = format!("Aerie-Orthus canonical benchmark record {record_index}").into_bytes();
    let zero_signature = CanonicalSignatureBytes::<Falcon512>::parse(&encode_signature(
        &salt,
        &[0_i16; FALCON_DEGREE],
    ))?;
    let hash_record =
        DirectRecord::<Falcon512>::new(public_key.clone(), zero_signature, message.clone());
    let target = HashToPointTarget::<Falcon512>::derive(&hash_record);

    let mut s2 = [0_i16; FALCON_DEGREE];
    for (output, &target_coefficient) in s2.iter_mut().zip(target.t_centered()) {
        let target_coefficient = i32::from(target_coefficient);
        let rounded = if target_coefficient >= 0 {
            (target_coefficient + i32::from(CONSTRUCTED_H) / 2) / i32::from(CONSTRUCTED_H)
        } else {
            (target_coefficient - i32::from(CONSTRUCTED_H) / 2) / i32::from(CONSTRUCTED_H)
        };
        *output = i16::try_from(rounded).expect("constructed quotient fits in i16");
    }
    let signature = CanonicalSignatureBytes::<Falcon512>::parse(&encode_signature(&salt, &s2))?;
    Ok(DirectRecord::<Falcon512>::new(
        public_key.clone(),
        signature,
        message,
    ))
}

fn fixture_salt(record_index: usize) -> [u8; FALCON_512_SALT_BYTES] {
    let mut salt = [0_u8; FALCON_512_SALT_BYTES];
    let mut state = (record_index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for chunk in salt.chunks_mut(8) {
        state ^= state >> 30;
        state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94d0_49bb_1331_11eb);
        state ^= state >> 31;
        chunk.copy_from_slice(&state.to_le_bytes()[..chunk.len()]);
    }
    salt
}

fn encode_public_key(coefficients: &[u16; FALCON_DEGREE]) -> [u8; FALCON_512_PUBLIC_KEY_BYTES] {
    let mut encoded = [0_u8; FALCON_512_PUBLIC_KEY_BYTES];
    encoded[0] = 0x09;
    let mut writer = BitWriter::new(&mut encoded[1..]);
    for &coefficient in coefficients {
        writer.write_bits(coefficient, 14);
    }
    encoded
}

fn encode_signature(
    salt: &[u8; FALCON_512_SALT_BYTES],
    coefficients: &[i16; FALCON_DEGREE],
) -> [u8; FALCON_512_SIGNATURE_BYTES] {
    let mut encoded = [0_u8; FALCON_512_SIGNATURE_BYTES];
    encoded[0] = 0x39;
    encoded[1..=FALCON_512_SALT_BYTES].copy_from_slice(salt);
    let mut writer = BitWriter::new(&mut encoded[1 + FALCON_512_SALT_BYTES..]);
    for &coefficient in coefficients {
        let magnitude = coefficient.unsigned_abs();
        writer.write_bit(coefficient < 0);
        writer.write_bits(magnitude & 127, 7);
        for _ in 0..magnitude >> 7 {
            writer.write_bit(false);
        }
        writer.write_bit(true);
    }
    encoded
}

struct BitWriter<'a> {
    bytes: &'a mut [u8],
    position: usize,
}

impl<'a> BitWriter<'a> {
    const fn new(bytes: &'a mut [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn write_bit(&mut self, value: bool) {
        if value {
            self.bytes[self.position / 8] |= 1 << (7 - self.position % 8);
        }
        self.position += 1;
    }

    fn write_bits(&mut self, value: u16, count: usize) {
        for shift in (0..count).rev() {
            self.write_bit(((value >> shift) & 1) != 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use aerie_falcon_native::witness::verify_signature_native;

    use super::*;

    #[test]
    fn constructed_records_are_canonical_and_distinct() {
        let statement = constructed_statement(2).unwrap();
        let first = verify_signature_native(&statement.records()[0]).unwrap();
        let second = verify_signature_native(&statement.records()[1]).unwrap();
        assert_ne!(
            statement.records()[0].signature(),
            statement.records()[1].signature()
        );
        assert_ne!(first.t_centered(), second.t_centered());
    }
}
