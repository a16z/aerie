//! Reproducible benchmark fixtures.
//!
//! The constructed canonical records exercise byte decoding and `HashToPoint`,
//! but their public key is deliberately weak and not produced by FN-DSA key
//! generation. They must never be labelled honest signatures.

use crate::native::{record::DirectRecord, target::HashToPointTarget};

use aerie_falcon_native::{
    params::{filled, FALCON_SALT_BYTES},
    source::{CanonicalPublicKeyBytes, CanonicalSignatureBytes},
    FalconParams,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{AggregationRecord, AggregationStatement, AggregationWitness, ProductionError};

const CONSTRUCTED_H: i16 = 111;

pub fn constructed_aggregation<P: FalconParams>(
    batch_size: usize,
) -> Result<(AggregationStatement<P>, AggregationWitness<P>), ProductionError> {
    let mut public_coefficients = filled::<P, u16>(0);
    public_coefficients.as_mut()[0] = CONSTRUCTED_H as u16;
    let public_key =
        CanonicalPublicKeyBytes::<P>::parse(&encode_public_key::<P>(public_coefficients.as_ref()))?;
    let records = constructed_records(batch_size, &public_key)?;
    let statement_records = records
        .iter()
        .map(|record| {
            AggregationRecord::new(record.public_key().clone(), record.message().to_vec())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let signatures = records
        .into_iter()
        .map(|record| record.signature().clone())
        .collect();
    Ok((
        AggregationStatement::new(statement_records)?,
        AggregationWitness::new(signatures)?,
    ))
}

#[cfg(feature = "parallel")]
fn constructed_records<P: FalconParams>(
    batch_size: usize,
    public_key: &CanonicalPublicKeyBytes<P>,
) -> Result<Vec<DirectRecord<P>>, aerie_falcon_native::AerieError> {
    (0..batch_size)
        .into_par_iter()
        .map(|index| constructed_record(index, public_key))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn constructed_records<P: FalconParams>(
    batch_size: usize,
    public_key: &CanonicalPublicKeyBytes<P>,
) -> Result<Vec<DirectRecord<P>>, aerie_falcon_native::AerieError> {
    (0..batch_size)
        .map(|index| constructed_record(index, public_key))
        .collect()
}

fn constructed_record<P: FalconParams>(
    record_index: usize,
    public_key: &CanonicalPublicKeyBytes<P>,
) -> Result<DirectRecord<P>, aerie_falcon_native::AerieError> {
    let salt = fixture_salt(record_index);
    let message = format!("Aerie-Orthus canonical benchmark record {record_index}").into_bytes();
    let zero_signature = CanonicalSignatureBytes::<P>::parse(&encode_signature::<P>(
        &salt,
        filled::<P, i16>(0).as_ref(),
    ))?;
    let hash_record = DirectRecord::<P>::new(public_key.clone(), zero_signature, message.clone());
    let target = HashToPointTarget::<P>::derive(&hash_record);

    let mut s2 = filled::<P, i16>(0);
    for (output, &target_coefficient) in s2.as_mut().iter_mut().zip(target.t_centered().as_ref()) {
        let target_coefficient = i32::from(target_coefficient);
        let rounded = if target_coefficient >= 0 {
            (target_coefficient + i32::from(CONSTRUCTED_H) / 2) / i32::from(CONSTRUCTED_H)
        } else {
            (target_coefficient - i32::from(CONSTRUCTED_H) / 2) / i32::from(CONSTRUCTED_H)
        };
        *output = i16::try_from(rounded).expect("constructed quotient fits in i16");
    }
    let signature =
        CanonicalSignatureBytes::<P>::parse(&encode_signature::<P>(&salt, s2.as_ref()))?;
    Ok(DirectRecord::<P>::new(
        public_key.clone(),
        signature,
        message,
    ))
}

fn fixture_salt(record_index: usize) -> [u8; FALCON_SALT_BYTES] {
    let mut salt = [0_u8; FALCON_SALT_BYTES];
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

fn encode_public_key<P: FalconParams>(coefficients: &[u16]) -> Vec<u8> {
    let mut encoded = vec![0_u8; P::PUBLIC_KEY_BYTES];
    encoded[0] = P::PUBLIC_KEY_HEADER;
    let mut writer = BitWriter::new(&mut encoded[1..]);
    for &coefficient in coefficients {
        writer.write_bits(coefficient, 14);
    }
    encoded
}

fn encode_signature<P: FalconParams>(
    salt: &[u8; FALCON_SALT_BYTES],
    coefficients: &[i16],
) -> Vec<u8> {
    let mut encoded = vec![0_u8; P::SIGNATURE_BYTES];
    encoded[0] = P::PADDED_SIGNATURE_HEADER;
    encoded[1..=FALCON_SALT_BYTES].copy_from_slice(salt);
    let mut writer = BitWriter::new(&mut encoded[1 + FALCON_SALT_BYTES..]);
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
    use crate::native::witness::verify_signature_native;

    use super::*;

    #[test]
    fn constructed_records_are_canonical_and_distinct() {
        check_records::<aerie_falcon_native::Falcon512>();
        check_records::<aerie_falcon_native::Falcon1024>();
    }

    fn check_records<P: FalconParams>() {
        let (statement, witness) = constructed_aggregation::<P>(2).unwrap();
        let records = statement
            .public_keys()
            .iter()
            .zip(statement.messages())
            .zip(witness.signatures())
            .map(|((public_key, message), signature)| {
                DirectRecord::new(public_key.clone(), signature.clone(), message.clone())
            })
            .collect::<Vec<_>>();
        let first = verify_signature_native(records[0].as_record_ref()).unwrap();
        let second = verify_signature_native(records[1].as_record_ref()).unwrap();
        assert_ne!(records[0].signature(), records[1].signature());
        assert_ne!(first.t_centered(), second.t_centered());
    }
}
