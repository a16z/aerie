//! Honest FN-DSA fixtures backed by real key generation and signing.
//!
//! Every record here is produced by the `fn-dsa` crate: Falcon-512 key
//! generation, then FN-DSA signing in the fixed v0 mode: raw messages
//! (`HASH_ID_RAW`) with the empty context inside the standardized framing.
//! Records are re-verified natively before any witness is derived, so this
//! module is the only fixture source permitted to score the
//! honest-FN-DSA-vectors obligation used by the constructed fixtures.
//!
//! Randomness is a seeded SHAKE256 stream so fixtures are reproducible.
//! That makes the keys benchmark material, not security-grade key material;
//! they must never leave benchmark and test code.

use aerie_falcon_native::{
    record::DirectRecord,
    source::{CanonicalPublicKeyBytes, CanonicalSignatureBytes},
    witness::verify_signature_native,
    Falcon1024, Falcon512,
};
use fn_dsa::{
    sign_key_size, signature_size, vrfy_key_size, CryptoRng, DomainContext, KeyPairGenerator,
    KeyPairGenerator1024, KeyPairGenerator512, RngCore, SigningKey, SigningKey1024, SigningKey512,
    FN_DSA_LOGN_1024, FN_DSA_LOGN_512, HASH_ID_RAW,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use shake::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256, Shake256Reader,
};

use crate::{
    AggregationRecord, AggregationStatement, AggregationWitness, DirectRow, ProductionError,
    ProductionStatement,
};

/// The v0 protocol fixes the empty FN-DSA context; honest fixtures sign
/// under it so the standardized framing matches the aggregate verifier's.
const EMPTY_CONTEXT: &[u8] = b"";

/// Deterministic SHAKE256-backed randomness for reproducible fixtures.
///
/// Implements the `rand_core` traits `fn-dsa` requires. The stream is
/// keyed by a fixture seed and a record index, so records are independent
/// and parallel construction is stable.
struct FixtureRng {
    stream: Shake256Reader,
    supplied_salt: Option<[u8; 40]>,
}

impl FixtureRng {
    fn new(seed: u64, record_index: u64) -> Self {
        let mut hasher = Shake256::default();
        hasher.update(b"aerie-orthus/honest-fixture-rng/v1");
        hasher.update(&seed.to_le_bytes());
        hasher.update(&record_index.to_le_bytes());
        Self {
            stream: hasher.finalize_xof(),
            supplied_salt: None,
        }
    }

    fn supply_salt(&mut self, salt: [u8; 40]) {
        self.supplied_salt = Some(salt);
    }
}

impl RngCore for FixtureRng {
    fn next_u32(&mut self) -> u32 {
        let mut word = [0_u8; 4];
        self.stream.read(&mut word);
        u32::from_le_bytes(word)
    }

    fn next_u64(&mut self) -> u64 {
        let mut word = [0_u8; 8];
        self.stream.read(&mut word);
        u64::from_le_bytes(word)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        if let (Some(salt), 40) = (self.supplied_salt, dest.len()) {
            // fn-dsa requests the 40-byte nonce before each signing attempt,
            // then draws a separate 56-byte sampler seed. Keep retries on
            // the caller's nonce while leaving sampler randomness intact.
            dest.copy_from_slice(&salt);
        } else {
            self.stream.read(dest);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), fn_dsa::RngError> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for FixtureRng {}

/// One honest Falcon-512 record: fresh keypair, real FN-DSA signature.
pub fn honest_record(
    seed: u64,
    record_index: u64,
    message: Vec<u8>,
) -> Result<DirectRecord<Falcon512>, ProductionError> {
    honest_record_inner(seed, record_index, message, None)
}

/// One honest Falcon-512 record signed with the caller-supplied FN-DSA
/// salt. The signature still comes from the reference signer.
pub fn honest_record_with_salt(
    seed: u64,
    record_index: u64,
    message: Vec<u8>,
    salt: [u8; 40],
) -> Result<DirectRecord<Falcon512>, ProductionError> {
    honest_record_inner(seed, record_index, message, Some(salt))
}

fn honest_record_inner(
    seed: u64,
    record_index: u64,
    message: Vec<u8>,
    supplied_salt: Option<[u8; 40]>,
) -> Result<DirectRecord<Falcon512>, ProductionError> {
    let mut rng = FixtureRng::new(seed, record_index);
    let mut sign_key = vec![0_u8; sign_key_size(FN_DSA_LOGN_512)];
    let mut vrfy_key = vec![0_u8; vrfy_key_size(FN_DSA_LOGN_512)];
    KeyPairGenerator512::default().keygen(FN_DSA_LOGN_512, &mut rng, &mut sign_key, &mut vrfy_key);
    let mut signer = SigningKey512::decode(&sign_key).ok_or_else(|| {
        ProductionError::MalformedWire("fn-dsa signing key failed to decode".to_owned())
    })?;
    if let Some(salt) = supplied_salt {
        rng.supply_salt(salt);
    }
    let mut signature = vec![0_u8; signature_size(FN_DSA_LOGN_512)];
    signer.sign(
        &mut rng,
        &DomainContext(EMPTY_CONTEXT),
        &HASH_ID_RAW,
        &message,
        &mut signature,
    );
    let public_key = CanonicalPublicKeyBytes::<Falcon512>::parse(&vrfy_key)?;
    let signature = CanonicalSignatureBytes::<Falcon512>::parse(&signature)?;
    Ok(DirectRecord::<Falcon512>::new(
        public_key, signature, message,
    ))
}

/// One honest Falcon-1024 record: fresh FN-DSA-1024 keypair, real signature
/// in the same raw-message, empty-context mode. The randomness stream is
/// domain-separated from the Falcon-512 fixtures by the parameter set.
pub fn honest_record_1024(
    seed: u64,
    record_index: u64,
    message: Vec<u8>,
) -> Result<DirectRecord<Falcon1024>, ProductionError> {
    honest_record_1024_inner(seed, record_index, message, None)
}

/// One honest Falcon-1024 record signed with the caller-supplied FN-DSA
/// salt. The signature still comes from the reference signer.
pub fn honest_record_1024_with_salt(
    seed: u64,
    record_index: u64,
    message: Vec<u8>,
    salt: [u8; 40],
) -> Result<DirectRecord<Falcon1024>, ProductionError> {
    honest_record_1024_inner(seed, record_index, message, Some(salt))
}

fn honest_record_1024_inner(
    seed: u64,
    record_index: u64,
    message: Vec<u8>,
    supplied_salt: Option<[u8; 40]>,
) -> Result<DirectRecord<Falcon1024>, ProductionError> {
    let mut rng = FixtureRng::new(seed ^ 0x0400_0000_0000_0000, record_index);
    let mut sign_key = vec![0_u8; sign_key_size(FN_DSA_LOGN_1024)];
    let mut vrfy_key = vec![0_u8; vrfy_key_size(FN_DSA_LOGN_1024)];
    KeyPairGenerator1024::default().keygen(
        FN_DSA_LOGN_1024,
        &mut rng,
        &mut sign_key,
        &mut vrfy_key,
    );
    let mut signer = SigningKey1024::decode(&sign_key).ok_or_else(|| {
        ProductionError::MalformedWire("fn-dsa signing key failed to decode".to_owned())
    })?;
    if let Some(salt) = supplied_salt {
        rng.supply_salt(salt);
    }
    let mut signature = vec![0_u8; signature_size(FN_DSA_LOGN_1024)];
    signer.sign(
        &mut rng,
        &DomainContext(EMPTY_CONTEXT),
        &HASH_ID_RAW,
        &message,
        &mut signature,
    );
    let public_key = CanonicalPublicKeyBytes::<Falcon1024>::parse(&vrfy_key)?;
    let signature = CanonicalSignatureBytes::<Falcon1024>::parse(&signature)?;
    Ok(DirectRecord::<Falcon1024>::new(
        public_key, signature, message,
    ))
}

/// An honest Falcon-1024 aggregation statement and its positional witness:
/// distinct keypairs, salts and messages per record.
pub fn honest_aggregation_1024(
    batch_size: usize,
    seed: u64,
) -> Result<
    (
        AggregationStatement<Falcon1024>,
        AggregationWitness<Falcon1024>,
    ),
    ProductionError,
> {
    let build = |index: usize| {
        honest_record_1024(
            seed,
            index as u64,
            format!("Aerie honest Falcon-1024 fixture record {index}").into_bytes(),
        )
    };
    #[cfg(feature = "parallel")]
    let records = (0..batch_size)
        .into_par_iter()
        .map(build)
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(not(feature = "parallel"))]
    let records = (0..batch_size).map(build).collect::<Result<Vec<_>, _>>()?;
    let statement_records = records
        .iter()
        .map(|record| {
            AggregationRecord::new(record.public_key().clone(), record.message().to_vec())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let signatures = records
        .iter()
        .map(|record| record.signature().clone())
        .collect();
    Ok((
        AggregationStatement::new(statement_records)?,
        AggregationWitness::new(signatures)?,
    ))
}

/// Honest Falcon-512 aggregation statement whose signatures share one
/// seed-derived salt supplied to the FN-DSA signer.
pub fn honest_aggregation_common_salt(
    batch_size: usize,
    seed: u64,
) -> Result<
    (
        AggregationStatement<Falcon512>,
        AggregationWitness<Falcon512>,
    ),
    ProductionError,
> {
    let salt = common_fixture_salt(seed, 512);
    let build = |index: usize| {
        honest_record_with_salt(
            seed,
            index as u64,
            format!("Aerie-Orthus honest fixture record {index}").into_bytes(),
            salt,
        )
    };
    #[cfg(feature = "parallel")]
    let records = (0..batch_size)
        .into_par_iter()
        .map(build)
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(not(feature = "parallel"))]
    let records = (0..batch_size).map(build).collect::<Result<Vec<_>, _>>()?;
    let statement_records = records
        .iter()
        .map(|record| {
            AggregationRecord::new(record.public_key().clone(), record.message().to_vec())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let signatures = records
        .iter()
        .map(|record| record.signature().clone())
        .collect();
    Ok((
        AggregationStatement::new(statement_records)?,
        AggregationWitness::new(signatures)?,
    ))
}

/// Honest Falcon-1024 aggregation statement whose signatures share one
/// seed-derived salt supplied to the FN-DSA signer.
pub fn honest_aggregation_1024_common_salt(
    batch_size: usize,
    seed: u64,
) -> Result<
    (
        AggregationStatement<Falcon1024>,
        AggregationWitness<Falcon1024>,
    ),
    ProductionError,
> {
    let salt = common_fixture_salt(seed, 1024);
    let build = |index: usize| {
        honest_record_1024_with_salt(
            seed,
            index as u64,
            format!("Aerie honest Falcon-1024 fixture record {index}").into_bytes(),
            salt,
        )
    };
    #[cfg(feature = "parallel")]
    let records = (0..batch_size)
        .into_par_iter()
        .map(build)
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(not(feature = "parallel"))]
    let records = (0..batch_size).map(build).collect::<Result<Vec<_>, _>>()?;
    let statement_records = records
        .iter()
        .map(|record| {
            AggregationRecord::new(record.public_key().clone(), record.message().to_vec())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let signatures = records
        .iter()
        .map(|record| record.signature().clone())
        .collect();
    Ok((
        AggregationStatement::new(statement_records)?,
        AggregationWitness::new(signatures)?,
    ))
}

fn common_fixture_salt(seed: u64, degree: u32) -> [u8; 40] {
    let mut hasher = Shake256::default();
    hasher.update(b"aerie-orthus/honest-common-salt/v1");
    hasher.update(&seed.to_le_bytes());
    hasher.update(&degree.to_le_bytes());
    let mut salt = [0_u8; 40];
    hasher.finalize_xof().read(&mut salt);
    salt
}

/// One honest, natively verified direct row.
pub fn honest_direct_row(
    seed: u64,
    record_index: u64,
) -> Result<DirectRow<Falcon512>, ProductionError> {
    let record = honest_record(
        seed,
        record_index,
        format!("Aerie-Orthus honest fixture record {record_index}").into_bytes(),
    )?;
    let verified = verify_signature_native(&record)?;
    Ok(DirectRow::from_verified(&verified)?)
}

/// A production statement of distinct honest records.
///
/// Each record has its own keypair, salt, and message. Key generation
/// dominates construction time; records are built in parallel when the
/// `parallel` feature is enabled.
#[tracing::instrument(
    skip_all,
    fields(batch_size = batch_size),
    name = "statement.honest.keygen_sign"
)]
pub fn honest_statement(
    batch_size: usize,
    seed: u64,
) -> Result<ProductionStatement, ProductionError> {
    let records = honest_records(batch_size, seed)?;
    ProductionStatement::new(records)
}

#[cfg(feature = "parallel")]
fn honest_records(
    batch_size: usize,
    seed: u64,
) -> Result<Vec<DirectRecord<Falcon512>>, ProductionError> {
    (0..batch_size)
        .into_par_iter()
        .map(|index| {
            honest_record(
                seed,
                index as u64,
                format!("Aerie-Orthus honest fixture record {index}").into_bytes(),
            )
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn honest_records(
    batch_size: usize,
    seed: u64,
) -> Result<Vec<DirectRecord<Falcon512>>, ProductionError> {
    (0..batch_size)
        .map(|index| {
            honest_record(
                seed,
                index as u64,
                format!("Aerie-Orthus honest fixture record {index}").into_bytes(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn honest_record_verifies_natively_and_is_nontrivial() {
        let record = honest_record(7, 0, b"honest test message".to_vec()).unwrap();
        let verified = verify_signature_native(&record).unwrap();
        assert!(verified.s2().iter().any(|&coefficient| coefficient != 0));
        assert!(verified.s1().iter().any(|&coefficient| coefficient != 0));
    }

    #[test]
    fn caller_salt_is_signed_and_natively_verified_for_both_degrees() {
        let salt = [0x5a; 40];
        let record_512 = honest_record_with_salt(17, 0, b"common salt 512".to_vec(), salt).unwrap();
        assert_eq!(record_512.signature().salt(), &salt);
        verify_signature_native(&record_512).unwrap();

        let record_1024 =
            honest_record_1024_with_salt(17, 0, b"common salt 1024".to_vec(), salt).unwrap();
        assert_eq!(record_1024.signature().salt(), &salt);
        verify_signature_native(&record_1024).unwrap();
    }

    #[test]
    fn honest_rows_are_reproducible_and_distinct_across_indices() {
        let first = honest_direct_row(7, 0).unwrap();
        let again = honest_direct_row(7, 0).unwrap();
        let other = honest_direct_row(7, 1).unwrap();
        assert_eq!(first.s2, again.s2);
        assert_ne!(first.s2, other.s2);
        assert_ne!(first.h, other.h);
    }

    #[test]
    fn honest_falcon_1024_aggregation_verifies_and_packs_at_degree_1024() {
        let (statement, witness) = honest_aggregation_1024(3, 0x1024_f00d).unwrap();
        assert_eq!(statement.records().len(), 3);
        // Canonical encodings round-trip at the Falcon-1024 key and
        // signature lengths.
        let encoded = statement.canonical_encoding().unwrap();
        assert_eq!(
            AggregationStatement::<Falcon1024>::parse_canonical(&encoded).unwrap(),
            statement
        );
        let encoded_witness = witness.canonical_encoding().unwrap();
        assert_eq!(
            encoded_witness.len(),
            6 + 3 * <Falcon1024 as aerie_falcon_native::FalconParams>::SIGNATURE_BYTES
        );
        assert_eq!(
            AggregationWitness::<Falcon1024>::parse_canonical(&encoded_witness).unwrap(),
            witness
        );
        // The Falcon-512 parser rejects the Falcon-1024 encoding.
        assert!(crate::AggregationStatement::<Falcon512>::parse_canonical(&encoded).is_err());
        assert!(crate::AggregationWitness::<Falcon512>::parse_canonical(&encoded_witness).is_err());
        // The prepared witness is a degree-1024 compact batch that validates.
        let prepared =
            crate::FalconPreparedWitness::from_aggregate_of::<Falcon1024>(&statement, &witness)
                .unwrap();
        let batch = prepared.witness();
        assert_eq!(batch.degree(), 1024);
        assert_eq!((batch.live_rows, batch.padded_rows), (3, 4));
        batch.validate_semantics().unwrap();
        assert_eq!(statement.public_batch().h.len(), 4 * 1024);
    }

    #[test]
    fn honest_batch_has_unique_keys_and_signatures_and_verifies_natively() {
        let records = honest_records(64, 0x051a_97e5).unwrap();
        let public_keys = records
            .iter()
            .map(|record| record.public_key().as_bytes().to_vec())
            .collect::<HashSet<_>>();
        let signatures = records
            .iter()
            .map(|record| record.signature().as_bytes().to_vec())
            .collect::<HashSet<_>>();
        let salts = records
            .iter()
            .map(|record| *record.signature().salt())
            .collect::<HashSet<_>>();
        let messages = records
            .iter()
            .map(|record| record.message().to_vec())
            .collect::<HashSet<_>>();
        assert_eq!(public_keys.len(), records.len());
        assert_eq!(signatures.len(), records.len());
        assert_eq!(salts.len(), records.len());
        assert_eq!(messages.len(), records.len());
        for record in records {
            verify_signature_native(&record).expect("honest signature must verify natively");
        }
    }
}

/// Differential harness against the reference `fn-dsa` verifier, on both
/// parameter sets.
///
/// Every case asserts two things: our end-to-end verdict (canonical byte
/// parsing, record construction, native verification) EQUALS the reference
/// crate's verdict under the fixed v0 mode (raw messages, empty context
/// inside the standardized framing), and that the joint verdict is the
/// expected one. A divergence in either direction is a finding: accepting
/// what the reference rejects would break soundness of the aggregate
/// statement, rejecting what it accepts would silently shrink coverage.
#[cfg(test)]
mod differential {
    use super::*;
    use aerie_falcon_native::params::FALCON_SALT_BYTES;
    use aerie_falcon_native::{Falcon512, FalconParams};
    use fn_dsa::{VerifyingKey, VerifyingKey1024, VerifyingKey512};

    const RECORDS: u64 = 8;

    /// The reference `fn-dsa` key generation, signing, and verification of
    /// one parameter set.
    trait Reference: FalconParams {
        /// Differential seed; Falcon-512 keeps its original value.
        const SEED: u64;
        fn keygen(rng: &mut FixtureRng, sign_key: &mut [u8], public_key: &mut [u8]);
        fn sign(sign_key: &[u8], rng: &mut FixtureRng, message: &[u8], signature: &mut [u8]);
        fn verify(public_key: &[u8], signature: &[u8], context: &[u8], message: &[u8]) -> bool;
    }

    impl Reference for Falcon512 {
        const SEED: u64 = 0xd1ff_0001;
        fn keygen(rng: &mut FixtureRng, sign_key: &mut [u8], public_key: &mut [u8]) {
            KeyPairGenerator512::default().keygen(FN_DSA_LOGN_512, rng, sign_key, public_key);
        }
        fn sign(sign_key: &[u8], rng: &mut FixtureRng, message: &[u8], signature: &mut [u8]) {
            let mut signer = SigningKey512::decode(sign_key).expect("signing key decodes");
            signer.sign(
                rng,
                &DomainContext(EMPTY_CONTEXT),
                &HASH_ID_RAW,
                message,
                signature,
            );
        }
        fn verify(public_key: &[u8], signature: &[u8], context: &[u8], message: &[u8]) -> bool {
            VerifyingKey512::decode(public_key).is_some_and(|key| {
                key.verify(signature, &DomainContext(context), &HASH_ID_RAW, message)
            })
        }
    }

    impl Reference for Falcon1024 {
        const SEED: u64 = 0xd1ff_1024;
        fn keygen(rng: &mut FixtureRng, sign_key: &mut [u8], public_key: &mut [u8]) {
            KeyPairGenerator1024::default().keygen(FN_DSA_LOGN_1024, rng, sign_key, public_key);
        }
        fn sign(sign_key: &[u8], rng: &mut FixtureRng, message: &[u8], signature: &mut [u8]) {
            let mut signer = SigningKey1024::decode(sign_key).expect("signing key decodes");
            signer.sign(
                rng,
                &DomainContext(EMPTY_CONTEXT),
                &HASH_ID_RAW,
                message,
                signature,
            );
        }
        fn verify(public_key: &[u8], signature: &[u8], context: &[u8], message: &[u8]) -> bool {
            VerifyingKey1024::decode(public_key).is_some_and(|key| {
                key.verify(signature, &DomainContext(context), &HASH_ID_RAW, message)
            })
        }
    }

    struct RawRecord {
        public_key: Vec<u8>,
        signature: Vec<u8>,
        message: Vec<u8>,
    }

    /// Same construction flow as [`honest_record`] and
    /// [`honest_record_1024`], keeping the raw bytes.
    fn raw_record<P: Reference>(index: u64) -> RawRecord {
        let mut rng = FixtureRng::new(P::SEED, index);
        let mut sign_key = vec![0_u8; sign_key_size(P::LOGN)];
        let mut public_key = vec![0_u8; vrfy_key_size(P::LOGN)];
        P::keygen(&mut rng, &mut sign_key, &mut public_key);
        let message = format!("Aerie-Orthus differential record {index}").into_bytes();
        let mut signature = vec![0_u8; signature_size(P::LOGN)];
        P::sign(&sign_key, &mut rng, &message, &mut signature);
        RawRecord {
            public_key,
            signature,
            message,
        }
    }

    fn native_accepts<P: Reference>(public_key: &[u8], signature: &[u8], message: &[u8]) -> bool {
        let Ok(public_key) = CanonicalPublicKeyBytes::<P>::parse(public_key) else {
            return false;
        };
        let Ok(signature) = CanonicalSignatureBytes::<P>::parse(signature) else {
            return false;
        };
        let record = DirectRecord::<P>::new(public_key, signature, message.to_vec());
        verify_signature_native(&record).is_ok()
    }

    fn assert_agree<P: Reference>(
        case: &str,
        index: u64,
        expected: bool,
        public_key: &[u8],
        signature: &[u8],
        message: &[u8],
    ) {
        let degree = P::DEGREE;
        let native = native_accepts::<P>(public_key, signature, message);
        let reference = P::verify(public_key, signature, EMPTY_CONTEXT, message);
        assert_eq!(
            native, reference,
            "Falcon-{degree} verifier divergence on case `{case}` (record {index}): \
             native={native} reference={reference}"
        );
        assert_eq!(
            native, expected,
            "unexpected Falcon-{degree} joint verdict on case `{case}` (record {index})"
        );
    }

    fn native_and_reference_agree<P: Reference>() {
        let records = (0..RECORDS).map(raw_record::<P>).collect::<Vec<_>>();
        for (index, record) in records.iter().enumerate() {
            let index = index as u64;
            let pk = &record.public_key;
            let sig = &record.signature;
            let msg = &record.message;
            assert_eq!(pk.len(), P::PUBLIC_KEY_BYTES);
            assert_eq!(sig.len(), P::SIGNATURE_BYTES);

            assert_agree::<P>("honest", index, true, pk, sig, msg);

            let mut tampered = msg.clone();
            tampered[0] ^= 1;
            assert_agree::<P>("flipped-message-bit", index, false, pk, sig, &tampered);

            let mut tampered = sig.clone();
            tampered[0] ^= 0xff;
            assert_agree::<P>("flipped-signature-header", index, false, pk, &tampered, msg);

            let mut tampered = sig.clone();
            tampered[1] ^= 1;
            assert_agree::<P>("flipped-salt-byte", index, false, pk, &tampered, msg);

            let mut tampered = sig.clone();
            let body_middle = 1 + FALCON_SALT_BYTES + 200;
            tampered[body_middle] ^= 1;
            assert_agree::<P>("flipped-body-byte", index, false, pk, &tampered, msg);

            let mut tampered = pk.clone();
            let last = tampered.len() - 1;
            tampered[last] ^= 1;
            assert_agree::<P>("flipped-public-key-byte", index, false, &tampered, sig, msg);

            // The fixed v0 framing corresponds to the empty FN-DSA context:
            // the same bytes must fail reference verification under any
            // nonempty context.
            assert!(
                !P::verify(pk, sig, b"Aerie-Orthus differential wrong context", msg),
                "reference verifier accepted a nonempty context (record {index})"
            );

            if *sig.last().expect("nonempty signature") == 0 {
                let mut tampered = sig.clone();
                let last = tampered.len() - 1;
                tampered[last] = 1;
                assert_agree::<P>("nonzero-trailing-padding", index, false, pk, &tampered, msg);
            }
        }

        for pair in records.windows(2) {
            assert_agree::<P>(
                "cross-key",
                0,
                false,
                &pair[1].public_key,
                &pair[0].signature,
                &pair[0].message,
            );
        }
    }

    #[test]
    fn native_and_reference_verifiers_agree() {
        native_and_reference_agree::<Falcon512>();
    }

    #[test]
    fn falcon_1024_native_and_reference_verifiers_agree() {
        native_and_reference_agree::<Falcon1024>();
    }

    /// Bytes of the other parameter set, alone or mixed with this set's
    /// bytes, fail both this set's native and reference verifiers.
    #[test]
    fn cross_parameter_set_bytes_are_rejected_by_both_verifiers() {
        let small = raw_record::<Falcon512>(0);
        let large = raw_record::<Falcon1024>(0);
        for (public_key, signature, message) in [
            (&large.public_key, &large.signature, &large.message),
            (&small.public_key, &large.signature, &large.message),
            (&large.public_key, &small.signature, &small.message),
        ] {
            assert_agree::<Falcon512>("cross-set", 0, false, public_key, signature, message);
        }
        for (public_key, signature, message) in [
            (&small.public_key, &small.signature, &small.message),
            (&large.public_key, &small.signature, &small.message),
            (&small.public_key, &large.signature, &large.message),
        ] {
            assert_agree::<Falcon1024>("cross-set", 0, false, public_key, signature, message);
        }
    }
}
