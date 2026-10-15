//! Honest FN-DSA fixtures backed by real key generation and signing.
//!
//! Every record here is produced by the `fn-dsa` crate: parameter-set key
//! generation, then FN-DSA signing in the fixed v0 mode: raw messages
//! (`HASH_ID_RAW`) with the empty context inside the standardized framing.
//! Signing yields canonical fixture inputs. Native verification occurs when
//! deriving direct or prepared arithmetic witnesses. This module is the only
//! fixture source permitted to score the honest-FN-DSA-vectors obligation
//! used by the constructed fixtures.
//!
//! Randomness is a seeded SHAKE256 stream so fixtures are reproducible.
//! That makes the keys benchmark material, not security-grade key material;
//! they must never leave benchmark and test code.

use crate::native::{record::DirectRecord, witness::verify_signature_native};

use aerie_falcon_native::{
    source::{CanonicalPublicKeyBytes, CanonicalSignatureBytes},
    Falcon1024, Falcon512,
};
use fn_dsa::{
    sign_key_size, signature_size, vrfy_key_size, CryptoRng, DomainContext, KeyPairGenerator,
    KeyPairGenerator1024, KeyPairGenerator512, RngCore, SigningKey, SigningKey1024, SigningKey512,
    HASH_ID_RAW,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use shake::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256, Shake256Reader,
};

use crate::{
    AggregationRecord, AggregationStatement, AggregationWitness, DirectRow, ProductionError,
};

/// The protocol fixes the empty FN-DSA context; honest fixtures sign
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

/// Reference signer and deterministic fixture domains for a parameter set.
pub trait HonestParams: aerie_falcon_native::FalconParams {
    type Generator: KeyPairGenerator;
    type Signer: SigningKey;
    const SEED_MASK: u64;
    const MESSAGE_PREFIX: &'static str;
}

impl HonestParams for Falcon512 {
    type Generator = KeyPairGenerator512;
    type Signer = SigningKey512;
    const SEED_MASK: u64 = 0;
    const MESSAGE_PREFIX: &'static str = "Aerie-Orthus honest fixture record";
}

impl HonestParams for Falcon1024 {
    type Generator = KeyPairGenerator1024;
    type Signer = SigningKey1024;
    const SEED_MASK: u64 = 0x0400_0000_0000_0000;
    const MESSAGE_PREFIX: &'static str = "Aerie honest Falcon-1024 fixture record";
}

/// Nonce selection for deterministic reference signatures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SaltMode {
    PerRecord,
    Common,
}

fn honest_record<P: HonestParams>(
    seed: u64,
    record_index: u64,
    message: Vec<u8>,
    supplied_salt: Option<[u8; 40]>,
) -> Result<DirectRecord<P>, ProductionError> {
    let mut rng = FixtureRng::new(seed ^ P::SEED_MASK, record_index);
    let mut sign_key = vec![0_u8; sign_key_size(P::LOGN)];
    let mut vrfy_key = vec![0_u8; vrfy_key_size(P::LOGN)];
    P::Generator::default().keygen(P::LOGN, &mut rng, &mut sign_key, &mut vrfy_key);
    let mut signer = P::Signer::decode(&sign_key).ok_or_else(|| {
        ProductionError::MalformedWire("fn-dsa signing key failed to decode".to_owned())
    })?;
    if let Some(salt) = supplied_salt {
        rng.supply_salt(salt);
    }
    let mut signature = vec![0_u8; signature_size(P::LOGN)];
    signer.sign(
        &mut rng,
        &DomainContext(EMPTY_CONTEXT),
        &HASH_ID_RAW,
        &message,
        &mut signature,
    );
    let public_key = CanonicalPublicKeyBytes::<P>::parse(&vrfy_key)?;
    let signature = CanonicalSignatureBytes::<P>::parse(&signature)?;
    Ok(DirectRecord::<P>::new(public_key, signature, message))
}

fn honest_records<P: HonestParams>(
    batch_size: usize,
    seed: u64,
    salt_mode: SaltMode,
) -> Result<Vec<DirectRecord<P>>, ProductionError> {
    let supplied_salt = match salt_mode {
        SaltMode::PerRecord => None,
        SaltMode::Common => Some(common_fixture_salt(seed, P::DEGREE as u32)),
    };
    let build = |index: usize| {
        honest_record::<P>(
            seed,
            index as u64,
            format!("{} {index}", P::MESSAGE_PREFIX).into_bytes(),
            supplied_salt,
        )
    };
    #[cfg(feature = "parallel")]
    let records = (0..batch_size).into_par_iter().map(build).collect();
    #[cfg(not(feature = "parallel"))]
    let records = (0..batch_size).map(build).collect();
    records
}

/// Distinct reference keypairs and messages with deterministic nonce selection.
/// Record indices fix the output order under parallel construction.
#[tracing::instrument(
    skip_all,
    fields(batch_size = batch_size),
    name = "statement.honest.keygen_sign"
)]
pub fn honest_aggregation<P: HonestParams>(
    batch_size: usize,
    seed: u64,
    salt_mode: SaltMode,
) -> Result<(AggregationStatement<P>, AggregationWitness<P>), ProductionError> {
    let records = honest_records::<P>(batch_size, seed, salt_mode)?;
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

fn common_fixture_salt(seed: u64, degree: u32) -> [u8; 40] {
    let mut hasher = Shake256::default();
    hasher.update(b"aerie-orthus/honest-common-salt/v1");
    hasher.update(&seed.to_le_bytes());
    hasher.update(&degree.to_le_bytes());
    let mut salt = [0_u8; 40];
    hasher.finalize_xof().read(&mut salt);
    salt
}

/// One reference-signed, natively verified direct row.
pub fn honest_direct_row<P: HonestParams + crate::DirectParams>(
    seed: u64,
    record_index: u64,
) -> Result<DirectRow<P>, ProductionError> {
    let record = honest_record::<P>(
        seed,
        record_index,
        format!("{} {record_index}", P::MESSAGE_PREFIX).into_bytes(),
        None,
    )?;
    let verified = verify_signature_native(record.as_record_ref())?;
    Ok(DirectRow::from_verified(&verified)?)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fmt::Write;

    use super::*;

    const FIXTURE_512_PER_RECORD: &str = "21170d44e8e8c000911c954dbbafa079";
    const FIXTURE_512_COMMON: &str = "d1367aab3f6f655404d082fdd44e0c75";
    const FIXTURE_1024_PER_RECORD: &str = "6d09356bb6ac1d0b31cf0e0e45bb51e3";
    const FIXTURE_1024_COMMON: &str = "cef102362b1568497a6b49009f5ecd48";

    fn honest_fixture_digest<P: HonestParams>(salt_mode: SaltMode) -> String {
        let (statement, witness) =
            honest_aggregation::<P>(2, 0x5eed_1024_0512_u64, salt_mode).unwrap();
        let signatures = witness.signatures();
        assert_eq!(signatures.len(), 2);
        match salt_mode {
            SaltMode::Common => assert_eq!(signatures[0].salt(), signatures[1].salt()),
            SaltMode::PerRecord => assert_ne!(signatures[0].salt(), signatures[1].salt()),
        }
        let statement_bytes = statement.canonical_encoding().unwrap();
        let witness_bytes = witness.canonical_encoding().unwrap();
        let mut hasher = Shake256::default();
        hasher.update(b"aerie/honest-fixture-pin/v1");
        hasher.update(&(statement_bytes.len() as u64).to_le_bytes());
        hasher.update(&statement_bytes);
        hasher.update(&(witness_bytes.len() as u64).to_le_bytes());
        hasher.update(&witness_bytes);
        let mut digest = [0_u8; 16];
        hasher.finalize_xof().read(&mut digest);
        let mut hex = String::with_capacity(32);
        for byte in digest {
            write!(hex, "{byte:02x}").unwrap();
        }
        hex
    }

    #[test]
    fn honest_fixture_bytes_512_per_record() {
        assert_eq!(
            honest_fixture_digest::<Falcon512>(SaltMode::PerRecord),
            FIXTURE_512_PER_RECORD
        );
    }

    #[test]
    fn honest_fixture_bytes_512_common() {
        assert_eq!(
            honest_fixture_digest::<Falcon512>(SaltMode::Common),
            FIXTURE_512_COMMON
        );
    }

    #[test]
    fn honest_fixture_bytes_1024_per_record() {
        assert_eq!(
            honest_fixture_digest::<Falcon1024>(SaltMode::PerRecord),
            FIXTURE_1024_PER_RECORD
        );
    }

    #[test]
    fn honest_fixture_bytes_1024_common() {
        assert_eq!(
            honest_fixture_digest::<Falcon1024>(SaltMode::Common),
            FIXTURE_1024_COMMON
        );
    }

    #[test]
    fn honest_record_verifies_natively_and_is_nontrivial() {
        let record =
            honest_record::<Falcon512>(7, 0, b"honest test message".to_vec(), None).unwrap();
        let verified = verify_signature_native(record.as_record_ref()).unwrap();
        assert!(verified.s2().iter().any(|&coefficient| coefficient != 0));
        assert!(verified.s1().iter().any(|&coefficient| coefficient != 0));
    }

    #[test]
    fn caller_salt_is_signed_and_natively_verified_for_both_degrees() {
        let salt = [0x5a; 40];
        let record_512 =
            honest_record::<Falcon512>(17, 0, b"common salt 512".to_vec(), Some(salt)).unwrap();
        assert_eq!(record_512.signature().salt(), &salt);
        verify_signature_native(record_512.as_record_ref()).unwrap();

        let record_1024 =
            honest_record::<Falcon1024>(17, 0, b"common salt 1024".to_vec(), Some(salt)).unwrap();
        assert_eq!(record_1024.signature().salt(), &salt);
        verify_signature_native(record_1024.as_record_ref()).unwrap();
    }

    #[test]
    fn honest_rows_are_reproducible_and_distinct_across_indices() {
        let first = honest_direct_row::<Falcon512>(7, 0).unwrap();
        let again = honest_direct_row::<Falcon512>(7, 0).unwrap();
        let other = honest_direct_row::<Falcon512>(7, 1).unwrap();
        assert_eq!(first.s2, again.s2);
        assert_ne!(first.s2, other.s2);
        assert_ne!(first.h, other.h);
    }

    #[test]
    fn honest_falcon_1024_aggregation_verifies_and_packs_at_degree_1024() {
        let (statement, witness) =
            honest_aggregation::<Falcon1024>(3, 0x1024_f00d, SaltMode::PerRecord).unwrap();
        assert_eq!(statement.len(), 3);
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
            crate::FalconPreparedWitness::from_aggregate::<Falcon1024>(&statement, &witness)
                .unwrap();
        let batch = prepared.witness();
        assert_eq!(batch.degree(), 1024);
        assert_eq!((batch.live_rows, batch.padded_rows), (3, 4));
        batch.validate_semantics().unwrap();
        assert_eq!(batch.h.len(), 4 * 1024);
    }

    #[test]
    fn honest_batch_has_unique_keys_and_signatures_and_verifies_natively() {
        let records = honest_records::<Falcon512>(64, 0x051a_97e5, SaltMode::PerRecord).unwrap();
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
            verify_signature_native(record.as_record_ref())
                .expect("honest signature must verify natively");
        }
    }
}
