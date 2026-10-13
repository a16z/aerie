//! Differential tests against the `fn-dsa` reference implementation for both
//! standardized parameter sets.
//!
//! Keys and signatures come from real FN-DSA key generation and signing in the
//! fixed raw-message, empty-context mode. Randomness is a seeded SHAKE256
//! stream, so the vectors are reproducible but are not security-grade keys.

use aerie_falcon_native::{
    encodability::FourCellEncodabilityWitness,
    record::DirectRecord,
    source::{CanonicalPublicKeyBytes, CanonicalSignatureBytes},
    witness::{verify_signature_native, verify_signatures_native},
    Falcon1024, Falcon512, FalconParams,
};
use fn_dsa::{
    sign_key_size, signature_size, vrfy_key_size, CryptoRng, DomainContext, KeyPairGenerator,
    KeyPairGenerator1024, KeyPairGenerator512, RngCore, SigningKey, SigningKey1024, SigningKey512,
    VerifyingKey, VerifyingKey1024, VerifyingKey512, HASH_ID_RAW,
};
use shake::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256, Shake256Reader,
};

const EMPTY_CONTEXT: DomainContext = DomainContext(b"");

struct FixtureRng(Shake256Reader);

impl FixtureRng {
    fn new(label: &str, index: u64) -> Self {
        let mut hasher = Shake256::default();
        hasher.update(b"aerie/falcon/fn-dsa-differential/v1");
        hasher.update(label.as_bytes());
        hasher.update(&index.to_le_bytes());
        Self(hasher.finalize_xof())
    }
}

impl RngCore for FixtureRng {
    fn next_u32(&mut self) -> u32 {
        let mut word = [0_u8; 4];
        self.0.read(&mut word);
        u32::from_le_bytes(word)
    }

    fn next_u64(&mut self) -> u64 {
        let mut word = [0_u8; 8];
        self.0.read(&mut word);
        u64::from_le_bytes(word)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.read(dest);
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), fn_dsa::RngError> {
        self.0.read(dest);
        Ok(())
    }
}

impl CryptoRng for FixtureRng {}

/// The reference key generator, signer, and verifier for one parameter set.
trait Reference: FalconParams {
    fn keygen(rng: &mut FixtureRng) -> (Vec<u8>, Vec<u8>);
    fn sign(sign_key: &[u8], rng: &mut FixtureRng, message: &[u8]) -> Vec<u8>;
    fn verify(vrfy_key: &[u8], signature: &[u8], message: &[u8]) -> bool;
}

macro_rules! reference_impl {
    ($params:ty, $kgen:ty, $sign:ty, $vrfy:ty) => {
        impl Reference for $params {
            fn keygen(rng: &mut FixtureRng) -> (Vec<u8>, Vec<u8>) {
                let mut sign_key = vec![0_u8; sign_key_size(Self::LOGN)];
                let mut vrfy_key = vec![0_u8; vrfy_key_size(Self::LOGN)];
                <$kgen>::default().keygen(Self::LOGN, rng, &mut sign_key, &mut vrfy_key);
                (sign_key, vrfy_key)
            }

            fn sign(sign_key: &[u8], rng: &mut FixtureRng, message: &[u8]) -> Vec<u8> {
                let mut signer = <$sign>::decode(sign_key).expect("reference signing key");
                let mut signature = vec![0_u8; signature_size(Self::LOGN)];
                signer.sign(rng, &EMPTY_CONTEXT, &HASH_ID_RAW, message, &mut signature);
                signature
            }

            fn verify(vrfy_key: &[u8], signature: &[u8], message: &[u8]) -> bool {
                <$vrfy>::decode(vrfy_key)
                    .expect("reference verifying key")
                    .verify(signature, &EMPTY_CONTEXT, &HASH_ID_RAW, message)
            }
        }
    };
}

reference_impl!(
    Falcon512,
    KeyPairGenerator512,
    SigningKey512,
    VerifyingKey512
);
reference_impl!(
    Falcon1024,
    KeyPairGenerator1024,
    SigningKey1024,
    VerifyingKey1024
);

struct Vector<P: FalconParams> {
    vrfy_key: Vec<u8>,
    signature: Vec<u8>,
    message: Vec<u8>,
    record: DirectRecord<P>,
}

fn vector<P: Reference>(index: u64) -> Vector<P> {
    let mut rng = FixtureRng::new(P::NAME, index);
    let (sign_key, vrfy_key) = P::keygen(&mut rng);
    let message = format!("{} differential record {index}", P::NAME).into_bytes();
    let signature = P::sign(&sign_key, &mut rng, &message);
    assert_eq!(vrfy_key.len(), P::PUBLIC_KEY_BYTES);
    assert_eq!(signature.len(), P::SIGNATURE_BYTES);
    assert_eq!(vrfy_key[0], P::PUBLIC_KEY_HEADER);
    assert_eq!(signature[0], P::PADDED_SIGNATURE_HEADER);
    let record = DirectRecord::new(
        CanonicalPublicKeyBytes::<P>::parse(&vrfy_key).expect("canonical reference key"),
        CanonicalSignatureBytes::<P>::parse(&signature).expect("canonical reference signature"),
        message.clone(),
    );
    Vector {
        vrfy_key,
        signature,
        message,
        record,
    }
}

const RECORDS: u64 = 3;

fn honest_vectors_verify_natively_for<P: Reference>() {
    let vectors = (0..RECORDS).map(vector::<P>).collect::<Vec<_>>();
    for vector in &vectors {
        assert!(P::verify(
            &vector.vrfy_key,
            &vector.signature,
            &vector.message
        ));
        let verified = verify_signature_native(&vector.record).expect("reference signature");
        assert!(verified.norm_squared() <= P::BETA_SQUARED);
        assert_eq!(verified.s2(), vector.record.signature().decode().s2());
        assert_eq!(
            verified.h_centered(),
            vector.record.public_key().decode().h_centered()
        );
        let encodability = verified
            .four_cell_encodability_witness()
            .expect("real signatures are encodable");
        for (index, &coefficient) in verified.s2().as_ref().iter().enumerate() {
            assert_eq!(encodability.coefficient(index), Some(coefficient));
        }
        let high_sum = verified
            .s2()
            .as_ref()
            .iter()
            .map(|coefficient| i16::try_from(coefficient.unsigned_abs() >> 7).unwrap())
            .sum::<i16>();
        assert_eq!(
            encodability.budget_slack(),
            i16::try_from(P::COMPRESSED_HIGH_BITS_BUDGET).unwrap() - high_sum
        );
    }
    let records = vectors
        .iter()
        .map(|vector| vector.record.clone())
        .collect::<Vec<_>>();
    let batch = verify_signatures_native(&records).expect("reference batch");
    for (batched, vector) in batch.iter().zip(&vectors) {
        assert_eq!(
            batched,
            &verify_signature_native(&vector.record).expect("reference signature")
        );
    }
}

#[test]
fn honest_falcon_512_vectors_verify_natively() {
    honest_vectors_verify_natively_for::<Falcon512>();
}

#[test]
fn honest_falcon_1024_vectors_verify_natively() {
    honest_vectors_verify_natively_for::<Falcon1024>();
}

#[test]
fn falcon_1024_signatures_exceed_the_falcon_512_norm_bound() {
    // Honest Falcon-1024 signatures have squared norm near
    // 2 * 1024 * sigma^2, about 58 million: valid for Falcon-1024, far above
    // the Falcon-512 bound. The Falcon-1024 verifier must use its own bound.
    for index in 0..RECORDS {
        let vector = vector::<Falcon1024>(index);
        let verified = verify_signature_native(&vector.record).expect("reference signature");
        assert!(verified.norm_squared() > Falcon512::BETA_SQUARED);
        assert!(verified.norm_squared() <= Falcon1024::BETA_SQUARED);
    }
}

fn tampering_rejects_for<P: Reference>() {
    let sample = vector::<P>(7);

    let mut other_message = sample.message.clone();
    other_message[0] ^= 0x01;
    let record = DirectRecord::<P>::new(
        sample.record.public_key().clone(),
        sample.record.signature().clone(),
        other_message.clone(),
    );
    assert!(!P::verify(
        &sample.vrfy_key,
        &sample.signature,
        &other_message
    ));
    assert!(verify_signature_native(&record).is_err());

    // Flip the sign bit of the first compressed coefficient: still canonical
    // bytes unless the coefficient is zero, but no longer a valid signature.
    let mut flipped = sample.signature.clone();
    flipped[41] ^= 0x80;
    assert!(!P::verify(&sample.vrfy_key, &flipped, &sample.message));
    // A zero first coefficient turns into negative zero, which the parser
    // rejects; otherwise the bytes stay canonical and native verification
    // must reject them.
    if let Ok(signature) = CanonicalSignatureBytes::<P>::parse(&flipped) {
        let record = DirectRecord::<P>::new(
            sample.record.public_key().clone(),
            signature,
            sample.message.clone(),
        );
        assert!(verify_signature_native(&record).is_err());
    }

    // A different key's salt-bound target does not verify either.
    let other = vector::<P>(8);
    let record = DirectRecord::<P>::new(
        other.record.public_key().clone(),
        sample.record.signature().clone(),
        sample.message.clone(),
    );
    assert!(verify_signature_native(&record).is_err());
}

#[test]
fn tampering_rejects_falcon_512() {
    tampering_rejects_for::<Falcon512>();
}

#[test]
fn tampering_rejects_falcon_1024() {
    tampering_rejects_for::<Falcon1024>();
}

#[test]
fn parameter_sets_reject_each_other_s_reference_bytes() {
    let wide = vector::<Falcon1024>(0);
    let narrow = vector::<Falcon512>(0);
    assert!(CanonicalPublicKeyBytes::<Falcon512>::parse(&wide.vrfy_key).is_err());
    assert!(CanonicalSignatureBytes::<Falcon512>::parse(&wide.signature).is_err());
    assert!(CanonicalPublicKeyBytes::<Falcon1024>::parse(&narrow.vrfy_key).is_err());
    assert!(CanonicalSignatureBytes::<Falcon1024>::parse(&narrow.signature).is_err());
    assert!(!Falcon512::verify(
        &narrow.vrfy_key,
        &wide.signature,
        &wide.message
    ));
}

#[test]
fn encodability_uses_the_parameter_set_budget() {
    let wide = vector::<Falcon1024>(1);
    let verified = verify_signature_native(&wide.record).expect("reference signature");
    let witness =
        FourCellEncodabilityWitness::<Falcon1024>::build(verified.s2()).expect("encodable");
    assert!(witness.budget_slack() <= 696);
    assert!(witness.budget_slack() >= 0);
    assert_eq!(witness.a0().as_ref().len(), 1024);
}
