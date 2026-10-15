//! Differential tests against the `fn-dsa` reference implementation for both
//! standardized parameter sets.
//!
//! Keys and signatures come from real FN-DSA key generation and signing in the
//! fixed raw-message, empty-context mode. Randomness is a seeded SHAKE256
//! stream, so the vectors are reproducible but are not security-grade keys.

use aerie_falcon_native::{
    params::{COMPRESSED_COEFFICIENT_BASE_BITS, FALCON_SALT_BYTES},
    source::{CanonicalPublicKeyBytes, CanonicalSignatureBytes},
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
    fn verify(vrfy_key: &[u8], signature: &[u8], context: &[u8], message: &[u8]) -> bool;
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

            fn verify(vrfy_key: &[u8], signature: &[u8], context: &[u8], message: &[u8]) -> bool {
                <$vrfy>::decode(vrfy_key).is_some_and(|key| {
                    key.verify(signature, &DomainContext(context), &HASH_ID_RAW, message)
                })
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
    public_key: CanonicalPublicKeyBytes<P>,
    canonical_signature: CanonicalSignatureBytes<P>,
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
    let public_key =
        CanonicalPublicKeyBytes::<P>::parse(&vrfy_key).expect("canonical reference key");
    let canonical_signature =
        CanonicalSignatureBytes::<P>::parse(&signature).expect("canonical reference signature");
    Vector {
        vrfy_key,
        signature,
        message,
        public_key,
        canonical_signature,
    }
}

const RECORDS: u64 = 8;

fn honest_reference_bytes_decode_for<P: Reference>() {
    for index in 0..RECORDS {
        let sample = vector::<P>(index);
        assert!(P::verify(
            &sample.vrfy_key,
            &sample.signature,
            b"",
            &sample.message
        ));
        assert_eq!(sample.public_key.as_bytes().as_ref(), sample.vrfy_key);
        assert_eq!(
            sample.canonical_signature.as_bytes().as_ref(),
            sample.signature
        );
        let decoded_key = sample.public_key.decode();
        assert_eq!(decoded_key.h_centered().as_ref().len(), P::DEGREE);
        assert_eq!(
            sample
                .public_key
                .centered_coefficients()
                .collect::<Vec<_>>(),
            decoded_key.h_centered().as_ref()
        );
        let decoded_signature = sample.canonical_signature.decode();
        assert_eq!(decoded_signature.s2().as_ref().len(), P::DEGREE);
        assert_eq!(decoded_signature.salt(), sample.canonical_signature.salt());
        assert_eq!(
            decoded_signature.salt().as_slice(),
            &sample.signature[1..=FALCON_SALT_BYTES]
        );

        let mut invalid_header = sample.signature.clone();
        invalid_header[0] ^= 0xff;
        assert!(CanonicalSignatureBytes::<P>::parse(&invalid_header).is_err());
        assert!(!P::verify(
            &sample.vrfy_key,
            &invalid_header,
            b"",
            &sample.message
        ));

        let body_bits = decoded_signature
            .s2()
            .as_ref()
            .iter()
            .map(|coefficient| {
                COMPRESSED_COEFFICIENT_BASE_BITS + usize::from(coefficient.unsigned_abs() >> 7)
            })
            .sum::<usize>();
        if body_bits <= P::PADDED_BODY_BITS - 8 {
            assert_eq!(sample.signature.last(), Some(&0));
            let mut invalid_padding = sample.signature.clone();
            *invalid_padding.last_mut().expect("padded signature") = 1;
            assert!(CanonicalSignatureBytes::<P>::parse(&invalid_padding).is_err());
            assert!(!P::verify(
                &sample.vrfy_key,
                &invalid_padding,
                b"",
                &sample.message
            ));
        }
    }
}

#[test]
fn honest_falcon_512_reference_bytes_decode_canonically() {
    honest_reference_bytes_decode_for::<Falcon512>();
}

#[test]
fn honest_falcon_1024_reference_bytes_decode_canonically() {
    honest_reference_bytes_decode_for::<Falcon1024>();
}

#[test]
fn codecs_reject_other_parameter_set_reference_bytes() {
    let wide = vector::<Falcon1024>(0);
    let narrow = vector::<Falcon512>(0);
    assert!(CanonicalPublicKeyBytes::<Falcon512>::parse(&wide.vrfy_key).is_err());
    assert!(CanonicalSignatureBytes::<Falcon512>::parse(&wide.signature).is_err());
    assert!(CanonicalPublicKeyBytes::<Falcon1024>::parse(&narrow.vrfy_key).is_err());
    assert!(CanonicalSignatureBytes::<Falcon1024>::parse(&narrow.signature).is_err());
}
