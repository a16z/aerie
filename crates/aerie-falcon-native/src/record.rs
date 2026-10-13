use crate::{
    params::FalconParams,
    source::{CanonicalPublicKeyBytes, CanonicalSignatureBytes},
};

/// Direct record assembled only from canonically validated Falcon source bytes.
///
/// The v0 protocol fixes raw message mode and the empty context inside the
/// pinned Falcon `HashToPoint` framing, so a record carries only the
/// message bytes alongside the canonical key and signature.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectRecord<P: FalconParams> {
    public_key: CanonicalPublicKeyBytes<P>,
    signature: CanonicalSignatureBytes<P>,
    message: Vec<u8>,
}

impl<P: FalconParams> DirectRecord<P> {
    #[must_use]
    pub const fn new(
        public_key: CanonicalPublicKeyBytes<P>,
        signature: CanonicalSignatureBytes<P>,
        message: Vec<u8>,
    ) -> Self {
        Self {
            public_key,
            signature,
            message,
        }
    }

    #[must_use]
    pub const fn public_key(&self) -> &CanonicalPublicKeyBytes<P> {
        &self.public_key
    }

    #[must_use]
    pub const fn signature(&self) -> &CanonicalSignatureBytes<P> {
        &self.signature
    }

    #[must_use]
    pub fn message(&self) -> &[u8] {
        &self.message
    }
}

/// Test-only structurally valid records with all-zero keys and bodies.
#[cfg(test)]
pub(crate) mod test_records {
    use super::*;
    use crate::params::FALCON_SALT_BYTES;

    /// A record whose key and `s2` are all zero: canonical bytes, not a
    /// signature that verifies.
    pub(crate) fn zero_record<P: FalconParams>(message: &[u8]) -> DirectRecord<P> {
        let mut public_key = vec![0_u8; P::PUBLIC_KEY_BYTES];
        public_key[0] = P::PUBLIC_KEY_HEADER;
        let mut signature = vec![0_u8; P::SIGNATURE_BYTES];
        signature[0] = P::PADDED_SIGNATURE_HEADER;
        // Zero coefficients use 9 bits each: eight zero bits, then a one.
        let mut bit = 0;
        for _ in 0..P::DEGREE {
            bit += 8;
            let byte = 1 + FALCON_SALT_BYTES + bit / 8;
            let shift = 7 - bit % 8;
            signature[byte] |= 1 << shift;
            bit += 1;
        }
        DirectRecord::new(
            CanonicalPublicKeyBytes::<P>::parse(&public_key).expect("canonical public key"),
            CanonicalSignatureBytes::<P>::parse(&signature).expect("canonical signature"),
            message.to_vec(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Falcon512;

    #[test]
    fn rejects_wrong_outer_header() {
        let public_key = vec![0_u8; crate::source::FALCON_512_PUBLIC_KEY_BYTES];
        assert!(CanonicalPublicKeyBytes::<Falcon512>::parse(&public_key).is_err());
    }
}
