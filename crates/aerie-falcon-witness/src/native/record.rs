use super::{
    params::FalconParams,
    source::{CanonicalPublicKeyBytes, CanonicalSignatureBytes},
};

/// Direct record assembled only from canonically validated Falcon source bytes.
///
/// The fixed framing is Falcon v1.2 padded signature format with key-prefixed
/// message framing (raw mode, empty context), as implemented by fn-dsa 0.3.0.
/// A record carries only message bytes alongside the canonical key and
/// signature, whose salt is used in `HashToPoint` input
/// `salt || SHAKE256(pk,64) || 0x00 || 0x00 || message`.
/// Original `r || m` framing is unsupported.
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

    #[inline]
    #[must_use]
    pub const fn public_key(&self) -> &CanonicalPublicKeyBytes<P> {
        &self.public_key
    }

    #[inline]
    #[must_use]
    pub const fn signature(&self) -> &CanonicalSignatureBytes<P> {
        &self.signature
    }

    #[inline]
    #[must_use]
    pub fn message(&self) -> &[u8] {
        &self.message
    }

    /// Borrow the canonical sources without copying their payloads.
    #[inline]
    #[must_use]
    pub fn as_record_ref(&self) -> DirectRecordRef<'_, P> {
        DirectRecordRef::new(self.public_key(), self.signature(), self.message())
    }
}

/// Borrowed canonical key, signature, and message for native verification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectRecordRef<'a, P: FalconParams> {
    public_key: &'a CanonicalPublicKeyBytes<P>,
    signature: &'a CanonicalSignatureBytes<P>,
    message: &'a [u8],
}

impl<'a, P: FalconParams> DirectRecordRef<'a, P> {
    #[inline]
    #[must_use]
    pub const fn new(
        public_key: &'a CanonicalPublicKeyBytes<P>,
        signature: &'a CanonicalSignatureBytes<P>,
        message: &'a [u8],
    ) -> Self {
        Self {
            public_key,
            signature,
            message,
        }
    }

    #[inline]
    #[must_use]
    pub const fn public_key(&self) -> &'a CanonicalPublicKeyBytes<P> {
        self.public_key
    }

    #[inline]
    #[must_use]
    pub const fn signature(&self) -> &'a CanonicalSignatureBytes<P> {
        self.signature
    }

    #[inline]
    #[must_use]
    pub const fn message(&self) -> &'a [u8] {
        self.message
    }
}

/// Test-only structurally valid records with all-zero keys and bodies.
#[cfg(test)]
pub(crate) mod test_records {
    use super::super::params::FALCON_SALT_BYTES;
    use super::*;

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
    use super::super::params::{Falcon512, FalconParams};
    use super::*;

    #[test]
    fn record_view_borrows_the_owned_sources() {
        let record = test_records::zero_record::<Falcon512>(b"borrowed message");
        let view = record.as_record_ref();
        let copied = view;
        assert!(core::ptr::eq(copied.public_key(), record.public_key()));
        assert!(core::ptr::eq(copied.signature(), record.signature()));
        assert!(core::ptr::eq(copied.message(), record.message()));
    }

    #[test]
    fn rejects_wrong_outer_header() {
        let public_key = vec![0_u8; Falcon512::PUBLIC_KEY_BYTES];
        assert!(CanonicalPublicKeyBytes::<Falcon512>::parse(&public_key).is_err());
    }
}
