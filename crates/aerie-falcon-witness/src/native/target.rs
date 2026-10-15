use super::{
    params::{FalconParams, FALCON_SALT_BYTES},
    preprocess::hash_to_point_public_raw,
    record::DirectRecord,
};

/// Verifier-local hash target bound to the source tuple that produced it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HashToPointTarget<P: FalconParams> {
    salt: [u8; FALCON_SALT_BYTES],
    message: Vec<u8>,
    t_centered: P::Array<i16>,
}

impl<P: FalconParams> HashToPointTarget<P> {
    /// Derive the Falcon target from one canonical direct record.
    ///
    /// The fixed framing is Falcon v1.2 padded signature format with key-prefixed
    /// message framing (raw mode, empty context), as implemented by fn-dsa 0.3.0.
    /// The `HashToPoint` input is
    /// `salt || SHAKE256(pk,64) || 0x00 || 0x00 || message`.
    /// Original `r || m` framing is unsupported.
    #[must_use]
    pub fn derive(record: &DirectRecord<P>) -> Self {
        let salt = record.signature().salt();
        let t_centered = hash_to_point_public_raw(record.public_key(), salt, record.message());
        Self {
            salt: *salt,
            message: record.message().to_vec(),
            t_centered,
        }
    }

    #[must_use]
    pub const fn salt(&self) -> &[u8; FALCON_SALT_BYTES] {
        &self.salt
    }

    #[must_use]
    pub fn message(&self) -> &[u8] {
        &self.message
    }

    #[must_use]
    pub const fn t_centered(&self) -> &P::Array<i16> {
        &self.t_centered
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        params::{Falcon1024, Falcon512},
        preprocess::HashedVerifyingKey,
        record::test_records::zero_record,
    };
    use super::*;

    #[test]
    fn hash_target_binds_message() {
        let first = HashToPointTarget::<Falcon512>::derive(&zero_record(b"message"));
        let changed_message = HashToPointTarget::<Falcon512>::derive(&zero_record(b"message2"));
        assert_ne!(first.t_centered, changed_message.t_centered);
        let wide = HashToPointTarget::<Falcon1024>::derive(&zero_record(b"message"));
        assert_eq!(wide.t_centered().as_ref().len(), 1024);
        // Same framing, longer squeeze: the first 512 coefficients agree
        // whenever the keys hash identically, which the zero keys do not.
        assert_ne!(
            HashedVerifyingKey::derive(zero_record::<Falcon512>(b"").public_key()),
            HashedVerifyingKey::derive(zero_record::<Falcon1024>(b"").public_key())
        );
    }

    fn fixed_mode_framing_for<P: FalconParams>() {
        let message = b"framing pin";
        let record = zero_record::<P>(message);
        let target = HashToPointTarget::<P>::derive(&record);
        // Native's independent SHAKE framing oracle checks the public helper;
        // retain the signed-record carrier's equivalence to that same helper.
        assert_eq!(
            target.t_centered(),
            &hash_to_point_public_raw(record.public_key(), record.signature().salt(), message)
        );
    }

    #[test]
    fn fixed_mode_framing_absorbs_salt_hpk_and_two_zero_bytes() {
        fixed_mode_framing_for::<Falcon512>();
        fixed_mode_framing_for::<Falcon1024>();
    }
}
