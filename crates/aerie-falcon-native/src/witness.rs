use crate::{
    arithmetic::{center_mod_q, FalconWitness, PublicRow},
    encodability::FourCellEncodabilityWitness,
    error::Result,
    params::{filled, FalconParams},
    preprocess::{hash_to_point_public_raw, hash_to_point_public_raw_batch, HashToPointInput},
    record::DirectRecord,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A source-bound Falcon signature accepted by the native Aerie relation.
///
/// Construction is possible only through this module's native verification
/// functions, which share the canonical decoders and `HashToPoint` path used
/// by later witness derivation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedFalconSignature<P: FalconParams> {
    public: PublicRow<P>,
    witness: FalconWitness<P>,
}

impl<P: FalconParams> VerifiedFalconSignature<P> {
    #[must_use]
    pub const fn h_centered(&self) -> &P::Array<i16> {
        self.public.h_centered()
    }

    #[must_use]
    pub const fn t_centered(&self) -> &P::Array<i16> {
        self.public.t_centered()
    }

    #[must_use]
    pub const fn s1(&self) -> &P::Array<i16> {
        self.witness.s1()
    }

    #[must_use]
    pub const fn s2(&self) -> &P::Array<i16> {
        self.witness.s2()
    }

    /// Read-only view of the verified Falcon witness. Reading is safe;
    /// construction remains sealed behind [`verify_signature_native`].
    #[must_use]
    pub const fn witness(&self) -> &FalconWitness<P> {
        &self.witness
    }

    #[must_use]
    pub const fn norm_squared(&self) -> u64 {
        self.witness.norm_squared()
    }

    /// The verified centered residue of `h * s2` in the Falcon ring.
    ///
    /// Native verification established `s1 = t - h * s2 (mod q)`, so the
    /// product residue is recovered without repeating the Falcon NTT. The
    /// returned coefficients are centered representatives modulo `q`.
    #[must_use]
    pub fn product_mod_q(&self) -> P::Array<i16> {
        let mut product = filled::<P, i16>(0);
        for ((slot, target), s1) in product
            .as_mut()
            .iter_mut()
            .zip(self.t_centered().as_ref())
            .zip(self.s1().as_ref())
        {
            *slot = center_mod_q(i32::from(*target) - i32::from(*s1));
        }
        product
    }

    /// Build the exact four-cell `s2` source committed by the protocol.
    pub fn four_cell_encodability_witness(&self) -> Result<FourCellEncodabilityWitness<P>> {
        FourCellEncodabilityWitness::<P>::build(self.s2())
    }
}

struct DecodedNativeSignature<P: FalconParams> {
    h_centered: P::Array<i16>,
    s2: P::Array<i16>,
}

struct BoundNativeSignature<P: FalconParams> {
    h_centered: P::Array<i16>,
    t_centered: P::Array<i16>,
    s2: P::Array<i16>,
}

fn decode_native_signature<P: FalconParams>(record: &DirectRecord<P>) -> DecodedNativeSignature<P> {
    DecodedNativeSignature {
        h_centered: *record.public_key().decode().h_centered(),
        s2: *record.signature().decode().s2(),
    }
}

fn bind_native_target<P: FalconParams>(
    record: &DirectRecord<P>,
    decoded: &DecodedNativeSignature<P>,
) -> BoundNativeSignature<P> {
    BoundNativeSignature {
        h_centered: decoded.h_centered,
        t_centered: hash_to_point_public_raw(
            record.public_key(),
            record.signature().salt(),
            record.message(),
        ),
        s2: decoded.s2,
    }
}

fn bind_precomputed_target<P: FalconParams>(
    decoded: &DecodedNativeSignature<P>,
    t_centered: &P::Array<i16>,
) -> BoundNativeSignature<P> {
    BoundNativeSignature {
        h_centered: decoded.h_centered,
        t_centered: *t_centered,
        s2: decoded.s2,
    }
}

fn hash_to_point_inputs<P: FalconParams>(
    records: &[DirectRecord<P>],
) -> Vec<HashToPointInput<'_, P>> {
    records
        .iter()
        .map(|record| {
            HashToPointInput::new(
                record.public_key(),
                record.signature().salt(),
                record.message(),
            )
        })
        .collect()
}

fn hash_to_point_batch<P: FalconParams>(records: &[DirectRecord<P>]) -> Vec<P::Array<i16>> {
    const RAYON_GRAIN: usize = 64;
    let inputs = hash_to_point_inputs(records);
    #[cfg(feature = "parallel")]
    {
        inputs
            .par_chunks(RAYON_GRAIN)
            .map(hash_to_point_public_raw_batch::<P>)
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = RAYON_GRAIN;
        hash_to_point_public_raw_batch(&inputs)
    }
}

fn verify_native_relation<P: FalconParams>(
    bound: &BoundNativeSignature<P>,
) -> Result<VerifiedFalconSignature<P>> {
    let public = PublicRow::<P>::new(&bound.h_centered, &bound.t_centered)?;
    let witness = FalconWitness::<P>::derive(&public, &bound.s2)?;
    Ok(VerifiedFalconSignature { public, witness })
}

/// Decode and verify one canonical Falcon direct record.
///
/// This is the sole public constructor for verified arithmetic values. It
/// derives `h`, `s2`, and `t` from the source bytes, reconstructs
/// `s1 = t - h*s2` with the Falcon negacyclic adapter, and enforces the exact
/// integer Falcon norm bound of the record's parameter set.
pub fn verify_signature_native<P: FalconParams>(
    record: impl std::borrow::Borrow<DirectRecord<P>>,
) -> Result<VerifiedFalconSignature<P>> {
    let record = record.borrow();
    let decoded = decode_native_signature(record);
    let bound = bind_native_target(record, &decoded);
    verify_native_relation(&bound)
}

/// Decode and verify a batch while exposing aggregate phase spans.
///
/// Hash-to-point stays batched. Each result is then decoded, bound to its
/// corresponding target, and checked locally, without retaining batch-wide
/// decoded or bound coefficient arrays.
pub fn verify_signatures_native<P: FalconParams>(
    records: &[DirectRecord<P>],
) -> Result<Vec<VerifiedFalconSignature<P>>> {
    let hash_span = tracing::info_span!("v1.prove.witness.native_verify.hash_to_point");
    let targets = hash_span.in_scope(|| hash_to_point_batch(records));

    let verify_span = tracing::info_span!("v1.prove.witness.native_verify.verify_records");
    #[cfg(feature = "parallel")]
    let verified = verify_span.in_scope(|| {
        records
            .par_iter()
            .zip(targets.par_iter())
            .map(|(record, target)| {
                let decoded = decode_native_signature(record);
                let bound = bind_precomputed_target(&decoded, target);
                verify_native_relation(&bound)
            })
            .collect::<Result<Vec<_>>>()
    });

    #[cfg(not(feature = "parallel"))]
    let verified = verify_span.in_scope(|| {
        records
            .iter()
            .zip(&targets)
            .map(|(record, target)| {
                let decoded = decode_native_signature(record);
                let bound = bind_precomputed_target(&decoded, target);
                verify_native_relation(&bound)
            })
            .collect()
    });
    verified
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        falcon_ntt::negacyclic_product,
        params::{Falcon1024, Falcon512},
        record::test_records,
    };

    fn structurally_valid_bytes_do_not_bypass_native_norm_for<P: FalconParams>() {
        let record = test_records::zero_record::<P>(b"not signed");
        assert!(verify_signature_native(&record).is_err());
        assert!(verify_signatures_native(std::slice::from_ref(&record)).is_err());
    }

    #[test]
    fn structurally_valid_bytes_do_not_bypass_native_norm() {
        structurally_valid_bytes_do_not_bypass_native_norm_for::<Falcon512>();
        structurally_valid_bytes_do_not_bypass_native_norm_for::<Falcon1024>();
    }

    fn verified_product_residue_matches_for<P: FalconParams>() {
        let mut h_centered = filled::<P, i16>(0);
        h_centered.as_mut()[0] = 3;
        h_centered.as_mut()[5] = -2;
        let mut s2 = filled::<P, i16>(0);
        s2.as_mut()[0] = 7;
        s2.as_mut()[2] = -4;
        let bound = BoundNativeSignature::<P> {
            h_centered,
            t_centered: filled::<P, i16>(0),
            s2,
        };
        let verified = verify_native_relation(&bound).expect("small native witness");

        assert_eq!(
            verified.product_mod_q(),
            negacyclic_product::<P>(&h_centered, &s2),
        );
    }

    #[test]
    fn verified_product_residue_matches_the_native_ring_product() {
        verified_product_residue_matches_for::<Falcon512>();
        verified_product_residue_matches_for::<Falcon1024>();
    }
}
