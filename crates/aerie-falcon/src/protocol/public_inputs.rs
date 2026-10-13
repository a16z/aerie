//! Public statement inputs shared by the PCS prover and verifier.

use super::ring::{PublicRingKeys, PublicTargets};
use super::FalconDegree;

/// Public statement data the verifier derives independently of the proof.
#[derive(Clone, Debug)]
pub struct PublicInputs<'a, C = i64> {
    /// The LIVE record count `N`; the padded domain `N_sig*` is derived.
    pub records: usize,
    /// Salt transport mode, bound independently of the public-record digest.
    pub salt_mode: super::wire::SaltMode,
    /// The 32-byte public-record binding digest.
    pub public_records: &'a [u8],
    /// Public ring keys as a padded coefficient table or canonical bytes.
    pub h: PublicRingKeys<'a, C>,
    /// Public `HashToPoint` targets as a padded table or live source tuples.
    pub t: PublicTargets<'a, C>,
}

impl<C> PublicInputs<'_, C>
where
    C: Copy + Sync + Into<i64>,
{
    /// The parameter set, derived from the public ring-key representation.
    /// An untyped expanded table is accepted only when its shape identifies
    /// one degree uniquely.
    pub fn degree(&self) -> Result<FalconDegree, &'static str> {
        match self.h {
            PublicRingKeys::ExpandedOf(degree, _) => Ok(degree),
            PublicRingKeys::Canonical(_) => Ok(FalconDegree::Falcon512),
            PublicRingKeys::Canonical1024(_) => Ok(FalconDegree::Falcon1024),
            PublicRingKeys::Expanded(table) => {
                let mut candidates = [FalconDegree::Falcon512, FalconDegree::Falcon1024]
                    .into_iter()
                    .filter(|degree| {
                        degree
                            .padded_records(self.records)
                            .is_ok_and(|rows| table.len() == rows * degree.degree())
                    });
                let degree = candidates
                    .next()
                    .ok_or("public ring-key table has the wrong shape")?;
                if candidates.next().is_some() {
                    return Err("ambiguous small expanded table: use canonical public keys");
                }
                Ok(degree)
            }
        }
    }
}
