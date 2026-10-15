//! Public profile and exact commitment geometry for the Falcon protocol.
//!
//! These types bind the three implemented stages and PCS opening to the
//! public commitment layout and wire profile.

use crate::UnsupportedBatch;

use aerie_falcon_native::{Falcon1024, Falcon512, FalconParams};

#[doc(hidden)]
pub mod algebra;
#[doc(hidden)]
pub mod grinding;
#[doc(hidden)]
pub mod instance;
pub mod manifest;
pub mod openings;
pub mod pcs;
pub mod projection;
#[doc(hidden)]
pub mod public_inputs;
#[doc(hidden)]
pub mod public_records_binding;
pub mod relations;
pub mod ring;
mod small_signed;

#[doc(hidden)]
pub mod stage;
pub mod stage1;
pub mod stage2;
pub mod stage3;
pub mod transcript;
pub mod transport;
pub mod wire;
pub mod zero_extension;

/// Byte length of the protocol identity: the little-endian `u16`
/// [`crate::WIRE_VERSION`] followed by the one-byte parameter set
/// (`log2 n`). The wire header and every transcript binding carry it.
pub(crate) const PROTOCOL_IDENTITY_BYTES: usize = 3;
/// Identifier of the exact four-group Falcon-512 commitment layout in
/// Section 5, bound in the transcript prefix of Section 10.1.
pub(crate) const FALCON_512_COMMITMENT_LAYOUT: u32 = 5;
/// Identifier of the exact four-group Falcon-1024 commitment layout
/// in Section 5, bound in the transcript prefix of Section 10.1.
pub(crate) const FALCON_1024_COMMITMENT_LAYOUT: u32 = 6;
/// Maximum supported live Falcon record batch.
pub const MAX_RECORDS: usize = aerie_falcon_statement::MAX_PRODUCTION_BATCH_SIZE;
/// Logical batch minimum `n * N* >= 4096`: 8 padded rows at Falcon-512
/// and 4 at Falcon-1024. It implies the first norm core's weaker two-lane
/// requirement `2 * n * N* >= 4096`. Physical PCS tables are separately
/// zero-extended to at least 8192 cells by `zero_extension::pcs_length`.
#[doc(hidden)]
pub const MIN_PADDED_COEFFICIENTS: usize = 1 << 12;

/// Run two independent closures concurrently when the caller's measured gate
/// enables it, with an identical serial fallback when Rayon is unavailable.
pub fn join_if<A, B>(
    parallel: bool,
    left: impl FnOnce() -> A + Send,
    right: impl FnOnce() -> B + Send,
) -> (A, B)
where
    A: Send,
    B: Send,
{
    #[cfg(feature = "parallel")]
    if parallel {
        return rayon::join(left, right);
    }
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;
    (left(), right())
}

/// Derive the canonical Falcon-512 padded row count.
#[cfg(any(test, feature = "internals"))]
pub fn padded_records(records: usize) -> Result<usize, UnsupportedBatch> {
    FalconDegree::Falcon512.padded_records(records)
}

/// Standardized Falcon parameter set selected for one aggregate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FalconDegree {
    Falcon512,
    Falcon1024,
}

impl FalconDegree {
    /// Derive N* using this parameter set's minimum admitted shape.
    pub fn padded_records(self, records: usize) -> Result<usize, UnsupportedBatch> {
        if records == 0 || records > MAX_RECORDS {
            return Err(UnsupportedBatch::RecordCount { records });
        }
        Ok(records
            .next_power_of_two()
            .max(MIN_PADDED_COEFFICIENTS / self.degree()))
    }

    /// Ring degree `n`.
    #[must_use]
    pub const fn degree(self) -> usize {
        match self {
            Self::Falcon512 => 512,
            Self::Falcon1024 => 1024,
        }
    }

    /// Squared verification bound `floor(beta^2)`.
    #[must_use]
    pub const fn beta_squared(self) -> u64 {
        match self {
            Self::Falcon512 => Falcon512::BETA_SQUARED,
            Self::Falcon1024 => Falcon1024::BETA_SQUARED,
        }
    }

    /// Additive constant of the budget-slack recomposition.
    #[must_use]
    pub const fn budget_constant(self) -> i16 {
        match self {
            Self::Falcon512 => Falcon512::BUDGET_CONSTANT,
            Self::Falcon1024 => Falcon1024::BUDGET_CONSTANT,
        }
    }

    /// Bits available for all unary high parts of one padded signature.
    #[must_use]
    pub const fn compressed_high_bits_budget(self) -> usize {
        match self {
            Self::Falcon512 => Falcon512::COMPRESSED_HIGH_BITS_BUDGET,
            Self::Falcon1024 => Falcon1024::COMPRESSED_HIGH_BITS_BUDGET,
        }
    }

    /// `log2 n`: the number of coefficient variables.
    #[must_use]
    pub const fn logn(self) -> usize {
        match self {
            Self::Falcon512 => 9,
            Self::Falcon1024 => 10,
        }
    }

    /// Recover the parameter set from a ring degree.
    pub fn from_degree(degree: usize) -> Result<Self, UnsupportedBatch> {
        match degree {
            512 => Ok(Self::Falcon512),
            1024 => Ok(Self::Falcon1024),
            _ => Err(UnsupportedBatch::Degree { degree }),
        }
    }

    /// Parameter-set byte of the wire header and every transcript binding:
    /// `log2 n`, so 9 for Falcon-512 and 10 for Falcon-1024.
    #[must_use]
    pub const fn parameter_set(self) -> u8 {
        self.logn() as u8
    }

    /// Recover the parameter set from its wire byte.
    pub const fn from_parameter_set(parameter_set: u8) -> Result<Self, UnsupportedBatch> {
        match parameter_set {
            9 => Ok(Self::Falcon512),
            10 => Ok(Self::Falcon1024),
            _ => Err(UnsupportedBatch::ParameterSet { parameter_set }),
        }
    }

    /// Protocol identity bytes: little-endian `u16` [`crate::WIRE_VERSION`]
    /// followed by [`Self::parameter_set`]. Both parameter sets share the
    /// version and the root domain [`crate::PROTOCOL_DOMAIN`]; this pair is
    /// what separates them on the wire and in every transcript binding.
    #[must_use]
    pub const fn identity_bytes(self) -> [u8; PROTOCOL_IDENTITY_BYTES] {
        let version = crate::WIRE_VERSION.to_le_bytes();
        [version[0], version[1], self.parameter_set()]
    }

    /// Commitment layout identifier.
    #[must_use]
    pub const fn commitment_layout(self) -> u32 {
        match self {
            Self::Falcon512 => FALCON_512_COMMITMENT_LAYOUT,
            Self::Falcon1024 => FALCON_1024_COMMITMENT_LAYOUT,
        }
    }

    /// The public clear-projection profile of this parameter set for one
    /// live batch (profile of its padded domain `N_sig*`).
    pub fn projection_profile(
        self,
        records: usize,
    ) -> Result<projection::ProjectionProfile, UnsupportedBatch> {
        projection::ProjectionProfile::for_batch(self, self.padded_records(records)?)
    }
}

/// A standardized Falcon parameter set as the aggregation protocol names
/// it: [`Falcon512`] or [`Falcon1024`].
///
/// This is the type parameter of the public API (`aerie_prover::FalconAggregateSetup`,
/// [`pcs::FalconVerifierSetup`], `aerie_prover::prove_aggregate`,
/// [`transport::verify_aggregate`]). It extends the statement layer's
/// [`aerie_falcon_statement::DirectParams`], which only these two types implement. The
/// associated constants are the per-set protocol identity; they are derived
/// from `FalconDegree`, the runtime form the untyped internals use.
pub trait AggregationParams: aerie_falcon_statement::DirectParams {
    /// The parameter set as a runtime value.
    const DEGREE_ID: FalconDegree;
    /// The wire header and transcript parameter-set byte (`log2 n`).
    const PARAMETER_SET: u8 = Self::DEGREE_ID.parameter_set();
    /// The number of committed groups.
    const GROUP_COUNT: usize = manifest::GROUP_COUNT;
    /// Wrap canonical public keys as the stage-three public ring keys.
    fn ring_keys(
        public_keys: &[aerie_falcon_native::source::CanonicalPublicKeyBytes<Self>],
    ) -> ring::PublicRingKeys<'_, i16>;
}

impl AggregationParams for Falcon512 {
    const DEGREE_ID: FalconDegree = FalconDegree::Falcon512;
    #[inline]
    fn ring_keys(
        public_keys: &[aerie_falcon_native::source::CanonicalPublicKeyBytes<Self>],
    ) -> ring::PublicRingKeys<'_, i16> {
        ring::PublicRingKeys::Canonical512(public_keys)
    }
}

impl AggregationParams for Falcon1024 {
    const DEGREE_ID: FalconDegree = FalconDegree::Falcon1024;
    #[inline]
    fn ring_keys(
        public_keys: &[aerie_falcon_native::source::CanonicalPublicKeyBytes<Self>],
    ) -> ring::PublicRingKeys<'_, i16> {
        ring::PublicRingKeys::Canonical1024(public_keys)
    }
}

/// Exact per-polynomial cell count of each version 1 commitment group.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommitmentCellCounts {
    pub s1: usize,
    pub slack: usize,
    pub s2_sources: usize,
    pub budget: usize,
}

impl CommitmentCellCounts {
    /// Derive all four groups' per-polynomial shapes from the public batch. Logical
    /// lengths are in padded `N_sig*` cells; the first `N` rows are live and the
    /// padded suffix is semantically zero.
    pub fn for_batch(degree: FalconDegree, records: usize) -> Result<Self, UnsupportedBatch> {
        let padded = degree.padded_records(records)?;
        let n = degree.degree();
        Ok(Self {
            s1: n * padded,
            slack: 4 * padded,
            s2_sources: n * padded,
            budget: padded,
        })
    }

    /// Number of committed groups in canonical wire order.
    #[must_use]
    #[cfg(any(test, feature = "internals"))]
    pub const fn commitment_count() -> usize {
        manifest::GROUP_COUNT
    }

    /// Total committed base-field cells, with no implicit rounding or padding.
    #[must_use]
    #[cfg(any(test, feature = "internals"))]
    pub const fn total(self) -> usize {
        self.s1 + self.slack + 4 * self.s2_sources + 3 * self.budget
    }
}

/// Work threshold shared by producer parallel scheduling.
#[doc(hidden)]
pub const PARALLEL_MIN_WORK: usize = 1 << 12;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_commitments_have_exact_unpadded_sizes() {
        for records in [8, 1 << 14, MAX_RECORDS] {
            let counts = CommitmentCellCounts::for_batch(FalconDegree::Falcon512, records)
                .expect("target commitment counts");
            assert_eq!(counts.s1, 512 * records);
            assert_eq!(counts.slack, 4 * records);
            assert_eq!(counts.s2_sources, 512 * records);
            assert_eq!(counts.budget, records);
            assert_eq!(counts.total(), 2_567 * records);
            assert_eq!(CommitmentCellCounts::commitment_count(), 4);
            let wide = CommitmentCellCounts::for_batch(FalconDegree::Falcon1024, records)
                .expect("Falcon-1024 commitment counts");
            assert_eq!(wide.total(), 5_127 * records);
        }
    }

    #[test]
    fn arbitrary_batches_pad_to_the_minimum_shape() {
        for (records, expected_padded) in [
            (1, 8),
            (3, 8),
            (5, 8),
            (11, 16),
            (5_000, 8_192),
            (65_535, 65_536),
        ] {
            assert_eq!(padded_records(records), Ok(expected_padded));
            let counts = CommitmentCellCounts::for_batch(FalconDegree::Falcon512, records)
                .expect("arbitrary-batch commitment counts");
            assert_eq!(
                counts,
                CommitmentCellCounts::for_batch(FalconDegree::Falcon512, expected_padded).unwrap(),
                "logical lengths are functions of the padded domain"
            );
            assert_eq!(counts.s1, 512 * expected_padded);
        }
    }

    #[test]
    fn cataloged_power_of_two_batches_are_runtime_supported() {
        for records in [1 << 12, 1 << 13, 1 << 14, 1 << 15, 1 << 16] {
            assert_eq!(padded_records(records), Ok(records));
            for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
                assert!(degree.projection_profile(records).is_ok());
            }
        }
    }

    #[test]
    fn projection_profiles_need_the_minimum_padded_domain() {
        // The PCS minima also satisfy the projection geometry.
        assert!(FalconDegree::Falcon512.projection_profile(2).is_ok());
        assert!(FalconDegree::Falcon512.projection_profile(3).is_ok());
        assert!(FalconDegree::Falcon1024.projection_profile(1).is_ok());
        assert!(FalconDegree::Falcon1024.projection_profile(2).is_ok());
    }

    #[test]
    fn malformed_batches_reject() {
        for records in [0, (1 << 16) + 1, 1 << 17] {
            assert!(padded_records(records).is_err());
            assert!(CommitmentCellCounts::for_batch(FalconDegree::Falcon512, records).is_err());
        }
    }

    #[test]
    fn degree_identity_is_bound_separately() {
        assert_eq!(FalconDegree::from_degree(512), Ok(FalconDegree::Falcon512));
        assert_eq!(
            FalconDegree::from_degree(1024),
            Ok(FalconDegree::Falcon1024)
        );
        assert!(FalconDegree::from_degree(256).is_err());
        // One version and one root domain; the parameter-set byte separates.
        assert_eq!(FalconDegree::Falcon512.identity_bytes(), [1, 0, 9]);
        assert_eq!(FalconDegree::Falcon1024.identity_bytes(), [1, 0, 10]);
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            assert_eq!(
                FalconDegree::from_parameter_set(degree.parameter_set()),
                Ok(degree)
            );
        }
        for byte in [0_u8, 8, 11, 255] {
            assert!(FalconDegree::from_parameter_set(byte).is_err());
        }
        assert_ne!(
            FalconDegree::Falcon512.commitment_layout(),
            FalconDegree::Falcon1024.commitment_layout()
        );
        assert_eq!(FalconDegree::Falcon1024.logn(), 10);
    }
}
