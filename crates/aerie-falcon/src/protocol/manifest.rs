//! Exact commitment manifest for the Falcon protocols.
//!
//! The manifest fixes the canonical wire order, native polynomial counts,
//! per-polynomial cell counts, and PCS schedule family of every committed
//! group, and serializes the complete public profile binding absorbed before any
//! commitment. Both parameter sets commit four groups, all before the
//! projection seed (`specs/falcon.md` Section 5). Budget is the final Akita
//! group, so PCS execution follows canonical wire order.

use super::{projection::ProjectionProfile, CommitmentCellCounts, FalconDegree};

/// Canonical role of one committed polynomial group, in wire order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GroupRole {
    S1,
    Slack,
    S2Sources,
    Budget,
}

impl GroupRole {
    /// Canonical byte absorbed into the profile binding.
    #[must_use]
    pub const fn transcript_id(self) -> u8 {
        match self {
            Self::S1 => 0,
            Self::Slack => 1,
            Self::S2Sources => 2,
            Self::Budget => 3,
        }
    }
}

/// PCS schedule family selected for one group. These are extraction
/// parameters, not substitutes for the three stages' range, sign, norm, and budget predicates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcsFamily {
    Width14,
    Width4,
}

impl PcsFamily {
    #[must_use]
    pub const fn transcript_id(self) -> u8 {
        match self {
            Self::Width14 => 0,
            Self::Width4 => 1,
        }
    }
}

/// One committed group's exact public shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupDescriptor {
    pub role: GroupRole,
    pub logical_polynomials: usize,
    /// Logical cell count of each same-shaped polynomial in the group.
    pub logical_cells: usize,
    pub family: PcsFamily,
}

/// Number of committed groups in the Falcon-512 protocol.
pub const FALCON_512_GROUP_COUNT: usize = 4;
/// Number of committed groups in the Falcon-1024 protocol.
pub const FALCON_1024_GROUP_COUNT: usize = 4;
/// Group count shared by both parameter sets.
pub const GROUP_COUNT: usize = 4;

const _: () = assert!(FALCON_512_GROUP_COUNT == GROUP_COUNT);
const _: () = assert!(FALCON_1024_GROUP_COUNT == GROUP_COUNT);

/// Complete public commitment manifest for one batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommitmentManifest {
    degree: FalconDegree,
    records: usize,
    padded_records: usize,
    projection: ProjectionProfile,
    groups: [GroupDescriptor; GROUP_COUNT],
}

impl CommitmentManifest {
    /// Derive the canonical Falcon-512 manifest for a checked batch.
    #[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
    pub fn for_batch(records: usize) -> Result<Self, &'static str> {
        Self::for_batch_of(FalconDegree::Falcon512, records)
    }

    /// Derive the canonical manifest of one parameter set.
    ///
    /// The domain includes the parameter set's minimum admitted PCS shape.
    pub fn for_batch_of(degree: FalconDegree, records: usize) -> Result<Self, &'static str> {
        let padded = degree.padded_records(records)?;
        let projection = ProjectionProfile::for_batch(degree, padded)?;
        let counts = CommitmentCellCounts::for_batch(degree, records)?;
        let descriptor = |role, logical_polynomials, logical_cells, family| GroupDescriptor {
            role,
            logical_polynomials,
            logical_cells,
            family,
        };
        let groups = [
            descriptor(GroupRole::S1, 1, counts.s1, PcsFamily::Width14),
            descriptor(GroupRole::Slack, 1, counts.slack, PcsFamily::Width14),
            descriptor(
                GroupRole::S2Sources,
                4,
                counts.s2_sources,
                PcsFamily::Width4,
            ),
            descriptor(GroupRole::Budget, 3, counts.budget, PcsFamily::Width4),
        ];
        Ok(Self {
            degree,
            records,
            padded_records: padded,
            projection,
            groups,
        })
    }

    #[must_use]
    #[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
    pub const fn degree(&self) -> FalconDegree {
        self.degree
    }

    #[must_use]
    #[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
    pub const fn records(&self) -> usize {
        self.records
    }

    #[must_use]
    pub const fn padded_records(&self) -> usize {
        self.padded_records
    }

    /// The public clear-projection profile bound by this manifest.
    #[must_use]
    #[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
    pub const fn projection_profile(&self) -> &ProjectionProfile {
        &self.projection
    }

    /// Every committed group in canonical wire order.
    #[must_use]
    pub fn groups(&self) -> &[GroupDescriptor] {
        &self.groups
    }

    #[must_use]
    pub const fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Total logical committed cells.
    #[must_use]
    #[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
    pub fn total_logical_cells(&self) -> usize {
        self.groups()
            .iter()
            .map(|group| group.logical_polynomials * group.logical_cells)
            .sum()
    }

    /// Canonical byte encoding of the complete public profile.
    ///
    /// The encoding is the same for both parameter sets: the protocol
    /// identity (`u16` version, parameter-set byte), `LE32` layout
    /// identifier, `LE32` ring degree, `LE64` live and padded record counts,
    /// the self-labelled [`ProjectionProfile::binding_bytes`], and every
    /// exact group shape (`U8` role, `U8` polynomial count, `LE64` cells, `U8`
    /// family). All of it precedes every commitment and the projection seed
    /// in the transcript.
    #[must_use]
    pub fn binding_bytes(&self) -> Vec<u8> {
        let projection = self.projection.binding_bytes();
        let mut bytes = Vec::with_capacity(27 + projection.len() + 11 * GROUP_COUNT);
        bytes.extend_from_slice(&self.degree.identity_bytes());
        bytes.extend_from_slice(&self.degree.commitment_layout().to_le_bytes());
        bytes.extend_from_slice(&(self.degree.degree() as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.records as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.padded_records as u64).to_le_bytes());
        bytes.extend_from_slice(&projection);
        for group in self.groups() {
            bytes.push(group.role.transcript_id());
            bytes.push(group.logical_polynomials as u8);
            bytes.extend_from_slice(&(group.logical_cells as u64).to_le_bytes());
            bytes.push(group.family.transcript_id());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target_manifest() -> CommitmentManifest {
        CommitmentManifest::for_batch(8).expect("target manifest")
    }

    fn wide_manifest(records: usize) -> CommitmentManifest {
        CommitmentManifest::for_batch_of(FalconDegree::Falcon1024, records)
            .expect("Falcon-1024 manifest")
    }

    #[test]
    fn manifest_matches_the_exact_commitment_geometry() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            let manifest = CommitmentManifest::for_batch_of(degree, 8).expect("manifest");
            let counts = CommitmentCellCounts::for_batch(degree, 8).unwrap();
            let cells = [counts.s1, counts.slack, counts.s2_sources, counts.budget];
            let polynomials = [1, 1, 4, 3];
            assert_eq!(manifest.groups().len(), GROUP_COUNT);
            for ((group, &expected_cells), &expected_polynomials) in
                manifest.groups().iter().zip(&cells).zip(&polynomials)
            {
                assert_eq!(group.logical_cells, expected_cells);
                assert_eq!(group.logical_polynomials, expected_polynomials);
            }
            assert_eq!(manifest.total_logical_cells(), counts.total());
            assert_eq!(manifest.degree(), degree);
            assert_eq!(
                manifest.projection_profile(),
                &ProjectionProfile::for_batch(degree, 8).unwrap()
            );
        }
    }

    #[test]
    fn manifest_pins_widths_and_canonical_group_order() {
        for manifest in [target_manifest(), wide_manifest(8)] {
            let roles: Vec<_> = manifest.groups().iter().map(|group| group.role).collect();
            assert_eq!(
                roles,
                vec![
                    GroupRole::S1,
                    GroupRole::Slack,
                    GroupRole::S2Sources,
                    GroupRole::Budget,
                ]
            );
            for (index, group) in manifest.groups().iter().enumerate() {
                let family = if index < 2 {
                    PcsFamily::Width14
                } else {
                    PcsFamily::Width4
                };
                assert_eq!(group.family, family, "group {index}");
                assert_eq!(usize::from(group.role.transcript_id()), index);
            }
        }
    }

    #[test]
    fn falcon_512_binding_bytes_are_pinned() {
        // Regression pin: u16 version, parameter set, layout, degree, N, N*,
        // the projection profile, then four groups of eleven bytes.
        let manifest = target_manifest();
        let bytes = manifest.binding_bytes();
        let mut expected = Vec::new();
        expected.extend_from_slice(&2_u16.to_le_bytes());
        expected.push(9);
        expected.extend_from_slice(&5_u32.to_le_bytes());
        expected.extend_from_slice(&512_u32.to_le_bytes());
        expected.extend_from_slice(&8_u64.to_le_bytes());
        expected.extend_from_slice(&8_u64.to_le_bytes());
        let projection = ProjectionProfile::for_batch(FalconDegree::Falcon512, 8)
            .unwrap()
            .binding_bytes();
        expected.extend_from_slice(&projection);
        for (role, polynomials, cells, family) in [
            (0_u8, 1_u8, 512 * 8_u64, 0_u8),
            (1, 1, 4 * 8, 0),
            (2, 4, 512 * 8, 1),
            (3, 3, 8, 1),
        ] {
            expected.push(role);
            expected.push(polynomials);
            expected.extend_from_slice(&cells.to_le_bytes());
            expected.push(family);
        }
        assert_eq!(bytes, expected);
        assert_eq!(bytes.len(), 3 + 4 + 4 + 8 + 8 + projection.len() + 4 * 11);
    }

    #[test]
    fn falcon_1024_manifest_has_four_groups_and_5127_cells_per_record() {
        for records in [8_usize, 1 << 14] {
            let manifest = wide_manifest(records);
            assert_eq!(manifest.degree(), FalconDegree::Falcon1024);
            assert_eq!(manifest.group_count(), 4);
            let cells: Vec<_> = manifest
                .groups()
                .iter()
                .map(|group| group.logical_cells)
                .collect();
            assert_eq!(
                cells,
                vec![1_024 * records, 4 * records, 1_024 * records, records]
            );
            assert_eq!(
                manifest
                    .groups()
                    .iter()
                    .map(|group| group.logical_polynomials)
                    .collect::<Vec<_>>(),
                vec![1, 1, 4, 3]
            );
            assert_eq!(manifest.total_logical_cells(), 5_127 * records);
        }
    }

    #[test]
    fn binding_bytes_separate_every_public_choice() {
        let base = target_manifest().binding_bytes();
        let variants = [
            CommitmentManifest::for_batch(16),
            CommitmentManifest::for_batch_of(FalconDegree::Falcon1024, 8),
        ];
        for variant in variants {
            assert_ne!(variant.expect("valid manifest").binding_bytes(), base);
        }
        assert_eq!(target_manifest().binding_bytes(), base);
        let wide = wide_manifest(8).binding_bytes();
        assert_eq!(&wide[..2], &2_u16.to_le_bytes());
        assert_eq!(wide[2], 10);
        assert_eq!(&wide[3..7], &6_u32.to_le_bytes());
        assert_eq!(&wide[7..11], &1_024_u32.to_le_bytes());
        assert_eq!(base[2], 9, "the parameter-set byte separates the sets");
        assert_ne!(wide, wide_manifest(16).binding_bytes());
    }

    #[test]
    fn arbitrary_batches_bind_both_live_and_padded_counts() {
        let manifest = CommitmentManifest::for_batch(11).expect("arbitrary-batch manifest");
        assert_eq!(manifest.records(), 11);
        assert_eq!(manifest.padded_records(), 16);
        assert_eq!(manifest.groups()[0].logical_cells, 512 * 16);
        // The binding separates a live batch from its padded power of two.
        let full = CommitmentManifest::for_batch(16).expect("full manifest");
        assert_ne!(manifest.binding_bytes(), full.binding_bytes());
        let wide = wide_manifest(11);
        assert_eq!(wide.padded_records(), 16);
        assert_eq!(wide.groups()[0].logical_cells, 1_024 * 16);
    }

    #[test]
    fn single_record_uses_minimum_manifest_shape() {
        let single = CommitmentManifest::for_batch(1).unwrap();
        let minimum = CommitmentManifest::for_batch(8).unwrap();
        assert_eq!(single.padded_records(), minimum.padded_records());
        assert_eq!(single.groups(), minimum.groups());
        assert_eq!(single.projection_profile(), minimum.projection_profile());
        assert_ne!(single.binding_bytes(), minimum.binding_bytes());
    }

    #[test]
    fn malformed_batches_reject() {
        for records in [0, (1 << 16) + 1, 1 << 17] {
            assert!(CommitmentManifest::for_batch(records).is_err());
            assert!(CommitmentManifest::for_batch_of(FalconDegree::Falcon1024, records).is_err());
        }
    }
}
