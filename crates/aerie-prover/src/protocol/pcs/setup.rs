//! Producer setup, matrix sharing, and cache warming.

use aerie_falcon_witness::prepare_direct_witness_convolution;
#[cfg(test)]
use akita_config::SetupRequirements;
#[cfg(test)]
use akita_error::AkitaError;
use akita_params::{PrecommittedGroupProfiles, ScheduleLookupKey};
use akita_pcs::{new_prover_setup, AkitaProverSetup, CpuBackend, GroupContext};
#[cfg(test)]
use akita_verifier::AkitaVerifier;
use std::sync::Arc;

use super::group::group_polynomial;
use super::{Error, FalconVerifierSetup};
use crate::protocol::manifest::{CommitmentManifest, PcsFamily};
use crate::protocol::{AggregationParams, FalconDegree};
use crate::substrate::{BaseField, GroupOpeningScheme, OpeningField, Width14Config, Width4Config};
use aerie_falcon::protocol::pcs::{
    group_cells_of, max_physical_vars, shared_setup_requirements, total_group_polynomials,
    verifier_group_profiles_of,
};

pub(crate) struct ProverCore {
    pub(super) degree: FalconDegree,
    pub(super) records: usize,
    #[cfg(test)]
    pub(super) instance_mutation: Option<usize>,
    #[cfg(test)]
    pub(super) session_mutation: bool,
    pub(super) manifest: CommitmentManifest,
    pub(super) logical_cells: Vec<usize>,
    /// The catalog profiles of every precommitted group in manifest order.
    /// They are public shape data, so the final group commits against them
    /// without waiting for the precommits.
    pub(super) precommitted: PrecommittedGroupProfiles,
    pub(super) prover: AkitaProverSetup<BaseField>,
    /// Commits every group, each under its own family's catalog, and proves
    /// the batched opening.
    pub(super) backend: CpuBackend<BaseField, OpeningField>,
}

impl ProverCore {
    #[tracing::instrument(skip_all, fields(records), name = "v1.setup.total")]
    pub(super) fn new(degree: FalconDegree, records: usize) -> Result<Self, Error> {
        tracing::info_span!("v1.setup.witness_ntt").in_scope(prepare_direct_witness_convolution);
        let manifest = CommitmentManifest::for_batch_of(degree, records)?;
        let (precommitteds, final_group, _) = verifier_group_profiles_of(degree, records)?;
        let logical_cells = group_cells_of(&manifest);
        let max_physical_vars = max_physical_vars(&logical_cells);
        let requirements =
            shared_setup_requirements(max_physical_vars, total_group_polynomials(&manifest))?;
        let prover = new_prover_setup(&requirements)?;
        let backend = CpuBackend::new(Arc::clone(&prover.expanded))?;
        let schedule = Width4Config::catalog()?.resolve_key(&ScheduleLookupKey {
            final_group: final_group.group,
            precommitteds: precommitteds.clone(),
        })?;
        backend.prewarm(schedule.schedule())?;
        let precommitted = PrecommittedGroupProfiles::from_profiles(precommitteds)?;
        tracing::info_span!("v1.setup.compression_caches")
            .in_scope(|| warm_compression_caches(&backend, &manifest, &precommitted))?;
        Ok(Self {
            degree,
            records,
            #[cfg(test)]
            instance_mutation: None,
            #[cfg(test)]
            session_mutation: false,
            manifest,
            logical_cells,
            precommitted,
            prover,
            backend,
        })
    }

    /// The public verifier derived from this prover setup.
    #[cfg(test)]
    pub(super) fn verifier(&self) -> Result<AkitaVerifier<Width4Config>, AkitaError> {
        let pcs = GroupOpeningScheme::new(Width4Config::catalog()?.clone());
        pcs.verifier(pcs.setup_verifier(&self.prover)?)
    }
}

/// Reusable, shape-dependent prover and verifier state for one live count.
pub struct FalconAggregateSetup<P: AggregationParams> {
    pub(super) core: ProverCore,
    verifier: FalconVerifierSetup<P>,
}

impl<P: AggregationParams> FalconAggregateSetup<P> {
    /// Whether the generated PCS catalog supports this live record count.
    /// Uses the same padded geometry and schedule lookup as setup and the
    /// verifier, without allocating or preparing prover state.
    #[must_use]
    pub fn supports(records: usize) -> bool {
        FalconVerifierSetup::<P>::supports(records)
    }

    /// Build reusable prover and verifier state for one live record count.
    pub fn new(records: usize) -> Result<Self, Error> {
        let core = ProverCore::new(P::DEGREE_ID, records)?;
        let pcs = GroupOpeningScheme::new(Width4Config::catalog()?.clone());
        let verifier =
            FalconVerifierSetup::from_derived_setup(records, pcs.setup_verifier(&core.prover)?)?;
        Ok(Self { core, verifier })
    }

    /// The live record count this setup was built for.
    #[must_use]
    pub const fn records(&self) -> usize {
        self.core.records
    }

    /// The public verifier half.
    #[must_use]
    pub const fn verifier(&self) -> &FalconVerifierSetup<P> {
        &self.verifier
    }
}

fn warm_compression_caches(
    backend: &CpuBackend<BaseField, OpeningField>,
    manifest: &CommitmentManifest,
    precommitted: &PrecommittedGroupProfiles,
) -> Result<(), Error> {
    // Manifest order, families and commit contexts: the cache shape depends
    // only on the physical arity, the width and the scheduled commit
    // parameters, so zero tables of each logical length suffice.
    let final_slot = manifest.group_count() - 1;
    for (slot, descriptor) in manifest.groups().iter().enumerate() {
        let cells = vec![0_i16; descriptor.logical_cells];
        let polynomials = (0..descriptor.logical_polynomials)
            .map(|_| group_polynomial(&cells))
            .collect::<Result<Vec<_>, _>>()?;
        let source = backend.import_source(polynomials)?;
        let context = if slot == final_slot {
            GroupContext::scheduler_with_precommitted_groups(precommitted)
        } else {
            GroupContext::scheduler_without_precommitted_groups()
        };
        match descriptor.family {
            PcsFamily::Width14 => {
                backend.commit(Width14Config::catalog()?, &source, context)?;
            }
            PcsFamily::Width4 => {
                backend.commit(Width4Config::catalog()?, &source, context)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerie_falcon_native::{Falcon1024, Falcon512};
    use std::collections::BTreeSet;

    fn compare_verifier_setups<P: AggregationParams>(minimum_power: u32, expected_shapes: usize) {
        let expected: BTreeSet<_> = (minimum_power..=16).map(|power| 1 << power).collect();
        let mut shapes = BTreeSet::new();
        // Sweep support without matrix derivation, including both sides of every
        // padding transition and the 16K and 65,536 live-count boundaries.
        for records in 1..=65_536 {
            assert!(
                FalconVerifierSetup::<P>::supports(records),
                "live count {records}"
            );
            let manifest = CommitmentManifest::for_batch_of(P::DEGREE_ID, records)
                .expect("supported live count manifest");
            shapes.insert(manifest.padded_records());
        }
        assert_eq!(shapes.len(), expected_shapes);
        assert_eq!(shapes, expected);
        for records in [0, 65_537] {
            assert!(!FalconVerifierSetup::<P>::supports(records));
            assert!(matches!(
                FalconVerifierSetup::<P>::new(records),
                Err(Error::UnsupportedBatch(_))
            ));
        }

        for padded in shapes {
            // One comparison pair per distinct padded shape; the minimum uses
            // live count one to exercise its canonical zero padding.
            let records = if padded == 1 << minimum_power {
                1
            } else {
                padded
            };
            let manifest = CommitmentManifest::for_batch_of(P::DEGREE_ID, records)
                .expect("comparison manifest");
            let max_vars = max_physical_vars(&group_cells_of(&manifest));
            let polys = total_group_polynomials(&manifest);
            // The original two-catalog union is the independent sizing oracle.
            let reference_requirements = SetupRequirements::from_catalog(
                Width4Config::catalog().expect("width4 catalog"),
                max_vars,
                polys,
            )
            .expect("width4 requirements")
            .union(
                SetupRequirements::from_catalog(
                    Width14Config::catalog().expect("width14 catalog"),
                    max_vars,
                    polys,
                )
                .expect("width14 requirements"),
            )
            .expect("catalog union");
            assert_eq!(
                shared_setup_requirements(max_vars, polys).expect("shared requirements"),
                reference_requirements
            );
            let prover = new_prover_setup(&reference_requirements).expect("producer provisioning");
            let pcs =
                GroupOpeningScheme::new(Width4Config::catalog().expect("width4 catalog").clone());
            let reference = pcs
                .verifier(
                    pcs.setup_verifier(&prover)
                        .expect("full-capacity conversion"),
                )
                .expect("producer-derived verifier");
            let standalone = FalconVerifierSetup::<P>::new(records).expect("seed-only verifier");
            assert_eq!(standalone.records(), records);
            let actual = &standalone.akita();
            assert_eq!(
                actual.setup().expanded().descriptor(),
                reference.setup().expanded().descriptor()
            );
            let actual_matrix = actual.setup().expanded().shared_matrix().as_field_slice();
            let reference_matrix = reference
                .setup()
                .expanded()
                .shared_matrix()
                .as_field_slice();
            assert_eq!(actual_matrix.len(), reference_matrix.len());
            for (index, (actual, reference)) in
                actual_matrix.iter().zip(reference_matrix).enumerate()
            {
                assert_eq!(
                    actual, reference,
                    "matrix coefficient {index}, padded {padded}"
                );
            }
            assert_eq!(
                actual.setup().prefix_slots(),
                reference.setup().prefix_slots()
            );
            assert!(actual.setup().prefix_slots().is_empty());
            assert!(reference.setup().prefix_slots().is_empty());
            assert_eq!(actual.admitted_rows(), reference.admitted_rows());
            assert_eq!(
                actual.terminal_ntt_cache_bytes(),
                reference.terminal_ntt_cache_bytes()
            );
            assert!(actual.terminal_ntt_cache_bytes() > 0);
            let (_, _, selected) =
                verifier_group_profiles_of(P::DEGREE_ID, records).expect("selected row");
            assert!(actual.admits(selected.row_digest));
            assert!(reference.admits(selected.row_digest));
        }
    }

    #[test]
    fn verifier_setup_matches_prover_derived() {
        compare_verifier_setups::<Falcon512>(3, 14);
        compare_verifier_setups::<Falcon1024>(2, 15);
    }

    fn assert_aggregate_sharing<P: AggregationParams>() {
        let setup = FalconAggregateSetup::<P>::new(1).expect("aggregate setup");
        assert_eq!(setup.records(), 1);
        assert_eq!(setup.verifier().records(), 1);
        assert!(Arc::ptr_eq(
            &setup.core.prover.expanded,
            setup.verifier().akita().setup().expanded()
        ));
    }

    #[test]
    fn aggregate_setup_reuses_prover_matrix() {
        assert_aggregate_sharing::<Falcon512>();
        assert_aggregate_sharing::<Falcon1024>();
    }
}
