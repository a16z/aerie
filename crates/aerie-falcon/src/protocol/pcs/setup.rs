//! Shape-dependent PCS setup and catalog-derived commitment profiles.

use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use akita_config::{CommitmentConfig, SetupRequirements, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::{
    GroupCommitPhaseParams, GroupOpenPhaseParams, OpeningScheduleSelection, PolynomialGroupLayout,
    ScheduleLookupKey, SetupPrefixSlotId,
};
use akita_serialization::Valid;
use akita_types::{
    derive_public_matrix_prefix, sample_akita_setup_seed, AkitaExpandedSetup, AkitaSetupDescriptor,
    AkitaVerifierSetup, SetupPrefixVerifierRegistry,
};
use akita_verifier::AkitaVerifier;

#[cfg(feature = "internals")]
use super::verifier::verifier_instance;
use super::Error;
use crate::protocol::manifest::{CommitmentManifest, PcsFamily};
use crate::protocol::zero_extension::pcs_length;
#[cfg(feature = "internals")]
use crate::protocol::{instance, public_inputs::PublicInputs};
use crate::protocol::{AggregationParams, FalconDegree};
use crate::substrate::{BaseField, Width14Config, Width4Config};
use crate::UnsupportedBatch;

/// One Akita setup requirements envelope for both commitment families.
///
/// The width-4 and width-14 groups commit on the same family-agnostic
/// backend, so the setup is sized by the union of both catalogs'
/// requirements rather than by either catalog alone.
#[doc(hidden)]
pub fn shared_setup_requirements(
    max_physical_vars: usize,
    polys: usize,
) -> Result<SetupRequirements<BaseField>, AkitaError> {
    validate_seed_only_catalogs()?;
    let requirements =
        SetupRequirements::from_catalog(Width4Config::catalog()?, max_physical_vars, polys)?
            .union(SetupRequirements::from_catalog(
                Width14Config::catalog()?,
                max_physical_vars,
                polys,
            )?)?;
    validate_seed_only_metadata(false, std::iter::empty(), requirements.prefix_slot_ids())?;
    Ok(requirements)
}

/// Reject recursive setup policy, marked groups, or required prefix commitments
/// before allocating the seed-derived public matrix.
fn validate_seed_only_metadata<'a>(
    recursive_setup: bool,
    groups: impl IntoIterator<Item = &'a GroupOpenPhaseParams>,
    prefix_slot_ids: &[SetupPrefixSlotId],
) -> Result<(), AkitaError> {
    if recursive_setup
        || groups
            .into_iter()
            .any(|group| group.setup_natural_len.is_some())
        || !prefix_slot_ids.is_empty()
    {
        return Err(AkitaError::InvalidSetup(
            "Aerie seed-only setup requires direct-only catalogs without setup prefixes".to_owned(),
        ));
    }
    Ok(())
}

fn validate_seed_only_catalog<Cfg: CommitmentConfig>(
    catalog: &TrustedScheduleCatalog<Cfg>,
) -> Result<(), AkitaError> {
    validate_seed_only_metadata(
        Cfg::recursive_setup_planning(),
        catalog.rows().flat_map(|row| {
            let schedule = row.schedule();
            std::iter::once(&schedule.root)
                .chain(&schedule.recursive_folds)
                .flat_map(|fold| fold.params.groups())
        }),
        &[],
    )
}

fn validate_seed_only_catalogs() -> Result<(), AkitaError> {
    static VALIDATION: OnceLock<Result<(), AkitaError>> = OnceLock::new();
    VALIDATION
        .get_or_init(|| {
            validate_seed_only_catalog(Width4Config::catalog()?)?;
            validate_seed_only_catalog(Width14Config::catalog()?)
        })
        .as_ref()
        .copied()
        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))
}

/// The public verifier setup for aggregates of parameter set `P` at one
/// live record count. Holds no prover state; a verifier builds it with
/// [`FalconVerifierSetup::new`] or borrows it from the producer setup.
pub struct FalconVerifierSetup<P: AggregationParams> {
    pub(super) records: usize,
    pub(super) akita: AkitaVerifier<Width4Config>,
    params: PhantomData<fn() -> P>,
}

impl<P: AggregationParams> FalconVerifierSetup<P> {
    /// Whether the generated PCS catalog supports this live record count.
    #[must_use]
    pub fn supports(records: usize) -> bool {
        verifier_group_profiles_of(P::DEGREE_ID, records).is_ok()
    }

    /// Build the public verifier setup for one live record count.
    ///
    /// Derives the shared public matrix from Akita's fixed setup seed and
    /// prepares the admitted catalog rows' terminal caches.
    pub fn new(records: usize) -> Result<Self, Error> {
        let manifest = CommitmentManifest::for_batch_of(P::DEGREE_ID, records)?;
        // Validate all commitment group shapes and their opening schedule against the catalogs.
        verifier_group_profiles_of(P::DEGREE_ID, records)?;
        let cells = group_cells_of(&manifest);
        let requirements = shared_setup_requirements(
            max_physical_vars(&cells),
            total_group_polynomials(&manifest),
        )?;
        let descriptor = AkitaSetupDescriptor {
            max_num_vars: requirements.max_num_vars(),
            max_num_batched_polys: requirements.max_num_batched_polys(),
            num_field_elements: requirements.matrix_capacity().num_field_elements,
            setup_seed: sample_akita_setup_seed(),
        };
        descriptor.check().map_err(|error| {
            AkitaError::InvalidSetup(format!("setup seed validation failed: {error}"))
        })?;
        let matrix = derive_public_matrix_prefix::<BaseField>(
            descriptor.num_field_elements,
            &descriptor.setup_seed,
        );
        let registry = SetupPrefixVerifierRegistry::new(descriptor.setup_seed.clone());
        let expanded = Arc::new(
            AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(descriptor, matrix),
        );
        Self::from_derived_setup(records, AkitaVerifierSetup::from_parts(expanded, registry)?)
    }

    /// Prepare one verifier over an already-derived, full-capacity setup.
    #[doc(hidden)]
    pub fn from_derived_setup(
        records: usize,
        setup: AkitaVerifierSetup<BaseField>,
    ) -> Result<Self, Error> {
        Ok(Self {
            records,
            akita: AkitaVerifier::<Width4Config>::new(setup, Width4Config::catalog()?.clone())?,
            params: PhantomData,
        })
    }

    /// The live record count this setup was built for.
    #[must_use]
    pub const fn records(&self) -> usize {
        self.records
    }

    /// Backend verifier access for setup equivalence tests.
    #[cfg(feature = "internals")]
    #[doc(hidden)]
    #[must_use]
    pub fn akita(&self) -> &AkitaVerifier<Width4Config> {
        &self.akita
    }

    /// Reconstruct the canonical instance for adversarial transcript tests.
    #[cfg(feature = "internals")]
    #[doc(hidden)]
    pub fn transcript_instance<C: Copy + Sync + Into<i64>>(
        &self,
        public: &PublicInputs<'_, C>,
    ) -> Result<[u8; instance::INSTANCE_BYTES], Error> {
        let degree = public.degree;
        let manifest = CommitmentManifest::for_batch_of(degree, public.records)?;
        let (_, _, selection) = verifier_group_profiles_of(degree, public.records)?;
        verifier_instance(public, &manifest, selection, &self.akita)
    }
}

/// Exact per-polynomial logical cell counts of every group in canonical order.
#[doc(hidden)]
#[must_use]
pub fn group_cells_of(manifest: &CommitmentManifest) -> Vec<usize> {
    manifest
        .groups()
        .iter()
        .map(|descriptor| descriptor.logical_cells)
        .collect()
}

#[doc(hidden)]
#[must_use]
pub fn max_physical_vars(logical_cells: &[usize]) -> usize {
    logical_cells
        .iter()
        .map(|&cells| pcs_length(cells).trailing_zeros() as usize)
        .max()
        .expect("a manifest has at least one group")
}

#[doc(hidden)]
#[must_use]
pub fn total_group_polynomials(manifest: &CommitmentManifest) -> usize {
    manifest
        .groups()
        .iter()
        .map(|descriptor| descriptor.logical_polynomials)
        .sum()
}

/// Verifier-derived group profiles and the public schedule selection,
/// computed from the statement shape and the generated catalogs alone
/// ("verifier-derivable; never prover-supplied"). Every group but the last
/// is a precommitted group in manifest order; the last is the final group.
#[doc(hidden)]
pub fn verifier_group_profiles_of(
    degree: FalconDegree,
    records: usize,
) -> Result<
    (
        Vec<GroupCommitPhaseParams>,
        GroupCommitPhaseParams,
        OpeningScheduleSelection,
    ),
    Error,
> {
    let manifest = CommitmentManifest::for_batch_of(degree, records)?;
    let groups = manifest.groups();
    let layout = |descriptor: &crate::protocol::manifest::GroupDescriptor| {
        PolynomialGroupLayout::new(
            pcs_length(descriptor.logical_cells).trailing_zeros() as usize,
            descriptor.logical_polynomials,
        )
    };
    let (final_group, precommitted) = groups
        .split_last()
        .ok_or_else(|| Error::internal("commitment manifest is empty"))?;
    let mut precommits = Vec::with_capacity(precommitted.len());
    for descriptor in precommitted {
        let profile = match descriptor.family {
            PcsFamily::Width14 => {
                Width14Config::commitment_profile(layout(descriptor)).map_err(|_| {
                    Error::UnsupportedBatch(UnsupportedBatch::Shape {
                        degree: degree.degree(),
                        records,
                    })
                })?
            }
            PcsFamily::Width4 => {
                Width4Config::commitment_profile(layout(descriptor)).map_err(|_| {
                    Error::UnsupportedBatch(UnsupportedBatch::Shape {
                        degree: degree.degree(),
                        records,
                    })
                })?
            }
        };
        precommits.push(profile);
    }
    if final_group.family != PcsFamily::Width4 {
        return Err(Error::internal(
            "final group uses an incompatible commitment family",
        ));
    }
    let key = ScheduleLookupKey {
        final_group: layout(final_group),
        precommitteds: precommits.clone(),
    };
    let row = Width4Config::catalog()?.resolve_key(&key).map_err(|_| {
        Error::UnsupportedBatch(UnsupportedBatch::Shape {
            degree: degree.degree(),
            records,
        })
    })?;
    Ok((precommits, row.profiles().final_group, row.selection()))
}

/// Shaped wire context for the codec: the manifest's commitment coefficient
/// counts in canonical order, derived from the statement shape and the
/// generated catalogs.
#[doc(hidden)]
pub fn commitment_coefficient_counts_of(
    degree: FalconDegree,
    records: usize,
) -> Result<Vec<usize>, Error> {
    let (precommits, final_profile, _) = verifier_group_profiles_of(degree, records)?;
    let mut coefficient_counts = Vec::with_capacity(precommits.len() + 1);
    for profile in precommits.iter().chain(std::iter::once(&final_profile)) {
        coefficient_counts.push(crate::substrate::commitment_coefficients(profile)?);
    }
    Ok(coefficient_counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_only_setup_rejects_prefix_metadata() {
        let catalog = Width4Config::catalog().expect("width4 catalog");
        let schedule = catalog.rows().next().expect("catalog row").schedule();
        let ordinary = schedule.root.params.final_group();
        let prefix_id = SetupPrefixSlotId {
            natural_len: 1,
            commitment_profile: ordinary.profile,
        };
        assert!(validate_seed_only_metadata(false, [&ordinary], &[]).is_ok());
        assert!(matches!(
            validate_seed_only_metadata(false, [&ordinary], &[prefix_id]),
            Err(AkitaError::InvalidSetup(_))
        ));
        let marked = GroupOpenPhaseParams {
            setup_natural_len: Some(1),
            ..ordinary
        };
        assert!(matches!(
            validate_seed_only_metadata(false, [&marked], &[]),
            Err(AkitaError::InvalidSetup(_))
        ));
        assert!(matches!(
            validate_seed_only_metadata(true, [&ordinary], &[]),
            Err(AkitaError::InvalidSetup(_))
        ));
    }
}
