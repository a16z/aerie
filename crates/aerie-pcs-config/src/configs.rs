//! Aerie's fp64 commitment configuration presets.
//!
//! These presets inherit Akita fp64 `Dense`'s adaptive ring-dimension policy.
//! The planner searches exact A/B/D dimension tuples for the first two fold
//! levels (A over 64 through 2048, B and D over 64 through 256), then uses a
//! monotone sequence of uniform dimensions from {64, 128}. Each trusted catalog
//! row records the dimensions it selected. Preset names describe the field and
//! source contract; the implementation below declares the ring policy.
//!
//! Akita enforces each declared signed coefficient interval at commitment
//! admission. Aerie's separate range proofs establish the application relation;
//! producer admission does not replace those verifier predicates.

use akita_config::proof_optimized::fp64::{Dense, ExtensionField, Field};
use akita_config::{CommitmentConfig, RingDimensionSchedule};
use akita_error::AkitaError;
use akita_params::{DecompositionParams, SisModulusProfileId};

/// Fp64 preset for signed width-14 root sources in `[-8192, 8191]`.
///
/// A/source decomposition uses Dense's inner-basis search policy, clipped by
/// Akita to log bases 3 through 14 for these raw coefficients. The catalog
/// selects log basis/digit count 5/3 at 14 variables, 4/4 at 17, 8/2 at
/// 18–19, and 14/1 at the other catalog shapes (13 through 26 variables),
/// independently of the opening B/D decomposition (nominal log basis 3,
/// search range 3 through 6).
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp64SignedWidth14;

/// Fp64 preset for signed width-4 sources in `[-8, 7]`.
///
/// A/source decomposition uses exactly one balanced base-16 digit (inner log
/// basis 4). This is independent of the opening B/D decomposition, whose
/// nominal log basis is 3 and whose search range is 3 through 6.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp64SignedWidth4;

macro_rules! impl_aerie_preset {
    ($cfg:ty, $log_commit_bound:expr, $table:ident, $inner_basis_range:expr) => {
        impl $cfg {
            /// Select the scalar producer profile from the trusted family.
            pub fn commitment_profile(
                layout: akita_params::PolynomialGroupLayout,
            ) -> Result<akita_params::GroupCommitPhaseParams, AkitaError> {
                Ok(Self::catalog()?
                    .resolve_key(&akita_params::ScheduleLookupKey::single(layout))?
                    .profiles()
                    .final_group)
            }
            /// Load and audit the application-owned schedule family once.
            pub fn catalog(
            ) -> Result<&'static akita_config::TrustedScheduleCatalog<Self>, AkitaError> {
                #[cfg(feature = "schedules-fp64-aerie")]
                {
                    static CATALOG: std::sync::OnceLock<
                        Result<akita_config::TrustedScheduleCatalog<$cfg>, AkitaError>,
                    > = std::sync::OnceLock::new();
                    CATALOG
                        .get_or_init(|| {
                            let bytes = include_bytes!(concat!(
                                env!("CARGO_MANIFEST_DIR"),
                                "/src/schedules/",
                                stringify!($table),
                                ".aks"
                            ));
                            akita_config::TrustedScheduleCatalog::from_artifact_bytes(bytes)
                        })
                        .as_ref()
                        .map_err(|error| AkitaError::InvalidSetup(error.to_string()))
                }
                #[cfg(not(feature = "schedules-fp64-aerie"))]
                {
                    Err(AkitaError::InvalidSetup(
                        "aerie fp64 schedule catalogs are disabled; enable the \
                         `schedules-fp64-aerie` feature outside offline regeneration"
                            .to_owned(),
                    ))
                }
            }
        }
        impl CommitmentConfig for $cfg {
            type Field = Field;
            type ExtField = ExtensionField;

            const RING_DIMENSION_SCHEDULE: RingDimensionSchedule = Dense::RING_DIMENSION_SCHEDULE;

            fn decomposition() -> DecompositionParams {
                DecompositionParams {
                    log_basis: 3,
                    log_commit_bound: $log_commit_bound,
                    log_open_bound: Some(64),
                }
            }

            fn ring_challenge_config(
                d: usize,
            ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
                Dense::ring_challenge_config(d)
            }

            fn sis_modulus_profile() -> SisModulusProfileId {
                SisModulusProfileId::Q64Offset59
            }

            fn opening_basis_range() -> (u32, u32) {
                (3, 6)
            }

            fn inner_basis_range() -> (u32, u32) {
                $inner_basis_range
            }

            fn committed_source_class() -> akita_params::sis::CommittedSourceClass {
                akita_params::sis::CommittedSourceClass::BalancedSignedDigit
            }

            fn schedule_family_name() -> &'static str {
                stringify!($table)
            }
        }
    };
}

impl_aerie_preset!(
    Fp64SignedWidth14,
    14,
    aerie_fp64_signed_width14,
    Dense::inner_basis_range()
);
impl_aerie_preset!(Fp64SignedWidth4, 4, aerie_fp64_signed_width4, (4, 4));

#[cfg(all(test, feature = "schedules-fp64-aerie"))]
mod tests {
    use super::*;
    use akita_params::PolynomialGroupLayout;

    #[test]
    fn source_group_has_an_independent_width4_profile() {
        let layout = PolynomialGroupLayout::new(18, 4);
        let profile =
            Fp64SignedWidth4::commitment_profile(layout).expect("native source-group producer row");
        assert_eq!(profile.group, layout);
    }
}

#[cfg(all(test, not(feature = "schedules-fp64-aerie")))]
mod disabled_tests {
    use super::*;

    #[test]
    fn catalog_reports_the_disabled_regeneration_mode() {
        let error = Fp64SignedWidth4::catalog()
            .err()
            .expect("catalog feature is disabled");
        assert!(error.to_string().contains("schedules-fp64-aerie"));
    }
}

#[cfg(test)]
mod source_width_tests {
    use super::*;
    use akita_params::sis::num_digits_open;

    #[test]
    fn source_basis_policy_is_independent_of_opening_basis() {
        assert_eq!(
            Fp64SignedWidth14::inner_basis_range(),
            Dense::inner_basis_range()
        );
        assert_eq!(Fp64SignedWidth4::inner_basis_range(), (4, 4));
        for params in [
            Fp64SignedWidth14::decomposition(),
            Fp64SignedWidth4::decomposition(),
        ] {
            params.validate().expect("valid decomposition");
            assert_eq!(params.log_basis, 3);
            assert_eq!(num_digits_open(params), 22);
        }
        assert_eq!(Fp64SignedWidth14::opening_basis_range(), (3, 6));
        assert_eq!(Fp64SignedWidth4::opening_basis_range(), (3, 6));
    }

    #[cfg(feature = "schedules-fp64-aerie")]
    #[test]
    fn selected_source_digits_cover_every_honest_width_endpoint() {
        use akita_params::PolynomialGroupLayout;
        let main_contract =
            Fp64SignedWidth14::committed_source_contract().expect("width14 contract");
        let digit_contract =
            Fp64SignedWidth4::committed_source_contract().expect("width4 contract");
        for num_vars in 13..=26 {
            let profile =
                Fp64SignedWidth14::commitment_profile(PolynomialGroupLayout::new(num_vars, 1))
                    .expect("width14 producer profile");
            let digits = profile.inner.digits;
            assert_eq!(
                main_contract.accepted_bounds(digits.log_basis, digits.num_digits),
                (Some(8192), Some(8191))
            );
            let profile =
                Fp64SignedWidth4::commitment_profile(PolynomialGroupLayout::new(num_vars, 4))
                    .expect("width4 producer profile");
            let digits = profile.inner.digits;
            assert_eq!((digits.log_basis, digits.num_digits), (4, 1));
            assert_eq!(
                digit_contract.accepted_bounds(digits.log_basis, digits.num_digits),
                (Some(8), Some(7))
            );
        }
    }
}
