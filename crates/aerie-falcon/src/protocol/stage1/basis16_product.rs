//! Prover for stage one's weighted basis-16 product relations.
//!
//! Coefficient digits share one `ProductGroup`, so its compact lookup and
//! source scan are reused across `a1` and `a2`. Falcon-1024's signature digit
//! uses a separate group on its shorter suffix domain. All source tables stay
//! borrowed as prepared coded sources (packed pair codes or direct cells)
//! until the product kernel folds them.

use crate::protocol::kernels::product_kernel::ProductGroup;
use crate::protocol::source_tables::CodedSource;
use akita_algebra::split_eq::GruenSplitEq;
use akita_error::AkitaError;
use jolt_field::{Field as _, One as _, Ring as _, Zero as _};

use crate::substrate::OpeningField as E;

use super::evaluate_compressed;

pub(super) struct Prover<'a> {
    group: ProductGroup<'a>,
    equality: GruenSplitEq<E>,
    inactive: usize,
    inactive_scale: E,
    claim: E,
    half: E,
}

impl<'a> Prover<'a> {
    pub(super) fn new(
        sources: Vec<&CodedSource<'a>>,
        source_weights: Vec<E>,
        tau: &[E],
        inactive: usize,
    ) -> Result<Self, AkitaError> {
        let rounds = tau.len();
        let group = ProductGroup::new(sources, source_weights, rounds)?;
        Self::from_group(group, tau, inactive)
    }

    fn from_group(group: ProductGroup<'a>, tau: &[E], inactive: usize) -> Result<Self, AkitaError> {
        Ok(Self {
            group,
            equality: GruenSplitEq::new(tau)?,
            inactive,
            inactive_scale: E::one(),
            claim: E::zero(),
            half: E::from_u64(2).inverse().expect("two is invertible"),
        })
    }

    pub(super) fn round(&mut self, stage_round: usize) -> [E; 3] {
        if stage_round < self.inactive {
            return [self.claim * self.half, E::zero(), E::zero()];
        }
        let (eq_low, eq_high) = self.equality.remaining_eq_tables();
        let [q1, q2] = self.group.round_coefficients(eq_low, eq_high);
        let tau = self.equality.current_tau();
        let prefix_scale = self.equality.current_scalar() * self.inactive_scale;
        let q0_scaled = self.claim - prefix_scale * tau * (q1 + q2);
        let linear = tau + tau - E::one();
        [
            (E::one() - tau) * q0_scaled,
            prefix_scale * ((E::one() - tau) * q2 + linear * q1),
            prefix_scale * linear * q2,
        ]
    }

    pub(super) fn bind(&mut self, stage_round: usize, challenge: E, message: [E; 3]) {
        self.claim = evaluate_compressed(message, self.claim, challenge);
        if stage_round < self.inactive {
            self.inactive_scale *= self.half;
            return;
        }
        self.group.bind(challenge);
        self.equality.bind(challenge);
    }

    pub(super) fn terminal_pairs(&self) -> Result<Vec<[E; 2]>, AkitaError> {
        let leaves = self.group.terminal_leaves()?;
        if !leaves.len().is_multiple_of(2) {
            return Err(AkitaError::InvalidInput(
                "basis16 product returned an odd number of child leaves".into(),
            ));
        }
        Ok(leaves
            .chunks_exact(2)
            .map(|pair| [pair[0], pair[1]])
            .collect())
    }
}
