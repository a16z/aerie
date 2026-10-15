//! Prover for stage one's weighted basis-16 product relations.
//!
//! Coefficient digits share one `ProductGroup`, so its compact lookup and
//! source scan are reused across `a1` and `a2`. Falcon-1024's signature digit
//! uses a separate group on its shorter suffix domain. All source tables stay
//! borrowed as prepared coded sources (packed pair codes or direct cells)
//! until the product kernel folds them.

use crate::error::AggregationError;
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

    pub(super) fn round(&mut self, stage_round: usize) -> Result<[E; 3], AggregationError> {
        if stage_round < self.inactive {
            return Ok([self.claim * self.half, E::zero(), E::zero()]);
        }
        let (eq_low, eq_high) = self.equality.remaining_eq_tables();
        let [q1, q2] = self
            .group
            .round_coefficients(eq_low, eq_high)
            .map_err(AggregationError::backend)?;
        let tau = self.equality.current_tau();
        let prefix_scale = self.equality.current_scalar() * self.inactive_scale;
        let q0_scaled = self.claim - prefix_scale * tau * (q1 + q2);
        let linear = tau + tau - E::one();
        Ok([
            (E::one() - tau) * q0_scaled,
            prefix_scale * ((E::one() - tau) * q2 + linear * q1),
            prefix_scale * linear * q2,
        ])
    }

    pub(super) fn bind(
        &mut self,
        stage_round: usize,
        challenge: E,
        message: [E; 3],
    ) -> Result<(), AggregationError> {
        if stage_round < self.inactive {
            self.claim = evaluate_compressed(message, self.claim, challenge);
            self.inactive_scale *= self.half;
            return Ok(());
        }
        self.group
            .bind(challenge)
            .map_err(AggregationError::backend)?;
        self.claim = evaluate_compressed(message, self.claim, challenge);
        self.equality.bind(challenge);
        Ok(())
    }

    pub(super) fn terminal_pairs(&self) -> Result<Vec<[E; 2]>, AggregationError> {
        let leaves = self.group.terminal_leaves()?;
        if !leaves.len().is_multiple_of(2) {
            return Err(AggregationError::internal(
                "basis16 product returned an odd number of child leaves",
            ));
        }
        Ok(leaves
            .chunks_exact(2)
            .map(|pair| [pair[0], pair[1]])
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhausted_group_errors_preserve_claim_and_equality_state() {
        let digits = [-8, -1, 2, 7];
        let coded = CodedSource::new(&digits, -8, 7).unwrap();
        for inactive in [0, 10] {
            let mut prover = Prover::new(
                vec![&coded],
                vec![E::one()],
                &[E::from_u64(3), E::from_u64(5)],
                inactive,
            )
            .unwrap();
            for stage_round in 0..=inactive {
                let message = prover.round(stage_round).unwrap();
                if stage_round < inactive {
                    assert_eq!(message, [prover.claim * prover.half, E::zero(), E::zero()]);
                }
                prover.bind(stage_round, E::from_u64(2), message).unwrap();
            }
            // Complete the group while equality still has an active coordinate.
            prover.group.bind(E::from_u64(7)).unwrap();
            let claim = prover.claim;
            let scale = prover.inactive_scale;
            let scalar = prover.equality.current_scalar();
            let tau = prover.equality.current_tau();
            let (first, second) = prover.equality.remaining_eq_tables();
            let equality_tables = (first.to_vec(), second.to_vec());
            let terminal = prover.terminal_pairs().unwrap();
            for _ in 0..2 {
                assert!(matches!(
                    prover.round(inactive + 1),
                    Err(AggregationError::Backend(_))
                ));
                assert!(matches!(
                    prover.bind(inactive + 1, E::from_u64(11), [E::one(); 3]),
                    Err(AggregationError::Backend(_))
                ));
                assert_eq!(prover.claim, claim);
                assert_eq!(prover.inactive_scale, scale);
                assert_eq!(prover.equality.current_scalar(), scalar);
                assert_eq!(prover.equality.current_tau(), tau);
                assert_eq!(
                    prover.equality.remaining_eq_tables(),
                    (equality_tables.0.as_slice(), equality_tables.1.as_slice())
                );
                assert_eq!(prover.terminal_pairs().unwrap(), terminal);
            }
        }
    }
}
