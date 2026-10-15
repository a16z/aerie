//! Terminal sign-check identity used by the fused stage-three prover.

use crate::substrate::OpeningField as E;
use jolt_field::One;

#[must_use]
pub fn terminal(gamma: E, r2: &[E], r3: &[E], sigma: E) -> E {
    gamma
        * crate::protocol::algebra::eq_evaluation(r2, r3)
        * (sigma - E::one())
        * (sigma + E::one())
}
