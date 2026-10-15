//! Lookup-table construction for the range-source compact prefix.

use jolt_field::{Fold, Ring};

use super::kernel_common::decode_quartet_indices;
use crate::substrate::OpeningField as E;

/// Build the first-bind lookup over zero-based alphabet-index pairs.
pub(super) fn build_pair(pattern_base: usize, challenge: E) -> Vec<E> {
    let context = E::precompute(challenge);
    (0..pattern_base.pow(2))
        .map(|code| {
            E::fold_one(
                &context,
                E::from_i64((code / pattern_base) as i64),
                E::from_i64((code % pattern_base) as i64),
            )
        })
        .collect()
}

/// Build the two-bind lookup over zero-based alphabet-index quartets.
pub(super) fn build_quartet(
    pattern_base: usize,
    first_challenge: E,
    second_challenge: E,
) -> Vec<E> {
    let quartet_patterns = pattern_base.pow(4);
    let first_context = E::precompute(first_challenge);
    let second_context = E::precompute(second_challenge);
    let mut lut = Vec::with_capacity(quartet_patterns);
    for pattern in 0..quartet_patterns {
        let slots: [E; 4] =
            decode_quartet_indices(pattern, pattern_base).map(|slot| E::from_i64(slot as i64));
        let left = E::fold_one(&first_context, slots[0], slots[1]);
        let right = E::fold_one(&first_context, slots[2], slots[3]);
        lut.push(E::fold_one(&second_context, left, right));
    }
    lut
}
