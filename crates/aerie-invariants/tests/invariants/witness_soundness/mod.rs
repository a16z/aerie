//! Witness-soundness invariant tests.

use std::sync::OnceLock;

use aerie_invariants::invariants::WitnessSoundness;

mod fast;
mod pcs;

/// Share the fixed eight-record PCS setup across every witness-soundness test.
fn invariant() -> &'static WitnessSoundness {
    static INVARIANT: OnceLock<WitnessSoundness> = OnceLock::new();
    INVARIANT.get_or_init(WitnessSoundness::default)
}

fn fixture() -> aerie_falcon::FalconWitness {
    let rows = (0..8)
        .map(|seed| {
            aerie_falcon_statement::DirectRow::<aerie_falcon_native::Falcon512>::synthetic(
                seed + 263,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    aerie_falcon::FalconWitness::from_rows(&rows)
}

fn original_a0(index: usize) -> i64 {
    let batch = fixture();
    let start =
        index / aerie_falcon_statement::FALCON_DEGREE * aerie_falcon_statement::FALCON_DEGREE;
    let s2 = batch.s2[start..start + aerie_falcon_statement::FALCON_DEGREE]
        .try_into()
        .unwrap();
    let row = aerie_falcon_native::encodability::FourCellEncodabilityWitness::<
        aerie_falcon::Falcon512,
    >::build(s2)
    .unwrap();
    i64::from(row.a0()[index - start])
}
