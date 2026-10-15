//! Witness-soundness invariant tests.

use std::sync::OnceLock;

use aerie_falcon::{Falcon1024, Falcon512, FalconParams};
use aerie_falcon_native::params::FixedArray;
use aerie_falcon_statement::DirectParams;
use aerie_falcon_witness::DirectRow;
use aerie_invariants::invariants::{WitnessInput, WitnessSoundness, WitnessTamper};
use aerie_invariants::{CheckResult, Invariant};

mod fast;
mod pcs;

/// Share the catalog-backed eight-record PCS setups across witness candidates.
fn invariant() -> &'static WitnessSoundness {
    static INVARIANT: OnceLock<WitnessSoundness> = OnceLock::new();
    INVARIANT.get_or_init(WitnessSoundness::default)
}

fn check(parameter_set: u8, tampers: &[WitnessTamper]) -> CheckResult {
    invariant().check(&WitnessInput {
        parameter_set: Some(parameter_set),
        tampers: tampers.to_vec(),
    })
}

fn fixture(parameter_set: u8) -> aerie_prover::FalconWitness {
    fn build<P: DirectParams>() -> aerie_prover::FalconWitness {
        let rows = (0..8)
            .map(|seed| DirectRow::<P>::synthetic(seed + 263).unwrap())
            .collect::<Vec<_>>();
        aerie_prover::FalconWitness::from_rows(&rows)
    }
    match parameter_set {
        9 => build::<Falcon512>(),
        10 => build::<Falcon1024>(),
        _ => panic!("unsupported fixture parameter set"),
    }
}

fn original_a0(parameter_set: u8, index: usize) -> i64 {
    fn original<P: FalconParams>(batch: &aerie_prover::FalconWitness, index: usize) -> i64 {
        let start = index / P::DEGREE * P::DEGREE;
        let s2 = P::Array::<i16>::from_slice(&batch.s2[start..start + P::DEGREE]).unwrap();
        let row =
            aerie_falcon_witness::native::encodability::FourCellEncodabilityWitness::<P>::build(
                &s2,
            )
            .unwrap();
        i64::from(row.a0().as_ref()[index - start])
    }
    let batch = fixture(parameter_set);
    match parameter_set {
        9 => original::<Falcon512>(&batch, index),
        10 => original::<Falcon1024>(&batch, index),
        _ => panic!("unsupported fixture parameter set"),
    }
}
