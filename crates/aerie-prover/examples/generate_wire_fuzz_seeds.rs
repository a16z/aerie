//! Generate four deterministic wire-fuzzer seed proofs in a new directory.
//!
//! From the workspace root, run:
//!
//! ```text
//! cargo run --locked -p aerie-prover --profile ci \
//!   --example generate_wire_fuzz_seeds \
//!   --features internals,honest-vectors -- OUTPUT_DIRECTORY
//! ```
//!
//! Exactly one output-directory argument is required. Its parent must exist,
//! and the output directory must not exist. Each proof is verified and decoded
//! to check its record count and salt mode before writing its seed file.

use aerie_falcon_witness::honest::honest_aggregation;
use aerie_prover::transport::prove_aggregate_with_mode;
use aerie_prover::wire::{self, SaltMode};
use aerie_prover::{
    verify_aggregate, AggregationParams, Falcon1024, Falcon512, FalconAggregateSetup,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let usage = "usage: generate_wire_fuzz_seeds OUTPUT_DIRECTORY";
    let output = args.next().ok_or(usage)?;
    if args.next().is_some() {
        return Err(usage.into());
    }
    let output = std::path::PathBuf::from(output);
    std::fs::create_dir(&output)?;

    macro_rules! generate_for {
        ($params:ty, $seed:expr, $name:literal) => {{
            let (statement, witness) = honest_aggregation::<$params>(
                1,
                $seed,
                aerie_falcon_witness::honest::SaltMode::Common,
            )
            .expect("build deterministic honest vector");
            let setup =
                FalconAggregateSetup::<$params>::new(1).expect("build minimum-record setup");
            for (mode, suffix) in [(SaltMode::PerRecord, "v1p"), (SaltMode::Common, "v1c")] {
                let proof = prove_aggregate_with_mode(&setup, &statement, &witness, mode)
                    .expect("prove fuzz seed");
                verify_aggregate(setup.verifier(), &statement, &proof).expect("verify fuzz seed");
                let decoded =
                    wire::decode_pcs_proof(<$params>::DEGREE_ID, &proof).expect("decode fuzz seed");
                assert_eq!(decoded.records, 1);
                assert_eq!(decoded.salt_mode, mode);
                std::fs::write(output.join(format!("{}-{suffix}", $name)), proof)
                    .expect("write fuzz seed");
            }
        }};
    }

    generate_for!(Falcon512, 0x0512_f022_u64, "falcon512");
    generate_for!(Falcon1024, 0x1024_f022_u64, "falcon1024");
    Ok(())
}
