//! The four machine-checkable invariants and shared constructed-fixture
//! helpers used by the `statement-binding` and `verifier-totality` checks.

mod completeness;
mod statement_binding;
mod verifier_totality;
mod witness_soundness;

pub use completeness::Completeness;
pub use statement_binding::{sample_statement, StatementBinding, StatementInput, StatementPair};
pub use verifier_totality::{TotalityInput, VerifierTotality};
pub use witness_soundness::{
    BatchColumn, OutputLeg, ProofField, ProofStage, SourceCell, WitnessInput, WitnessSoundness,
    WitnessTamper,
};

use aerie_prover::FalconAggregateSetup;
use std::sync::OnceLock;

use aerie_falcon::{Falcon1024, Falcon512, FalconDegree};
use aerie_falcon_statement::AggregationStatement;
use aerie_falcon_witness::fixtures::constructed_aggregation;
use aerie_falcon_witness::honest::{honest_aggregation, SaltMode};
use aerie_prover::transport::prove_aggregate;

/// Live record counts accepted by completeness and statement binding.
/// Intersect the generated PCS catalog with the harness's 16-record resource
/// cap. Both validation and caller briefings use this domain.
#[must_use]
pub fn supported_statement_record_counts(degree: FalconDegree) -> &'static [usize] {
    const MAX_STATEMENT_RECORDS: usize = 16;
    static COUNTS_512: OnceLock<Vec<usize>> = OnceLock::new();
    static COUNTS_1024: OnceLock<Vec<usize>> = OnceLock::new();
    let counts = match degree {
        FalconDegree::Falcon512 => &COUNTS_512,
        FalconDegree::Falcon1024 => &COUNTS_1024,
    };
    counts.get_or_init(|| {
        (1..=MAX_STATEMENT_RECORDS)
            .filter(|&records| match degree {
                FalconDegree::Falcon512 => FalconAggregateSetup::<Falcon512>::supports(records),
                FalconDegree::Falcon1024 => FalconAggregateSetup::<Falcon1024>::supports(records),
            })
            .collect()
    })
}

fn validate_statement_record_count(degree: FalconDegree, records: usize) -> Result<(), String> {
    let supported = supported_statement_record_counts(degree);
    if supported.contains(&records) {
        Ok(())
    } else {
        Err(format!(
            "record count {records} is out of scope; supported live record counts are {supported:?}"
        ))
    }
}

/// The constructed fixture's public statement for a given record count — cheap, no
/// proving. Used by the verifier-totality check to verify caller-supplied bytes
/// against reproducible keys and messages.
pub(crate) fn constructed_public_statement(
    records: usize,
) -> Result<AggregationStatement<Falcon512>, String> {
    let (statement, _) =
        constructed_aggregation::<Falcon512>(records).map_err(|error| error.to_string())?;
    Ok(statement)
}

/// Seed of the honest FN-DSA-1024 fixture behind the Falcon-1024 totality
/// keys and [`sample_envelope_1024`].
const HONEST_1024_SEED: u64 = 0x1024_7074;

/// The honest Falcon-1024 fixture's public statement for a given record
/// count: real FN-DSA-1024 keys and messages, no proving. Key generation is
/// the cost, so the Falcon-1024 totality path only calls this for frames
/// that name Falcon-1024.
pub(crate) fn honest_public_statement_1024(
    records: usize,
) -> Result<AggregationStatement<Falcon1024>, String> {
    honest_aggregation::<Falcon1024>(records, HONEST_1024_SEED, SaltMode::PerRecord)
        .map(|(statement, _)| statement)
        .map_err(|error| error.to_string())
}

/// Prove one constructed-fixture aggregate and return the envelope bytes.
pub(crate) fn build_envelope(records: usize) -> Result<Vec<u8>, String> {
    let (statement, witness) =
        constructed_aggregation::<Falcon512>(records).map_err(|error| error.to_string())?;
    let setup =
        FalconAggregateSetup::<Falcon512>::new(records).map_err(|error| error.to_string())?;
    let envelope =
        prove_aggregate(&setup, &statement, &witness).map_err(|error| error.to_string())?;
    Ok(envelope)
}

/// A valid constructed-fixture envelope for feeding [`VerifierTotality`]
/// real bytes (e.g. from a test, or as the base a search driver mutates).
///
/// # Errors
/// If the fixture cannot be built or proved.
pub fn sample_envelope(records: usize) -> Result<Vec<u8>, String> {
    crate::support::run_check(move || build_envelope(records)).and_then(|inner| inner)
}

/// A valid Falcon-1024 envelope over the honest FN-DSA-1024 fixture:
/// the Falcon-1024 counterpart of [`sample_envelope`].
/// `records` must be a Falcon-1024 catalog row.
///
/// # Errors
/// If the fixture cannot be built or proved.
pub fn sample_envelope_1024(records: usize) -> Result<Vec<u8>, String> {
    crate::support::run_check(move || {
        let (statement, witness) =
            honest_aggregation::<Falcon1024>(records, HONEST_1024_SEED, SaltMode::PerRecord)
                .map_err(|error| error.to_string())?;
        let setup =
            FalconAggregateSetup::<Falcon1024>::new(records).map_err(|error| error.to_string())?;
        prove_aggregate(&setup, &statement, &witness).map_err(|error| error.to_string())
    })
    .and_then(|inner| inner)
}
