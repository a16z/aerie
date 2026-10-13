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
    BatchColumn, OutputLeg, ProofField, ProofStage, SourceCell, WitnessSoundness, WitnessTamper,
};

use std::sync::OnceLock;

use aerie_falcon::transport::prove_aggregate;
use aerie_falcon::{Falcon1024, Falcon512, FalconAggregateSetup};
use aerie_falcon_statement::fixtures::constructed_statement;
use aerie_falcon_statement::honest::honest_aggregation_1024;
use aerie_falcon_statement::{split_source_statement, AggregationStatement};

/// Live record counts accepted by completeness and statement binding.
/// Intersect the generated PCS catalog with the harness's 16-record resource
/// cap. Both validation and agent briefings use this domain.
#[must_use]
pub fn supported_statement_record_counts() -> &'static [usize] {
    const MAX_STATEMENT_RECORDS: usize = 16;
    static COUNTS: OnceLock<Vec<usize>> = OnceLock::new();
    COUNTS.get_or_init(|| {
        (1..=MAX_STATEMENT_RECORDS)
            .filter(|&records| FalconAggregateSetup::<Falcon512>::supports(records))
            .collect()
    })
}

fn validate_statement_record_count(records: usize) -> Result<(), String> {
    let supported = supported_statement_record_counts();
    if supported.contains(&records) {
        Ok(())
    } else {
        Err(format!(
            "record count {records} is out of scope; supported live record counts are {supported:?}"
        ))
    }
}

/// The constructed fixture's public statement for a given record count — cheap, no
/// proving. Used by the verifier-totality check to verify agent-supplied bytes
/// against reproducible keys and messages.
pub(crate) fn constructed_public_statement(
    records: usize,
) -> Result<AggregationStatement<Falcon512>, String> {
    let source = constructed_statement(records).map_err(|error| error.to_string())?;
    let (statement, _) = split_source_statement(&source).map_err(|error| error.to_string())?;
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
    honest_aggregation_1024(records, HONEST_1024_SEED)
        .map(|(statement, _)| statement)
        .map_err(|error| error.to_string())
}

/// Prove one constructed-fixture aggregate and return the envelope bytes. Must
/// be called on a deep stack (it runs the real PCS prover).
pub(crate) fn build_envelope(records: usize) -> Result<Vec<u8>, String> {
    let source = constructed_statement(records).map_err(|error| error.to_string())?;
    let (statement, witness) =
        split_source_statement(&source).map_err(|error| error.to_string())?;
    let setup =
        FalconAggregateSetup::<Falcon512>::new(records).map_err(|error| error.to_string())?;
    let (envelope, _) =
        prove_aggregate(&setup, &statement, &witness).map_err(|error| error.to_string())?;
    Ok(envelope)
}

/// A valid constructed-fixture envelope, proved on a deep stack — a starting
/// point for feeding [`VerifierTotality`] real bytes (e.g. from a test, or as
/// the base an agent mutates).
///
/// # Errors
/// If the fixture cannot be built or proved.
pub fn sample_envelope(records: usize) -> Result<Vec<u8>, String> {
    crate::support::on_deep_stack(move || build_envelope(records)).and_then(|inner| inner)
}

/// A valid Falcon-1024 envelope over the honest FN-DSA-1024 fixture, proved
/// on a deep stack: the Falcon-1024 counterpart of [`sample_envelope`].
/// `records` must be a Falcon-1024 catalog row.
///
/// # Errors
/// If the fixture cannot be built or proved.
pub fn sample_envelope_1024(records: usize) -> Result<Vec<u8>, String> {
    crate::support::on_deep_stack(move || {
        let (statement, witness) = honest_aggregation_1024(records, HONEST_1024_SEED)
            .map_err(|error| error.to_string())?;
        let setup =
            FalconAggregateSetup::<Falcon1024>::new(records).map_err(|error| error.to_string())?;
        prove_aggregate(&setup, &statement, &witness)
            .map(|(envelope, _)| envelope)
            .map_err(|error| error.to_string())
    })
    .and_then(|inner| inner)
}
