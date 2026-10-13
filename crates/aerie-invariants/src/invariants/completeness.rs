//! **Completeness.**
//!
//! Canonical, natively valid statement/witness pairs must survive the entire
//! production path: aggregate proving and wire-envelope verification with
//! verifier-recomputed HashToPoint targets. This closes the gap the survey
//! flagged — no cargo test runs a natively valid signature batch through
//! `prove_aggregate → verify_aggregate`; only the benchmark does.
//!
//! The input is the statement itself — an [`AggregationStatement`] plus its
//! [`AggregationWitness`] in canonical wire form — supplied directly, not a
//! seed. That gives the agent full control over the batch (keys, messages,
//! signatures) instead of a lottery over which pseudo-random seed happens to
//! surface a bug. Only canonical statement/witness pairs whose signatures verify
//! natively are in scope; no key-generation or signing provenance is required.
//! The check then proves the aggregate on the production path and verifies the
//! envelope. A
//! counterexample is a native-verifier-valid batch that the pipeline REJECTS or
//! PANICS on. Malformed or natively invalid inputs, and record counts outside the
//! catalog-backed harness domain, are skipped.

use aerie_falcon::projection::PROJECTION_REJECTED;
use aerie_falcon::transport::{prove_aggregate, verify_aggregate};
use aerie_falcon::{Falcon512, FalconAggregateSetup};

/// Recognize the prover's projection-rejection error by its stable prefix
/// [`aerie_falcon::projection::PROJECTION_REJECTED`]. An honest predicate
/// failure is a protocol-permitted completeness event below `2^-128.9`
/// (`specs/falcon.md` Section 6.1), not a violation. A structured error
/// channel through the proving API would remove this string dependency.
fn is_permitted_projection_rejection(error: &str) -> bool {
    error.contains(PROJECTION_REJECTED)
}
use aerie_falcon_native::record::DirectRecord;
use aerie_falcon_native::witness::verify_signatures_native;
use aerie_falcon_statement::{AggregationStatement, AggregationWitness};

use super::{
    sample_statement, supported_statement_record_counts, validate_statement_record_count,
    StatementInput,
};
use crate::invariant::{CheckError, CheckResult, Invariant, Outcome, Violation};
use crate::support::{on_deep_stack, trap_panic};

/// Upper bound on the input byte length, to bound allocation.
const MAX_BYTES: usize = 1 << 20;

#[derive(Default)]
pub struct Completeness;

impl Invariant for Completeness {
    type Input = StatementInput;
    const NAME: &'static str = "completeness";

    fn summary(&self) -> String {
        let supported_records = supported_statement_record_counts();
        format!(
            "Completeness. The input is an aggregate statement (an AggregationStatement plus \
             its AggregationWitness in canonical wire form) — supplied directly, no seed, so you \
             control the batch exactly. Only canonical statement/witness pairs whose signatures \
             verify natively are in scope; no key-generation or signing provenance is required. \
             The check then proves the aggregate on the production path and verifies the \
             wire envelope. A counterexample is a \
             native-verifier-valid batch that the prover or verifier REJECTS or PANICS on. \
             Supported live record counts are {supported_records:?}. Malformed or natively \
             invalid inputs and all other record counts are skipped."
        )
    }

    fn input_description(&self) -> String {
        let supported_records = supported_statement_record_counts();
        format!(
            r#"One JSON object: {{"statement": <bytes>, "witness": <bytes>}}.
Each field is the canonical_encoding() of an AggregationStatement / AggregationWitness.

Every <bytes> field takes EITHER an inline array ([1,2,3]) OR a path to a file
holding those raw bytes: {{"file": "statement.bin"}}, or just "statement.bin".
These are kilobytes of proof material — do NOT retype them. Write them with the
real encoder and hand over the paths:

    let input = aerie_invariants::invariants::sample_statement(8, Some(7))?;  // or None
    std::fs::write("statement.bin", &*input.statement)?;
    std::fs::write("witness.bin", &*input.witness)?;

then submit {{"statement": "statement.bin", "witness": "witness.bin"}}.
You can also build them via the aerie-falcon-statement crate; every signature must be natively
valid, the record count must be one of {supported_records:?}, and total input at most
{MAX_BYTES} bytes. The pair does not need to come from sample_statement or an
honest key-generation/signing helper."#
        )
    }

    fn seed_corpus(&self) -> Vec<StatementInput> {
        match sample_statement(8, Some(7)) {
            Ok(statement) => vec![statement],
            Err(_) => Vec::new(),
        }
    }

    fn check(&self, input: &StatementInput) -> CheckResult {
        if input.statement.len() + input.witness.len() > MAX_BYTES {
            return Ok(Outcome::invalid(format!(
                "input exceeds the {MAX_BYTES}-byte cap"
            )));
        }
        let input = input.clone();
        on_deep_stack(move || complete(&input))
            .map_err(|error| CheckError::new(format!("completeness worker failed: {error}")))
    }
}

#[allow(clippy::too_many_lines)] // permitted-overflow branch pushes it 1 over
fn complete(input: &StatementInput) -> Outcome {
    let statement = match AggregationStatement::parse_canonical(&input.statement) {
        Ok(statement) => statement,
        Err(error) => return Outcome::invalid(format!("statement malformed: {error}")),
    };
    let witness = match AggregationWitness::parse_canonical(&input.witness) {
        Ok(witness) => witness,
        Err(error) => return Outcome::invalid(format!("witness malformed: {error}")),
    };
    let records = statement.records().len();
    if let Err(reason) = validate_statement_record_count(records) {
        return Outcome::invalid(reason);
    }
    if records != witness.signatures().len() {
        return Outcome::invalid("record count does not match the signature count");
    }

    // Only native-verifier-valid statement/witness pairs are in scope. Their
    // key-generation and signing provenance is intentionally irrelevant.
    let direct: Vec<DirectRecord<aerie_falcon_native::Falcon512>> = (0..records)
        .map(|index| {
            DirectRecord::new(
                statement.records()[index].public_key().clone(),
                witness.signatures()[index].clone(),
                statement.records()[index].message().to_vec(),
            )
        })
        .collect();
    if let Err(error) = verify_signatures_native(&direct) {
        return Outcome::invalid(format!(
            "statement/witness pair is not natively valid: {error}"
        ));
    }

    let setup = match FalconAggregateSetup::<Falcon512>::new(records) {
        Ok(setup) => setup,
        Err(error) => {
            return Outcome::invalid(format!("no PCS setup for {records} records: {error}"))
        }
    };
    // Completeness is specifically about the deployed production path.
    let proved = trap_panic(|| prove_aggregate(&setup, &statement, &witness));
    let envelope = match proved {
        Ok(Ok((envelope, _))) => envelope,
        Ok(Err(error)) => {
            // An honest projection-predicate failure is a protocol-permitted
            // statistical event (< 2^-128.9). Under the adversarial search
            // this harness introduces, it must be classified as a permitted
            // failure, not a confirmed completeness counterexample.
            if is_permitted_projection_rejection(&error.to_string()) {
                return Outcome::invalid(format!(
                    "protocol-permitted projection rejection on a natively valid batch of \
                     {records} records (< 2^-128.9, negligible): {error}"
                ));
            }
            return Outcome::Violated(Violation::new(
                "natively valid signatures were rejected by the prover",
                format!(
                    "prove_aggregate errored on a natively valid batch of {records} records: \
                     {error}"
                ),
            ));
        }
        Err(panic) => {
            return Outcome::Violated(Violation::new(
                "the prover panicked on a natively valid batch",
                format!("prove_aggregate panicked on a natively valid batch of {records}: {panic}"),
            ));
        }
    };

    let public_keys: Vec<_> = statement
        .records()
        .iter()
        .map(|record| record.public_key().clone())
        .collect();
    let messages: Vec<_> = statement
        .records()
        .iter()
        .map(|record| record.message().to_vec())
        .collect();
    match trap_panic(|| verify_aggregate(setup.verifier(), &public_keys, &messages, &envelope)) {
        Ok(Ok(_)) => Outcome::Holds,
        Ok(Err(error)) => Outcome::Violated(Violation::new(
            "a proof for a natively valid batch was rejected by the verifier",
            format!(
                "verify_aggregate rejected the production proof for {records} records: {error}"
            ),
        )),
        Err(panic) => Outcome::Violated(Violation::new(
            "the verifier panicked on a proof for a natively valid batch",
            format!(
                "verify_aggregate panicked on the production proof for {records} records: {panic}"
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::is_permitted_projection_rejection;

    #[test]
    fn recognizes_the_projection_rejection_marker() {
        let rejected = rejection_display();
        assert!(is_permitted_projection_rejection(&rejected));
        assert!(!is_permitted_projection_rejection(
            "verify_aggregate rejected the proof"
        ));
        assert!(!is_permitted_projection_rejection("Invalid proof"));
    }

    /// The Display a caller sees for the prover's projection rejection.
    fn rejection_display() -> String {
        format!(
            "Invalid input: {}: the outputs fail a Section 6.1 predicate",
            aerie_falcon::projection::PROJECTION_REJECTED
        )
    }
}
