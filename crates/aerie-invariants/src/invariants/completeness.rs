//! **Completeness.**
//!
//! Canonical, natively valid statement/witness pairs must survive the entire
//! production path: aggregate proving and wire-envelope verification with
//! verifier-recomputed HashToPoint targets for Falcon-512 and Falcon-1024.
//!
//! The input is the statement itself — an [`AggregationStatement`] plus its
//! [`AggregationWitness`] in canonical wire form — supplied directly, not a
//! seed. That gives the caller full control over the batch (keys, messages,
//! signatures) instead of a lottery over which pseudo-random seed happens to
//! surface a bug. Only canonical statement/witness pairs whose signatures verify
//! natively are in scope; no key-generation or signing provenance is required.
//! The check then proves the aggregate on the production path and verifies the
//! envelope. A
//! counterexample is a native-verifier-valid batch that the pipeline REJECTS or
//! PANICS on. Malformed or natively invalid inputs, and record counts outside the
//! catalog-backed harness domain, are skipped.

use aerie_falcon::transport::verify_aggregate;
use aerie_falcon::AggregationError;
use aerie_falcon::{AggregationParams, Falcon1024, Falcon512, FalconDegree};
use aerie_prover::transport::prove_aggregate;
use aerie_prover::FalconAggregateSetup;

/// An honest projection predicate failure is a permitted completeness event
/// below `2^-128.9` (Section 6.1), identified by its dedicated variant.
fn is_permitted_projection_rejection(error: &AggregationError) -> bool {
    matches!(error, AggregationError::ProjectionRejected)
}
use aerie_falcon_statement::AggregationStatement;
use aerie_falcon_witness::native::record::DirectRecordRef;
use aerie_falcon_witness::native::witness::verify_signatures_native;
use aerie_falcon_witness::AggregationWitness;

use super::{
    sample_statement, supported_statement_record_counts, validate_statement_record_count,
    StatementInput,
};
use crate::invariant::{CheckError, CheckResult, Invariant, Outcome, Violation};
use crate::support::{run_check, trap_panic};

/// Upper bound on the input byte length, to bound allocation.
const MAX_BYTES: usize = 1 << 20;

#[derive(Default)]
pub struct Completeness;

impl Invariant for Completeness {
    type Input = StatementInput;
    const NAME: &'static str = "completeness";

    fn summary(&self) -> String {
        let supported_records = format!(
            "Falcon-512 {:?}; Falcon-1024 {:?}",
            supported_statement_record_counts(FalconDegree::Falcon512),
            supported_statement_record_counts(FalconDegree::Falcon1024),
        );
        format!(
            "Completeness. The input is an aggregate statement (an AggregationStatement plus \
             its AggregationWitness in canonical wire form) — supplied directly, no seed, so you \
             control the batch exactly. Only canonical statement/witness pairs whose signatures \
             verify natively are in scope; no key-generation or signing provenance is required. \
             The check then proves the aggregate on the production path and verifies the \
             wire envelope. A counterexample is a \
             native-verifier-valid batch that the prover or verifier REJECTS or PANICS on. \
             Supported live record counts are {supported_records}. Malformed or natively \
             invalid inputs and all other record counts are skipped."
        )
    }

    fn input_description(&self) -> String {
        let supported_records = format!(
            "Falcon-512 {:?}; Falcon-1024 {:?}",
            supported_statement_record_counts(FalconDegree::Falcon512),
            supported_statement_record_counts(FalconDegree::Falcon1024),
        );
        format!(
            r#"One JSON object: {{"parameter_set": 9, "statement": <bytes>, "witness": <bytes>}}.
parameter_set is required: 9 selects Falcon-512 and 10 selects Falcon-1024.
Each byte field is the canonical_encoding() of an AggregationStatement / AggregationWitness.

Every <bytes> field takes EITHER an inline array ([1,2,3]) OR a path to a file
holding those raw bytes: {{"file": "statement.bin"}}, or just "statement.bin".
These are kilobytes of proof material — do NOT retype them. Write them with the
real encoder and hand over the paths:

    use aerie_falcon::FalconDegree;
    let input = aerie_invariants::invariants::sample_statement(FalconDegree::Falcon512, 8, Some(7))?;  // or None
    std::fs::write("statement.bin", &*input.statement)?;
    std::fs::write("witness.bin", &*input.witness)?;

then submit {{"parameter_set": 9, "statement": "statement.bin", "witness": "witness.bin"}}.
You can also build them via the aerie-falcon-witness crate; every signature must be natively
valid, the record count must be one of {supported_records}, and total input at most
{MAX_BYTES} bytes. The pair does not need to come from sample_statement or an
honest key-generation/signing helper."#
        )
    }

    fn seed_corpus(&self) -> Vec<StatementInput> {
        [FalconDegree::Falcon512, FalconDegree::Falcon1024]
            .into_iter()
            .map(|degree| {
                sample_statement(degree, 8, Some(7)).expect(
                    "completeness seed corpus: honest 8-record statement for each parameter set",
                )
            })
            .collect()
    }

    fn check(&self, input: &StatementInput) -> CheckResult {
        if input.statement.len() + input.witness.len() > MAX_BYTES {
            return Ok(Outcome::invalid(format!(
                "input exceeds the {MAX_BYTES}-byte cap"
            )));
        }
        let input = input.clone();
        run_check(move || match input.parameter_set {
            Some(9) => complete::<Falcon512>(&input),
            Some(10) => complete::<Falcon1024>(&input),
            _ => Outcome::invalid("parameter_set must be 9 (Falcon-512) or 10 (Falcon-1024)"),
        })
        .map_err(|error| CheckError::new(format!("completeness worker failed: {error}")))
    }
}

#[allow(clippy::too_many_lines)] // Native validity and both production phases form one check.
fn complete<P: AggregationParams>(input: &StatementInput) -> Outcome {
    let statement = match AggregationStatement::<P>::parse_canonical(&input.statement) {
        Ok(statement) => statement,
        Err(error) => return Outcome::invalid(format!("statement malformed: {error}")),
    };
    let witness = match AggregationWitness::<P>::parse_canonical(&input.witness) {
        Ok(witness) => witness,
        Err(error) => return Outcome::invalid(format!("witness malformed: {error}")),
    };
    let records = statement.len();
    if let Err(reason) = validate_statement_record_count(P::DEGREE_ID, records) {
        return Outcome::invalid(reason);
    }
    if records != witness.signatures().len() {
        return Outcome::invalid("record count does not match the signature count");
    }

    // Only native-verifier-valid statement/witness pairs are in scope. Their
    // key-generation and signing provenance is intentionally irrelevant.
    let direct: Vec<DirectRecordRef<'_, P>> = (0..records)
        .map(|index| {
            DirectRecordRef::new(
                &statement.public_keys()[index],
                &witness.signatures()[index],
                &statement.messages()[index],
            )
        })
        .collect();
    if let Err(error) = verify_signatures_native(&direct) {
        return Outcome::invalid(format!(
            "statement/witness pair is not natively valid: {error}"
        ));
    }

    let setup = match FalconAggregateSetup::<P>::new(records) {
        Ok(setup) => setup,
        Err(error) => {
            return Outcome::invalid(format!("no PCS setup for {records} records: {error}"))
        }
    };
    // Completeness is specifically about the deployed production path.
    let proved = trap_panic(|| prove_aggregate(&setup, &statement, &witness));
    let envelope = match proved {
        Ok(Ok(envelope)) => envelope,
        Ok(Err(error)) => {
            // An honest projection-predicate failure is a protocol-permitted
            // statistical event (< 2^-128.9). Under the adversarial search
            // this harness introduces, it must be classified as a permitted
            // failure, not a confirmed completeness counterexample.
            if is_permitted_projection_rejection(&error) {
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

    match trap_panic(|| verify_aggregate(setup.verifier(), &statement, &envelope)) {
        Ok(Ok(())) => Outcome::Holds,
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
    use super::*;
    use aerie_falcon::RejectionStage;

    #[test]
    fn recognizes_the_projection_rejection_variant() {
        assert!(is_permitted_projection_rejection(
            &AggregationError::ProjectionRejected
        ));
        for stage in [
            RejectionStage::Stage1,
            RejectionStage::Stage2,
            RejectionStage::Stage3,
            RejectionStage::Grinding,
            RejectionStage::PcsOpening,
        ] {
            assert!(!is_permitted_projection_rejection(
                &AggregationError::ProofRejected { stage }
            ));
        }
        assert!(!is_permitted_projection_rejection(
            &AggregationError::MalformedProof(aerie_falcon::MalformedProof::Truncated)
        ));
    }
}
