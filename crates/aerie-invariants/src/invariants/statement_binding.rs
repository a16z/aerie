//! **Statement binding — no proof verifies under two statements.**
//!
//! A statement is an aggregate of `records` records — public keys, salts, and
//! messages. An accepted envelope must bind exactly one of them. The input is a
//! *pair* of statements A and B, each given directly as an [`AggregationStatement`]
//! plus its [`AggregationWitness`] in canonical wire form. The check proves an
//! aggregate envelope for A, verifies that baseline under A, then re-presents
//! A's proof under B's public data — B's keys and messages supplied to the
//! verifier, with B's salts spliced into the transport frame — and verifies
//! again. A counterexample is
//! both verifications ACCEPTING: one proof authenticated two distinct
//! statements, a binding break.
//!
//! Because the agent supplies both statements verbatim (no pseudo-random seed
//! deciding the content), it has full control over what A and B are. A must be
//! provable and baseline-verifiable, including constructed relation-positive
//! fixtures. B needs only well-formed public data and may
//! differ from A in keys, salts/messages, or both — subsuming the old key- and
//! frame-mutation probes. Identical statements and record counts outside the
//! catalog-backed harness domain are skipped. Runs the real PCS on a deep
//! stack.

use serde::{Deserialize, Serialize};

use aerie_falcon::transport::{prove_aggregate, verify_aggregate};
use aerie_falcon::wire::{
    decode_pcs_proof, encode_pcs_proof, DecodedAggregateProof, TRANSPORT_SALT_BYTES,
};
use aerie_falcon::FalconDegree;
use aerie_falcon::{Falcon512, FalconAggregateSetup};
use aerie_falcon_native::source::CanonicalPublicKeyBytes;
use aerie_falcon_statement::fixtures::constructed_statement;
use aerie_falcon_statement::honest::honest_statement;
use aerie_falcon_statement::{split_source_statement, AggregationStatement, AggregationWitness};

use super::{supported_statement_record_counts, validate_statement_record_count};
use crate::bytes::Bytes;
use crate::invariant::{CheckError, CheckResult, Invariant, Outcome, Violation};
use crate::support::on_deep_stack;

/// Upper bound on the total input byte length, to bound allocation.
const MAX_BYTES: usize = 1 << 20;
/// One statement of the pair: an `AggregationStatement` and its
/// `AggregationWitness`, each as `canonical_encoding()` bytes — supplied inline
/// or, far more practically at this size, as a path to a file holding them (see
/// [`Bytes`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatementInput {
    pub statement: Bytes,
    pub witness: Bytes,
}

/// A pair of statements to test the binding between.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatementPair {
    /// The statement whose envelope is proved.
    pub a: StatementInput,
    /// The second statement A's proof is re-presented under.
    pub b: StatementInput,
}

#[derive(Default)]
pub struct StatementBinding;

impl Invariant for StatementBinding {
    type Input = StatementPair;
    const NAME: &'static str = "statement-binding";

    fn summary(&self) -> String {
        let supported_records = supported_statement_record_counts();
        format!(
            "Statement binding. A statement is an aggregate of `records` records (keys, \
             salts, messages). The input is a PAIR (A, B), each an AggregationStatement plus its \
             AggregationWitness in canonical wire form — you build them exactly, no seed lottery. \
             The check proves an aggregate envelope for A, first verifies it under A, then \
             re-presents A's proof under B's public data (B's keys and messages supplied to the \
             verifier, with B's salts spliced into the frame) and verifies again. A counterexample \
             is BOTH verifications ACCEPTING \
             — one proof authenticated A and B. A must be provable and baseline-verifiable, \
             including constructed relation-positive fixtures. B needs only well-formed public \
             data and may differ from A in keys, salts/messages, or both. Record counts must \
             match between A and B and be one of {supported_records:?}; all other record counts \
             and identical statements are skipped."
        )
    }

    fn input_description(&self) -> String {
        let supported_records = supported_statement_record_counts();
        format!(
            r#"One JSON object: {{"a": {{"statement": <bytes>, "witness": <bytes>}}, "b": {{...}}}}.
Each is the canonical_encoding() of an AggregationStatement / AggregationWitness.

Every <bytes> field takes EITHER an inline array ([1,2,3]) OR a path to a file
holding those raw bytes: {{"file": "a_statement.bin"}}, or just "a_statement.bin".
These are kilobytes of proof material — do NOT retype them. Write them with the
real encoder and hand over the paths:

    let input = aerie_invariants::invariants::sample_statement(8, Some(7))?;  // or None
    std::fs::write("a_statement.bin", &*input.statement)?;
    std::fs::write("a_witness.bin", &*input.witness)?;

then submit {{"a": {{"statement": "a_statement.bin", "witness": "a_witness.bin"}}, ...}}.
You can also build them via the statement crate (constructed_statement/honest_statement +
split_source_statement). A must be provable and baseline-verifiable; constructed
relation-positive fixtures are accepted. B needs only well-formed public data.
A and B must have the same record count, chosen from {supported_records:?},
and differ in keys/salts/messages; total input at most {MAX_BYTES} bytes."#
        )
    }

    fn seed_corpus(&self) -> Vec<StatementPair> {
        match (sample_statement(8, None), sample_statement(8, Some(7))) {
            (Ok(a), Ok(b)) => vec![StatementPair { a, b }],
            _ => Vec::new(),
        }
    }

    fn check(&self, input: &StatementPair) -> CheckResult {
        let total = input.a.statement.len()
            + input.a.witness.len()
            + input.b.statement.len()
            + input.b.witness.len();
        if total > MAX_BYTES {
            return Ok(Outcome::invalid(format!(
                "total input {total} exceeds the {MAX_BYTES}-byte cap"
            )));
        }
        let pair = input.clone();
        on_deep_stack(move || run_pair(&pair))
            .map_err(|error| CheckError::new(format!("statement-binding worker failed: {error}")))
    }
}

/// One parsed statement: its aggregate form (for proving) and its public data
/// (for re-presenting).
struct Prepared {
    statement: AggregationStatement<Falcon512>,
    witness: AggregationWitness<Falcon512>,
    keys: Vec<CanonicalPublicKeyBytes<aerie_falcon_native::Falcon512>>,
    salts: Vec<[u8; TRANSPORT_SALT_BYTES]>,
    messages: Vec<Vec<u8>>,
}

fn prepare(input: &StatementInput) -> Result<Prepared, String> {
    let statement = AggregationStatement::parse_canonical(&input.statement)
        .map_err(|error| error.to_string())?;
    let witness =
        AggregationWitness::parse_canonical(&input.witness).map_err(|error| error.to_string())?;
    if statement.records().len() != witness.signatures().len() {
        return Err("record count does not match the signature count".to_owned());
    }
    let keys = statement
        .records()
        .iter()
        .map(|record| record.public_key().clone())
        .collect();
    let messages = statement
        .records()
        .iter()
        .map(|record| record.message().to_vec())
        .collect();
    let salts = witness
        .signatures()
        .iter()
        .map(|signature| *signature.salt())
        .collect();
    Ok(Prepared {
        statement,
        witness,
        keys,
        salts,
        messages,
    })
}

fn run_pair(pair: &StatementPair) -> Outcome {
    let a = match prepare(&pair.a) {
        Ok(prepared) => prepared,
        Err(error) => return Outcome::invalid(format!("statement A malformed: {error}")),
    };
    let b = match prepare(&pair.b) {
        Ok(prepared) => prepared,
        Err(error) => return Outcome::invalid(format!("statement B malformed: {error}")),
    };
    let records = a.keys.len();
    for count in [records, b.keys.len()] {
        if let Err(reason) = validate_statement_record_count(count) {
            return Outcome::invalid(reason);
        }
    }
    if a.keys.len() != b.keys.len() {
        return Outcome::invalid("A and B must have the same record count");
    }
    if a.keys == b.keys && a.salts == b.salts && a.messages == b.messages {
        return Outcome::invalid("A and B are the same statement (a no-op pair)");
    }

    let setup = match FalconAggregateSetup::<Falcon512>::new(records) {
        Ok(setup) => setup,
        Err(error) => {
            return Outcome::invalid(format!("no PCS setup for {records} records: {error}"))
        }
    };
    // Proving, checking A's encoded frame, and baseline verification are one
    // attempt; if any stage fails, A is out of scope.
    let decoded_a = match prove_and_verify_a(&setup, &a) {
        Ok(decoded) => decoded,
        Err(error) => {
            return Outcome::invalid(format!(
                "statement A was not provable and baseline-verifiable at {records}: {error}"
            ));
        }
    };

    let spliced = match present_a_as_b(&decoded_a, &b) {
        Ok(bytes) => bytes,
        Err(error) => return Outcome::invalid(error),
    };
    match verify_aggregate(setup.verifier(), &b.keys, &b.messages, &spliced) {
        Ok(_) => Outcome::Violated(Violation::new(
            "an envelope verified under a second, distinct statement",
            format!(
                "statement A's proof (over {records} records) verified when re-presented under \
                 statement B's public data — its keys and salts/messages — after first verifying \
                 under A, a binding break."
            ),
        )),
        Err(_) => Outcome::Holds,
    }
}

/// Produce one envelope for A and establish that it really verifies under A's
/// complete public statement. `verify_aggregate` receives keys and messages
/// externally but reads salts from the envelope, so both the decoded-frame equality
/// and the verifier call are necessary provenance checks.
fn prove_and_verify_a(
    setup: &FalconAggregateSetup<Falcon512>,
    a: &Prepared,
) -> Result<DecodedAggregateProof, String> {
    let (envelope, _) = prove_aggregate(setup, &a.statement, &a.witness)
        .map_err(|error| format!("prover rejected A: {error}"))?;
    let decoded = decode_pcs_proof(FalconDegree::Falcon512, &envelope)
        .map_err(|error| format!("A's freshly proved envelope did not decode: {error}"))?;
    if decoded.records != a.keys.len() || decoded.salts != a.salts {
        return Err("A's freshly proved envelope did not contain A's exact salt frame".to_owned());
    }
    verify_aggregate(setup.verifier(), &a.keys, &a.messages, &envelope)
        .map_err(|error| format!("A's freshly proved envelope did not verify under A: {error}"))?;
    Ok(decoded)
}

/// Canonically re-encode A's exact decoded proof under B's public
/// salt frame. Verifying this envelope under B's keys and messages asks whether the
/// proof already accepted under A also authenticates statement B.
fn present_a_as_b(decoded_a: &DecodedAggregateProof, b: &Prepared) -> Result<Vec<u8>, String> {
    encode_pcs_proof(
        decoded_a.degree,
        decoded_a.records,
        &b.salts,
        &decoded_a.proof,
    )
    .map_err(|error| format!("could not re-present statement A's proof under B: {error}"))
}

/// Build one statement's canonical `StatementInput` from a fixture on a deep
/// stack — a starting point for feeding [`StatementBinding`] (e.g. from a test
/// or as a base an agent adapts). `honest_seed` selects a real FN-DSA fixture;
/// `None` gives the deterministic constructed fixture.
/// This helper also permits counts outside [`supported_statement_record_counts`]
/// so callers can construct out-of-scope inputs for tests.
///
/// # Errors
/// If the fixture cannot be generated, split, or encoded.
pub fn sample_statement(
    records: usize,
    honest_seed: Option<u64>,
) -> Result<StatementInput, String> {
    on_deep_stack(move || build_statement_input(records, honest_seed)).and_then(|inner| inner)
}

fn build_statement_input(
    records: usize,
    honest_seed: Option<u64>,
) -> Result<StatementInput, String> {
    let source = match honest_seed {
        None => constructed_statement(records).map_err(|error| error.to_string())?,
        Some(seed) => honest_statement(records, seed).map_err(|error| error.to_string())?,
    };
    let (statement, witness) =
        split_source_statement(&source).map_err(|error| error.to_string())?;
    Ok(StatementInput {
        statement: statement
            .canonical_encoding()
            .map_err(|error| error.to_string())?
            .into(),
        witness: witness
            .canonical_encoding()
            .map_err(|error| error.to_string())?
            .into(),
    })
}
