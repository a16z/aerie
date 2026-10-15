//! **Verifier totality.**
//!
//! `decode_pcs_proof` and `verify_aggregate` must be *total*: for any input bytes
//! they return `Ok`/`Err`, never panic or abort. A counterexample is a byte
//! string that makes either function unwind.
//!
//! The input is a record count plus the bytes to feed the verifier. The check
//! builds the constructed fixture's reproducible keys and messages for that many
//! records and runs `decode_pcs_proof` then `verify_aggregate` against them. Pure garbage
//! fuzzes the decoder (and fails fast at the magic/version/shape checks). To
//! reach the *deep* verifier, the caller builds a valid envelope for `records`
//! records with `aerie_invariants::invariants::sample_envelope` (or `prove_aggregate` on a
//! `constructed_aggregation`) and hands over a mutated copy with the matching
//! `records`.
//!
//! `records` and the byte length are capped to keep one check bounded and
//! discourage OOM; inputs past the caps are skipped, not counted. Note this
//! traps unwinding panics — an abort (OOM, or a stack overflow the guard page
//! turns into SIGSEGV) is not catchable here and is out of scope.
//!
//! Both parameter sets are covered. The decoder runs under both. The deep
//! verifier runs under Falcon-512 always, and under Falcon-1024 when the
//! header's parameter-set byte names Falcon-1024 and the Falcon-1024 catalog
//! supports `records` (any other Falcon-1024 input stops in the decoder,
//! which already ran). Falcon-1024 keys and messages come from the honest
//! FN-DSA-1024 fixture ([`super::honest_public_statement_1024`]).

use serde::{Deserialize, Serialize};

use aerie_falcon::transport::verify_aggregate;
use aerie_falcon::wire::{decode_pcs_proof, HEADER_PARAMETER_SET_OFFSET};
use aerie_falcon::FalconDegree;
use aerie_falcon::{Falcon1024, Falcon512, FalconVerifierSetup};

use super::{constructed_public_statement, honest_public_statement_1024};
use crate::bytes::Bytes;
use crate::invariant::{CheckError, CheckResult, Invariant, Outcome, Violation};
use crate::support::{run_check, trap_panic};

/// Upper bound on the record count, to bound fixture/setup/verify work and
/// discourage OOM. Well below the protocol's 2^16 ceiling.
const MAX_RECORDS: usize = 256;
/// Upper bound on the input byte length. A valid envelope is a few KiB across
/// the supported record range; this leaves ample room for mutation while
/// bounding the allocation.
const MAX_BYTES: usize = 4 * 1024 * 1024;

/// Bytes to feed the verifier, for a batch of `records` records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TotalityInput {
    /// Records the bytes claim to be for; the verifier gets this many keys and messages.
    pub records: usize,
    /// The raw input handed to the verifier: an inline array, or a path to a
    /// file holding it (see [`Bytes`]).
    pub bytes: Bytes,
}

#[derive(Default)]
pub struct VerifierTotality;

impl Invariant for VerifierTotality {
    type Input = TotalityInput;
    const NAME: &'static str = "verifier-totality";

    fn summary(&self) -> String {
        format!(
            "Verifier totality. Input is a record count and the raw bytes to feed the \
             verifier. The check builds the constructed fixture's keys and messages for `records` \
             records and runs decode_pcs_proof (both parameter sets) then \
             verify_aggregate(setup, statement, bytes) under Falcon-512, and also under \
             Falcon-1024 when the header names parameter set 10 and the Falcon-1024 catalog \
             supports `records` (sample_envelope_1024 builds such an envelope); a \
             counterexample is any input that makes either function PANIC. A graceful Err is \
             correct and counts as holding. Pure garbage exercises the decoder and fails fast; to \
             reach the deep verifier, build a valid envelope for `records` records \
             (aerie_invariants::invariants::sample_envelope, or prove_aggregate on a \
             constructed_aggregation) and submit a mutated copy with the matching `records`. \
             records must be 1..={MAX_RECORDS} and bytes at most {MAX_BYTES} long; larger inputs \
             are skipped. Aborts (OOM, SIGSEGV) are out of scope."
        )
    }

    fn input_description(&self) -> String {
        format!(
            r#"One JSON object: {{"records": 1..={MAX_RECORDS}, "bytes": <bytes>}}.
`bytes` is the exact input handed to the verifier (any length up to {MAX_BYTES});
`records` is the batch size the bytes target (also the verifier's key and message count).

<bytes> takes EITHER an inline array ([0,171,255]) OR a path to a file holding
the raw bytes: {{"file": "envelope.bin"}}, or just "envelope.bin". For a
mutated real envelope — the only way past the decoder into the deep verifier —
write it to a file and submit the path rather than retyping kilobytes:

    let mut envelope = aerie_invariants::invariants::sample_envelope(8)?;
    envelope[1234] ^= 0x01;                      // your mutation
    std::fs::write("envelope.bin", &envelope)?;

then submit {{"records": 8, "bytes": "envelope.bin"}}."#
        )
    }

    fn seed_corpus(&self) -> Vec<TotalityInput> {
        vec![
            // The decoder must reject these without panicking.
            TotalityInput {
                records: 8,
                bytes: Vec::new().into(),
            },
            TotalityInput {
                records: 8,
                bytes: vec![0xAB; 64].into(),
            },
            // The PCS magic then nothing: one step past the magic check.
            TotalityInput {
                records: 8,
                bytes: b"aerie/falcon/v1p".to_vec().into(),
            },
        ]
    }

    fn check(&self, input: &TotalityInput) -> CheckResult {
        let records = input.records;
        if records == 0 || records > MAX_RECORDS {
            return Ok(Outcome::invalid(format!(
                "records must be 1..={MAX_RECORDS}"
            )));
        }
        if input.bytes.len() > MAX_BYTES {
            return Ok(Outcome::invalid(format!(
                "byte length {} exceeds the {MAX_BYTES}-byte cap",
                input.bytes.len()
            )));
        }
        let bytes = input.bytes.to_vec();
        run_check(move || {
            if let Err(panic) = trap_panic(|| {
                for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
                    drop(decode_pcs_proof(degree, &bytes));
                }
            }) {
                return Outcome::Violated(Violation::new(
                    "decode_pcs_proof panicked on hostile bytes",
                    format!("decode panicked: {panic}"),
                ));
            }
            let statement = match constructed_public_statement(records) {
                Ok(statement) => statement,
                Err(error) => {
                    return Outcome::invalid(format!(
                        "fixture statement for {records} records unavailable: {error}"
                    ));
                }
            };
            let setup = match FalconVerifierSetup::<Falcon512>::new(records) {
                Ok(setup) => setup,
                Err(error) => {
                    return Outcome::invalid(format!(
                        "no verifier setup for {records} records: {error}"
                    ));
                }
            };
            if let Err(panic) = trap_panic(|| drop(verify_aggregate(&setup, &statement, &bytes))) {
                return Outcome::Violated(Violation::new(
                    "verify_aggregate panicked on hostile bytes",
                    format!("Falcon-512 verify panicked: {panic}"),
                ));
            }
            let names_falcon_1024 = bytes.get(HEADER_PARAMETER_SET_OFFSET)
                == Some(&FalconDegree::Falcon1024.parameter_set());
            if !names_falcon_1024 || !FalconVerifierSetup::<Falcon1024>::supports(records) {
                return Outcome::Holds;
            }
            let statement = match honest_public_statement_1024(records) {
                Ok(statement) => statement,
                Err(error) => {
                    return Outcome::invalid(format!(
                        "Falcon-1024 fixture statement for {records} records unavailable: {error}"
                    ));
                }
            };
            let setup = match FalconVerifierSetup::<Falcon1024>::new(records) {
                Ok(setup) => setup,
                Err(error) => {
                    return Outcome::invalid(format!(
                        "no Falcon-1024 verifier setup for {records} records: {error}"
                    ));
                }
            };
            match trap_panic(|| drop(verify_aggregate(&setup, &statement, &bytes))) {
                Ok(()) => Outcome::Holds,
                Err(panic) => Outcome::Violated(Violation::new(
                    "verify_aggregate panicked on hostile bytes",
                    format!("Falcon-1024 verify panicked: {panic}"),
                )),
            }
        })
        .map_err(|error| CheckError::new(format!("verifier-totality worker failed: {error}")))
    }
}
