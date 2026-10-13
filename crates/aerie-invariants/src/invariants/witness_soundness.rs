//! **Witness soundness.**
//!
//! No accepting proof may correspond to a committed witness that violates a
//! verifier-checked predicate. This is Aerie's analog of Jolt's `soundness`
//! invariant: the counterexample carries the cheating witness (as a structured
//! tamper), the harness proves with it and verifies, and *acceptance is the
//! violation*.
//!
//! The input is a *list* of tampers applied together, so the agent can
//! coordinate them — for example, a claim move that preserves a stage terminal
//! equation, or an out-of-alphabet cell masked by neighboring edits.
//! Witness-side tampers (source cells, projection outputs, batch cells) are
//! applied before proving; proof-field perturbations are applied after. The
//! complete four-group PCS verifier then runs once.
//!
//! The net witness is validated independently of the verifier: exact integer
//! ring and norm identities, encodability, derived bounds, source decoding and
//! budget consistency, and projection outputs `(y, z)` equal to the clear
//! projection of the batch under the actual commitment-derived seed. Proof
//! claims alone do not establish a witness defect.
//!
//! All writes are last-write-wins. A restored or semantically equivalent
//! witness is out of scope. Coverage is limited to eight synthetic records,
//! the production cell encoding, and the exposed tamper sites;
//! record-count, live-mask, and semantic-padding attacks are not exercised.

use jolt_field::Ring;
use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};

use aerie_falcon::pcs::{self, AggregateProof};
use aerie_falcon::projection::{self, ProjectionOutputs, NORM_TERMINAL_ROWS, RESIDUAL_WIDTH};
use aerie_falcon::ring::{PublicRingKeys, PublicTargets};
use aerie_falcon::source_tables::SourceTables;
use aerie_falcon::PublicInputs;
use aerie_falcon::{BaseField, FalconWitness, OpeningField};
use aerie_falcon::{Falcon512, FalconAggregateSetup};
use aerie_falcon_statement::{DirectParams, DirectRow, FALCON_DEGREE, FALCON_Q};

use crate::invariant::{CheckError, CheckResult, Invariant, Outcome, Violation};
use crate::support::{ensure_deep_thread_pool, on_deep_stack};

const RECORDS: usize = 8;
const PUBLIC_RECORDS: &[u8; 32] = &[0xa1; 32];
/// The fixed eight-record fixture has twelve coefficient and signature bits.
const SUMCHECK_ROUNDS: usize = 9 + RECORDS.trailing_zeros() as usize;
/// Bound parsing, reduction, mutation, and persisted diagnostics for one input.
const MAX_TAMPERS: usize = 1024;

/// A structured cheating-witness or proof tamper. A [`WitnessSoundness`] input
/// is a list of these, applied together.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WitnessTamper {
    /// Set a committed source-table cell to `value` (cheating prover).
    SourceCell {
        table: SourceCell,
        index: usize,
        value: i64,
    },
    /// Add the signed integer `delta` in `F_p` to one public projection
    /// output, `y_index` (156 norm outputs) or `z_index` (31 residual
    /// outputs), before it is absorbed (cheating prover). The prover's
    /// predicate check is skipped; the verifier's is not.
    Output {
        leg: OutputLeg,
        index: usize,
        delta: i64,
    },
    /// Set a raw batch witness cell, then prove honestly (cheating prover).
    /// Counts when the net batch violates an exact semantic predicate.
    BatchCell {
        column: BatchColumn,
        index: usize,
        value: i64,
    },
    /// Perturb one current stage proof field by adding the signed integer
    /// `delta` directly in the field. For `rounds`, `slot` flattens
    /// `(round, coefficient)` in row-major order; other fields use their
    /// canonical array order. This can coordinate with an invalid witness but
    /// does not establish witness invalidity on its own.
    ProofClaim {
        stage: ProofStage,
        field: ProofField,
        slot: usize,
        delta: i64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceCell {
    A0,
    A1,
    A2,
    Sigma,
    U0,
    U1,
    U2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputLeg {
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchColumn {
    S1,
    S2,
    Slack,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofStage {
    Stage1,
    Stage2,
    Stage3,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofField {
    Rounds,
    Basis16Claims,
    ProjectionClaims,
    FoldedDigits,
    TerminalEvaluations,
}

/// See the module docs.
#[derive(Default)]
pub struct WitnessSoundness {
    setup: OnceLock<Result<Arc<FalconAggregateSetup<Falcon512>>, String>>,
}

impl WitnessSoundness {
    /// Reusable PCS state for the supported eight-record shape. Built once, on
    /// a deep stack, and shared by every candidate checked by this instance.
    fn setup(&self) -> Result<Arc<FalconAggregateSetup<Falcon512>>, String> {
        self.setup
            .get_or_init(|| match on_deep_stack(build_setup) {
                Ok(inner) => inner,
                Err(spawn) => Err(spawn),
            })
            .clone()
    }
}

impl Invariant for WitnessSoundness {
    type Input = Vec<WitnessTamper>;
    const NAME: &'static str = "witness-soundness";

    fn summary(&self) -> String {
        format!(
            "Witness soundness. The input is a JSON ARRAY of at most \
             {MAX_TAMPERS} tampers applied \
             together to a supported batch of {RECORDS} synthetic records proved and verified \
             through the real four-group Akita PCS: witness-side tampers \
             (source cells, projection outputs, batch cells) before proving, proof-side \
             claim perturbations after. \
             A counterexample is: the full verifier ACCEPTS and the NET \
             witness violates an exact semantic predicate: ring or norm/slack identity, \
             coefficient/quotient bounds, unary budget, source decoding/budget tie, \
             or projection outputs (y, z) differing from the actual seeded projection. \
             Proof-side edits can coordinate with a witness defect but are out of scope \
             on their own. Writes are last-write-wins; restoring a valid witness is \
             out of scope. This fixed eight-record fixture does not exercise record-count, \
             live-mask, or semantic-padding failures."
        )
    }

    fn input_description(&self) -> String {
        format!("At most {MAX_TAMPERS} tampers are accepted per candidate.\n\n{SCHEMA}")
    }

    fn seed_corpus(&self) -> Vec<Vec<WitnessTamper>> {
        vec![
            vec![WitnessTamper::BatchCell {
                column: BatchColumn::S1,
                index: 9,
                value: i64::from(synthetic_batch(RECORDS).s1[9]) + 1,
            }],
            (0..FALCON_DEGREE)
                .map(|index| WitnessTamper::BatchCell {
                    column: BatchColumn::S2,
                    index,
                    value: 1024,
                })
                .collect(),
            vec![WitnessTamper::SourceCell {
                table: SourceCell::A2,
                index: 5,
                value: 8,
            }],
            // A residual output moved by q: both predicates still pass.
            vec![WitnessTamper::Output {
                leg: OutputLeg::Z,
                index: 4,
                delta: 12_289,
            }],
            vec![WitnessTamper::BatchCell {
                column: BatchColumn::S2,
                index: 9,
                value: 2048,
            }],
            // Proof-side changes matter to witness soundness only when coordinated with an
            // invalid witness. These edits target two Stage1 projection claims.
            vec![
                WitnessTamper::SourceCell {
                    table: SourceCell::A0,
                    index: 5,
                    value: 8,
                },
                WitnessTamper::ProofClaim {
                    stage: ProofStage::Stage1,
                    field: ProofField::ProjectionClaims,
                    slot: 0,
                    delta: 1,
                },
                WitnessTamper::ProofClaim {
                    stage: ProofStage::Stage1,
                    field: ProofField::ProjectionClaims,
                    slot: 1,
                    delta: 1,
                },
            ],
            // Coordinated: an out-of-alphabet cell masked by an in-alphabet edit
            // at a different index (the bad cell survives).
            vec![
                WitnessTamper::SourceCell {
                    table: SourceCell::A0,
                    index: 5,
                    value: 8,
                },
                WitnessTamper::SourceCell {
                    table: SourceCell::A0,
                    index: 6,
                    value: 3,
                },
            ],
        ]
    }

    // The signature is fixed by `type Input = Vec<_>`; a slice arg is not an
    // option for a trait-method impl.
    #[allow(clippy::ptr_arg)]
    fn check(&self, input: &Vec<WitnessTamper>) -> CheckResult {
        if input.is_empty() {
            return Ok(Outcome::invalid("empty tamper list"));
        }
        if input.len() > MAX_TAMPERS {
            return Ok(Outcome::invalid(format!(
                "tamper list has {} entries, past the {MAX_TAMPERS}-entry cap",
                input.len()
            )));
        }
        let Partitioned {
            sources,
            outputs,
            batches,
            claims,
        } = match partition(input) {
            Ok(partitioned) => partitioned,
            Err(outcome) => return Ok(outcome),
        };

        // Validation and source preparation also use Rayon. Initialize its
        // prover-sized workers before either can create the default pool.
        ensure_deep_thread_pool();
        let mut batch = synthetic_batch(RECORDS);
        for &(column, index, value) in &batches {
            if !apply_batch_cell(&mut batch, column, index, value) {
                return Ok(Outcome::invalid(format!(
                    "batch index {index} past the {column:?} column"
                )));
            }
        }
        rederive_quotients(&mut batch);
        // These owners also validate normal witness preparation. Do not add
        // another list of Falcon predicates to the invariant itself.
        let mut defect = batch
            .validate_semantics()
            .err()
            .map(|error| error.to_string());
        match SourceTables::build(&batch) {
            Ok(mut tables) => {
                for &(table, index, cell) in &sources {
                    write_source(&mut tables, table, index, cell);
                }
                if let Err(error) = tables.validate_semantics(&batch) {
                    defect.get_or_insert(error);
                }
            }
            // The encoder can reject a proven-invalid batch before producing
            // tables. The full proving entry point below must reject it too.
            Err(_) if defect.is_some() => {}
            Err(error) => {
                return Err(CheckError::new(format!(
                    "source preparation failed: {error}"
                )))
            }
        }
        if defect.is_none() && outputs.iter().all(|&(_, _, delta)| delta == 0) {
            return Ok(no_witness_defect());
        }
        let reason = format!(
            "{} coordinated tampers ({} source, {} output, {} batch, {} proof claim)",
            input.len(),
            sources.len(),
            outputs.len(),
            batches.len(),
            claims.len()
        );
        let setup = self.setup().map_err(|error| {
            CheckError::new(format!("witness-soundness PCS setup failed: {error}"))
        })?;
        deep(move || {
            let defect = RefCell::new(defect);
            let result = pcs::prove_prepared_with_witness_tamper(
                &batch,
                PUBLIC_RECORDS,
                &setup,
                |tables| {
                    for &(table, index, cell) in &sources {
                        write_source(tables, table, index, cell);
                    }
                },
                |seed, emitted| {
                    for &(leg, index, delta) in &outputs {
                        apply_output(emitted, leg, index, delta);
                    }
                    // Compare with the projection under the real
                    // commitment-derived seed, never a probe built with an
                    // unrelated seed.
                    if let Some(error) = output_defect(&batch, seed, emitted) {
                        defect.borrow_mut().get_or_insert(error);
                    }
                },
            );
            match (result, defect.into_inner()) {
                (Err(_), Some(_)) => Ok(Outcome::Holds),
                (Err(error), None) => Err(CheckError::new(format!(
                    "proving failed before a net witness defect was established: {error}"
                ))),
                (Ok(_), None) => Ok(no_witness_defect()),
                (Ok((proof, _)), Some(defect)) => Ok(finish(
                    &batch,
                    proof,
                    &claims,
                    &setup,
                    format!("{reason}; net witness defect: {defect}"),
                )),
            }
        })
    }
}

/// The tampers grouped by execution stage.
struct Partitioned {
    sources: Vec<(SourceCell, usize, i8)>,
    outputs: Vec<(OutputLeg, usize, i64)>,
    batches: Vec<(BatchColumn, usize, i64)>,
    claims: Vec<(ProofStage, ProofField, usize, i64)>,
}

/// Validate every element's bounds and group them by stage. A malformed element
/// makes the whole list out of scope (`Err`). Whether a tamper is *invalidating*
/// is decided later from the net state, not here.
fn partition(input: &[WitnessTamper]) -> Result<Partitioned, Outcome> {
    let mut parts = Partitioned {
        sources: Vec::new(),
        outputs: Vec::new(),
        batches: Vec::new(),
        claims: Vec::new(),
    };
    for (i, tamper) in input.iter().enumerate() {
        match *tamper {
            WitnessTamper::SourceCell {
                table,
                index,
                value,
            } => {
                let Ok(cell) = i8::try_from(value) else {
                    return Err(Outcome::invalid(format!("tamper {i}: source cells are i8")));
                };
                let len = source_len(table);
                if index >= len {
                    return Err(Outcome::invalid(format!(
                        "tamper {i}: index {index} past the {len}-cell {table:?} table"
                    )));
                }
                parts.sources.push((table, index, cell));
            }
            WitnessTamper::Output { leg, index, delta } => {
                let len = output_len(leg);
                if index >= len {
                    return Err(Outcome::invalid(format!(
                        "tamper {i}: index {index} past the {len} {leg:?} outputs"
                    )));
                }
                parts.outputs.push((leg, index, delta));
            }
            WitnessTamper::BatchCell {
                column,
                index,
                value,
            } => {
                if i16::try_from(value).is_err() {
                    return Err(Outcome::invalid(format!(
                        "tamper {i}: value does not fit the {column:?} carrier"
                    )));
                }
                parts.batches.push((column, index, value));
            }
            WitnessTamper::ProofClaim {
                stage,
                field,
                slot,
                delta,
            } => {
                if !claim_site_valid(stage, field, slot) {
                    return Err(Outcome::invalid(format!(
                        "tamper {i}: no proof field {stage:?}/{field:?} at slot {slot}"
                    )));
                }
                parts.claims.push((stage, field, slot, delta));
            }
        }
    }
    Ok(parts)
}

/// Run a proving/verifying closure on a prover-sized stack, mapping a
/// spawn/join failure to a checker error.
fn deep<F>(body: F) -> CheckResult
where
    F: FnOnce() -> CheckResult + Send + 'static,
{
    on_deep_stack(body)
        .map_err(|error| CheckError::new(format!("witness-soundness worker failed: {error}")))?
}

fn build_setup() -> Result<Arc<FalconAggregateSetup<Falcon512>>, String> {
    FalconAggregateSetup::<Falcon512>::new(RECORDS)
        .map(Arc::new)
        .map_err(|error| error.to_string())
}

/// Apply optional coordinated proof-side tampers, then run the full PCS
/// verifier. The caller has already established a net-invalid witness.
fn finish(
    batch: &FalconWitness,
    mut proof: AggregateProof,
    claims: &[(ProofStage, ProofField, usize, i64)],
    setup: &FalconAggregateSetup<Falcon512>,
    reason: String,
) -> Outcome {
    for &(stage, field, slot, delta) in claims {
        if delta != 0 {
            apply_claim(&mut proof, stage, field, slot, delta);
        }
    }
    if pcs::verify(&public_inputs(batch), setup.verifier(), &proof).is_ok() {
        Outcome::Violated(Violation::new(
            "the full PCS verifier accepted a proof for a provably-invalid witness",
            reason,
        ))
    } else {
        Outcome::Holds
    }
}

fn apply_claim(
    proof: &mut AggregateProof,
    stage: ProofStage,
    field: ProofField,
    slot: usize,
    delta: i64,
) {
    let magnitude = field_magnitude(delta.unsigned_abs());
    let target: Option<&mut OpeningField> = match (stage, field) {
        (ProofStage::Stage1, ProofField::Rounds) => proof
            .stage1
            .rounds
            .get_mut(slot / 3)
            .and_then(|round| round.get_mut(slot % 3)),
        (ProofStage::Stage1, ProofField::Basis16Claims) => proof
            .stage1
            .basis16_claims
            .get_mut(slot / 2)
            .and_then(|claims| claims.get_mut(slot % 2)),
        (ProofStage::Stage1, ProofField::ProjectionClaims) => {
            proof.stage1.projection_claims.get_mut(slot)
        }
        (ProofStage::Stage2, ProofField::Rounds) => proof
            .stage2
            .rounds
            .get_mut(slot / 4)
            .and_then(|round| round.get_mut(slot % 4)),
        (ProofStage::Stage2, ProofField::FoldedDigits) => proof.stage2.folded_digits.get_mut(slot),
        (ProofStage::Stage3, ProofField::Rounds) => proof
            .stage3
            .rounds
            .get_mut(slot / 3)
            .and_then(|round| round.get_mut(slot % 3)),
        (ProofStage::Stage3, ProofField::TerminalEvaluations) => {
            proof.stage3.terminal_evaluations.get_mut(slot)
        }
        _ => None,
    };
    if let Some(entry) = target {
        if delta >= 0 {
            *entry += magnitude;
        } else {
            *entry -= magnitude;
        }
    }
}

const fn claim_site_valid(stage: ProofStage, field: ProofField, slot: usize) -> bool {
    match (stage, field) {
        (ProofStage::Stage1 | ProofStage::Stage3, ProofField::Rounds) => slot < SUMCHECK_ROUNDS * 3,
        (ProofStage::Stage1, ProofField::Basis16Claims) => slot < 4,
        (ProofStage::Stage1, ProofField::ProjectionClaims) => slot < 3,
        (ProofStage::Stage2, ProofField::Rounds) => slot < SUMCHECK_ROUNDS * 4,
        (ProofStage::Stage2, ProofField::FoldedDigits) => slot < 6,
        (ProofStage::Stage3, ProofField::TerminalEvaluations) => slot < 9,
        _ => false,
    }
}

const fn source_len(table: SourceCell) -> usize {
    match table {
        SourceCell::A0 | SourceCell::A1 | SourceCell::A2 | SourceCell::Sigma => {
            RECORDS * FALCON_DEGREE
        }
        SourceCell::U0 | SourceCell::U1 | SourceCell::U2 => RECORDS,
    }
}

const fn output_len(leg: OutputLeg) -> usize {
    match leg {
        OutputLeg::Y => NORM_TERMINAL_ROWS,
        OutputLeg::Z => RESIDUAL_WIDTH,
    }
}

fn apply_output(outputs: &mut ProjectionOutputs, leg: OutputLeg, index: usize, delta: i64) {
    let magnitude = BaseField::from_u64(delta.unsigned_abs());
    let entry = match leg {
        OutputLeg::Y => &mut outputs.y[index],
        OutputLeg::Z => &mut outputs.z[index],
    };
    if delta >= 0 {
        *entry += magnitude;
    } else {
        *entry -= magnitude;
    }
}

/// The emitted outputs are a defect when they differ from the clear
/// projection of the (net) batch under the transcript's seed, or when that
/// projection itself fails a predicate (a batch defect the prover's own
/// check would have caught).
fn output_defect(
    batch: &FalconWitness,
    seed: &[u8],
    emitted: &ProjectionOutputs,
) -> Option<String> {
    match projection::project(batch, seed) {
        Ok(honest) if honest.outputs() == emitted => None,
        Ok(_) => Some("projection outputs differ from the seeded projection".to_owned()),
        Err(error) => Some(format!("the seeded projection fails a predicate: {error}")),
    }
}

fn no_witness_defect() -> Outcome {
    Outcome::invalid(
        "the net witness satisfies the semantic predicates; proof-only changes are out of scope",
    )
}

fn apply_batch_cell(
    batch: &mut FalconWitness,
    column: BatchColumn,
    index: usize,
    value: i64,
) -> bool {
    match column {
        BatchColumn::S1 => set(&mut batch.s1, index, value as i16),
        BatchColumn::S2 => set(&mut batch.s2, index, value as i16),
        BatchColumn::Slack => set(&mut batch.slack, index, value as i16),
    }
}

/// Version 1 commits no quotient table: `v = (s_1 + h (*) s_2 - t) / q` is
/// auxiliary and recovered by extraction (`specs/falcon.md` Sections 2.2 and
/// 9.4). Recompute it from the net witness so the semantic check sees the
/// quotient a verifier-accepted witness actually determines. A residual that
/// `q` does not divide keeps its truncated quotient and fails the relation.
fn rederive_quotients(batch: &mut FalconWitness) {
    rederive_quotients_of::<aerie_falcon_native::Falcon512>(batch);
}

fn rederive_quotients_of<P: DirectParams>(batch: &mut FalconWitness) {
    let mut residual = vec![0i64; P::DEGREE];
    for record in 0..batch.live_rows {
        let range = record * P::DEGREE..(record + 1) * P::DEGREE;
        let product =
            P::convolution_negacyclic_i16(&batch.h[range.clone()], &batch.s2[range.clone()]);
        for (((slot, &s1), &t), &product) in residual
            .iter_mut()
            .zip(&batch.s1[range.clone()])
            .zip(&batch.t[range.clone()])
            .zip(&product)
        {
            *slot = i64::from(s1) + product - i64::from(t);
        }
        for (quotient, &value) in batch.v[range].iter_mut().zip(&residual) {
            // |Delta| < 2^41 for any i16 cells, so Delta / q fits an i32.
            *quotient = i32::try_from(value / FALCON_Q).expect("quotient fits i32");
        }
    }
}

fn set<T: Copy>(slice: &mut [T], index: usize, value: T) -> bool {
    if let Some(slot) = slice.get_mut(index) {
        *slot = value;
        true
    } else {
        false
    }
}

fn synthetic_batch(records: usize) -> FalconWitness {
    let rows: Vec<DirectRow<Falcon512>> = (0..records)
        .map(|seed| {
            DirectRow::<aerie_falcon_native::Falcon512>::synthetic(seed as u64 + 263)
                .expect("synthetic row")
        })
        .collect();
    FalconWitness::from_rows(&rows)
}

fn public_inputs(batch: &FalconWitness) -> PublicInputs<'_, i16> {
    PublicInputs {
        records: batch.live_rows,
        salt_mode: aerie_falcon::wire::SaltMode::PerRecord,
        public_records: PUBLIC_RECORDS,
        h: PublicRingKeys::Expanded(&batch.h),
        t: PublicTargets::Expanded(&batch.t),
    }
}

fn write_source(tables: &mut SourceTables, table: SourceCell, index: usize, cell: i8) {
    match table {
        SourceCell::A0 => tables.a0_mut()[index] = cell,
        SourceCell::A1 => tables.a1_mut()[index] = cell,
        SourceCell::A2 => tables.a2_mut()[index] = cell,
        SourceCell::Sigma => tables.sigma_mut()[index] = cell,
        SourceCell::U0 => tables.u0_mut()[index] = cell,
        SourceCell::U1 => tables.u1_mut()[index] = cell,
        SourceCell::U2 => tables.u2_mut()[index] = cell,
    }
}

/// Direct conversion supports the full signed-i64 delta domain, including MIN.
fn field_magnitude(count: u64) -> OpeningField {
    OpeningField::from_u64(count)
}

const SCHEMA: &str = r#"A JSON ARRAY of one or more tamper objects, each internally tagged by "kind",
applied together (witness-side before proving, proof-side after) then verified
once through the real four-group Akita PCS at the supported 8-record shape.
A violation requires the full verifier to ACCEPT and the NET witness to be
provably invalidating; proof-only candidates and witness tampers that cancel
are skipped.

- {"kind":"source_cell","table":T,"index":usize,"value":int}
    T in {"a0","a1","a2","sigma","u0","u1","u2"}.
    Alphabets: a0 [-4,3]; a1/a2 [-8,7]; sigma {-1,1};
    u0/u1/u2 [-4,3] for this Falcon-512 fixture.
    Length 8*512 = 4096 (a0/a1/a2/sigma) or 8 (u0/u1/u2).
    Final values are checked for alphabets, decoded s2 equality, and the
    per-row unary_sum + slack = 392 identity.
- {"kind":"output","leg":L,"index":usize,"delta":i64}
    L in {"y","z"}: y has 156 norm outputs, z has 31 residual outputs. Adds the
    signed integer delta in F_p before the outputs are absorbed; the prover's
    predicate check is skipped. The net outputs must equal the clear projection
    of the batch under the actual commitment seed. The verifier requires
    sum ctr_p(y_j)^2 <= C_S and q | ctr_p(z_j), q = 12289.
- {"kind":"batch_cell","column":C,"index":usize,"value":int}
    C in {"s1","s2","slack"}, all i16; s1/s2 length 4096, slack length
    32. The quotient v is not committed: it is re-derived from the net
    (s1, s2, t) as (s1 + h (*) s2 - t) / q. Checks the negacyclic ring
    identity, exact norm/slack sum 34034726, unary budget <=392, and
    bounds: |s2|<=2047, |s1| and |slack|<=5833, |v|<=65998.
- {"kind":"proof_claim","stage":S,"field":F,"slot":usize,"delta":i64}
    S in {"stage1","stage2","stage3"}. Stage1 fields: rounds (slot<36),
    basis16_claims (flattened pair slot<4), projection_claims (slot<3).
    Stage2 fields: rounds (slot<48), folded_digits (slot<6).
    Stage3 fields: rounds (slot<36), terminal_evaluations (slot<9).
    For rounds, slot flattens round and coefficient in row-major order.
    Adds the signed integer delta directly in the opening field. It may
    coordinate with an invalid witness, but is out of scope by itself.
Coverage is fixed to eight synthetic records; record-count, live-mask, and
semantic-padding-dependent failures are not exercised.
Coordinate tampers by combining an invalid source cell with a proof-field edit:
[{"kind":"source_cell","table":"a0","index":5,"value":8},
 {"kind":"proof_claim","stage":"stage1","field":"projection_claims","slot":0,"delta":1}]"#;

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{One, Zero};

    #[test]
    fn claim_magnitudes_cover_zero_and_signed_extremes() {
        assert_eq!(field_magnitude(0), OpeningField::zero());
        assert_eq!(
            field_magnitude(65),
            OpeningField::from_u64(64) + OpeningField::one()
        );
        assert_eq!(
            field_magnitude(i64::MIN.unsigned_abs()),
            field_magnitude(i64::MAX as u64) + OpeningField::one(),
        );
        for delta in [i64::MIN, i64::MAX] {
            assert!(partition(&[WitnessTamper::ProofClaim {
                stage: ProofStage::Stage2,
                field: ProofField::FoldedDigits,
                slot: 0,
                delta,
            }])
            .is_ok());
        }
    }
}
