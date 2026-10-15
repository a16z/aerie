//! Commitment-first PIOP proving and grouped PCS opening.

use akita_params::CommittedGroupBatchProfile;
use akita_serialization::AkitaSerialize;
use akita_types::CommittedGroup;

use super::commit::{commit_group, is_s2_only_role, signed_group};
#[cfg(any(test, feature = "internals"))]
use super::group::delta_evaluation;
use super::group::SignedGroup;
use super::opening::{open_groups, physical_point, OpeningGroup};
use super::verifier_group_profiles_of;
use super::ProverCore;
use super::{AggregateProof, Error, FalconAggregateSetup, PcsStageMetrics, S2Precommit};
use crate::protocol::manifest::{GroupRole, GROUP_COUNT};
use crate::protocol::openings::opening_ledger;
use crate::protocol::projection;
#[cfg(any(test, feature = "internals"))]
use crate::protocol::projection::ProjectionOutputs;
use crate::protocol::source_tables::{PreparedCoefficients, SourceTables};
use crate::protocol::transcript::ProtocolTranscript;
use crate::protocol::{instance, wire::SaltMode, AggregationParams, FalconDegree, FalconWitness};
use crate::protocol::{
    stage1::{self, Stage1Proof},
    stage2::{self, Stage2Proof},
    stage3::{self, Stage3Proof},
};
use crate::substrate::{BaseField, Width4Config};
use crate::{StatementMismatch, UnsupportedBatch};

// Measurements showed that smaller padded batches do not repay the scheduling
// cost of overlapping stage two with stage-three preparation.
#[cfg(feature = "parallel")]
pub(super) const STAGE3_PREPARE_OVERLAP_MIN_ROWS: usize = 256;

struct PiopOutput {
    stage1_proof: Stage1Proof,
    stage1_ns: u128,
    stage2_proof: Stage2Proof,
    stage2_ns: u128,
    stage3_proof: Stage3Proof,
    third: stage3::Stage3Output,
    stage3_ns: u128,
}

/// Prove the complete version 1 protocol with real commitments, building
/// a one-off setup for the witness's parameter set and record count.
#[cfg(any(test, feature = "internals"))]
pub fn prove(
    batch: &FalconWitness,
    public_records: &[u8],
) -> Result<(AggregateProof, PcsStageMetrics), Error> {
    let setup_start = std::time::Instant::now();
    let degree = FalconDegree::from_degree(batch.degree())?;
    let setup = tracing::info_span!("v1.setup.prover")
        .in_scope(|| ProverCore::new(degree, batch.live_rows))?;
    let mut result = prove_prepared_impl(
        batch,
        public_records,
        SaltMode::PerRecord,
        &setup,
        #[cfg(any(test, feature = "internals"))]
        ProverHooks::default(),
        None,
    )?;
    result.1.setup_ns = setup_start.elapsed().as_nanos();
    Ok(result)
}

/// Prove with reusable shape-dependent PCS state.
#[cfg(any(test, feature = "internals"))]
pub fn prove_prepared<P: AggregationParams>(
    batch: &FalconWitness,
    public_records: &[u8],
    setup: &FalconAggregateSetup<P>,
) -> Result<(AggregateProof, PcsStageMetrics), Error> {
    prove_prepared_impl(
        batch,
        public_records,
        SaltMode::PerRecord,
        &setup.core,
        #[cfg(any(test, feature = "internals"))]
        ProverHooks::default(),
        None,
    )
}

/// [`prove_prepared`] with the `s2`-only groups already committed by
/// [`super::precommit_s2_groups`], typically concurrently with the ingestion that
/// produced `batch`. The tables and commitments must have been built for
/// this batch's degree, padded size and profiles; the proof bytes are
/// identical to [`prove_prepared`]'s.
pub fn prove_prepared_with_precommit<P: AggregationParams>(
    batch: &FalconWitness,
    public_records: &[u8],
    salt_mode: SaltMode,
    setup: &FalconAggregateSetup<P>,
    precommit: S2Precommit,
) -> Result<(AggregateProof, PcsStageMetrics), Error> {
    prove_prepared_impl(
        batch,
        public_records,
        salt_mode,
        &setup.core,
        #[cfg(any(test, feature = "internals"))]
        ProverHooks::default(),
        Some(precommit),
    )
}

/// Adversarial-test hook: apply a cheating witness consistently to both the
/// committed groups and the PIOP before producing the real PCS opening proof.
///
/// Source-table edits happen before the four commitments. Output edits see
/// the projection seed and replace the forward projection's `(y, z)` before
/// they are absorbed, so stage one proves against the edited outputs; with
/// an output hook the prover skips its own predicate check.
/// Unlike [`prove_with_group_tamper`], this hook proves against the edited
/// witness rather than intentionally creating a commitment/ledger mismatch.
#[cfg(feature = "internals")]
#[doc(hidden)]
pub fn prove_prepared_with_witness_tamper<P: AggregationParams>(
    batch: &FalconWitness,
    public_records: &[u8],
    setup: &FalconAggregateSetup<P>,
    tamper_tables: impl Fn(&mut SourceTables),
    tamper_outputs: impl Fn(&[u8], &mut ProjectionOutputs),
) -> Result<(AggregateProof, PcsStageMetrics), Error> {
    prove_prepared_impl(
        batch,
        public_records,
        SaltMode::PerRecord,
        &setup.core,
        ProverHooks {
            tables: Some(&tamper_tables),
            outputs: Some(&tamper_outputs),
            ..ProverHooks::default()
        },
        None,
    )
}

/// Adversarial-test hook (Section 12): commit tampered group cells while
/// proving the PIOP against the honest witness. The batched opening proves
/// the committed (tampered, zero-extended) polynomials' true evaluations,
/// which the Akita prover requires, so a proof is always emitted; the
/// verifier's ledger claim differs from the committed physical
/// table, and the multi-group PCS verifier must reject.
#[cfg(feature = "internals")]
#[doc(hidden)]
pub fn prove_with_group_tamper(
    batch: &FalconWitness,
    public_records: &[u8],
    tamper: &dyn Fn(GroupRole, &mut Vec<i64>),
) -> Result<(AggregateProof, PcsStageMetrics), Error> {
    let setup_start = std::time::Instant::now();
    let degree = FalconDegree::from_degree(batch.degree())?;
    let setup = tracing::info_span!("v1.setup.prover")
        .in_scope(|| ProverCore::new(degree, batch.live_rows))?;
    let mut result = prove_prepared_impl(
        batch,
        public_records,
        SaltMode::PerRecord,
        &setup,
        ProverHooks {
            group: Some(tamper),
            ..ProverHooks::default()
        },
        None,
    )?;
    result.1.setup_ns = setup_start.elapsed().as_nanos();
    Ok(result)
}

#[cfg(any(test, feature = "internals"))]
#[derive(Clone, Copy, Default)]
pub(super) struct ProverHooks<'a> {
    group: Option<&'a dyn Fn(GroupRole, &mut Vec<i64>)>,
    tables: Option<&'a dyn Fn(&mut SourceTables)>,
    outputs: Option<&'a dyn Fn(&[u8], &mut ProjectionOutputs)>,
}

#[allow(clippy::too_many_lines)]
pub(super) fn prove_prepared_impl(
    batch: &FalconWitness,
    public_records: &[u8],
    salt_mode: SaltMode,
    setup: &ProverCore,
    #[cfg(any(test, feature = "internals"))] hooks: ProverHooks<'_>,
    precommit: Option<S2Precommit>,
) -> Result<(AggregateProof, PcsStageMetrics), Error> {
    let records = batch.live_rows;
    if setup.records != records {
        return Err(Error::UnsupportedBatch(UnsupportedBatch::SetupSize {
            expected: setup.records,
            actual: records,
        }));
    }
    if setup.degree.degree() != batch.degree() {
        return Err(StatementMismatch::ParameterSet {
            expected: setup.degree.parameter_set(),
            actual: FalconDegree::from_degree(batch.degree())?.parameter_set(),
        }
        .into());
    }
    let mut padded_batch;
    let batch = if batch.padded_rows < setup.manifest.padded_records() {
        padded_batch = batch.clone();
        super::super::pad_witness(&mut padded_batch)?;
        &padded_batch
    } else {
        batch
    };
    let manifest = setup.manifest;
    if manifest.padded_records() != batch.padded_rows {
        return Err(Error::UnsupportedBatch(UnsupportedBatch::Shape {
            degree: batch.degree(),
            records,
        }));
    }
    let mut encodability_ns: u128 = 0;
    let mut early_commit_ns: u128 = 0;
    let (tables, early_groups) = if let Some(early) = precommit {
        #[cfg(any(test, feature = "internals"))]
        if hooks.group.is_some() || hooks.tables.is_some() {
            return Err(Error::internal("precommitted tables cannot be edited"));
        }
        if early.tables.degree() != batch.degree() || early.tables.records() != batch.padded_rows {
            return Err(Error::UnsupportedBatch(UnsupportedBatch::Shape {
                degree: batch.degree(),
                records,
            }));
        }
        encodability_ns = early.encodability_ns;
        early_commit_ns = early.commit_ns;
        (early.tables, Some(early.groups))
    } else {
        let tables = aerie_falcon_statement::staged!(
            "v1.prove.witness.encodability",
            encodability_ns,
            SourceTables::build(batch)?
        );
        (tables, None)
    };
    #[cfg(any(test, feature = "internals"))]
    let mut tables = tables;
    #[cfg(any(test, feature = "internals"))]
    if let Some(tamper) = hooks.tables {
        tamper(&mut tables);
    }
    // Compact encodings are prepared after any adversarial mutation and
    // before the PIOP timer so every proof path uses the same validated data.
    let coefficients = PreparedCoefficients::new(&tables)?;
    let logical_cells = &setup.logical_cells;

    let (_, _, selection) = verifier_group_profiles_of(setup.degree, records)?;
    let instance = instance::record(
        &manifest,
        salt_mode,
        public_records,
        [
            akita_types::setup_seed_digest(&setup.prover.expanded.descriptor.setup_seed)
                .map_err(Error::backend)?,
            Width4Config::catalog()?.catalog().catalog_digest(),
            *selection.row_digest.as_bytes(),
        ],
    )?;
    #[cfg(test)]
    let registry_instance = instance;
    #[cfg(test)]
    let instance = {
        let mut instance = instance;
        if let Some(offset) = setup.instance_mutation {
            instance[offset] ^= 1;
        }
        instance
    };
    #[cfg(not(test))]
    let mut bound = ProtocolTranscript::new(&manifest, &instance)?;
    #[cfg(test)]
    let mut bound =
        ProtocolTranscript::new_with_test_record(&manifest, &instance, &registry_instance)?;

    let mut commit_group_ns = [0_u128; GROUP_COUNT];
    // Adversarial group tamper only: each role's sparse deltas per polynomial, so the
    // opening proves the tampered polynomials' true evaluations.
    #[cfg(any(test, feature = "internals"))]
    let mut group_deltas: Vec<(GroupRole, Vec<Vec<(usize, i64)>>)> = Vec::new();

    // Every group not already committed by the early `s2` path, in
    // canonical order.
    let mut group_materialization_ns: u128 = 0;
    let pending =
        aerie_falcon_statement::staged!("v1.prove.witness.groups", group_materialization_ns, {
            let groups: Vec<(GroupRole, SignedGroup<'_>)> = manifest
                .groups()
                .iter()
                .map(|descriptor| descriptor.role)
                .filter(|role| early_groups.is_none() || !is_s2_only_role(*role))
                .map(|role| (role, signed_group(role, batch, &tables)))
                .collect();
            #[cfg(any(test, feature = "internals"))]
            let mut groups = groups;
            #[cfg(any(test, feature = "internals"))]
            if let Some(tamper) = hooks.group {
                for (role, group) in &mut groups {
                    group_deltas.push((*role, group.tamper(*role, tamper)?));
                }
            }
            groups
        });
    let mut commit_ns: u128 = 0;
    // Commitment is the first prover phase. Keep the overall prover span
    // open through the final PCS opening proof.
    let prove_span = tracing::info_span!("v1.prove.protocol").entered();
    let committed = aerie_falcon_statement::staged!("v1.commit", commit_ns, {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            pending
                .par_iter()
                .map(|(role, cells)| commit_group(setup, *role, cells).map(|group| (*role, group)))
                .collect::<Result<Vec<_>, Error>>()?
        }
        #[cfg(not(feature = "parallel"))]
        {
            pending
                .iter()
                .map(|(role, cells)| commit_group(setup, *role, cells).map(|group| (*role, group)))
                .collect::<Result<Vec<_>, Error>>()?
        }
    });
    drop(pending);
    // Canonical slot order S1, slack, S2 sources, budget: the early groups,
    // when present, are exactly the final two.
    let mut ordered = committed;
    if let Some(early) = early_groups {
        ordered.extend(early);
    }
    if ordered.len() != manifest.group_count() {
        return Err(Error::internal(
            "prover committed an incorrect number of groups",
        ));
    }
    let mut groups: Vec<CommittedGroup<BaseField>> = Vec::with_capacity(GROUP_COUNT);
    let mut handles = Vec::with_capacity(GROUP_COUNT);
    for (slot, (role, group)) in ordered.into_iter().enumerate() {
        if manifest.groups()[slot].role != role {
            return Err(Error::internal(
                "prover committed groups in an incorrect order",
            ));
        }
        bound.absorb_group_commitment(role, &group.payload)?;
        groups.push(group.committed);
        handles.push(group.handle);
        commit_group_ns[slot] = group.group_ns;
    }
    // The early commit ran concurrently with the ingestion; its wall time is
    // added so `commit_ns` still accounts every commit, but the prover total
    // differs from the sum of its stage timers.
    commit_ns += early_commit_ns;
    // Sanity: the committed profiles are the catalog profiles the final
    // group was scheduled against, and they resolve to a generated catalog
    // row. The verifier re-derives its own selection from the catalogs.
    let (final_group, precommitted) = groups.split_last().expect("four groups");
    if precommitted.iter().map(|group| *group.profile()).ne(setup
        .precommitted
        .as_slice()
        .iter()
        .copied())
    {
        return Err(Error::internal(
            "committed group differs from its catalog profile",
        ));
    }
    Width4Config::catalog()?.resolve_profiles(&CommittedGroupBatchProfile {
        final_group: *final_group.profile(),
        precommitteds: setup.precommitted.as_slice().to_vec(),
    })?;
    let commitment_bytes = groups
        .iter()
        .map(|group| group.commitment().compressed_size())
        .sum::<usize>();

    let seed = bound.derive_projection_seed()?;
    let mut transcript = bound.finish()?;

    // Forward projection and the public outputs `(y, z)`.
    let mut projection_ns: u128 = 0;
    let projected = aerie_falcon_statement::staged!("v1.projection", projection_ns, {
        // A cheating prover may emit any outputs; the predicates are then
        // the verifier's to check.
        #[cfg(any(test, feature = "internals"))]
        let projected = match hooks.outputs {
            Some(tamper) => {
                let mut projected = projection::project_unchecked(batch, &seed)?;
                tamper(&seed, &mut projected.outputs);
                projected
            }
            None => projection::project(batch, &seed)?,
        };
        #[cfg(not(any(test, feature = "internals")))]
        let projected = projection::project(batch, &seed)?;
        projected
    });

    // The PIOP blocks.
    #[cfg(feature = "parallel")]
    let piop_parent = tracing::Span::current();
    let piop_span = tracing::info_span!("v1.piop.prove");
    let degree = setup.degree;
    let mut piop = || -> Result<_, Error> {
        let _piop_span = piop_span.enter();
        let mut stage1_ns = 0;
        let (stage1_proof, first) = aerie_falcon_statement::staged!(
            "v1.stage1.prove",
            stage1_ns,
            stage1::prove_prepared(batch, &tables, &coefficients, &projected, &mut transcript)?
        );
        let mut stage2_ns = 0;
        let (stage2_result, stage3_prepared) =
            aerie_falcon_statement::staged!("v1.stage2.prove_with_stage3_prepare", stage2_ns, {
                #[cfg(feature = "parallel")]
                let parallel = batch.padded_rows >= STAGE3_PREPARE_OVERLAP_MIN_ROWS
                    && rayon::current_num_threads() > 1;
                #[cfg(not(feature = "parallel"))]
                let parallel = false;
                super::super::join_if(
                    parallel,
                    || {
                        stage2::prove_prepared(
                            degree,
                            &tables,
                            &coefficients,
                            &first,
                            &mut transcript,
                        )
                    },
                    || stage3::prepare(batch, &tables, &first),
                )
            });
        let (stage2_proof, second) = stage2_result?;
        let stage3_prepared = stage3_prepared?;
        let mut stage3_ns = 0;
        let (stage3_proof, third) = aerie_falcon_statement::staged!(
            "v1.stage3.prove",
            stage3_ns,
            stage3::prove_prepared(
                batch,
                &tables,
                &coefficients,
                &first,
                &second,
                stage3_prepared,
                &mut transcript,
            )?
        );
        Ok(PiopOutput {
            stage1_proof,
            stage1_ns,
            stage2_proof,
            stage2_ns,
            stage3_proof,
            third,
            stage3_ns,
        })
    };
    #[cfg(feature = "parallel")]
    let piop_result = if rayon::current_num_threads() > 1 {
        // Start the PIOP on a pool worker so nested parallel iterators and the
        // stage-two/prepare join can be stolen without cold pool injection.
        // Re-enter the caller span because tracing context is thread-local.
        rayon::scope(|_| piop_parent.in_scope(&mut piop))
    } else {
        piop()
    };
    #[cfg(not(feature = "parallel"))]
    let piop_result = piop();
    let PiopOutput {
        stage1_proof,
        stage1_ns,
        stage2_proof,
        stage2_ns,
        stage3_proof,
        third,
        stage3_ns,
    } = piop_result?;
    #[cfg(test)]
    if setup.session_mutation {
        transcript.alter_export_prefix_for_test();
    }
    let pcs_session = transcript.pcs_session()?;
    transcript.ensure_complete()?;
    let piop_work = transcript.work_bytes().to_vec();
    let projection_outputs = projected.outputs;
    let ledger = opening_ledger(degree, &third);
    drop(tables);
    drop(piop_span);

    let wire_commitments = groups.iter().map(|g| g.commitment().clone()).collect();
    let groups = groups
        .into_iter()
        .zip(handles)
        .zip(ledger.groups.iter().zip(logical_cells))
        .zip(manifest.groups())
        .map(|(((committed, handle), (claim, cells)), descriptor)| {
            #[cfg(not(any(test, feature = "internals")))]
            let _ = descriptor;
            let point = physical_point(&claim.point, *cells);
            #[cfg(any(test, feature = "internals"))]
            let values = group_deltas
                .iter()
                .filter(|(role, _)| *role == descriptor.role)
                .fold(claim.values.clone(), |values, (_, deltas)| {
                    values
                        .iter()
                        .zip(deltas)
                        .map(|(&value, polynomial_deltas)| {
                            value + delta_evaluation(polynomial_deltas, &point)
                        })
                        .collect()
                });
            #[cfg(not(any(test, feature = "internals")))]
            let values = claim.values.clone();
            OpeningGroup {
                committed,
                handle,
                point,
                values,
            }
        })
        .collect();
    let mut pcs_prove_ns = 0;
    let pcs_proof = aerie_falcon_statement::staged!(
        "v1.pcs.prove",
        pcs_prove_ns,
        open_groups(groups, setup, &pcs_session)?
    );
    let pcs_proof_bytes = pcs_proof.len();
    drop(prove_span);

    Ok((
        AggregateProof {
            commitments: wire_commitments,
            projection_outputs,
            stage1: stage1_proof,
            stage2: stage2_proof,
            stage3: stage3_proof,
            piop_work,
            pcs: pcs_proof,
        },
        PcsStageMetrics {
            records,
            padded_records: batch.padded_rows,
            encodability_ns,
            group_materialization_ns,
            setup_ns: 0,
            commit_ns,
            projection_ns,
            stage1_ns,
            stage2_ns,
            stage3_ns,
            pcs_prove_ns,
            target_evaluation_ns: 0,
            stage_verify_ns: 0,
            pcs_verify_ns: 0,
            target_evaluation_single_thread_ns: 0,
            stage_verify_single_thread_ns: 0,
            pcs_verify_single_thread_ns: 0,
            commit_group_ns,
            commitment_bytes,
            pcs_proof_bytes,
        },
    ))
}
