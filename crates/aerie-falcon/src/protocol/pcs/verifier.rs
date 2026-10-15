//! Public-statement verification and PCS claim reduction.

use akita_params::{BasisMode, OpeningScheduleSelection};
use akita_types::{CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use akita_verifier::AkitaVerifier;

use super::serialized_commitment;
use super::setup::group_cells_of;
use super::verifier_group_profiles_of;
use super::{AggregateProof, Error, FalconVerifierSetup, VerifiedOpening};
use crate::protocol::manifest::CommitmentManifest;
use crate::protocol::openings::{opening_ledger, OpeningLedger};
use crate::protocol::public_inputs::PublicInputs;
use crate::protocol::transcript::VerifierProtocolTranscript;
use crate::protocol::zero_extension::physical_point;
use crate::protocol::{instance, projection, stage1, stage2, stage3, AggregationParams};
use crate::substrate::{BaseField, OpeningField, Width4Config};
use crate::{MalformedProof, RejectionStage, StatementMismatch, UnsupportedBatch};

/// The verifier's reconstruction of every PIOP block: the committed groups
/// and their per-polynomial claims. Nothing is accepted until the opening
/// verifies against these claims.
struct ReducedClaims {
    session: [u8; 32],
    groups: Vec<CommittedGroup<BaseField>>,
    claims: Vec<(Vec<OpeningField>, Vec<OpeningField>)>,
    ledger: OpeningLedger,
    selection: OpeningScheduleSelection,
    target_evaluation_ns: u128,
    stage_verify_ns: u128,
}

/// Verify the complete version 1 protocol from public inputs and wire data.
///
/// Every group profile and the schedule selection are re-derived from the
/// statement shape and the generated catalogs; only commitment payloads
/// and proof messages are consumed from the proof.
///
/// The public inputs must be of the setup's parameter set and record count.
pub fn verify<P: AggregationParams, C>(
    public: &PublicInputs<'_, C>,
    verifier_setup: &FalconVerifierSetup<P>,
    proof: &AggregateProof,
) -> Result<VerifiedOpening, Error>
where
    C: Copy + Sync + Into<i64>,
{
    if public.degree != P::DEGREE_ID {
        return Err(StatementMismatch::ParameterSet {
            expected: P::DEGREE_ID.parameter_set(),
            actual: public.degree.parameter_set(),
        }
        .into());
    }
    if public.records != verifier_setup.records {
        return Err(Error::UnsupportedBatch(UnsupportedBatch::SetupSize {
            expected: verifier_setup.records,
            actual: public.records,
        }));
    }
    verify_opening(public, &verifier_setup.akita, proof)
}

#[allow(clippy::too_many_lines)]
#[tracing::instrument(skip_all, name = "v1.verify.protocol")]
#[doc(hidden)]
pub fn verify_opening<C>(
    public: &PublicInputs<'_, C>,
    verifier: &AkitaVerifier<Width4Config>,
    proof: &AggregateProof,
) -> Result<VerifiedOpening, Error>
where
    C: Copy + Sync + Into<i64>,
{
    let reduced = reduce_pcs(public, verifier, proof)?;
    let groups = reduced
        .groups
        .iter()
        .zip(&reduced.claims)
        .map(|(group, (point, values))| {
            PolynomialGroupClaims::new(point.as_slice(), values.clone(), group)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| Error::MalformedProof(MalformedProof::CommitmentShape))?;
    let claims = OpeningClaims::from_groups(groups)
        .map_err(|_| Error::MalformedProof(MalformedProof::CommitmentShape))?;
    let mut pcs_verify_ns = 0;
    aerie_falcon_statement::staged!(
        "v1.pcs.verify",
        pcs_verify_ns,
        verifier
            .batched_verify(
                &proof.pcs,
                &reduced.session,
                GroupBatchStatement::new(reduced.selection, claims)
                    .map_err(|_| Error::MalformedProof(MalformedProof::CommitmentShape))?,
                BasisMode::Lagrange,
            )
            .map_err(|_| Error::ProofRejected {
                stage: RejectionStage::PcsOpening,
            })?
    );
    Ok(VerifiedOpening {
        ledger: reduced.ledger,
        target_evaluation_ns: reduced.target_evaluation_ns,
        stage_verify_ns: reduced.stage_verify_ns,
        pcs_verify_ns,
    })
}

pub(super) fn verifier_instance<C: Copy + Sync + Into<i64>>(
    public: &PublicInputs<'_, C>,
    manifest: &CommitmentManifest,
    selection: OpeningScheduleSelection,
    verifier: &AkitaVerifier<Width4Config>,
) -> Result<[u8; instance::INSTANCE_BYTES], Error> {
    instance::record(
        manifest,
        public.salt_mode,
        public.public_records,
        [
            akita_types::setup_seed_digest(&verifier.setup().expanded().descriptor.setup_seed)
                .map_err(Error::backend)?,
            Width4Config::catalog()?.catalog().catalog_digest(),
            *selection.row_digest.as_bytes(),
        ],
    )
}

/// Replay every PIOP block and reconstruct the group claims the batched
/// opening must discharge.
#[allow(clippy::too_many_lines)]
fn reduce_pcs<C>(
    public: &PublicInputs<'_, C>,
    verifier: &AkitaVerifier<Width4Config>,
    proof: &AggregateProof,
) -> Result<ReducedClaims, Error>
where
    C: Copy + Sync + Into<i64>,
{
    let preprocessing_span = tracing::info_span!("v1.verify.preprocessing").entered();
    let live_rows = public.records;
    let degree = public.degree;
    let manifest = CommitmentManifest::for_batch_of(degree, live_rows)?;
    let records = manifest.padded_records();
    public
        .h
        .validate_shape_of(degree.degree(), live_rows, records)?;
    public
        .t
        .validate_shape_of(degree.degree(), live_rows, records)?;
    let group_count = manifest.group_count();
    if proof.commitments.len() != group_count {
        return Err(Error::MalformedProof(MalformedProof::CommitmentShape));
    }

    let logical_cells = group_cells_of(&manifest);
    let (precommit_profiles, final_profile, opening_selection) =
        verifier_group_profiles_of(degree, live_rows)?;
    let (final_commitment, precommitted) = proof
        .commitments
        .split_last()
        .expect("the group count is positive");
    let groups: Vec<CommittedGroup<BaseField>> = precommit_profiles
        .into_iter()
        .zip(precommitted)
        .map(|(profile, commitment)| CommittedGroup::new(profile, commitment.clone()))
        .collect();
    let final_group = CommittedGroup::new(final_profile, final_commitment.clone());

    let instance = verifier_instance(public, &manifest, opening_selection, verifier)?;
    let mut bound = VerifierProtocolTranscript::new(&manifest, &instance, &proof.piop_work)?;
    for (descriptor, commitment) in manifest.groups().iter().zip(&proof.commitments) {
        bound.absorb_group_commitment(descriptor.role, &serialized_commitment(commitment)?)?;
    }
    let seed = bound.derive_projection_seed()?;
    let mut transcript = bound.finish()?;
    drop(preprocessing_span);

    let mut stage_verify_inclusive_ns: u128 = 0;
    let mut target_evaluation_ns: u128 = 0;
    let ledger = aerie_falcon_statement::staged!("v1.stages.verify", stage_verify_inclusive_ns, {
        if !projection::check_outputs(degree, &proof.projection_outputs) {
            return Err(Error::ProofRejected {
                stage: RejectionStage::Stage1,
            });
        }
        let first = stage1::verify(
            degree,
            records,
            live_rows,
            &seed,
            &proof.projection_outputs,
            &proof.stage1,
            &mut transcript,
        )?;
        let second = stage2::verify(degree, records, &first, &proof.stage2, &mut transcript)?;
        let third = stage3::verify(
            records,
            public,
            &first,
            &second,
            &proof.stage3,
            &mut transcript,
            &mut target_evaluation_ns,
        )?;
        opening_ledger(degree, &third)
    });

    let mut all_groups = groups;
    all_groups.push(final_group);
    let claims = ledger
        .groups
        .iter()
        .zip(&logical_cells)
        .map(|(opening, cells)| {
            (
                physical_point(&opening.point, *cells),
                opening.values.clone(),
            )
        })
        .collect();
    let stage_verify_ns = stage_verify_inclusive_ns.saturating_sub(target_evaluation_ns);
    let session = transcript.pcs_session()?;
    transcript.ensure_complete()?;
    Ok(ReducedClaims {
        session,
        groups: all_groups,
        claims,
        ledger,
        selection: opening_selection,
        target_evaluation_ns,
        stage_verify_ns,
    })
}
