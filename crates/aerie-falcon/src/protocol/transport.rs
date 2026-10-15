//! Transport binding: canonical record bytes and verifier-local targets.
//!
//! The statement is bound to its source records through the canonical
//! public-record tree digest over the record count, hashed public
//! keys, salts, and messages, absorbed before commitments in the Section 10.1
//! transcript. The wire proof transports only what the verifier does not
//! already hold: the `HashToPoint` salts and the proof material. Common-salt
//! mode carries one salt and binds its mode through the digest customization.
//! Public keys and messages are statement inputs supplied to the verifier.
//! The verifier streams both public tables from these inputs: stage three
//! unpacks each canonical key's `h` row and samples each `HashToPoint`
//! target row `t` exactly once, contracting it at the stage point without
//! materializing a padded table. Neither table is absorbed into the
//! transcript; the public-record digest binds their sources.
//! Nothing target-shaped is ever trusted from the prover.

#[cfg(any(test, feature = "internals"))]
use aerie_falcon_native::FalconParams;
use aerie_falcon_native::{preprocess::PrehashedHashToPointBatch, source::CanonicalPublicKeyBytes};
use aerie_falcon_statement::AggregationStatement;
#[cfg(all(feature = "parallel", any(test, feature = "internals")))]
use rayon::prelude::*;

use super::pcs::{self, FalconVerifierSetup, VerifiedOpening};
use super::public_records_binding::{hashed_keys, public_records_binding_from_view};
use super::wire::{self, SaltView};
use super::AggregationParams;

#[cfg(any(test, feature = "internals"))]
pub use super::public_records_binding::public_records_binding;
#[cfg(feature = "internals")]
pub use super::public_records_binding::PUBLIC_RECORDS_BINDING_BLOCK_BYTES;

use crate::{AggregationError as Error, StatementMismatch, UnsupportedBatch};

/// Complete public-input-and-proof-to-acceptance timing for the verifier.
#[cfg_attr(
    not(feature = "internals"),
    allow(
        dead_code,
        clippy::struct_field_names,
        reason = "read only by the measured entry points; the suffix names the unit"
    )
)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AggregateVerifyMetrics {
    pub total_ns: u128,
    pub decode_ns: u128,
    /// Derivation of the fixed-width `hpk` values needed before stage one.
    pub hashed_keys_ns: u128,
    /// End-to-end public-target work: `hpk` derivation plus the streamed
    /// target evaluation.
    pub hash_to_point_ns: u128,
    /// Streamed `HashToPoint` sampling and contraction of every target row at
    /// the stage point.
    pub target_evaluation_ns: u128,
    pub binding_ns: u128,
    /// Stage verification excluding target evaluation.
    pub stage_verify_ns: u128,
    pub pcs_verify_ns: u128,
}

/// The zero-extended public ring-key table decoded from canonical bytes.
#[cfg(any(test, feature = "internals"))]
#[doc(hidden)]
pub fn derived_ring_keys<P: FalconParams>(
    public_keys: &[CanonicalPublicKeyBytes<P>],
    padded: usize,
) -> Vec<i16> {
    let degree = P::DEGREE;
    let mut table = vec![0_i16; padded * degree];
    #[cfg(feature = "parallel")]
    table[..public_keys.len() * degree]
        .par_chunks_mut(degree)
        .zip(public_keys.par_iter())
        .for_each(|(row, public_key)| {
            copy_centered_key_row(row, public_key);
        });
    #[cfg(not(feature = "parallel"))]
    for (record, public_key) in public_keys.iter().enumerate() {
        copy_centered_key_row(
            &mut table[record * degree..(record + 1) * degree],
            public_key,
        );
    }
    table
}

#[cfg(any(test, feature = "internals"))]
fn copy_centered_key_row<P: FalconParams>(
    row: &mut [i16],
    public_key: &CanonicalPublicKeyBytes<P>,
) {
    debug_assert_eq!(row.len(), P::DEGREE);
    for (destination, block) in row
        .chunks_exact_mut(4)
        .zip(public_key.centered_coefficient_blocks())
    {
        destination.copy_from_slice(&block);
    }
}

/// Verify one aggregate proof of parameter set `P` FROM WIRE BYTES: no
/// prover-side data. The arguments follow the usual order: the public
/// verifier setup, the public statement (ordered canonical public-key bytes
/// and ordered raw messages), then the proof bytes. The wire header must name
/// `P` and the setup's record count. The salts come from the proof bytes, and
/// every ring key and target is derived locally. A message list that differs
/// from the one the prover bound changes the public-record digest and the
/// recomputed targets, so it rejects.
#[inline]
pub fn verify_aggregate<P: AggregationParams>(
    verifier_setup: &FalconVerifierSetup<P>,
    statement: &AggregationStatement<P>,
    proof: &[u8],
) -> Result<(), Error> {
    verify_aggregate_impl(
        verifier_setup,
        statement.public_keys(),
        statement.messages(),
        proof,
    )
    .map(|_| ())
}

/// [`verify_aggregate`] with complete online timing and the opening ledger.
#[cfg(any(test, feature = "internals"))]
#[inline]
pub fn verify_aggregate_measured<P: AggregationParams>(
    verifier_setup: &FalconVerifierSetup<P>,
    statement: &AggregationStatement<P>,
    proof: &[u8],
) -> Result<(VerifiedOpening, AggregateVerifyMetrics), Error> {
    verify_aggregate_impl(
        verifier_setup,
        statement.public_keys(),
        statement.messages(),
        proof,
    )
}

#[inline]
/// Shared verifier replay for producer-owned regression tests.
#[doc(hidden)]
pub fn verify_aggregate_impl<P: AggregationParams>(
    verifier_setup: &FalconVerifierSetup<P>,
    public_keys: &[CanonicalPublicKeyBytes<P>],
    messages: &[Vec<u8>],
    proof: &[u8],
) -> Result<(VerifiedOpening, AggregateVerifyMetrics), Error> {
    let total_start = std::time::Instant::now();
    let total_span = tracing::info_span!("v1.verify.total").entered();
    // Reject fixed-width shape mismatches before decoding or allocating for
    // any attacker-supplied proof body.
    let (_, records) = wire::peek_pcs_proof_header(P::DEGREE_ID, proof)?;
    if public_keys.len() != records {
        return Err(StatementMismatch::RecordCount {
            expected: public_keys.len(),
            actual: records,
        }
        .into());
    }
    if records != verifier_setup.records() {
        return Err(UnsupportedBatch::SetupSize {
            expected: verifier_setup.records(),
            actual: records,
        }
        .into());
    }
    let mut decode_ns = 0;
    let decoded = aerie_falcon_statement::staged!(
        "v1.verify.input.decode",
        decode_ns,
        // The header's parameter-set byte must name `P`; the decoder checks
        // it before parsing anything after it.
        wire::decode_pcs_proof(P::DEGREE_ID, proof)?
    );
    debug_assert_eq!(decoded.records, records);
    let mut hashed_keys_ns = 0;
    let hashed_keys = aerie_falcon_statement::staged!(
        "v1.verify.input.hashed_keys",
        hashed_keys_ns,
        hashed_keys(public_keys)
    );
    let mut binding_ns = 0;
    let binding = aerie_falcon_statement::staged!(
        "v1.verify.input.binding",
        binding_ns,
        public_records_binding_from_view(&hashed_keys, decoded.salts.view(), messages)?
    );
    // The digest above binds the hashed keys, salts and messages that these
    // views read; the tables they stream are never absorbed themselves.
    let target_inputs = match decoded.salts.view() {
        SaltView::Common(salt) => {
            PrehashedHashToPointBatch::new_common(&hashed_keys, salt, messages)
        }
        SaltView::PerRecord(salts) => PrehashedHashToPointBatch::new(&hashed_keys, salts, messages),
    }
    .map_err(|_| Error::StatementMismatch(StatementMismatch::PublicInputShape))?;
    let public: super::public_inputs::PublicInputs<'_, i16> = super::public_inputs::PublicInputs {
        degree: P::DEGREE_ID,
        records,
        salt_mode: decoded.salts.view().mode(),
        public_records: &binding,
        h: P::ring_keys(public_keys),
        t: super::ring::PublicTargets::SourceBound(target_inputs),
    };
    let verified = pcs::verify(&public, verifier_setup, &decoded.proof)?;
    let hash_to_point_ns = hashed_keys_ns.saturating_add(verified.target_evaluation_ns);
    let metrics = AggregateVerifyMetrics {
        total_ns: total_start.elapsed().as_nanos(),
        decode_ns,
        hashed_keys_ns,
        hash_to_point_ns,
        target_evaluation_ns: verified.target_evaluation_ns,
        binding_ns,
        stage_verify_ns: verified.stage_verify_ns,
        pcs_verify_ns: verified.pcs_verify_ns,
    };
    drop(total_span);
    Ok((verified, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "parallel")]
    use crate::protocol::ring::{target_evaluation, PublicTargets};
    use crate::protocol::FalconDegree;
    #[cfg(feature = "parallel")]
    use crate::substrate::OpeningField;
    use aerie_falcon_native::Falcon512;
    use jolt_field::Ring as _;
    /// A syntactically valid canonical public key that differs from the
    /// constructed fixture's shared key: header byte plus all-zero
    /// coefficients.
    fn zero_public_key() -> CanonicalPublicKeyBytes<aerie_falcon_native::Falcon512> {
        let mut bytes = [0_u8; Falcon512::PUBLIC_KEY_BYTES];
        bytes[0] = 0x09;
        CanonicalPublicKeyBytes::parse(&bytes).expect("canonical zero key")
    }

    fn zero_public_key_of<P: FalconParams>() -> CanonicalPublicKeyBytes<P> {
        let mut bytes = vec![0_u8; P::PUBLIC_KEY_BYTES];
        bytes[0] = P::PUBLIC_KEY_HEADER;
        CanonicalPublicKeyBytes::parse(&bytes).expect("canonical zero key")
    }

    /// The wire verifier's streamed targets agree with a padded table of
    /// scalar native `HashToPoint` rows at both degrees.
    fn source_bound_targets_match_native_scalar_of<P: AggregationParams>() {
        let live_rows = 3;
        let padded_rows = P::DEGREE_ID.padded_records(live_rows).unwrap();
        let key = zero_public_key_of::<P>();
        let public_keys = vec![key; live_rows];
        let salts = (0..live_rows)
            .map(|record| {
                core::array::from_fn(|index| record.wrapping_mul(31).wrapping_add(index) as u8)
            })
            .collect::<Vec<_>>();
        let messages = (0..live_rows)
            .map(|record| vec![record as u8; record * 19])
            .collect::<Vec<_>>();
        let hpks = hashed_keys(&public_keys);
        let source = PrehashedHashToPointBatch::new(&hpks, &salts, &messages).unwrap();
        let mut expanded = public_keys
            .iter()
            .zip(&salts)
            .zip(&messages)
            .flat_map(|((key, salt), message)| {
                aerie_falcon_native::preprocess::hash_to_point_public_raw(key, salt, message)
                    .as_ref()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        expanded.resize(padded_rows * P::DEGREE, 0);
        let variables = P::DEGREE_ID.logn() + padded_rows.trailing_zeros() as usize;
        let point = (0..variables)
            .map(|index| {
                crate::substrate::OpeningField::from_u64(
                    (index as u64).wrapping_mul(0x9e37_79b9).wrapping_add(7),
                )
            })
            .collect::<Vec<_>>();
        let expected = crate::protocol::ring::target_evaluation(
            P::DEGREE_ID,
            padded_rows,
            live_rows,
            crate::protocol::ring::PublicTargets::Expanded(&expanded),
            &point,
        );
        let actual = crate::protocol::ring::target_evaluation(
            P::DEGREE_ID,
            padded_rows,
            live_rows,
            crate::protocol::ring::PublicTargets::<i16>::SourceBound(source),
            &point,
        );
        assert_eq!(actual, expected, "{}", P::NAME);
    }

    #[test]
    fn streamed_targets_match_scalar_hash_to_point_for_both_degrees() {
        source_bound_targets_match_native_scalar_of::<Falcon512>();
        source_bound_targets_match_native_scalar_of::<aerie_falcon_native::Falcon1024>();
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn streamed_target_matches_expanded_across_chunks_and_worker_counts() {
        let live_rows = 129_usize;
        let padded_rows = FalconDegree::Falcon512.padded_records(live_rows).unwrap();
        let public_key = zero_public_key();
        let salts = (0..live_rows)
            .map(|record| {
                core::array::from_fn(|index| record.wrapping_mul(31).wrapping_add(index) as u8)
            })
            .collect::<Vec<_>>();
        let messages = (0..live_rows)
            .map(|record| vec![record as u8; record % 37])
            .collect::<Vec<_>>();
        let hpks = vec![
            aerie_falcon_native::preprocess::HashedVerifyingKey::derive(&public_key);
            live_rows
        ];
        let source = PrehashedHashToPointBatch::new(&hpks, &salts, &messages).unwrap();
        let inputs = salts
            .iter()
            .zip(&messages)
            .map(|(salt, message)| {
                aerie_falcon_native::preprocess::HashToPointInput::new(&public_key, salt, message)
            })
            .collect::<Vec<_>>();
        let mut expanded = aerie_falcon_native::preprocess::hash_to_point_public_raw_batch(&inputs)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        expanded.resize(padded_rows * Falcon512::DEGREE, 0);
        let row_vars = padded_rows.trailing_zeros() as usize;
        let s1_point: Vec<_> = (0..crate::protocol::ring::COEFFICIENT_VARS + row_vars)
            .map(|index| OpeningField::from_u64((index as u64).wrapping_mul(17).wrapping_add(3)))
            .collect();
        let expected = target_evaluation(
            FalconDegree::Falcon512,
            padded_rows,
            live_rows,
            PublicTargets::Expanded(&expanded),
            &s1_point,
        );
        for workers in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap();
            let actual = pool.install(|| {
                target_evaluation(
                    FalconDegree::Falcon512,
                    padded_rows,
                    live_rows,
                    PublicTargets::<i16>::SourceBound(source),
                    &s1_point,
                )
            });
            assert_eq!(actual, expected, "{workers} Rayon workers");
        }
    }

    #[test]
    fn verifier_rejects_setup_record_mismatch_before_body_decode() {
        let verifier_setup = FalconVerifierSetup::<Falcon512>::new(1).unwrap();
        let mut header_only = b"aerie/falcon/v1p".to_vec();
        header_only.extend_from_slice(&FalconDegree::Falcon512.identity_bytes());
        header_only.extend_from_slice(&2_u64.to_le_bytes());
        assert_eq!(header_only.len(), wire::HEADER_BYTES);

        let statement = AggregationStatement::new(
            (0..2)
                .map(|_| {
                    aerie_falcon_statement::AggregationRecord::new(zero_public_key(), Vec::new())
                        .unwrap()
                })
                .collect(),
        )
        .unwrap();
        let error = verify_aggregate(&verifier_setup, &statement, &header_only)
            .expect_err("setup mismatch must reject before the missing body is decoded");
        assert!(matches!(
            error,
            Error::UnsupportedBatch(UnsupportedBatch::SetupSize {
                expected: 1,
                actual: 2
            })
        ));
        let shorter =
            AggregationStatement::new(vec![aerie_falcon_statement::AggregationRecord::new(
                zero_public_key(),
                Vec::new(),
            )
            .unwrap()])
            .unwrap();
        assert!(matches!(
            verify_aggregate(&verifier_setup, &shorter, &header_only),
            Err(Error::StatementMismatch(StatementMismatch::RecordCount {
                expected: 1,
                actual: 2
            }))
        ));
    }
}
