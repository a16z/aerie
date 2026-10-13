//! Nine authenticated evaluations at the four groups' three native points.
use super::{manifest::GroupRole, stage3::Stage3Output, FalconDegree};
use crate::substrate::OpeningField as E;
#[derive(Clone, Debug, PartialEq)]
pub struct GroupOpening {
    pub role: GroupRole,
    pub point: Vec<E>,
    pub values: Vec<E>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct OpeningLedger {
    pub groups: Vec<GroupOpening>,
}
#[must_use]
pub fn opening_ledger(degree: FalconDegree, output: &Stage3Output) -> OpeningLedger {
    let point = &output.point;
    let v = &output.terminal_evaluations;
    OpeningLedger {
        groups: vec![
            GroupOpening {
                role: GroupRole::S1,
                point: point.clone(),
                values: vec![v[0]],
            },
            GroupOpening {
                role: GroupRole::Slack,
                point: point[degree.logn() - 2..].to_vec(),
                values: vec![v[1]],
            },
            GroupOpening {
                role: GroupRole::S2Sources,
                point: point.clone(),
                values: v[2..6].to_vec(),
            },
            GroupOpening {
                role: GroupRole::Budget,
                point: point[degree.logn()..].to_vec(),
                values: v[6..].to_vec(),
            },
        ],
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::algebra::{evaluate_mle, lift_table};
    use crate::protocol::manifest::{CommitmentManifest, GroupRole};
    use crate::protocol::pcs::{self, FalconVerifierSetup};
    use crate::protocol::public_inputs::PublicInputs;
    use crate::protocol::ring::{PublicRingKeys, PublicTargets};
    use crate::protocol::source_tables::{slack_group, SourceTables};
    use crate::protocol::AggregationParams;
    use crate::protocol::{FalconDegree, FalconWitness};
    use aerie_falcon_native::{Falcon1024, Falcon512};
    use aerie_falcon_statement::DirectRow;

    fn on_deep_stack<T: Send>(body: impl FnOnce() -> T + Send) -> T {
        std::thread::scope(|scope| {
            let handle = std::thread::Builder::new()
                .stack_size(512 * 1024 * 1024)
                .spawn_scoped(scope, move || {
                    #[cfg(feature = "parallel")]
                    {
                        let pool = rayon::ThreadPoolBuilder::new()
                            .num_threads(1)
                            .stack_size(512 * 1024 * 1024)
                            .build()
                            .expect("proof thread pool");
                        pool.install(body)
                    }
                    #[cfg(not(feature = "parallel"))]
                    body()
                })
                .expect("spawn PCS verifier");
            handle.join().expect("join PCS verifier")
        })
    }

    /// Native polynomial tables in canonical group and polynomial order.
    fn group_tables(batch: &FalconWitness, tables: &SourceTables) -> [Vec<Vec<i64>>; 4] {
        [
            vec![batch.s1.iter().copied().map(i64::from).collect()],
            vec![slack_group(batch).iter().copied().map(i64::from).collect()],
            [tables.a0_view(), tables.a1_view(), tables.a2_view()]
                .into_iter()
                .map(|cells| {
                    (0..cells.len())
                        .map(|index| i64::from(cells.get(index)))
                        .collect()
                })
                .chain(std::iter::once({
                    let signs = tables.sigma_view();
                    (0..signs.len())
                        .map(|index| i64::from(signs.get(index)))
                        .collect()
                }))
                .collect(),
            (0..3)
                .map(|index| {
                    tables
                        .budget(index)
                        .iter()
                        .copied()
                        .map(i64::from)
                        .collect()
                })
                .collect(),
        ]
    }

    fn verify_opening<P: AggregationParams>(
        batch: &FalconWitness,
        degree: FalconDegree,
        records: usize,
    ) -> OpeningLedger {
        let proof = on_deep_stack(|| {
            pcs::prove(batch, b"opening ledger fixture          ")
                .expect("production PCS prover")
                .0
        });
        on_deep_stack(|| {
            let verifier_setup = FalconVerifierSetup::<P>::new(records).expect("verifier setup");
            let public = PublicInputs {
                records,
                salt_mode: crate::protocol::wire::SaltMode::PerRecord,
                public_records: b"opening ledger fixture          ",
                h: PublicRingKeys::ExpandedOf(degree, &batch.h),
                t: PublicTargets::Expanded(&batch.t),
            };
            pcs::verify(&public, &verifier_setup, &proof)
                .expect("production proof verifies")
                .ledger
        })
    }

    #[test]
    fn every_group_claim_is_the_true_native_mle_at_its_shared_point() {
        for (degree, records) in [
            (FalconDegree::Falcon512, 3_usize),
            (FalconDegree::Falcon512, 8),
            (FalconDegree::Falcon1024, 3),
            (FalconDegree::Falcon1024, 4),
        ] {
            let batch = if degree == FalconDegree::Falcon512 {
                let rows = (0..records)
                    .map(|seed| {
                        DirectRow::<Falcon512>::synthetic(seed as u64 + 101).expect("synthetic row")
                    })
                    .collect::<Vec<_>>();
                let mut batch = FalconWitness::from_rows(&rows);
                crate::protocol::pad_witness(&mut batch).unwrap();
                batch
            } else {
                let rows = (0..records)
                    .map(|seed| {
                        DirectRow::<Falcon1024>::synthetic(seed as u64 + 101)
                            .expect("synthetic row")
                    })
                    .collect::<Vec<_>>();
                let mut batch = FalconWitness::from_rows(&rows);
                crate::protocol::pad_witness(&mut batch).unwrap();
                batch
            };
            let tables = SourceTables::build(&batch).expect("source tables");
            let ledger = if degree == FalconDegree::Falcon512 {
                verify_opening::<Falcon512>(&batch, degree, records)
            } else {
                verify_opening::<Falcon1024>(&batch, degree, records)
            };
            let manifest = CommitmentManifest::for_batch_of(degree, records).expect("manifest");
            assert_eq!(ledger.groups.len(), manifest.group_count());
            for (((opening, polynomials), descriptor), expected_role) in ledger
                .groups
                .iter()
                .zip(group_tables(&batch, &tables))
                .zip(manifest.groups())
                .zip([
                    GroupRole::S1,
                    GroupRole::Slack,
                    GroupRole::S2Sources,
                    GroupRole::Budget,
                ])
            {
                assert_eq!(opening.role, expected_role, "canonical group order");
                assert_eq!(opening.role, descriptor.role, "manifest role");
                assert_eq!(polynomials.len(), descriptor.logical_polynomials);
                assert_eq!(opening.values.len(), descriptor.logical_polynomials);
                assert_eq!(
                    opening.point.len(),
                    descriptor.logical_cells.trailing_zeros() as usize,
                    "{:?} per-polynomial point arity",
                    descriptor.role
                );
                for (polynomial, &value) in polynomials.iter().zip(&opening.values) {
                    assert_eq!(polynomial.len(), descriptor.logical_cells);
                    assert_eq!(
                        value,
                        evaluate_mle(&lift_table(polynomial), &opening.point),
                        "{:?} must authenticate every native polynomial at records {records}",
                        descriptor.role
                    );
                }
            }
        }
    }
}
