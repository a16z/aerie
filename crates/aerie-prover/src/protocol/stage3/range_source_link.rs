//! Link all six stage-two range images to the committed digit sources.

use crate::error::AggregationError;
use crate::protocol::kernels::range_source_kernel::{QuadraticGroup, QuadraticSource};
use akita_algebra::split_eq::GruenSplitEq;
use jolt_field::{Field, One, Ring, Zero};

use crate::protocol::{
    source_tables::{CodedSource, PreparedCoefficients, SourceTables},
    stage::evaluate_compressed,
    stage2::Stage2Output,
    FalconDegree,
};
use crate::substrate::OpeningField as E;

/// The two compact source kernels that implement the six canonical claims.
/// The coefficient kernel has the full stage point; the budget kernel has
/// only its signature suffix and is inactive during leading coefficient rounds.
pub(super) struct Provers<'a> {
    pub(super) coefficient: Prover<'a>,
    pub(super) budget: Prover<'a>,
}

/// Compact prefix output transferred into the fused dense sweep.
pub(super) struct DenseHandoff {
    pub(super) sources: Vec<E>,
    /// Equality-contracted coefficients for the first dense round, when one
    /// follows the third compact bind.
    pub(super) first_dense_coefficients: Option<[E; 2]>,
}

pub(super) struct Prover<'a> {
    kernel: Option<QuadraticGroup<'a>>,
    pub(super) equality: GruenSplitEq<E>,
    pub(super) claim: E,
    pub(super) scale: E,
}

impl<'a> Provers<'a> {
    pub(super) fn new(
        tables: &'a SourceTables,
        prepared: &PreparedCoefficients<'a>,
        degree: FalconDegree,
        stage2: &Stage2Output,
        gammas: &[E; 6],
    ) -> Result<Self, AggregationError> {
        let coefficient_vars = degree.logn();
        let rounds = stage2.point.len();
        let row_rounds = rounds - coefficient_vars;
        let coefficient_sources = (0..3)
            .map(|index| QuadraticSource::new(prepared.source(index), gammas[index]))
            .collect();
        let budget_coded = [
            CodedSource::new(tables.u0(), -4, 3)?,
            CodedSource::new(tables.u1(), -4, 3)?,
            CodedSource::new(
                tables.u2(),
                if degree == FalconDegree::Falcon1024 {
                    -8
                } else {
                    -4
                },
                if degree == FalconDegree::Falcon1024 {
                    7
                } else {
                    3
                },
            )?,
        ];
        let budget_sources = vec![
            QuadraticSource::new(&budget_coded[0], gammas[3]),
            QuadraticSource::new(&budget_coded[1], gammas[4]),
            QuadraticSource::new(&budget_coded[2], gammas[5]),
        ];
        let coefficient_claim = (0..3).fold(E::zero(), |sum, index| {
            sum + gammas[index] * stage2.folded_digits[index]
        });
        let budget_claim = (0..3).fold(E::zero(), |sum, index| {
            sum + gammas[index + 3] * stage2.folded_digits[index + 3]
        });
        let coefficient_kernel = QuadraticGroup::new(coefficient_sources, rounds)?;
        Ok(Self {
            coefficient: Prover {
                kernel: Some(coefficient_kernel),
                equality: GruenSplitEq::new(&stage2.point)?,
                claim: coefficient_claim,
                scale: E::one(),
            },
            budget: Prover {
                kernel: Some(QuadraticGroup::new(budget_sources, row_rounds)?),
                equality: GruenSplitEq::new(&stage2.point[coefficient_vars..])?,
                claim: budget_claim,
                scale: E::one(),
            },
        })
    }
}

impl Prover<'_> {
    pub(super) fn claim(&self) -> E {
        self.claim
    }

    pub(super) fn round_compact(&mut self) -> Result<[E; 3], AggregationError> {
        let (first, second) = self.equality.remaining_eq_tables();
        let [linear, quadratic] = self
            .kernel
            .as_mut()
            .ok_or_else(|| AggregationError::internal("range source is already dense"))?
            .round_coefficients(first, second)?;
        Ok(self.dense_message([linear, quadratic]))
    }

    /// Compact QG message and a second affine relation contribution derived
    /// from the same materialized compact source representation. The caller
    /// supplies a separate split-equality tensor and one linear term per QG
    /// source lane.
    pub(super) fn round_compact_with_linear(
        &mut self,
        linear_first: &[E],
        linear_second: &[E],
        linear_factors: &[E],
        linear_offsets: &[i64],
    ) -> Result<([E; 3], [E; 2]), AggregationError> {
        let (first, second) = self.equality.remaining_eq_tables();
        let (quadratic, linear) = self
            .kernel
            .as_mut()
            .ok_or_else(|| AggregationError::internal("range source is already dense"))?
            .round_coefficients_with_linear(
                first,
                second,
                linear_first,
                linear_second,
                linear_factors,
                linear_offsets,
            )?;
        Ok((self.dense_message(quadratic), linear))
    }

    pub(super) fn is_compact(&self) -> bool {
        self.kernel
            .as_ref()
            .is_some_and(QuadraticGroup::is_compact_prefix)
    }

    /// Reconstruct the relation's transmitted coefficients from pair sums
    /// produced by the fused dense sweep.
    pub(super) fn dense_message(&self, [linear, quadratic]: [E; 2]) -> [E; 3] {
        let tau = self.equality.current_tau();
        let scalar = self.equality.current_scalar();
        let scalar = scalar * self.scale;
        let constant = self.claim - scalar * tau * (linear + quadratic);
        let ell_constant = E::one() - tau;
        let ell_linear = tau + tau - E::one();
        let coefficients = if let Some(polynomial) =
            crate::protocol::algebra::eq_factored_cubic_from_constant_and_quadratic(
                tau,
                self.claim,
                constant,
                scalar * quadratic,
            ) {
            let coefficients = polynomial.coefficients();
            [coefficients[0], coefficients[2], coefficients[3]]
        } else {
            let linear = scalar * linear;
            [
                ell_constant * constant,
                ell_constant * scalar * quadratic + ell_linear * linear,
                ell_linear * scalar * quadratic,
            ]
        };
        coefficients
    }

    /// Apply a compact bind and transfer the materialized source lanes when
    /// the compact prefix ends. The third bind contracts the next dense range
    /// message while materializing those lanes.
    pub(super) fn bind_compact(
        &mut self,
        message: [E; 3],
        challenge: E,
    ) -> Result<Option<DenseHandoff>, AggregationError> {
        self.claim = evaluate_compressed(message, self.claim, challenge);
        self.equality.bind(challenge);
        let kernel = self
            .kernel
            .as_mut()
            .ok_or_else(|| AggregationError::internal("range source is already dense"))?;
        let next_round_equality = (kernel.rounds_completed() == 2 && kernel.num_rounds() > 3)
            .then(|| self.equality.remaining_eq_tables());
        let first_dense_coefficients = kernel.bind_compact(challenge, next_round_equality)?;
        if kernel.is_compact_prefix() {
            return Ok(None);
        }
        let sources = self
            .kernel
            .take()
            .ok_or_else(|| AggregationError::internal("range source was already transferred"))?
            .into_dense_sources()?;
        Ok(Some(DenseHandoff {
            sources,
            first_dense_coefficients,
        }))
    }

    /// Update only relation metadata after a dense message has been
    /// appended; the fused engine advances the source table separately.
    pub(super) fn bind_metadata(&mut self, message: [E; 3], challenge: E) {
        self.claim = evaluate_compressed(message, self.claim, challenge);
        self.equality.bind(challenge);
    }

    pub(super) fn halve_inactive(&mut self) {
        let half = E::from_u64(2).inverse().expect("two is invertible");
        self.claim *= half;
        self.scale *= half;
    }
}
#[cfg(feature = "internals")]
pub use aerie_falcon::protocol::stage3::range_source_link::terminal;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::algebra::{eq_evaluation, eq_table, evaluate_mle, fold};
    use crate::protocol::kernels::DigitSource;
    use crate::protocol::stage::evaluate_compressed;
    use crate::protocol::{stage2::Stage2Output, FalconWitness};
    use crate::substrate::BaseField as F;
    use aerie_falcon_native::Falcon1024;
    use aerie_falcon_witness::DirectRow;
    use jolt_field::One;

    fn k(a: u64, b: u64) -> E {
        E::new(F::from_u64(a), F::from_u64(b))
    }

    fn synthetic_batch(degree: FalconDegree) -> FalconWitness {
        synthetic_batch_with_rows(degree, 4)
    }

    fn synthetic_batch_with_rows(degree: FalconDegree, num_rows: usize) -> FalconWitness {
        match degree {
            FalconDegree::Falcon512 => {
                let rows = (0..num_rows)
                    .map(|seed| {
                        DirectRow::<aerie_falcon_native::Falcon512>::synthetic(seed as u64 + 811)
                            .expect("synthetic row")
                    })
                    .collect::<Vec<_>>();
                FalconWitness::from_rows(&rows)
            }
            FalconDegree::Falcon1024 => {
                let rows = (0..num_rows)
                    .map(|seed| {
                        DirectRow::<Falcon1024>::synthetic(seed as u64 + 811)
                            .expect("synthetic row")
                    })
                    .collect::<Vec<_>>();
                FalconWitness::from_rows(&rows)
            }
        }
    }

    fn lift(table: &[i8]) -> Vec<E> {
        table
            .iter()
            .map(|&value| E::from_i64(i64::from(value)))
            .collect()
    }

    fn lift_view(source: DigitSource<'_>) -> Vec<E> {
        (0..source.len())
            .map(|index| E::from_i64(i64::from(source.get(index))))
            .collect()
    }

    fn source_claim(table: &[E], point: &[E]) -> E {
        let quadratic = table
            .iter()
            .map(|&value| value * (value + E::one()))
            .collect::<Vec<_>>();
        evaluate_mle(&quadratic, point)
    }

    fn direct_round(table: &[E], gamma: E, point: &[E], challenges: &[E], x: E) -> E {
        let round = challenges.len();
        let suffix = eq_table(&point[round + 1..]);
        let prefix = gamma * eq_evaluation(&point[..round], challenges);
        let ell = E::one() - point[round] + x * (point[round] + point[round] - E::one());
        let inner = table
            .chunks_exact(2)
            .zip(&suffix)
            .fold(E::zero(), |sum, (pair, &weight)| {
                let value = pair[0] + x * (pair[1] - pair[0]);
                sum + weight * value * (value + E::one())
            });
        prefix * ell * inner
    }

    fn direct_dense_coefficients(
        sources: &[Vec<E>; 3],
        gammas: &[E; 3],
        equality_first: &[E],
        equality_second: &[E],
    ) -> [E; 2] {
        let mut result = [E::zero(); 2];
        for pair in 0..sources[0].len() / 2 {
            let equality = equality_first[pair % equality_first.len()]
                * equality_second[pair / equality_first.len()];
            for source in 0..3 {
                let left = sources[source][2 * pair];
                let right = sources[source][2 * pair + 1];
                let delta = right - left;
                result[0] += equality * gammas[source] * delta * (left + left + E::one());
                result[1] += equality * gammas[source] * delta * delta;
            }
        }
        result
    }

    #[test]
    fn grouped_source_rounds_match_each_compact_quadratic_relation() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            let batch = synthetic_batch(degree);
            let tables = SourceTables::build(&batch).expect("source tables");
            let coefficient_vars = degree.logn();
            let row_vars = batch.padded_rows.trailing_zeros() as usize;
            let point = (0..coefficient_vars)
                .map(|index| k(17 + index as u64, 31 + index as u64))
                .chain([E::zero(), E::one()])
                .collect::<Vec<_>>();
            let mut digit_tables =
                [tables.a0_view(), tables.a1_view(), tables.a2_view()].map(lift_view);
            let mut budget_tables = [tables.u0(), tables.u1(), tables.u2()].map(lift);
            let folded_digits = [
                source_claim(&digit_tables[0], &point),
                source_claim(&digit_tables[1], &point),
                source_claim(&digit_tables[2], &point),
                source_claim(&budget_tables[0], &point[coefficient_vars..]),
                source_claim(&budget_tables[1], &point[coefficient_vars..]),
                source_claim(&budget_tables[2], &point[coefficient_vars..]),
            ];
            let stage2 = Stage2Output {
                point: point.clone(),
                folded_digits,
            };
            let gammas = std::array::from_fn(|i| k(43 + i as u64, 61 + i as u64));
            let coefficients = PreparedCoefficients::new(&tables).unwrap();
            let mut provers =
                Provers::new(&tables, &coefficients, degree, &stage2, &gammas).unwrap();
            let mut coefficient_challenges = Vec::new();
            let mut coefficient_handoff = None;
            for stage_round in 0..3 {
                let coefficient_claim = provers.coefficient.claim();
                let coefficient_message = provers.coefficient.round_compact().unwrap();
                for x in [E::zero(), E::one(), k(73, 97)] {
                    let direct = (0..3).fold(E::zero(), |sum, source| {
                        sum + direct_round(
                            &digit_tables[source],
                            gammas[source],
                            &point,
                            &coefficient_challenges,
                            x,
                        )
                    });
                    assert_eq!(
                        evaluate_compressed(coefficient_message, coefficient_claim, x),
                        direct
                    );
                }
                let challenge = match stage_round % 3 {
                    0 => E::zero(),
                    1 => E::one(),
                    _ => k(137 + stage_round as u64, 163 + stage_round as u64),
                };
                let handoff = provers
                    .coefficient
                    .bind_compact(coefficient_message, challenge)
                    .unwrap();
                if stage_round == 2 {
                    coefficient_handoff = handoff;
                } else {
                    assert!(handoff.is_none());
                }
                for table in &mut digit_tables {
                    fold(table, challenge);
                }
                coefficient_challenges.push(challenge);
            }

            let dense_handoff = coefficient_handoff.expect("third bind materializes sources");
            assert!(dense_handoff.first_dense_coefficients.is_some());
            let dense_coefficients = dense_handoff.sources;
            let expected_coefficients = (0..digit_tables[0].len())
                .flat_map(|index| {
                    [
                        digit_tables[0][index],
                        digit_tables[1][index],
                        digit_tables[2][index],
                    ]
                })
                .collect::<Vec<_>>();
            assert_eq!(dense_coefficients, expected_coefficients);

            for _ in 0..coefficient_vars {
                provers.budget.halve_inactive();
            }
            let mut budget_challenges = Vec::new();
            let mut budget_handoff = None;
            let row_point = &point[coefficient_vars..];
            for row_round in 0..row_vars {
                let claim = provers.budget.claim();
                let (linear_first, linear_second) = provers.budget.equality.remaining_eq_tables();
                let (linear_first, linear_second) = (linear_first.to_vec(), linear_second.to_vec());
                let (linear_first, linear_second) =
                    (linear_first.as_slice(), linear_second.as_slice());
                let (message, [linear_constant, linear_slope]) = provers
                    .budget
                    .round_compact_with_linear(
                        linear_first,
                        linear_second,
                        &[E::one(), E::from_u64(8), E::from_u64(64)],
                        &[0; 3],
                    )
                    .unwrap();
                assert_eq!(provers.budget.round_compact().unwrap(), message);
                for x in [E::zero(), E::one(), k(103, 127)] {
                    let scale = super::super::inverse_power_of_two(coefficient_vars);
                    let direct = (0..3).fold(E::zero(), |sum, source| {
                        sum + scale
                            * direct_round(
                                &budget_tables[source],
                                gammas[source + 3],
                                row_point,
                                &budget_challenges,
                                x,
                            )
                    });
                    assert_eq!(evaluate_compressed(message, claim, x), direct);

                    let direct_linear = budget_tables
                        .iter()
                        .enumerate()
                        .flat_map(|(source, table)| {
                            table
                                .chunks_exact(2)
                                .enumerate()
                                .map(move |(pair, values)| {
                                    let weight = linear_first[pair % linear_first.len()]
                                        * linear_second[pair / linear_first.len()];
                                    weight
                                        * E::from_u64(1_u64 << (3 * source))
                                        * (values[0] + x * (values[1] - values[0]))
                                })
                        })
                        .fold(E::zero(), |sum, term| sum + term);
                    assert_eq!(linear_constant + x * linear_slope, direct_linear);
                }
                let challenge = match row_round {
                    0 => E::one(),
                    _ => k(137 + row_round as u64, 163 + row_round as u64),
                };
                let handoff = provers.budget.bind_compact(message, challenge).unwrap();
                if row_round + 1 == row_vars {
                    budget_handoff = handoff;
                } else {
                    assert!(handoff.is_none());
                }
                for table in &mut budget_tables {
                    fold(table, challenge);
                }
                budget_challenges.push(challenge);
            }
            let dense_handoff = budget_handoff.expect("terminal budget bind materializes sources");
            assert!(dense_handoff.first_dense_coefficients.is_none());
            let dense_budget = dense_handoff.sources;
            assert_eq!(
                dense_budget,
                [
                    budget_tables[0][0],
                    budget_tables[1][0],
                    budget_tables[2][0]
                ]
            );
            assert_eq!(row_vars, 2);
        }
    }

    #[test]
    fn third_bind_contracts_first_dense_range_round_for_record_boundaries() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            for num_rows in [2, 4, 8, 17] {
                let batch = synthetic_batch_with_rows(degree, num_rows);
                let tables = SourceTables::build(&batch).expect("source tables");
                let coefficient_vars = degree.logn();
                let row_vars = batch.padded_rows.trailing_zeros() as usize;
                let point = (0..coefficient_vars + row_vars)
                    .map(|index| match index % 4 {
                        0 => E::zero(),
                        1 => E::one(),
                        _ => k(211 + index as u64, 263 + index as u64),
                    })
                    .collect::<Vec<_>>();
                let mut source_tables =
                    [tables.a0_view(), tables.a1_view(), tables.a2_view()].map(lift_view);
                let folded_digits = [
                    source_claim(&source_tables[0], &point),
                    source_claim(&source_tables[1], &point),
                    source_claim(&source_tables[2], &point),
                    source_claim(&lift(tables.u0()), &point[coefficient_vars..]),
                    source_claim(&lift(tables.u1()), &point[coefficient_vars..]),
                    source_claim(&lift(tables.u2()), &point[coefficient_vars..]),
                ];
                let stage2 = Stage2Output {
                    point: point.clone(),
                    folded_digits,
                };
                let gammas = std::array::from_fn(|index| k(307 + index as u64, 359 + index as u64));
                let source_gammas = [gammas[0], gammas[1], gammas[2]];
                let coefficients = PreparedCoefficients::new(&tables).unwrap();
                let mut prover = Provers::new(&tables, &coefficients, degree, &stage2, &gammas)
                    .unwrap()
                    .coefficient;
                let mut challenges = Vec::new();
                let round_challenges = [E::zero(), E::one(), k(401, 463)];

                for (round, challenge) in round_challenges.into_iter().enumerate() {
                    let claim = prover.claim();
                    let message = prover.round_compact().unwrap();
                    for x in [E::zero(), E::one(), k(509, 563)] {
                        let direct = (0..3).fold(E::zero(), |sum, source| {
                            sum + direct_round(
                                &source_tables[source],
                                source_gammas[source],
                                &point,
                                &challenges,
                                x,
                            )
                        });
                        assert_eq!(evaluate_compressed(message, claim, x), direct);
                    }

                    let handoff = prover.bind_compact(message, challenge).unwrap();
                    for table in &mut source_tables {
                        fold(table, challenge);
                    }
                    challenges.push(challenge);
                    if round < 2 {
                        assert!(handoff.is_none());
                        continue;
                    }

                    let handoff = handoff.expect("third compact bind materializes sources");
                    let expected_sources = (0..source_tables[0].len())
                        .flat_map(|index| {
                            [
                                source_tables[0][index],
                                source_tables[1][index],
                                source_tables[2][index],
                            ]
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(handoff.sources, expected_sources);

                    let coefficients = handoff
                        .first_dense_coefficients
                        .expect("a dense round follows the third bind");
                    let (equality_first, equality_second) = prover.equality.remaining_eq_tables();
                    assert_eq!(
                        coefficients,
                        direct_dense_coefficients(
                            &source_tables,
                            &source_gammas,
                            equality_first,
                            equality_second,
                        )
                    );
                    let dense_message = prover.dense_message(coefficients);
                    let dense_claim = prover.claim();
                    for x in [E::zero(), E::one(), k(601, 653)] {
                        let direct = (0..3).fold(E::zero(), |sum, source| {
                            sum + direct_round(
                                &source_tables[source],
                                source_gammas[source],
                                &point,
                                &challenges,
                                x,
                            )
                        });
                        assert_eq!(evaluate_compressed(dense_message, dense_claim, x), direct);
                    }
                }
            }
        }
    }

    #[test]
    fn terminal_short_budget_handoffs_do_not_recontract_or_refold() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            for num_rows in [2, 4, 8] {
                let batch = synthetic_batch_with_rows(degree, num_rows);
                let tables = SourceTables::build(&batch).expect("source tables");
                let coefficient_vars = degree.logn();
                let row_vars = batch.padded_rows.trailing_zeros() as usize;
                let point = (0..coefficient_vars + row_vars)
                    .map(|index| k(701 + index as u64, 757 + index as u64))
                    .collect::<Vec<_>>();
                let digit_tables =
                    [tables.a0_view(), tables.a1_view(), tables.a2_view()].map(lift_view);
                let mut budget_tables = [tables.u0(), tables.u1(), tables.u2()].map(lift);
                let folded_digits = [
                    source_claim(&digit_tables[0], &point),
                    source_claim(&digit_tables[1], &point),
                    source_claim(&digit_tables[2], &point),
                    source_claim(&budget_tables[0], &point[coefficient_vars..]),
                    source_claim(&budget_tables[1], &point[coefficient_vars..]),
                    source_claim(&budget_tables[2], &point[coefficient_vars..]),
                ];
                let stage2 = Stage2Output {
                    point: point.clone(),
                    folded_digits,
                };
                let gammas = std::array::from_fn(|index| k(809 + index as u64, 863 + index as u64));
                let coefficients = PreparedCoefficients::new(&tables).unwrap();
                let mut provers =
                    Provers::new(&tables, &coefficients, degree, &stage2, &gammas).unwrap();
                for _ in 0..coefficient_vars {
                    provers.budget.halve_inactive();
                }

                let mut challenges = Vec::new();
                let mut final_handoff = None;
                let scale = super::super::inverse_power_of_two(coefficient_vars);
                for row_round in 0..row_vars {
                    let claim = provers.budget.claim();
                    let (first, second) = provers.budget.equality.remaining_eq_tables();
                    let (first, second) = (first.to_vec(), second.to_vec());
                    let (message, _) = provers
                        .budget
                        .round_compact_with_linear(
                            &first,
                            &second,
                            &[E::one(), E::one(), E::one()],
                            &[0; 3],
                        )
                        .unwrap();
                    assert_eq!(provers.budget.round_compact().unwrap(), message);
                    for x in [E::zero(), E::one(), k(887, 919)] {
                        let direct = (0..3).fold(E::zero(), |sum, source| {
                            sum + scale
                                * direct_round(
                                    &budget_tables[source],
                                    gammas[source + 3],
                                    &point[coefficient_vars..],
                                    &challenges,
                                    x,
                                )
                        });
                        assert_eq!(evaluate_compressed(message, claim, x), direct);
                    }
                    let challenge = match row_round % 3 {
                        0 => E::zero(),
                        1 => E::one(),
                        _ => k(907, 953),
                    };
                    let handoff = provers.budget.bind_compact(message, challenge).unwrap();
                    for table in &mut budget_tables {
                        fold(table, challenge);
                    }
                    challenges.push(challenge);
                    if row_round + 1 == row_vars {
                        final_handoff = handoff;
                    } else {
                        assert!(handoff.is_none());
                    }
                }

                let handoff = final_handoff.expect("short budget terminal bind materializes");
                assert!(handoff.first_dense_coefficients.is_none());
                let expected = (0..budget_tables[0].len())
                    .flat_map(|index| {
                        [
                            budget_tables[0][index],
                            budget_tables[1][index],
                            budget_tables[2][index],
                        ]
                    })
                    .collect::<Vec<_>>();
                assert_eq!(handoff.sources, expected);
                assert_eq!(challenges.len(), row_vars);
            }
        }
    }
}
