//! Stage two: one normalized shared-equality sumcheck for all range leaves.
pub mod range_leaf;
use super::transcript::PioTranscript;
use super::{
    source_tables::{CodedSource, PreparedCoefficients, SourceTables},
    stage1::Stage1Output,
    FalconDegree,
};
use crate::protocol::kernels::quartic_kernel::{QuarticGroup, QuarticSource, WeightedQuartic};
use crate::substrate::OpeningField as E;
use akita_algebra::split_eq::GruenSplitEq;
use akita_error::AkitaError;
use jolt_field::Zero;
use range_leaf::{advance, instances};
const GAMMA_SITE: &str = "aerie/falcon/v2/stage2/gamma";
const TERMINAL_SITE: &str = "aerie/falcon/v2/stage2/terminal";
#[derive(Clone, Debug, PartialEq)]
pub struct Stage2Proof {
    pub rounds: Vec<[E; 4]>,
    pub folded_digits: [E; 6],
}
#[derive(Clone, Debug, PartialEq)]
pub struct Stage2Output {
    pub point: Vec<E>,
    pub folded_digits: [E; 6],
}
fn invalid(s: &str) -> AkitaError {
    AkitaError::InvalidInput(s.into())
}
fn setup(
    degree: FalconDegree,
    records: usize,
    stage1: &Stage1Output,
    t: &mut impl PioTranscript,
) -> Result<(Vec<E>, E), AkitaError> {
    if !records.is_power_of_two()
        || stage1.point.len() != degree.logn() + records.trailing_zeros() as usize
        || stage1.basis16_claims.len() != 2 + usize::from(degree == FalconDegree::Falcon1024)
    {
        return Err(invalid("invalid range leaf input shape"));
    }
    let relations = instances(degree);
    let gamma = t.challenge_extensions(GAMMA_SITE, relations.len())?;
    let claim = stage1
        .basis16_claims
        .iter()
        .flatten()
        .zip(&gamma)
        .fold(E::zero(), |s, (&v, &g)| s + g * v);
    Ok((gamma, claim))
}
#[cfg(any(test, feature = "bench-internals", feature = "adversarial-internals"))]
pub fn prove(
    degree: FalconDegree,
    tables: &SourceTables,
    stage1: &Stage1Output,
    t: &mut impl PioTranscript,
) -> Result<(Stage2Proof, Stage2Output), AkitaError> {
    let prepared = PreparedCoefficients::new(tables)?;
    prove_prepared(degree, tables, &prepared, stage1, t)
}

pub(crate) fn prove_prepared(
    degree: FalconDegree,
    tables: &SourceTables,
    prepared: &PreparedCoefficients<'_>,
    stage1: &Stage1Output,
    t: &mut impl PioTranscript,
) -> Result<(Stage2Proof, Stage2Output), AkitaError> {
    let records = tables.records();
    let (gamma, mut claim) = setup(degree, records, stage1, t)?;
    let relations = instances(degree);
    let budget_sources = [
        CodedSource::new(tables.budget(0), -4, 3)?,
        CodedSource::new(tables.budget(1), -4, 3)?,
        CodedSource::new(
            tables.budget(2),
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
    let terms = |i: usize| {
        relations
            .iter()
            .zip(&gamma)
            .filter(|((index, _), _)| *index == i)
            .map(|((_, polynomial), &weight)| WeightedQuartic {
                polynomial: *polynomial,
                weight,
            })
            .collect()
    };
    let rounds = stage1.point.len();
    let mut coefficients = QuarticGroup::new(
        (0..3)
            .map(|i| QuarticSource::new(prepared.source(i), terms(i)))
            .collect(),
        rounds,
    )?;
    let mut budget = QuarticGroup::new(
        (3..6)
            .map(|i| QuarticSource::new(&budget_sources[i - 3], terms(i)))
            .collect(),
        rounds - degree.logn(),
    )?;
    let mut equality = GruenSplitEq::new(&stage1.point)?;
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let (first, second) = equality.remaining_eq_tables();
        let mut message = coefficients.round_coefficients(first, second)?;
        if round >= degree.logn() {
            for (v, w) in message
                .iter_mut()
                .zip(budget.round_coefficients(first, second)?)
            {
                *v += w;
            }
        }
        t.absorb_extensions(
            &format!("aerie/falcon/v2/stage2/round/{round}/message"),
            &message,
        )?;
        let challenge = t.challenge_extensions(
            &format!("aerie/falcon/v2/stage2/round/{round}/challenge"),
            1,
        )?[0];
        claim = advance(claim, stage1.point[round], message, challenge);
        coefficients.bind(challenge)?;
        if round >= degree.logn() {
            budget.bind(challenge)?;
        }
        equality.bind(challenge);
        point.push(challenge);
        messages.push(message);
    }
    let mut folded_digits = [E::zero(); 6];
    folded_digits[..3].copy_from_slice(coefficients.terminal_images()?);
    folded_digits[3..].copy_from_slice(budget.terminal_images()?);
    check_terminal(degree, &gamma, &folded_digits, claim)?;
    t.absorb_extensions(TERMINAL_SITE, &folded_digits)?;
    Ok((
        Stage2Proof {
            rounds: messages,
            folded_digits,
        },
        Stage2Output {
            point,
            folded_digits,
        },
    ))
}
pub fn verify(
    degree: FalconDegree,
    records: usize,
    stage1: &Stage1Output,
    proof: &Stage2Proof,
    t: &mut impl PioTranscript,
) -> Result<Stage2Output, AkitaError> {
    let (gamma, mut claim) = setup(degree, records, stage1, t)?;
    if proof.rounds.len() != stage1.point.len() {
        return Err(invalid("invalid range leaf round count"));
    }
    let mut point = Vec::with_capacity(proof.rounds.len());
    for (round, (&message, &alpha)) in proof.rounds.iter().zip(&stage1.point).enumerate() {
        t.absorb_extensions(
            &format!("aerie/falcon/v2/stage2/round/{round}/message"),
            &message,
        )?;
        let challenge = t.challenge_extensions(
            &format!("aerie/falcon/v2/stage2/round/{round}/challenge"),
            1,
        )?[0];
        claim = advance(claim, alpha, message, challenge);
        point.push(challenge);
    }
    check_terminal(degree, &gamma, &proof.folded_digits, claim)?;
    t.absorb_extensions(TERMINAL_SITE, &proof.folded_digits)?;
    Ok(Stage2Output {
        point,
        folded_digits: proof.folded_digits,
    })
}
fn check_terminal(
    degree: FalconDegree,
    gamma: &[E],
    values: &[E; 6],
    claim: E,
) -> Result<(), AkitaError> {
    let expected = instances(degree)
        .iter()
        .zip(gamma)
        .fold(E::zero(), |s, (&(index, q), &g)| {
            s + g * q.evaluate(values[index])
        });
    if claim != expected {
        return Err(invalid("range leaf terminal equation failed"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::kernels::quartic_kernel::QuarticPolynomial;
    use crate::protocol::kernels::DigitSource;
    use crate::protocol::{
        algebra::{eq_table, sample_point},
        transcript::TestTranscript,
        FalconWitness,
    };
    use aerie_falcon_native::{Falcon1024, Falcon512};
    use aerie_falcon_statement::{DirectParams, DirectRow};
    use jolt_field::{One, Ring};

    fn check<P: DirectParams>(degree: FalconDegree) {
        let rows = (0..3)
            .map(|i| DirectRow::<P>::synthetic(i + 29).unwrap())
            .collect::<Vec<_>>();
        let batch = FalconWitness::from_rows(&rows);
        let tables = SourceTables::build(&batch).unwrap();
        let mut point = sample_point(
            &mut TestTranscript::new(b"range leaf reference"),
            "point",
            degree.logn() + 2,
        )
        .unwrap();
        // Equality endpoints are legal transcript values; normalization must
        // not divide by either the coordinate or the equality prefix.
        point[0] = E::zero();
        point[1] = E::one();
        point[degree.logn()] = E::zero();
        let decode = |source: DigitSource<'_>| {
            let mut digits = vec![0; source.len()];
            source.decode_into(0, &mut digits);
            digits
        };
        let mut basis_sources = vec![decode(tables.a1_view()), decode(tables.a2_view())];
        if degree == FalconDegree::Falcon1024 {
            basis_sources.push(tables.u2().to_vec());
        }
        let basis16_claims = basis_sources
            .iter()
            .enumerate()
            .map(|(index, source)| {
                let weights = eq_table(if index == 2 {
                    &point[degree.logn()..]
                } else {
                    &point
                });
                [QuarticPolynomial::Low, QuarticPolynomial::High].map(|q| {
                    source
                        .iter()
                        .zip(&weights)
                        .fold(E::zero(), |sum, (&digit, &weight)| {
                            let z = E::from_i64(i64::from(digit));
                            sum + weight * q.evaluate(z * (z + E::one()))
                        })
                })
            })
            .collect();
        let first = Stage1Output {
            point,
            basis16_claims,
            projection_point: Vec::new(),
            slack_projection_point: Vec::new(),
            projection_claims: [E::zero(); 3],
        };
        let (proof, output) = prove(
            degree,
            &tables,
            &first,
            &mut TestTranscript::new(b"stage2 endpoints"),
        )
        .unwrap();
        let verified = verify(
            degree,
            batch.padded_rows,
            &first,
            &proof,
            &mut TestTranscript::new(b"stage2 endpoints"),
        )
        .unwrap();
        assert_eq!(output, verified);
        for (index, source) in [
            decode(tables.a0_view()),
            decode(tables.a1_view()),
            decode(tables.a2_view()),
            tables.u0().to_vec(),
            tables.u1().to_vec(),
            tables.u2().to_vec(),
        ]
        .into_iter()
        .enumerate()
        {
            let weights = eq_table(if index < 3 {
                &output.point
            } else {
                &output.point[degree.logn()..]
            });
            let expected = source
                .iter()
                .zip(weights)
                .fold(E::zero(), |sum, (&digit, weight)| {
                    let z = E::from_i64(i64::from(digit));
                    sum + weight * z * (z + E::one())
                });
            assert_eq!(proof.folded_digits[index], expected, "range image {index}");
        }
    }

    #[test]
    fn normalized_range_leaves_match_native_mles_at_equality_endpoints() {
        check::<Falcon512>(FalconDegree::Falcon512);
        check::<Falcon1024>(FalconDegree::Falcon1024);
    }
}
