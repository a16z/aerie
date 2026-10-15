//! Producer folding kernels.
use crate::protocol::kernels::{DigitSource, SignSource};
use crate::substrate::OpeningField as E;
pub use aerie_falcon::protocol::algebra::*;
#[cfg(feature = "parallel")]
use jolt_field::Fold;
use jolt_field::{Field, One as _, Ring as _, Zero as _};
use jolt_poly::UnivariatePoly;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
const MIN_PARALLEL_CELLS: usize = 1 << 13;
/// The small-alphabet lookup shared by standalone compact folds and the
/// integrated stage-three materialize-and-contract sweep. Construct once per
/// challenge; each caller may write its output directly into a disjoint lane.
pub(crate) struct LiftedFoldPlan {
    lookup: Option<LiftedFoldLookup>,
    challenge: E,
}

struct LiftedFoldLookup {
    minimum: i64,
    span: i64,
    lows: Vec<E>,
    deltas: Vec<E>,
}

impl LiftedFoldPlan {
    pub(crate) fn new<C>(table: &[C], challenge: E) -> Self
    where
        C: Copy + Sync + Into<i64>,
    {
        Self::new_indexed(table.len(), |index| table[index].into(), challenge)
    }

    pub(crate) fn new_digit(table: DigitSource<'_>, challenge: E) -> Self {
        match table {
            // A nibble code bounds the alphabet without a scan.
            DigitSource::Packed { minimum, .. } => Self::with_bounds(
                table.len(),
                i64::from(minimum),
                i64::from(minimum) + 15,
                challenge,
            ),
            DigitSource::Direct(cells) => Self::new(cells, challenge),
        }
    }

    pub(crate) fn new_sign(table: SignSource<'_>, challenge: E) -> Self {
        match table {
            SignSource::Bits(_) => Self::with_bounds(table.len(), -1, 1, challenge),
            #[cfg(any(test, feature = "internals"))]
            SignSource::Direct(cells) => Self::new(cells, challenge),
        }
    }

    fn new_indexed(len: usize, get: impl Fn(usize) -> i64 + Sync, challenge: E) -> Self {
        if len < 1 << 13 {
            return Self {
                lookup: None,
                challenge,
            };
        }
        #[cfg(feature = "parallel")]
        let (min, max) = (0..len)
            .into_par_iter()
            .map(&get)
            .fold(
                || (i64::MAX, i64::MIN),
                |(min, max), cell| (min.min(cell), max.max(cell)),
            )
            .reduce(|| (i64::MAX, i64::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)));
        #[cfg(not(feature = "parallel"))]
        let (min, max) = (0..len).fold((i64::MAX, i64::MIN), |(min, max), index| {
            let cell = get(index);
            (min.min(cell), max.max(cell))
        });
        Self::with_bounds(len, min, max, challenge)
    }

    /// Every lookup entry is an exact field value, so any alphabet bound that
    /// contains the table yields the same folds as its exact range.
    fn with_bounds(len: usize, min: i64, max: i64, challenge: E) -> Self {
        let lookup = if len >= 1 << 13 {
            max.checked_sub(min)
                .filter(|&span| span <= 65_535 && (3 * span as usize + 2) <= len / 4)
                .map(|span| {
                    let mut value = E::from_i64(min);
                    let lows = (0..=span)
                        .map(|_| {
                            let old = value;
                            value += E::one();
                            old
                        })
                        .collect();
                    let mut value = E::from_i64(-span) * challenge;
                    let deltas = (0..=2 * span)
                        .map(|_| {
                            let old = value;
                            value += challenge;
                            old
                        })
                        .collect();
                    LiftedFoldLookup {
                        minimum: min,
                        span,
                        lows,
                        deltas,
                    }
                })
        } else {
            None
        };
        Self { lookup, challenge }
    }

    /// Both values must belong to the source alphabet used by `new`.
    /// The integrated engine builds a separate plan for each compact lane.
    #[inline]
    pub(crate) fn fold_pair(&self, low: i64, high: i64) -> E {
        if let Some(lookup) = &self.lookup {
            lookup.lows[(low - lookup.minimum) as usize]
                + lookup.deltas[(high - low + lookup.span) as usize]
        } else {
            E::from_i64(low) + E::from_i64(high - low) * self.challenge
        }
    }
}

/// Sum a per-pair round contribution over the low-bit pairs of a domain.
#[inline]
pub(crate) fn pair_sums<const D: usize, C>(half: usize, contribution: C) -> [E; D]
where
    C: Fn(usize) -> [E; D] + Send + Sync,
{
    #[cfg(feature = "parallel")]
    if 2 * half >= MIN_PARALLEL_CELLS {
        return (0..half).into_par_iter().map(&contribution).reduce(
            || [E::zero(); D],
            |mut left, right| {
                for (slot, value) in left.iter_mut().zip(right) {
                    *slot += value;
                }
                left
            },
        );
    }
    let mut sums = [E::zero(); D];
    for pair_index in 0..half {
        for (slot, value) in sums.iter_mut().zip(contribution(pair_index)) {
            *slot += value;
        }
    }
    sums
}

/// Halve a table by binding its low variable.
///
/// Large tables collect disjoint output pairs in parallel. This avoids the
/// sequential prefix and repeated Rayon barriers of overlapping in-place
/// waves; indexed collection initializes each destination cell exactly once.
/// Small tables retain their allocation and fold in place.
pub(crate) fn fold(table: &mut Vec<E>, challenge: E) {
    let half = table.len() / 2;
    #[cfg(feature = "parallel")]
    if table.len() >= MIN_PARALLEL_CELLS {
        let context = E::precompute(challenge);
        *table = table
            .par_chunks_exact(2)
            .map(|pair| E::fold_one(&context, pair[0], pair[1]))
            .collect();
        return;
    }
    for index in 0..half {
        table[index] = table[2 * index] + (table[2 * index + 1] - table[2 * index]) * challenge;
    }
    table.truncate(half);
}

/// Degree-three eq-factored reconstruction from the constant and quadratic
/// coefficients of the inner polynomial.
///
/// For `s(X) = ell(X) q(X)`, `deg(ell) = 1`, and `deg(q) <= 2`, a pair
/// traversal can accumulate only `q_0` and `q_2`. The current claim
/// `s(0) + s(1)` determines `q(1)`, hence the omitted `q_1`. This
/// coefficient-form reconstruction matches Akita's
/// Gruen `try_gruen_poly_deg_3` path.
pub(crate) fn eq_factored_cubic_from_constant_and_quadratic(
    tau: E,
    previous_claim: E,
    q_constant: E,
    q_quadratic: E,
) -> Option<UnivariatePoly<E>> {
    if tau == E::zero() || tau == E::one() {
        return None;
    }
    let ell_constant = E::one() - tau;
    let ell_linear = tau + tau - E::one();
    let q_at_one = (previous_claim - ell_constant * q_constant)
        * tau.inverse().expect("nonzero tau has an inverse");
    let q_linear = q_at_one - q_constant - q_quadratic;
    Some(UnivariatePoly::new(vec![
        ell_constant * q_constant,
        ell_constant * q_linear + ell_linear * q_constant,
        ell_constant * q_quadratic + ell_linear * q_linear,
        ell_linear * q_quadratic,
    ]))
}

/// One fold applied directly to a signed-integer table, producing the
/// half-size lifted table without ever materializing the full-size one.
/// Field-exact match of `fold` after a centered lift: the lift is a ring
/// homomorphism and `(1 - r)a + rb = a + (b - a)r`, so the output bytes
/// equal the eager lift-then-fold path's.
pub(crate) fn fold_lifted_i64<C>(table: &[C], challenge: E) -> Vec<E>
where
    C: Copy + Sync + Into<i64>,
{
    let plan = LiftedFoldPlan::new(table, challenge);
    let pair = |pair: &[C]| plan.fold_pair(pair[0].into(), pair[1].into());
    #[cfg(feature = "parallel")]
    if table.len() >= MIN_PARALLEL_CELLS {
        return table.par_chunks_exact(2).map(pair).collect();
    }
    table.chunks_exact(2).map(pair).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::transcript::TestTranscript;
    use crate::substrate::BaseField as F;

    fn line_points<const M: usize>(v0: E, v1: E) -> [E; M] {
        let slope = v1 - v0;
        let mut out = [v0; M];
        let mut current = v1;
        for slot in out.iter_mut().skip(1) {
            current += slope;
            *slot = current;
        }
        out
    }

    #[test]
    fn compact_fold_plan_matches_affine_formula_at_alphabet_boundaries() {
        let challenges = [
            E::zero(),
            E::one(),
            E::new(F::from_u64(17), F::from_u64(31)),
        ];
        for len in [2, 8_192] {
            for kind in 0..3 {
                let table: Vec<i16> = (0..len)
                    .map(|index| match kind {
                        0 => i16::MIN + (index % 8) as i16,
                        1 => i16::MAX - (index % 8) as i16,
                        _ => {
                            if index % 2 == 0 {
                                i16::MIN
                            } else {
                                i16::MAX
                            }
                        }
                    })
                    .collect();
                for challenge in challenges {
                    let plan = LiftedFoldPlan::new(&table, challenge);
                    for pair in table.chunks_exact(2) {
                        let [low, high] = [i64::from(pair[0]), i64::from(pair[1])];
                        assert_eq!(
                            plan.fold_pair(low, high),
                            E::from_i64(low) * (E::one() - challenge)
                                + E::from_i64(high) * challenge,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn in_place_fold_matches_the_pairwise_definition() {
        // Sizes on both sides of the parallel threshold, including
        // large parallel collections and the serial prefix.
        let mut transcript = TestTranscript::new(b"in-place fold test");
        let challenge = sample(&mut transcript, "challenge").unwrap();
        for log_len in [1, 5, 12, 13, 14, 17] {
            let table = sample_point(&mut transcript, "table", 1 << log_len).unwrap();
            let expected: Vec<E> = table
                .chunks_exact(2)
                .map(|pair| pair[0] + (pair[1] - pair[0]) * challenge)
                .collect();
            let mut folded = table;
            fold(&mut folded, challenge);
            assert_eq!(folded, expected, "log_len {log_len}");
        }
    }

    #[test]
    fn eq_factored_reconstruction_matches_the_direct_round_polynomial() {
        // One eq-weighted cubic relation over eight cells: the factored
        // path must reproduce the direct evaluation bit for bit.
        let mut transcript = TestTranscript::new(b"eq factored test");
        let point = sample_point(&mut transcript, "eq-point", 3).unwrap();
        let table = sample_point(&mut transcript, "table", 8).unwrap();
        let equality = eq_table(&point);
        const D: usize = 3;
        let direct = pair_sums::<D, _>(equality.len() / 2, |pair| {
            let low = 2 * pair;
            let high = low + 1;
            let weight = line_points::<D>(equality[low], equality[high]);
            let source = line_points::<D>(table[low], table[high]);
            let mut out = [E::zero(); D];
            for slot in 0..D {
                out[slot] = weight[slot] * source[slot] * (source[slot] + E::one());
            }
            out
        });
        // The true claim is the full sum of the relation over the cube.
        let claim: E = equality
            .iter()
            .zip(&table)
            .map(|(&w, &s)| w * s * (s + E::one()))
            .fold(E::zero(), |sum, term| sum + term);
        let direct_poly =
            UnivariatePoly::from_evals(&[direct[0], claim - direct[0], direct[1], direct[2]]);
        let [q_constant, q_quadratic] = pair_sums::<2, _>(equality.len() / 2, |pair| {
            let low = 2 * pair;
            let high = low + 1;
            let weight = equality[low] + equality[high];
            let constant = table[low];
            let slope = table[high] - constant;
            [
                weight * constant * (constant + E::one()),
                weight * slope * slope,
            ]
        });
        let coefficient_factored =
            eq_factored_cubic_from_constant_and_quadratic(point[0], claim, q_constant, q_quadratic)
                .expect("challenge coordinates are neither zero nor one");
        assert_eq!(
            coefficient_factored.coefficients(),
            direct_poly.coefficients()
        );
    }

    #[test]
    fn folded_lift_matches_wide_arithmetic_for_compact_signed_tables() {
        let challenge = sample(&mut TestTranscript::new(b"compact fold test"), "r").unwrap();
        let compact = [i16::MIN, i16::MAX, -1234, 5678];
        let wide = compact.map(i64::from);
        assert_eq!(
            fold_lifted_i64(&compact, challenge),
            fold_lifted_i64(&wide, challenge),
        );
        assert_eq!(lift_table(&compact), lift_table(&wide));

        let compact = [i32::MIN, i32::MAX, -123_456_789, 987_654_321];
        let wide = compact.map(i64::from);
        assert_eq!(
            fold_lifted_i64(&compact, challenge),
            fold_lifted_i64(&wide, challenge),
        );
        assert_eq!(lift_table(&compact), lift_table(&wide));
    }

    #[test]
    fn alphabet_fold_matches_direct_arithmetic_for_all_signed_byte_pairs() {
        let mut transcript = TestTranscript::new(b"alphabet fold test");
        let mut table = Vec::with_capacity(2 * 256 * 256);
        for low in i8::MIN..=i8::MAX {
            for high in i8::MIN..=i8::MAX {
                table.extend([low, high]);
            }
        }
        for challenge in [E::zero(), E::one(), sample(&mut transcript, "r").unwrap()] {
            let expected: Vec<_> = table
                .chunks_exact(2)
                .map(|pair| {
                    let low = i64::from(pair[0]);
                    let high = i64::from(pair[1]);
                    E::from_i64(low) + E::from_i64(high - low) * challenge
                })
                .collect();
            assert_eq!(fold_lifted_i64(&table, challenge), expected);
        }
        let challenge = sample(&mut transcript, "r").unwrap();
        for (offset, width) in [
            (-1200, 2400),
            (9000, 2),
            (-13, 0),
            (i64::from(i32::MIN), 1 << 31),
        ] {
            let table: Vec<_> = (0..1 << 17)
                .map(|i| offset + (i * 37) % (width + 1))
                .collect();
            let expected: Vec<_> = table
                .chunks_exact(2)
                .map(|p| E::from_i64(p[0]) + E::from_i64(p[1] - p[0]) * challenge)
                .collect();
            assert_eq!(fold_lifted_i64(&table, challenge), expected);
        }
    }
}
