//! Section 8 verifier evaluation of the public ring functional.

use aerie_falcon_native::source::CanonicalPublicKeyBytes;
use aerie_falcon_native::{Falcon1024, Falcon512, FalconParams};
use aerie_falcon_statement::FALCON_Q;
use aerie_simd::{KeyUnpackPlan, SmallSignedDotPlan};
use jolt_field::One as _;
use jolt_field::Unreduced;
use jolt_field::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::substrate::OpeningField as E;

use super::delayed_base_inner_product;
#[cfg(test)]
use super::COEFFICIENT_VARS;
use crate::protocol::small_signed::SmallSignedFactors;

/// Verifier-side public-key representation.
///
/// Adversarial tests and prover-adjacent paths may retain a padded coefficient
/// table. The wire verifier streams the canonical key bytes it bound into the
/// public-records digest, unpacking each key once inside the functional.
#[derive(Clone, Copy, Debug)]
pub enum PublicRingKeys<'a, C = i64> {
    Expanded(&'a [C]),
    /// Explicit degree for expanded tables whose minimum-padded lengths coincide.
    ExpandedOf(super::super::FalconDegree, &'a [C]),
    /// Canonical Falcon-512 public-key bytes, live rows only.
    Canonical(&'a [CanonicalPublicKeyBytes<aerie_falcon_native::Falcon512>]),
    /// Canonical Falcon-1024 public-key bytes, live rows only.
    Canonical1024(&'a [CanonicalPublicKeyBytes<Falcon1024>]),
}

impl<C> PublicRingKeys<'_, C>
where
    C: Copy + Sync + Into<i64>,
{
    /// Validate the Falcon-512 representation against the live and padded
    /// relation domains. Canonical keys carry only live rows; their padded
    /// suffix is implicit.
    pub fn validate_shape(self, live_rows: usize, records: usize) -> Result<(), &'static str> {
        self.validate_shape_of(Falcon512::DEGREE, live_rows, records)
    }

    /// Validate the representation for one ring degree. Canonical keys are
    /// Falcon-512 bytes and reject any other degree.
    pub fn validate_shape_of(
        self,
        degree: usize,
        live_rows: usize,
        records: usize,
    ) -> Result<(), &'static str> {
        if let Self::ExpandedOf(expected, _) = self {
            if degree != expected.degree() {
                return Err("expanded public-key degree mismatch");
            }
        }
        match self {
            Self::Expanded(coefficients) | Self::ExpandedOf(_, coefficients) => {
                if coefficients.len() != records * degree {
                    return Err("public ring-key table has the wrong shape");
                }
                if coefficients[live_rows * degree..]
                    .iter()
                    .any(|&value| value.into() != 0)
                {
                    return Err("padded public ring-key rows must be zero");
                }
            }
            Self::Canonical(public_keys) => {
                if degree != Falcon512::DEGREE {
                    return Err("Falcon-512 canonical public-key bytes for another degree");
                }
                if public_keys.len() != live_rows {
                    return Err("canonical public-key count does not match the live rows");
                }
            }
            Self::Canonical1024(public_keys) => {
                if degree != Falcon1024::DEGREE {
                    return Err("Falcon-1024 canonical public-key bytes for another degree");
                }
                if public_keys.len() != live_rows {
                    return Err("canonical public-key count does not match the live rows");
                }
            }
        }
        Ok(())
    }

    /// Evaluate the Section 8 public functional in the representation's
    /// native form.
    #[must_use]
    pub fn functional_evaluation(
        self,
        records: usize,
        projection_point: &[E],
        stage3_point: &[E],
    ) -> E {
        match self {
            Self::Expanded(coefficients) | Self::ExpandedOf(_, coefficients) => {
                functional_evaluation(coefficients, records, projection_point, stage3_point)
            }
            Self::Canonical(public_keys) => functional_evaluation_canonical(
                public_keys,
                records,
                projection_point,
                stage3_point,
            ),
            Self::Canonical1024(public_keys) => {
                functional_evaluation_canonical_of::<Falcon1024, 1024>(
                    public_keys,
                    records,
                    projection_point,
                    stage3_point,
                    SmallSignedDotPlan::best_available(),
                )
            }
        }
    }
}

/// Evaluate one canonical key row four coefficients at a time, matching the
/// key's native seven-byte packing groups.
#[inline]
fn delayed_canonical_public_key_inner_product<P: FalconParams>(
    public_key: &CanonicalPublicKeyBytes<P>,
    factors: &[E],
    centering_correction: E,
) -> E {
    debug_assert_eq!(factors.len(), P::DEGREE);
    let mut accumulators = [<E as Unreduced>::SmallProduct::zero(); 4];
    for (coefficients, factors) in public_key
        .shifted_centered_coefficient_blocks()
        .zip(factors.chunks_exact(4))
    {
        accumulators[0] += factors[0].mul_u64_unreduced(u64::from(coefficients[0]));
        accumulators[1] += factors[1].mul_u64_unreduced(u64::from(coefficients[1]));
        accumulators[2] += factors[2].mul_u64_unreduced(u64::from(coefficients[2]));
        accumulators[3] += factors[3].mul_u64_unreduced(u64::from(coefficients[3]));
    }
    let accumulator = accumulators
        .into_iter()
        .fold(<E as Unreduced>::SmallProduct::zero(), |sum, part| {
            sum + part
        });
    E::reduce_small_product(accumulator) - centering_correction
}

/// The carry-recurrence values `Lambda_a`, Equations (8.9) to (8.11).
fn lambda_values(l_coefficient: &[E], n_coefficient: &[E]) -> Vec<E> {
    let bits = l_coefficient.len();
    let length = 1_usize << bits;
    debug_assert_eq!(n_coefficient.len(), bits);

    // All a values share their low-bit prefixes. Build that binary trie once
    // rather than replaying the carry recurrence independently at every leaf.
    let mut states = vec![[E::zero(); 2]; length];
    states[0] = [E::one(), E::zero()];
    let mut width = 1_usize;
    for (&n_coordinate, &l_coordinate) in n_coefficient.iter().zip(l_coefficient) {
        let product = n_coordinate * l_coordinate;
        let equal = E::one() - n_coordinate - l_coordinate + product + product;
        let l_only = l_coordinate - product;
        let n_only = n_coordinate - product;
        let (parents, children) = states.split_at_mut(width);
        for prefix in 0..width {
            let [carry_zero, carry_one] = parents[prefix];
            children[prefix] = [l_only * carry_zero, n_only * carry_zero + equal * carry_one];
            parents[prefix] = [equal * carry_zero + l_only * carry_one, n_only * carry_one];
        }
        width *= 2;
    }
    states
        .into_iter()
        .map(|[carry_zero, carry_one]| carry_zero - carry_one)
        .collect()
}

/// Pointwise product of two equality tables without materializing either
/// input table.
fn joint_eq_table(left: &[E], right: &[E]) -> Vec<E> {
    debug_assert_eq!(left.len(), right.len());
    let mut weights = vec![E::zero(); 1_usize << left.len()];
    weights[0] = E::one();
    let mut length = 1_usize;
    for (&left_coordinate, &right_coordinate) in left.iter().zip(right) {
        let high_factor = left_coordinate * right_coordinate;
        let low_factor = E::one() - left_coordinate - right_coordinate + high_factor;
        let (low, suffix) = weights.split_at_mut(length);
        let high = &mut suffix[..length];
        #[cfg(feature = "parallel")]
        if 2 * length >= 1 << 13 {
            low.par_iter_mut()
                .zip(high.par_iter_mut())
                .for_each(|(low, high)| {
                    let weight = *low;
                    *low = weight * low_factor;
                    *high = weight * high_factor;
                });
        } else {
            for (low, high) in low.iter_mut().zip(high) {
                let weight = *low;
                *low = weight * low_factor;
                *high = weight * high_factor;
            }
        }
        #[cfg(not(feature = "parallel"))]
        for (low, high) in low.iter_mut().zip(high) {
            let weight = *low;
            *low = weight * low_factor;
            *high = weight * high_factor;
        }
        length *= 2;
    }
    weights
}

fn functional_factors(
    coefficient_vars: usize,
    projection_point: &[E],
    stage3_point: &[E],
) -> (Vec<E>, Vec<E>) {
    let (l_coefficient, l_row) = projection_point.split_at(coefficient_vars);
    let (n_coefficient, n_row) = stage3_point.split_at(coefficient_vars);
    #[cfg(feature = "parallel")]
    return rayon::join(
        || {
            tracing::info_span!("stage3::ring_functional.lambda")
                .in_scope(|| lambda_values(l_coefficient, n_coefficient))
        },
        || {
            tracing::info_span!("stage3::ring_functional.row_weights")
                .in_scope(|| joint_eq_table(l_row, n_row))
        },
    );
    #[cfg(not(feature = "parallel"))]
    (
        tracing::info_span!("stage3::ring_functional.lambda")
            .in_scope(|| lambda_values(l_coefficient, n_coefficient)),
        tracing::info_span!("stage3::ring_functional.row_weights")
            .in_scope(|| joint_eq_table(l_row, n_row)),
    )
}

/// Linear-time verifier evaluation `w_ring~(r_N)`, Equation (8.11). The ring
/// degree is `h.len() / records`.
#[must_use]
pub fn functional_evaluation<C>(
    h: &[C],
    records: usize,
    projection_point: &[E],
    stage3_point: &[E],
) -> E
where
    C: Copy + Sync + Into<i64>,
{
    assert!(
        records > 0,
        "a functional evaluation needs at least one row"
    );
    let degree = h.len() / records;
    assert!(
        degree.is_power_of_two() && degree * records == h.len(),
        "public-key table shape"
    );
    let coefficient_vars = degree.trailing_zeros() as usize;
    let (lambdas, row_weights) =
        functional_factors(coefficient_vars, projection_point, stage3_point);
    let evaluate_record = |record: usize| {
        let h_row = &h[record * degree..(record + 1) * degree];
        let inner = delayed_base_inner_product(h_row, &lambdas);
        row_weights[record].mul_unreduced(inner)
    };
    #[cfg(feature = "parallel")]
    let accumulator = (0..records)
        .into_par_iter()
        .map(evaluate_record)
        .reduce(<E as Unreduced>::Product::zero, |left, right| left + right);
    #[cfg(not(feature = "parallel"))]
    let accumulator = (0..records)
        .map(evaluate_record)
        .fold(<E as Unreduced>::Product::zero(), |sum, term| sum + term);
    E::reduce_product(accumulator)
}

/// Linear-time verifier evaluation directly from canonical public-key bytes.
#[must_use]
pub fn functional_evaluation_canonical(
    public_keys: &[CanonicalPublicKeyBytes<aerie_falcon_native::Falcon512>],
    records: usize,
    projection_point: &[E],
    stage3_point: &[E],
) -> E {
    functional_evaluation_canonical_with_plan(
        public_keys,
        records,
        projection_point,
        stage3_point,
        SmallSignedDotPlan::best_available(),
    )
}

fn functional_evaluation_canonical_with_plan(
    public_keys: &[CanonicalPublicKeyBytes<aerie_falcon_native::Falcon512>],
    records: usize,
    projection_point: &[E],
    stage3_point: &[E],
    dot: Option<SmallSignedDotPlan>,
) -> E {
    functional_evaluation_canonical_of::<Falcon512, { Falcon512::DEGREE }>(
        public_keys,
        records,
        projection_point,
        stage3_point,
        dot,
    )
}

/// Linear-time verifier evaluation from canonical public-key bytes of any
/// parameter set; `N` is the ring degree `P::DEGREE`.
fn functional_evaluation_canonical_of<P: FalconParams, const N: usize>(
    public_keys: &[CanonicalPublicKeyBytes<P>],
    records: usize,
    projection_point: &[E],
    stage3_point: &[E],
    dot: Option<SmallSignedDotPlan>,
) -> E {
    debug_assert_eq!(P::DEGREE, N);
    debug_assert!(public_keys.len() <= records);
    let coefficient_vars = N.trailing_zeros() as usize;
    let (lambdas, row_weights) =
        functional_factors(coefficient_vars, projection_point, stage3_point);
    let packed_lambdas = dot.map(|dot| SmallSignedFactors::<N>::packed(&lambdas, dot));
    let centering_correction = packed_lambdas.is_none().then(|| {
        let lambda_sum = lambdas
            .iter()
            .copied()
            .fold(E::zero(), |sum, value| sum + value);
        E::reduce_small_product(
            lambda_sum.mul_u64_unreduced(u64::try_from((FALCON_Q - 1) / 2).expect("q is positive")),
        )
    });
    debug_assert_eq!(row_weights.len(), records);
    let unpack = KeyUnpackPlan::best_available();
    let evaluate_key = |record: usize, public_key: &CanonicalPublicKeyBytes<P>| {
        let inner = if let Some(factors) = &packed_lambdas {
            // Parsing checked the header and that every residue is below q;
            // the unpacker skips the header byte and centers each residue.
            let mut row = [0_i16; N];
            unpack.unpack_centered_14(&public_key.as_bytes().as_ref()[1..], &mut row);
            factors.dot_i16(&row)
        } else {
            delayed_canonical_public_key_inner_product(
                public_key,
                &lambdas,
                centering_correction.expect("field path has a centering correction"),
            )
        };
        row_weights[record].mul_unreduced(inner)
    };
    #[cfg(feature = "parallel")]
    let accumulator = tracing::info_span!("stage3::ring_functional.key_dot").in_scope(|| {
        const KEYS_PER_CHUNK: usize = 64;
        public_keys
            .par_chunks(KEYS_PER_CHUNK)
            .enumerate()
            .map(|(chunk, keys)| {
                let first = chunk * KEYS_PER_CHUNK;
                keys.iter().enumerate().fold(
                    <E as Unreduced>::Product::zero(),
                    |sum, (offset, public_key)| sum + evaluate_key(first + offset, public_key),
                )
            })
            .reduce(<E as Unreduced>::Product::zero, |left, right| left + right)
    });
    #[cfg(not(feature = "parallel"))]
    let accumulator = tracing::info_span!("stage3::ring_functional.key_dot").in_scope(|| {
        public_keys.iter().enumerate().fold(
            <E as Unreduced>::Product::zero(),
            |sum, (record, public_key)| sum + evaluate_key(record, public_key),
        )
    });
    E::reduce_product(accumulator)
}

/// Benchmark-only access to the canonical verifier kernel's field and SIMD paths.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
#[must_use]
pub fn functional_evaluation_canonical_for_bench(
    public_keys: &[CanonicalPublicKeyBytes<aerie_falcon_native::Falcon512>],
    records: usize,
    projection_point: &[E],
    stage3_point: &[E],
    dot: Option<SmallSignedDotPlan>,
) -> E {
    functional_evaluation_canonical_with_plan(
        public_keys,
        records,
        projection_point,
        stage3_point,
        dot,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::transcript::TestTranscript;
    use aerie_falcon_native::source::CanonicalPublicKeyBytes;

    use crate::protocol::algebra::{eq_table, sample_point};

    fn random_point(label: &[u8], count: usize) -> Vec<E> {
        sample_point(
            &mut TestTranscript::new(b"aerie/falcon/v1/ring-functional-test"),
            core::str::from_utf8(label).unwrap(),
            count,
        )
        .unwrap()
    }

    fn lambda_values_reference(l_coefficient: &[E], n_coefficient: &[E]) -> Vec<E> {
        let length = 1_usize << l_coefficient.len();
        (0..length)
            .map(|a| {
                let mut states = [E::one(), E::zero()];
                for (bit, (&n_coordinate, &l_coordinate)) in
                    n_coefficient.iter().zip(l_coefficient).enumerate()
                {
                    let alpha = (a >> bit) & 1;
                    let mut next = [E::zero(); 2];
                    for (carry, &state) in states.iter().enumerate() {
                        for j_bit in 0..2_usize {
                            let total = alpha + j_bit + carry;
                            let u = if j_bit == 1 {
                                n_coordinate
                            } else {
                                E::one() - n_coordinate
                            };
                            let w = if total & 1 == 1 {
                                l_coordinate
                            } else {
                                E::one() - l_coordinate
                            };
                            next[total >> 1] += state * u * w;
                        }
                    }
                    states = next;
                }
                states[0] - states[1]
            })
            .collect()
    }

    #[test]
    fn shared_prefix_lambda_trie_matches_independent_carry_recurrences() {
        let left = random_point(b"lambda trie left", COEFFICIENT_VARS);
        let right = random_point(b"lambda trie right", COEFFICIENT_VARS);
        for bits in 1..=COEFFICIENT_VARS {
            assert_eq!(
                lambda_values(&left[..bits], &right[..bits]),
                lambda_values_reference(&left[..bits], &right[..bits]),
                "carry trie differs at {bits} bits"
            );
        }
    }

    #[test]
    fn joint_equality_table_matches_pointwise_product() {
        let left = random_point(b"joint eq left", 8);
        let right = random_point(b"joint eq right", 8);
        let expected = eq_table(&left)
            .into_iter()
            .zip(eq_table(&right))
            .map(|(left, right)| left * right)
            .collect::<Vec<_>>();
        assert_eq!(joint_eq_table(&left, &right), expected);
    }

    fn canonical_keys_match_expanded<P: FalconParams, const N: usize>() {
        let live_rows = 3_usize;
        let records = 4_usize;
        let public_keys = (0..live_rows)
            .map(|record| {
                let mut encoded = vec![0_u8; P::PUBLIC_KEY_BYTES];
                encoded[0] = P::PUBLIC_KEY_HEADER;
                for (block, destination) in encoded[1..].chunks_exact_mut(7).enumerate() {
                    let residues: [u16; 4] = std::array::from_fn(|lane| {
                        let index = 4 * block + lane;
                        let residue = match index % 5 {
                            // Centering boundaries and the largest residue.
                            0 => [0, 6_144, 6_145, 12_288][(index / 5 + record) % 4],
                            _ => (record * 4_099 + index * 7_919 + 1_237) % FALCON_Q as usize,
                        };
                        u16::try_from(residue).expect("residue fits u16")
                    });
                    let word = (u64::from(residues[0]) << 42)
                        | (u64::from(residues[1]) << 28)
                        | (u64::from(residues[2]) << 14)
                        | u64::from(residues[3]);
                    destination.copy_from_slice(&word.to_be_bytes()[1..]);
                }
                CanonicalPublicKeyBytes::<P>::parse(&encoded).expect("canonical patterned key")
            })
            .collect::<Vec<_>>();
        let mut expanded = vec![0_i16; records * N];
        for (destination, public_key) in expanded.chunks_exact_mut(N).zip(&public_keys) {
            destination.copy_from_slice(public_key.decode().h_centered().as_ref());
        }
        let variables = N.trailing_zeros() as usize + records.trailing_zeros() as usize;
        let projection_point = random_point(b"canonical rL", variables);
        let stage3_point = random_point(b"canonical rN", variables);

        let expected = functional_evaluation(&expanded, records, &projection_point, &stage3_point);
        for dot in [
            None,
            Some(SmallSignedDotPlan::scalar()),
            SmallSignedDotPlan::neon(),
            SmallSignedDotPlan::avx2(),
            SmallSignedDotPlan::avx512(),
            SmallSignedDotPlan::best_available(),
        ] {
            assert_eq!(
                functional_evaluation_canonical_of::<P, N>(
                    &public_keys,
                    records,
                    &projection_point,
                    &stage3_point,
                    dot,
                ),
                expected,
                "{} with {:?}",
                P::NAME,
                dot.map(SmallSignedDotPlan::name),
            );
        }
    }

    #[test]
    fn canonical_public_keys_match_expanded_coefficients_with_implicit_padding() {
        canonical_keys_match_expanded::<Falcon512, 512>();
        canonical_keys_match_expanded::<Falcon1024, 1024>();
    }
}
