//! Producer materialization of the public ring functional.
use crate::protocol::kernels::SignSource;
use crate::substrate::OpeningField as E;
pub use aerie_falcon::protocol::ring::*;
mod exact_convolution;
/// Construct the fused public ring table from expanded keys and packed sign bits.
#[cfg(feature = "internals")]
#[doc(hidden)]
#[must_use]
pub fn functional_kernel_table_for_bench(
    h: &[i16],
    records: usize,
    live: usize,
    point: &[E],
    sigma: &[u64],
    magnitude: &[i16],
    row_scale: &[E],
) -> (Vec<E>, [E; 3]) {
    functional_kernel_table_i16_with_round(
        h,
        records,
        live,
        point,
        SignSource::Bits(sigma),
        magnitude,
        row_scale,
    )
}
/// Materialize the signed public ring table and its first raw coefficient
/// triple in the same row-generation pass. `row_scale` contains the existing
/// `gamma * eq(row_point, row)` factors; it scales the round message only.
#[must_use]
#[tracing::instrument(
    skip_all,
    name = "functional_kernel_table_round0",
    fields(
        cells = h.len(),
        storage_bytes = h.len().saturating_mul(core::mem::size_of::<E>())
    )
)]

pub(super) fn functional_kernel_table_i16_with_round(
    h: &[i16],
    records: usize,
    live_rows: usize,
    coefficient_point: &[E],
    sigma: SignSource<'_>,
    magnitude: &[i16],
    row_scale: &[E],
) -> (Vec<E>, [E; 3]) {
    assert!(live_rows <= records, "live rows exceed padded rows");
    match coefficient_point.len() {
        9 => exact_convolution::table_with_round0::<512>(
            h,
            records,
            live_rows,
            coefficient_point,
            sigma,
            magnitude,
            row_scale,
        ),
        10 => exact_convolution::table_with_round0::<1024>(
            h,
            records,
            live_rows,
            coefficient_point,
            sigma,
            magnitude,
            row_scale,
        ),
        _ => panic!("fused ring table supports Falcon-512 and Falcon-1024"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::algebra::{evaluate_mle, sample_point};
    use crate::protocol::transcript::TestTranscript;
    use aerie_falcon_native::params::FALCON_Q;
    use jolt_field::{One, Ring, Zero};

    fn random_point(label: &[u8], count: usize) -> Vec<E> {
        sample_point(
            &mut TestTranscript::new(b"aerie/falcon/v1/ring-test"),
            core::str::from_utf8(label).unwrap(),
            count,
        )
        .unwrap()
    }

    fn random_coefficients(seed: u64, count: usize) -> Vec<i16> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % FALCON_Q as u64) as i16 - 6144
            })
            .collect()
    }

    // Low-bit-first equality weights, computed from the Boolean definition.
    fn equality_weights(point: &[E]) -> Vec<E> {
        (0..1 << point.len())
            .map(|index| {
                point.iter().enumerate().fold(E::one(), |value, (bit, &r)| {
                    value
                        * if index & (1 << bit) == 0 {
                            E::one() - r
                        } else {
                            r
                        }
                })
            })
            .collect()
    }

    // The coefficient of s[j] in the evaluation of h(X)s(X) modulo X^n + 1.
    fn kernel_from_definition(h: &[i64], point: &[E]) -> Vec<E> {
        let degree = h.len();
        let weights = equality_weights(point);
        (0..degree)
            .map(|j| {
                h.iter()
                    .enumerate()
                    .fold(E::zero(), |sum, (a, &coefficient)| {
                        let term = E::from_i64(coefficient) * weights[(a + j) % degree];
                        if a + j < degree {
                            sum + term
                        } else {
                            sum - term
                        }
                    })
            })
            .collect()
    }

    #[test]
    fn compact_functional_widens_before_negation() {
        for bits in [1_usize, 9, 10] {
            let degree = 1 << bits;
            let mut compact = vec![0_i16; degree];
            compact[..2].copy_from_slice(&[17, i16::MIN]);
            let wide: Vec<_> = compact.iter().copied().map(i64::from).collect();
            let projection = random_point(b"compact kernel point", bits);
            let terminal = random_point(b"compact terminal point", bits);
            let expected = kernel_from_definition(&wide, &projection);
            assert_eq!(
                functional_evaluation(&compact, 1, &projection, &terminal),
                evaluate_mle(&expected, &terminal),
            );
            if bits >= 9 {
                let sigma = vec![0_i8; degree];
                let magnitude = vec![0_i16; degree];
                let (actual, round) = functional_kernel_table_i16_with_round(
                    &compact,
                    1,
                    1,
                    &projection,
                    SignSource::Direct(&sigma),
                    &magnitude,
                    &[E::one()],
                );
                assert_eq!(actual, expected);
                assert_eq!(round, [E::zero(); 3]);
            }
        }
    }

    #[test]
    fn materialized_table_matches_the_weighted_definition() {
        let bits = 9;
        let degree = 1 << bits;
        let records = 4;
        let h = random_coefficients(0x5441_424c, degree * records);
        let projection = random_point(b"direct table point", bits + 2);
        let sigma = vec![0_i8; h.len()];
        let magnitude = vec![0_i16; h.len()];
        let rows = equality_weights(&projection[bits..]);
        let (actual, round) = functional_kernel_table_i16_with_round(
            &h,
            records,
            records,
            &projection[..bits],
            SignSource::Direct(&sigma),
            &magnitude,
            &rows,
        );
        for row in 0..records {
            let wide: Vec<_> = h[row * degree..(row + 1) * degree]
                .iter()
                .copied()
                .map(i64::from)
                .collect();
            let expected = kernel_from_definition(&wide, &projection[..bits]);
            for (j, &value) in expected.iter().enumerate() {
                assert_eq!(actual[row * degree + j], value);
            }
        }
        assert_eq!(round, [E::zero(); 3]);
    }

    #[test]
    fn carry_recurrence_and_fused_table_match_the_definition_with_padding() {
        for bits in [9_usize, 10] {
            let degree = 1 << bits;
            let records = 4;
            let mut h = random_coefficients(0x5041_4444, degree * records);
            h[3 * degree..].fill(0);
            let wide: Vec<_> = h.iter().copied().map(i64::from).collect();
            let sigma = vec![0_i8; h.len()];
            let magnitude = vec![0_i16; h.len()];
            let mut transcript = TestTranscript::new(b"aerie/falcon/v1/ring-test");
            for _ in 0..3 {
                let projection =
                    sample_point(&mut transcript, "projection point", bits + 2).unwrap();
                let terminal = sample_point(&mut transcript, "terminal point", bits + 2).unwrap();
                let rows = equality_weights(&projection[bits..]);
                let mut expected = Vec::with_capacity(h.len());
                for row in wide.chunks_exact(degree) {
                    expected.extend(kernel_from_definition(row, &projection[..bits]));
                }
                let weighted: Vec<_> = expected
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| value * rows[i / degree])
                    .collect();
                let direct = evaluate_mle(&weighted, &terminal);
                assert_eq!(
                    functional_evaluation(&h, records, &projection, &terminal),
                    direct
                );
                assert_eq!(
                    functional_evaluation(&wide, records, &projection, &terminal),
                    direct
                );
                let (actual, round) = functional_kernel_table_i16_with_round(
                    &h,
                    records,
                    3,
                    &projection[..bits],
                    SignSource::Direct(&sigma),
                    &magnitude,
                    &rows,
                );
                assert_eq!(actual, expected);
                assert_eq!(round, [E::zero(); 3]);
                assert!(actual[3 * degree..].iter().all(Zero::is_zero));
            }
        }
    }
}
