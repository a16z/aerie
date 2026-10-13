//! Narrow signed source tables and polynomial-major field-polynomial materialization.

use std::sync::OnceLock;

use akita_error::AkitaError;
use akita_pcs::DensePoly;
use jolt_field::{One as _, Ring as _, Zero as _};

use crate::protocol::kernels::{DigitSource, SignSource};
use crate::substrate::{BaseField, OpeningField};

use super::{group_polynomial, Error, GroupRole};

/// Cells decoded per step of the narrow field conversion.
const FIELD_CONVERSION_CHUNK: usize = 1 << 12;

/// One homogeneous group of signed tables. Multi-polynomial variants retain
/// the producer's narrow representation and materialize each polynomial
/// separately, in the same order as the manifest group.
pub(super) enum SignedGroup<'a> {
    I16(&'a [i16]),
    MultiI8Three([&'a [i8]; 3]),
    /// The `s2` source digits and signs, in the source tables' storage.
    S2Sources {
        digits: [DigitSource<'a>; 3],
        signs: SignSource<'a>,
    },
    /// Adversarial edits are stored as one logical table per polynomial.
    OwnedI64(Vec<Vec<i64>>),
}

impl SignedGroup<'_> {
    /// Logical cells in each polynomial. All members of a group have the
    /// same shape, as required by Akita's `PolynomialGroupLayout`.
    pub(super) fn logical_cells(&self) -> usize {
        match self {
            Self::I16(cells) => cells.len(),
            Self::MultiI8Three(polys) => equal_poly_len(polys),
            Self::S2Sources { digits, signs } => {
                let cells = signs.len();
                debug_assert!(digits.iter().all(|digits| digits.len() == cells));
                cells
            }
            Self::OwnedI64(polys) => equal_poly_len(polys),
        }
    }

    pub(super) fn polynomial_count(&self) -> usize {
        match self {
            Self::I16(_) => 1,
            Self::MultiI8Three(_) => 3,
            Self::S2Sources { .. } => 4,
            Self::OwnedI64(polys) => polys.len(),
        }
    }

    /// Materialize one Akita source polynomial per logical table, preserving
    /// polynomial-major order within a group.
    pub(super) fn polynomials(&self) -> Result<Vec<DensePoly<BaseField>>, Error> {
        match self {
            Self::I16(cells) => Ok(vec![group_polynomial(cells)?]),
            Self::MultiI8Three(_) | Self::S2Sources { .. } => i8_group_polynomials(
                self.logical_cells(),
                self.polynomial_count(),
                |poly, start, output| {
                    self.decode_i8(poly, start, output);
                },
            ),
            Self::OwnedI64(polys) => Ok(polys
                .iter()
                .map(|cells| group_polynomial(cells))
                .collect::<Result<_, _>>()?),
        }
    }

    /// Decode cells `start..start + output.len()` of polynomial `poly` of a
    /// narrow `i8` group.
    fn decode_i8(&self, poly: usize, start: usize, output: &mut [i8]) {
        match self {
            Self::MultiI8Three(polys) => {
                DigitSource::Direct(polys[poly]).decode_into(start, output);
            }
            Self::S2Sources { digits, signs } => match digits.get(poly) {
                Some(digits) => digits.decode_into(start, output),
                None => signs.decode_into(start, output),
            },
            Self::I16(_) | Self::OwnedI64(_) => unreachable!("not a narrow i8 group"),
        }
    }

    /// Apply the adversarial group edit to a flattened polynomial-major
    /// vector and return sparse deltas per polynomial. Keeping the group's
    /// arity and each polynomial's logical length fixed ensures the emitted
    /// commitments still match the verifier-derived group profile.
    pub(super) fn tamper(
        &mut self,
        role: GroupRole,
        tamper: &dyn Fn(GroupRole, &mut Vec<i64>),
    ) -> Result<Vec<Vec<(usize, i64)>>, Error> {
        let original = self.to_i64_polynomials();
        let logical_cells = self.logical_cells();
        let expected_len = logical_cells
            .checked_mul(original.len())
            .ok_or("adversarial group cell count overflow")?;
        if logical_cells == 0 {
            return Err(format!("adversarial group {role:?} has no logical cells").into());
        }
        let mut flattened: Vec<i64> = original.iter().flatten().copied().collect();
        debug_assert_eq!(flattened.len(), expected_len);
        tamper(role, &mut flattened);
        if flattened.len() != expected_len {
            return Err(format!(
                "adversarial edit for {role:?} changed the polynomial-major cell count from {expected_len} to {}",
                flattened.len()
            )
            .into());
        }
        let edited: Vec<Vec<i64>> = flattened
            .chunks_exact(logical_cells)
            .map(<[i64]>::to_vec)
            .collect();
        let deltas = original
            .iter()
            .zip(&edited)
            .map(|(before, after)| {
                before
                    .iter()
                    .zip(after)
                    .enumerate()
                    .filter_map(|(index, (&before, &after))| {
                        (before != after).then_some((index, after - before))
                    })
                    .collect()
            })
            .collect();
        *self = Self::OwnedI64(edited);
        Ok(deltas)
    }

    fn to_i64_polynomials(&self) -> Vec<Vec<i64>> {
        match self {
            Self::I16(cells) => vec![cells.iter().copied().map(i64::from).collect()],
            Self::MultiI8Three(_) | Self::S2Sources { .. } => (0..self.polynomial_count())
                .map(|poly| {
                    let mut cells = vec![0; self.logical_cells()];
                    self.decode_i8(poly, 0, &mut cells);
                    cells.into_iter().map(i64::from).collect()
                })
                .collect(),
            Self::OwnedI64(polys) => polys.clone(),
        }
    }
}

static I8_FIELD_VALUES: OnceLock<[BaseField; 256]> = OnceLock::new();

fn i8_field_values() -> &'static [BaseField; 256] {
    I8_FIELD_VALUES.get_or_init(|| {
        std::array::from_fn(|byte| BaseField::from_i64(i64::from(byte as u8 as i8)))
    })
}

/// Materialize the small signed tables concurrently while keeping the
/// field conversion out of the per-cell hot path. Each polynomial is decoded
/// chunk by chunk directly into its zero-padded evaluation vector, so no
/// full-length narrow copy is made. `DensePoly`'s own centered i8 scan
/// remains part of construction because Akita uses that mirror in its
/// small-coefficient commit kernels.
fn i8_group_polynomials(
    logical_cells: usize,
    polynomial_count: usize,
    decode: impl Fn(usize, usize, &mut [i8]) + Sync,
) -> Result<Vec<DensePoly<BaseField>>, Error> {
    if polynomial_count == 0 {
        return Err(Error::from("an i8 group must contain a polynomial"));
    }
    let physical_len = super::pcs_length(logical_cells);
    let num_vars = physical_len.trailing_zeros() as usize;
    let field_values = i8_field_values();
    let convert = |poly: usize, chunk: usize, output: &mut [BaseField]| {
        let mut cells = [0_i8; FIELD_CONVERSION_CHUNK];
        let cells = &mut cells[..output.len()];
        decode(poly, chunk * FIELD_CONVERSION_CHUNK, cells);
        for (value, &cell) in output.iter_mut().zip(cells.iter()) {
            *value = field_values[usize::from(cell as u8)];
        }
    };

    let evaluations = tracing::info_span!(
        "v1.commit.group.field_conversion",
        logical_cells,
        physical_cells = physical_len,
        polynomial_count,
        carrier = "i8"
    )
    .in_scope(|| {
        let polynomial = |poly: usize| {
            let mut evaluations = vec![BaseField::zero(); physical_len];
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                evaluations[..logical_cells]
                    .par_chunks_mut(FIELD_CONVERSION_CHUNK)
                    .enumerate()
                    .for_each(|(chunk, output)| convert(poly, chunk, output));
            }
            #[cfg(not(feature = "parallel"))]
            for (chunk, output) in evaluations[..logical_cells]
                .chunks_mut(FIELD_CONVERSION_CHUNK)
                .enumerate()
            {
                convert(poly, chunk, output);
            }
            evaluations
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            (0..polynomial_count)
                .into_par_iter()
                .map(polynomial)
                .collect::<Vec<_>>()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..polynomial_count).map(polynomial).collect::<Vec<_>>()
        }
    });

    let polynomials = tracing::info_span!(
        "v1.commit.group.dense_poly",
        logical_cells,
        physical_cells = physical_len,
        polynomial_count,
        carrier = "i8"
    )
    .in_scope(|| {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            evaluations
                .into_par_iter()
                .map(|evaluations| DensePoly::from_field_evals(num_vars, evaluations))
                .collect::<Result<Vec<_>, AkitaError>>()
        }
        #[cfg(not(feature = "parallel"))]
        {
            evaluations
                .into_iter()
                .map(|evaluations| DensePoly::from_field_evals(num_vars, evaluations))
                .collect::<Result<Vec<_>, AkitaError>>()
        }
    })
    .map_err(Error::from)?;

    Ok(polynomials)
}

/// The multilinear extension of sparse cell deltas at a physical opening
/// point, low index bit first: `sum_j delta_j * eq(j, point)`.
pub(super) fn delta_evaluation(deltas: &[(usize, i64)], point: &[OpeningField]) -> OpeningField {
    deltas
        .iter()
        .fold(OpeningField::zero(), |sum, &(index, delta)| {
            let eq = point.iter().enumerate().fold(
                OpeningField::one(),
                |product, (bit, &coordinate)| {
                    if (index >> bit) & 1 == 1 {
                        product * coordinate
                    } else {
                        product * (OpeningField::one() - coordinate)
                    }
                },
            );
            sum + OpeningField::from_i64(delta) * eq
        })
}

fn equal_poly_len<T, C: AsRef<[T]>>(polys: &[C]) -> usize {
    let Some(first) = polys.first() else {
        return 0;
    };
    let cells = first.as_ref().len();
    debug_assert!(polys.iter().all(|poly| poly.as_ref().len() == cells));
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i8_field_lookup_matches_direct_conversion_for_every_byte() {
        let lookup = i8_field_values();
        for byte in u8::MIN..=u8::MAX {
            let value = byte as i8;
            assert_eq!(
                lookup[usize::from(byte)],
                BaseField::from_i64(i64::from(value)),
                "signed byte {value}"
            );
        }
    }

    #[test]
    fn compact_s2_group_materializes_like_its_direct_cells() {
        // Crosses several conversion chunks and the u64 sign-word boundary.
        let cells = 1 << 14;
        let minima = [-4_i8, -8, -8];
        let codes: [Vec<u8>; 3] = std::array::from_fn(|source| {
            (0..cells / 2)
                .map(|pair| {
                    let modulus = if source == 0 { 8 } else { 16 };
                    let digit = |index: usize| ((index * 7 + source * 3) % modulus) as u8;
                    digit(2 * pair) * 16 + digit(2 * pair + 1)
                })
                .collect()
        });
        let words: Vec<u64> = (0..cells / 64)
            .map(|word| 0x9e37_79b9_7f4a_7c15_u64.rotate_left(word as u32))
            .collect();
        let digits: [DigitSource<'_>; 3] = std::array::from_fn(|source| DigitSource::Packed {
            codes: &codes[source],
            minimum: minima[source],
        });
        let signs = SignSource::Bits(&words);
        let direct: Vec<Vec<i8>> = (0..4)
            .map(|poly| {
                let mut cells = vec![0; cells];
                match digits.get(poly) {
                    Some(digits) => digits.decode_into(0, &mut cells),
                    None => signs.decode_into(0, &mut cells),
                }
                cells
            })
            .collect();
        let compact = SignedGroup::S2Sources { digits, signs };
        let wide = SignedGroup::S2Sources {
            digits: std::array::from_fn(|poly| DigitSource::Direct(&direct[poly])),
            signs: SignSource::Direct(&direct[3]),
        };
        assert_eq!(compact.logical_cells(), cells);
        assert_eq!(
            compact.polynomials().expect("compact s2 group"),
            wide.polynomials().expect("direct s2 group")
        );
        assert_eq!(
            compact.to_i64_polynomials(),
            direct
                .iter()
                .map(|cells| cells.iter().copied().map(i64::from).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn i8_group_materialization_preserves_values_order_and_zero_padding() {
        let values: [Vec<i8>; 3] = [
            (u8::MIN..=u8::MAX).map(|byte| byte as i8).collect(),
            (u8::MIN..=u8::MAX).rev().map(|byte| byte as i8).collect(),
            (0_u8..=u8::MAX)
                .map(|byte| byte.wrapping_add(71) as i8)
                .collect(),
        ];
        let group = SignedGroup::MultiI8Three([
            values[0].as_slice(),
            values[1].as_slice(),
            values[2].as_slice(),
        ]);
        let polynomials = group.polynomials().expect("valid i8 source group");
        let physical_len = super::super::pcs_length(values[0].len());

        assert_eq!(polynomials.len(), 3);
        for (polynomial, source) in polynomials.iter().zip(&values) {
            let coefficients = polynomial.field_coeffs();
            assert_eq!(coefficients.len(), physical_len.max(1024));
            for (&actual, &value) in coefficients.iter().zip(source) {
                assert_eq!(actual, BaseField::from_i64(i64::from(value)));
            }
            assert!(coefficients[source.len()..]
                .iter()
                .all(|&coefficient| coefficient == BaseField::zero()));
        }
    }
}
