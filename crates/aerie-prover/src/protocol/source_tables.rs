//! Batch-level committed source tables of the Falcon protocols.
//!
//! One [`SourceTables`] holds `a0`, `a1`, `a2`, `sigma`, and the three budget
//! polynomials `u0`, `u1`, and `u2`. Each source remains an independent table.
//! Falcon-512 uses budget radices `(8, 8, 8)`; Falcon-1024 uses `(8, 8, 16)`.

use aerie_falcon_native::params::FixedArray;
use aerie_falcon_native::{Falcon1024, Falcon512, FalconParams};
use aerie_falcon_witness::native::encodability::FourCellEncodabilityWitness;
#[cfg(any(test, feature = "internals"))]
use aerie_falcon_witness::native::encodability::A2_OFFSET;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::sync::Arc;

use super::kernels::kernel_common::encode_pairs;
use super::kernels::{DigitSource, SignSource};
use super::{FalconDegree, FalconWitness};

/// A failure while constructing or checking the committed source tables.
#[derive(Debug)]
#[non_exhaustive]
pub enum SourceTableError {
    Unsupported(crate::UnsupportedBatch),
    Shape,
    Alphabet {
        minimum: i8,
        maximum: i8,
    },
    Digit {
        index: usize,
        digit: i8,
        minimum: i8,
        maximum: i8,
    },
    CompactAlphabet {
        minimum: i8,
        maximum: i8,
    },
    Backend(crate::BackendError),
    SourceAlphabet {
        cell: usize,
    },
    SourceDecode {
        cell: usize,
    },
    BudgetAlphabet {
        row: usize,
    },
    BudgetEquation {
        row: usize,
    },
    RecordLength {
        record: usize,
    },
    Encodability {
        record: usize,
        source: crate::StatementError,
    },
}
impl std::fmt::Display for SourceTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(error) => error.fmt(f),
            Self::Alphabet { minimum, maximum } => {
                write!(f, "unsupported source alphabet [{minimum}, {maximum}]")
            }
            Self::Digit {
                index,
                digit,
                minimum,
                maximum,
            } => write!(
                f,
                "source digit {digit} at index {index} is outside [{minimum}, {maximum}]"
            ),
            Self::CompactAlphabet { minimum, maximum } => {
                write!(f, "compact source digit is outside [{minimum}, {maximum}]")
            }
            Self::Backend(error) => error.fmt(f),
            Self::Shape => f.write_str("source table shape mismatch"),
            Self::SourceAlphabet { cell } => {
                write!(f, "source cell {cell} is outside its alphabet")
            }
            Self::SourceDecode { cell } => write!(f, "source cell {cell} does not decode to s2"),
            Self::BudgetAlphabet { row } => write!(f, "budget row {row} is outside its alphabet"),
            Self::BudgetEquation { row } => write!(f, "budget equation failed at row {row}"),
            Self::RecordLength { record } => {
                write!(f, "source record {record} has incorrect length")
            }
            Self::Encodability { record, .. } => {
                write!(f, "source record {record} is not encodable")
            }
        }
    }
}
impl std::error::Error for SourceTableError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Unsupported(e) => Some(e),
            Self::Encodability { source, .. } => Some(source),
            Self::Backend(error) => Some(error),
            _ => None,
        }
    }
}
impl From<crate::UnsupportedBatch> for SourceTableError {
    fn from(error: crate::UnsupportedBatch) -> Self {
        Self::Unsupported(error)
    }
}
impl From<SourceTableError> for crate::AggregationError {
    fn from(error: SourceTableError) -> Self {
        match error {
            SourceTableError::Unsupported(e) => e.into(),
            SourceTableError::Encodability { record, source } => {
                Self::InvalidStatement(source.at_record(record))
            }
            SourceTableError::Backend(error) => Self::Backend(error),
            other => Self::backend(other),
        }
    }
}

/// Exact per-batch source tables for the committed coefficient digits and
/// budget digits.
#[derive(Clone, Debug)]
pub struct SourceTables {
    degree: FalconDegree,
    records: usize,
    coefficients: CoefficientCells,
    signs: SignCells,
    u: [Vec<i8>; 3],
}

#[derive(Clone, Debug)]
enum CoefficientCells {
    /// Canonical storage: raw radix-16 and image pair codes, shared with the
    /// prepared [`CodedSource`]s without copying.
    Compact {
        raw16: [Arc<[u8]>; 3],
        image8: [Arc<[u8]>; 3],
    },
    #[cfg(any(test, feature = "internals"))]
    Wide([Vec<i8>; 3]),
}

#[derive(Clone, Debug)]
enum SignCells {
    Bits(Vec<u64>),
    #[cfg(any(test, feature = "internals"))]
    Wide(Vec<i8>),
}

const COEFFICIENT_MINIMA: [i8; 3] = [-4, -8, -8];
const COEFFICIENT_MAXIMA: [i8; 3] = [3, 7, 7];

/// A digit source bundled with the compact codes derived from it.
///
/// Construction validates the digit alphabet and is the only place kernels
/// obtain pair codes. Canonical coefficient sources share the codes already
/// stored in [`SourceTables`]; direct cells (budget digits and adversarial
/// hooks) are encoded here. Cloning shares the immutable codes.
#[derive(Clone, Debug)]
pub(crate) struct CodedSource<'a> {
    digits: DigitSource<'a>,
    minimum: i8,
    maximum: i8,
    image8: Arc<[u8]>,
    raw16: Arc<[u8]>,
}

/// Coefficient encodings prepared once before the timed PIOP stages.
pub(crate) struct PreparedCoefficients<'a> {
    sources: [CodedSource<'a>; 3],
    #[cfg(any(test, feature = "internals"))]
    sigma_is_sign: bool,
}

impl<'a> CodedSource<'a> {
    /// Validate and encode direct `i8` cells.
    pub(crate) fn new(
        digits: &'a [i8],
        minimum: i8,
        maximum: i8,
    ) -> Result<Self, SourceTableError> {
        Self::from_digits(DigitSource::Direct(digits), minimum, maximum)
    }

    fn from_digits(
        digits: DigitSource<'a>,
        minimum: i8,
        maximum: i8,
    ) -> Result<Self, SourceTableError> {
        check_alphabet(minimum, maximum)?;
        let in_range = |index: usize| {
            let digit = digits.get(index);
            if (minimum..=maximum).contains(&digit) {
                Ok(())
            } else {
                Err(SourceTableError::Digit {
                    index,
                    digit,
                    minimum,
                    maximum,
                })
            }
        };
        for index in 0..digits.len() {
            in_range(index)?;
        }
        let image8 = encode_pairs(digits.len() / 2, |pair| {
            Ok(image_pair_code(
                digits.get(2 * pair),
                digits.get(2 * pair + 1),
            ))
        })
        .map_err(|error| SourceTableError::Backend(crate::BackendError::new(error)))?;
        let raw16 = encode_pairs(digits.len() / 2, |pair| {
            Ok(raw_pair_code(
                digits.get(2 * pair),
                digits.get(2 * pair + 1),
                minimum,
            ))
        })
        .map_err(|error| SourceTableError::Backend(crate::BackendError::new(error)))?;
        Ok(Self {
            digits,
            minimum,
            maximum,
            image8: image8.into(),
            raw16: raw16.into(),
        })
    }

    /// Adopt the compact codes produced by [`SourceTables::build`]. Only the
    /// raw nibbles are rescanned, and only when the alphabet is narrower than
    /// sixteen slots; the image codes come from the same encoder.
    fn from_compact(
        raw16: &'a Arc<[u8]>,
        image8: &Arc<[u8]>,
        minimum: i8,
        maximum: i8,
    ) -> Result<Self, SourceTableError> {
        check_alphabet(minimum, maximum)?;
        if raw16.len() != image8.len() {
            return Err(SourceTableError::Shape);
        }
        let top = u8::try_from(i16::from(maximum) - i16::from(minimum))
            .expect("checked alphabet width fits u8");
        if top < 15
            && raw16
                .iter()
                .any(|&code| code >> 4 > top || code & 0x0f > top)
        {
            return Err(SourceTableError::CompactAlphabet { minimum, maximum });
        }
        Ok(Self {
            digits: DigitSource::Packed {
                codes: raw16,
                minimum,
            },
            minimum,
            maximum,
            image8: Arc::clone(image8),
            raw16: Arc::clone(raw16),
        })
    }

    #[cfg(test)]
    pub(crate) fn digits(&self) -> Vec<i8> {
        let mut digits = vec![0; self.digits.len()];
        self.digits.decode_into(0, &mut digits);
        digits
    }

    pub(crate) const fn cell_count(&self) -> usize {
        self.digits.len()
    }

    #[inline]
    pub(crate) fn digit(&self, index: usize) -> Option<i8> {
        (index < self.digits.len()).then(|| self.digits.get(index))
    }

    pub(crate) const fn minimum(&self) -> i8 {
        self.minimum
    }

    pub(crate) const fn maximum(&self) -> i8 {
        self.maximum
    }

    #[cfg(test)]
    pub(crate) fn image8(&self) -> &[u8] {
        &self.image8
    }

    pub(crate) fn image8_codes(&self) -> Arc<[u8]> {
        Arc::clone(&self.image8)
    }

    #[cfg(test)]
    pub(crate) fn raw16(&self) -> &[u8] {
        &self.raw16
    }

    pub(crate) fn raw16_codes(&self) -> Arc<[u8]> {
        Arc::clone(&self.raw16)
    }
}

impl<'a> PreparedCoefficients<'a> {
    pub(crate) fn new(tables: &'a SourceTables) -> Result<Self, SourceTableError> {
        let source = |index: usize| match &tables.coefficients {
            CoefficientCells::Compact { raw16, image8 } => CodedSource::from_compact(
                &raw16[index],
                &image8[index],
                COEFFICIENT_MINIMA[index],
                COEFFICIENT_MAXIMA[index],
            ),
            #[cfg(any(test, feature = "internals"))]
            CoefficientCells::Wide(cells) => CodedSource::new(
                &cells[index],
                COEFFICIENT_MINIMA[index],
                COEFFICIENT_MAXIMA[index],
            ),
        };
        Ok(Self {
            sources: [source(0)?, source(1)?, source(2)?],
            // Bit-packed signs are in {-1, 1} by construction; adversarial
            // direct cells may hold other values.
            #[cfg(any(test, feature = "internals"))]
            sigma_is_sign: match tables.sigma_view() {
                SignSource::Bits(_) => true,
                #[cfg(any(test, feature = "internals"))]
                SignSource::Direct(cells) => cells.iter().all(|&sign| matches!(sign, -1 | 1)),
            },
        })
    }

    pub(crate) fn source(&self, index: usize) -> &CodedSource<'a> {
        &self.sources[index]
    }

    #[cfg(any(test, feature = "internals"))]
    pub(crate) const fn sigma_is_sign(&self) -> bool {
        self.sigma_is_sign
    }
}

fn check_alphabet(minimum: i8, maximum: i8) -> Result<(), SourceTableError> {
    let width = i16::from(maximum) - i16::from(minimum) + 1;
    if !(1..=16).contains(&width) || minimum < -8 || maximum > 7 {
        return Err(SourceTableError::Alphabet { minimum, maximum });
    }
    Ok(())
}

/// Pair code over the eight images of `d(d + 1)`, for digits in `[-8, 7]`.
fn image_pair_code(left: i8, right: i8) -> u8 {
    let image_index = |digit: i8| {
        if digit >= 0 {
            digit as u8
        } else {
            (-i16::from(digit) - 1) as u8
        }
    };
    image_index(left) * 8 + image_index(right)
}

/// Radix-16 pair code of two digits relative to `minimum`.
fn raw_pair_code(left: i8, right: i8, minimum: i8) -> u8 {
    let raw_index = |digit: i8| (i16::from(digit) - i16::from(minimum)) as u8;
    raw_index(left) * 16 + raw_index(right)
}

impl PartialEq for SourceTables {
    fn eq(&self, other: &Self) -> bool {
        self.degree == other.degree
            && self.records == other.records
            && (0..3).all(|source| {
                let left = self.coefficient_view(source);
                let right = other.coefficient_view(source);
                left.len() == right.len()
                    && (0..left.len()).all(|cell| left.get(cell) == right.get(cell))
            })
            && {
                let left = self.sigma_view();
                let right = other.sigma_view();
                left.len() == right.len()
                    && (0..left.len()).all(|cell| left.get(cell) == right.get(cell))
            }
            && self.u == other.u
    }
}

impl Eq for SourceTables {}

impl SourceTables {
    /// Encode every record's `s2` polynomial into `a0`, `a1`, `a2`, and
    /// `sigma`, with three balanced budget cells over the padded domain.
    ///
    /// Padded rows are the all-zero signature, so they carry the canonical
    /// encoding of zero coefficients and the full-slack budget
    /// digits required by Section 3.4.
    pub fn build(batch: &FalconWitness) -> Result<Self, SourceTableError> {
        super::validate_witness_shape(batch)?;
        Self::build_from_s2(batch.degree(), batch.padded_rows, &batch.s2)
    }

    /// [`Self::build`] from the padded `s2` table alone (`padded_rows *
    /// degree` record-major coefficients, zero on padded rows). The tables
    /// depend on nothing else in the batch, so a prover can build and commit
    /// them before native verification produces `s1`.
    pub fn build_from_s2(
        degree: usize,
        padded_rows: usize,
        s2: &[i16],
    ) -> Result<Self, SourceTableError> {
        if padded_rows == 0 || !padded_rows.is_power_of_two() || s2.len() != padded_rows * degree {
            return Err(SourceTableError::Shape);
        }
        match degree {
            Falcon512::DEGREE => Self::build_of::<Falcon512>(s2, padded_rows),
            Falcon1024::DEGREE => Self::build_of::<Falcon1024>(s2, padded_rows),
            other => Err(SourceTableError::Unsupported(
                crate::UnsupportedBatch::Degree { degree: other },
            )),
        }
    }

    fn build_of<P: FalconParams>(s2: &[i16], records: usize) -> Result<Self, SourceTableError> {
        let degree = P::DEGREE;
        let cells = records * degree;
        let mut tables = Self {
            degree: FalconDegree::from_degree(degree)?,
            records,
            coefficients: CoefficientCells::Compact {
                raw16: std::array::from_fn(|_| zeroed_codes(cells / 2)),
                image8: std::array::from_fn(|_| zeroed_codes(cells / 2)),
            },
            signs: SignCells::Bits(vec![0; cells / 64]),
            u: std::array::from_fn(|_| vec![0; records]),
        };
        #[cfg(feature = "parallel")]
        {
            let [u0, u1, u2] = &mut tables.u;
            let (raw16, image8) = match &mut tables.coefficients {
                CoefficientCells::Compact { raw16, image8 } => (raw16, image8),
                #[cfg(any(test, feature = "internals"))]
                CoefficientCells::Wide(_) => {
                    unreachable!("new source tables use compact coefficient storage")
                }
            };
            #[cfg(not(any(test, feature = "internals")))]
            let SignCells::Bits(sign_bits) = &mut tables.signs;
            #[cfg(any(test, feature = "internals"))]
            let SignCells::Bits(sign_bits) = &mut tables.signs
            else {
                unreachable!("new source tables use bit-packed sign storage")
            };
            let [image0, image1, image2] = image8.each_mut().map(unique_codes);
            let [raw0, raw1, raw2] = raw16.each_mut().map(unique_codes);
            let image_chunks = image0
                .par_chunks_mut(degree / 2)
                .zip(image1.par_chunks_mut(degree / 2))
                .zip(image2.par_chunks_mut(degree / 2))
                .map(|((image0, image1), image2)| [image0, image1, image2]);
            let raw_chunks = raw0
                .par_chunks_mut(degree / 2)
                .zip(raw1.par_chunks_mut(degree / 2))
                .zip(raw2.par_chunks_mut(degree / 2))
                .map(|((raw0, raw1), raw2)| [raw0, raw1, raw2]);
            let records = image_chunks
                .zip(raw_chunks)
                .zip(sign_bits.par_chunks_mut(degree / 64))
                .zip(u0.par_iter_mut())
                .zip(u1.par_iter_mut())
                .zip(u2.par_iter_mut())
                .map(|(((((images, raws), signs), u0), u1), u2)| (images, raws, signs, u0, u1, u2));
            records
                .enumerate()
                .try_for_each(|(record, (images, raws, signs, u0, u1, u2))| {
                    let witness =
                        encodability_row::<P>(&s2[record * degree..(record + 1) * degree], record)?;
                    let digits = witness.budget_digits();
                    *u0 = digits[0];
                    *u1 = digits[1];
                    *u2 = digits[2];
                    for (((source, minimum), image), raw) in
                        [witness.a0(), witness.a1(), witness.a2()]
                            .into_iter()
                            .map(AsRef::as_ref)
                            .zip(COEFFICIENT_MINIMA)
                            .zip(images)
                            .zip(raws)
                    {
                        encode_coefficient_pair_codes(source, minimum, image, raw);
                    }
                    encode_sign_bits(witness.sigma().as_ref(), signs);
                    Ok::<_, SourceTableError>(())
                })?;
        }
        #[cfg(not(feature = "parallel"))]
        for record in 0..records {
            let start = record * degree;
            let witness = encodability_row::<P>(&s2[start..start + degree], record)?;
            for (table, &digit) in tables.u.iter_mut().zip(witness.budget_digits()) {
                table[record] = digit;
            }
            let pair_start = start / 2;
            let pair_end = pair_start + degree / 2;
            let (raw16, image8) = match &mut tables.coefficients {
                CoefficientCells::Compact { raw16, image8 } => (raw16, image8),
                #[cfg(any(test, feature = "internals"))]
                CoefficientCells::Wide(_) => {
                    unreachable!("new source tables use compact coefficient storage")
                }
            };
            for (((source, minimum), image), raw) in [witness.a0(), witness.a1(), witness.a2()]
                .into_iter()
                .map(AsRef::as_ref)
                .zip(COEFFICIENT_MINIMA)
                .zip(
                    image8
                        .each_mut()
                        .map(|codes| &mut unique_codes(codes)[pair_start..pair_end]),
                )
                .zip(
                    raw16
                        .each_mut()
                        .map(|codes| &mut unique_codes(codes)[pair_start..pair_end]),
                )
            {
                encode_coefficient_pair_codes(source, minimum, image, raw);
            }
            #[cfg(not(any(test, feature = "internals")))]
            let SignCells::Bits(sign_bits) = &mut tables.signs;
            #[cfg(any(test, feature = "internals"))]
            let SignCells::Bits(sign_bits) = &mut tables.signs
            else {
                unreachable!("new source tables use bit-packed sign storage")
            };
            encode_sign_bits(
                witness.sigma().as_ref(),
                &mut sign_bits[start / 64..(start + degree) / 64],
            );
        }
        Ok(tables)
    }

    /// Check source alphabets, decoded coefficients, and the exact budget
    /// identity against a compact witness. This is an encoder self-check for
    /// tests and the adversarial harness, not part of the production prove
    /// path: soundness is the verifier's job, and the honest encoder in
    /// [`Self::build`] cannot fail it. Equality is semantic: either sign
    /// of a zero magnitude is valid, including on padded rows.
    #[cfg(any(test, feature = "internals"))]
    pub fn validate_semantics(&self, batch: &FalconWitness) -> Result<(), SourceTableError> {
        super::validate_witness_shape(batch)?;
        if self.records != batch.padded_rows || self.degree() != batch.degree() {
            return Err(SourceTableError::Shape);
        }
        match self.degree {
            FalconDegree::Falcon512 => self.validate_semantics_of::<Falcon512>(batch),
            FalconDegree::Falcon1024 => self.validate_semantics_of::<Falcon1024>(batch),
        }
    }

    #[cfg(any(test, feature = "internals"))]
    fn validate_semantics_of<P: FalconParams>(
        &self,
        batch: &FalconWitness,
    ) -> Result<(), SourceTableError> {
        let degree = P::DEGREE;
        let a0 = self.a0_view();
        let a1 = self.a1_view();
        let a2 = self.a2_view();
        let sigma = self.sigma_view();
        for cell in 0..a0.len() {
            if !(-4..=3).contains(&a0.get(cell))
                || !(-8..=7).contains(&a1.get(cell))
                || !(-8..=7).contains(&a2.get(cell))
                || !matches!(sigma.get(cell), -1 | 1)
            {
                return Err(SourceTableError::SourceAlphabet { cell });
            }
            if self.decode_coefficient(cell) != i64::from(batch.s2[cell]) {
                return Err(SourceTableError::SourceDecode { cell });
            }
        }
        let radices = P::BUDGET_DIGIT_RADICES;
        let budget = i64::try_from(P::COMPRESSED_HIGH_BITS_BUDGET).expect("budget fits i64");
        for row in 0..self.records {
            let digits = self.u.each_ref().map(|table| table[row]);
            for (digit, radix) in digits.iter().zip(radices) {
                let half = i8::try_from(radix / 2).expect("half radix fits i8");
                if !(-half..half).contains(digit) {
                    return Err(SourceTableError::BudgetAlphabet { row });
                }
            }
            let unary_sum: i64 = (row * degree..(row + 1) * degree)
                .map(|cell| i64::from(a2.get(cell)) + i64::from(A2_OFFSET))
                .sum();
            if unary_sum + self.budget_slack(row) != budget {
                return Err(SourceTableError::BudgetEquation { row });
            }
        }
        Ok(())
    }

    /// Ring degree of the encoded batch.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree.degree()
    }

    /// Radices of the three budget digits, least significant first.
    #[must_use]
    #[cfg(any(test, feature = "internals"))]
    pub fn budget_radices(&self) -> [u8; 3] {
        match self.degree {
            FalconDegree::Falcon512 => Falcon512::BUDGET_DIGIT_RADICES,
            FalconDegree::Falcon1024 => Falcon1024::BUDGET_DIGIT_RADICES,
        }
    }

    /// Additive constant of the budget-slack recomposition.
    #[must_use]
    #[cfg(any(test, feature = "internals"))]
    pub fn budget_constant(&self) -> i64 {
        match self.degree {
            FalconDegree::Falcon512 => i64::from(Falcon512::BUDGET_CONSTANT),
            FalconDegree::Falcon1024 => i64::from(Falcon1024::BUDGET_CONSTANT),
        }
    }

    /// The unsigned budget slack of one row, recomposed from its digits.
    #[must_use]
    #[cfg(any(test, feature = "internals"))]
    pub fn budget_slack(&self, row: usize) -> i64 {
        let radices = self.budget_radices();
        let weight1 = i64::from(radices[0]);
        let weight2 = weight1 * i64::from(radices[1]);
        i64::from(self.u[0][row])
            + weight1 * i64::from(self.u[1][row])
            + weight2 * i64::from(self.u[2][row])
            + self.budget_constant()
    }

    #[must_use]
    pub const fn records(&self) -> usize {
        self.records
    }

    /// Decode one coefficient digit source for internal regression harnesses.
    #[cfg(feature = "internals")]
    #[doc(hidden)]
    pub fn coefficient_digits(&self, source: usize) -> impl Iterator<Item = i8> + '_ {
        let cells = self.coefficient_view(source);
        (0..cells.len()).map(move |index| cells.get(index))
    }

    /// Decode the sign source for internal regression harnesses.
    #[cfg(feature = "internals")]
    #[doc(hidden)]
    pub fn sign_digits(&self) -> impl Iterator<Item = i8> + '_ {
        let cells = self.sigma_view();
        (0..cells.len()).map(move |index| cells.get(index))
    }

    #[must_use]
    pub(crate) fn a0_view(&self) -> DigitSource<'_> {
        self.coefficient_view(0)
    }

    #[must_use]
    pub(crate) fn a1_view(&self) -> DigitSource<'_> {
        self.coefficient_view(1)
    }

    #[must_use]
    pub(crate) fn a2_view(&self) -> DigitSource<'_> {
        self.coefficient_view(2)
    }

    fn coefficient_view(&self, source: usize) -> DigitSource<'_> {
        match &self.coefficients {
            CoefficientCells::Compact { raw16, .. } => DigitSource::Packed {
                codes: &raw16[source],
                minimum: COEFFICIENT_MINIMA[source],
            },
            #[cfg(any(test, feature = "internals"))]
            CoefficientCells::Wide(cells) => DigitSource::Direct(&cells[source]),
        }
    }

    /// Stored image pair codes for coefficient source `0 = a0`, `1 = a1`, or
    /// `2 = a2`; `None` once an adversarial hook widened the storage.
    #[cfg(test)]
    fn coefficient_image8_pair_codes(&self, source: usize) -> Option<&[u8]> {
        match &self.coefficients {
            CoefficientCells::Compact { image8, .. } => image8.get(source).map(AsRef::as_ref),
            #[cfg(any(test, feature = "internals"))]
            CoefficientCells::Wide(_) => None,
        }
    }

    /// Stored raw radix-16 pair codes, relative to the source minimum.
    #[cfg(test)]
    fn coefficient_raw16_pair_codes(&self, source: usize) -> Option<&[u8]> {
        match &self.coefficients {
            CoefficientCells::Compact { raw16, .. } => raw16.get(source).map(AsRef::as_ref),
            #[cfg(any(test, feature = "internals"))]
            CoefficientCells::Wide(_) => None,
        }
    }

    #[must_use]
    pub(crate) fn sigma_view(&self) -> SignSource<'_> {
        match &self.signs {
            SignCells::Bits(words) => SignSource::Bits(words),
            #[cfg(any(test, feature = "internals"))]
            SignCells::Wide(cells) => SignSource::Direct(cells),
        }
    }

    /// Budget digit `u0` on the row domain.
    #[must_use]
    pub fn u0(&self) -> &[i8] {
        &self.u[0]
    }

    /// Budget digit `u1` on the row domain.
    #[must_use]
    pub fn u1(&self) -> &[i8] {
        &self.u[1]
    }

    /// Budget digit `u2` on the row domain.
    #[must_use]
    pub fn u2(&self) -> &[i8] {
        &self.u[2]
    }

    /// Budget digit by its least-significant-first index.
    #[must_use]
    pub fn budget(&self, index: usize) -> &[i8] {
        &self.u[index]
    }

    /// Adversarial-test hook (Section 12): mutable access to `a_0`.
    #[doc(hidden)]
    #[cfg(any(test, feature = "internals"))]
    pub fn a0_mut(&mut self) -> &mut [i8] {
        self.make_coefficients_wide();
        let CoefficientCells::Wide(cells) = &mut self.coefficients else {
            unreachable!("coefficient storage was widened")
        };
        &mut cells[0]
    }

    /// Adversarial-test hook (Section 12): mutable access to `a_1`.
    #[doc(hidden)]
    #[cfg(feature = "internals")]
    pub fn a1_mut(&mut self) -> &mut [i8] {
        self.make_coefficients_wide();
        let CoefficientCells::Wide(cells) = &mut self.coefficients else {
            unreachable!("coefficient storage was widened")
        };
        &mut cells[1]
    }

    /// Adversarial-test hook (Section 12): mutable access to `a2`.
    #[doc(hidden)]
    #[cfg(feature = "internals")]
    pub fn a2_mut(&mut self) -> &mut [i8] {
        self.make_coefficients_wide();
        let CoefficientCells::Wide(cells) = &mut self.coefficients else {
            unreachable!("coefficient storage was widened")
        };
        &mut cells[2]
    }

    /// Adversarial-test hook (Section 12): mutable access to `sigma`.
    #[doc(hidden)]
    #[cfg(any(test, feature = "internals"))]
    pub fn sigma_mut(&mut self) -> &mut [i8] {
        if let SignCells::Bits(words) = &self.signs {
            let source = SignSource::Bits(words);
            let mut cells = vec![0; source.len()];
            source.decode_into(0, &mut cells);
            self.signs = SignCells::Wide(cells);
        }
        let SignCells::Wide(cells) = &mut self.signs else {
            unreachable!("sign storage was widened")
        };
        cells
    }

    /// Adversarial-test hook (Section 12): mutable access to `u0`.
    #[doc(hidden)]
    #[cfg(any(test, feature = "internals"))]
    pub fn u0_mut(&mut self) -> &mut [i8] {
        &mut self.u[0]
    }

    /// Adversarial-test hook (Section 12): mutable access to `u1`.
    #[doc(hidden)]
    #[cfg(feature = "internals")]
    pub fn u1_mut(&mut self) -> &mut [i8] {
        &mut self.u[1]
    }

    /// Adversarial-test hook (Section 12): mutable access to `u2`.
    #[doc(hidden)]
    #[cfg(any(test, feature = "internals"))]
    pub fn u2_mut(&mut self) -> &mut [i8] {
        &mut self.u[2]
    }

    /// Adversarial-test hook (Section 12): mutable budget digit by index.
    #[doc(hidden)]
    #[cfg(feature = "internals")]
    pub fn budget_mut(&mut self, index: usize) -> &mut [i8] {
        &mut self.u[index]
    }

    /// Decode one virtual `s2` coefficient from the stored tables.
    #[must_use]
    #[cfg(any(test, feature = "internals"))]
    pub fn decode_coefficient(&self, cell: usize) -> i64 {
        let magnitude = i64::from(self.a0_view().get(cell))
            + 8 * i64::from(self.a1_view().get(cell))
            + 128 * i64::from(self.a2_view().get(cell))
            + 1_092;
        i64::from(self.sigma_view().get(cell)) * magnitude
    }

    #[cfg(any(test, feature = "internals"))]
    fn make_coefficients_wide(&mut self) {
        if let CoefficientCells::Compact { raw16, .. } = &self.coefficients {
            let cells = std::array::from_fn(|source| {
                let digits = DigitSource::Packed {
                    codes: &raw16[source],
                    minimum: COEFFICIENT_MINIMA[source],
                };
                let mut cells = vec![0; digits.len()];
                digits.decode_into(0, &mut cells);
                cells
            });
            self.coefficients = CoefficientCells::Wide(cells);
        }
    }
}

fn encode_coefficient_pair_codes(source: &[i8], minimum: i8, image8: &mut [u8], raw16: &mut [u8]) {
    debug_assert_eq!(source.len(), 2 * image8.len());
    debug_assert_eq!(image8.len(), raw16.len());
    for ((digits, image), raw) in source
        .chunks_exact(2)
        .zip(image8.iter_mut())
        .zip(raw16.iter_mut())
    {
        *image = image_pair_code(digits[0], digits[1]);
        *raw = raw_pair_code(digits[0], digits[1], minimum);
    }
}

/// Zero-initialized pair-code storage, allocated directly as a shared slice.
fn zeroed_codes(len: usize) -> Arc<[u8]> {
    (0..len).map(|_| 0).collect()
}

/// Mutable access to pair codes that `build_of` has not yet shared.
fn unique_codes(codes: &mut Arc<[u8]>) -> &mut [u8] {
    Arc::get_mut(codes).expect("freshly allocated pair codes are unshared")
}

fn encode_sign_bits(source: &[i8], output: &mut [u64]) {
    debug_assert_eq!(source.len(), 64 * output.len());
    for (signs, word) in source.chunks_exact(64).zip(output) {
        *word = signs.iter().enumerate().fold(0, |bits, (index, &sign)| {
            bits | u64::from(sign == -1) << index
        });
    }
}

fn encodability_row<P: FalconParams>(
    source: &[i16],
    record: usize,
) -> Result<FourCellEncodabilityWitness<P>, SourceTableError> {
    let s2 = <P::Array<i16> as FixedArray<i16>>::from_slice(source)
        .ok_or(SourceTableError::RecordLength { record })?;
    FourCellEncodabilityWitness::<P>::build(&s2).map_err(|source| SourceTableError::Encodability {
        record,
        source: crate::StatementError::new(aerie_falcon_statement::ProductionError::Aerie(source)),
    })
}

/// The four-component slack vector for each live record, in row order.
#[must_use]
pub fn slack_group(batch: &FalconWitness) -> &[i16] {
    &batch.slack
}

#[cfg(test)]
mod tests {
    use super::*;
    use aerie_falcon_witness::DirectRow;

    fn decode_digits(source: DigitSource<'_>) -> Vec<i8> {
        let mut cells = vec![0; source.len()];
        source.decode_into(0, &mut cells);
        cells
    }

    fn decode_signs(source: SignSource<'_>) -> Vec<i8> {
        let mut cells = vec![0; source.len()];
        source.decode_into(0, &mut cells);
        cells
    }

    fn synthetic_batch(rows: usize) -> FalconWitness {
        let rows = (0..rows)
            .map(|seed| {
                DirectRow::<aerie_falcon_native::Falcon512>::synthetic(seed as u64 + 11)
                    .expect("synthetic row")
            })
            .collect::<Vec<_>>();
        FalconWitness::from_rows(&rows)
    }

    fn wide_synthetic_batch(rows: usize) -> FalconWitness {
        let rows = (0..rows)
            .map(|seed| DirectRow::<Falcon1024>::synthetic(seed as u64 + 11).expect("row"))
            .collect::<Vec<_>>();
        FalconWitness::from_rows(&rows)
    }

    fn assert_native_cells<P: FalconParams>(batch: &FalconWitness) {
        let degree = P::DEGREE;
        let tables = SourceTables::build(batch).expect("source tables");
        for record in 0..batch.padded_rows {
            let start = record * degree;
            let s2 =
                <P::Array<i16> as FixedArray<i16>>::from_slice(&batch.s2[start..start + degree])
                    .expect("row has the parameter-set degree");
            let expected = FourCellEncodabilityWitness::<P>::build(&s2).expect("encodable row");
            for offset in 0..degree {
                let cell = start + offset;
                assert_eq!(tables.a0_view().get(cell), expected.a0().as_ref()[offset]);
                assert_eq!(tables.a1_view().get(cell), expected.a1().as_ref()[offset]);
                assert_eq!(tables.a2_view().get(cell), expected.a2().as_ref()[offset]);
                assert_eq!(
                    tables.sigma_view().get(cell),
                    expected.sigma().as_ref()[offset]
                );
            }
        }
    }

    #[test]
    fn built_tables_match_each_native_encodability_cell() {
        assert_native_cells::<Falcon512>(&synthetic_batch(3));
        assert_native_cells::<Falcon1024>(&wide_synthetic_batch(3));
    }

    #[test]
    fn packed_digit_views_round_trip_every_alphabet_pair() {
        for (minimum, width) in [(-4_i8, 8_u8), (-8, 16)] {
            let expected: Vec<i8> = (0..width)
                .flat_map(|left| {
                    (0..width).flat_map(move |right| {
                        [
                            minimum + i8::try_from(left).unwrap(),
                            minimum + i8::try_from(right).unwrap(),
                        ]
                    })
                })
                .collect();
            let mut codes = vec![0; expected.len() / 2];
            let mut images = vec![0; expected.len() / 2];
            encode_coefficient_pair_codes(&expected, minimum, &mut images, &mut codes);
            let source = DigitSource::Packed {
                codes: &codes,
                minimum,
            };
            assert_eq!(source.len(), expected.len());
            for (index, &digit) in expected.iter().enumerate() {
                assert_eq!(source.get(index), digit);
            }
            for start in 0..expected.len() {
                for len in 0..=(expected.len() - start).min(17) {
                    let mut decoded = vec![0; len];
                    source.decode_into(start, &mut decoded);
                    assert_eq!(decoded, expected[start..start + len]);
                }
            }
        }
    }

    #[test]
    fn packed_sign_view_round_trips_across_word_boundaries() {
        let expected: Vec<i8> = (0..192)
            .map(|cell| {
                if cell % 3 == 0 || cell % 11 == 5 {
                    -1
                } else {
                    1
                }
            })
            .collect();
        let mut words = vec![0_u64; expected.len() / 64];
        encode_sign_bits(&expected, &mut words);
        let source = SignSource::Bits(&words);
        for (index, &sign) in expected.iter().enumerate() {
            assert_eq!(source.get(index), sign);
        }
        for start in 0..expected.len() {
            for len in 0..=(expected.len() - start).min(150) {
                let mut decoded = vec![0; len];
                source.decode_into(start, &mut decoded);
                assert_eq!(decoded, expected[start..start + len]);
            }
        }
    }

    #[test]
    fn source_tables_decode_back_to_the_committed_s2_table() {
        for batch in [synthetic_batch(4), wide_synthetic_batch(4)] {
            let tables = SourceTables::build(&batch).unwrap();
            assert_eq!(tables.degree(), batch.degree());
            for cell in 0..batch.s2.len() {
                assert_eq!(tables.decode_coefficient(cell), i64::from(batch.s2[cell]));
            }
            for a2 in decode_digits(tables.a2_view()) {
                assert!(
                    (-8..=7).contains(&a2),
                    "a2 must be a balanced radix-16 digit"
                );
            }
            tables.validate_semantics(&batch).unwrap();
        }
    }

    #[test]
    fn source_tables_keep_each_native_group_polynomial_separate() {
        let batch = synthetic_batch(8);
        let tables = SourceTables::build(&batch).unwrap();
        assert_eq!(tables.a0_view().len(), 512 * 8);
        assert_eq!(tables.a1_view().len(), 512 * 8);
        assert_eq!(tables.a2_view().len(), 512 * 8);
        assert_eq!(tables.sigma_view().len(), 512 * 8);
        assert_eq!(tables.u0().len(), 8);
        assert_eq!(tables.u1().len(), 8);
        assert_eq!(tables.u2().len(), 8);
        assert_eq!(slack_group(&batch).len(), 32);

        let wide = wide_synthetic_batch(8);
        let tables = SourceTables::build(&wide).unwrap();
        assert_eq!(tables.a0_view().len(), 1_024 * 8);
        assert_eq!(tables.a1_view().len(), 1_024 * 8);
        assert_eq!(tables.a2_view().len(), 1_024 * 8);
        assert_eq!(tables.sigma_view().len(), 1_024 * 8);
        assert_eq!(tables.budget_radices(), [8, 8, 16]);
        assert_eq!(tables.budget_constant(), 548);
    }

    #[test]
    fn coefficient_pair_code_cache_matches_independent_encodings() {
        assert_pair_codes::<Falcon512>(&synthetic_batch(4));
        assert_pair_codes::<Falcon1024>(&wide_synthetic_batch(4));
    }

    /// Both pair-code tables against encodings of the native witness digits,
    /// not of the tables' own decoded cells.
    fn assert_pair_codes<P: FalconParams>(batch: &FalconWitness) {
        let degree = P::DEGREE;
        let tables = SourceTables::build(batch).unwrap();
        let mut native: [Vec<i8>; 3] = Default::default();
        for record in 0..batch.padded_rows {
            let start = record * degree;
            let s2 =
                <P::Array<i16> as FixedArray<i16>>::from_slice(&batch.s2[start..start + degree])
                    .expect("row has the parameter-set degree");
            let witness = FourCellEncodabilityWitness::<P>::build(&s2).expect("encodable row");
            for (digits, row) in native
                .iter_mut()
                .zip([witness.a0(), witness.a1(), witness.a2()])
            {
                digits.extend_from_slice(row.as_ref());
            }
        }
        for (source, (digits, minimum)) in native.into_iter().zip(COEFFICIENT_MINIMA).enumerate() {
            let expected_image: Vec<u8> = digits
                .chunks_exact(2)
                .map(|pair| {
                    const IMAGES: [i16; 8] = [0, 2, 6, 12, 20, 30, 42, 56];
                    let image = |digit: i8| {
                        let digit = i16::from(digit);
                        let value = digit * (digit + 1);
                        IMAGES.iter().position(|&image| image == value).unwrap() as u8
                    };
                    image(pair[0]) * 8 + image(pair[1])
                })
                .collect();
            let expected_raw: Vec<u8> = digits
                .chunks_exact(2)
                .map(|pair| {
                    let index =
                        |digit: i8| u8::try_from(i16::from(digit) - i16::from(minimum)).unwrap();
                    index(pair[0]) * 16 + index(pair[1])
                })
                .collect();
            assert_eq!(
                tables.coefficient_image8_pair_codes(source),
                Some(expected_image.as_slice())
            );
            assert_eq!(
                tables.coefficient_raw16_pair_codes(source),
                Some(expected_raw.as_slice())
            );
        }
        assert_eq!(tables.coefficient_image8_pair_codes(3), None);
        assert_eq!(tables.coefficient_raw16_pair_codes(3), None);
    }

    #[test]
    fn mutable_coefficient_access_invalidates_all_pair_codes() {
        let mut tables = SourceTables::build(&synthetic_batch(4)).unwrap();
        let expected = tables.clone();
        assert!(tables.coefficient_image8_pair_codes(0).is_some());
        assert!(tables.coefficient_raw16_pair_codes(2).is_some());
        let value = tables.a0_view().get(0);
        tables.a0_mut()[0] = value;
        for source in 0..3 {
            assert_eq!(tables.coefficient_image8_pair_codes(source), None);
            assert_eq!(tables.coefficient_raw16_pair_codes(source), None);
        }
        assert_eq!(tables, expected);
    }

    #[test]
    fn prepared_coefficients_share_compact_codes_and_match_direct_encoding() {
        for batch in [synthetic_batch(4), wide_synthetic_batch(4)] {
            let tables = SourceTables::build(&batch).unwrap();
            let prepared = PreparedCoefficients::new(&tables).unwrap();
            assert!(prepared.sigma_is_sign());
            let CoefficientCells::Compact { raw16, image8 } = &tables.coefficients else {
                panic!("built tables are compact");
            };
            for source in 0..3 {
                let coded = prepared.source(source);
                assert!(Arc::ptr_eq(&coded.raw16_codes(), &raw16[source]));
                assert!(Arc::ptr_eq(&coded.image8_codes(), &image8[source]));
                assert_eq!(coded.minimum(), COEFFICIENT_MINIMA[source]);
                assert_eq!(coded.maximum(), COEFFICIENT_MAXIMA[source]);
                let digits = coded.digits();
                assert_eq!(digits, decode_digits(tables.coefficient_view(source)));
                let direct = CodedSource::new(
                    &digits,
                    COEFFICIENT_MINIMA[source],
                    COEFFICIENT_MAXIMA[source],
                )
                .unwrap();
                assert_eq!(direct.image8(), coded.image8());
                assert_eq!(direct.raw16(), coded.raw16());
                assert_eq!(direct.cell_count(), coded.cell_count());
                for index in [0, 1, digits.len() - 1, digits.len()] {
                    assert_eq!(direct.digit(index), coded.digit(index));
                }
            }
        }
    }

    #[test]
    fn prepared_coefficients_are_rebuilt_after_mutation() {
        let mut tables = SourceTables::build(&synthetic_batch(4)).unwrap();
        let before = PreparedCoefficients::new(&tables)
            .unwrap()
            .source(0)
            .raw16()
            .to_vec();
        let value = tables.a0_view().get(0);
        tables.a0_mut()[0] = i8::from(value == 0);
        let prepared = PreparedCoefficients::new(&tables).unwrap();
        assert_eq!(prepared.source(0).digits(), decode_digits(tables.a0_view()));
        assert_ne!(prepared.source(0).raw16(), before);
        drop(prepared);

        tables.a0_mut()[1] = 4;
        assert!(matches!(
            PreparedCoefficients::new(&tables),
            Err(SourceTableError::Digit {
                index: 1,
                digit: 4,
                minimum: -4,
                maximum: 3
            })
        ));

        let mut signs = SourceTables::build(&synthetic_batch(4)).unwrap();
        signs.sigma_mut()[0] = 0;
        assert!(!PreparedCoefficients::new(&signs).unwrap().sigma_is_sign());
    }

    #[test]
    fn coded_sources_validate_pairs_and_singletons() {
        assert!(matches!(
            CodedSource::new(&[0, 8], -4, 3),
            Err(SourceTableError::Digit {
                index: 1,
                digit: 8,
                minimum: -4,
                maximum: 3
            })
        ));
        assert!(CodedSource::new(&[3], -4, 3).is_ok());
        assert!(matches!(
            CodedSource::new(&[4], -4, 3),
            Err(SourceTableError::Digit {
                index: 0,
                digit: 4,
                minimum: -4,
                maximum: 3
            })
        ));
        assert!(matches!(
            CodedSource::new(&[0, 0], -9, 3),
            Err(SourceTableError::Alphabet {
                minimum: -9,
                maximum: 3
            })
        ));
    }

    fn assert_no_op_mutable_access(batch: &FalconWitness) {
        let expected = SourceTables::build(batch).unwrap();

        let mut signs_wide = expected.clone();
        let sign = signs_wide.sigma_view().get(65);
        signs_wide.sigma_mut()[65] = sign;
        assert_eq!(signs_wide, expected);
        signs_wide.validate_semantics(batch).unwrap();
        for source in 0..3 {
            assert!(signs_wide.coefficient_image8_pair_codes(source).is_some());
            assert!(signs_wide.coefficient_raw16_pair_codes(source).is_some());
        }
        assert!(matches!(
            &signs_wide.coefficients,
            CoefficientCells::Compact { .. }
        ));

        let mut a0_wide = expected.clone();
        let digit = a0_wide.a0_view().get(0);
        a0_wide.a0_mut()[0] = digit;
        assert_eq!(a0_wide, expected);
        a0_wide.validate_semantics(batch).unwrap();
        for source in 0..3 {
            assert_eq!(a0_wide.coefficient_image8_pair_codes(source), None);
            assert_eq!(a0_wide.coefficient_raw16_pair_codes(source), None);
        }
        assert!(matches!(&a0_wide.signs, SignCells::Bits(_)));

        #[cfg(feature = "internals")]
        {
            let mut a1_wide = expected.clone();
            let digit = a1_wide.a1_view().get(1);
            a1_wide.a1_mut()[1] = digit;
            assert_eq!(a1_wide, expected);
            a1_wide.validate_semantics(batch).unwrap();
            for source in 0..3 {
                assert_eq!(a1_wide.coefficient_image8_pair_codes(source), None);
                assert_eq!(a1_wide.coefficient_raw16_pair_codes(source), None);
            }
            assert!(matches!(&a1_wide.signs, SignCells::Bits(_)));

            let mut a2_wide = expected.clone();
            let digit = a2_wide.a2_view().get(2);
            a2_wide.a2_mut()[2] = digit;
            assert_eq!(a2_wide, expected);
            a2_wide.validate_semantics(batch).unwrap();
            for source in 0..3 {
                assert_eq!(a2_wide.coefficient_image8_pair_codes(source), None);
                assert_eq!(a2_wide.coefficient_raw16_pair_codes(source), None);
            }
            assert!(matches!(&a2_wide.signs, SignCells::Bits(_)));
        }
    }

    #[test]
    fn no_op_mutable_access_preserves_semantics_and_switches_only_its_storage() {
        assert_no_op_mutable_access(&synthetic_batch(4));
        assert_no_op_mutable_access(&wide_synthetic_batch(4));
    }

    #[test]
    fn every_emitted_source_and_budget_value_fits_its_specified_alphabet() {
        for batch in [synthetic_batch(3), wide_synthetic_batch(3)] {
            let tables = SourceTables::build(&batch).unwrap();
            for (name, cells) in [
                ("a0", decode_digits(tables.a0_view())),
                ("a1", decode_digits(tables.a1_view())),
                ("a2", decode_digits(tables.a2_view())),
                ("sigma", decode_signs(tables.sigma_view())),
            ] {
                for (index, cell) in cells.into_iter().enumerate() {
                    let valid = match name {
                        "a0" => (-4..=3).contains(&cell),
                        "sigma" => matches!(cell, -1 | 1),
                        _ => (-8..=7).contains(&cell),
                    };
                    assert!(valid, "{name}[{index}] = {cell} is outside its alphabet");
                }
            }
            let budget_tables = [tables.u0(), tables.u1(), tables.u2()];
            for (digit, table) in budget_tables.into_iter().enumerate() {
                let radix = tables.budget_radices()[digit];
                let half = i8::try_from(radix / 2).unwrap();
                for (row, &cell) in table.iter().enumerate() {
                    assert!(
                        (-half..half).contains(&cell),
                        "u{digit} at row {row} = {cell} is outside radix {radix}"
                    );
                }
            }

            // A padded all-zero signature contributes a zero source row and
            // the full-slack budget representation; verify that representation
            // also remains in the width-4 alphabet for both degrees.
            assert!(batch.live_rows < batch.padded_rows);
            let padded_start = batch.live_rows * batch.degree();
            for (name, cells) in [
                ("a0", decode_digits(tables.a0_view())),
                ("a1", decode_digits(tables.a1_view())),
                ("a2", decode_digits(tables.a2_view())),
                ("sigma", decode_signs(tables.sigma_view())),
            ] {
                let (lower, upper) = match name {
                    "a0" => (-4, 3),
                    "sigma" => (-1, 1),
                    _ => (-8, 7),
                };
                assert!(
                    cells[padded_start..]
                        .iter()
                        .all(|&cell| (lower..=upper).contains(&cell)),
                    "padded {name} values fit their alphabet"
                );
            }
            for (digit, table) in budget_tables.into_iter().enumerate() {
                let half = i8::try_from(tables.budget_radices()[digit] / 2).unwrap();
                assert!(
                    table[batch.live_rows..]
                        .iter()
                        .all(|&cell| (-half..half).contains(&cell)),
                    "padded u{digit} values fit their alphabet"
                );
            }
        }
    }

    #[test]
    fn budget_cells_recompose_each_record_slack() {
        for (batch, budget) in [(synthetic_batch(4), 392), (wide_synthetic_batch(4), 696)] {
            let degree = batch.degree();
            let tables = SourceTables::build(&batch).unwrap();
            for record in 0..4 {
                let start = record * degree;
                let unary_sum: i64 = batch.s2[start..start + degree]
                    .iter()
                    .map(|&coefficient| i64::from(coefficient).abs() >> 7)
                    .sum();
                assert_eq!(unary_sum + tables.budget_slack(record), budget);
            }
            if degree == 512 {
                let slack = i64::from(tables.u0()[0])
                    + 8 * i64::from(tables.u1()[0])
                    + 64 * i64::from(tables.u2()[0])
                    + 292;
                assert_eq!(slack, tables.budget_slack(0));
            }
        }
    }

    #[test]
    fn semantic_validation_checks_in_alphabet_edits_and_allows_zero_signs() {
        for batch in [synthetic_batch(3), wide_synthetic_batch(3)] {
            let degree = batch.degree();
            let tables = SourceTables::build(&batch).unwrap();
            let mut changed = tables.clone();
            let original = changed.a0_view().get(5);
            changed.a0_mut()[5] = if original == 3 { 2 } else { original + 1 };
            assert!(matches!(
                changed.validate_semantics(&batch),
                Err(SourceTableError::SourceDecode { cell: 5 })
            ));
            changed.a0_mut()[5] = original;
            changed.validate_semantics(&batch).unwrap();

            let mut changed = tables.clone();
            let digit = changed.u0()[0];
            changed.u0_mut()[0] = if digit == 3 { 2 } else { digit + 1 };
            assert!(matches!(
                changed.validate_semantics(&batch),
                Err(SourceTableError::BudgetEquation { row: 0 })
            ));

            let mut changed = tables.clone();
            changed.sigma_mut()[5] = 0;
            assert!(matches!(
                changed.validate_semantics(&batch),
                Err(SourceTableError::SourceAlphabet { cell: 5 })
            ));

            // No canonical-encoding equality check: a zero's sign is free.
            let mut changed = tables;
            let cell = 3 * degree;
            changed.sigma_mut()[cell] = -changed.sigma_view().get(cell);
            assert_eq!(changed.decode_coefficient(cell), 0);
            changed.validate_semantics(&batch).unwrap();
        }
    }

    #[test]
    fn falcon_1024_top_budget_digit_uses_radix_16() {
        let batch = wide_synthetic_batch(3);
        let tables = SourceTables::build(&batch).unwrap();
        // The padded fourth row uses the canonical budget digits
        // (u0, u1, u2) = (-4, 3, 2), giving 548 - 4 + 24 + 128 = 696.
        assert_eq!(tables.u0()[3], -4);
        assert_eq!(tables.u1()[3], 3);
        assert_eq!(tables.u2()[3], 2);
        assert_eq!(tables.budget_slack(3), 696);
        // A radix-16 digit that a Falcon-512 table would reject is valid here.
        let mut changed = tables.clone();
        changed.u2_mut()[3] = 7;
        assert!(matches!(
            changed.validate_semantics(&batch),
            Err(SourceTableError::BudgetEquation { row: 3 })
        ));
        changed.u2_mut()[3] = 8;
        assert!(matches!(
            changed.validate_semantics(&batch),
            Err(SourceTableError::BudgetAlphabet { row: 3 })
        ));
    }

    #[test]
    fn padded_rows_carry_the_canonical_zero_encoding() {
        let rows = (0..3)
            .map(|seed| {
                DirectRow::<aerie_falcon_native::Falcon512>::synthetic(seed + 21)
                    .expect("synthetic row")
            })
            .collect::<Vec<_>>();
        let batch = FalconWitness::from_rows(&rows);
        let tables = SourceTables::build(&batch).expect("arbitrary-batch source tables");
        assert_eq!(tables.records(), 4);
        // The padded fourth row decodes to zero coefficients and full slack.
        const DEGREE: usize = 512;
        for cell in 3 * DEGREE..4 * DEGREE {
            assert_eq!(tables.decode_coefficient(cell), 0);
            assert_eq!(tables.a0_view().get(cell), -4);
            assert_eq!(tables.a1_view().get(cell), -8);
            assert_eq!(tables.a2_view().get(cell), -8);
            assert_eq!(tables.sigma_view().get(cell), 1);
        }
        let slack = i64::from(tables.u0()[3])
            + 8 * i64::from(tables.u1()[3])
            + 64 * i64::from(tables.u2()[3])
            + 292;
        assert_eq!(slack, 392, "padded rows carry the full unary budget");
    }

    #[test]
    fn padded_budget_digits_match_both_parameter_set_canonical_values() {
        let small = SourceTables::build(&synthetic_batch(3)).unwrap();
        assert_eq!((small.u0()[3], small.u1()[3], small.u2()[3]), (-4, -3, 2));
        assert_eq!(small.budget_slack(3), 392);

        let wide = SourceTables::build(&wide_synthetic_batch(3)).unwrap();
        assert_eq!((wide.u0()[3], wide.u1()[3], wide.u2()[3]), (-4, 3, 2));
        assert_eq!(wide.budget_slack(3), 696);
        let padded_start = 3 * Falcon1024::DEGREE;
        for cell in padded_start..4 * Falcon1024::DEGREE {
            assert_eq!(wide.a0_view().get(cell), -4);
            assert_eq!(wide.a1_view().get(cell), -8);
            assert_eq!(wide.a2_view().get(cell), -8);
            assert_eq!(wide.sigma_view().get(cell), 1);
            assert_eq!(wide.decode_coefficient(cell), 0);
        }
    }
}
