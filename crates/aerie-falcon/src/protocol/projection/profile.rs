//! Public projection profile and the tensor geometry it induces.
//!
//! The profile is a function of the parameter set and the padded row count
//! `N_sig*` alone. The tensor geometry follows `specs/falcon.md` Section
//! 6.2: with `nu = log2 n`, `ell = log2 N_sig*` and `k = 12 - (nu + 1)`, the
//! norm first block is the low 12 bits `(j, a, i_{<k})` of the norm source
//! index and the residual first block the low 11 bits `(j, i_{<k})` of the
//! residual index. The remaining `R = ell - k` record bits are the "rest"
//! bits shared by both legs.

use super::super::{FalconDegree, MAX_RECORDS};

/// Independent norm-leg groups.
pub const NORM_GROUPS: usize = 4;
/// Prefix cores per norm group.
pub const NORM_PREFIX_LAYERS: usize = 4;
/// Output channels of every norm prefix core.
pub const NORM_PREFIX_WIDTH: usize = 43;
/// Rows of the norm terminal core, and so the length of `y`.
pub const NORM_TERMINAL_ROWS: usize = 156;
/// Concatenated channels entering the norm terminal core.
pub const NORM_CHANNELS: usize = NORM_GROUPS * NORM_PREFIX_WIDTH;
/// Residual cores.
pub const RESIDUAL_LAYERS: usize = 3;
/// Output channels of every residual core, and so the length of `z`.
pub const RESIDUAL_WIDTH: usize = 31;
/// Residual core entries are uniform on `{-bound, ..., bound}`.
pub const RESIDUAL_BOUND: u8 = 12;
/// Falcon's modulus `q`, the residual predicate's divisor.
pub const FALCON_Q: u64 = aerie_falcon_native::params::FALCON_Q as u64;
const _: () = assert!(aerie_falcon_native::params::FALCON_Q > 0);
/// Projection outputs `(y, z)`.
pub const OUTPUT_COUNT: usize = NORM_TERMINAL_ROWS + RESIDUAL_WIDTH;
/// First-core output channels of both legs: `4 * 43` norm then 31 residual.
pub const FIRST_CHANNELS: usize = NORM_CHANNELS + RESIDUAL_WIDTH;
/// Bits of the norm first block `(j, a, i_{<k})`.
#[doc(hidden)]
pub const NORM_FIRST_BLOCK_BITS: usize = 12;
/// Bits of the residual first block `(j, i_{<k})`.
#[doc(hidden)]
pub const RESIDUAL_FIRST_BLOCK_BITS: usize = 11;
/// Maximum bits taken by norm cores 2 through 4.
#[doc(hidden)]
pub const NORM_LATER_BLOCK_BITS: usize = 4;
/// Norm cap `C_S` for Falcon-512 (`specs/falcon.md` Section 6.1).
pub const NORM_CAP_FALCON_512: u128 = 5_092_300_183_991_442_303_232;
/// Norm cap `C_S` for Falcon-1024 (`specs/falcon.md` Section 6.1).
pub const NORM_CAP_FALCON_1024: u128 = 10_513_136_047_130_310_946_696;

/// Label opening [`ProjectionProfile::binding_bytes`].
pub const PROFILE_BINDING_LABEL: &[u8] = b"aerie/falcon/v1/projection/profile";

/// Entry distribution of one leg's cores.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoreDistribution {
    /// `-1, 0, 1` with probabilities `1/4, 1/2, 1/4` through the Section 6.3
    /// bit-pair map.
    TernaryQuarterHalfQuarter,
    /// Uniform on `{-bound, ..., bound}` by byte rejection sampling.
    UniformSymmetric { bound: u8 },
}

impl CoreDistribution {
    /// Binding tag followed by the distribution parameters.
    #[must_use]
    pub fn binding_bytes(self) -> Vec<u8> {
        match self {
            Self::TernaryQuarterHalfQuarter => vec![0],
            Self::UniformSymmetric { bound } => vec![1, bound],
        }
    }

    /// Largest absolute entry.
    #[must_use]
    pub const fn bound(self) -> u8 {
        match self {
            Self::TernaryQuarterHalfQuarter => 1,
            Self::UniformSymmetric { bound } => bound,
        }
    }
}

/// Slot layout of one norm-source record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NormSourceOrder {
    /// `2n` slots: `s_1` in lane `a = 0`, slack in the first four slots
    /// of lane `a = 1`, zeros elsewhere.
    S1ThenSlack,
}

impl NormSourceOrder {
    /// Binding tag.
    #[must_use]
    pub const fn transcript_id(self) -> u8 {
        match self {
            Self::S1ThenSlack => 0,
        }
    }
}

/// Public norm-leg parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NormLeg {
    pub groups: usize,
    pub prefix_widths: [usize; NORM_PREFIX_LAYERS],
    pub terminal_rows: usize,
    /// `[4096, r_2, r_3, r_4, r_5]`; radix-1 cores are kept.
    pub radices: [usize; NORM_PREFIX_LAYERS + 1],
    pub distribution: CoreDistribution,
    pub source_order: NormSourceOrder,
    /// `C_S`.
    pub cap: u128,
}

/// Public residual-leg parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidualLeg {
    pub widths: [usize; RESIDUAL_LAYERS],
    /// `[2048, r'_2, r'_3]`.
    pub radices: [usize; RESIDUAL_LAYERS],
    pub distribution: CoreDistribution,
    pub modulus: u64,
}

/// The public projection profile (`specs/falcon.md` Section 6.4).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionProfile {
    pub norm: NormLeg,
    pub residual: ResidualLeg,
}

impl ProjectionProfile {
    /// The profile of a padded batch of `padded_records = N_sig*` rows.
    ///
    /// `N_sig*` must be a power of two in `[2^k, 2^16]`, where `k = 2` for
    /// Falcon-512 and `k = 1` for Falcon-1024.
    pub fn for_batch(
        degree: FalconDegree,
        padded_records: usize,
    ) -> Result<Self, crate::UnsupportedBatch> {
        Ok(Geometry::new(degree, padded_records)?.profile())
    }

    /// Deterministic encoding of every profile field.
    ///
    /// Layout: [`PROFILE_BINDING_LABEL`], then the norm leg as
    /// `LE64(groups) || LE64(w_i) x4 || LE64(terminal_rows) || LE64(r_i) x5
    /// || distribution || U8(source_order) || LE128(cap)`, then the residual
    /// leg as `LE64(w_i) x3 || LE64(r_i) x3 || distribution || LE64(modulus)`.
    /// A distribution is `U8(0)` for the ternary map and `U8(1) || U8(bound)`
    /// for the uniform one.
    #[must_use]
    pub fn binding_bytes(&self) -> Vec<u8> {
        let le64 =
            |out: &mut Vec<u8>, value: usize| out.extend_from_slice(&(value as u64).to_le_bytes());
        let mut out = Vec::with_capacity(PROFILE_BINDING_LABEL.len() + 160);
        out.extend_from_slice(PROFILE_BINDING_LABEL);
        le64(&mut out, self.norm.groups);
        for &width in &self.norm.prefix_widths {
            le64(&mut out, width);
        }
        le64(&mut out, self.norm.terminal_rows);
        for &radix in &self.norm.radices {
            le64(&mut out, radix);
        }
        out.extend_from_slice(&self.norm.distribution.binding_bytes());
        out.push(self.norm.source_order.transcript_id());
        out.extend_from_slice(&self.norm.cap.to_le_bytes());
        for &width in &self.residual.widths {
            le64(&mut out, width);
        }
        for &radix in &self.residual.radices {
            le64(&mut out, radix);
        }
        out.extend_from_slice(&self.residual.distribution.binding_bytes());
        out.extend_from_slice(&self.residual.modulus.to_le_bytes());
        out
    }
}

/// The norm cap `C_S` of a parameter set.
#[must_use]
pub const fn norm_cap(degree: FalconDegree) -> u128 {
    match degree {
        FalconDegree::Falcon512 => NORM_CAP_FALCON_512,
        FalconDegree::Falcon1024 => NORM_CAP_FALCON_1024,
    }
}

/// Tensor geometry of one padded batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct Geometry {
    pub degree: FalconDegree,
    /// `n`.
    pub n: usize,
    /// `nu = log2 n`.
    pub logn: usize,
    /// `N_sig*`.
    pub records: usize,
    /// `ell = log2 N_sig*`.
    pub record_bits: usize,
    /// `k = 12 - (nu + 1)`: record bits inside both first blocks.
    pub(crate) low_record_bits: usize,
    /// `R = ell - k`.
    pub rest_bits: usize,
    /// Bits per norm core, `[12, b_2, .., b_5]`.
    pub(crate) norm_bits: [usize; NORM_PREFIX_LAYERS + 1],
    /// Bits per residual core, `[11, floor(R/2), ceil(R/2)]`.
    pub(crate) residual_bits: [usize; RESIDUAL_LAYERS],
}

impl Geometry {
    pub fn new(degree: FalconDegree, records: usize) -> Result<Self, crate::UnsupportedBatch> {
        if records == 0 || !records.is_power_of_two() || records > MAX_RECORDS {
            return Err(crate::UnsupportedBatch::Shape {
                degree: degree.degree(),
                records,
            });
        }
        let logn = degree.logn();
        let record_bits = records.trailing_zeros() as usize;
        let low_record_bits = NORM_FIRST_BLOCK_BITS - (logn + 1);
        if record_bits < low_record_bits {
            return Err(crate::UnsupportedBatch::Shape {
                degree: degree.degree(),
                records,
            });
        }
        let rest_bits = record_bits - low_record_bits;
        let mut norm_bits = [0; NORM_PREFIX_LAYERS + 1];
        norm_bits[0] = NORM_FIRST_BLOCK_BITS;
        let mut remaining = rest_bits;
        for bits in &mut norm_bits[1..NORM_PREFIX_LAYERS] {
            *bits = remaining.min(NORM_LATER_BLOCK_BITS);
            remaining -= *bits;
        }
        norm_bits[NORM_PREFIX_LAYERS] = remaining;
        let residual_bits = [
            RESIDUAL_FIRST_BLOCK_BITS,
            rest_bits / 2,
            rest_bits - rest_bits / 2,
        ];
        debug_assert_eq!(logn + low_record_bits, RESIDUAL_FIRST_BLOCK_BITS);
        Ok(Self {
            degree,
            n: degree.degree(),
            logn,
            records,
            record_bits,
            low_record_bits,
            rest_bits,
            norm_bits,
            residual_bits,
        })
    }

    #[must_use]
    pub fn profile(&self) -> ProjectionProfile {
        ProjectionProfile {
            norm: NormLeg {
                groups: NORM_GROUPS,
                prefix_widths: [NORM_PREFIX_WIDTH; NORM_PREFIX_LAYERS],
                terminal_rows: NORM_TERMINAL_ROWS,
                radices: self.norm_radices(),
                distribution: CoreDistribution::TernaryQuarterHalfQuarter,
                source_order: NormSourceOrder::S1ThenSlack,
                cap: norm_cap(self.degree),
            },
            residual: ResidualLeg {
                widths: [RESIDUAL_WIDTH; RESIDUAL_LAYERS],
                radices: self.residual_radices(),
                distribution: CoreDistribution::UniformSymmetric {
                    bound: RESIDUAL_BOUND,
                },
                modulus: FALCON_Q,
            },
        }
    }

    #[must_use]
    pub fn norm_radices(&self) -> [usize; NORM_PREFIX_LAYERS + 1] {
        self.norm_bits.map(|bits| 1 << bits)
    }

    #[must_use]
    pub fn residual_radices(&self) -> [usize; RESIDUAL_LAYERS] {
        self.residual_bits.map(|bits| 1 << bits)
    }

    /// `2^R`: fibers of the rest bits.
    #[must_use]
    pub const fn rest_size(&self) -> usize {
        1 << self.rest_bits
    }

    /// `2^k`: records inside one first block.
    #[must_use]
    pub const fn low_records(&self) -> usize {
        1 << self.low_record_bits
    }
}
