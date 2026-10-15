//! Core derivation (`specs/falcon.md` Section 6.3).
//!
//! Every row of every core has its own cSHAKE256 stream,
//!
//! ```text
//! stream(core, row) = cSHAKE256(prefix(core) || LE64(row), N="", S="aerie/falcon/v1/projection")
//! prefix(core)      = "aerie/falcon/v1/projection/core-v1"
//!                     || LE16(WIRE_VERSION = 1) || U8(log2 n)
//!                     || LE64(N_sig) || LE64(N_sig*)
//!                     || U8(leg) || U8(group) || U8(layer)
//!                     || LE64(rows) || LE64(columns)
//!                     || U8(radix count) || LE64(radix) for each radix of the leg
//!                     || LE64(seed length) || seed
//! ```
//!
//! - `leg` is 0 for the norm leg and 1 for the residual leg.
//! - `group` is the norm group `0..4`, [`TERMINAL_GROUP`] for the norm
//!   terminal core, and 0 for the residual cores.
//! - `layer` is the one-based core index along its leg: 1 to 4 for the norm
//!   prefix cores, 5 for the norm terminal core, 1 to 3 for the residual
//!   cores.
//! - The radix tuple is the leg's full tuple: five radices for the norm leg,
//!   three for the residual leg.
//!
//! The label is not length-prefixed; it is a fixed constant and the first
//! field. Every later variable-length field is length-prefixed, so the
//! prefix is injective in its fields.
//!
//! Entries are decoded from the row stream alone, so rows are random-access
//! and independent of thread count and SIMD width.
//!
//! - **Ternary** (norm cores): entry `c` is bit pair `c mod 4` (low bits
//!   first) of stream byte `floor(c / 4)`, mapped as in Section 6.3:
//!   `00 -> -1`, `11 -> 1`, `01, 10 -> 0`.
//! - **Uniform** (residual cores): entry `c` is the `c`-th stream byte below
//!   250, mapped to `(byte mod 25) - 12`. Bytes `250..=255` are skipped.
//!
//! Core `(c', column)` entries act on a column index that places the
//! incoming channel low: a later core's column is `c + width * u`, where `c`
//! is the incoming channel and `u` the radix digit it contracts. The norm
//! terminal core's column is `(43 g + c) + 172 u_5`, with `g` the group.

use aerie_keccak::{CShake256, SHAKE256_RATE_BYTES};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::super::FalconDegree;
use super::profile::{
    CoreDistribution, Geometry, NORM_CHANNELS, NORM_GROUPS, NORM_PREFIX_LAYERS, NORM_PREFIX_WIDTH,
    NORM_TERMINAL_ROWS, RESIDUAL_BOUND, RESIDUAL_LAYERS, RESIDUAL_WIDTH,
};

/// XOF label of every core row.
pub const CORE_XOF_LABEL: &[u8] = b"aerie/falcon/v1/projection/core-v1";
/// Group byte of the norm terminal core.
pub const TERMINAL_GROUP: u8 = 0xff;
/// Uniform rejection threshold: the largest multiple of 25 not above 256.
const UNIFORM_ACCEPT_BELOW: u8 = 250;
/// Values of the uniform alphabet.
const UNIFORM_ALPHABET: u8 = 2 * RESIDUAL_BOUND + 1;
/// Rows per parallel task; a multiple of every Keccak backend width.
#[cfg(feature = "parallel")]
const PARALLEL_CORE_ROWS: usize = 8;

/// Leg of a core.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Leg {
    Norm,
    Residual,
}

impl Leg {
    pub(crate) const fn byte(self) -> u8 {
        match self {
            Self::Norm => 0,
            Self::Residual => 1,
        }
    }
}

/// Identity of one core.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CoreId {
    pub(crate) leg: Leg,
    pub(crate) group: u8,
    pub(crate) layer: u8,
}

/// Public inputs every core stream binds besides its own shape.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CoreContext<'a> {
    pub(crate) seed: &'a [u8],
    pub(crate) degree: FalconDegree,
    pub(crate) live_rows: usize,
    pub(crate) records: usize,
}

/// One sampled core, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub struct Core {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) entries: Vec<i8>,
}

impl Core {
    /// Construct a small explicit core for arithmetic regression tests.
    #[cfg(any(test, feature = "internals"))]
    #[doc(hidden)]
    #[must_use]
    pub fn from_test_entries(rows: usize, cols: usize, entries: Vec<i8>) -> Self {
        Self {
            rows,
            cols,
            entries,
        }
    }

    #[inline]
    #[must_use]
    pub fn row(&self, row: usize) -> &[i8] {
        &self.entries[row * self.cols..(row + 1) * self.cols]
    }

    /// Borrow the row-major sampled entries.
    #[inline]
    #[must_use]
    pub fn entries(&self) -> &[i8] {
        &self.entries
    }

    /// Number of sampled rows.
    #[inline]
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of entries per row.
    #[inline]
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    #[cfg(any(test, feature = "internals"))]
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> i8 {
        self.entries[row * self.cols + col]
    }
}

/// The stream prefix of one core; see the module documentation.
pub(crate) fn core_xof_prefix(
    context: &CoreContext<'_>,
    id: CoreId,
    rows: usize,
    cols: usize,
    radices: &[usize],
) -> Vec<u8> {
    let le64 =
        |out: &mut Vec<u8>, value: usize| out.extend_from_slice(&(value as u64).to_le_bytes());
    let mut prefix =
        Vec::with_capacity(CORE_XOF_LABEL.len() + 64 + 8 * radices.len() + context.seed.len());
    prefix.extend_from_slice(CORE_XOF_LABEL);
    prefix.extend_from_slice(&context.degree.identity_bytes());
    le64(&mut prefix, context.live_rows);
    le64(&mut prefix, context.records);
    prefix.push(id.leg.byte());
    prefix.push(id.group);
    prefix.push(id.layer);
    le64(&mut prefix, rows);
    le64(&mut prefix, cols);
    prefix.push(u8::try_from(radices.len()).expect("a leg has at most five radices"));
    for &radix in radices {
        le64(&mut prefix, radix);
    }
    le64(&mut prefix, context.seed.len());
    prefix.extend_from_slice(context.seed);
    prefix
}

/// The Section 6.3 bit-pair map.
#[inline]
pub(crate) const fn decode_ternary(byte: u8, slot: usize) -> i8 {
    match (byte >> (2 * slot)) & 3 {
        0 => -1,
        3 => 1,
        _ => 0,
    }
}

/// The four entries [`decode_ternary`] gives each byte, in slot order.
const fn ternary_quads() -> [[i8; 4]; 256] {
    let mut table = [[0_i8; 4]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut slot = 0;
        while slot < 4 {
            #[allow(clippy::cast_possible_truncation)]
            let value = decode_ternary(byte as u8, slot);
            table[byte][slot] = value;
            slot += 1;
        }
        byte += 1;
    }
    table
}

static TERNARY_QUADS: [[i8; 4]; 256] = ternary_quads();

/// The uniform map of one accepted byte, or `None` for a rejected byte. The
/// row fill applies the same map without a branch; tests check it against this.
#[cfg(test)]
pub(crate) const fn decode_uniform(byte: u8) -> Option<i8> {
    if byte < UNIFORM_ACCEPT_BELOW {
        Some(uniform_value(byte))
    } else {
        None
    }
}

/// The uniform map of one byte, before the rejection test.
#[inline]
const fn uniform_value(byte: u8) -> i8 {
    (byte % UNIFORM_ALPHABET) as i8 - RESIDUAL_BOUND as i8
}

/// The cSHAKE256 state after absorbing one core's prefix, shared by its rows.
pub(crate) fn core_stream(prefix: &[u8]) -> CShake256 {
    let mut stream = CShake256::new(b"", b"aerie/falcon/v1/projection");
    stream.update(prefix);
    stream
}

/// Fill consecutive core rows, starting at `first_row`, from their streams.
///
/// `out` holds whole rows of `cols` entries. Row `r` reads the output of
/// `core` extended with `LE64(r)`.
pub(crate) fn fill_core_rows(
    core: &CShake256,
    first_row: usize,
    distribution: CoreDistribution,
    cols: usize,
    out: &mut [i8],
) {
    debug_assert!(cols > 0 && out.len().is_multiple_of(cols));
    let rows = out.len() / cols;
    let suffixes = (first_row..first_row + rows)
        .map(|row| (row as u64).to_le_bytes())
        .collect::<Vec<_>>();
    let mut filled = vec![0_usize; rows];
    match distribution {
        CoreDistribution::TernaryQuarterHalfQuarter => {
            core.squeeze_forks(&suffixes, |row, block| {
                let start = filled[row];
                let end = cols.min(start + 4 * block.len());
                let mut quads = out[row * cols + start..row * cols + end].chunks_exact_mut(4);
                for (quad, &byte) in (&mut quads).zip(block) {
                    quad.copy_from_slice(&TERNARY_QUADS[usize::from(byte)]);
                }
                let tail = quads.into_remainder();
                if !tail.is_empty() {
                    let byte = block[(end - start) / 4];
                    tail.copy_from_slice(&TERNARY_QUADS[usize::from(byte)][..tail.len()]);
                }
                filled[row] = end;
                end < cols
            });
        }
        CoreDistribution::UniformSymmetric { bound } => {
            assert_eq!(
                bound, RESIDUAL_BOUND,
                "only the certified uniform bound is supported"
            );
            core.squeeze_forks(&suffixes, |row, block| {
                // Decode every byte, then keep the accepted prefix; the
                // write index never passes the byte index.
                let mut accepted = [0_i8; SHAKE256_RATE_BYTES];
                let mut count = 0;
                for &byte in block {
                    accepted[count.min(SHAKE256_RATE_BYTES - 1)] = uniform_value(byte);
                    count += usize::from(byte < UNIFORM_ACCEPT_BELOW);
                }
                let start = filled[row];
                let take = count.min(cols - start);
                out[row * cols + start..row * cols + start + take]
                    .copy_from_slice(&accepted[..take]);
                filled[row] = start + take;
                filled[row] < cols
            });
        }
    }
}

/// Shape and identity of one core.
#[derive(Clone, Copy, Debug)]
struct CoreSpec<'r> {
    id: CoreId,
    rows: usize,
    cols: usize,
    radices: &'r [usize],
    distribution: CoreDistribution,
}

fn derive_core(context: &CoreContext<'_>, spec: &CoreSpec<'_>) -> Core {
    let prefix = core_xof_prefix(context, spec.id, spec.rows, spec.cols, spec.radices);
    let mut entries = vec![0_i8; spec.rows * spec.cols];
    let stream = core_stream(&prefix);
    #[cfg(feature = "parallel")]
    entries
        .par_chunks_mut(PARALLEL_CORE_ROWS * spec.cols)
        .enumerate()
        .for_each(|(chunk, out)| {
            fill_core_rows(
                &stream,
                chunk * PARALLEL_CORE_ROWS,
                spec.distribution,
                spec.cols,
                out,
            );
        });
    #[cfg(not(feature = "parallel"))]
    fill_core_rows(&stream, 0, spec.distribution, spec.cols, &mut entries);
    Core {
        rows: spec.rows,
        cols: spec.cols,
        entries,
    }
}

/// All 20 cores of one projection: 16 norm prefix cores, the norm terminal
/// core and 3 residual cores.
#[derive(Clone, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub struct ProjectionCores {
    /// `A_{g,i}` at index `NORM_PREFIX_LAYERS * g + (i - 1)`.
    norm_prefix: Vec<Core>,
    /// `T`.
    pub(crate) norm_terminal: Core,
    /// `B_i` at index `i - 1`.
    residual: Vec<Core>,
}

impl ProjectionCores {
    /// Derive every core of a batch geometry from the projection seed.
    #[must_use]
    pub fn derive(seed: &[u8], geometry: &Geometry, live_rows: usize) -> Self {
        let context = CoreContext {
            seed,
            degree: geometry.degree,
            live_rows,
            records: geometry.records,
        };
        let norm_radices = geometry.norm_radices();
        let residual_radices = geometry.residual_radices();
        let ternary = CoreDistribution::TernaryQuarterHalfQuarter;
        let uniform = CoreDistribution::UniformSymmetric {
            bound: RESIDUAL_BOUND,
        };
        let mut specs = Vec::with_capacity(NORM_GROUPS * NORM_PREFIX_LAYERS + 1 + RESIDUAL_LAYERS);
        for group in 0..NORM_GROUPS {
            for layer in 1..=NORM_PREFIX_LAYERS {
                let incoming = if layer == 1 { 1 } else { NORM_PREFIX_WIDTH };
                specs.push(CoreSpec {
                    id: CoreId {
                        leg: Leg::Norm,
                        group: group as u8,
                        layer: layer as u8,
                    },
                    rows: NORM_PREFIX_WIDTH,
                    cols: incoming * norm_radices[layer - 1],
                    radices: &norm_radices,
                    distribution: ternary,
                });
            }
        }
        specs.push(CoreSpec {
            id: CoreId {
                leg: Leg::Norm,
                group: TERMINAL_GROUP,
                layer: (NORM_PREFIX_LAYERS + 1) as u8,
            },
            rows: NORM_TERMINAL_ROWS,
            cols: NORM_CHANNELS * norm_radices[NORM_PREFIX_LAYERS],
            radices: &norm_radices,
            distribution: ternary,
        });
        for layer in 1..=RESIDUAL_LAYERS {
            let incoming = if layer == 1 { 1 } else { RESIDUAL_WIDTH };
            specs.push(CoreSpec {
                id: CoreId {
                    leg: Leg::Residual,
                    group: 0,
                    layer: layer as u8,
                },
                rows: RESIDUAL_WIDTH,
                cols: incoming * residual_radices[layer - 1],
                radices: &residual_radices,
                distribution: uniform,
            });
        }
        #[cfg(feature = "parallel")]
        let mut cores: Vec<Core> = specs
            .par_iter()
            .map(|spec| derive_core(&context, spec))
            .collect();
        #[cfg(not(feature = "parallel"))]
        let mut cores: Vec<Core> = specs
            .iter()
            .map(|spec| derive_core(&context, spec))
            .collect();
        let residual = cores.split_off(NORM_GROUPS * NORM_PREFIX_LAYERS + 1);
        let norm_terminal = cores.pop().expect("the terminal core was derived");
        Self {
            norm_prefix: cores,
            norm_terminal,
            residual,
        }
    }

    /// Borrow the terminal norm core.
    #[inline]
    #[must_use]
    pub const fn norm_terminal(&self) -> &Core {
        &self.norm_terminal
    }

    /// `A_{g,i}` for the one-based layer `i`.
    #[inline]
    #[must_use]
    pub fn norm(&self, group: usize, layer: usize) -> &Core {
        debug_assert!((1..=NORM_PREFIX_LAYERS).contains(&layer));
        &self.norm_prefix[NORM_PREFIX_LAYERS * group + layer - 1]
    }

    /// `B_i` for the one-based layer `i`.
    #[inline]
    #[must_use]
    pub fn residual(&self, layer: usize) -> &Core {
        debug_assert!((1..=RESIDUAL_LAYERS).contains(&layer));
        &self.residual[layer - 1]
    }
}
