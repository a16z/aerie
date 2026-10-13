//! Compact grouped executor for the basis 16 product prefix.
//!
//! This module has no transcript or proof-format logic. It keeps borrowed
//! `i8` sources through the first three rounds, then retains only the folded
//! virtual quartic leaves required by the terminal claims.

use std::sync::Arc;

use akita_error::AkitaError;
use jolt_field::{Field, One, Ring, Zero};
use jolt_field::{Fold, Unreduced};

#[cfg(feature = "parallel")]
use super::kernel_common::PARALLEL_MIN_WORK;
use super::kernel_common::{fold_rows_in_place, record_phase};
use super::opening_field_ops::mul;
use crate::protocol::source_tables::CodedSource;
use crate::substrate::OpeningField as E;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const LEAVES_PER_SOURCE: usize = 2;
const STORED_COEFFICIENTS: usize = 2;

enum CompactLut {
    Pairs(Vec<[E; LEAVES_PER_SOURCE]>),
    Quartets(Vec<[E; LEAVES_PER_SOURCE]>),
}

pub(crate) struct ProductGroup<'a> {
    sources: Vec<CodedSource<'a>>,
    source_weights: Vec<E>,
    pair_indices: Vec<Arc<[u8]>>,
    leaf_lookup: [[E; LEAVES_PER_SOURCE]; RANGE_IMAGE_PATTERNS],
    domain_len: usize,
    num_rounds: usize,
    rounds_completed: usize,
    defer_dense_materialization: bool,
    compact_challenges: [Option<E>; 2],
    /// Folded leaves indexed by compact range-image patterns. After the first
    /// bind this holds 64 pair entries; after the second it holds 4,096
    /// quartet entries. The state is shared by every source.
    compact_lut: Option<CompactLut>,
    bivariate_suffix_masses: Option<Vec<E>>,
    cached_second_round: Option<[E; STORED_COEFFICIENTS]>,
    folded_leaves: Option<Vec<E>>,
}

impl<'a> ProductGroup<'a> {
    #[tracing::instrument(
        skip_all,
        name = "ProductGroup::new",
        fields(source_count = sources.len(), num_rounds)
    )]
    pub(crate) fn new(
        sources: Vec<&CodedSource<'a>>,
        source_weights: Vec<E>,
        num_rounds: usize,
    ) -> Result<Self, AkitaError> {
        if sources.is_empty() || sources.len() != source_weights.len() {
            return Err(AkitaError::InvalidInput(
                "basis16 product sources and weights must have the same nonzero length".to_owned(),
            ));
        }
        let shift = u32::try_from(num_rounds).map_err(|_| {
            AkitaError::InvalidInput("basis16 product domain overflows usize".into())
        })?;
        let domain_len = 1_usize.checked_shl(shift).ok_or_else(|| {
            AkitaError::InvalidInput("basis16 product domain overflows usize".into())
        })?;
        if sources
            .iter()
            .any(|source| source.cell_count() != domain_len)
        {
            return Err(AkitaError::InvalidInput(
                "basis16 product source length does not match its equality point".to_owned(),
            ));
        }
        let pair_indices = sources
            .iter()
            .map(|source| source.image8_codes())
            .collect::<Vec<_>>();
        if pair_indices.iter().any(|codes| {
            codes.len() != domain_len / 2
                || codes.iter().any(|&code| usize::from(code) >= PAIR_PATTERNS)
        }) {
            return Err(AkitaError::InvalidInput(
                "basis16 pair codes do not match the source domain".to_owned(),
            ));
        }
        if domain_len == 1 {
            for source in &sources {
                let digit = source.digit(0).ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "basis16 affine source digit overflows i8 at index 0".to_owned(),
                    )
                })?;
                if !(-8..=7).contains(&digit) {
                    return Err(AkitaError::InvalidInput(format!(
                        "basis16 source digit {digit} at index 0 is outside [-8, 7]"
                    )));
                }
            }
        }

        let mut group = Self {
            sources: sources.into_iter().cloned().collect(),
            source_weights,
            pair_indices,
            leaf_lookup: core::array::from_fn(|index| {
                basis16_leaf_values(E::from_i64(
                    i64::try_from(index).expect("basis 16 range-image index fits i64"),
                ))
            }),
            domain_len,
            num_rounds,
            rounds_completed: 0,
            defer_dense_materialization: num_rounds >= 3,
            compact_challenges: [None; 2],
            compact_lut: None,
            bivariate_suffix_masses: None,
            cached_second_round: None,
            folded_leaves: None,
        };
        if num_rounds == 0 {
            // A one-cell domain has no round to fold: the terminal leaves are
            // the range-image leaves of that cell (mirrors the quartic group).
            let mut terminal = vec![E::zero(); group.lane_count()];
            for source in 0..group.sources.len() {
                let leaves = group.leaves(source, 0);
                terminal[LEAVES_PER_SOURCE * source..LEAVES_PER_SOURCE * (source + 1)]
                    .copy_from_slice(&leaves);
            }
            group.folded_leaves = Some(terminal);
            group.pair_indices.clear();
        }
        Ok(group)
    }

    #[must_use]
    pub(crate) fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    #[tracing::instrument(
        skip_all,
        name = "product_group_round",
        fields(
            round = self.rounds_completed,
            phase = tracing::field::Empty,
            source_count = self.sources.len(),
            domain_len = self.domain_len,
            num_rounds = self.num_rounds,
            remaining_pairs = (self.domain_len >> self.rounds_completed) / 2
        )
    )]
    pub(crate) fn round_coefficients(
        &mut self,
        equality_first: &[E],
        equality_second: &[E],
    ) -> [E; STORED_COEFFICIENTS] {
        debug_assert!(self.rounds_completed < self.num_rounds());
        let pair_count = (self.domain_len >> self.rounds_completed) / 2;
        debug_assert_eq!(equality_first.len() * equality_second.len(), pair_count);
        match self.rounds_completed {
            0 if self.bivariate_prefix_applies(pair_count) => {
                record_phase("compact-bivariate-prefix");
                self.bivariate_first_round(equality_first, equality_second, pair_count)
            }
            0 if pair_count >= BUCKET_MIN_RATIO * PAIR_PATTERNS => {
                record_phase("compact-pairs-bucketed");
                self.bucketed_pair_coefficients(equality_first, equality_second, pair_count)
            }
            0 => {
                record_phase("compact-pairs-direct");
                self.direct_pair_coefficients(equality_first, equality_second, pair_count)
            }
            1 if self.cached_second_round.is_some() => {
                record_phase("compact-bivariate-cached");
                self.cached_second_round
                    .expect("the bivariate prefix cached round two")
            }
            1 => {
                let first_challenge = self.compact_challenges[0]
                    .expect("the first challenge is present in round one");
                if pair_count >= BUCKET_MIN_RATIO * QUARTET_PATTERNS {
                    record_phase("compact-quartets-bucketed");
                    self.bucketed_quartet_coefficients(
                        equality_first,
                        equality_second,
                        pair_count,
                        first_challenge,
                    )
                } else {
                    record_phase("compact-quartets-direct");
                    self.direct_quartet_coefficients(
                        equality_first,
                        equality_second,
                        pair_count,
                        first_challenge,
                    )
                }
            }
            2 if self.defer_dense_materialization => {
                record_phase("compact-octets");
                let challenges = [
                    self.compact_challenges[0]
                        .expect("the first compact challenge is present in round two"),
                    self.compact_challenges[1]
                        .expect("the second compact challenge is present in round two"),
                ];
                sum_split_coefficients(equality_first, equality_second, |octet| {
                    self.compact_octet_coefficients(octet, &challenges)
                })
            }
            _ => {
                record_phase("dense");
                let table = self
                    .folded_leaves
                    .as_ref()
                    .expect("folded leaves exist after the compact rounds");
                let lanes = self.lane_count();
                debug_assert_eq!(table.len(), 2 * pair_count * lanes);
                sum_split_coefficients(equality_first, equality_second, |pair_index| {
                    let left = &table[(2 * pair_index) * lanes..(2 * pair_index + 1) * lanes];
                    let right = &table[(2 * pair_index + 1) * lanes..(2 * pair_index + 2) * lanes];
                    product_coefficients(left, right, &self.source_weights)
                })
            }
        }
    }

    #[tracing::instrument(
        skip_all,
        name = "product_group_bind",
        fields(
            round = self.rounds_completed,
            phase = tracing::field::Empty,
            source_count = self.sources.len(),
            domain_len = self.domain_len,
            num_rounds = self.num_rounds,
            remaining_rows = self.domain_len >> self.rounds_completed
        )
    )]
    pub(crate) fn bind(&mut self, challenge: E) {
        debug_assert!(self.rounds_completed < self.num_rounds());
        match self.rounds_completed {
            0 if self.num_rounds == 1 => {
                record_phase("materialize-after-first");
                self.folded_leaves = Some(self.materialize_after_one_round(challenge));
                self.pair_indices.clear();
            }
            0 => {
                record_phase("retain-compact-pairs");
                self.compact_challenges[0] = Some(challenge);
                self.compact_lut = Some(CompactLut::Pairs(self.build_folded_pair_lut(challenge)));
                let masses = self.bivariate_suffix_masses.take();
                if let Some(masses) = masses {
                    self.cached_second_round =
                        Some(self.bivariate_second_round(&masses, challenge));
                }
            }
            1 if !self.defer_dense_materialization => {
                record_phase("materialize-after-second");
                let first_challenge = self.compact_challenges[0]
                    .expect("the first challenge is present before round two");
                self.folded_leaves =
                    Some(self.materialize_after_two_rounds(first_challenge, challenge));
                self.pair_indices.clear();
                self.compact_lut = None;
                self.cached_second_round = None;
            }
            1 => {
                record_phase("retain-compact-quartets");
                self.compact_challenges[1] = Some(challenge);
                self.compact_lut = self
                    .build_folded_quartet_lut(challenge)
                    .map(CompactLut::Quartets);
                self.cached_second_round = None;
            }
            2 if self.defer_dense_materialization => {
                record_phase("materialize-after-third");
                let compact_challenges = [
                    self.compact_challenges[0]
                        .expect("the first compact challenge is present before round three"),
                    self.compact_challenges[1]
                        .expect("the second compact challenge is present before round three"),
                ];
                self.folded_leaves =
                    Some(self.materialize_after_three_rounds(&compact_challenges, challenge));
                self.pair_indices.clear();
                self.compact_lut = None;
            }
            _ => {
                record_phase("fold-dense");
                let table = self
                    .folded_leaves
                    .as_mut()
                    .expect("folded leaves exist after the compact rounds");
                fold_rows_in_place(
                    table,
                    self.sources.len() * LEAVES_PER_SOURCE,
                    challenge,
                    "product",
                );
            }
        }
        self.rounds_completed += 1;
    }

    #[tracing::instrument(skip_all, name = "product_group_terminal")]
    pub(crate) fn terminal_leaves(&self) -> Result<Vec<E>, AkitaError> {
        if self.rounds_completed != self.num_rounds() {
            return Err(AkitaError::InvalidInput(
                "basis16 product executor has not completed every round".to_owned(),
            ));
        }
        let leaves = self.folded_leaves.as_ref().ok_or_else(|| {
            AkitaError::InvalidInput("basis16 product has no terminal leaves".into())
        })?;
        if leaves.len() != self.lane_count() {
            return Err(AkitaError::InvalidSize {
                expected: self.lane_count(),
                actual: leaves.len(),
            });
        }
        Ok(leaves.clone())
    }

    fn lane_count(&self) -> usize {
        self.sources.len() * LEAVES_PER_SOURCE
    }

    fn materialize_rows(&self, rows: usize, fill: impl Fn(usize, &mut [E]) + Sync) -> Vec<E> {
        let lanes = self.lane_count();
        let mut table = vec![E::zero(); rows * lanes];
        #[cfg(feature = "parallel")]
        if rows >= PARALLEL_MIN_WORK {
            table
                .par_chunks_mut(lanes)
                .enumerate()
                .for_each(|(row, output)| fill(row, output));
            return table;
        }
        for (row, output) in table.chunks_mut(lanes).enumerate() {
            fill(row, output);
        }
        table
    }

    fn bivariate_prefix_applies(&self, pair_count: usize) -> bool {
        self.num_rounds >= 2
            && (cfg!(test) || pair_count / 2 >= BUCKET_MIN_RATIO * QUARTET_PATTERNS)
    }

    /// Build the transient two-variable grid in pattern-mass form. The two
    /// current round messages are reconstructed from it, but remain ordinary
    /// univariate transcript messages.
    fn bivariate_first_round(
        &mut self,
        equality_first: &[E],
        equality_second: &[E],
        pair_count: usize,
    ) -> [E; STORED_COEFFICIENTS] {
        let quartets = pair_count / 2;
        let slots = self.sources.len() * QUARTET_PATTERNS;
        let (linear_zero, linear_one) = bivariate_linear_evals(equality_first, equality_second);
        let accumulate = |masses: &mut Vec<E>, quartet: usize| {
            let weight = bivariate_suffix_weight(equality_first, equality_second, quartet);
            for source in 0..self.sources.len() {
                let pair = 2 * quartet;
                let pattern = usize::from(self.pair_indices[source][pair]) * PAIR_PATTERNS
                    + usize::from(self.pair_indices[source][pair + 1]);
                let slot = source * QUARTET_PATTERNS + pattern;
                masses[slot] += weight;
            }
        };
        #[cfg(feature = "parallel")]
        let masses = if quartets >= PARALLEL_MIN_WORK {
            (0..quartets)
                .into_par_iter()
                .with_min_len(1 << 12)
                .fold(
                    || vec![E::zero(); slots],
                    |mut masses, quartet| {
                        accumulate(&mut masses, quartet);
                        masses
                    },
                )
                .reduce(
                    || vec![E::zero(); slots],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                        left
                    },
                )
        } else {
            let mut masses = vec![E::zero(); slots];
            for quartet in 0..quartets {
                accumulate(&mut masses, quartet);
            }
            masses
        };
        #[cfg(not(feature = "parallel"))]
        let masses = {
            let mut masses = vec![E::zero(); slots];
            for quartet in 0..quartets {
                accumulate(&mut masses, quartet);
            }
            masses
        };

        let mut output = [E::zero(); STORED_COEFFICIENTS];
        let mut suffix_masses = vec![E::zero(); QUARTET_PATTERNS];
        for pattern in 0..QUARTET_PATTERNS {
            let mut mass = E::zero();
            for (source, &source_weight) in self.source_weights.iter().enumerate() {
                mass += source_weight * masses[source * QUARTET_PATTERNS + pattern];
            }
            suffix_masses[pattern] = mass;
            let digits: [usize; 4] = core::array::from_fn(|slot| {
                (pattern / RANGE_IMAGE_PATTERNS.pow(3 - slot as u32)) % RANGE_IMAGE_PATTERNS
            });
            add_scaled_coefficients(
                &mut output,
                product_coefficients(
                    &self.leaf_lookup[digits[0]],
                    &self.leaf_lookup[digits[1]],
                    &[E::one()],
                ),
                linear_zero * mass,
            );
            add_scaled_coefficients(
                &mut output,
                product_coefficients(
                    &self.leaf_lookup[digits[2]],
                    &self.leaf_lookup[digits[3]],
                    &[E::one()],
                ),
                linear_one * mass,
            );
        }
        self.bivariate_suffix_masses = Some(suffix_masses);
        output
    }

    fn bivariate_second_round(
        &self,
        suffix_masses: &[E],
        first_challenge: E,
    ) -> [E; STORED_COEFFICIENTS] {
        let context = E::precompute(first_challenge);
        let mut output = [E::zero(); STORED_COEFFICIENTS];
        for (pattern, &mass) in suffix_masses.iter().enumerate() {
            let digits: [usize; 4] = core::array::from_fn(|slot| {
                (pattern / RANGE_IMAGE_PATTERNS.pow(3 - slot as u32)) % RANGE_IMAGE_PATTERNS
            });
            let left = fold_leaf_pair_with_context(
                self.leaf_lookup[digits[0]],
                self.leaf_lookup[digits[1]],
                &context,
            );
            let right = fold_leaf_pair_with_context(
                self.leaf_lookup[digits[2]],
                self.leaf_lookup[digits[3]],
                &context,
            );
            add_scaled_coefficients(
                &mut output,
                product_coefficients(&left, &right, &[E::one()]),
                mass,
            );
        }
        output
    }

    fn leaves(&self, source: usize, index: usize) -> [E; LEAVES_PER_SOURCE] {
        let digit = self.sources[source]
            .digit(index)
            .expect("basis16 product source digit was validated");
        self.leaf_lookup[range_image_index(digit)]
    }

    #[must_use]
    fn direct_pair_coefficients(
        &self,
        equality_first: &[E],
        equality_second: &[E],
        _pair_count: usize,
    ) -> [E; STORED_COEFFICIENTS] {
        sum_split_coefficients(equality_first, equality_second, |pair_index| {
            self.compact_pair_coefficients(pair_index)
        })
    }

    #[must_use]
    fn direct_quartet_coefficients(
        &self,
        equality_first: &[E],
        equality_second: &[E],
        _quartet_count: usize,
        first_challenge: E,
    ) -> [E; STORED_COEFFICIENTS] {
        sum_split_coefficients(equality_first, equality_second, |pair_index| {
            self.compact_quartet_coefficients(pair_index, first_challenge)
        })
    }

    /// Per-source equality-weight sums bucketed by the digit pattern of
    /// each item's cells (digits are validated into `[-8, 7]`, sixteen
    /// values). Only field additions run per item; the leaf-product
    /// arithmetic runs once per pattern in the caller. Field addition is
    /// associative and commutative, so the round message equals the direct
    /// path's bit for bit.
    /// Block-factored variant of [`Self::pattern_weight_sums`]: the split
    /// equality weight is `first[i] * second[j]`, so per `j`-block the
    /// `first` values accumulate into per-pattern partials (additions
    /// only) and one flush scales them by `second[j]`. Distributivity
    /// makes the sums equal the plain accumulation bit for bit; the gate
    /// in the caller keeps the flush cost below the multiplies saved.
    #[tracing::instrument(skip_all, name = "ProductGroup::accumulate_factored_pattern_weights")]
    fn factored_pattern_weight_sums(
        &self,
        equality_first: &[E],
        equality_second: &[E],
        cells_per_item: usize,
        patterns: usize,
    ) -> Vec<E> {
        let sources = self.sources.len();
        let first_len = equality_first.len();
        let blocks = equality_second.len();
        let accumulate = |sums: &mut Vec<E>, inner: &mut Vec<E>, block: usize| {
            for value in inner.iter_mut() {
                *value = E::zero();
            }
            let base = block * first_len;
            for (low, &first_weight) in equality_first.iter().enumerate() {
                let item_base = (base + low) * cells_per_item;
                for source in 0..sources {
                    let pair = item_base / 2;
                    let pattern = if cells_per_item == 2 {
                        usize::from(self.pair_indices[source][pair])
                    } else {
                        usize::from(self.pair_indices[source][pair]) * PAIR_PATTERNS
                            + usize::from(self.pair_indices[source][pair + 1])
                    };
                    inner[source * patterns + pattern] += first_weight;
                }
            }
            let scale = equality_second[block];
            for (sum, &partial) in sums.iter_mut().zip(inner.iter()) {
                *sum += scale * partial;
            }
        };
        #[cfg(feature = "parallel")]
        if blocks * first_len >= PARALLEL_MIN_WORK {
            return (0..blocks)
                .into_par_iter()
                .fold(
                    || {
                        (
                            vec![E::zero(); sources * patterns],
                            vec![E::zero(); sources * patterns],
                        )
                    },
                    |(mut sums, mut inner), block| {
                        accumulate(&mut sums, &mut inner, block);
                        (sums, inner)
                    },
                )
                .map(|(sums, _)| sums)
                .reduce(
                    || vec![E::zero(); sources * patterns],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                        left
                    },
                );
        }
        let mut sums = vec![E::zero(); sources * patterns];
        let mut inner = vec![E::zero(); sources * patterns];
        for block in 0..blocks {
            accumulate(&mut sums, &mut inner, block);
        }
        sums
    }

    #[tracing::instrument(skip_all, name = "ProductGroup::accumulate_pattern_weights")]
    fn pattern_weight_sums(
        &self,
        equality_first: &[E],
        equality_second: &[E],
        items: usize,
        cells_per_item: usize,
        patterns: usize,
    ) -> Vec<E> {
        let sources = self.sources.len();
        // Factor the per-item weight multiply out when the per-block flush
        // stays below the multiplies it replaces.
        if sources * patterns <= equality_first.len() / 2
            && items == equality_first.len() * equality_second.len()
        {
            return self.factored_pattern_weight_sums(
                equality_first,
                equality_second,
                cells_per_item,
                patterns,
            );
        }
        let accumulate = |sums: &mut Vec<E>, item: usize| {
            let weight = split_equality_weight(equality_first, equality_second, item);
            let base = item * cells_per_item;
            for source in 0..sources {
                let pair = base / 2;
                let pattern = if cells_per_item == 2 {
                    usize::from(self.pair_indices[source][pair])
                } else {
                    usize::from(self.pair_indices[source][pair]) * PAIR_PATTERNS
                        + usize::from(self.pair_indices[source][pair + 1])
                };
                sums[source * patterns + pattern] += weight;
            }
        };
        #[cfg(feature = "parallel")]
        if items >= PARALLEL_MIN_WORK {
            return (0..items)
                .into_par_iter()
                .with_min_len(1 << 12)
                .fold(
                    || vec![E::zero(); sources * patterns],
                    |mut sums, item| {
                        accumulate(&mut sums, item);
                        sums
                    },
                )
                .reduce(
                    || vec![E::zero(); sources * patterns],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                        left
                    },
                );
        }
        let mut sums = vec![E::zero(); sources * patterns];
        for item in 0..items {
            accumulate(&mut sums, item);
        }
        sums
    }

    #[must_use]
    fn bucketed_pair_coefficients(
        &self,
        equality_first: &[E],
        equality_second: &[E],
        pair_count: usize,
    ) -> [E; STORED_COEFFICIENTS] {
        let sums = self.pattern_weight_sums(
            equality_first,
            equality_second,
            pair_count,
            2,
            PAIR_PATTERNS,
        );
        let mut output = [E::zero(); STORED_COEFFICIENTS];
        for pattern in 0..PAIR_PATTERNS {
            let mass = self
                .source_weights
                .iter()
                .enumerate()
                .fold(E::zero(), |mass, (source, &weight)| {
                    mass + weight * sums[source * PAIR_PATTERNS + pattern]
                });
            let left = self.leaf_lookup[pattern / RANGE_IMAGE_PATTERNS];
            let right = self.leaf_lookup[pattern % RANGE_IMAGE_PATTERNS];
            add_scaled_coefficients(
                &mut output,
                product_coefficients(&left, &right, &[E::one()]),
                mass,
            );
        }
        output
    }

    #[must_use]
    fn bucketed_quartet_coefficients(
        &self,
        equality_first: &[E],
        equality_second: &[E],
        quartet_count: usize,
        first_challenge: E,
    ) -> [E; STORED_COEFFICIENTS] {
        let sums = self.pattern_weight_sums(
            equality_first,
            equality_second,
            quartet_count,
            4,
            QUARTET_PATTERNS,
        );
        let mut output = [E::zero(); STORED_COEFFICIENTS];
        for pattern in 0..QUARTET_PATTERNS {
            let mass = self
                .source_weights
                .iter()
                .enumerate()
                .fold(E::zero(), |mass, (source, &weight)| {
                    mass + weight * sums[source * QUARTET_PATTERNS + pattern]
                });
            let digits: [usize; 4] = core::array::from_fn(|slot| {
                (pattern / RANGE_IMAGE_PATTERNS.pow(3 - slot as u32)) % RANGE_IMAGE_PATTERNS
            });
            let first = fold_leaf_pair(
                self.leaf_lookup[digits[0]],
                self.leaf_lookup[digits[1]],
                first_challenge,
            );
            let second = fold_leaf_pair(
                self.leaf_lookup[digits[2]],
                self.leaf_lookup[digits[3]],
                first_challenge,
            );
            add_scaled_coefficients(
                &mut output,
                product_coefficients(&first, &second, &[E::one()]),
                mass,
            );
        }
        output
    }

    fn compact_pair_coefficients(&self, pair_index: usize) -> [E; STORED_COEFFICIENTS] {
        let mut coefficients = [E::zero(); STORED_COEFFICIENTS];
        for (source, &source_weight) in self.source_weights.iter().enumerate() {
            let left = self.leaves(source, 2 * pair_index);
            let right = self.leaves(source, 2 * pair_index + 1);
            add_scaled_coefficients(
                &mut coefficients,
                product_coefficients(&left, &right, &[E::one()]),
                source_weight,
            );
        }
        coefficients
    }

    fn compact_quartet_coefficients(
        &self,
        quartet_index: usize,
        first_challenge: E,
    ) -> [E; STORED_COEFFICIENTS] {
        let mut coefficients = [E::zero(); STORED_COEFFICIENTS];
        let start = 4 * quartet_index;
        for (source, &source_weight) in self.source_weights.iter().enumerate() {
            let first = fold_leaf_pair(
                self.leaves(source, start),
                self.leaves(source, start + 1),
                first_challenge,
            );
            let second = fold_leaf_pair(
                self.leaves(source, start + 2),
                self.leaves(source, start + 3),
                first_challenge,
            );
            add_scaled_coefficients(
                &mut coefficients,
                product_coefficients(&first, &second, &[E::one()]),
                source_weight,
            );
        }
        coefficients
    }

    #[tracing::instrument(skip_all, name = "ProductGroup::materialize_after_first_round")]
    fn materialize_after_one_round(&self, challenge: E) -> Vec<E> {
        let rows = self.domain_len / 2;
        self.materialize_rows(rows, |row, output| {
            let start = 2 * row;
            for source in 0..self.sources.len() {
                let folded = fold_leaf_pair(
                    self.leaves(source, start),
                    self.leaves(source, start + 1),
                    challenge,
                );
                output[LEAVES_PER_SOURCE * source] = folded[0];
                output[LEAVES_PER_SOURCE * source + 1] = folded[1];
            }
        })
    }

    #[tracing::instrument(skip_all, name = "ProductGroup::materialize_after_second_round")]
    fn materialize_after_two_rounds(&self, first_challenge: E, second_challenge: E) -> Vec<E> {
        let rows = self.domain_len / 4;
        let first_context = E::precompute(first_challenge);
        let second_context = E::precompute(second_challenge);
        self.materialize_rows(rows, |row, output| {
            let start = 4 * row;
            for source in 0..self.sources.len() {
                let folded = self.folded_quartet_for_two_round_domain(
                    source,
                    start,
                    &first_context,
                    &second_context,
                    first_challenge,
                );
                output[LEAVES_PER_SOURCE * source] = folded[0];
                output[LEAVES_PER_SOURCE * source + 1] = folded[1];
            }
        })
    }

    fn compact_octet_coefficients(
        &self,
        octet: usize,
        challenges: &[E; 2],
    ) -> [E; STORED_COEFFICIENTS] {
        let start = 8 * octet;
        let mut output = [E::zero(); STORED_COEFFICIENTS];
        for (source, &weight) in self.source_weights.iter().enumerate() {
            if self.octet_zero_lane(source, octet).is_some() {
                continue;
            }
            let left = self.folded_quartet_after_two(source, start, challenges);
            let right = self.folded_quartet_after_two(source, start + 4, challenges);
            let delta0 = right[0] - left[0];
            let delta1 = right[1] - left[1];
            output[0] += mul(weight, mul(left[0], delta1) + mul(delta0, left[1]));
            output[1] += mul(mul(weight, delta0), delta1);
        }
        output
    }

    #[tracing::instrument(skip_all, name = "ProductGroup::materialize_after_third_round")]
    fn materialize_after_three_rounds(
        &self,
        compact_challenges: &[E; 2],
        third_challenge: E,
    ) -> Vec<E> {
        let rows = self.domain_len / 8;
        let third_context = E::precompute(third_challenge);
        self.materialize_rows(rows, |row, output| {
            let start = 8 * row;
            for source in 0..self.sources.len() {
                let left = self.folded_quartet_after_two(source, start, compact_challenges);
                let right = self.folded_quartet_after_two(source, start + 4, compact_challenges);
                output[LEAVES_PER_SOURCE * source] = E::fold_one(&third_context, left[0], right[0]);
                output[LEAVES_PER_SOURCE * source + 1] =
                    E::fold_one(&third_context, left[1], right[1]);
            }
        })
    }

    #[tracing::instrument(skip_all, name = "ProductGroup::build_pair_fold_lut")]
    fn build_folded_pair_lut(&self, first_challenge: E) -> Vec<[E; LEAVES_PER_SOURCE]> {
        let context = E::precompute(first_challenge);
        let mut lut = Vec::with_capacity(PAIR_PATTERNS);
        for pattern in 0..PAIR_PATTERNS {
            lut.push(fold_leaf_pair_with_context(
                self.leaf_lookup[pattern / RANGE_IMAGE_PATTERNS],
                self.leaf_lookup[pattern % RANGE_IMAGE_PATTERNS],
                &context,
            ));
        }
        lut
    }

    #[tracing::instrument(skip_all, name = "ProductGroup::build_quartet_fold_lut")]
    fn build_folded_quartet_lut(&self, second_challenge: E) -> Option<Vec<[E; LEAVES_PER_SOURCE]>> {
        let lut = self.compact_lut.as_ref()?;
        let CompactLut::Pairs(pair_lut) = lut else {
            unreachable!("round one requires the pair lookup")
        };
        let context = E::precompute(second_challenge);
        let mut lut = Vec::with_capacity(QUARTET_PATTERNS);
        for pattern in 0..QUARTET_PATTERNS {
            lut.push(fold_leaf_pair_with_context(
                pair_lut[pattern / PAIR_PATTERNS],
                pair_lut[pattern % PAIR_PATTERNS],
                &context,
            ));
        }
        Some(lut)
    }

    fn digit_pair_pattern(&self, source: usize, start: usize) -> usize {
        usize::from(self.pair_indices[source][start / 2])
    }

    fn quartet_pattern(&self, source: usize, start: usize) -> usize {
        self.digit_pair_pattern(source, start) * PAIR_PATTERNS
            + self.digit_pair_pattern(source, start + 2)
    }

    /// Return the leaf lane that vanishes when all eight digit classes in an
    /// octet lie in the same half of the range image.
    fn octet_zero_lane(&self, source: usize, octet: usize) -> Option<usize> {
        let first_pair = 4 * octet;
        let pair_codes = &self.pair_indices[source][first_pair..first_pair + 4];
        let mut high_bits_or = 0_u8;
        let mut high_bits_and = RANGE_IMAGE_HALF_MASK;
        for &pair_code in pair_codes {
            let high_bits = pair_code & RANGE_IMAGE_HALF_MASK;
            high_bits_or |= high_bits;
            high_bits_and &= high_bits;
        }
        if high_bits_or == 0 {
            Some(0)
        } else if high_bits_and == RANGE_IMAGE_HALF_MASK {
            Some(1)
        } else {
            None
        }
    }

    fn folded_quartet_for_two_round_domain(
        &self,
        source: usize,
        start: usize,
        first_context: &<E as Fold>::Ctx,
        second_context: &<E as Fold>::Ctx,
        first_challenge: E,
    ) -> [E; LEAVES_PER_SOURCE] {
        match self.compact_lut.as_ref() {
            Some(CompactLut::Pairs(lut)) => {
                debug_assert_eq!(Some(first_challenge), self.compact_challenges[0]);
                return fold_leaf_pair_with_context(
                    lut[self.digit_pair_pattern(source, start)],
                    lut[self.digit_pair_pattern(source, start + 2)],
                    second_context,
                );
            }
            Some(CompactLut::Quartets(_)) => {
                unreachable!("two-round domains require the pair lookup")
            }
            None => {}
        }
        let first = fold_leaf_pair_with_context(
            self.leaves(source, start),
            self.leaves(source, start + 1),
            first_context,
        );
        let second = fold_leaf_pair_with_context(
            self.leaves(source, start + 2),
            self.leaves(source, start + 3),
            first_context,
        );
        fold_leaf_pair_with_context(first, second, second_context)
    }

    fn folded_quartet_after_two(
        &self,
        source: usize,
        start: usize,
        challenges: &[E; 2],
    ) -> [E; LEAVES_PER_SOURCE] {
        match self.compact_lut.as_ref() {
            Some(CompactLut::Quartets(lut)) => {
                return lut[self.quartet_pattern(source, start)];
            }
            Some(CompactLut::Pairs(_)) => {
                unreachable!("round two requires the quartet lookup")
            }
            None => {}
        }
        let first = fold_leaf_pair(
            self.leaves(source, start),
            self.leaves(source, start + 1),
            challenges[0],
        );
        let second = fold_leaf_pair(
            self.leaves(source, start + 2),
            self.leaves(source, start + 3),
            challenges[0],
        );
        fold_leaf_pair(first, second, challenges[1])
    }
}

fn basis16_leaf_values<E: Field + Ring>(digit: E) -> [E; 2] {
    let y = digit * (digit + E::one());
    [
        y * (y - E::from_i64(2)) * (y - E::from_i64(6)) * (y - E::from_i64(12)),
        (y - E::from_i64(20))
            * (y - E::from_i64(30))
            * (y - E::from_i64(42))
            * (y - E::from_i64(56)),
    ]
}

fn split_equality_weight<E: Field>(first: &[E], second: &[E], index: usize) -> E {
    let first_bits = first.len().trailing_zeros() as usize;
    first[index & (first.len() - 1)] * second[index >> first_bits]
}

fn bivariate_linear_evals<E: Field>(first: &[E], second: &[E]) -> (E, E) {
    let table = if first.len() > 1 { first } else { second };
    table
        .chunks_exact(2)
        .fold((E::zero(), E::zero()), |(zero, one), pair| {
            (zero + pair[0], one + pair[1])
        })
}

fn bivariate_suffix_weight<E: Field>(first: &[E], second: &[E], index: usize) -> E {
    if first.len() > 1 {
        let first_index = 2 * index;
        let first_bits = first.len().trailing_zeros() as usize;
        (first[first_index & (first.len() - 1)] + first[(first_index & (first.len() - 1)) + 1])
            * second[first_index >> first_bits]
    } else {
        first[0] * (second[2 * index] + second[2 * index + 1])
    }
}

fn product_coefficients(left: &[E], right: &[E], source_weights: &[E]) -> [E; STORED_COEFFICIENTS] {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), LEAVES_PER_SOURCE * source_weights.len());
    let mut output = [E::zero(); STORED_COEFFICIENTS];
    let zero = E::zero();
    for (source, &weight) in source_weights.iter().enumerate() {
        let first = LEAVES_PER_SOURCE * source;
        let a0 = left[first];
        let a1 = left[first + 1];
        let b0 = right[first];
        let b1 = right[first + 1];
        if (a0 == zero && b0 == zero) || (a1 == zero && b1 == zero) {
            continue;
        }
        let d0 = b0 - a0;
        let d1 = b1 - a1;
        output[0] += mul(weight, mul(a0, d1) + mul(d0, a1));
        output[1] += mul(mul(weight, d0), d1);
    }
    output
}

fn fold_leaf_pair<E: Field>(left: [E; 2], right: [E; 2], challenge: E) -> [E; 2] {
    [
        left[0] + challenge * (right[0] - left[0]),
        left[1] + challenge * (right[1] - left[1]),
    ]
}

fn fold_leaf_pair_with_context<E: Fold>(left: [E; 2], right: [E; 2], context: &E::Ctx) -> [E; 2] {
    [
        E::fold_one(context, left[0], right[0]),
        E::fold_one(context, left[1], right[1]),
    ]
}

fn scale_coefficients<E: Field>(mut coefficients: [E; 2], scale: E) -> [E; 2] {
    for coefficient in &mut coefficients {
        *coefficient *= scale;
    }
    coefficients
}

fn add_scaled_coefficients<E: Field>(destination: &mut [E; 2], source: [E; 2], scale: E) {
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += scale * source;
    }
}

/// The map `z -> z(z + 1)` identifies `z` with `-1-z`, so the sixteen
/// validated digits in `[-8, 7]` have only eight distinct leaf images.
const RANGE_IMAGE_PATTERNS: usize = 8;
const PAIR_PATTERNS: usize = RANGE_IMAGE_PATTERNS * RANGE_IMAGE_PATTERNS;
const QUARTET_PATTERNS: usize = PAIR_PATTERNS * PAIR_PATTERNS;
/// Pair-code bits recording whether each digit class lies in `[4, 7]`.
const RANGE_IMAGE_HALF_MASK: u8 = 0x24;
/// Bucketing pays only when the items dominate the pattern space.
const BUCKET_MIN_RATIO: usize = 4;

/// Lookup slot of a validated digit modulo `z ~ -1-z`.
fn range_image_index(digit: i8) -> usize {
    if digit >= 0 {
        digit as usize
    } else {
        (-i16::from(digit) - 1) as usize
    }
}

/// Contract coefficient vectors against `first[i] * second[j]` without
/// materializing that product per pair. Each block accumulates against the
/// first tensor factor, then applies its second-factor scalar once.
fn sum_split_coefficients<E>(
    first: &[E],
    second: &[E],
    coefficients_at: impl Fn(usize) -> [E; 2] + Send + Sync,
) -> [E; 2]
where
    E: Field + Unreduced + Send + Sync,
{
    let first_len = first.len();
    if E::SUM_IS_EXACT {
        let block_coefficients = |block: usize| {
            let mut inner = [<E as Unreduced>::Product::zero(); 2];
            let base = block * first_len;
            for (low, &weight) in first.iter().enumerate() {
                let coefficients = coefficients_at(base + low);
                for (inner, coefficient) in inner.iter_mut().zip(coefficients) {
                    *inner += weight.mul_unreduced(coefficient);
                }
            }
            let mut output = [<E as Unreduced>::Product::zero(); 2];
            for (output, inner) in output.iter_mut().zip(inner) {
                *output += second[block].mul_unreduced(E::reduce_product(inner));
            }
            output
        };
        #[cfg(feature = "parallel")]
        if first_len * second.len() >= PARALLEL_MIN_WORK && second.len() > 1 {
            return (0..second.len())
                .into_par_iter()
                .map(block_coefficients)
                .reduce(
                    || [<E as Unreduced>::Product::zero(); 2],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                        left
                    },
                )
                .map(E::reduce_product);
        }
        return (0..second.len())
            .map(block_coefficients)
            .fold([<E as Unreduced>::Product::zero(); 2], |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            })
            .map(E::reduce_product);
    }
    let block_coefficients = |block: usize| {
        let mut output = [E::zero(); 2];
        let base = block * first_len;
        for (low, &weight) in first.iter().enumerate() {
            let coefficients = coefficients_at(base + low);
            for (output, coefficient) in output.iter_mut().zip(coefficients) {
                *output += weight * coefficient;
            }
        }
        scale_coefficients(output, second[block])
    };
    #[cfg(feature = "parallel")]
    if first_len * second.len() >= PARALLEL_MIN_WORK && second.len() > 1 {
        return (0..second.len())
            .into_par_iter()
            .map(block_coefficients)
            .reduce(|| [E::zero(); 2], add_coefficients);
    }
    (0..second.len())
        .map(block_coefficients)
        .fold([E::zero(); 2], add_coefficients)
}

fn add_coefficients<E: Field>(mut left: [E; 2], right: [E; 2]) -> [E; 2] {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Prime64Offset59 as F;

    fn next_random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn random_extension(state: &mut u64) -> E {
        E::new(
            F::from_u64(next_random(state)),
            F::from_u64(next_random(state)),
        )
    }

    fn reference_product_coefficients(
        left: &[E],
        right: &[E],
        source_weights: &[E],
    ) -> [E; STORED_COEFFICIENTS] {
        let mut output = [E::zero(); STORED_COEFFICIENTS];
        for (source, &weight) in source_weights.iter().enumerate() {
            let first = LEAVES_PER_SOURCE * source;
            let a0 = left[first];
            let a1 = left[first + 1];
            let d0 = right[first] - a0;
            let d1 = right[first + 1] - a1;
            output[0] += weight * (a0 * d1 + d0 * a1);
            output[1] += weight * d0 * d1;
        }
        output
    }

    fn reference_compact_octet(
        group: &ProductGroup<'_>,
        octet: usize,
        challenges: &[E; 2],
    ) -> [E; STORED_COEFFICIENTS] {
        let fold_quartet = |source: usize, start: usize| {
            let first = fold_leaf_pair(
                group.leaves(source, start),
                group.leaves(source, start + 1),
                challenges[0],
            );
            let second = fold_leaf_pair(
                group.leaves(source, start + 2),
                group.leaves(source, start + 3),
                challenges[0],
            );
            fold_leaf_pair(first, second, challenges[1])
        };

        let start = 8 * octet;
        let mut output = [E::zero(); STORED_COEFFICIENTS];
        for (source, &weight) in group.source_weights.iter().enumerate() {
            let left = fold_quartet(source, start);
            let right = fold_quartet(source, start + 4);
            let delta0 = right[0] - left[0];
            let delta1 = right[1] - left[1];
            output[0] += weight * (left[0] * delta1 + delta0 * left[1]);
            output[1] += weight * delta0 * delta1;
        }
        output
    }

    #[test]
    fn product_coefficients_skip_only_a_shared_zero_lane() {
        let zero = E::zero();
        let mut state = 0x7a42_dac9_61e5_083b_u64;
        let a = random_extension(&mut state);
        let b = random_extension(&mut state);
        let c = random_extension(&mut state);
        let d = random_extension(&mut state);
        let left = [zero, a, b, zero, zero, a, b, c];
        let right = [zero, b, d, zero, c, zero, a, d];
        let weights: Vec<E> = (0..4).map(|_| random_extension(&mut state)).collect();

        let actual = product_coefficients(&left, &right, &weights);
        let expected = reference_product_coefficients(&left, &right, &weights);
        assert_eq!(actual, expected);

        // Source two has a zero at opposite endpoints (left low, right high),
        // so both polynomial coefficients can still be nonzero.
        let mixed_zero = product_coefficients(&left[4..6], &right[4..6], &[E::one()]);
        assert_ne!(mixed_zero[0], zero);
        assert_ne!(mixed_zero[1], zero);
    }

    #[test]
    fn compact_octet_zero_lanes_match_direct_folds_for_mixed_ranges() {
        let digits = [
            [0, 1, 2, 3, 0, 1, 2, 3],
            [4, 5, 6, 7, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 4, 1, 5, 2, 6, 3, 7],
        ];
        let coded = digits
            .iter()
            .map(|source| CodedSource::new(source, -8, 7).unwrap())
            .collect::<Vec<_>>();
        let mut state = 0xd3c8_0a51_b749_2e61_u64;
        let weights: Vec<E> = (0..digits.len())
            .map(|_| random_extension(&mut state))
            .collect();
        let group = ProductGroup::new(coded.iter().collect(), weights, 3).unwrap();

        assert_eq!(group.octet_zero_lane(0, 0), Some(0));
        assert_eq!(group.octet_zero_lane(1, 0), Some(1));
        assert_eq!(group.octet_zero_lane(2, 0), None);
        assert_eq!(group.octet_zero_lane(3, 0), None);

        let challenges = [
            [E::zero(), E::zero()],
            [E::one(), E::one()],
            [random_extension(&mut state), random_extension(&mut state)],
        ];
        for challenge_pair in challenges {
            assert_eq!(
                group.compact_octet_coefficients(0, &challenge_pair),
                reference_compact_octet(&group, 0, &challenge_pair)
            );
        }
    }

    #[test]
    fn three_round_materialization_matches_two_round_materialization_then_fold() {
        let digits = [[-8, -7, -1, 0, 1, 2, 6, 7], [7, 6, 2, 1, 0, -1, -7, -8]];
        let mut state = 0xa187_f09c_7d35_42be_u64;
        let weights = vec![random_extension(&mut state), random_extension(&mut state)];
        let coded = digits
            .iter()
            .map(|digits| CodedSource::new(digits, -8, 7).unwrap())
            .collect::<Vec<_>>();
        let group = ProductGroup::new(coded.iter().collect(), weights, 3).unwrap();
        let first = random_extension(&mut state);
        let second = random_extension(&mut state);
        let third = random_extension(&mut state);

        let mut after_two = group.materialize_after_two_rounds(first, second);
        fold_rows_in_place(&mut after_two, group.lane_count(), third, "product-test");
        assert_eq!(
            group.materialize_after_three_rounds(&[first, second], third),
            after_two
        );
    }

    #[test]
    fn coded_product_group_is_reusable() {
        let digits = [
            [-8, -7, -1, 0, 1, 2, 6, 7, -4, 3, -2, 5, -6, 4, -3, -5],
            [7, 6, 2, 1, 0, -1, -7, -8, 3, -4, 5, -2, 4, -6, -5, -3],
        ];
        let expected_codes: Vec<Vec<u8>> = digits
            .iter()
            .map(|source| {
                source
                    .chunks_exact(2)
                    .map(|pair| {
                        (range_image_index(pair[0]) * RANGE_IMAGE_PATTERNS
                            + range_image_index(pair[1])) as u8
                    })
                    .collect()
            })
            .collect();
        let coded = digits
            .iter()
            .map(|source| CodedSource::new(source, -8, 7).unwrap())
            .collect::<Vec<_>>();
        for (source, expected) in coded.iter().zip(expected_codes) {
            assert_eq!(source.image8(), expected);
        }
        let mut state = 0x219c_4e76_88b5_f0d3_u64;
        let weights = vec![random_extension(&mut state), random_extension(&mut state)];
        let mut indexed = ProductGroup::new(coded.iter().collect(), weights.clone(), 4).unwrap();
        let mut live = ProductGroup::new(coded.iter().collect(), weights, 4).unwrap();

        for round in 0..4 {
            let pairs = (digits[0].len() >> round) / 2;
            let equality: Vec<E> = (0..pairs).map(|_| random_extension(&mut state)).collect();
            assert_eq!(
                indexed.round_coefficients(&equality, &[E::one()]),
                live.round_coefficients(&equality, &[E::one()])
            );
            let challenge = random_extension(&mut state);
            indexed.bind(challenge);
            live.bind(challenge);
        }
        assert_eq!(
            indexed.terminal_leaves().unwrap(),
            live.terminal_leaves().unwrap()
        );
    }

    #[test]
    fn product_coefficients_match_weighted_affine_products() {
        let mut state = 0x382f_671b_528a_cfea_u64;
        let mut random = || {
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                F::from_u64(state)
            };
            E::new(next(), next())
        };
        for sources in 1..=3 {
            for _ in 0..256 {
                let left: Vec<E> = (0..2 * sources).map(|_| random()).collect();
                let right: Vec<E> = (0..2 * sources).map(|_| random()).collect();
                let weights: Vec<E> = (0..sources).map(|_| random()).collect();
                let coefficients = product_coefficients(&left, &right, &weights);
                for challenge in [E::zero(), E::one(), random()] {
                    let mut expected = E::zero();
                    for (source, &weight) in weights.iter().enumerate() {
                        let index = 2 * source;
                        let a = left[index] + challenge * (right[index] - left[index]);
                        let b = left[index + 1] + challenge * (right[index + 1] - left[index + 1]);
                        expected += weight * (a * b - left[index] * left[index + 1]);
                    }
                    assert_eq!(
                        expected,
                        challenge * (coefficients[0] + challenge * coefficients[1])
                    );
                }
            }
        }
    }
}
