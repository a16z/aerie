//! Compact executor for the range quartic leaf.
//!
//! This module owns no transcript or proof types. It reads compact digit
//! sources, computes virtual range images, and retains only folded range-image
//! lanes after the first three active rounds.

use std::sync::Arc;

use akita_error::AkitaError;
use jolt_field::{ExtField as _, Field, Ring, Zero};
use jolt_field::{Fold, Unreduced};

#[cfg(feature = "parallel")]
use super::kernel_common::PARALLEL_MIN_WORK;
use super::kernel_common::{fold_rows_in_place, record_phase};
use super::opening_field_ops::mul;
use super::quartic_relation::{
    aggregate_weights, weighted_affine_coefficients, weighted_affine_coefficients_shifted,
    PreparedWeights, KIND_COUNT,
};
use crate::protocol::source_tables::CodedSource;
use crate::substrate::{BaseField as F, OpeningField as E};

pub(crate) use super::quartic_relation::{QuarticPolynomial, WeightedQuartic};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const STORED_COEFFICIENTS: usize = 4;

pub(crate) struct QuarticSource<'a> {
    source: CodedSource<'a>,
    pub(crate) terms: Vec<WeightedQuartic>,
}

impl<'a> QuarticSource<'a> {
    pub(crate) fn new(source: &CodedSource<'a>, terms: Vec<WeightedQuartic>) -> Self {
        Self {
            source: source.clone(),
            terms,
        }
    }

    #[cfg(test)]
    fn from_digits(
        digits: &'a [i8],
        minimum: i8,
        maximum: i8,
        terms: Vec<WeightedQuartic>,
    ) -> Self {
        let source = CodedSource::new(digits, minimum, maximum).unwrap();
        Self::new(&source, terms)
    }
}

struct PreparedQuarticSource<'a> {
    source: CodedSource<'a>,
    weights: PreparedWeights,
}

pub(crate) struct QuarticGroup<'a> {
    sources: Vec<PreparedQuarticSource<'a>>,
    pattern_base: usize,
    pair_indices: Vec<Arc<[u8]>>,
    domain_len: usize,
    num_rounds: usize,
    rounds_completed: usize,
    compact_challenges: [Option<E>; 2],
    /// Folded quartet values per range-image pattern, built once both
    /// compact challenges are known (the fold is the same for every
    /// quartet with the same images).
    quartet_lut: Option<Vec<E>>,
    quartet_taylor: Vec<Vec<[E; STORED_COEFFICIENTS]>>,
    /// Holds `Y - shift` throughout dense rounds and the restored `Y` values
    /// after the terminal bind.
    folded_range_images: Option<Vec<E>>,
    bivariate_suffix_masses: Option<Vec<[E; KIND_COUNT]>>,
    cached_second_round: Option<[E; STORED_COEFFICIENTS]>,
}

impl<'a> QuarticGroup<'a> {
    #[tracing::instrument(
        skip_all,
        name = "QuarticGroup::new",
        fields(source_count = sources.len(), num_rounds)
    )]
    pub(crate) fn new(
        sources: Vec<QuarticSource<'a>>,
        num_rounds: usize,
    ) -> Result<Self, AkitaError> {
        if sources.is_empty() || sources.iter().any(|source| source.terms.is_empty()) {
            return Err(AkitaError::InvalidInput(
                "range leaf groups need nonempty sources and quartic terms".to_owned(),
            ));
        }
        let sources = sources
            .into_iter()
            .map(|source| PreparedQuarticSource {
                source: source.source,
                weights: aggregate_weights(&source.terms),
            })
            .collect::<Vec<_>>();
        let shift = u32::try_from(num_rounds)
            .map_err(|_| AkitaError::InvalidInput("range leaf domain overflows usize".into()))?;
        let domain_len = 1_usize
            .checked_shl(shift)
            .ok_or_else(|| AkitaError::InvalidInput("range leaf domain overflows usize".into()))?;
        let pattern_base = if sources
            .iter()
            .all(|source| source.source.minimum() >= -4 && source.source.maximum() <= 3)
        {
            4
        } else {
            IMAGE_PATTERNS
        };
        // Narrow groups re-encode the image8 codes once, so the round kernels
        // read a pair pattern with a single byte load.
        let pair_indices = sources
            .iter()
            .map(|source| {
                let image8 = source.source.image8_codes();
                if pattern_base == IMAGE_PATTERNS {
                    image8
                } else {
                    let base = pattern_base as u8;
                    image8
                        .iter()
                        .map(|&code| (code / 8) * base + code % 8)
                        .collect()
                }
            })
            .collect::<Vec<_>>();
        // CodedSource construction already validates every digit and is the
        // sole producer of image8 codes. Only the local-domain shape remains
        // to check here; rescanning the full coefficient domain is redundant.
        if sources.iter().zip(&pair_indices).any(|(source, codes)| {
            source.source.cell_count() != domain_len || codes.len() != domain_len / 2
        }) {
            return Err(AkitaError::InvalidInput(
                "range leaf pair codes do not match the source domain".to_owned(),
            ));
        }
        if domain_len == 1 {
            for source in &sources {
                validate_compact_source(source, domain_len)?;
            }
        }
        let mut group = Self {
            sources,
            pattern_base,
            pair_indices,
            domain_len,
            num_rounds,
            rounds_completed: 0,
            compact_challenges: [None; 2],
            quartet_lut: None,
            quartet_taylor: Vec::new(),
            folded_range_images: None,
            bivariate_suffix_masses: None,
            cached_second_round: None,
        };
        if num_rounds == 0 {
            let mut terminal = zeroed_vec(group.sources.len())?;
            for (source, value) in terminal.iter_mut().enumerate() {
                *value = group.range_image(source, 0);
            }
            group.folded_range_images = Some(terminal);
        }
        Ok(group)
    }

    #[tracing::instrument(
        skip_all,
        name = "quartic_group_round",
        fields(
            round = self.rounds_completed,
            phase = tracing::field::Empty,
            source_count = self.sources.len(),
            domain_len = self.domain_len,
            num_rounds = self.num_rounds,
            pattern_base = self.pattern_base,
            remaining_pairs = (self.domain_len >> self.rounds_completed) / 2
        )
    )]
    pub(crate) fn round_coefficients(
        &mut self,
        equality_first: &[E],
        equality_second: &[E],
    ) -> Result<[E; STORED_COEFFICIENTS], AkitaError> {
        if self.rounds_completed >= self.num_rounds {
            return Err(AkitaError::InvalidInput(
                "range leaf executor has no remaining round".to_owned(),
            ));
        }
        let pair_count = (self.domain_len >> self.rounds_completed) / 2;
        if equality_first.len().checked_mul(equality_second.len()) != Some(pair_count) {
            return Err(AkitaError::InvalidInput(
                "range leaf equality tables do not match the remaining domain".to_owned(),
            ));
        }
        let coefficients = match self.rounds_completed {
            0 if self.bivariate_prefix_applies(pair_count) => {
                record_phase("compact-bivariate-prefix");
                self.bivariate_first_round(equality_first, equality_second, pair_count)
            }
            0 if self.bucketing_applies(pair_count, self.pattern_base().pow(2)) => {
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
                    .expect("the bivariate leaf prefix cached round two")
            }
            1 => {
                let first = self.compact_challenges[0].ok_or_else(|| {
                    AkitaError::InvalidInput("range leaf compact state is incomplete".to_owned())
                })?;
                if self.bucketing_applies(pair_count, self.pattern_base().pow(4)) {
                    record_phase("compact-quartets-bucketed");
                    self.bucketed_quartet_coefficients(
                        equality_first,
                        equality_second,
                        pair_count,
                        first,
                    )
                } else {
                    record_phase("compact-quartets-direct");
                    self.direct_quartet_coefficients(
                        equality_first,
                        equality_second,
                        pair_count,
                        first,
                    )
                }
            }
            2 if self.folded_range_images.is_some() => {
                record_phase("dense");
                let table = self.folded_range_images.as_ref().ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "range leaf folded range-image state is missing".to_owned(),
                    )
                })?;
                let lanes = self.sources.len();
                sum_split_coefficients(equality_first, equality_second, |pair| {
                    self.source_pair_coefficients(
                        &table[(2 * pair) * lanes..(2 * pair + 1) * lanes],
                        &table[(2 * pair + 1) * lanes..(2 * pair + 2) * lanes],
                    )
                })
            }
            2 => {
                record_phase("compact-octets");
                let first = self.compact_challenges[0].ok_or_else(|| {
                    AkitaError::InvalidInput("range leaf compact state is incomplete".to_owned())
                })?;
                let second = self.compact_challenges[1].ok_or_else(|| {
                    AkitaError::InvalidInput("range leaf compact state is incomplete".to_owned())
                })?;
                let bucket_narrow = self.pattern_base == 8
                    && self.quartet_lut.is_some()
                    && pair_count >= BUCKET_MIN_RATIO * NARROW_OCTET_PATTERNS;
                let mut output = sum_split_coefficients(equality_first, equality_second, |pair| {
                    self.compact_octet_coefficients(pair, first, second, bucket_narrow)
                });
                if bucket_narrow {
                    for source in 0..self.sources.len() {
                        if self.is_narrow_source(source) {
                            let bucketed = self.narrow_octet_coefficients(
                                source,
                                equality_first,
                                equality_second,
                            );
                            for (sum, coefficient) in output.iter_mut().zip(bucketed) {
                                *sum += coefficient;
                            }
                        }
                    }
                }
                output
            }
            _ => {
                record_phase("dense");
                let table = self.folded_range_images.as_ref().ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "range leaf folded range-image state is missing".to_owned(),
                    )
                })?;
                let lanes = self.sources.len();
                debug_assert_eq!(table.len(), 2 * pair_count * lanes);
                sum_split_coefficients(equality_first, equality_second, |pair| {
                    self.source_pair_coefficients(
                        &table[(2 * pair) * lanes..(2 * pair + 1) * lanes],
                        &table[(2 * pair + 1) * lanes..(2 * pair + 2) * lanes],
                    )
                })
            }
        };
        Ok(coefficients)
    }

    #[tracing::instrument(
        skip_all,
        name = "quartic_group_bind",
        fields(
            round = self.rounds_completed,
            phase = tracing::field::Empty,
            source_count = self.sources.len(),
            domain_len = self.domain_len,
            num_rounds = self.num_rounds,
            pattern_base = self.pattern_base,
            remaining_rows = self.domain_len >> self.rounds_completed
        )
    )]
    pub(crate) fn bind(&mut self, challenge: E) -> Result<(), AkitaError> {
        if self.rounds_completed >= self.num_rounds {
            return Err(AkitaError::InvalidInput(
                "range leaf executor has no remaining round".to_owned(),
            ));
        }
        match self.rounds_completed {
            0 if self.num_rounds == 1 => {
                record_phase("materialize-after-first");
                self.folded_range_images = Some(self.materialize_after_one(challenge)?);
                self.pair_indices.clear();
            }
            0 => {
                record_phase("retain-compact-pairs");
                self.compact_challenges[0] = Some(challenge);
                let masses = self.bivariate_suffix_masses.take();
                if let Some(masses) = masses {
                    self.cached_second_round =
                        Some(self.bivariate_second_round(&masses, challenge));
                }
            }
            1 if self.num_rounds == 2 => {
                record_phase("materialize-after-second");
                let first = self.compact_challenges[0].ok_or_else(|| {
                    AkitaError::InvalidInput("range leaf compact state is incomplete".to_owned())
                })?;
                self.folded_range_images = Some(self.materialize_after_two(first, challenge)?);
                self.pair_indices.clear();
                self.cached_second_round = None;
            }
            1 => {
                record_phase("retain-compact-quartets");
                self.compact_challenges[1] = Some(challenge);
                self.quartet_lut = self.build_quartet_lut(challenge);
                if let Some(lut) = &self.quartet_lut {
                    self.quartet_taylor = self
                        .sources
                        .iter()
                        .map(|source| {
                            lut.iter()
                                .map(|&left| source.weights.taylor(left))
                                .collect()
                        })
                        .collect();
                }
            }
            2 if self.folded_range_images.is_some() => {
                record_phase("fold-dense");
                let table = self.folded_range_images.as_mut().ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "range leaf folded range-image state is missing".to_owned(),
                    )
                })?;
                fold_rows_in_place(table, self.sources.len(), challenge, "quartic");
            }
            2 => {
                record_phase("materialize-after-third");
                let first = self.compact_challenges[0].ok_or_else(|| {
                    AkitaError::InvalidInput("range leaf compact state is incomplete".to_owned())
                })?;
                let second = self.compact_challenges[1].ok_or_else(|| {
                    AkitaError::InvalidInput("range leaf compact state is incomplete".to_owned())
                })?;
                self.folded_range_images =
                    Some(self.materialize_after_three(first, second, challenge)?);
                self.pair_indices.clear();
                self.quartet_lut = None;
                self.quartet_taylor.clear();
            }
            _ => {
                record_phase("fold-dense");
                let table = self.folded_range_images.as_mut().ok_or_else(|| {
                    AkitaError::InvalidInput(
                        "range leaf folded range-image state is missing".to_owned(),
                    )
                })?;
                fold_rows_in_place(table, self.sources.len(), challenge, "quartic");
            }
        }
        if self.rounds_completed + 1 == self.num_rounds {
            self.restore_terminal_images();
        }
        self.rounds_completed += 1;
        Ok(())
    }

    pub(crate) fn terminal_images(&self) -> Result<&[E], AkitaError> {
        if self.rounds_completed != self.num_rounds {
            return Err(AkitaError::InvalidInput(
                "range leaf executor has not completed every local round".to_owned(),
            ));
        }
        let values = self.folded_range_images.as_deref().ok_or_else(|| {
            AkitaError::InvalidInput("range leaf executor has no terminal images".into())
        })?;
        if values.len() != self.sources.len() {
            return Err(AkitaError::InvalidSize {
                expected: self.sources.len(),
                actual: values.len(),
            });
        }
        Ok(values)
    }

    #[must_use]
    fn direct_pair_coefficients(
        &self,
        equality_first: &[E],
        equality_second: &[E],
        _pair_count: usize,
    ) -> [E; STORED_COEFFICIENTS] {
        sum_split_coefficients(equality_first, equality_second, |pair| {
            self.compact_pair_coefficients(pair)
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
        sum_split_coefficients(equality_first, equality_second, |pair| {
            self.compact_quartet_coefficients(pair, first_challenge)
        })
    }

    /// Whether every source digit maps into the eight-value range-image
    /// alphabet and the item count dwarfs the pattern space.
    fn bucketing_applies(&self, items: usize, patterns: usize) -> bool {
        items >= BUCKET_MIN_RATIO * patterns
            && self
                .sources
                .iter()
                .all(|source| source.source.minimum() >= -8 && source.source.maximum() <= 7)
    }

    /// Radix-8-only groups use four range images; mixed/basis-16 groups use eight.
    fn pattern_base(&self) -> usize {
        self.pattern_base
    }

    fn bivariate_prefix_applies(&self, pair_count: usize) -> bool {
        self.num_rounds >= 2
            && (cfg!(test) || pair_count / 2 >= BUCKET_MIN_RATIO * self.pattern_base.pow(4))
    }

    /// Cache sufficient statistics for the first two active variables. This
    /// is the bivariate prefix represented by class masses rather than a
    /// literal evaluation grid.
    fn bivariate_first_round(
        &mut self,
        equality_first: &[E],
        equality_second: &[E],
        pair_count: usize,
    ) -> [E; STORED_COEFFICIENTS] {
        let patterns = self.pattern_base.pow(4);
        let quartets = pair_count / 2;
        let slots = self.sources.len() * patterns;
        let (linear_zero, linear_one) = bivariate_linear_evals(equality_first, equality_second);
        let accumulate = |masses: &mut Vec<E>, quartet: usize| {
            let weight = bivariate_suffix_weight(equality_first, equality_second, quartet);
            for source in 0..self.sources.len() {
                let pair = 2 * quartet;
                let pattern = self.pair_pattern(source, pair) * self.pattern_base.pow(2)
                    + self.pair_pattern(source, pair + 1);
                let slot = source * patterns + pattern;
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

        let output = self.bivariate_first_output(&masses, linear_zero, linear_one);
        let mut suffix_masses = vec![[E::zero(); KIND_COUNT]; patterns];
        for (pattern, suffix_mass) in suffix_masses.iter_mut().enumerate() {
            let masses_by_polynomial = self.kind_masses(&masses, patterns, pattern);
            *suffix_mass = masses_by_polynomial;
        }
        self.bivariate_suffix_masses = Some(suffix_masses);
        output
    }

    /// Contract the first-round pattern masses after summing away the half
    /// that does not affect each endpoint pair. Polynomial coefficients are
    /// small integers here, so the hot 4,096-pattern pass uses additions
    /// only. Extension-by-base products run over the two 64-pattern
    /// marginals, followed by a constant number of extension-field products
    /// per source.
    fn bivariate_first_output(
        &self,
        masses: &[E],
        linear_zero: E,
        linear_one: E,
    ) -> [E; STORED_COEFFICIENTS] {
        let pair_patterns = self.pattern_base.pow(2);
        let patterns = pair_patterns * pair_patterns;
        debug_assert_eq!(masses.len(), self.sources.len() * patterns);
        let mut half_masses = vec![E::zero(); self.sources.len() * 2 * pair_patterns];
        for source in 0..self.sources.len() {
            let source_masses = &masses[source * patterns..(source + 1) * patterns];
            let halves =
                &mut half_masses[source * 2 * pair_patterns..(source + 1) * 2 * pair_patterns];
            let (zero, one) = halves.split_at_mut(pair_patterns);
            for (pattern, &mass) in source_masses.iter().enumerate() {
                zero[pattern / pair_patterns] += mass;
                one[pattern % pair_patterns] += mass;
            }
        }

        let mut half_outputs = [[E::zero(); STORED_COEFFICIENTS]; 2];
        for (source, specification) in self.sources.iter().enumerate() {
            for (half, half_output) in half_outputs.iter_mut().enumerate() {
                let start = (2 * source + half) * pair_patterns;
                let pair_masses = &half_masses[start..start + pair_patterns];
                let mut kind_outputs = [[E::zero(); STORED_COEFFICIENTS]; KIND_COUNT];
                for (pattern, &mass) in pair_masses.iter().enumerate() {
                    let left = image_i64_from_index(pattern / self.pattern_base);
                    let right = image_i64_from_index(pattern % self.pattern_base);
                    for (kind_output, kind) in kind_outputs.iter_mut().zip(QuarticPolynomial::KINDS)
                    {
                        for (sum, coefficient) in kind_output
                            .iter_mut()
                            .zip(kind.affine_coefficients_i64(left, right - left))
                        {
                            *sum += mass.mul_base(F::from_i64(coefficient));
                        }
                    }
                }
                for (kind_output, weight) in kind_outputs
                    .into_iter()
                    .zip(specification.weights.kind_weights())
                {
                    add_scaled_coefficients(half_output, kind_output, weight);
                }
            }
        }
        let mut output = scale_coefficients(half_outputs[0], linear_zero);
        let one = scale_coefficients(half_outputs[1], linear_one);
        for (output, one) in output.iter_mut().zip(one) {
            *output += one;
        }
        output
    }

    fn bivariate_second_round(
        &self,
        suffix_masses: &[[E; KIND_COUNT]],
        first_challenge: E,
    ) -> [E; STORED_COEFFICIENTS] {
        let context = E::precompute(first_challenge);
        let mut output = [E::zero(); STORED_COEFFICIENTS];
        for (pattern, &masses) in suffix_masses.iter().enumerate() {
            let images: [E; 4] = core::array::from_fn(|slot| {
                image_from_index::<E>(
                    (pattern / self.pattern_base.pow(3 - slot as u32)) % self.pattern_base,
                )
            });
            let left = E::fold_one(&context, images[0], images[1]);
            let right = E::fold_one(&context, images[2], images[3]);
            let delta = right - left;
            let coefficients =
                weighted_affine_coefficients(PreparedWeights::new_unshifted(masses), left, delta);
            for (sum, coefficient) in output.iter_mut().zip(coefficients) {
                *sum += coefficient;
            }
        }
        output
    }

    /// Per-source equality-weight sums bucketed by the range-image pattern
    /// of each item's cells. Only field additions run per item; the
    /// polynomial arithmetic runs once per pattern in the caller. Field
    /// addition is associative and commutative, so the resulting round
    /// message equals the direct path's bit for bit.
    /// Block-factored variant of [`Self::pattern_weight_sums`]: the split
    /// equality weight is `first[i] * second[j]`, so per `j`-block the
    /// `first` values accumulate into per-pattern partials (additions
    /// only) and one flush scales them by `second[j]`. Distributivity
    /// makes the sums equal the plain accumulation bit for bit; the gate
    /// in the caller keeps the flush cost below the multiplies saved.
    #[tracing::instrument(skip_all, name = "QuarticGroup::accumulate_factored_pattern_weights")]
    fn factored_pattern_weight_sums(
        &self,
        equality_first: &[E],
        equality_second: &[E],
        cells_per_item: usize,
        pattern_base: usize,
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
                        self.pair_pattern(source, pair)
                    } else {
                        self.pair_pattern(source, pair) * pattern_base.pow(2)
                            + self.pair_pattern(source, pair + 1)
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

    #[tracing::instrument(skip_all, name = "QuarticGroup::accumulate_pattern_weights")]
    fn pattern_weight_sums(
        &self,
        equality_first: &[E],
        equality_second: &[E],
        items: usize,
        cells_per_item: usize,
        pattern_base: usize,
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
                pattern_base,
                patterns,
            );
        }
        let accumulate = |sums: &mut Vec<E>, item: usize| {
            let weight = split_equality_weight(equality_first, equality_second, item);
            let base = item * cells_per_item;
            for source in 0..sources {
                let pair = base / 2;
                let pattern = if cells_per_item == 2 {
                    self.pair_pattern(source, pair)
                } else {
                    self.pair_pattern(source, pair) * pattern_base.pow(2)
                        + self.pair_pattern(source, pair + 1)
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
        let pattern_base = self.pattern_base();
        let pair_patterns = pattern_base.pow(2);
        let sums = self.pattern_weight_sums(
            equality_first,
            equality_second,
            pair_count,
            2,
            pattern_base,
            pair_patterns,
        );
        let mut output = [E::zero(); STORED_COEFFICIENTS];
        for pattern in 0..pair_patterns {
            let masses = self.kind_masses(&sums, pair_patterns, pattern);
            let left = image_from_index::<E>(pattern / pattern_base);
            let right = image_from_index::<E>(pattern % pattern_base);
            let delta = right - left;
            add_kind_coefficients(&mut output, masses, left, delta);
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
        let pattern_base = self.pattern_base();
        let quartet_patterns = pattern_base.pow(4);
        let sums = self.pattern_weight_sums(
            equality_first,
            equality_second,
            quartet_count,
            4,
            pattern_base,
            quartet_patterns,
        );
        let mut output = [E::zero(); STORED_COEFFICIENTS];
        for pattern in 0..quartet_patterns {
            let masses = self.kind_masses(&sums, quartet_patterns, pattern);
            let images: [E; 4] = core::array::from_fn(|slot| {
                image_from_index::<E>((pattern / pattern_base.pow(3 - slot as u32)) % pattern_base)
            });
            // Matches `folded_source(_, _, 2, &[first_challenge])`.
            let left = images[0] + first_challenge * (images[1] - images[0]);
            let right = images[2] + first_challenge * (images[3] - images[2]);
            let delta = right - left;
            add_kind_coefficients(&mut output, masses, left, delta);
        }
        output
    }

    /// Per-polynomial-kind mass of one pattern: every source's bucketed
    /// equality weight times that source's aggregated kind weights.
    fn kind_masses(&self, sums: &[E], patterns: usize, pattern: usize) -> [E; KIND_COUNT] {
        let mut masses = [E::zero(); KIND_COUNT];
        for (source, specification) in self.sources.iter().enumerate() {
            let bucket = sums[source * patterns + pattern];
            for (mass, weight) in masses.iter_mut().zip(specification.weights.kind_weights()) {
                *mass += weight * bucket;
            }
        }
        masses
    }

    fn range_image(&self, source: usize, index: usize) -> E {
        let digit = self.sources[source]
            .source
            .digit(index)
            .expect("range leaf digit was validated");
        let digit = i16::from(digit);
        E::from_i64(i64::from(digit * (digit + 1)))
    }

    fn compact_pair_coefficients(&self, pair: usize) -> [E; STORED_COEFFICIENTS] {
        let mut output = [E::zero(); STORED_COEFFICIENTS];
        for source in 0..self.sources.len() {
            self.add_source_coefficients(
                &mut output,
                source,
                self.range_image(source, 2 * pair),
                self.range_image(source, 2 * pair + 1),
            );
        }
        output
    }

    fn compact_quartet_coefficients(
        &self,
        quartet: usize,
        first_challenge: E,
    ) -> [E; STORED_COEFFICIENTS] {
        let start = 4 * quartet;
        let mut output = [E::zero(); STORED_COEFFICIENTS];
        for source in 0..self.sources.len() {
            self.add_source_coefficients(
                &mut output,
                source,
                self.folded_source(source, start, 2, &[first_challenge]),
                self.folded_source(source, start + 2, 2, &[first_challenge]),
            );
        }
        output
    }

    /// Build the folded-quartet lookup when the octet count dominates the
    /// pattern space and every alphabet maps into the image index. The
    /// entries replay `folded_source(_, _, 4, &[first, second])` exactly,
    /// so lookups equal the direct folds bit for bit.
    #[tracing::instrument(skip_all, name = "QuarticGroup::build_quartet_fold_lut")]
    fn build_quartet_lut(&self, second_challenge: E) -> Option<Vec<E>> {
        let first_challenge = self.compact_challenges[0]?;
        let octets = self.domain_len / 8;
        let pattern_base = self.pattern_base();
        let quartet_patterns = pattern_base.pow(4);
        let alphabets_fit = self
            .sources
            .iter()
            .all(|source| source.source.minimum() >= -8 && source.source.maximum() <= 7);
        if octets < quartet_patterns || !alphabets_fit {
            return None;
        }
        let first_context = E::precompute(first_challenge);
        let second_context = E::precompute(second_challenge);
        let mut lut = Vec::with_capacity(quartet_patterns);
        for pattern in 0..quartet_patterns {
            let images: [E; 4] = core::array::from_fn(|slot| {
                image_from_index::<E>((pattern / pattern_base.pow(3 - slot as u32)) % pattern_base)
            });
            let left = E::fold_one(&first_context, images[0], images[1]);
            let right = E::fold_one(&first_context, images[2], images[3]);
            lut.push(E::fold_one(&second_context, left, right));
        }
        Some(lut)
    }

    /// Image pattern of four consecutive cells of one source.
    fn image_quartet_pattern(&self, source: usize, start: usize) -> usize {
        let pair = start / 2;
        self.pair_pattern(source, pair) * self.pattern_base.pow(2)
            + self.pair_pattern(source, pair + 1)
    }

    #[inline]
    fn pair_pattern(&self, source: usize, pair: usize) -> usize {
        usize::from(self.pair_indices[source][pair])
    }

    fn compact_octet_coefficients(
        &self,
        octet: usize,
        first_challenge: E,
        second_challenge: E,
        bucket_narrow: bool,
    ) -> [E; STORED_COEFFICIENTS] {
        let start = 8 * octet;
        let challenges = [first_challenge, second_challenge];
        let mut output = [E::zero(); STORED_COEFFICIENTS];
        if let Some(lut) = self.quartet_lut.as_deref() {
            for source in 0..self.sources.len() {
                if bucket_narrow && self.is_narrow_source(source) {
                    continue;
                }
                let left_index = self.image_quartet_pattern(source, start);
                let right_index = self.image_quartet_pattern(source, start + 4);
                let delta = lut[right_index] - lut[left_index];
                let delta_squared = mul(delta, delta);
                let powers = [
                    delta,
                    delta_squared,
                    mul(delta_squared, delta),
                    mul(delta_squared, delta_squared),
                ];
                for ((sum, power), derivative) in output
                    .iter_mut()
                    .zip(powers)
                    .zip(self.quartet_taylor[source][left_index])
                {
                    *sum += mul(power, derivative);
                }
            }
            return output;
        }
        for source in 0..self.sources.len() {
            self.add_source_coefficients(
                &mut output,
                source,
                self.folded_source(source, start, 4, &challenges),
                self.folded_source(source, start + 4, 4, &challenges),
            );
        }
        output
    }

    fn is_narrow_source(&self, source: usize) -> bool {
        self.sources[source].source.minimum() >= -4 && self.sources[source].source.maximum() <= 3
    }

    /// A narrow source has only 4^8 octet classes even when its group also
    /// contains wide sources. Contract those classes before doing the quartic
    /// arithmetic; the wide sources keep the quartet Taylor path.
    fn narrow_octet_coefficients(
        &self,
        source: usize,
        first: &[E],
        second: &[E],
    ) -> [E; STORED_COEFFICIENTS] {
        debug_assert_eq!(self.pattern_base, 8);
        debug_assert!(self.is_narrow_source(source));
        let pairs = &self.pair_indices[source];
        let accumulate = |masses: &mut Vec<E>, octet: usize| {
            let mut pattern = 0;
            for &pair in &pairs[4 * octet..4 * octet + 4] {
                // Both image indices are below four by source validation.
                let pair = usize::from(pair);
                pattern = 16 * pattern + 4 * (pair / 8) + pair % 8;
            }
            masses[pattern] += split_equality_weight(first, second, octet);
        };
        let octets = first.len() * second.len();
        #[cfg(feature = "parallel")]
        let masses =
            super::kernel_common::chunked_bucket_sums(octets, NARROW_OCTET_PATTERNS, accumulate);
        #[cfg(not(feature = "parallel"))]
        let masses = {
            let mut masses = vec![E::zero(); NARROW_OCTET_PATTERNS];
            for octet in 0..octets {
                accumulate(&mut masses, octet);
            }
            masses
        };
        // Map a base-four quartet index into the group's base-eight LUT.
        let quartet_indices: [usize; 256] = core::array::from_fn(|pattern| {
            let mut mapped = 0;
            for shift in [6, 4, 2, 0] {
                mapped = mapped * 8 + ((pattern >> shift) & 3);
            }
            mapped
        });
        let lut = self
            .quartet_lut
            .as_deref()
            .expect("octet bucketing requires the quartet LUT");
        let accumulate_pattern = |mut sums: [<E as Unreduced>::Product; STORED_COEFFICIENTS],
                                  (pattern, mass): (usize, E)| {
            if mass.is_zero() {
                return sums;
            }
            let left = quartet_indices[pattern / 256];
            let right = quartet_indices[pattern % 256];
            let delta = lut[right] - lut[left];
            let square = mul(delta, delta);
            let powers = [delta, square, mul(square, delta), mul(square, square)];
            for ((sum, power), derivative) in sums
                .iter_mut()
                .zip(powers)
                .zip(self.quartet_taylor[source][left])
            {
                *sum += mass.mul_unreduced(mul(power, derivative));
            }
            sums
        };
        #[cfg(feature = "parallel")]
        let sums = masses
            .into_par_iter()
            .enumerate()
            .fold(
                || [<E as Unreduced>::Product::zero(); STORED_COEFFICIENTS],
                accumulate_pattern,
            )
            .reduce(
                || [<E as Unreduced>::Product::zero(); STORED_COEFFICIENTS],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    left
                },
            );
        #[cfg(not(feature = "parallel"))]
        let sums = masses.into_iter().enumerate().fold(
            [<E as Unreduced>::Product::zero(); STORED_COEFFICIENTS],
            accumulate_pattern,
        );
        sums.map(E::reduce_product)
    }

    fn source_pair_coefficients(&self, left: &[E], right: &[E]) -> [E; STORED_COEFFICIENTS] {
        let mut output = [E::zero(); STORED_COEFFICIENTS];
        for (source, (&left, &right)) in left.iter().zip(right).enumerate() {
            let coefficients = weighted_affine_coefficients_shifted(
                self.sources[source].weights,
                left,
                right - left,
            );
            for (output, coefficient) in output.iter_mut().zip(coefficients) {
                *output += coefficient;
            }
        }
        output
    }

    fn add_source_coefficients(
        &self,
        output: &mut [E; STORED_COEFFICIENTS],
        source: usize,
        left: E,
        right: E,
    ) {
        let coefficients =
            weighted_affine_coefficients(self.sources[source].weights, left, right - left);
        for (output, coefficient) in output.iter_mut().zip(coefficients) {
            *output += coefficient;
        }
    }

    fn folded_source(&self, source: usize, start: usize, width: usize, challenges: &[E]) -> E {
        debug_assert!(matches!(width, 2 | 4 | 8));
        let mut values = [E::zero(); 8];
        for (offset, value) in values.iter_mut().take(width).enumerate() {
            *value = self.range_image(source, start + offset);
        }
        let mut active = width;
        for &challenge in challenges {
            for index in 0..active / 2 {
                values[index] =
                    values[2 * index] + challenge * (values[2 * index + 1] - values[2 * index]);
            }
            active /= 2;
        }
        debug_assert_eq!(active, 1);
        values[0]
    }

    fn folded_source_with_contexts(
        &self,
        source: usize,
        start: usize,
        width: usize,
        contexts: &[<E as Fold>::Ctx],
    ) -> E {
        let mut values = [E::zero(); 8];
        for (offset, value) in values.iter_mut().take(width).enumerate() {
            *value = self.range_image(source, start + offset);
        }
        let mut active = width;
        for context in contexts {
            for index in 0..active / 2 {
                values[index] = E::fold_one(context, values[2 * index], values[2 * index + 1]);
            }
            active /= 2;
        }
        values[0]
    }

    #[tracing::instrument(skip_all, name = "QuarticGroup::materialize_after_first_round")]
    fn materialize_after_one(&self, challenge: E) -> Result<Vec<E>, AkitaError> {
        self.materialize_folded(2, &[challenge])
    }

    #[tracing::instrument(skip_all, name = "QuarticGroup::materialize_after_second_round")]
    fn materialize_after_two(&self, first: E, second: E) -> Result<Vec<E>, AkitaError> {
        self.materialize_folded(4, &[first, second])
    }

    #[tracing::instrument(skip_all, name = "QuarticGroup::materialize_after_third_round")]
    fn materialize_after_three(&self, first: E, second: E, third: E) -> Result<Vec<E>, AkitaError> {
        if self.quartet_lut.is_none() {
            return self.materialize_folded(8, &[first, second, third]);
        }
        let rows = self.domain_len / 8;
        let lanes = self.sources.len();
        let cells = rows.checked_mul(lanes).ok_or_else(|| {
            AkitaError::InvalidInput("range leaf allocation overflows usize".into())
        })?;
        let mut table = zeroed_vec(cells)?;
        let third_context = E::precompute(third);
        let lut = self
            .quartet_lut
            .as_deref()
            .expect("the three-round lookup path checked its table");
        let fill = |row: usize, output: &mut [E]| {
            for (source, value) in output.iter_mut().enumerate() {
                let start = row * 8;
                *value = E::fold_one(
                    &third_context,
                    lut[self.image_quartet_pattern(source, start)],
                    lut[self.image_quartet_pattern(source, start + 4)],
                ) - self.sources[source].weights.shift();
            }
        };
        #[cfg(feature = "parallel")]
        if rows >= PARALLEL_MIN_WORK {
            table
                .par_chunks_mut(lanes)
                .enumerate()
                .for_each(|(row, output)| fill(row, output));
            return Ok(table);
        }
        for (row, output) in table.chunks_mut(lanes).enumerate() {
            fill(row, output);
        }
        Ok(table)
    }

    fn materialize_folded(&self, width: usize, challenges: &[E]) -> Result<Vec<E>, AkitaError> {
        let rows = self.domain_len / width;
        let lanes = self.sources.len();
        let cells = rows.checked_mul(lanes).ok_or_else(|| {
            AkitaError::InvalidInput("range leaf allocation overflows usize".into())
        })?;
        let mut table = zeroed_vec(cells)?;
        let contexts = challenges
            .iter()
            .copied()
            .map(E::precompute)
            .collect::<Vec<_>>();

        #[cfg(feature = "parallel")]
        if rows >= PARALLEL_MIN_WORK {
            table
                .par_chunks_mut(lanes)
                .enumerate()
                .for_each(|(row, output)| {
                    for (source, value) in output.iter_mut().enumerate() {
                        *value =
                            self.folded_source_with_contexts(source, row * width, width, &contexts)
                                - self.sources[source].weights.shift();
                    }
                });
        } else {
            for (row, output) in table.chunks_mut(lanes).enumerate() {
                for (source, value) in output.iter_mut().enumerate() {
                    *value =
                        self.folded_source_with_contexts(source, row * width, width, &contexts)
                            - self.sources[source].weights.shift();
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        for (row, output) in table.chunks_mut(lanes).enumerate() {
            for (source, value) in output.iter_mut().enumerate() {
                *value = self.folded_source_with_contexts(source, row * width, width, &contexts)
                    - self.sources[source].weights.shift();
            }
        }
        Ok(table)
    }

    fn restore_terminal_images(&mut self) {
        let Some(values) = self.folded_range_images.as_mut() else {
            return;
        };
        debug_assert_eq!(values.len(), self.sources.len());
        for (value, source) in values.iter_mut().zip(&self.sources) {
            *value += source.weights.shift();
        }
    }
}

fn zeroed_vec<E: Field>(len: usize) -> Result<Vec<E>, AkitaError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| {
        AkitaError::InvalidInput("range leaf range-image allocation failed".to_owned())
    })?;
    values.resize(len, E::zero());
    Ok(values)
}

/// Distinct range-image values of a digit in `[-8, 7]`: `d(d+1)` collapses
/// the sixteen digits onto eight images.
const IMAGE_PATTERNS: usize = 8;
const NARROW_OCTET_PATTERNS: usize = 4_usize.pow(8);
/// Bucketing pays only when the items dominate the pattern space.
const BUCKET_MIN_RATIO: usize = 4;

fn validate_compact_source(
    source: &PreparedQuarticSource<'_>,
    domain_len: usize,
) -> Result<(), AkitaError> {
    if source.source.cell_count() != domain_len {
        return Err(AkitaError::InvalidInput(
            "range leaf source length does not match its local domain".to_owned(),
        ));
    }
    let digit = source.source.digit(0).ok_or_else(|| {
        AkitaError::InvalidInput("range leaf compact source overflows at index 0".into())
    })?;
    if !(source.source.minimum()..=source.source.maximum()).contains(&digit) {
        return Err(AkitaError::InvalidInput(format!(
            "range leaf digit {digit} at index 0 is outside [{}, {}]",
            source.source.minimum(),
            source.source.maximum()
        )));
    }
    Ok(())
}

/// Index of `d(d+1)` in the ascending image alphabet, for `d` in `[-8, 7]`.
#[cfg(test)]
fn image_index(digit: i8) -> usize {
    if digit >= 0 {
        digit as usize
    } else {
        (-digit - 1) as usize
    }
}

/// The image value at an [`image_index`] slot: slot `i` holds `i(i + 1)`.
fn image_from_index<E: Field + Ring>(index: usize) -> E {
    E::from_i64(image_i64_from_index(index))
}

fn image_i64_from_index(index: usize) -> i64 {
    (index * (index + 1)) as i64
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

fn scale_coefficients<E: Field, const N: usize>(mut values: [E; N], scale: E) -> [E; N] {
    for value in &mut values {
        *value *= scale;
    }
    values
}

/// Add every polynomial kind's affine coefficients at `left + delta X`,
/// scaled by that kind's mass.
fn add_kind_coefficients(
    output: &mut [E; STORED_COEFFICIENTS],
    masses: [E; KIND_COUNT],
    left: E,
    delta: E,
) {
    for (kind, mass) in QuarticPolynomial::KINDS.into_iter().zip(masses) {
        add_scaled_coefficients(output, kind.affine_coefficients(left, delta), mass);
    }
}

fn add_scaled_coefficients<E: Field, const N: usize>(
    destination: &mut [E; N],
    source: [E; N],
    scale: E,
) {
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += scale * source;
    }
}

/// Contract coefficient vectors against the split equality tensor without
/// multiplying its two factors for every pair.
fn sum_split_coefficients<E, const N: usize>(
    first: &[E],
    second: &[E],
    coefficients_at: impl Fn(usize) -> [E; N] + Send + Sync,
) -> [E; N]
where
    E: Field + Unreduced + Send + Sync,
{
    let first_len = first.len();
    if E::SUM_IS_EXACT {
        let block_coefficients = |block: usize| {
            let mut inner = [E::Product::zero(); N];
            let base = block * first_len;
            for (low, &weight) in first.iter().enumerate() {
                let coefficients = coefficients_at(base + low);
                for (inner, coefficient) in inner.iter_mut().zip(coefficients) {
                    *inner += weight.mul_unreduced(coefficient);
                }
            }
            let mut output = [E::Product::zero(); N];
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
                    || [E::Product::zero(); N],
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
            .fold([E::Product::zero(); N], |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            })
            .map(E::reduce_product);
    }
    let block_coefficients = |block: usize| {
        let mut output = [E::zero(); N];
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
            .reduce(|| [E::zero(); N], add_coefficients);
    }
    (0..second.len())
        .map(block_coefficients)
        .fold([E::zero(); N], add_coefficients)
}

fn add_coefficients<E: Field, const N: usize>(mut left: [E; N], right: [E; N]) -> [E; N] {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{One, Prime64Offset59 as F};

    fn random(state: &mut u64) -> E {
        let mut next = || {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            F::from_u64(*state)
        };
        E::new(next(), next())
    }

    fn equality(point: &[E]) -> Vec<E> {
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

    #[test]
    fn compact_and_dense_rounds_match_polynomial_expansion_and_terminal_images() {
        let mut state = 0xb37e_a129_840c_f561;
        // Includes every compact exit, both alphabets, and the quartet-LUT boundary.
        for rounds in [0, 1, 2, 3, 4, 12, 15, 21] {
            for mode in 0..3 {
                if rounds == 21 && mode != 2 {
                    continue;
                }
                let digits: Vec<Vec<i8>> = (0..3)
                    .map(|source| {
                        (0..1 << rounds)
                            .map(|index| {
                                let width = if source == 0 || mode == 1 { 8 } else { 16 };
                                ((index * 7 + index / 5 + source * 3) % width) as i8
                                    - (width / 2) as i8
                            })
                            .collect()
                    })
                    .collect();
                let weights: Vec<[E; 2]> = (0..3)
                    .map(|source| {
                        let low = random(&mut state);
                        [
                            low,
                            if source == 0 {
                                E::zero()
                            } else if mode == 0 {
                                -low
                            } else {
                                random(&mut state)
                            },
                        ]
                    })
                    .collect();
                let sources = digits
                    .iter()
                    .zip(&weights)
                    .enumerate()
                    .map(|(source, (digits, weights))| {
                        QuarticSource::from_digits(
                            digits,
                            if source == 0 || mode == 1 { -4 } else { -8 },
                            if source == 0 || mode == 1 { 3 } else { 7 },
                            QuarticPolynomial::KINDS
                                .into_iter()
                                .zip(weights)
                                .map(|(polynomial, &weight)| WeightedQuartic { polynomial, weight })
                                .collect(),
                        )
                    })
                    .collect();
                let mut group = QuarticGroup::new(sources, rounds).unwrap();
                let mut images: Vec<Vec<E>> = digits
                    .iter()
                    .map(|source| {
                        source
                            .iter()
                            .map(|&digit| E::from_i64(i64::from(digit) * (i64::from(digit) + 1)))
                            .collect()
                    })
                    .collect();
                let tau: Vec<E> = (0..rounds)
                    .map(|_| match mode {
                        0 => E::zero(),
                        1 => E::one(),
                        _ => random(&mut state),
                    })
                    .collect();
                for round in 0..rounds {
                    let remaining = &tau[round + 1..];
                    let split = remaining.len() / 2;
                    let first = equality(&remaining[..split]);
                    let second = equality(&remaining[split..]);
                    let mut expected = [E::zero(); 4];
                    for (source, weights) in images.iter().zip(&weights) {
                        for (pair, values) in source.chunks_exact(2).enumerate() {
                            let eq = first[pair % first.len()] * second[pair / first.len()];
                            for (polynomial, &weight) in
                                QuarticPolynomial::KINDS.into_iter().zip(weights)
                            {
                                let coefficients = polynomial
                                    .affine_coefficients(values[0], values[1] - values[0]);
                                for (sum, coefficient) in expected.iter_mut().zip(coefficients) {
                                    *sum += eq * weight * coefficient;
                                }
                            }
                        }
                    }
                    assert_eq!(
                        group.round_coefficients(&first, &second).unwrap(),
                        expected,
                        "rounds={rounds}, round={round}, mode={mode}"
                    );
                    let challenge = match mode {
                        0 => E::zero(),
                        1 => E::one(),
                        _ => random(&mut state),
                    };
                    group.bind(challenge).unwrap();
                    for source in &mut images {
                        *source = source
                            .chunks_exact(2)
                            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                            .collect();
                    }
                }
                assert_eq!(
                    group.terminal_images().unwrap(),
                    images.iter().map(|source| source[0]).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn integer_bivariate_pass_matches_generic_all_patterns() {
        let digits = [
            [-8, -7, -1, 0, 1, 2, 6, 7, -8, -7, -1, 0, 1, 2, 6, 7],
            [7, 6, 2, 1, 0, -1, -7, -8, 7, 6, 2, 1, 0, -1, -7, -8],
            [-4, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3],
        ];
        let mut state = 0xda12_f76b_99c4_083e_u64;
        let sources = digits
            .iter()
            .enumerate()
            .map(|(source, digits)| {
                QuarticSource::from_digits(
                    digits,
                    if source == 2 { -4 } else { -8 },
                    if source == 2 { 3 } else { 7 },
                    QuarticPolynomial::KINDS
                        .into_iter()
                        .map(|polynomial| WeightedQuartic {
                            polynomial,
                            weight: random(&mut state),
                        })
                        .collect(),
                )
            })
            .collect();
        let group = QuarticGroup::new(sources, 4).unwrap();
        let patterns = group.pattern_base.pow(4);
        assert_eq!(patterns, 4096);
        let masses: Vec<E> = (0..group.sources.len() * patterns)
            .map(|_| random(&mut state))
            .collect();
        let linear_zero = random(&mut state);
        let linear_one = random(&mut state);

        let actual = group.bivariate_first_output(&masses, linear_zero, linear_one);
        let mut expected = [E::zero(); STORED_COEFFICIENTS];
        for pattern in 0..patterns {
            let kind_masses = group.kind_masses(&masses, patterns, pattern);
            let images: [E; 4] = core::array::from_fn(|slot| {
                image_from_index::<E>(
                    (pattern / group.pattern_base.pow(3 - slot as u32)) % group.pattern_base,
                )
            });
            for (kind, mass) in QuarticPolynomial::KINDS.into_iter().zip(kind_masses) {
                add_scaled_coefficients(
                    &mut expected,
                    kind.affine_coefficients(images[0], images[1] - images[0]),
                    linear_zero * mass,
                );
                add_scaled_coefficients(
                    &mut expected,
                    kind.affine_coefficients(images[2], images[3] - images[2]),
                    linear_one * mass,
                );
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn coded_quartic_group_is_reusable() {
        let digits = [
            [-4, -3, -2, -1, 0, 1, 2, 3, -4, -1, 2, 0, 3, -2, 1, -3],
            [-8, -7, -1, 0, 1, 2, 6, 7, -4, 3, -2, 5, -6, 4, -3, -5],
            [7, 6, 2, 1, 0, -1, -7, -8, 3, -4, 5, -2, 4, -6, -5, -3],
        ];
        let expected_codes: Vec<Vec<u8>> = digits
            .iter()
            .map(|source| {
                source
                    .chunks_exact(2)
                    .map(|pair| (image_index(pair[0]) * 8 + image_index(pair[1])) as u8)
                    .collect()
            })
            .collect();
        let mut state = 0x9a51_2d73_e604_b8cf_u64;
        let term_weights: Vec<[E; 2]> = (0..3)
            .map(|_| [random(&mut state), random(&mut state)])
            .collect();
        let sources = || {
            digits
                .iter()
                .enumerate()
                .map(|(source, digits)| {
                    QuarticSource::from_digits(
                        digits,
                        if source == 0 { -4 } else { -8 },
                        if source == 0 { 3 } else { 7 },
                        QuarticPolynomial::KINDS
                            .into_iter()
                            .zip(term_weights[source])
                            .map(|(polynomial, weight)| WeightedQuartic { polynomial, weight })
                            .collect(),
                    )
                })
                .collect::<Vec<_>>()
        };
        for (source, expected) in sources().iter().zip(expected_codes) {
            assert_eq!(source.source.image8(), expected);
        }
        let mut indexed = QuarticGroup::new(sources(), 4).unwrap();
        let mut live = QuarticGroup::new(sources(), 4).unwrap();

        for round in 0..4 {
            let pairs = (digits[0].len() >> round) / 2;
            let equality: Vec<E> = (0..pairs).map(|_| random(&mut state)).collect();
            assert_eq!(
                indexed.round_coefficients(&equality, &[E::one()]).unwrap(),
                live.round_coefficients(&equality, &[E::one()]).unwrap()
            );
            let challenge = random(&mut state);
            indexed.bind(challenge).unwrap();
            live.bind(challenge).unwrap();
        }
        assert_eq!(
            indexed.terminal_images().unwrap(),
            live.terminal_images().unwrap()
        );
    }

    #[test]
    fn combined_kind_second_round_matches_generic_all_patterns() {
        let digits = vec![-8; 16];
        let mut state = 0x8c30_624e_b9af_17d5_u64;
        let group = QuarticGroup::new(
            vec![QuarticSource::from_digits(
                &digits,
                -8,
                7,
                QuarticPolynomial::KINDS
                    .into_iter()
                    .map(|polynomial| WeightedQuartic {
                        polynomial,
                        weight: random(&mut state),
                    })
                    .collect(),
            )],
            4,
        )
        .unwrap();
        let patterns = group.pattern_base.pow(4);
        assert_eq!(patterns, 4096);
        let suffix_masses: Vec<[E; KIND_COUNT]> = (0..patterns)
            .map(|_| [random(&mut state), random(&mut state)])
            .collect();
        let challenge = random(&mut state);

        let actual = group.bivariate_second_round(&suffix_masses, challenge);
        let context = E::precompute(challenge);
        let mut expected = [E::zero(); STORED_COEFFICIENTS];
        for (pattern, &masses) in suffix_masses.iter().enumerate() {
            let images: [E; 4] = core::array::from_fn(|slot| {
                image_from_index::<E>(
                    (pattern / group.pattern_base.pow(3 - slot as u32)) % group.pattern_base,
                )
            });
            let left = E::fold_one(&context, images[0], images[1]);
            let right = E::fold_one(&context, images[2], images[3]);
            add_kind_coefficients(&mut expected, masses, left, right - left);
        }
        assert_eq!(actual, expected);
    }
    #[test]
    fn narrow_octet_histogram_matches_direct_expansion_at_prefix_endpoints() {
        let mut state = 0x845b_f5ac_718c_d951;
        let digits: Vec<i8> = (0..1 << 15)
            .map(|i| ((i * 7 + i / 11) % 8) as i8 - 4)
            .collect();
        let terms = [
            WeightedQuartic {
                polynomial: QuarticPolynomial::Low,
                weight: random(&mut state),
            },
            WeightedQuartic {
                polynomial: QuarticPolynomial::High,
                weight: random(&mut state),
            },
        ];
        for mode in 0..3 {
            let mut group = QuarticGroup::new(
                vec![
                    QuarticSource::from_digits(&digits, -4, 3, terms.to_vec()),
                    QuarticSource::from_digits(&digits, -8, 7, terms.to_vec()),
                ],
                15,
            )
            .unwrap();
            let mut values: Vec<E> = digits
                .iter()
                .map(|&d| E::from_i64(i64::from(d) * (i64::from(d) + 1)))
                .collect();
            for _ in 0..2 {
                let challenge = match mode {
                    0 => E::zero(),
                    1 => E::one(),
                    _ => random(&mut state),
                };
                group.bind(challenge).unwrap();
                values = values
                    .chunks_exact(2)
                    .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                    .collect();
            }
            let first: Vec<E> = (0..64).map(|_| random(&mut state)).collect();
            let second: Vec<E> = (0..64).map(|_| random(&mut state)).collect();
            let mut expected = [E::zero(); 4];
            for (pair, values) in values.chunks_exact(2).enumerate() {
                let weight = first[pair % 64] * second[pair / 64];
                for term in terms {
                    let coefficients = term
                        .polynomial
                        .affine_coefficients(values[0], values[1] - values[0]);
                    for (sum, coefficient) in expected.iter_mut().zip(coefficients) {
                        *sum += weight * term.weight * coefficient;
                    }
                }
            }
            assert_eq!(
                group.narrow_octet_coefficients(0, &first, &second),
                expected
            );
        }
    }
}
