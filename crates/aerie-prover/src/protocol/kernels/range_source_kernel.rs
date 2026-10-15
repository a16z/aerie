//! Compact executor for the Falcon range-source link.
//!
//! This module owns no transcript or proof types. It reads borrowed compact
//! source tables, evaluates the quadratic source-link relation, and retains
//! folded raw-source lanes only after the first three active rounds.

use std::sync::Arc;

use akita_error::AkitaError;
use jolt_field::{Fold, Unreduced};
use jolt_field::{One, Ring, Zero};

#[cfg(feature = "parallel")]
use super::kernel_common::chunked_bucket_sums;
#[cfg(feature = "parallel")]
use super::kernel_common::PARALLEL_MIN_WORK;
use super::kernel_common::{
    add_scaled_coefficients, record_phase, split_equality_weight, sum_split_coefficients,
    zeroed_vec, BUCKET_MIN_RATIO, QUADRATIC_COEFFICIENTS,
};
use super::range_source_lut;
use crate::protocol::source_tables::CodedSource;
use crate::substrate::OpeningField as E;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub(crate) struct QuadraticSource<'a> {
    source: CodedSource<'a>,
    pub(crate) weight: E,
}

impl<'a> QuadraticSource<'a> {
    pub(crate) fn new(source: &CodedSource<'a>, weight: E) -> Self {
        Self {
            source: source.clone(),
            weight,
        }
    }

    #[cfg(test)]
    fn from_cells(cells: &'a [i8], minimum: i8, maximum: i8, weight: E) -> Self {
        let source = CodedSource::new(cells, minimum, maximum).unwrap();
        Self::new(&source, weight)
    }
}

pub(crate) struct QuadraticGroup<'a> {
    sources: Vec<QuadraticSource<'a>>,
    minimums: Vec<E>,
    pattern_base: usize,
    pair_indices: Vec<Arc<[u8]>>,
    domain_len: usize,
    num_rounds: usize,
    rounds_completed: usize,
    compact_challenges: [Option<E>; 2],
    /// Folded pair values per pair of zero-based alphabet indices. A source's
    /// folded value is the lookup entry plus its minimum because folding is
    /// affine with coefficient sum one.
    pair_lut: Option<Vec<E>>,
    /// Folded quartet values per slot-index pattern, built once both
    /// compact challenges are known. The two-challenge fold is linear
    /// with coefficient sum one, so a source's folded value is the
    /// index-pattern entry plus the source minimum, exactly.
    quartet_lut: Option<Vec<E>>,
    /// Scratch storage reused while scaling source weights for compact rounds.
    scaled_weights: Vec<E>,
    folded_sources: Option<Vec<E>>,
}

#[derive(Clone, Copy)]
enum CompactPairPhase<'a> {
    Raw,
    Pairs {
        pair_lut: &'a [E],
    },
    Quartets {
        pair_lut: &'a [E],
        second_challenge: E,
    },
}

impl<'a> QuadraticGroup<'a> {
    #[tracing::instrument(
        skip_all,
        name = "QuadraticGroup::new",
        fields(source_count = sources.len(), num_rounds)
    )]
    pub(crate) fn new(
        sources: Vec<QuadraticSource<'a>>,
        num_rounds: usize,
    ) -> Result<Self, AkitaError> {
        if sources.is_empty() {
            return Err(AkitaError::InvalidInput(
                "source groups need at least one source".to_owned(),
            ));
        }
        let shift = u32::try_from(num_rounds)
            .map_err(|_| AkitaError::InvalidInput("source-link domain overflows usize".into()))?;
        let domain_len = 1_usize
            .checked_shl(shift)
            .ok_or_else(|| AkitaError::InvalidInput("source-link domain overflows usize".into()))?;
        let mut pattern_base = 0_usize;
        for source in &sources {
            let width = i16::from(source.source.maximum()) - i16::from(source.source.minimum()) + 1;
            pattern_base = pattern_base.max(width as usize);
        }
        // Narrow groups re-encode the width-16 raw codes once, so the round
        // kernels read a pair pattern with a single byte load.
        let pair_indices = sources
            .iter()
            .map(|source| {
                let raw16 = source.source.raw16_codes();
                if pattern_base == 16 {
                    raw16
                } else {
                    let base = pattern_base as u8;
                    raw16
                        .iter()
                        .map(|&code| (code >> 4) * base + (code & 0x0f))
                        .collect()
                }
            })
            .collect::<Vec<_>>();
        if sources
            .iter()
            .any(|source| source.source.cell_count() != domain_len)
        {
            return Err(AkitaError::InvalidInput(
                "range-source pairs do not match the local domain".to_owned(),
            ));
        }
        let minimums = sources
            .iter()
            .map(|source| E::from_i64(i64::from(source.source.minimum())))
            .collect();
        let mut group = Self {
            sources,
            minimums,
            pattern_base,
            pair_indices,
            domain_len,
            num_rounds,
            rounds_completed: 0,
            compact_challenges: [None; 2],
            pair_lut: None,
            quartet_lut: None,
            scaled_weights: Vec::new(),
            folded_sources: None,
        };
        if num_rounds == 0 {
            let mut terminal = zeroed_vec(group.sources.len(), "source allocation failed")?;
            for (source, value) in terminal.iter_mut().enumerate() {
                *value = group.raw_value(source, 0);
            }
            group.folded_sources = Some(terminal);
        }
        Ok(group)
    }

    #[must_use]
    pub(crate) const fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    /// Number of prefix and fused rounds already bound into this group.
    #[must_use]
    pub(crate) const fn rounds_completed(&self) -> usize {
        self.rounds_completed
    }

    /// The compact executor owns only its optimized first three rounds.
    #[must_use]
    pub(crate) const fn is_compact_prefix(&self) -> bool {
        self.rounds_completed < 3 && self.rounds_completed < self.num_rounds
    }

    #[tracing::instrument(
        skip_all,
        name = "range_source_round",
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
    ) -> Result<[E; QUADRATIC_COEFFICIENTS], AkitaError> {
        if self.rounds_completed >= self.num_rounds {
            return Err(AkitaError::InvalidInput(
                "source-link executor has no remaining round".to_owned(),
            ));
        }
        let pair_count = (self.domain_len >> self.rounds_completed) / 2;
        if equality_first.len().checked_mul(equality_second.len()) != Some(pair_count) {
            return Err(AkitaError::InvalidInput(
                "equality tables do not match the remaining domain".to_owned(),
            ));
        }
        if matches!(self.rounds_completed, 1 | 2) {
            self.scaled_weights.clear();
            self.scaled_weights
                .reserve(equality_first.len().saturating_mul(self.sources.len()));
            for &weight in equality_first {
                self.scaled_weights
                    .extend(self.sources.iter().map(|source| source.weight * weight));
            }
        }
        let coefficients = match self.rounds_completed {
            0 if Self::bucketing_applies(pair_count, self.pattern_base.pow(2)) => {
                record_phase("compact-pairs-bucketed");
                self.bucketed_pair_coefficients(equality_first, equality_second, pair_count)
            }
            0 => {
                record_phase("compact-pairs-direct");
                self.direct_pair_coefficients(equality_first, equality_second)
            }
            1 => {
                let pair_lut = self
                    .pair_lut
                    .as_deref()
                    .expect("the first compact bind builds the pair lookup");
                record_phase("compact-quartets-lut");
                self.weighted_coefficients(equality_first, equality_second, |quartet, source| {
                    (
                        self.folded_pair(pair_lut, source, self.pair_pattern(source, 2 * quartet)),
                        self.folded_pair(
                            pair_lut,
                            source,
                            self.pair_pattern(source, 2 * quartet + 1),
                        ),
                    )
                })
            }
            2 => {
                record_phase("compact-octets");
                let second = self.compact_challenges[1]
                    .expect("each completed compact bind stores its challenge");
                let pair_lut = self
                    .pair_lut
                    .as_deref()
                    .expect("the first compact bind builds the pair lookup");
                self.weighted_coefficients(equality_first, equality_second, |octet, source| {
                    (
                        self.folded_quartet(pair_lut, source, 8 * octet, second),
                        self.folded_quartet(pair_lut, source, 8 * octet + 4, second),
                    )
                })
            }
            3.. => {
                return Err(AkitaError::InvalidInput(
                    "dense range-source rounds belong to the fused stage-three engine".to_owned(),
                ));
            }
        };
        Ok(coefficients)
    }

    /// Compact quadratic coefficients plus one independent affine source
    /// relation. The auxiliary relation reads the compact indexed pairs
    /// without materializing dense field tables; `linear_items_per_weight`
    /// lets a row-only relation reuse each row factor across its coefficient
    /// pairs.
    pub(crate) fn round_coefficients_with_linear(
        &mut self,
        equality_first: &[E],
        equality_second: &[E],
        linear_first: &[E],
        linear_second: &[E],
        linear_factors: &[E],
        linear_offsets: &[i64],
    ) -> Result<([E; QUADRATIC_COEFFICIENTS], [E; 2]), AkitaError> {
        if !self.is_compact_prefix() {
            return Err(AkitaError::InvalidInput(
                "linear auxiliary contributions are available only in the compact prefix"
                    .to_owned(),
            ));
        }
        if linear_factors.len() != self.sources.len()
            || linear_offsets.len() != self.sources.len()
            || linear_first.is_empty()
            || linear_second.is_empty()
        {
            return Err(AkitaError::InvalidInput(
                "compact auxiliary source shape mismatch".to_owned(),
            ));
        }
        let quadratic = self.round_coefficients(equality_first, equality_second)?;
        let phase = self.compact_pair_phase();
        let pair_count = equality_first.len() * equality_second.len();
        let weight_count = linear_first.len() * linear_second.len();
        if !pair_count.is_multiple_of(weight_count) {
            return Err(AkitaError::InvalidInput(
                "compact auxiliary equality does not cover source pairs".to_owned(),
            ));
        }
        let linear_items_per_weight = pair_count / weight_count;
        let accumulate = |range: std::ops::Range<usize>| {
            let mut sums = [E::zero(); 2];
            for pair in range {
                let weight_index = pair / linear_items_per_weight;
                let weight = linear_first[weight_index % linear_first.len()]
                    * linear_second[weight_index / linear_first.len()];
                if weight.is_zero() {
                    continue;
                }
                for source in 0..self.sources.len() {
                    let (left, right) = self.compact_pair_values(phase, source, pair);
                    let factor = linear_factors[source] * weight;
                    sums[0] += factor * (left + E::from_i64(linear_offsets[source]));
                    sums[1] += factor * (right - left);
                }
            }
            sums
        };
        #[cfg(feature = "parallel")]
        let linear = if pair_count >= PARALLEL_MIN_WORK {
            (0..pair_count.div_ceil(PARALLEL_MIN_WORK))
                .into_par_iter()
                .map(|chunk| {
                    accumulate(
                        chunk * PARALLEL_MIN_WORK..pair_count.min((chunk + 1) * PARALLEL_MIN_WORK),
                    )
                })
                .reduce(
                    || [E::zero(); 2],
                    |mut left, right| {
                        for lane in 0..2 {
                            left[lane] += right[lane];
                        }
                        left
                    },
                )
        } else {
            accumulate(0..pair_count)
        };
        #[cfg(not(feature = "parallel"))]
        let linear = accumulate(0..pair_count);
        Ok((quadratic, linear))
    }

    #[tracing::instrument(
        skip_all,
        name = "range_source_bind",
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
    pub(crate) fn bind_compact(
        &mut self,
        challenge: E,
        next_round_equality: Option<(&[E], &[E])>,
    ) -> Result<Option<[E; QUADRATIC_COEFFICIENTS]>, AkitaError> {
        if self.rounds_completed >= self.num_rounds {
            return Err(AkitaError::InvalidInput(
                "source-link executor has no remaining round".to_owned(),
            ));
        }
        let coefficients = match self.rounds_completed {
            0 if self.num_rounds == 1 => {
                record_phase("materialize-after-first");
                self.folded_sources = Some(self.materialize_folded(2, &[challenge])?);
                self.pair_indices.clear();
                Ok::<_, AkitaError>(None)
            }
            0 => {
                record_phase("retain-compact-pairs");
                self.compact_challenges[0] = Some(challenge);
                self.pair_lut = Some(range_source_lut::build_pair(self.pattern_base, challenge));
                Ok(None)
            }
            1 if self.num_rounds == 2 => {
                record_phase("materialize-after-second");
                let first = self.compact_challenges[0]
                    .expect("each completed compact bind stores its challenge");
                self.folded_sources = Some(self.materialize_folded(4, &[first, challenge])?);
                self.pair_indices.clear();
                Ok(None)
            }
            1 => {
                record_phase("retain-compact-quartets");
                self.compact_challenges[1] = Some(challenge);
                self.quartet_lut = self.build_quartet_lut(challenge);
                Ok(None)
            }
            2 => {
                record_phase("materialize-after-third");
                let first = self.compact_challenges[0]
                    .expect("each completed compact bind stores its challenge");
                let second = self.compact_challenges[1]
                    .expect("each completed compact bind stores its challenge");
                let (values, coefficients) = if self.num_rounds > 3 {
                    let (equality_first, equality_second) = next_round_equality.expect(
                        "the third bind receives equality tables for the remaining dense rounds",
                    );
                    let (values, coefficients) = self.materialize_after_three_with_coefficients(
                        &[first, second, challenge],
                        equality_first,
                        equality_second,
                    )?;
                    (values, Some(coefficients))
                } else {
                    (
                        self.materialize_folded(8, &[first, second, challenge])?,
                        None,
                    )
                };
                self.folded_sources = Some(values);
                self.pair_indices.clear();
                self.pair_lut = None;
                self.quartet_lut = None;
                Ok(coefficients)
            }
            3.. => {
                return Err(AkitaError::InvalidInput(
                    "dense range-source binds belong to the fused stage-three engine".to_owned(),
                ));
            }
        }?;
        self.rounds_completed += 1;
        Ok(coefficients)
    }

    /// Consume the compact executor after its optimized prefix. The returned
    /// row-major table belongs to the stage-three fused engine for every
    /// remaining dense round.
    pub(crate) fn into_dense_sources(self) -> Result<Vec<E>, AkitaError> {
        if self.rounds_completed < self.num_rounds.min(3) {
            return Err(AkitaError::InvalidInput(
                "range-source compact prefix is incomplete".to_owned(),
            ));
        }
        let values = self
            .folded_sources
            .expect("completing the compact prefix materializes the source lanes");
        Ok(values)
    }

    /// Contract compact source pairs against a split equality tensor. The
    /// source weight and low equality factor are shared by both quadratic
    /// coefficients, leaving one reduced product and two unreduced products
    /// per source item. Each unreduced inner sum contains
    /// `first.len() * sources.len()` products: the same batch used by
    /// [`sum_split_coefficients`], multiplied by the source count. The
    /// `FpExt2<Fp64>` accumulator admits more than 2^62 products, while the
    /// materialized `Vec<E>` of scaled weights bounds a successful batch
    /// below 2^59 elements on a 64-bit target.
    fn weighted_coefficients(
        &self,
        first: &[E],
        second: &[E],
        values: impl Fn(usize, usize) -> (E, E) + Send + Sync,
    ) -> [E; QUADRATIC_COEFFICIENTS] {
        const { assert!(E::SUM_IS_EXACT) };
        type Product = <E as Unreduced>::Product;

        let source_count = self.sources.len();
        let first_len = first.len();
        let scaled = &self.scaled_weights;
        debug_assert_eq!(scaled.len(), first_len * source_count);
        let block = |high: usize| {
            let mut inner = [Product::zero(); QUADRATIC_COEFFICIENTS];
            let base = high * first_len;
            for low in 0..first_len {
                let weights = &scaled[low * source_count..(low + 1) * source_count];
                for (source, &weight) in weights.iter().enumerate() {
                    let (left, right) = values(base + low, source);
                    let delta = right - left;
                    let weighted_delta = weight * delta;
                    inner[0] += weighted_delta.mul_unreduced(left + left + E::one());
                    inner[1] += weighted_delta.mul_unreduced(delta);
                }
            }
            inner.map(|value| second[high].mul_unreduced(E::reduce_product(value)))
        };
        let add = |mut left: [Product; QUADRATIC_COEFFICIENTS],
                   right: [Product; QUADRATIC_COEFFICIENTS]| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        };
        #[cfg(feature = "parallel")]
        if first_len * second.len() >= PARALLEL_MIN_WORK && second.len() > 1 {
            return (0..second.len())
                .into_par_iter()
                .map(block)
                .reduce(|| [Product::zero(); QUADRATIC_COEFFICIENTS], add)
                .map(E::reduce_product);
        }
        (0..second.len())
            .map(block)
            .fold([Product::zero(); QUADRATIC_COEFFICIENTS], add)
            .map(E::reduce_product)
    }

    #[must_use]
    fn direct_pair_coefficients(
        &self,
        equality_first: &[E],
        equality_second: &[E],
    ) -> [E; QUADRATIC_COEFFICIENTS] {
        sum_split_coefficients(equality_first, equality_second, |pair| {
            self.compact_pair_coefficients(pair)
        })
    }

    /// Whether the item count dwarfs the validated source pattern space.
    fn bucketing_applies(items: usize, patterns: usize) -> bool {
        items >= BUCKET_MIN_RATIO * patterns
    }

    /// Per-source equality-weight sums bucketed by the cell-value pattern
    /// of each item (values are validated into per-source alphabets of at
    /// most sixteen values, indexed from the source minimum). Only field
    /// additions run per item; the quadratic arithmetic runs once per
    /// pattern in the caller. Field addition is associative and
    /// commutative, so the round message equals the direct path's bit for
    /// bit.
    #[tracing::instrument(skip_all, name = "QuadraticGroup::accumulate_pattern_weights")]
    fn pattern_weight_sums(
        &self,
        equality_first: &[E],
        equality_second: &[E],
        items: usize,
        patterns: usize,
    ) -> Vec<E> {
        let sources = self.sources.len();
        if sources * patterns <= equality_first.len() / 2 {
            return self.factored_pattern_weight_sums(equality_first, equality_second, patterns);
        }
        let accumulate = |sums: &mut Vec<E>, item: usize| {
            let weight = split_equality_weight(equality_first, equality_second, item);
            for source in 0..sources {
                let pattern = self.pair_pattern(source, item);
                sums[source * patterns + pattern] += weight;
            }
        };
        #[cfg(feature = "parallel")]
        if items >= PARALLEL_MIN_WORK {
            return chunked_bucket_sums(items, sources * patterns, accumulate);
        }
        let mut sums = vec![E::zero(); sources * patterns];
        for item in 0..items {
            accumulate(&mut sums, item);
        }
        sums
    }

    /// Accumulate the split equality tensor one second-factor block at a
    /// time. Each item contributes only its first-factor weight to a local
    /// pattern bucket; one flush per pattern then applies the shared second
    /// factor. The gate in [`Self::pattern_weight_sums`] ensures that the
    /// flush count remains below the per-item multiplications it replaces.
    #[tracing::instrument(skip_all, name = "QuadraticGroup::accumulate_factored_pattern_weights")]
    fn factored_pattern_weight_sums(
        &self,
        equality_first: &[E],
        equality_second: &[E],
        patterns: usize,
    ) -> Vec<E> {
        let sources = self.sources.len();
        let first_len = equality_first.len();
        let blocks = equality_second.len();
        let accumulate = |sums: &mut [E], inner: &mut [E], block: usize| {
            inner.fill(E::zero());
            let base = block * first_len;
            for (low, &first_weight) in equality_first.iter().enumerate() {
                for source in 0..sources {
                    let pattern = self.pair_pattern(source, base + low);
                    inner[source * patterns + pattern] += first_weight;
                }
            }
            let second_weight = equality_second[block];
            for (sum, &partial) in sums.iter_mut().zip(inner.iter()) {
                *sum += second_weight * partial;
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

    #[must_use]
    fn bucketed_pair_coefficients(
        &self,
        equality_first: &[E],
        equality_second: &[E],
        pair_count: usize,
    ) -> [E; QUADRATIC_COEFFICIENTS] {
        let pattern_base = self.pattern_base;
        let pair_patterns = pattern_base.pow(2);
        let sums =
            self.pattern_weight_sums(equality_first, equality_second, pair_count, pair_patterns);
        let mut output = [E::zero(); QUADRATIC_COEFFICIENTS];
        for pattern in 0..pair_patterns {
            let left_slot = pattern / pattern_base;
            let right_slot = pattern % pattern_base;
            let delta = E::from_i64(right_slot as i64 - left_slot as i64);
            let mut q1 = E::zero();
            let mut q2_mass = E::zero();
            for (source, specification) in self.sources.iter().enumerate() {
                let mass = specification.weight * sums[source * pair_patterns + pattern];
                let left =
                    E::from_i64(i64::from(specification.source.minimum()) + left_slot as i64);
                q1 += mass * delta * (left + left + E::one());
                q2_mass += mass;
            }
            output[0] += q1;
            output[1] += q2_mass * delta * delta;
        }
        output
    }

    fn raw_value(&self, source: usize, index: usize) -> E {
        E::from_i64(i64::from(
            self.sources[source]
                .source
                .digit(index)
                .expect("source was validated"),
        ))
    }

    fn compact_pair_coefficients(&self, pair: usize) -> [E; QUADRATIC_COEFFICIENTS] {
        let mut output = [E::zero(); QUADRATIC_COEFFICIENTS];
        for source in 0..self.sources.len() {
            self.add_source_coefficients(
                &mut output,
                source,
                self.raw_value(source, 2 * pair),
                self.raw_value(source, 2 * pair + 1),
            );
        }
        output
    }

    #[inline]
    fn folded_pair(&self, pair_lut: &[E], source: usize, code: usize) -> E {
        pair_lut[code] + self.minimums[source]
    }

    /// Build the folded-quartet lookup over slot-index patterns when the
    /// octet count dominates the pattern space and every alphabet fits the
    /// sixteen-slot base. Entries replay the fold on the index values; a
    /// source's folded value is the entry plus its minimum because the
    /// fold is linear and preserves constants, exactly.
    #[tracing::instrument(skip_all, name = "QuadraticGroup::build_quartet_fold_lut")]
    fn build_quartet_lut(&self, second_challenge: E) -> Option<Vec<E>> {
        let first_challenge =
            self.compact_challenges[0].expect("the first compact bind stores its challenge");
        let pattern_base = self.pattern_base;
        let quartet_patterns = pattern_base.pow(4);
        if self.domain_len / 8 < quartet_patterns {
            return None;
        }
        Some(range_source_lut::build_quartet(
            pattern_base,
            first_challenge,
            second_challenge,
        ))
    }

    /// Slot-index pattern of four consecutive cells of one source.
    fn index_quartet_pattern(&self, source: usize, start: usize) -> usize {
        let pair = start / 2;
        self.pair_pattern(source, pair) * self.pattern_base.pow(2)
            + self.pair_pattern(source, pair + 1)
    }

    #[inline]
    fn pair_pattern(&self, source: usize, pair: usize) -> usize {
        usize::from(self.pair_indices[source][pair])
    }

    /// A width-four fold through the quartet lookup when it exists.
    fn folded_quartet(
        &self,
        pair_lut: &[E],
        source: usize,
        start: usize,
        second_challenge: E,
    ) -> E {
        if let Some(lut) = self.quartet_lut.as_deref() {
            return lut[self.index_quartet_pattern(source, start)] + self.minimums[source];
        }
        let pair = start / 2;
        let left = self.folded_pair(pair_lut, source, self.pair_pattern(source, pair));
        let right = self.folded_pair(pair_lut, source, self.pair_pattern(source, pair + 1));
        left + second_challenge * (right - left)
    }

    /// Fold one compact octet while its values are still zero-based alphabet
    /// indices. Restoring the source minimum after all three folds avoids
    /// repeating the same affine offset for every intermediate quartet.
    #[inline]
    fn folded_octet(
        &self,
        pair_lut: &[E],
        source: usize,
        row: usize,
        second_context: &<E as Fold>::Ctx,
        third_context: &<E as Fold>::Ctx,
    ) -> E {
        let pair = 4 * row;
        let (left, right) = if let Some(lut) = self.quartet_lut.as_deref() {
            let quartet_base = self.pattern_base.pow(2);
            (
                lut[self.pair_pattern(source, pair) * quartet_base
                    + self.pair_pattern(source, pair + 1)],
                lut[self.pair_pattern(source, pair + 2) * quartet_base
                    + self.pair_pattern(source, pair + 3)],
            )
        } else {
            (
                E::fold_one(
                    second_context,
                    pair_lut[self.pair_pattern(source, pair)],
                    pair_lut[self.pair_pattern(source, pair + 1)],
                ),
                E::fold_one(
                    second_context,
                    pair_lut[self.pair_pattern(source, pair + 2)],
                    pair_lut[self.pair_pattern(source, pair + 3)],
                ),
            )
        };
        E::fold_one(third_context, left, right) + self.minimums[source]
    }

    /// Emit the post-third-bind table and contract its first dense
    /// quadratic round while each adjacent row pair is still in registers.
    #[tracing::instrument(
        skip_all,
        name = "QuadraticGroup::materialize_and_contract_after_third_round"
    )]
    fn materialize_after_three_with_coefficients(
        &self,
        challenges: &[E; 3],
        equality_first: &[E],
        equality_second: &[E],
    ) -> Result<(Vec<E>, [E; QUADRATIC_COEFFICIENTS]), AkitaError> {
        const { assert!(E::SUM_IS_EXACT) };

        let rows = self.domain_len / 8;
        let lanes = self.sources.len();
        let pairs = rows / 2;
        let expected_pairs = equality_first.len().checked_mul(equality_second.len());
        if !equality_first.len().is_power_of_two() || expected_pairs != Some(pairs) {
            return Err(AkitaError::InvalidInput(
                "next range-source equality does not match the dense domain".to_owned(),
            ));
        }
        let block_rows = equality_first.len() * 2;
        let block_cells = block_rows.checked_mul(lanes).ok_or_else(|| {
            AkitaError::InvalidInput("range source allocation overflows usize".into())
        })?;
        let cells = rows.checked_mul(lanes).ok_or_else(|| {
            AkitaError::InvalidInput("range source allocation overflows usize".into())
        })?;
        let mut table = zeroed_vec(cells, "source allocation failed")?;
        let pair_lut = self
            .pair_lut
            .as_deref()
            .expect("the first compact bind builds the pair lookup");
        let second_context = E::precompute(challenges[1]);
        let third_context = E::precompute(challenges[2]);
        let weighted_count = equality_first.len() * lanes;
        let mut weighted_first = Vec::new();
        weighted_first
            .try_reserve_exact(weighted_count)
            .map_err(|_| AkitaError::InvalidInput("source allocation failed".to_owned()))?;
        for &equality_weight in equality_first {
            weighted_first.extend(
                self.sources
                    .iter()
                    .map(|source| equality_weight * source.weight),
            );
        }
        let fill_block = |block: usize, output: &mut [E]| {
            let mut inner = [<E as Unreduced>::Product::zero(); QUADRATIC_COEFFICIENTS];
            for low in 0..equality_first.len() {
                let pair = block * equality_first.len() + low;
                let left_row = 2 * pair;
                let right_row = left_row + 1;
                for source in 0..lanes {
                    let left = self.folded_octet(
                        pair_lut,
                        source,
                        left_row,
                        &second_context,
                        &third_context,
                    );
                    let right = self.folded_octet(
                        pair_lut,
                        source,
                        right_row,
                        &second_context,
                        &third_context,
                    );
                    let row_offset = 2 * low * lanes;
                    output[row_offset + source] = left;
                    output[row_offset + lanes + source] = right;

                    let delta = right - left;
                    let weighted_delta = weighted_first[low * lanes + source] * delta;
                    inner[0] += weighted_delta.mul_unreduced(left + left + E::one());
                    inner[1] += weighted_delta.mul_unreduced(delta);
                }
            }
            [
                equality_second[block].mul_unreduced(E::reduce_product(inner[0])),
                equality_second[block].mul_unreduced(E::reduce_product(inner[1])),
            ]
        };
        let add_products =
            |mut left: [<E as Unreduced>::Product; QUADRATIC_COEFFICIENTS],
             right: [<E as Unreduced>::Product; QUADRATIC_COEFFICIENTS]| {
                for lane in 0..QUADRATIC_COEFFICIENTS {
                    left[lane] += right[lane];
                }
                left
            };
        #[cfg(feature = "parallel")]
        let coefficients = if rows >= PARALLEL_MIN_WORK && equality_second.len() > 1 {
            table
                .par_chunks_mut(block_cells)
                .enumerate()
                .map(|(block, output)| fill_block(block, output))
                .reduce(
                    || [<E as Unreduced>::Product::zero(); QUADRATIC_COEFFICIENTS],
                    add_products,
                )
        } else {
            table
                .chunks_mut(block_cells)
                .enumerate()
                .map(|(block, output)| fill_block(block, output))
                .fold(
                    [<E as Unreduced>::Product::zero(); QUADRATIC_COEFFICIENTS],
                    add_products,
                )
        };
        #[cfg(not(feature = "parallel"))]
        let coefficients = table
            .chunks_mut(block_cells)
            .enumerate()
            .map(|(block, output)| fill_block(block, output))
            .fold(
                [<E as Unreduced>::Product::zero(); QUADRATIC_COEFFICIENTS],
                add_products,
            );

        Ok((table, coefficients.map(E::reduce_product)))
    }

    fn compact_pair_phase(&self) -> CompactPairPhase<'_> {
        debug_assert!(
            self.is_compact_prefix(),
            "compact pair phases require an active prefix round"
        );
        match self.rounds_completed {
            0 => CompactPairPhase::Raw,
            1 => {
                let pair_lut = self
                    .pair_lut
                    .as_deref()
                    .expect("the first compact bind builds the pair lookup");
                CompactPairPhase::Pairs { pair_lut }
            }
            2.. => {
                let second = self.compact_challenges[1]
                    .expect("each completed compact bind stores its challenge");
                let pair_lut = self
                    .pair_lut
                    .as_deref()
                    .expect("the first compact bind builds the pair lookup");
                CompactPairPhase::Quartets {
                    pair_lut,
                    second_challenge: second,
                }
            }
        }
    }

    fn compact_pair_values(
        &self,
        phase: CompactPairPhase<'_>,
        source: usize,
        pair: usize,
    ) -> (E, E) {
        match phase {
            CompactPairPhase::Raw => (
                self.raw_value(source, 2 * pair),
                self.raw_value(source, 2 * pair + 1),
            ),
            CompactPairPhase::Pairs { pair_lut } => (
                self.folded_pair(pair_lut, source, self.pair_pattern(source, 2 * pair)),
                self.folded_pair(pair_lut, source, self.pair_pattern(source, 2 * pair + 1)),
            ),
            CompactPairPhase::Quartets {
                pair_lut,
                second_challenge,
            } => (
                self.folded_quartet(pair_lut, source, 8 * pair, second_challenge),
                self.folded_quartet(pair_lut, source, 8 * pair + 4, second_challenge),
            ),
        }
    }

    fn add_source_coefficients(
        &self,
        output: &mut [E; QUADRATIC_COEFFICIENTS],
        source: usize,
        left: E,
        right: E,
    ) {
        let source = &self.sources[source];
        let delta = right - left;
        let coefficients = [delta * (left + left + E::one()), delta * delta];
        add_scaled_coefficients(output, coefficients, source.weight);
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
            *value = self.raw_value(source, start + offset);
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

    #[tracing::instrument(
        skip_all,
        name = "QuadraticGroup::materialize_compact_sources",
        fields(width)
    )]
    fn materialize_folded(&self, width: usize, challenges: &[E]) -> Result<Vec<E>, AkitaError> {
        let rows = self.domain_len / width;
        let lanes = self.sources.len();
        let cells = rows.checked_mul(lanes).ok_or_else(|| {
            AkitaError::InvalidInput("range source allocation overflows usize".into())
        })?;
        let mut table = zeroed_vec(cells, "source allocation failed")?;
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
                            self.folded_source_with_contexts(source, row * width, width, &contexts);
                    }
                });
        } else {
            for (row, output) in table.chunks_mut(lanes).enumerate() {
                for (source, value) in output.iter_mut().enumerate() {
                    *value =
                        self.folded_source_with_contexts(source, row * width, width, &contexts);
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        for (row, output) in table.chunks_mut(lanes).enumerate() {
            for (source, value) in output.iter_mut().enumerate() {
                *value = self.folded_source_with_contexts(source, row * width, width, &contexts);
            }
        }
        Ok(table)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::BaseField as F;

    fn extension(seed: u64) -> E {
        E::new(F::from_u64(seed * 17 + 3), F::from_u64(seed * 29 + 5))
    }

    fn digit_tables(pattern_base: usize, domain_len: usize) -> Vec<Vec<i8>> {
        (0..3)
            .map(|source| {
                let mut state = 0x9e37_79b9_u64 ^ (source as u64 * 0x1000_0001);
                (0..domain_len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        let minimum = -(pattern_base as i8 / 2);
                        minimum + (state as usize % pattern_base) as i8
                    })
                    .collect()
            })
            .collect()
    }

    fn sources(tables: &[Vec<i8>], pattern_base: usize) -> Vec<QuadraticSource<'_>> {
        let minimum = -(pattern_base as i8 / 2);
        tables
            .iter()
            .enumerate()
            .map(|(index, cells)| {
                QuadraticSource::from_cells(
                    cells,
                    minimum,
                    minimum + pattern_base as i8 - 1,
                    extension(index as u64 + 11),
                )
            })
            .collect()
    }

    fn split_weights(pair_count: usize) -> (Vec<E>, Vec<E>) {
        let first_len = 1 << (pair_count.trailing_zeros() / 2);
        let first = (0..first_len)
            .map(|index| extension(index as u64 + 101))
            .collect();
        let second = (0..pair_count / first_len)
            .map(|index| extension(index as u64 + 701))
            .collect();
        (first, second)
    }

    fn folded_source(
        group: &QuadraticGroup<'_>,
        source: usize,
        start: usize,
        width: usize,
        challenges: &[E],
    ) -> E {
        debug_assert!(matches!(width, 2 | 4 | 8));
        let mut values = [E::zero(); 8];
        for (offset, value) in values.iter_mut().take(width).enumerate() {
            *value = group.raw_value(source, start + offset);
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

    fn compact_quartet_coefficients(
        group: &QuadraticGroup<'_>,
        quartet: usize,
        first: E,
    ) -> [E; QUADRATIC_COEFFICIENTS] {
        let start = 4 * quartet;
        let mut output = [E::zero(); QUADRATIC_COEFFICIENTS];
        for source in 0..group.sources.len() {
            group.add_source_coefficients(
                &mut output,
                source,
                folded_source(group, source, start, 2, &[first]),
                folded_source(group, source, start + 2, 2, &[first]),
            );
        }
        output
    }

    fn direct_quartet_coefficients(
        group: &QuadraticGroup<'_>,
        equality_first: &[E],
        equality_second: &[E],
        first_challenge: E,
    ) -> [E; QUADRATIC_COEFFICIENTS] {
        sum_split_coefficients(equality_first, equality_second, |quartet| {
            compact_quartet_coefficients(group, quartet, first_challenge)
        })
    }

    fn compact_octet_coefficients(
        group: &QuadraticGroup<'_>,
        octet: usize,
        first: E,
        second: E,
    ) -> [E; QUADRATIC_COEFFICIENTS] {
        let start = 8 * octet;
        let challenges = [first, second];
        let mut output = [E::zero(); QUADRATIC_COEFFICIENTS];
        for source in 0..group.sources.len() {
            group.add_source_coefficients(
                &mut output,
                source,
                folded_source(group, source, start, 4, &challenges),
                folded_source(group, source, start + 4, 4, &challenges),
            );
        }
        output
    }

    #[test]
    fn coded_raw_pairs_are_reusable_through_compact_prefix() {
        let domain_len = 1 << 10;
        let minimums = [-4_i8, -8, -8];
        let widths = [8_usize, 16, 16];
        let tables = minimums
            .into_iter()
            .zip(widths)
            .enumerate()
            .map(|(source, (minimum, width))| {
                let mut state = 0x6a09_e667_f3bc_c909_u64 ^ source as u64;
                (0..domain_len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        minimum + (state as usize % width) as i8
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let pair_indices = tables
            .iter()
            .zip(minimums)
            .map(|(cells, minimum)| {
                cells
                    .chunks_exact(2)
                    .map(|pair| {
                        let left = i16::from(pair[0]) - i16::from(minimum);
                        let right = i16::from(pair[1]) - i16::from(minimum);
                        (left as u8) * 16 + right as u8
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let make_sources = || {
            tables
                .iter()
                .zip(minimums)
                .zip(widths)
                .enumerate()
                .map(|(source, ((cells, minimum), width))| {
                    QuadraticSource::from_cells(
                        cells,
                        minimum,
                        minimum + width as i8 - 1,
                        extension(source as u64 + 41),
                    )
                })
                .collect::<Vec<_>>()
        };
        for (source, expected) in make_sources().iter().zip(pair_indices) {
            assert_eq!(source.source.raw16(), expected);
        }
        let mut validated = QuadraticGroup::new(make_sources(), 10).unwrap();
        let mut preindexed = QuadraticGroup::new(make_sources(), 10).unwrap();

        for round in 0..3 {
            let pair_count = (domain_len >> round) / 2;
            let (equality_first, equality_second) = split_weights(pair_count);
            assert_eq!(
                validated
                    .round_coefficients(&equality_first, &equality_second)
                    .unwrap(),
                preindexed
                    .round_coefficients(&equality_first, &equality_second)
                    .unwrap(),
                "round {round}"
            );
            let challenge = extension(round as u64 + 71);
            let next_equality = (round == 2).then(|| split_weights(domain_len / 16));
            let validated_next = next_equality
                .as_ref()
                .map(|(first, second)| (first.as_slice(), second.as_slice()));
            let preindexed_next = next_equality
                .as_ref()
                .map(|(first, second)| (first.as_slice(), second.as_slice()));
            assert_eq!(
                validated.bind_compact(challenge, validated_next).unwrap(),
                preindexed.bind_compact(challenge, preindexed_next).unwrap(),
                "bind {round}"
            );
        }
        assert_eq!(
            validated.into_dense_sources().unwrap(),
            preindexed.into_dense_sources().unwrap()
        );
    }

    #[test]
    fn pair_lut_round_one_matches_generic_fold() {
        for pattern_base in [4, 8, 16] {
            let domain_len = 1 << 10;
            let tables = digit_tables(pattern_base, domain_len);
            let mut group = QuadraticGroup::new(sources(&tables, pattern_base), 10).unwrap();
            let first_challenge = extension(1);
            group.bind_compact(first_challenge, None).unwrap();
            let pair_count = domain_len / 4;
            let (equality_first, equality_second) = split_weights(pair_count);

            let expected = direct_quartet_coefficients(
                &group,
                &equality_first,
                &equality_second,
                first_challenge,
            );
            let actual = group
                .round_coefficients(&equality_first, &equality_second)
                .unwrap();
            assert_eq!(actual, expected, "pattern base {pattern_base}");
        }
    }

    #[test]
    fn pair_lut_round_two_matches_generic_fold_without_quartet_lut() {
        for pattern_base in [4, 8, 16] {
            let domain_len = 1 << 8;
            let tables = digit_tables(pattern_base, domain_len);
            let mut group = QuadraticGroup::new(sources(&tables, pattern_base), 8).unwrap();
            let challenges = [extension(2), extension(3)];
            group.bind_compact(challenges[0], None).unwrap();
            group.bind_compact(challenges[1], None).unwrap();
            assert!(group.quartet_lut.is_none());
            let pair_count = domain_len / 8;
            let (equality_first, equality_second) = split_weights(pair_count);

            let expected = sum_split_coefficients(&equality_first, &equality_second, |octet| {
                compact_octet_coefficients(&group, octet, challenges[0], challenges[1])
            });
            let actual = group
                .round_coefficients(&equality_first, &equality_second)
                .unwrap();
            assert_eq!(actual, expected, "pattern base {pattern_base}");
        }
    }

    #[test]
    fn compact_rounds_clear_and_reuse_scaled_weights() {
        for pattern_base in [4, 8, 16] {
            for num_rounds in [1, 2, 3, 10, 16] {
                let domain_len = 1 << num_rounds;
                let tables = digit_tables(pattern_base, domain_len);
                let mut group =
                    QuadraticGroup::new(sources(&tables, pattern_base), num_rounds).unwrap();
                let mut challenges = Vec::new();
                for round in 0..num_rounds.min(3) {
                    let pair_count = (domain_len >> round) / 2;
                    let (first, second) = split_weights(pair_count);
                    let expected = sum_split_coefficients(&first, &second, |pair| {
                        let mut output = [E::zero(); QUADRATIC_COEFFICIENTS];
                        for source in 0..group.sources.len() {
                            let width = 1 << round;
                            let left = if round == 0 {
                                group.raw_value(source, 2 * pair)
                            } else {
                                folded_source(&group, source, 2 * pair * width, width, &challenges)
                            };
                            let right = if round == 0 {
                                group.raw_value(source, 2 * pair + 1)
                            } else {
                                folded_source(
                                    &group,
                                    source,
                                    (2 * pair + 1) * width,
                                    width,
                                    &challenges,
                                )
                            };
                            group.add_source_coefficients(&mut output, source, left, right);
                        }
                        output
                    });
                    assert_eq!(group.round_coefficients(&first, &second).unwrap(), expected);
                    let capacity = group.scaled_weights.capacity();
                    let allocation = group.scaled_weights.as_ptr();
                    assert_eq!(group.round_coefficients(&first, &second).unwrap(), expected);
                    let zeros = vec![E::zero(); first.len()];
                    assert_eq!(
                        group.round_coefficients(&zeros, &second).unwrap(),
                        [E::zero(); QUADRATIC_COEFFICIENTS]
                    );
                    let (quadratic, _) = group
                        .round_coefficients_with_linear(
                            &first,
                            &second,
                            &first,
                            &second,
                            &[E::one(); 3],
                            &[0; 3],
                        )
                        .unwrap();
                    assert_eq!(quadratic, expected);
                    assert_eq!(group.round_coefficients(&first, &second).unwrap(), expected);
                    assert_eq!(group.scaled_weights.capacity(), capacity);
                    assert_eq!(group.scaled_weights.as_ptr(), allocation);
                    assert_eq!(
                        group.scaled_weights.len(),
                        if round == 0 { 0 } else { first.len() * 3 }
                    );
                    let challenge = extension(round as u64 + 31);
                    let next_equality =
                        (round == 2 && num_rounds > 3).then(|| split_weights(domain_len / 16));
                    let next_equality = next_equality
                        .as_ref()
                        .map(|(first, second)| (first.as_slice(), second.as_slice()));
                    group.bind_compact(challenge, next_equality).unwrap();
                    challenges.push(challenge);
                }
            }
        }
    }

    fn assert_fused_materialization_matches_generic(
        pattern_base: usize,
        num_rounds: usize,
        expect_quartet_lut: bool,
    ) {
        let domain_len = 1 << num_rounds;
        let tables = digit_tables(pattern_base, domain_len);
        let mut group = QuadraticGroup::new(sources(&tables, pattern_base), num_rounds).unwrap();
        let challenges = [extension(4), extension(5), extension(6)];
        group.bind_compact(challenges[0], None).unwrap();
        group.bind_compact(challenges[1], None).unwrap();
        assert_eq!(group.quartet_lut.is_some(), expect_quartet_lut);
        let rows = domain_len / 8;
        let (equality_first, equality_second) = split_weights(rows / 2);

        let (actual_table, actual_coefficients) = group
            .materialize_after_three_with_coefficients(
                &challenges,
                &equality_first,
                &equality_second,
            )
            .unwrap();
        let expected_table = group.materialize_folded(8, &challenges).unwrap();
        let lanes = group.sources.len();
        let expected_coefficients =
            sum_split_coefficients(&equality_first, &equality_second, |pair| {
                let mut output = [E::zero(); QUADRATIC_COEFFICIENTS];
                for source in 0..lanes {
                    group.add_source_coefficients(
                        &mut output,
                        source,
                        expected_table[2 * pair * lanes + source],
                        expected_table[(2 * pair + 1) * lanes + source],
                    );
                }
                output
            });

        let case = format!("pattern base {pattern_base}, quartet LUT {expect_quartet_lut}");
        assert_eq!(actual_table, expected_table, "{case}");
        assert_eq!(actual_coefficients, expected_coefficients, "{case}");
    }

    #[test]
    fn fused_materialization_without_quartet_lut_matches_generic() {
        for pattern_base in [4, 8, 16] {
            assert_fused_materialization_matches_generic(pattern_base, 8, false);
        }
    }

    #[test]
    fn fused_materialization_with_quartet_lut_matches_generic() {
        for (pattern_base, num_rounds) in [(8, 15), (16, 19)] {
            assert_fused_materialization_matches_generic(pattern_base, num_rounds, true);
        }
    }

    #[test]
    fn compact_indexing_validates_pairs_and_singletons() {
        let invalid_pair = [0_i8, 8];
        assert!(CodedSource::new(&invalid_pair, -4, 3).is_err());

        let valid_singleton = [3_i8];
        assert!(CodedSource::new(&valid_singleton, -4, 3).is_ok());

        let invalid_singleton = [4_i8];
        assert!(CodedSource::new(&invalid_singleton, -4, 3).is_err());
    }
}
