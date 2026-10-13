//! Stage-one link kernels for the three-stage Falcon PIOP.
//!
//! The committed coefficient domain can contain 2^26 cells, so the prover
//! keeps the projection transpose in its 203-channel tensor form through the
//! high-row rounds. Only the native four-cell slack domain is materialized.

use jolt_field::{
    ExtField as _, Field as _, Fold, MulBaseUnreduced as _, One as _, Ring as _, Unreduced,
    Zero as _,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::substrate::{BaseField as F, OpeningField as E};

use super::evaluate_compressed;
use crate::protocol::algebra::{eq_table, fold, fold_lifted_i64, pair_sums};
use crate::protocol::projection::cores::ProjectionCores;
use crate::protocol::projection::forward::{ForwardProjection, RetainedDelta};
use crate::protocol::projection::profile::{
    Geometry, FIRST_CHANNELS, NORM_CHANNELS, NORM_GROUPS, NORM_PREFIX_WIDTH, NORM_TERMINAL_ROWS,
    RESIDUAL_WIDTH,
};
use crate::protocol::projection::reverse::{
    first_norm_weights, first_residual_weights, point_weights, rest_weights,
};
use crate::protocol::projection::OUTPUT_COUNT;
use crate::protocol::{FalconDegree, FalconWitness};

/// Two local-domain pieces of the single stage-one projection identity.
///
/// `coefficient` uses all `d` variables. `slack` uses the native
/// `(slot,row)` domain and sits out the first `d_coef - 2` rounds.
pub(crate) struct ProjectionLinkProver<'a> {
    batch: &'a FalconWitness,
    projection: &'a ForwardProjection,
    geometry: Geometry,
    rounds: usize,
    inactive_slack: usize,
    rest_weights: Vec<E>,
    rest_values: RestValues<'a>,
    rest_point: Vec<E>,
    rest_sums: Option<[E; 2]>,
    coefficient_claim: E,
    coefficient_weights_s1: Vec<E>,
    coefficient_weights_delta: Vec<E>,
    coefficient_values_s1: Vec<E>,
    coefficient_values_delta: Vec<E>,
    slack_weights: Vec<E>,
    slack_values: SlackValues,
    slack_claim: E,
    half: E,
    slack_scale: E,
}

enum RestValues<'a> {
    Raw(&'a [i64]),
    Folded(Vec<E>),
}

enum SlackValues {
    Raw(Vec<i16>),
    Folded(Vec<E>),
}

#[derive(Clone, Copy)]
struct RestRoundAccum {
    at_zero: Product,
    at_two: Product,
}

impl RestRoundAccum {
    fn zero() -> Self {
        Self {
            at_zero: Product::zero(),
            at_two: Product::zero(),
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.at_zero += other.at_zero;
        self.at_two += other.at_two;
        self
    }

    fn reduce(self) -> [E; 2] {
        [
            E::reduce_product(self.at_zero),
            E::reduce_product(self.at_two),
        ]
    }
}

const REST_TILE: usize = 16;
#[cfg(feature = "parallel")]
const MIN_PARALLEL_REST: usize = 1 << 13;
#[cfg(feature = "parallel")]
use crate::protocol::kernels::kernel_common::PARALLEL_MIN_WORK;

impl<'a> ProjectionLinkProver<'a> {
    pub(crate) fn new(
        batch: &'a FalconWitness,
        projection: &'a ForwardProjection,
        lambda: &[E],
        output_claim: E,
    ) -> Result<Self, akita_error::AkitaError> {
        let geometry = projection.geometry;
        if batch.degree() != geometry.n
            || batch.padded_rows != geometry.records
            || batch.live_rows != projection.live_rows
            || lambda.len() != OUTPUT_COUNT
        {
            return Err(invalid("stage-one projection link shape mismatch"));
        }

        let half = E::from_u64(2).inverse().expect("two is invertible");
        let rest_weights = tracing::info_span!("projection_link_rest_weights")
            .in_scope(|| rest_weights(&projection.cores, &geometry, lambda));
        // The native slack table has only 4 * N_sig* cells. Index bits are
        // high row bits, slot bits, then low row bits, exactly Section 7.2.
        let slack_weights = tracing::info_span!("projection_link_slack_weights")
            .in_scope(|| slack_projection_weights(&geometry, &projection.cores, &rest_weights));
        let mut slack_values = vec![0_i16; slack_weights.len()];
        for high in 0..geometry.rest_size() {
            for low in 0..geometry.low_records() {
                let record = high * geometry.low_records() + low;
                for slot in 0..4 {
                    let index = high + geometry.rest_size() * (slot + 4 * low);
                    slack_values[index] = batch.slack[4 * record + slot];
                }
            }
        }
        let slack_claim = slack_weights
            .iter()
            .zip(&slack_values)
            .fold(E::zero(), |sum, (&weight, &value)| {
                sum + weight.mul_base(F::from_i64(i64::from(value)))
            });
        let coefficient_claim = output_claim - slack_claim;

        let mut state = Self {
            batch,
            projection,
            geometry,
            rounds: geometry.logn + geometry.record_bits,
            inactive_slack: geometry.logn - 2,
            rest_weights,
            rest_values: RestValues::Raw(&projection.first_without_slack),
            rest_point: Vec::with_capacity(geometry.rest_bits),
            rest_sums: None,
            coefficient_claim,
            coefficient_weights_s1: Vec::new(),
            coefficient_weights_delta: Vec::new(),
            coefficient_values_s1: Vec::new(),
            coefficient_values_delta: Vec::new(),
            slack_weights,
            slack_values: SlackValues::Raw(slack_values),
            slack_claim,
            half,
            slack_scale: E::one(),
        };
        if geometry.rest_bits == 0 {
            state.enter_coefficient_block();
        }
        Ok(state)
    }

    pub(crate) fn initial_claim(&self) -> E {
        self.coefficient_claim + self.slack_claim
    }

    pub(crate) fn inactive_slack_rounds(&self) -> usize {
        self.inactive_slack
    }

    pub(crate) fn coefficient_round_phase(&self, round: usize) -> &'static str {
        if round < self.geometry.rest_bits {
            "rest"
        } else {
            "coefficient"
        }
    }

    /// Compressed ordinary-cubic message for the coefficient contribution.
    pub(crate) fn coefficient_round(&self, round: usize) -> [E; 3] {
        debug_assert!(round < self.rounds);
        if round < self.geometry.rest_bits {
            return self.rest_round(round);
        }
        product_round(
            &self.coefficient_weights_s1,
            &self.coefficient_values_s1,
            &self.coefficient_weights_delta,
            &self.coefficient_values_delta,
            self.coefficient_claim,
        )
    }

    /// Compressed ordinary-cubic message for the native slack contribution.
    pub(crate) fn slack_round(&self, round: usize) -> [E; 3] {
        if round < self.inactive_slack {
            return [self.slack_claim * self.half, E::zero(), E::zero()];
        }
        product_round_single(&self.slack_weights, &self.slack_values, self.slack_scale)
    }

    pub(crate) fn coefficient_bind(&mut self, round: usize, challenge: E, message: [E; 3]) {
        self.coefficient_claim = evaluate_compressed(message, self.coefficient_claim, challenge);
        if round < self.geometry.rest_bits {
            self.rest_point.push(challenge);
            self.fold_rest(challenge, round + 1 == self.geometry.rest_bits);
        } else {
            fold(&mut self.coefficient_weights_s1, challenge);
            fold(&mut self.coefficient_weights_delta, challenge);
            fold(&mut self.coefficient_values_s1, challenge);
            fold(&mut self.coefficient_values_delta, challenge);
        }
    }

    pub(crate) fn slack_bind(&mut self, round: usize, challenge: E, message: [E; 3]) {
        self.slack_claim = evaluate_compressed(message, self.slack_claim, challenge);
        if round < self.inactive_slack {
            self.slack_scale *= self.half;
        } else {
            fold(&mut self.slack_weights, challenge);
            let values =
                match core::mem::replace(&mut self.slack_values, SlackValues::Folded(Vec::new())) {
                    SlackValues::Raw(values) => {
                        SlackValues::Folded(fold_lifted_i64(&values, challenge))
                    }
                    SlackValues::Folded(mut values) => {
                        fold(&mut values, challenge);
                        SlackValues::Folded(values)
                    }
                };
            self.slack_values = values;
        }
    }

    pub(crate) fn terminal_claims(&self) -> [E; 3] {
        debug_assert_eq!(self.coefficient_values_s1.len(), 1);
        debug_assert_eq!(self.coefficient_values_delta.len(), 1);
        let slack_terminal = match &self.slack_values {
            SlackValues::Raw(values) => E::from_i64(i64::from(values[0])),
            SlackValues::Folded(values) => values[0],
        };
        debug_assert_eq!(slack_values_len(&self.slack_values), 1);
        [
            self.coefficient_values_s1[0],
            slack_terminal,
            self.coefficient_values_delta[0],
        ]
    }

    fn rest_round(&self, _round: usize) -> [E; 3] {
        let sums = match &self.rest_values {
            RestValues::Raw(values) => raw_round_sums(&self.rest_weights, values),
            RestValues::Folded(_) => self
                .rest_sums
                .expect("the previous rest fold computed this round"),
        };
        round_accum_to_message(sums, self.coefficient_claim)
    }

    fn fold_rest(&mut self, challenge: E, last: bool) {
        let (weights, next_values, sums) =
            fold_rest_round(&self.rest_weights, &self.rest_values, challenge, !last);
        self.rest_weights = weights;
        self.rest_values = RestValues::Folded(next_values);
        self.rest_sums = sums;
        if last {
            self.enter_coefficient_block();
        }
    }

    fn enter_coefficient_block(&mut self) {
        // Contract the already-folded public transpose weights through the
        // first cores. The rest values are only needed to form the rest-round
        // messages; after binding those challenges, the coefficient relation
        // is the folded committed sources against A_1^T U(r_rest).
        debug_assert_eq!(self.rest_weights.len(), FIRST_CHANNELS);
        let norm = first_norm_weights(&self.projection.cores, &self.rest_weights[..NORM_CHANNELS]);
        let residual =
            first_residual_weights(&self.projection.cores, &self.rest_weights[NORM_CHANNELS..]);
        let (s1, delta) = fold_coefficient_sources(
            self.batch,
            &self.projection.delta,
            &self.geometry,
            &self.rest_point,
        );
        let n = self.geometry.n;
        let low_records = self.geometry.low_records();
        let mut weights_s1 = vec![E::zero(); n * low_records];
        for low in 0..low_records {
            weights_s1[n * low..n * (low + 1)].copy_from_slice(&norm[2 * n * low..2 * n * low + n]);
        }
        self.coefficient_weights_s1 = weights_s1;
        self.coefficient_weights_delta = residual;
        self.coefficient_values_s1 = s1;
        self.coefficient_values_delta = delta;
        let folded_claim = self
            .coefficient_weights_s1
            .iter()
            .zip(&self.coefficient_values_s1)
            .chain(
                self.coefficient_weights_delta
                    .iter()
                    .zip(&self.coefficient_values_delta),
            )
            .fold(E::zero(), |sum, (&weight, &value)| sum + weight * value);
        let channel_claim = match &self.rest_values {
            RestValues::Raw(values) => self
                .rest_weights
                .iter()
                .zip(values.iter())
                .fold(E::zero(), |sum, (&weight, &value)| {
                    sum + weight * E::from_i64(value)
                }),
            RestValues::Folded(values) => self
                .rest_weights
                .iter()
                .zip(values)
                .fold(E::zero(), |sum, (&weight, &value)| sum + weight * value),
        };
        // This checks that the two prover-side kernels represent the same
        // folded source relation, independently of the public output claim.
        // A deliberately tampered projection output must produce a proof that
        // the verifier rejects rather than a prover-side panic.
        debug_assert_eq!(folded_claim, channel_claim);
    }
}

type Product = <E as Unreduced>::Product;

#[derive(Clone, Copy)]
struct ProductAccum {
    constant: Product,
    quadratic: Product,
}

impl ProductAccum {
    fn zero() -> Self {
        Self {
            constant: Product::zero(),
            quadratic: Product::zero(),
        }
    }

    fn add_pair(&mut self, weights: &[E], values: &[E], pair: usize) {
        let low = 2 * pair;
        let w0 = weights[low];
        let dw = weights[low + 1] - w0;
        let v0 = values[low];
        let dv = values[low + 1] - v0;
        self.constant += w0.mul_unreduced(v0);
        self.quadratic += dw.mul_unreduced(dv);
    }

    fn merge(mut self, other: Self) -> Self {
        self.constant += other.constant;
        self.quadratic += other.quadratic;
        self
    }

    fn reduce(self) -> [E; 2] {
        [
            E::reduce_product(self.constant),
            E::reduce_product(self.quadratic),
        ]
    }
}

fn folded_fiber_pair(weights: (&[E], &[E]), values: (&[E], &[E])) -> RestRoundAccum {
    let mut accum = RestRoundAccum::zero();
    for (((&w0, &w1), &v0), &v1) in weights.0.iter().zip(weights.1).zip(values.0).zip(values.1) {
        accum.at_zero += w0.mul_unreduced(v0);
        accum.at_two += (w1 + w1 - w0).mul_unreduced(v1 + v1 - v0);
    }
    accum
}

fn round_accum_to_message([at_zero, at_two]: [E; 2], claim: E) -> [E; 3] {
    // For this quadratic, claim = g(0) + g(1). Solving for the x^2
    // coefficient from g(0), g(2), and claim gives
    // c2 = (g(2) + 3 g(0) - 2 claim) / 2.
    let quadratic = (at_two + at_zero + at_zero + at_zero - claim - claim)
        * E::from_u64(2).inverse().expect("two is invertible");
    [at_zero, quadratic, E::zero()]
}

fn raw_round_sums(weights: &[E], values: &[i64]) -> [E; 2] {
    let pairs = weights.len() / (2 * FIRST_CHANNELS);
    let contribution = |pair: usize| {
        let low = 2 * pair * FIRST_CHANNELS;
        let high = low + FIRST_CHANNELS;
        let mut sums = [[0_i128; 2]; 2];
        for channel in 0..FIRST_CHANNELS {
            let (w0, w1) = (weights[low + channel], weights[high + channel]);
            let (v0, v1) = (values[low + channel], values[high + channel]);
            let points = [(w0, v0), (w1 + w1 - w0, v1 + v1 - v0)];
            for (sum, (weight, value)) in sums.iter_mut().zip(points) {
                let limbs = weight.coeffs.map(F::to_canonical_u64);
                sum[0] += i128::from(limbs[0]) * i128::from(value);
                sum[1] += i128::from(limbs[1]) * i128::from(value);
            }
        }
        sums.map(|[lo, hi]| E::new(F::from_i128(lo), F::from_i128(hi)))
    };
    pair_sums::<2, _>(pairs, contribution)
}

fn product_round(
    first_weights: &[E],
    first_values: &[E],
    second_weights: &[E],
    second_values: &[E],
    _claim: E,
) -> [E; 3] {
    let pairs = first_weights.len() / 2;
    let contribution = |pair: usize| {
        let mut accum = ProductAccum::zero();
        accum.add_pair(first_weights, first_values, pair);
        accum.add_pair(second_weights, second_values, pair);
        accum
    };
    #[cfg(feature = "parallel")]
    let sums = if first_weights.len() >= 1 << 13 {
        (0..pairs)
            .into_par_iter()
            .map(contribution)
            .reduce(ProductAccum::zero, ProductAccum::merge)
    } else {
        (0..pairs)
            .map(contribution)
            .fold(ProductAccum::zero(), ProductAccum::merge)
    };
    #[cfg(not(feature = "parallel"))]
    let sums = (0..pairs)
        .map(contribution)
        .fold(ProductAccum::zero(), ProductAccum::merge);
    let [c0, c2] = sums.reduce();
    [c0, c2, E::zero()]
}

fn product_round_single(weights: &[E], values: &SlackValues, scale: E) -> [E; 3] {
    let pairs = weights.len() / 2;
    let mut sums = ProductAccum::zero();
    match values {
        SlackValues::Raw(values) => {
            for pair in 0..pairs {
                let low = 2 * pair;
                let w0 = weights[low];
                let dw = weights[low + 1] - w0;
                let v0 = F::from_i64(i64::from(values[low]));
                let dv = F::from_i64(i64::from(values[low + 1]) - i64::from(values[low]));
                sums.constant += (scale * w0).mul_base_unreduced(v0);
                sums.quadratic += (scale * dw).mul_base_unreduced(dv);
            }
        }
        SlackValues::Folded(values) => {
            for pair in 0..pairs {
                let low = 2 * pair;
                let w0 = weights[low];
                let dw = weights[low + 1] - w0;
                let v0 = values[low];
                let dv = values[low + 1] - v0;
                sums.constant += (scale * w0).mul_unreduced(v0);
                sums.quadratic += (scale * dw).mul_unreduced(dv);
            }
        }
    }
    let [c0, c2] = sums.reduce();
    [c0, c2, E::zero()]
}

fn slack_values_len(values: &SlackValues) -> usize {
    match values {
        SlackValues::Raw(values) => values.len(),
        SlackValues::Folded(values) => values.len(),
    }
}

fn rest_values_len(values: &RestValues<'_>) -> usize {
    match values {
        RestValues::Raw(values) => values.len(),
        RestValues::Folded(values) => values.len(),
    }
}

fn fold_rest_round(
    weights: &[E],
    values: &RestValues<'_>,
    challenge: E,
    evaluate_next_round: bool,
) -> (Vec<E>, Vec<E>, Option<[E; 2]>) {
    debug_assert_eq!(weights.len(), rest_values_len(values));
    debug_assert!(weights.len().is_multiple_of(2 * FIRST_CHANNELS));
    let out_len = weights.len() / 2;
    let mut new_weights = vec![E::zero(); out_len];
    let mut new_values = vec![E::zero(); out_len];
    let context = E::precompute(challenge);
    let tile = REST_TILE * FIRST_CHANNELS;
    let apply = |task: usize, (out_weights, out_values): (&mut [E], &mut [E])| {
        let start = 2 * task * tile;
        let in_weights = &weights[start..start + 2 * out_weights.len()];
        for (slot, pair) in out_weights
            .chunks_exact_mut(FIRST_CHANNELS)
            .zip(in_weights.chunks_exact(2 * FIRST_CHANNELS))
        {
            let (low, high) = pair.split_at(FIRST_CHANNELS);
            for ((slot, &low), &high) in slot.iter_mut().zip(low).zip(high) {
                *slot = E::fold_one(&context, low, high);
            }
        }
        match values {
            RestValues::Raw(raw) => {
                let in_values = &raw[start..start + 2 * out_values.len()];
                for (slot, pair) in out_values
                    .chunks_exact_mut(FIRST_CHANNELS)
                    .zip(in_values.chunks_exact(2 * FIRST_CHANNELS))
                {
                    let (low, high) = pair.split_at(FIRST_CHANNELS);
                    for ((slot, &low), &high) in slot.iter_mut().zip(low).zip(high) {
                        *slot = E::from_i64(low) + challenge.mul_base(F::from_i64(high - low));
                    }
                }
            }
            RestValues::Folded(folded) => {
                let in_values = &folded[start..start + 2 * out_values.len()];
                for (slot, pair) in out_values
                    .chunks_exact_mut(FIRST_CHANNELS)
                    .zip(in_values.chunks_exact(2 * FIRST_CHANNELS))
                {
                    let (low, high) = pair.split_at(FIRST_CHANNELS);
                    for ((slot, &low), &high) in slot.iter_mut().zip(low).zip(high) {
                        *slot = E::fold_one(&context, low, high);
                    }
                }
            }
        }

        let mut accum = RestRoundAccum::zero();
        if evaluate_next_round {
            for (weights, values) in out_weights
                .chunks_exact(2 * FIRST_CHANNELS)
                .zip(out_values.chunks_exact(2 * FIRST_CHANNELS))
            {
                accum = accum.merge(folded_fiber_pair(
                    weights.split_at(FIRST_CHANNELS),
                    values.split_at(FIRST_CHANNELS),
                ));
            }
        }
        accum
    };
    #[cfg(feature = "parallel")]
    let accum = if evaluate_next_round && out_len >= MIN_PARALLEL_REST {
        new_weights
            .par_chunks_mut(tile)
            .zip(new_values.par_chunks_mut(tile))
            .enumerate()
            .map(|(task, item)| apply(task, item))
            .reduce(RestRoundAccum::zero, RestRoundAccum::merge)
    } else {
        new_weights
            .chunks_mut(tile)
            .zip(new_values.chunks_mut(tile))
            .enumerate()
            .fold(RestRoundAccum::zero(), |sum, (task, item)| {
                sum.merge(apply(task, item))
            })
    };
    #[cfg(not(feature = "parallel"))]
    let accum = new_weights
        .chunks_mut(tile)
        .zip(new_values.chunks_mut(tile))
        .enumerate()
        .fold(RestRoundAccum::zero(), |sum, (task, item)| {
            sum.merge(apply(task, item))
        });
    let sums = evaluate_next_round.then(|| accum.reduce());
    (new_weights, new_values, sums)
}

fn slack_projection_weights(
    geometry: &Geometry,
    cores: &ProjectionCores,
    rest_weights: &[E],
) -> Vec<E> {
    let rest_size = geometry.rest_size();
    let values_per_high = 4 * geometry.low_records();
    let mut high_major = vec![E::zero(); rest_size * values_per_high];
    let fill = |high: usize, output: &mut [E]| {
        let channels = &rest_weights[high * FIRST_CHANNELS..(high + 1) * FIRST_CHANNELS];
        for group in 0..NORM_GROUPS {
            let core = cores.norm(group, 1);
            for channel in 0..NORM_PREFIX_WIDTH {
                let weight = channels[group * NORM_PREFIX_WIDTH + channel];
                // Norm cores are ternary. Reuse their three scalar multiples
                // across all native slack columns instead of multiplying in K.
                let multiples = [-weight, E::zero(), weight];
                let row = core.row(channel);
                for low in 0..geometry.low_records() {
                    for slot in 0..4 {
                        let column = geometry.n + slot + 2 * geometry.n * low;
                        let local_index = slot + 4 * low;
                        output[local_index] += multiples[(row[column] + 1) as usize];
                    }
                }
            }
        }
    };
    #[cfg(feature = "parallel")]
    if high_major.len() >= PARALLEL_MIN_WORK {
        high_major
            .par_chunks_mut(values_per_high)
            .enumerate()
            .for_each(|(high, output)| fill(high, output));
    } else {
        high_major
            .chunks_mut(values_per_high)
            .enumerate()
            .for_each(|(high, output)| fill(high, output));
    }
    #[cfg(not(feature = "parallel"))]
    high_major
        .chunks_mut(values_per_high)
        .enumerate()
        .for_each(|(high, output)| fill(high, output));

    // Accumulate each fiber contiguously, then transpose to the wire's
    // canonical `(high, slot, low)` order.
    let mut out = vec![E::zero(); rest_size * values_per_high];
    for high in 0..rest_size {
        for low in 0..geometry.low_records() {
            for slot in 0..4 {
                let local_index = slot + 4 * low;
                let canonical_index = high + rest_size * local_index;
                out[canonical_index] = high_major[high * values_per_high + local_index];
            }
        }
    }
    out
}

fn slack_first_norm_weights(
    geometry: &Geometry,
    cores: &ProjectionCores,
    channel_weights: &[E],
) -> Vec<E> {
    let mut out = vec![E::zero(); 4 * geometry.low_records()];
    for group in 0..NORM_GROUPS {
        let core = cores.norm(group, 1);
        for channel in 0..NORM_PREFIX_WIDTH {
            let weight = channel_weights[group * NORM_PREFIX_WIDTH + channel];
            let row = core.row(channel);
            for record_low in 0..geometry.low_records() {
                for slot in 0..4 {
                    let column = geometry.n + slot + 2 * geometry.n * record_low;
                    match row[column] {
                        -1 => out[slot + 4 * record_low] -= weight,
                        0 => {}
                        1 => out[slot + 4 * record_low] += weight,
                        value => debug_assert!(
                            (-1..=1).contains(&value),
                            "derived norm core entry is not ternary"
                        ),
                    }
                }
            }
        }
    }
    out
}

const FOLD_BLOCK: usize = 64;
const FOLD_TILE: usize = 16;
const FOLD_RUN: usize = 1 << 16;
const I16_BIAS: i64 = 1 << 15;
const LIMB_HALVES: usize = 4;

#[inline]
fn limbs(value: E) -> [u64; 2] {
    value.coeffs.map(F::to_canonical_u64)
}

#[inline]
fn limb_halves(value: E) -> [u32; LIMB_HALVES] {
    let [c0, c1] = limbs(value);
    [c0 as u32, (c0 >> 32) as u32, c1 as u32, (c1 >> 32) as u32]
}

struct HalfAccum {
    lanes: [[u64; FOLD_BLOCK]; LIMB_HALVES],
    sums: [E; FOLD_BLOCK],
}

impl HalfAccum {
    fn new() -> Self {
        Self {
            lanes: [[0; FOLD_BLOCK]; LIMB_HALVES],
            sums: [E::zero(); FOLD_BLOCK],
        }
    }

    #[inline]
    fn add(&mut self, weight: &[u32; LIMB_HALVES], cells: &[u16; FOLD_BLOCK]) {
        for (lanes, &half) in self.lanes.iter_mut().zip(weight) {
            for (lane, &cell) in lanes.iter_mut().zip(cells) {
                *lane += u64::from(cell) * u64::from(half);
            }
        }
    }

    #[inline]
    fn add_tile(&mut self, weights: &[[u32; LIMB_HALVES]], cells: &[[u16; FOLD_BLOCK]; FOLD_TILE]) {
        for (half, lanes) in self.lanes.iter_mut().enumerate() {
            let tile_weights: [u64; FOLD_TILE] =
                core::array::from_fn(|tile| u64::from(weights[tile][half]));
            for (cell, lane) in lanes.iter_mut().enumerate() {
                let mut sum = *lane;
                for tile in 0..FOLD_TILE {
                    sum += u64::from(cells[tile][cell]) * tile_weights[tile];
                }
                *lane = sum;
            }
        }
    }

    fn flush(&mut self) {
        for (cell, sum) in self.sums.iter_mut().enumerate() {
            let limb = |low: usize| {
                F::from_u128(
                    u128::from(self.lanes[low][cell])
                        + (u128::from(self.lanes[low + 1][cell]) << 32),
                )
            };
            *sum += E::new(limb(0), limb(2));
        }
        self.lanes = [[0; FOLD_BLOCK]; LIMB_HALVES];
    }
}

#[inline]
fn biased(value: i16) -> u16 {
    (value as u16) ^ 0x8000
}

/// Fold the committed coefficient sources over the high-row point using the
/// old P-block's exact integer kernel. This streams 64 coefficient cells at
/// a time, keeps `S1` and narrow `Delta` in i16 limbs, and parallelizes over
/// independent first-block slices.
fn fold_coefficient_sources(
    batch: &FalconWitness,
    retained: &RetainedDelta,
    geometry: &Geometry,
    rest_point: &[E],
) -> (Vec<E>, Vec<E>) {
    let n = geometry.n;
    let low_records = geometry.low_records();
    let cells = low_records * n;
    debug_assert!(n.is_multiple_of(FOLD_BLOCK));
    let rest_eq = eq_table(rest_point);
    let weights: Vec<[u32; LIMB_HALVES]> =
        rest_eq.iter().map(|&weight| limb_halves(weight)).collect();
    let total_weight = rest_eq.iter().fold(E::zero(), |sum, &weight| sum + weight);
    let fold_block = |block: usize| -> ([E; FOLD_BLOCK], [E; FOLD_BLOCK]) {
        let low = block * FOLD_BLOCK / n;
        let column = block * FOLD_BLOCK % n;
        let mut s1_acc = HalfAccum::new();
        let mut delta_low = HalfAccum::new();
        let mut delta_high = HalfAccum::new();
        let mut wide_delta = [E::zero(); FOLD_BLOCK];
        let mut wide_weight = E::zero();
        let mut s1_cells = [[0_u16; FOLD_BLOCK]; FOLD_TILE];
        let mut low_cells = [[0_u16; FOLD_BLOCK]; FOLD_TILE];
        let mut high_cells = [[0_u16; FOLD_BLOCK]; FOLD_TILE];
        let mut narrow_tile = [false; FOLD_TILE];
        for (run, run_weights) in weights.chunks(FOLD_RUN).enumerate() {
            for (tile, tile_weights) in run_weights.chunks(FOLD_TILE).enumerate() {
                for slot in 0..tile_weights.len() {
                    let fiber = run * FOLD_RUN + FOLD_TILE * tile + slot;
                    let record = fiber * low_records + low;
                    let start = record * n + column;
                    for (cell, &value) in s1_cells[slot]
                        .iter_mut()
                        .zip(&batch.s1[start..][..FOLD_BLOCK])
                    {
                        *cell = biased(value);
                    }
                    narrow_tile[slot] = retained.wide[record].is_none();
                    match &retained.wide[record] {
                        None => {
                            let narrow = &retained.narrow[start..][..FOLD_BLOCK];
                            for ((lo, hi), &value) in low_cells[slot]
                                .iter_mut()
                                .zip(&mut high_cells[slot])
                                .zip(narrow)
                            {
                                let (lo_value, hi_value) =
                                    crate::protocol::projection::forward::halves(value);
                                (*lo, *hi) = (biased(lo_value), biased(hi_value));
                            }
                        }
                        Some(wide) => {
                            let eq = rest_eq[fiber];
                            wide_weight += eq;
                            for (sum, &value) in
                                wide_delta.iter_mut().zip(&wide[column..][..FOLD_BLOCK])
                            {
                                *sum += eq * E::from_i64(value);
                            }
                        }
                    }
                }
                if tile_weights.len() == FOLD_TILE {
                    s1_acc.add_tile(tile_weights, &s1_cells);
                    if narrow_tile == [true; FOLD_TILE] {
                        delta_low.add_tile(tile_weights, &low_cells);
                        delta_high.add_tile(tile_weights, &high_cells);
                        continue;
                    }
                } else {
                    for (weight, source) in tile_weights.iter().zip(&s1_cells) {
                        s1_acc.add(weight, source);
                    }
                }
                for (slot, weight) in tile_weights.iter().enumerate() {
                    if narrow_tile[slot] {
                        delta_low.add(weight, &low_cells[slot]);
                        delta_high.add(weight, &high_cells[slot]);
                    }
                }
            }
            s1_acc.flush();
            delta_low.flush();
            delta_high.flush();
        }
        let narrow_weight = total_weight - wide_weight;
        let s1_bias = total_weight * E::from_i64(I16_BIAS);
        let delta_bias = narrow_weight * E::from_i64(I16_BIAS + (I16_BIAS << 16));
        let shift = E::from_u64(1 << 16);
        (
            s1_acc.sums.map(|sum| sum - s1_bias),
            core::array::from_fn(|cell| {
                delta_low.sums[cell] + shift * delta_high.sums[cell] - delta_bias + wide_delta[cell]
            }),
        )
    };
    let block_count = cells / FOLD_BLOCK;
    #[cfg(feature = "parallel")]
    let blocks: Vec<_> = if cells >= PARALLEL_MIN_WORK {
        (0..block_count).into_par_iter().map(fold_block).collect()
    } else {
        (0..block_count).map(fold_block).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let blocks: Vec<_> = (0..block_count).map(fold_block).collect();
    let mut s1 = Vec::with_capacity(cells);
    let mut delta = Vec::with_capacity(cells);
    for (s1_block, delta_block) in blocks {
        s1.extend_from_slice(&s1_block);
        delta.extend_from_slice(&delta_block);
    }
    (s1, delta)
}

/// Public transpose evaluations used by the stage-one terminal check.
/// Returns `(w_S1(r_pi), w_e(r_pi,e), w_Delta(r_pi))`.
pub(crate) fn projection_weights_at(
    degree: FalconDegree,
    records: usize,
    live_rows: usize,
    seed: &[u8],
    lambda: &[E],
    coefficient_point: &[E],
    slack_point: &[E],
) -> Result<[E; 3], akita_error::AkitaError> {
    if lambda.len() != OUTPUT_COUNT
        || coefficient_point.len() != degree.logn() + records.trailing_zeros() as usize
        || slack_point.len() != 2 + records.trailing_zeros() as usize
    {
        return Err(invalid("stage-one projection terminal shape mismatch"));
    }
    let geometry = Geometry::new(degree, records).map_err(invalid)?;
    let cores = ProjectionCores::derive(seed, &geometry, live_rows);
    let (lambda_norm, lambda_residual) = lambda.split_at(NORM_TERMINAL_ROWS);
    let (coef, row) = coefficient_point.split_at(geometry.logn);
    let (low, high) = row.split_at(geometry.low_record_bits);
    let channels = point_weights(&cores, &geometry, lambda_norm, lambda_residual, high);
    let norm = first_norm_weights(&cores, &channels[..NORM_CHANNELS]);
    let residual = first_residual_weights(&cores, &channels[NORM_CHANNELS..]);

    let norm_point = {
        let mut point = coef.to_vec();
        point.push(E::zero());
        point.extend_from_slice(low);
        point
    };
    let residual_point = {
        let mut point = coef.to_vec();
        point.extend_from_slice(low);
        point
    };
    let w_norm_s1 = mle(&norm, &norm_point);
    let w_delta = mle(&residual, &residual_point);

    let (slack_slot, slack_row) = slack_point.split_at(2);
    let (slack_low, slack_high) = slack_row.split_at(geometry.low_record_bits);
    let norm_channels = point_weights(
        &cores,
        &geometry,
        lambda_norm,
        &vec![E::zero(); RESIDUAL_WIDTH],
        slack_high,
    );
    let slack_table = slack_first_norm_weights(&geometry, &cores, &norm_channels[..NORM_CHANNELS]);
    let mut canonical_slack_point = slack_slot.to_vec();
    canonical_slack_point.extend_from_slice(slack_low);
    let w_e = mle(&slack_table, &canonical_slack_point);
    Ok([w_norm_s1, w_e, w_delta])
}

fn mle(table: &[E], point: &[E]) -> E {
    eq_table(point)
        .iter()
        .zip(table)
        .fold(E::zero(), |sum, (&weight, &value)| sum + weight * value)
}

fn invalid(message: &str) -> akita_error::AkitaError {
    akita_error::AkitaError::InvalidInput(message.to_owned())
}

#[cfg(test)]
mod slack_weight_tests {
    use super::*;
    use crate::protocol::projection::profile::Geometry;

    #[test]
    fn parallel_slack_weights_match_the_direct_projection_formula() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            let low_bits = 12 - (degree.logn() + 1);
            let geometry = Geometry::new(degree, 1 << (low_bits + 2)).unwrap();
            let cores = ProjectionCores::derive(&[0x5a; 32], &geometry, geometry.records - 1);
            let rest_weights = (0..geometry.rest_size() * FIRST_CHANNELS)
                .map(|index| E::from_u64((index as u64 % 97) + 1))
                .collect::<Vec<_>>();
            let actual = slack_projection_weights(&geometry, &cores, &rest_weights);

            let mut expected = vec![E::zero(); 4 * geometry.records];
            for high in 0..geometry.rest_size() {
                let channels = &rest_weights[high * FIRST_CHANNELS..(high + 1) * FIRST_CHANNELS];
                for group in 0..NORM_GROUPS {
                    let core = cores.norm(group, 1);
                    for channel in 0..NORM_PREFIX_WIDTH {
                        let weight = channels[group * NORM_PREFIX_WIDTH + channel];
                        let row = core.row(channel);
                        for low in 0..geometry.low_records() {
                            for slot in 0..4 {
                                let column = geometry.n + slot + 2 * geometry.n * low;
                                let index = high + geometry.rest_size() * (slot + 4 * low);
                                expected[index] +=
                                    weight.mul_base(F::from_i64(i64::from(row[column])));
                            }
                        }
                    }
                }
            }

            assert_eq!(actual, expected, "{degree:?}");
        }
    }

    #[test]
    fn selected_slack_columns_match_full_first_norm_weights() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            let low_bits = 12 - (degree.logn() + 1);
            let geometry = Geometry::new(degree, 1 << (low_bits + 2)).unwrap();
            let cores = ProjectionCores::derive(&[0xa5; 32], &geometry, geometry.records - 1);
            let channel_weights = (0..NORM_CHANNELS)
                .map(|index| E::from_u64((index as u64 % 101) + 1))
                .collect::<Vec<_>>();

            let actual = slack_first_norm_weights(&geometry, &cores, &channel_weights);
            let full = first_norm_weights(&cores, &channel_weights);
            let expected = (0..geometry.low_records())
                .flat_map(|record_low| {
                    (0..4).map({
                        let full = &full;
                        move |slot| full[geometry.n + slot + 2 * geometry.n * record_low]
                    })
                })
                .collect::<Vec<_>>();

            assert_eq!(actual, expected, "{degree:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::stage::evaluate_compressed;

    #[test]
    fn rest_round_compression_recovers_quadratics_from_g0_g2_and_claim() {
        let c0 = E::new(F::from_u64(17), F::from_u64(29));
        let c1 = E::new(F::from_u64(31), F::from_u64(43));
        let c2 = E::new(F::from_u64(47), F::from_u64(59));
        let claim = c0 + c0 + c1 + c2;
        let at_two = c0 + E::from_u64(2) * c1 + E::from_u64(4) * c2;
        let message = round_accum_to_message([c0, at_two], claim);

        for point in [
            E::zero(),
            E::one(),
            E::from_u64(2),
            E::new(F::from_u64(61), F::from_u64(71)),
        ] {
            let expected = c0 + c1 * point + c2 * point * point;
            assert_eq!(evaluate_compressed(message, claim, point), expected);
        }
    }
}
