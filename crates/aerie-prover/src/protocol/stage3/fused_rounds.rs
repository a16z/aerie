//! Integrated stage-three folds and coefficient contractions.
//!
//! The first two folds retain only ring weights, the first folded in place;
//! small integers stay raw, and a genuine sign lane is read by bit pattern.
//! Round three materializes four interleaved lanes directly from raw blocks.
//! Each task folds and contracts disjoint blocks before they leave cache.

use crate::error::AggregationError;
use akita_algebra::split_eq::GruenSplitEq;
use jolt_field::{Fold, One, Ring, Unreduced, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(test)]
use super::norm_equality;
use super::{
    compact::{equality_weighted_cubic, weighted_integer_coefficients},
    encoding_budget, range_source_link, ring_functional,
};
use crate::protocol::{
    algebra::{eq_table, live_prefix_evaluation, LiftedFoldPlan},
    kernels::{fold_rows_in_place, DigitSource, SignSource},
    source_tables::{PreparedCoefficients, SourceTables},
    stage::{evaluate_compressed, inactive_round},
    stage1::Stage1Output,
    stage2::Stage2Output,
    FalconDegree, FalconWitness,
};
use crate::substrate::OpeningField as E;

type Product = <E as Unreduced>::Product;
type Messages = [[E; 3]; 10];
const CORE_CHUNK_CELLS: usize = 1024;

#[cfg(all(test, feature = "parallel"))]
std::thread_local! {
    static FORCE_PARALLEL_SINGLE_CHUNK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(all(test, feature = "parallel"))]
pub(super) fn with_forced_parallel_single_chunk<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCE_PARALLEL_SINGLE_CHUNK.with(|flag| flag.set(self.0));
        }
    }

    let previous = FORCE_PARALLEL_SINGLE_CHUNK.with(|flag| flag.replace(true));
    let _restore = Restore(previous);
    run()
}

#[inline]
#[cfg(feature = "parallel")]
fn force_parallel_single_chunk() -> bool {
    #[cfg(test)]
    {
        FORCE_PARALLEL_SINGLE_CHUNK.with(std::cell::Cell::get)
    }
    #[cfg(not(test))]
    {
        false
    }
}

/// Recycled lane storage shared by successive fused rounds.
///
/// `emit_blocks` may retain values from two rounds earlier. Every active cell
/// must therefore be written before it is read or contracted; debug builds
/// poison the resized buffers to make violations visible.
#[derive(Default)]
struct LaneBuffers {
    core: Vec<E>,
    digits: Vec<E>,
    high: Vec<E>,
    slack: Vec<E>,
    budget: Vec<E>,
}

struct EqualityStates {
    sign: GruenSplitEq<E>,
    coefficient_norm: GruenSplitEq<E>,
    slack_norm: GruenSplitEq<E>,
    high_budget: GruenSplitEq<E>,
    signature_budget: GruenSplitEq<E>,
    s1_reduction: GruenSplitEq<E>,
    slack_reduction: GruenSplitEq<E>,
}

#[derive(Clone, Copy)]
struct EqView<'a> {
    first: &'a [E],
    second: &'a [E],
    first_mask: usize,
    first_shift: u32,
    tau: E,
    scalar: E,
}

fn initial_sign_coefficients(
    sigma: SignSource<'_>,
    first: &[E],
    second: &[E],
    endpoint: bool,
    #[cfg(any(test, feature = "internals"))] sigma_is_sign: bool,
) -> [E; 3] {
    #[cfg(any(test, feature = "internals"))]
    if !sigma_is_sign {
        return weighted_integer_coefficients(first, second, |pair| {
            let low = i64::from(sigma.get(2 * pair));
            let delta = i64::from(sigma.get(2 * pair + 1)) - low;
            [
                low * low - 1,
                if endpoint { 2 * low * delta } else { 0 },
                delta * delta,
            ]
        });
    }
    let [flips] = weighted_integer_coefficients(first, second, |pair| {
        [i64::from(sigma.get(2 * pair) != sigma.get(2 * pair + 1))]
    });
    let four_flips = E::from_u64(4) * flips;
    [
        E::zero(),
        if endpoint { -four_flips } else { E::zero() },
        four_flips,
    ]
}

fn initial_s1_coefficients(s1: &[i16], first: &[E], second: &[E], endpoint: bool) -> [E; 3] {
    if endpoint {
        let [constant, linear] = weighted_integer_coefficients(first, second, |pair| {
            let low = i64::from(s1[2 * pair]);
            [low, i64::from(s1[2 * pair + 1]) - low]
        });
        [constant, linear, E::zero()]
    } else {
        let [constant] =
            weighted_integer_coefficients(first, second, |pair| [i64::from(s1[2 * pair])]);
        [constant, E::zero(), E::zero()]
    }
}

impl<'a> EqView<'a> {
    fn new(eq: &'a GruenSplitEq<E>, scale: E) -> Self {
        let (first, second) = eq.remaining_eq_tables();
        debug_assert!(first.len().is_power_of_two());
        Self {
            first,
            second,
            first_mask: first.len() - 1,
            first_shift: first.len().trailing_zeros(),
            tau: eq.current_tau(),
            scalar: eq.current_scalar() * scale,
        }
    }
    fn endpoint(self) -> bool {
        self.tau.is_zero() || self.tau == E::one()
    }
}

/// Small-integer sources remain raw through two binds; round three materializes
/// the core, while the high digit is only ever read from raw or handed-off digits.
/// Scalar claims and equality states have no relation-local table traversals.
pub(super) struct Engine<'a> {
    raw_s1: &'a [i16],
    raw_sigma: SignSource<'a>,
    #[cfg(any(test, feature = "internals"))]
    sigma_is_sign: bool,
    raw_magnitude: Vec<i16>,
    raw_high: DigitSource<'a>,
    raw_slack: &'a [i16],
    core: Vec<E>,
    ring_lane: Vec<E>,
    raw_plans: Option<[LiftedFoldPlan; 4]>,
    second_context: Option<<E as Fold>::Ctx>,
    slack: Vec<E>,
    digits: Vec<E>,
    budget: Vec<E>,
    spare: LaneBuffers,
    ring: ring_functional::FusedState,
    range: range_source_link::Provers<'a>,
    equalities: EqualityStates,
    claims: [E; 10],
    messages: Messages,
    row_weights: Vec<E>,
    gamma: [E; 12],
    coefficient_vars: usize,
    rounds: usize,
    round: usize,
}

impl<'a> Engine<'a> {
    pub(super) fn new(
        batch: &'a FalconWitness,
        tables: &'a SourceTables,
        coefficients: &PreparedCoefficients<'a>,
        stage1: &Stage1Output,
        stage2: &Stage2Output,
        gamma: [E; 12],
        prepared: super::Prepared,
    ) -> Result<Self, AggregationError> {
        let degree = FalconDegree::from_degree(batch.degree())?;
        let coefficient_vars = degree.logn();
        if batch.s1.len() != tables.a0_view().len() || batch.slack.len() != 4 * batch.padded_rows {
            return Err(AggregationError::internal(
                "stage-three witness source shape mismatch",
            ));
        }
        let row_point = &stage2.point[coefficient_vars..];
        let row_weights = eq_table(row_point);
        let raw_magnitude = prepared.magnitude;
        let [slack_claim] = weighted_integer_coefficients(&row_weights, &[E::one()], |row| {
            [batch.slack[4 * row..4 * row + 4]
                .iter()
                .map(|&x| i64::from(x).pow(2))
                .sum()]
        });
        let [budget_claim] = weighted_integer_coefficients(&row_weights, &[E::one()], |row| {
            [i64::from(tables.u0()[row])
                + 8 * i64::from(tables.u1()[row])
                + 64 * i64::from(tables.u2()[row])]
        });
        let ring_message = prepared.ring.initial_message(gamma[9]);
        let ring = prepared.ring.into_fused(gamma[9]);
        let range = range_source_link::Provers::new(
            tables,
            coefficients,
            degree,
            stage2,
            &gamma[..6].try_into().expect("six weights"),
        )?;
        let claims = [
            range.coefficient.claim(),
            range.budget.claim(),
            E::zero(),
            gamma[7]
                * (E::from_u64(degree.beta_squared())
                    * live_prefix_evaluation(batch.live_rows, row_point)
                    - slack_claim),
            gamma[7] * slack_claim,
            gamma[8] * (encoding_budget::input_claim(degree) - budget_claim),
            gamma[8] * budget_claim,
            ring.claim(),
            gamma[10] * stage1.projection_claims[0],
            gamma[11] * stage1.projection_claims[1],
        ];
        let equalities = EqualityStates {
            sign: GruenSplitEq::with_initial_scalar(&stage2.point, gamma[6])?,
            coefficient_norm: GruenSplitEq::with_initial_scalar(row_point, gamma[7])?,
            slack_norm: GruenSplitEq::with_initial_scalar(row_point, gamma[7])?,
            high_budget: GruenSplitEq::with_initial_scalar(row_point, gamma[8])?,
            signature_budget: GruenSplitEq::with_initial_scalar(row_point, gamma[8])?,
            s1_reduction: GruenSplitEq::with_initial_scalar(&stage1.projection_point, gamma[10])?,
            slack_reduction: GruenSplitEq::with_initial_scalar(
                &stage1.slack_projection_point,
                gamma[11],
            )?,
        };
        let mut engine = Self {
            raw_s1: &batch.s1,
            raw_sigma: tables.sigma_view(),
            #[cfg(any(test, feature = "internals"))]
            sigma_is_sign: coefficients.sigma_is_sign(),
            raw_magnitude,
            raw_high: tables.a2_view(),
            raw_slack: &batch.slack,
            core: Vec::new(),
            ring_lane: Vec::new(),
            raw_plans: None,
            second_context: None,
            slack: Vec::new(),
            digits: Vec::new(),
            budget: Vec::new(),
            spare: LaneBuffers::default(),
            ring,
            range,
            equalities,
            claims,
            messages: [[E::zero(); 3]; 10],
            row_weights,
            gamma,
            coefficient_vars,
            rounds: stage1.point.len(),
            round: 0,
        };
        tracing::info_span!("stage3_compact_round0")
            .in_scope(|| engine.initial_round(ring_message))?;
        Ok(engine)
    }

    fn initial_round(&mut self, ring_message: [E; 3]) -> Result<(), AggregationError> {
        self.messages[0] = self.range.coefficient.round_compact()?;
        for id in [1, 4, 6, 9] {
            self.messages[id] = inactive_round(self.claims[id]);
        }
        self.messages[7] = ring_message;
        let sign_view = EqView::new(&self.equalities.sign, E::one());
        let sign_inner = initial_sign_coefficients(
            self.raw_sigma,
            sign_view.first,
            sign_view.second,
            sign_view.endpoint(),
            #[cfg(any(test, feature = "internals"))]
            self.sigma_is_sign,
        );
        self.messages[2] =
            equality_weighted_cubic(self.claims[2], sign_view.tau, sign_view.scalar, sign_inner);

        let s1_view = EqView::new(&self.equalities.s1_reduction, E::one());
        let s1_inner = initial_s1_coefficients(
            self.raw_s1,
            s1_view.first,
            s1_view.second,
            s1_view.endpoint(),
        );
        self.messages[8] =
            equality_weighted_cubic(self.claims[8], s1_view.tau, s1_view.scalar, s1_inner);
        let width = 1 << self.coefficient_vars;
        let [constant, quadratic, high] =
            weighted_integer_coefficients(&self.row_weights, &[E::one()], |row| {
                let mut sums = [0_i64; 3];
                for pair in 0..width / 2 {
                    let low = row * width + 2 * pair;
                    let s = i64::from(self.raw_s1[low]);
                    let ds = i64::from(self.raw_s1[low + 1]) - s;
                    let m = i64::from(self.raw_magnitude[low]);
                    let dm = i64::from(self.raw_magnitude[low + 1]) - m;
                    sums[0] += s * s + m * m;
                    sums[1] += ds * ds + dm * dm;
                    sums[2] += i64::from(self.raw_high.get(low)) + 8;
                }
                sums
            });
        self.messages[3] = [
            self.gamma[7] * constant,
            self.gamma[7] * quadratic,
            E::zero(),
        ];
        self.messages[5] = [self.gamma[8] * high, E::zero(), E::zero()];
        Ok(())
    }

    pub(super) fn message(&self) -> [E; 3] {
        let mut result = [E::zero(); 3];
        for part in self.messages {
            for (sum, value) in result.iter_mut().zip(part) {
                *sum += value;
            }
        }
        result
    }
    pub(super) fn claim(&self) -> E {
        self.claims.iter().copied().sum()
    }
    pub(super) fn terminal_evaluations(&self) -> [E; 9] {
        assert_eq!(self.round, self.rounds);
        [
            self.core[0],
            self.slack[0],
            self.digits[0],
            self.digits[1],
            self.digits[2],
            self.core[2],
            self.budget[0],
            self.budget[1],
            self.budget[2],
        ]
    }

    pub(super) fn bind(&mut self, challenge: E) -> Result<(), AggregationError> {
        let previous = self.round;
        for id in 0..10 {
            self.claims[id] = evaluate_compressed(self.messages[id], self.claims[id], challenge);
        }
        self.ring.bind_metadata(self.messages[7], challenge);
        self.equalities.sign.bind(challenge);
        self.equalities.s1_reduction.bind(challenge);
        if previous >= self.coefficient_vars {
            self.equalities.coefficient_norm.bind(challenge);
            self.equalities.slack_norm.bind(challenge);
            self.equalities.high_budget.bind(challenge);
            self.equalities.signature_budget.bind(challenge);
        }
        if previous >= self.coefficient_vars - 2 {
            self.equalities.slack_reduction.bind(challenge);
        }
        let digit_handoff = if self.range.coefficient.is_compact() {
            self.range
                .coefficient
                .bind_compact(self.messages[0], challenge)?
        } else {
            self.range
                .coefficient
                .bind_metadata(self.messages[0], challenge);
            None
        };
        let budget_handoff = if previous < self.coefficient_vars {
            self.range.budget.halve_inactive();
            None
        } else if self.range.budget.is_compact() {
            self.range
                .budget
                .bind_compact(self.messages[1], challenge)?
        } else {
            self.range.budget.bind_metadata(self.messages[1], challenge);
            None
        };
        self.round += 1;
        self.sweep(challenge, digit_handoff, budget_handoff)?;
        Ok(())
    }

    fn views(&self) -> [Option<EqView<'_>>; 10] {
        let mut views = [None; 10];
        if self.round == self.rounds {
            return views;
        }
        if !self.range.coefficient.is_compact() {
            views[0] = Some(EqView::new(
                &self.range.coefficient.equality,
                self.range.coefficient.scale,
            ));
        }
        if self.round >= self.coefficient_vars && !self.range.budget.is_compact() {
            views[1] = Some(EqView::new(
                &self.range.budget.equality,
                self.range.budget.scale,
            ));
        }
        views[2] = Some(EqView::new(&self.equalities.sign, E::one()));
        views[8] = Some(EqView::new(&self.equalities.s1_reduction, E::one()));
        if self.round >= self.coefficient_vars {
            views[3] = Some(EqView::new(&self.equalities.coefficient_norm, E::one()));
            views[5] = Some(EqView::new(&self.equalities.high_budget, E::one()));
            views[4] = Some(EqView::new(
                &self.equalities.slack_norm,
                super::inverse_power_of_two(self.coefficient_vars - 2),
            ));
            views[6] = Some(EqView::new(
                &self.equalities.signature_budget,
                super::inverse_power_of_two(self.coefficient_vars),
            ));
        }
        if self.round >= self.coefficient_vars - 2 {
            views[9] = Some(EqView::new(
                &self.equalities.slack_reduction,
                super::inverse_power_of_two(self.coefficient_vars - 2),
            ));
        }
        views
    }

    fn sweep(
        &mut self,
        challenge: E,
        digit_handoff: Option<range_source_link::DenseHandoff>,
        budget_handoff: Option<range_source_link::DenseHandoff>,
    ) -> Result<(), AggregationError> {
        let (spare, round) =
            self.setup_round(challenge, digit_handoff.as_ref(), budget_handoff.as_ref());
        let contraction =
            tracing::info_span!("stage3_fused_contract", round = round.round).entered();
        let output = round.contract(spare);
        let sums = output.sums.map(|part| part.map(E::reduce_product));
        drop(contraction);
        let finalization = round.equality_messages(&sums);
        self.finish_round(output, &sums, &finalization, digit_handoff, budget_handoff)
    }

    fn setup_round<'s>(
        &'s mut self,
        challenge: E,
        digit_handoff: Option<&'s range_source_link::DenseHandoff>,
        budget_handoff: Option<&'s range_source_link::DenseHandoff>,
    ) -> (LaneBuffers, RoundContext<'s, 'a>) {
        let round = self.round;
        let context = E::precompute(challenge);
        if round <= 3 {
            debug_assert!(self.coefficient_vars >= 9);
            debug_assert!(self.rounds > self.coefficient_vars);
            debug_assert!(round < self.coefficient_vars - 2);
        }
        if round == 1 {
            let mut weights = std::mem::take(&mut self.ring.weights);
            fold_rows_in_place(&mut weights, 1, challenge, "stage3_ring");
            weights.shrink_to_fit();
            self.ring_lane = weights;
            self.raw_plans = Some([
                LiftedFoldPlan::new(self.raw_s1, challenge),
                LiftedFoldPlan::new(&self.raw_magnitude, challenge),
                LiftedFoldPlan::new_sign(self.raw_sigma, challenge),
                LiftedFoldPlan::new_digit(self.raw_high, challenge),
            ]);
        } else if round == 2 {
            self.second_context = Some(context);
        }
        let spare = std::mem::take(&mut self.spare);
        (
            spare,
            RoundContext::new(self, context, challenge, digit_handoff, budget_handoff),
        )
    }

    #[inline]
    fn finish_round(
        &mut self,
        output: BlockOutput,
        sums: &Messages,
        finalization: &RoundFinalization,
        digit_handoff: Option<range_source_link::DenseHandoff>,
        budget_handoff: Option<range_source_link::DenseHandoff>,
    ) -> Result<(), AggregationError> {
        let &RoundFinalization {
            round,
            terminal,
            high_active,
            digits_active,
            budget_active,
            transition,
            digit_coefficients,
            budget_coefficients,
            mut messages,
        } = finalization;
        let next_digits = if digits_active {
            digit_handoff
                .map(|handoff| handoff.sources)
                .unwrap_or(output.digits)
        } else {
            Vec::new()
        };
        let next_budget = if budget_active {
            budget_handoff
                .map(|handoff| handoff.sources)
                .unwrap_or(output.budget)
        } else {
            Vec::new()
        };
        let old_core = if round == 1 {
            output.core
        } else if high_active {
            std::mem::replace(&mut self.ring_lane, output.core)
        } else {
            let old_core = std::mem::replace(&mut self.core, output.core);
            if round == 3 {
                std::mem::take(&mut self.ring_lane)
            } else {
                old_core
            }
        };
        let old_digits = std::mem::replace(&mut self.digits, next_digits);
        let old_slack = std::mem::replace(&mut self.slack, output.slack);
        let old_budget = std::mem::replace(&mut self.budget, next_budget);
        self.spare = LaneBuffers {
            core: old_core,
            digits: old_digits,
            high: output.high,
            slack: old_slack,
            budget: old_budget,
        };
        if round == 3 {
            self.raw_magnitude = Vec::new();
            self.raw_plans = None;
            self.second_context = None;
        }
        if transition {
            self.ring.row_scale = None;
        }
        if !terminal {
            messages[0] = if digits_active {
                self.range
                    .coefficient
                    .dense_message(digit_coefficients.unwrap_or([sums[0][0], sums[0][1]]))
            } else {
                self.range.coefficient.round_compact()?
            };
            messages[1] = if round < self.coefficient_vars {
                inactive_round(self.claims[1])
            } else if budget_active {
                self.range
                    .budget
                    .dense_message(budget_coefficients.unwrap_or([sums[1][0], sums[1][1]]))
            } else {
                let view = EqView::new(
                    &self.equalities.signature_budget,
                    super::inverse_power_of_two(self.coefficient_vars),
                );
                let (message, [constant, linear]) = self.range.budget.round_compact_with_linear(
                    view.first,
                    view.second,
                    &[E::one(), E::from_u64(8), E::from_u64(64)],
                    &[0; 3],
                )?;
                // Replaces the dense-view message from `equality_messages`.
                messages[6] = equality_weighted_cubic(
                    self.claims[6],
                    view.tau,
                    view.scalar,
                    [constant, linear, E::zero()],
                );
                message
            };
            self.messages = messages;
        }
        Ok(())
    }
}

/// Shared inputs and lookup state for one round's disjoint chunk contractions.
/// Independent phase flags select lane writes and contraction dispatch.
#[allow(clippy::struct_excessive_bools)]
struct RoundContext<'s, 'a> {
    engine: &'s Engine<'a>,
    context: <E as Fold>::Ctx,
    round: usize,
    cells: usize,
    terminal: bool,
    views: [Option<EqView<'s>>; 10],
    chunk_cells: usize,
    slack_plan: Option<LiftedFoldPlan>,
    digit_handoff: Option<&'s range_source_link::DenseHandoff>,
    budget_handoff: Option<&'s range_source_link::DenseHandoff>,
    digit_coefficients: Option<[E; 2]>,
    budget_coefficients: Option<[E; 2]>,
    digit_ready: bool,
    budget_ready: bool,
    digits_active: bool,
    budget_active: bool,
    range_weights: [Vec<[E; 3]>; 2],
    high_active: bool,
    slack_active: bool,
    row_pairs: usize,
    transition: bool,
    equality_masks: [usize; 10],
    equality_mask_count: usize,
    sign_width: usize,
    sign_lut: Option<(Vec<E>, Vec<[E; 3]>, bool)>,
}

/// Owned coefficients and phase flags for message completion and lane recycling.
#[allow(clippy::struct_excessive_bools)]
struct RoundFinalization {
    round: usize,
    terminal: bool,
    high_active: bool,
    digits_active: bool,
    budget_active: bool,
    transition: bool,
    digit_coefficients: Option<[E; 2]>,
    budget_coefficients: Option<[E; 2]>,
    messages: Messages,
}

impl<'s, 'a> RoundContext<'s, 'a> {
    #[allow(clippy::too_many_lines)]
    fn new(
        engine: &'s Engine<'a>,
        context: <E as Fold>::Ctx,
        challenge: E,
        digit_handoff: Option<&'s range_source_link::DenseHandoff>,
        budget_handoff: Option<&'s range_source_link::DenseHandoff>,
    ) -> Self {
        let round = engine.round;
        let cells = engine.raw_s1.len() >> round;
        let terminal = round == engine.rounds;
        let views = engine.views();
        // Every supported coefficient row fits this cache-resident block after
        // the first bind. Larger equality factors are reduced in partial blocks.
        let chunk_cells = CORE_CHUNK_CELLS.min(cells);
        let slack_plan = (round == engine.coefficient_vars - 1)
            .then(|| LiftedFoldPlan::new(engine.raw_slack, challenge));
        let digit_coefficients = digit_handoff
            .as_ref()
            .and_then(|handoff| handoff.first_dense_coefficients);
        let budget_coefficients = budget_handoff
            .as_ref()
            .and_then(|handoff| handoff.first_dense_coefficients);
        let digit_ready = digit_handoff.is_some();
        let budget_ready = budget_handoff.is_some();
        let digits_active = round >= 3;
        let budget_active = !engine.range.budget.is_compact() && round >= engine.coefficient_vars;
        let range_weights: [Vec<[E; 3]>; 2] = std::array::from_fn(|id| {
            let precontracted = if id == 0 {
                digit_coefficients.is_some()
            } else {
                budget_coefficients.is_some()
            };
            views[id]
                .filter(|_| !precontracted)
                .map_or_else(Vec::new, |view| {
                    view.first
                        .iter()
                        .map(|&weight| {
                            std::array::from_fn(|lane| weight * engine.gamma[3 * id + lane])
                        })
                        .collect()
                })
        });
        let high_active = round < 3;
        let slack_active = round > engine.coefficient_vars - 2;
        let row_pairs = if round < engine.coefficient_vars {
            1 << (engine.coefficient_vars - round - 1)
        } else {
            1
        };
        debug_assert!(terminal || chunk_cells.is_multiple_of(2 * row_pairs));
        let transition = round == engine.coefficient_vars;
        // Relations created over the same original domain share one split
        // boundary. Gather the few distinct boundaries once per round rather
        // than testing every relation on every pair.
        let mut equality_masks = [0_usize; 10];
        let mut equality_mask_count = 0;
        for view in views.iter().flatten() {
            if !equality_masks[..equality_mask_count].contains(&view.first_mask) {
                equality_masks[equality_mask_count] = view.first_mask;
                equality_mask_count += 1;
            }
        }
        // While sigma is a genuine sign, a folded sign cell depends only on the
        // bit pattern of its raw block, and the sign relation's pair
        // coefficients only on two such patterns.
        let sign_width = 1_usize << round;
        debug_assert!(!terminal || !high_active);
        let use_sign_lut = high_active;
        #[cfg(any(test, feature = "internals"))]
        let use_sign_lut = use_sign_lut && engine.sigma_is_sign;
        let sign_lut = use_sign_lut.then(|| {
            let plans = engine
                .raw_plans
                .as_ref()
                .expect("raw plans live until round three");
            let second = engine.second_context.as_ref().unwrap_or(&context);
            let values: Vec<E> = (0..1_usize << sign_width)
                .map(|pattern| {
                    fold_raw_block(&plans[2], round, second, &context, |i| {
                        1 - 2 * ((pattern >> i) & 1) as i64
                    })
                })
                .collect();
            let linear = views[2].unwrap().endpoint();
            let coefficients: Vec<[E; 3]> = (0..values.len() * values.len())
                .map(|pair| {
                    let a = values[pair >> sign_width];
                    let delta = values[pair & (values.len() - 1)] - a;
                    [
                        a * a - E::one(),
                        if linear { (a + a) * delta } else { E::zero() },
                        delta * delta,
                    ]
                })
                .collect();
            (values, coefficients, linear)
        });
        Self {
            engine,
            context,
            round,
            cells,
            terminal,
            views,
            chunk_cells,
            slack_plan,
            digit_handoff,
            budget_handoff,
            digit_coefficients,
            budget_coefficients,
            digit_ready,
            budget_ready,
            digits_active,
            budget_active,
            range_weights,
            high_active,
            slack_active,
            row_pairs,
            transition,
            equality_masks,
            equality_mask_count,
            sign_width,
            sign_lut,
        }
    }

    #[inline]
    fn sign_pattern(&self, raw: usize) -> usize {
        let sign_width = self.sign_width;
        match self.engine.raw_sigma {
            SignSource::Bits(words) => {
                ((words[raw / 64] >> (raw % 64)) & ((1 << sign_width) - 1)) as usize
            }
            #[cfg(any(test, feature = "internals"))]
            SignSource::Direct(cells) => (0..sign_width)
                .map(|i| usize::from(cells[raw + i] < 0) << i)
                .sum(),
        }
    }

    #[expect(
        clippy::inline_always,
        reason = "Keep the chunk fold and contraction in the emitted block closure"
    )]
    #[inline(always)]
    #[allow(clippy::too_many_lines)]
    fn contract_chunk(
        &self,
        chunk: usize,
        core: &mut [E],
        digits: &mut [E],
        slack: &mut [E],
        budget: &mut [E],
    ) -> [[Product; 3]; 10] {
        let engine = self.engine;
        let context = &self.context;
        let round = self.round;
        let cells = self.cells;
        let terminal = self.terminal;
        let chunk_cells = self.chunk_cells;
        let digit_coefficients = self.digit_coefficients;
        let budget_coefficients = self.budget_coefficients;
        let digit_ready = self.digit_ready;
        let budget_ready = self.budget_ready;
        let digits_active = self.digits_active;
        let budget_active = self.budget_active;
        let high_active = self.high_active;
        let slack_active = self.slack_active;
        let row_pairs = self.row_pairs;
        let transition = self.transition;
        let equality_mask_count = self.equality_mask_count;
        let sign_width = self.sign_width;
        let digit_handoff = self.digit_handoff;
        let budget_handoff = self.budget_handoff;
        let views = &self.views;
        let slack_plan = &self.slack_plan;
        let range_weights = &self.range_weights;
        let equality_masks = &self.equality_masks;
        let sign_lut = &self.sign_lut;
        let start = chunk * chunk_cells;
        let mut inner = [[Product::zero(); 3]; 10];
        let mut sums = [[Product::zero(); 3]; 10];
        let mut high_sum = E::zero();
        let task_cells = chunk_cells.min(cells - start);
        let decoded = engine.raw_plans.as_ref().map(|plans| {
            let mut signs = [0_i8; 8 * CORE_CHUNK_CELLS];
            let mut highs = [0_i8; 4 * CORE_CHUNK_CELLS];
            if sign_lut.is_none() {
                engine
                    .raw_sigma
                    .decode_into(start << round, &mut signs[..task_cells << round]);
            }
            if high_active {
                engine
                    .raw_high
                    .decode_into(start << round, &mut highs[..task_cells << round]);
            }
            (plans, signs, highs)
        });
        let second_context = engine.second_context.as_ref().unwrap_or(context);
        let mut previous_values = [E::zero(); 4];
        let mut previous_high = E::zero();
        let mut sign_pair = 0;
        // Sign-relation equality weights summed per pair pattern, until
        // the pattern coefficients are applied at each split flush.
        let mut sign_weights = [E::zero(); 256];
        for local in 0..task_cells {
            let index = start + local;
            let old = 2 * index;
            let mut high_value = E::zero();
            let mut values = if let Some((plans, signs, highs)) = &decoded {
                let raw = index << round;
                let offset = local << round;
                if high_active {
                    high_value = fold_raw_block(&plans[3], round, second_context, context, |i| {
                        i64::from(highs[offset + i])
                    });
                }
                [
                    fold_raw_block(&plans[0], round, second_context, context, |i| {
                        i64::from(engine.raw_s1[raw + i])
                    }),
                    fold_raw_block(&plans[1], round, second_context, context, |i| {
                        i64::from(engine.raw_magnitude[raw + i])
                    }),
                    if let Some((sign_values, _, _)) = &sign_lut {
                        let pattern = self.sign_pattern(raw);
                        sign_pair =
                            ((sign_pair << sign_width) | pattern) & ((1 << (2 * sign_width)) - 1);
                        sign_values[pattern]
                    } else {
                        fold_raw_block(&plans[2], round, second_context, context, |i| {
                            i64::from(signs[offset + i])
                        })
                    },
                    if round == 1 {
                        engine.ring_lane[index]
                    } else {
                        E::fold_one(context, engine.ring_lane[old], engine.ring_lane[old + 1])
                    },
                ]
            } else {
                // Written out rather than `array::from_fn`: this is the
                // hottest fold of the sweep, and whether LLVM inlines the
                // `from_fn` closure adapter varies with the crate's
                // codegen-unit partition.
                let (low, high) = engine.core[4 * old..4 * old + 8].split_at(4);
                [
                    E::fold_one(context, low[0], high[0]),
                    E::fold_one(context, low[1], high[1]),
                    E::fold_one(context, low[2], high[2]),
                    E::fold_one(context, low[3], high[3]),
                ]
            };
            if transition {
                values[3] *= engine.ring.row_scale.as_ref().unwrap()[index];
            }
            if round == 2 {
                core[local] = values[3];
            } else if round > 2 {
                core[4 * local..4 * local + 4].copy_from_slice(&values);
            }
            if digits_active && !digit_ready {
                for lane in 0..3 {
                    digits[3 * local + lane] = E::fold_one(
                        context,
                        engine.digits[3 * old + lane],
                        engine.digits[3 * (old + 1) + lane],
                    );
                }
            }
            if budget_active && !budget_ready {
                for lane in 0..3 {
                    budget[3 * local + lane] = E::fold_one(
                        context,
                        engine.budget[3 * old + lane],
                        engine.budget[3 * (old + 1) + lane],
                    );
                }
            }
            if slack_active {
                slack[local] = if let Some(plan) = &slack_plan {
                    plan.fold_pair(
                        i64::from(engine.raw_slack[old]),
                        i64::from(engine.raw_slack[old + 1]),
                    )
                } else {
                    E::fold_one(context, engine.slack[old], engine.slack[old + 1])
                };
            }
            if terminal || local % 2 == 0 {
                if high_active {
                    previous_values = values;
                    previous_high = high_value;
                }
                continue;
            }
            let digits = if let Some(handoff) = &digit_handoff {
                &handoff.sources[3 * start..]
            } else {
                &digits[..]
            };
            let budget = if let Some(handoff) = &budget_handoff {
                &handoff.sources[3 * start..]
            } else {
                &budget[..]
            };
            let pair = index / 2;
            let low = local - 1;
            let a: [E; 4] = if high_active {
                previous_values
            } else {
                core[4 * low..4 * low + 4].try_into().unwrap()
            };
            let b: [E; 4] = values;
            let delta: [E; 4] = std::array::from_fn(|lane| b[lane] - a[lane]);
            let products = ring_functional::pair_products([a[3], b[3]], [a[2], b[2]], [a[1], b[1]]);
            for lane in 0..3 {
                inner[7][lane] += products[lane];
            }
            if round < engine.coefficient_vars {
                inner[3][0] += a[0].mul_unreduced(a[0]);
                inner[3][0] += a[1].mul_unreduced(a[1]);
                inner[3][2] += delta[0].mul_unreduced(delta[0]);
                inner[3][2] += delta[1].mul_unreduced(delta[1]);
            } else {
                let c0 = a[0] * a[0] + a[1] * a[1];
                let c2 = delta[0] * delta[0] + delta[1] * delta[1];
                let c1 = if views[3].unwrap().endpoint() {
                    E::from_u64(2) * (a[0] * delta[0] + a[1] * delta[1])
                } else {
                    E::zero()
                };
                add_eq(&mut inner, 3, pair, [c0, c1, c2], views[3].unwrap());
            }
            let sign_view = views[2].unwrap();
            if let Some((_, coefficients, linear)) = &sign_lut {
                sign_weights[sign_pair] += sign_view.first[pair & sign_view.first_mask];
                if (pair + 1) & sign_view.first_mask == 0 {
                    drain_sign_weights(&mut inner[2], &mut sign_weights, coefficients, *linear);
                }
            } else {
                let sign_linear = if sign_view.endpoint() {
                    (a[2] + a[2]) * delta[2]
                } else {
                    E::zero()
                };
                add_eq(
                    &mut inner,
                    2,
                    pair,
                    [a[2] * a[2] - E::one(), sign_linear, delta[2] * delta[2]],
                    sign_view,
                );
            }
            if views[8].unwrap().endpoint() {
                add_eq(&mut inner, 8, pair, [a[0], delta[0]], views[8].unwrap());
            } else {
                add_eq(&mut inner, 8, pair, [a[0]], views[8].unwrap());
            }
            let high_left = if digits_active {
                digits[3 * low + 2]
            } else {
                previous_high
            };
            let high_right = if digits_active {
                digits[3 * local + 2]
            } else {
                high_value
            };
            if round < engine.coefficient_vars {
                high_sum += high_left + E::from_u64(8);
            } else {
                add_eq(
                    &mut inner,
                    5,
                    pair,
                    [high_left + E::from_u64(8), high_right - high_left],
                    views[5].unwrap(),
                );
            }
            if digits_active && digit_coefficients.is_none() {
                add_range(
                    &mut inner[0],
                    &digits[3 * low..3 * low + 3],
                    &digits[3 * local..3 * local + 3],
                    &range_weights[0][pair & views[0].unwrap().first_mask],
                );
            }
            if budget_active {
                if budget_coefficients.is_none() {
                    add_range(
                        &mut inner[1],
                        &budget[3 * low..3 * low + 3],
                        &budget[3 * local..3 * local + 3],
                        &range_weights[1][pair & views[1].unwrap().first_mask],
                    );
                }
                let left = budget[3 * low]
                    + E::from_u64(8) * budget[3 * low + 1]
                    + E::from_u64(64) * budget[3 * low + 2];
                let right = budget[3 * local]
                    + E::from_u64(8) * budget[3 * local + 1]
                    + E::from_u64(64) * budget[3 * local + 2];
                add_eq(&mut inner, 6, pair, [left, right - left], views[6].unwrap());
            }
            if round >= engine.coefficient_vars - 2 {
                let (left, right) = if slack_active {
                    (slack[low], slack[local])
                } else {
                    (
                        E::from_i64(i64::from(engine.raw_slack[index - 1])),
                        E::from_i64(i64::from(engine.raw_slack[index])),
                    )
                };
                let ds = right - left;
                if round < engine.coefficient_vars {
                    inner[4][0] += left.mul_unreduced(left);
                    inner[4][2] += ds.mul_unreduced(ds);
                } else {
                    let linear = if views[4].unwrap().endpoint() {
                        (left + left) * ds
                    } else {
                        E::zero()
                    };
                    add_eq(
                        &mut inner,
                        4,
                        pair,
                        [left * left, linear, ds * ds],
                        views[4].unwrap(),
                    );
                }
                if views[9].unwrap().endpoint() {
                    add_eq(&mut inner, 9, pair, [left, ds], views[9].unwrap());
                } else {
                    add_eq(&mut inner, 9, pair, [left], views[9].unwrap());
                }
            }
            for &mask in &equality_masks[..equality_mask_count] {
                if (pair + 1) & mask == 0 {
                    flush_complete_equalities(views, &mut inner, &mut sums, pair, mask);
                }
            }
            if round < engine.coefficient_vars && (pair + 1) & (row_pairs - 1) == 0 {
                let row = pair >> row_pairs.trailing_zeros();
                for id in [3, 4, 7] {
                    let weight = if id == 7 {
                        engine.ring.row_scale.as_ref().unwrap()[row]
                    } else {
                        engine.row_weights[row]
                    };
                    for lane in 0..3 {
                        sums[id][lane] += weight.mul_unreduced(E::reduce_product(inner[id][lane]));
                        inner[id][lane] = Product::zero();
                    }
                }
                sums[5][0] += engine.row_weights[row].mul_unreduced(high_sum);
                high_sum = E::zero();
            }
        }
        if !terminal {
            let end_pair = start / 2 + task_cells / 2;
            if let Some((_, coefficients, linear)) = &sign_lut {
                if end_pair & views[2].unwrap().first_mask != 0 {
                    drain_sign_weights(&mut inner[2], &mut sign_weights, coefficients, *linear);
                }
            }
            flush_partial_equalities(views, &inner, &mut sums, end_pair);
        }
        if round >= engine.coefficient_vars {
            for lane in 0..3 {
                sums[7][lane] += inner[7][lane];
            }
        }
        sums
    }

    fn contract(&self, spare: LaneBuffers) -> BlockOutput {
        let round = self.round;
        let cells = self.cells;
        let chunk_cells = self.chunk_cells;
        let high_active = self.high_active;
        let digit_ready = self.digit_ready;
        let slack_active = self.slack_active;
        let budget_active = self.budget_active;
        let budget_ready = self.budget_ready;
        let run = |chunk,
                   core: &mut [E],
                   digits: &mut [E],
                   _high: &mut [E],
                   slack: &mut [E],
                   budget: &mut [E]| {
            self.contract_chunk(chunk, core, digits, slack, budget)
        };
        if round == 1 {
            emit_blocks::<0, 0, 0, 0, 0>(cells, chunk_cells, spare, run)
        } else if high_active {
            emit_blocks::<1, 0, 0, 0, 0>(cells, chunk_cells, spare, run)
        } else if digit_ready {
            // The compact materializer owns this already-folded output.
            emit_blocks::<4, 0, 0, 0, 0>(cells, chunk_cells, spare, run)
        } else if !slack_active {
            emit_blocks::<4, { 3 * CORE_CHUNK_CELLS }, 0, 0, 0>(cells, chunk_cells, spare, run)
        } else if budget_active && !budget_ready {
            emit_blocks::<4, { 3 * CORE_CHUNK_CELLS }, 0, CORE_CHUNK_CELLS, { 3 * CORE_CHUNK_CELLS }>(
                cells,
                chunk_cells,
                spare,
                run,
            )
        } else {
            emit_blocks::<4, { 3 * CORE_CHUNK_CELLS }, 0, CORE_CHUNK_CELLS, 0>(
                cells,
                chunk_cells,
                spare,
                run,
            )
        }
    }

    #[inline]
    /// Messages 2-9 from the contracted sums, with the flags `finish_round` needs
    /// to complete messages 0 and 1 and recycle the lanes.
    fn equality_messages(&self, sums: &Messages) -> RoundFinalization {
        let engine = self.engine;
        let round = self.round;
        let terminal = self.terminal;
        let views = &self.views;
        let mut messages = [[E::zero(); 3]; 10];
        if !terminal {
            for id in 0..10 {
                messages[id] = if id == 0 || id == 1 {
                    continue;
                } else if id == 7 {
                    sums[7].map(|coefficient| engine.ring.gamma * coefficient)
                } else if let Some(view) = views[id] {
                    equality_weighted_cubic(engine.claims[id], view.tau, view.scalar, sums[id])
                } else if id == 3 {
                    [
                        engine.gamma[7] * sums[3][0],
                        engine.gamma[7] * sums[3][2],
                        E::zero(),
                    ]
                } else if id == 5 {
                    [engine.gamma[8] * sums[5][0], E::zero(), E::zero()]
                } else if id == 4 && round >= engine.coefficient_vars - 2 {
                    let scalar =
                        engine.gamma[7] * super::inverse_power_of_two(engine.coefficient_vars - 2);
                    [scalar * sums[4][0], scalar * sums[4][2], E::zero()]
                } else {
                    inactive_round(engine.claims[id])
                };
            }
        }
        RoundFinalization {
            round,
            terminal,
            high_active: self.high_active,
            digits_active: self.digits_active,
            budget_active: self.budget_active,
            transition: self.transition,
            digit_coefficients: self.digit_coefficients,
            budget_coefficients: self.budget_coefficients,
            messages,
        }
    }
}

/// Fold a raw block in low-variable-first order, retaining the r1 lookup.
#[inline]
fn fold_raw_block(
    plan: &LiftedFoldPlan,
    round: usize,
    second: &<E as Fold>::Ctx,
    current: &<E as Fold>::Ctx,
    get: impl Fn(usize) -> i64,
) -> E {
    let left = plan.fold_pair(get(0), get(1));
    if round == 1 {
        return left;
    }
    let left = E::fold_one(second, left, plan.fold_pair(get(2), get(3)));
    if round == 2 {
        return left;
    }
    let right = E::fold_one(
        second,
        plan.fold_pair(get(4), get(5)),
        plan.fold_pair(get(6), get(7)),
    );
    E::fold_one(current, left, right)
}

/// Flush the equalities that share this complete first-factor boundary. The
/// pair loop tests each distinct split geometry once instead of testing every
/// relation.
#[inline]
fn flush_complete_equalities(
    views: &[Option<EqView<'_>>; 10],
    inner: &mut [[Product; 3]; 10],
    sums: &mut [[Product; 3]; 10],
    pair: usize,
    mask: usize,
) {
    for id in 0..10 {
        if let Some(view) = views[id] {
            if view.first_mask != mask {
                continue;
            }
            let outer = view.second[pair >> view.first_shift];
            for lane in 0..3 {
                sums[id][lane] += outer.mul_unreduced(E::reduce_product(inner[id][lane]));
                inner[id][lane] = Product::zero();
            }
        }
    }
}

struct BlockOutput {
    core: Vec<E>,
    digits: Vec<E>,
    high: Vec<E>,
    slack: Vec<E>,
    budget: Vec<E>,
    sums: [[Product; 3]; 10],
}

/// Emit only active lanes, without a separate table-wide initialization pass.
/// An already materialized handoff uses a zero-length output block and is reused.
fn emit_blocks<const C: usize, const D: usize, const H: usize, const S: usize, const B: usize>(
    cells: usize,
    chunk_cells: usize,
    buffers: LaneBuffers,
    run: impl Fn(usize, &mut [E], &mut [E], &mut [E], &mut [E], &mut [E]) -> [[Product; 3]; 10] + Sync,
) -> BlockOutput {
    let chunks = cells.div_ceil(chunk_cells);
    let d_lanes = D / CORE_CHUNK_CELLS;
    let h_lanes = H / CORE_CHUNK_CELLS;
    let s_lanes = S / CORE_CHUNK_CELLS;
    let b_lanes = B / CORE_CHUNK_CELLS;
    let resize = |mut values: Vec<E>, len: usize| {
        // Alternating buffers otherwise retain four times the next output.
        if values.capacity() > 2 * len {
            values = Vec::new();
        }
        values.resize(len, E::zero());
        #[cfg(debug_assertions)]
        values.fill(E::from_u64(0xdead));
        values
    };
    // Disabled lanes retain one dummy cell per task. This lets all five
    // mutable outputs use one safe zipped traversal without aliasing empty
    // slices; they are cleared before returning.
    let lane_len = |lanes: usize| if lanes == 0 { chunks } else { lanes * cells };
    let mut core = resize(buffers.core, lane_len(C));
    let mut digits = resize(buffers.digits, lane_len(d_lanes));
    let mut high = resize(buffers.high, lane_len(h_lanes));
    let mut slack = resize(buffers.slack, lane_len(s_lanes));
    let mut budget = resize(buffers.budget, lane_len(b_lanes));
    let lane_chunk = |lanes: usize| if lanes == 0 { 1 } else { lanes * chunk_cells };
    let tasks: Vec<_> = core
        .chunks_mut(lane_chunk(C))
        .zip(digits.chunks_mut(lane_chunk(d_lanes)))
        .zip(high.chunks_mut(lane_chunk(h_lanes)))
        .zip(slack.chunks_mut(lane_chunk(s_lanes)))
        .zip(budget.chunks_mut(lane_chunk(b_lanes)))
        .enumerate()
        .collect();
    let emit = |(chunk, ((((core, digits), high), slack), budget))| {
        run(chunk, core, digits, high, slack, budget)
    };
    let add = |mut left: [[Product; 3]; 10], right: [[Product; 3]; 10]| {
        for id in 0..10 {
            for lane in 0..3 {
                left[id][lane] += right[id][lane];
            }
        }
        left
    };
    #[cfg(feature = "parallel")]
    let sums = if chunks == 1 && !force_parallel_single_chunk() {
        emit(tasks.into_iter().next().expect("one output chunk"))
    } else {
        tasks
            .into_par_iter()
            .map(emit)
            .reduce(|| [[Product::zero(); 3]; 10], add)
    };
    #[cfg(not(feature = "parallel"))]
    let sums = tasks
        .into_iter()
        .map(emit)
        .fold([[Product::zero(); 3]; 10], add);
    let clear_dummy = |mut values: Vec<E>, lanes: usize| {
        if lanes == 0 {
            values.clear();
        }
        values
    };
    BlockOutput {
        core: clear_dummy(core, C),
        digits: clear_dummy(digits, d_lanes),
        high: clear_dummy(high, h_lanes),
        slack: clear_dummy(slack, s_lanes),
        budget: clear_dummy(budget, b_lanes),
        sums,
    }
}

/// Finish a task's incomplete first-factor blocks. Complete blocks were
/// already flushed by the pair loop, independently for each equality split.
#[inline]
fn flush_partial_equalities(
    views: &[Option<EqView<'_>>; 10],
    inner: &[[Product; 3]; 10],
    sums: &mut [[Product; 3]; 10],
    end_pair: usize,
) {
    for id in 0..10 {
        if let Some(view) = views[id] {
            if end_pair & view.first_mask != 0 {
                let outer = view.second[(end_pair - 1) >> view.first_shift];
                for lane in 0..3 {
                    sums[id][lane] += outer.mul_unreduced(E::reduce_product(inner[id][lane]));
                }
            }
        }
    }
}

fn drain_sign_weights(
    inner: &mut [Product; 3],
    weights: &mut [E; 256],
    coefficients: &[[E; 3]],
    linear: bool,
) {
    for (weight, coefficients) in weights.iter_mut().zip(coefficients) {
        let weight = std::mem::replace(weight, E::zero());
        inner[0] += weight.mul_unreduced(coefficients[0]);
        if linear {
            inner[1] += weight.mul_unreduced(coefficients[1]);
        }
        inner[2] += weight.mul_unreduced(coefficients[2]);
    }
}

/// Adds the leading `LANES` coefficients; the remaining lanes are zero.
#[inline]
fn add_eq<const LANES: usize>(
    inner: &mut [[Product; 3]; 10],
    id: usize,
    pair: usize,
    coefficients: [E; LANES],
    view: EqView<'_>,
) {
    let weight = view.first[pair & view.first_mask];
    for (sum, value) in inner[id].iter_mut().zip(coefficients) {
        *sum += weight.mul_unreduced(value);
    }
}
#[inline]
fn add_range(inner: &mut [Product; 3], left: &[E], right: &[E], weights: &[E; 3]) {
    // Source and first equality weights are combined once per round. Keep
    // each quadratic product unreduced until the split-factor contraction.
    const { assert!(E::SUM_IS_EXACT) };
    for lane in 0..3 {
        let delta = right[lane] - left[lane];
        let weighted_delta = weights[lane] * delta;
        inner[0] += weighted_delta.mul_unreduced((left[lane] + left[lane]) + E::one());
        inner[1] += weighted_delta.mul_unreduced(delta);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::protocol::algebra::{eq_evaluation, evaluate_mle, fold};
    use crate::substrate::BaseField as F;
    use aerie_falcon_native::Falcon1024;
    use aerie_falcon_witness::DirectRow;

    fn k(a: u64) -> E {
        E::new(F::from_u64(a + 3), F::from_u64(2 * a + 7))
    }

    #[test]
    fn sparse_initial_relations_match_direct_integer_coefficients() {
        let first = [k(1), k(2), k(3), k(4)];
        let second = [k(5), k(6)];
        let valid_sigma = [-1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1];
        let invalid_sigma = [0, 2, -2, 1, 1, 0, -1, 2, 2, -2, 0, -1, 1, 1, -2, 0];
        let s1 = [
            -37, 12, 0, 91, -5, -5, 1_024, -511, 7, -13, 99, 3, -1, 2, 8, -21,
        ];
        for tau in [E::zero(), E::one(), k(19)] {
            let endpoint = tau.is_zero() || tau == E::one();
            for sigma in [&valid_sigma[..], &invalid_sigma[..]] {
                let direct = weighted_integer_coefficients(&first, &second, |pair| {
                    let low = i64::from(sigma[2 * pair]);
                    let delta = i64::from(sigma[2 * pair + 1]) - low;
                    [
                        low * low - 1,
                        if endpoint { 2 * low * delta } else { 0 },
                        delta * delta,
                    ]
                });
                let sparse = initial_sign_coefficients(
                    SignSource::Direct(sigma),
                    &first,
                    &second,
                    endpoint,
                    sigma.iter().all(|&value| matches!(value, -1 | 1)),
                );
                assert_eq!(sparse, direct, "sigma at tau {tau:?}");
                assert_eq!(
                    equality_weighted_cubic(k(23), tau, k(29), sparse),
                    equality_weighted_cubic(k(23), tau, k(29), direct)
                );
            }

            let direct = weighted_integer_coefficients(&first, &second, |pair| {
                let low = i64::from(s1[2 * pair]);
                [
                    low,
                    if endpoint {
                        i64::from(s1[2 * pair + 1]) - low
                    } else {
                        0
                    },
                    0,
                ]
            });
            let sparse = initial_s1_coefficients(&s1, &first, &second, endpoint);
            assert_eq!(sparse, direct, "s1 at tau {tau:?}");
            assert_eq!(
                equality_weighted_cubic(k(31), tau, k(37), sparse),
                equality_weighted_cubic(k(31), tau, k(37), direct)
            );
        }
    }

    fn lift<C: Copy + Into<i64>>(raw: &[C]) -> Vec<E> {
        raw.iter().map(|&v| E::from_i64(v.into())).collect()
    }

    fn lift_digit(raw: DigitSource<'_>) -> Vec<E> {
        (0..raw.len())
            .map(|index| E::from_i64(i64::from(raw.get(index))))
            .collect()
    }

    fn lift_sign(raw: SignSource<'_>) -> Vec<E> {
        (0..raw.len())
            .map(|index| E::from_i64(i64::from(raw.get(index))))
            .collect()
    }

    /// Exercise the production pair accumulator with independent split factors.
    pub(in crate::protocol::stage3) fn contract_split_for_test(
        first: &[E],
        second: &[E],
        coefficients: impl Fn(usize) -> [E; 3],
    ) -> [E; 3] {
        assert!(first.len().is_power_of_two());
        let view = EqView {
            first,
            second,
            first_mask: first.len() - 1,
            first_shift: first.len().trailing_zeros(),
            tau: E::zero(),
            scalar: E::one(),
        };
        let mut total = [E::zero(); 3];
        for (block, &outer) in second.iter().enumerate() {
            let mut inner = [[Product::zero(); 3]; 10];
            for low in 0..first.len() {
                let pair = block * first.len() + low;
                add_eq(&mut inner, 0, pair, coefficients(pair), view);
            }
            for lane in 0..3 {
                total[lane] += outer * E::reduce_product(inner[0][lane]);
            }
        }
        total
    }

    struct Reference {
        // S1, M, sigma, W, a0, a1, a2, slack, u0, u1, u2.
        lanes: [Vec<E>; 11],
        stage1: Stage1Output,
        stage2: Stage2Output,
        gamma: [E; 12],
        initial: [E; 10],
        coefficient_vars: usize,
    }
    impl Reference {
        fn native_skip(&self, id: usize) -> usize {
            match id {
                1 | 6 => self.coefficient_vars,
                4 | 9 => self.coefficient_vars - 2,
                _ => 0,
            }
        }
        fn message(&self, id: usize, bound: &[E], x: E) -> E {
            let round = bound.len();
            let skip = self.native_skip(id);
            if round < skip {
                return self.initial[id] * super::super::inverse_power_of_two(round + 1);
            }
            let native_round = round - skip;
            let n = if matches!(id, 1 | 6) {
                self.lanes[8].len()
            } else if matches!(id, 4 | 9) {
                self.lanes[7].len()
            } else {
                self.lanes[0].len()
            };
            let row_phase = round >= self.coefficient_vars;
            let equality_point: Option<&[E]> = match id {
                0 | 2 => Some(&self.stage2.point),
                1 | 6 => Some(&self.stage2.point[self.coefficient_vars..]),
                3..=5 if row_phase => Some(&self.stage2.point[self.coefficient_vars..]),
                8 => Some(&self.stage1.projection_point),
                9 => Some(&self.stage1.slack_projection_point),
                _ => None,
            };
            let eq_round = if matches!(id, 3..=5) {
                round.saturating_sub(self.coefficient_vars)
            } else {
                native_round
            };
            let equality = equality_point.map(|tau| {
                let eq_bound = if matches!(id, 3..=5) {
                    &bound[self.coefficient_vars..]
                } else {
                    &bound[skip..]
                };
                let prefix = eq_evaluation(&tau[..eq_round], eq_bound);
                let coordinate =
                    E::one() - tau[eq_round] + x * (tau[eq_round] + tau[eq_round] - E::one());
                (prefix * coordinate, eq_table(&tau[eq_round + 1..]))
            });
            let rows = eq_table(&self.stage2.point[self.coefficient_vars..]);
            let row_pairs = if row_phase {
                1
            } else {
                1 << (self.coefficient_vars - round - 1)
            };
            let mut total = E::zero();
            for pair in 0..n / 2 {
                let at = |lane: usize| {
                    let a = self.lanes[lane][2 * pair];
                    a + x * (self.lanes[lane][2 * pair + 1] - a)
                };
                let value = match id {
                    0 | 1 => {
                        let start = if id == 0 { 4 } else { 8 };
                        let gamma_start = if id == 0 { 0 } else { 3 };
                        (0..3)
                            .map(|j| {
                                self.gamma[gamma_start + j]
                                    * at(start + j)
                                    * (at(start + j) + E::one())
                            })
                            .sum()
                    }
                    2 => self.gamma[6] * (at(2) * at(2) - E::one()),
                    3 => self.gamma[7] * (at(0) * at(0) + at(1) * at(1)),
                    4 => self.gamma[7] * at(7) * at(7),
                    5 => self.gamma[8] * (at(6) + E::from_u64(8)),
                    6 => {
                        self.gamma[8] * (at(8) + E::from_u64(8) * at(9) + E::from_u64(64) * at(10))
                    }
                    7 => self.gamma[9] * at(3) * at(2) * at(1),
                    8 => self.gamma[10] * at(0),
                    9 => self.gamma[11] * at(7),
                    _ => unreachable!(),
                };
                let weight = if let Some((prefix, suffix)) = &equality {
                    *prefix * suffix[pair]
                } else if matches!(id, 3..=5) {
                    rows[pair / row_pairs]
                } else {
                    E::one()
                };
                total += weight * value;
            }
            total * super::super::inverse_power_of_two(skip)
        }
        fn bind(&mut self, round: usize, challenge: E) {
            for lane in 0..11 {
                let skip = match lane {
                    7 => self.coefficient_vars - 2,
                    8..=10 => self.coefficient_vars,
                    _ => 0,
                };
                if round >= skip {
                    fold(&mut self.lanes[lane], challenge);
                }
            }
        }
    }

    /// Direct relation formulas exercise the integrated engine, including every
    /// handoff, equality endpoint, inactive prefix, and terminal source opening.
    pub(in crate::protocol::stage3) fn check_relations(
        ids: &[usize],
        degree: FalconDegree,
        records: usize,
        mode: usize,
    ) {
        let mut batch = match degree {
            FalconDegree::Falcon512 => FalconWitness::from_rows(
                &(0..records)
                    .map(|r| {
                        DirectRow::<aerie_falcon_native::Falcon512>::synthetic(r as u64 + 811)
                            .unwrap()
                    })
                    .collect::<Vec<_>>(),
            ),
            FalconDegree::Falcon1024 => FalconWitness::from_rows(
                &(0..records)
                    .map(|r| DirectRow::<Falcon1024>::synthetic(r as u64 + 811).unwrap())
                    .collect::<Vec<_>>(),
            ),
        };
        // Norm and reduction formulas must also handle the complete signed alphabet.
        for (i, value) in batch.s1.iter_mut().enumerate() {
            *value = [i16::MIN, i16::MAX, -8191, 0, 17][i % 5];
        }
        for (i, value) in batch.slack.iter_mut().enumerate() {
            *value = [i16::MIN, i16::MAX, 0, -23][i % 4];
        }
        batch.s1[..8].copy_from_slice(&[i16::MIN, i16::MAX, -5_833, 0, 8_191, -17, 64, 23]);
        batch.s2[..8].copy_from_slice(&[0, -2_047, -1, 2_046, 1_092, 1_093, -2, -2_045]);
        let mut tables = SourceTables::build(&batch).unwrap();
        if mode == 6 {
            for (index, sigma) in tables.sigma_mut().iter_mut().enumerate() {
                *sigma = [-3, 0, 2, 1, -1][index % 5];
            }
        }
        let coefficient_vars = degree.logn();
        let rounds = coefficient_vars + batch.padded_rows.trailing_zeros() as usize;
        let point = |seed: u64, count: usize| {
            (0..count)
                .map(|i| {
                    if mode == 4 {
                        E::zero()
                    } else if mode == 5 {
                        E::one()
                    } else if mode != 0 {
                        match (i + usize::from(mode == 3)) % 3 {
                            0 => E::zero(),
                            1 => E::one(),
                            _ => k(seed + i as u64),
                        }
                    } else {
                        k(seed + i as u64)
                    }
                })
                .collect::<Vec<_>>()
        };
        let stage1 = Stage1Output {
            point: point(11, rounds),
            basis16_claims: vec![],
            projection_point: point(31, rounds),
            slack_projection_point: point(51, rounds - coefficient_vars + 2),
            projection_claims: [E::zero(); 3],
        };
        let stage2 = Stage2Output {
            point: point(71, rounds),
            folded_digits: [E::zero(); 6],
        };
        let gamma = std::array::from_fn(|i| {
            if mode == 2 {
                E::zero()
            } else {
                k(101 + i as u64)
            }
        });
        let weights = eq_table(&stage1.projection_point[..coefficient_vars]);
        let row_weights = eq_table(&stage1.projection_point[coefficient_vars..]);
        let d = degree.degree();
        let mut ring = vec![E::zero(); batch.padded_rows * d];
        // Definition of the adjoint negacyclic product, independent of all
        // production table construction and transform code.
        for row in 0..batch.live_rows {
            for j in 0..d {
                for (i, &weight) in weights.iter().enumerate() {
                    let h = if i >= j {
                        i64::from(batch.h[row * d + i - j])
                    } else {
                        -i64::from(batch.h[row * d + d + i - j])
                    };
                    if h != 0 {
                        ring[row * d + j] += weight * E::from_i64(h);
                    }
                }
                ring[row * d + j] *= row_weights[row];
            }
        }
        let lanes = [
            lift(&batch.s1),
            lift(&norm_equality::magnitude_table(&tables)),
            lift_sign(tables.sigma_view()),
            ring,
            lift_digit(tables.a0_view()),
            lift_digit(tables.a1_view()),
            lift_digit(tables.a2_view()),
            lift(&batch.slack),
            lift(tables.u0()),
            lift(tables.u1()),
            lift(tables.u2()),
        ];
        let original = lanes.clone();
        let row_eq = eq_table(&stage2.point[coefficient_vars..]);
        let mut reference = Reference {
            lanes,
            stage1,
            stage2,
            gamma,
            initial: [E::zero(); 10],
            coefficient_vars,
        };
        reference.initial[1] = (0..3)
            .map(|j| {
                gamma[3 + j]
                    * evaluate_mle(
                        &reference.lanes[8 + j]
                            .iter()
                            .map(|&a| a * (a + E::one()))
                            .collect::<Vec<_>>(),
                        &reference.stage2.point[coefficient_vars..],
                    )
            })
            .sum();
        reference.initial[4] = gamma[7]
            * row_eq
                .iter()
                .enumerate()
                .map(|(r, &w)| {
                    w * reference.lanes[7][4 * r..4 * r + 4]
                        .iter()
                        .map(|&a| a * a)
                        .sum::<E>()
                })
                .sum::<E>();
        reference.initial[6] = gamma[8]
            * (0..batch.padded_rows)
                .map(|r| {
                    row_eq[r]
                        * (reference.lanes[8][r]
                            + E::from_u64(8) * reference.lanes[9][r]
                            + E::from_u64(64) * reference.lanes[10][r])
                })
                .sum::<E>();
        reference.initial[9] = gamma[11]
            * evaluate_mle(
                &reference.lanes[7],
                &reference.stage1.slack_projection_point,
            );
        for id in [0, 2, 3, 5, 7, 8] {
            reference.initial[id] =
                reference.message(id, &[], E::zero()) + reference.message(id, &[], E::one());
        }
        let prepared = super::super::prepare(&batch, &tables, &reference.stage1).unwrap();
        let coefficients = PreparedCoefficients::new(&tables).unwrap();
        let engine = Engine::new(
            &batch,
            &tables,
            &coefficients,
            &reference.stage1,
            &reference.stage2,
            gamma,
            prepared,
        );
        // Split equality requires at least one row variable. Larger batches
        // exercise the short budget handoffs.
        if records == 1 {
            let cause = engine
                .as_ref()
                .err()
                .and_then(|error| std::error::Error::source(error))
                .and_then(|source| source.downcast_ref::<akita_error::AkitaError>());
            assert!(matches!(
                cause,
                Some(akita_error::AkitaError::InvalidSize {
                    expected: 1,
                    actual: 0,
                })
            ));
            return;
        }
        let mut engine = engine.unwrap();
        engine.claims = reference.initial;
        engine.range.coefficient.claim = reference.initial[0];
        engine.range.budget.claim = reference.initial[1];
        engine.ring.claim = reference.initial[7];
        engine.initial_round(engine.messages[7]).unwrap();
        let mut bound = Vec::new();
        for round in 0..rounds {
            for &id in ids {
                for x in [E::zero(), E::one(), k(151)] {
                    assert_eq!(
                        evaluate_compressed(engine.messages[id], engine.claims[id], x),
                        reference.message(id, &bound, x),
                        "relation {id}, round {round}, degree {}, rows {records}, mode {mode}",
                        degree.degree()
                    );
                }
            }
            let challenge = if mode == 0 {
                k(181 + round as u64)
            } else {
                [E::zero(), E::one(), k(191)][(round + usize::from(mode == 3)) % 3]
            };
            let expected = ids
                .iter()
                .map(|&id| reference.message(id, &bound, challenge))
                .collect::<Vec<_>>();
            engine.bind(challenge).unwrap();
            for (&id, expected) in ids.iter().zip(expected) {
                assert_eq!(engine.claims[id], expected);
            }
            reference.bind(round, challenge);
            for (cell, values) in engine.core.chunks_exact(4).enumerate() {
                for (lane, value) in values.iter().take(3).enumerate() {
                    assert_eq!(*value, reference.lanes[lane][cell]);
                }
            }
            if round >= 2 {
                for (cell, values) in engine.digits.chunks_exact(3).enumerate() {
                    for (lane, value) in values.iter().enumerate() {
                        assert_eq!(*value, reference.lanes[lane + 4][cell]);
                    }
                }
            }
            bound.push(challenge);
        }
        let indices = [0, 7, 4, 5, 6, 2, 8, 9, 10];
        for (value, lane) in engine.terminal_evaluations().into_iter().zip(indices) {
            let skip = match lane {
                7 => coefficient_vars - 2,
                8..=10 => coefficient_vars,
                _ => 0,
            };
            assert_eq!(value, evaluate_mle(&original[lane], &bound[skip..]));
            assert_eq!(value, reference.lanes[lane][0]);
        }
    }

    #[test]
    fn deferred_raw_folds_match_all_relations_with_lookup_and_direct_signs() {
        // 32768 cells activate the small-alphabet lookup for signs and digits,
        // while the full i16 s1 alphabet still uses the arithmetic fallback.
        for mode in [0, 2, 6] {
            check_relations(
                &(0..10).collect::<Vec<_>>(),
                FalconDegree::Falcon512,
                64,
                mode,
            );
        }
    }

    #[test]
    fn partial_equality_blocks_match_direct_contractions() {
        check_relations(&[2, 8], FalconDegree::Falcon512, 2_048, 0);
    }

    #[test]
    fn partial_flush_uses_each_split_and_skips_exact_boundaries() {
        for point in [E::zero(), E::one(), k(41)] {
            let first_tables = [7, 9, 10].map(|n| eq_table(&vec![point; n]));
            let second_tables = [7, 9, 10].map(|_| eq_table(&[point; 3]));
            let mut views = [None; 10];
            for id in 0..3 {
                let first = &first_tables[id];
                views[id] = Some(EqView {
                    first,
                    second: &second_tables[id],
                    first_mask: first.len() - 1,
                    first_shift: first.len().trailing_zeros(),
                    tau: point,
                    scalar: E::one(),
                });
            }
            // Include a segment crossing complete blocks, outer factors above
            // zero, and endpoints exact for one split but partial for another.
            for (start, end) in [(0, 31), (97, 271), (128, 256), (513, 777), (897, 1024)] {
                let mut inner = [[Product::zero(); 3]; 10];
                let mut sums = [[Product::zero(); 3]; 10];
                let mut expected = [[E::zero(); 3]; 3];
                for pair in start..end {
                    let coefficients: [E; 3] =
                        std::array::from_fn(|lane| k((pair * 3 + lane) as u64));
                    for id in 0..3 {
                        let view = views[id].unwrap();
                        add_eq(&mut inner, id, pair, coefficients, view);
                        for lane in 0..3 {
                            expected[id][lane] += view.first[pair % view.first.len()]
                                * view.second[pair / view.first.len()]
                                * coefficients[lane];
                        }
                        if (pair + 1) & view.first_mask == 0 {
                            for lane in 0..3 {
                                sums[id][lane] += view.second[pair >> view.first_shift]
                                    .mul_unreduced(E::reduce_product(inner[id][lane]));
                                // Leave a nonzero sentinel at an exact segment
                                // boundary to detect an accidental second flush.
                                inner[id][lane] = if pair + 1 == end {
                                    k(7).mul_unreduced(k(13))
                                } else {
                                    Product::zero()
                                };
                            }
                        }
                    }
                }
                flush_partial_equalities(&views, &inner, &mut sums, end);
                for id in 0..3 {
                    assert_eq!(sums[id].map(E::reduce_product), expected[id]);
                }
            }
        }
    }

    #[test]
    fn integrated_rounds_match_all_relations_across_materialization_boundaries() {
        for degree in [FalconDegree::Falcon512, FalconDegree::Falcon1024] {
            for records in [1, 2, 4, 8, 17] {
                for mode in 0..3 {
                    check_relations(&(0..10).collect::<Vec<_>>(), degree, records, mode);
                }
            }
        }
    }
}
