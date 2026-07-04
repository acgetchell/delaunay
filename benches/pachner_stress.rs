#![forbid(unsafe_code)]

//! Stress benchmarks for the unified Pachner move API.
//!
//! This harness measures repeated accepted [`PachnerMove`] attempts on stable
//! PL-manifold fixtures. It complements `ci_performance_suite` by focusing on
//! the unified dispatch facade rather than the individual flip primitives.

use std::{
    env,
    fmt::{Display, Write as _},
    hint::black_box,
    num::{NonZeroUsize, TryFromIntError},
    sync::LazyLock,
    time::{Duration, Instant},
};

use criterion::{
    BatchSize, BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayTriangulationBuilder, RetryPolicy, TopologyGuarantee, Vertex,
    vertex,
};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{CoordinateRange, RobustKernel};
use delaunay::prelude::pachner::{
    EdgeKey, FacetHandle, FlipError, PachnerMove, PachnerMoveResult, PachnerMoves, PachnerProposal,
    RidgeHandle, SimplexKey, TriangleHandle, VertexKey,
};
use delaunay::prelude::triangulation::Triangulation;
use delaunay::try_vertices_from_points;
use markov_chain_monte_carlo::prelude::delayed::{
    Chain, ChainId, DelayedProposal, DelayedStep, DelayedStepError, Target, Trace, TraceRecorder,
    TraceStepOutcome,
};
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System, get_current_pid};

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, OrAbortWithContext, abort_benchmark};

#[path = "common/flip_fixtures.rs"]
#[expect(
    dead_code,
    reason = "shared fixture catalog is intentionally broader than this 4D stress target"
)]
mod flip_fixtures;
use flip_fixtures::STABLE_POINTS_4D;

#[path = "common/flip_workflows.rs"]
#[expect(
    dead_code,
    reason = "shared flip workflow module is reused by broader benchmark and test targets"
)]
mod flip_workflows;
use flip_workflows::{CandidateFilter, FlipTriangulation};

type MonteCarloTriangulation<const D: usize> = Triangulation<RobustKernel<f64>, (), (), D>;

static MOVES_PER_SAMPLE: LazyLock<NonZeroUsize> =
    LazyLock::new(|| nonzero_usize("MOVES_PER_SAMPLE", 256));
static MONTE_CARLO_ATTEMPTS: LazyLock<NonZeroUsize> =
    LazyLock::new(|| nonzero_usize("MONTE_CARLO_ATTEMPTS", 100_000));
const MONTE_CARLO_3D_VERTICES: usize = 10_000;
const MONTE_CARLO_4D_VERTICES: usize = 1_000;
static MONTE_CARLO_KEY_REFRESH_EVERY: LazyLock<NonZeroUsize> =
    LazyLock::new(|| nonzero_usize("MONTE_CARLO_KEY_REFRESH_EVERY", 256));
static MONTE_CARLO_RETRY_ATTEMPTS: LazyLock<NonZeroUsize> =
    LazyLock::new(|| nonzero_usize("MONTE_CARLO_RETRY_ATTEMPTS", 24));
static MONTE_CARLO_SAMPLE_SIZE: LazyLock<NonZeroUsize> =
    LazyLock::new(|| nonzero_usize("MONTE_CARLO_SAMPLE_SIZE", 10));
const MONTE_CARLO_TRACE_TAIL: usize = 32;
static MONTE_CARLO_VALIDATE_EVERY: LazyLock<NonZeroUsize> =
    LazyLock::new(|| nonzero_usize("MONTE_CARLO_VALIDATE_EVERY", 1_000));
static MONTE_CARLO_VERTEX_GROWTH_DIVISOR: LazyLock<NonZeroUsize> =
    LazyLock::new(|| nonzero_usize("MONTE_CARLO_VERTEX_GROWTH_DIVISOR", 10));
static MONTE_CARLO_VERTEX_SHRINK_DIVISOR: LazyLock<NonZeroUsize> =
    LazyLock::new(|| nonzero_usize("MONTE_CARLO_VERTEX_SHRINK_DIVISOR", 20));
const MONTE_CARLO_REPORT_ENV: &str = "DELAUNAY_PACHNER_STRESS_REPORT";

struct PachnerStressSetup {
    base_dt: FlipTriangulation<4>,
    simplex_key: SimplexKey,
    k1_vertex: Vertex<(), 4>,
    k1_remove_dt: FlipTriangulation<4>,
    k1_remove_vertex_key: VertexKey,
    facet: FacetHandle,
    k2_inverse_dt: FlipTriangulation<4>,
    k2_inverse_edge: EdgeKey,
    ridge: RidgeHandle,
    k3_inverse_dt: FlipTriangulation<4>,
    k3_inverse_triangle: TriangleHandle,
}

#[derive(Clone, Copy)]
struct MonteCarloConfig {
    label: &'static str,
    vertex_count: usize,
    move_attempts: NonZeroUsize,
    validate_every: NonZeroUsize,
    key_refresh_every: NonZeroUsize,
    min_vertex_count: usize,
    max_vertex_count: usize,
    seed: u64,
}

impl MonteCarloConfig {
    const fn move_attempts(self) -> usize {
        self.move_attempts.get()
    }

    const fn validate_every(self) -> usize {
        self.validate_every.get()
    }

    const fn key_refresh_every(self) -> usize {
        self.key_refresh_every.get()
    }
}

#[derive(Clone, Copy, Debug)]
struct MonteCarloReport {
    attempts: usize,
    accepted: usize,
    rejected: usize,
    candidate_misses: usize,
    proposal_rejections: usize,
    validations: usize,
    validation_nanos: u128,
    elapsed_nanos: u128,
    attempts_per_second: u128,
    final_vertices: usize,
    final_simplices: usize,
    start_rss_kib: u64,
    max_rss_kib: u64,
    final_rss_kib: u64,
}

struct MoveSampler {
    simplex_keys: Vec<SimplexKey>,
    vertex_keys: Vec<VertexKey>,
    facet_handles: Vec<FacetHandle>,
    edge_keys: Vec<EdgeKey>,
    ridge_handles: Vec<RidgeHandle>,
}

impl MoveSampler {
    /// Captures the current live key frontier used for randomized move proposals.
    fn from_triangulation<const D: usize>(dt: &MonteCarloTriangulation<D>) -> Self {
        let mut sampler = Self {
            simplex_keys: Vec::new(),
            vertex_keys: Vec::new(),
            facet_handles: Vec::new(),
            edge_keys: Vec::new(),
            ridge_handles: Vec::new(),
        };
        sampler.refresh(dt);
        sampler
    }

    /// Refreshes cached keys after enough accepted moves may have stale candidates.
    fn refresh<const D: usize>(&mut self, dt: &MonteCarloTriangulation<D>) {
        self.simplex_keys.clear();
        self.simplex_keys
            .extend(dt.simplices().map(|(simplex_key, _)| simplex_key));

        self.vertex_keys.clear();
        self.vertex_keys
            .extend(dt.vertices().map(|(vertex_key, _)| vertex_key));

        self.facet_handles.clear();
        self.facet_handles
            .extend(dt.facets().map(|facet| facet.or_abort().handle()));

        self.edge_keys.clear();
        self.edge_keys.extend(dt.edges());

        self.ridge_handles.clear();
        self.ridge_handles
            .extend(dt.ridge_handles().map(OrAbort::or_abort));
    }

    /// Selects a cached simplex key uniformly from the last refresh.
    fn random_simplex_key<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<SimplexKey> {
        random_cached(&self.simplex_keys, rng)
    }

    /// Selects a cached vertex key uniformly from the last refresh.
    fn random_vertex_key<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<VertexKey> {
        random_cached(&self.vertex_keys, rng)
    }

    /// Selects a cached facet handle uniformly from the last refresh.
    fn random_facet<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<FacetHandle> {
        random_cached(&self.facet_handles, rng)
    }

    /// Selects a cached edge key uniformly from the last refresh.
    fn random_edge<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<EdgeKey> {
        random_cached(&self.edge_keys, rng)
    }

    /// Selects a cached ridge handle uniformly from the last refresh.
    fn random_ridge<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<RidgeHandle> {
        random_cached(&self.ridge_handles, rng)
    }
}

/// Selects a cached proposal item uniformly while preserving empty-cache misses.
fn random_cached<T: Copy, R: Rng + ?Sized>(values: &[T], rng: &mut R) -> Option<T> {
    (!values.is_empty()).then(|| {
        let index = rng.random_range(0..values.len());
        values[index]
    })
}

/// Flat diagnostic target: successful planned Pachner moves accept with probability one.
struct FlatPachnerTarget;

impl<const D: usize> Target<MonteCarloTriangulation<D>> for FlatPachnerTarget {
    fn log_prob(&self, _state: &MonteCarloTriangulation<D>) -> f64 {
        0.0
    }
}

#[derive(Clone, Debug)]
struct PachnerChainPlan<const D: usize> {
    request: PachnerMove<(), D>,
    proposal: PachnerProposal<(), D>,
}

#[derive(Clone, Debug)]
enum PachnerStepInfo<const D: usize> {
    Proposed {
        request: PachnerMove<(), D>,
    },
    CandidateMiss,
    ProposalRejected {
        request: PachnerMove<(), D>,
        rejection: FlipError,
    },
}

struct PachnerProposalKernel<const D: usize> {
    config: MonteCarloConfig,
    sampler: MoveSampler,
    proposed_steps: usize,
    candidate_misses: usize,
    proposal_rejections: usize,
    last_request: Option<PachnerMove<(), D>>,
    last_result: Option<PachnerMoveResult<D>>,
    last_no_plan_info: Option<PachnerStepInfo<D>>,
}

impl<const D: usize> PachnerProposalKernel<D> {
    fn new(dt: &MonteCarloTriangulation<D>, config: MonteCarloConfig) -> Self {
        Self {
            config,
            sampler: MoveSampler::from_triangulation(dt),
            proposed_steps: 0,
            candidate_misses: 0,
            proposal_rejections: 0,
            last_request: None,
            last_result: None,
            last_no_plan_info: None,
        }
    }

    fn maybe_refresh(&mut self, dt: &MonteCarloTriangulation<D>) {
        if self
            .proposed_steps
            .is_multiple_of(self.config.key_refresh_every())
        {
            self.sampler.refresh(dt);
        }
    }
}

impl<const D: usize> DelayedProposal<MonteCarloTriangulation<D>> for PachnerProposalKernel<D> {
    type Plan = PachnerChainPlan<D>;
    type Info = PachnerStepInfo<D>;
    type Error = FlipError;

    fn propose_plan<R: Rng + ?Sized>(
        &mut self,
        state: &MonteCarloTriangulation<D>,
        rng: &mut R,
    ) -> Result<Option<Self::Plan>, Self::Error> {
        self.proposed_steps = self.proposed_steps.saturating_add(1);
        self.maybe_refresh(state);
        self.last_result = None;

        let Some(request) = random_pachner_move(state, &self.sampler, rng, self.config) else {
            self.candidate_misses = self.candidate_misses.saturating_add(1);
            self.last_request = None;
            self.last_no_plan_info = Some(PachnerStepInfo::CandidateMiss);
            return Ok(None);
        };

        self.last_request = Some(request);
        match state.propose_pachner(request) {
            Ok(proposal) => {
                self.last_no_plan_info = None;
                Ok(Some(PachnerChainPlan { request, proposal }))
            }
            Err(error) => {
                self.proposal_rejections = self.proposal_rejections.saturating_add(1);
                self.last_no_plan_info = Some(PachnerStepInfo::ProposalRejected {
                    request,
                    rejection: error,
                });
                Ok(None)
            }
        }
    }

    fn no_plan_info(&mut self) -> Option<Self::Info> {
        self.last_no_plan_info.take()
    }

    fn proposed_log_prob<T: Target<MonteCarloTriangulation<D>>>(
        &self,
        state: &MonteCarloTriangulation<D>,
        _plan: &Self::Plan,
        target: &T,
    ) -> Result<f64, Self::Error> {
        Ok(target.log_prob(state))
    }

    fn info(&self, plan: &Self::Plan) -> Self::Info {
        PachnerStepInfo::Proposed {
            request: plan.request,
        }
    }

    fn commit<R: Rng + ?Sized>(
        &mut self,
        state: &mut MonteCarloTriangulation<D>,
        plan: Self::Plan,
        _rng: &mut R,
    ) -> Result<(), Self::Error> {
        let result = plan.proposal.attempt_on(state)?;
        self.last_result = Some(result);
        Ok(())
    }
}

/// Reads a positive `usize` override, falling back to `default`.
fn configured_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

/// Reads a positive `usize` override, preserving the non-zero proof.
fn configured_nonzero_usize(name: &str, default: NonZeroUsize) -> NonZeroUsize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .and_then(NonZeroUsize::new)
        .unwrap_or(default)
}

/// Parses a benchmark-owned non-zero default.
fn nonzero_usize(name: &str, value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).or_abort(format_args!("{name} must be non-zero"))
}

/// Reads a `u64` override, falling back to `default`.
fn configured_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

/// Reads a case-specific override before the shared Monte Carlo override.
fn configured_case_usize(label: &str, field: &str, default: usize) -> usize {
    let case_name = format!(
        "DELAUNAY_PACHNER_STRESS_{field}_{}",
        label.to_ascii_uppercase()
    );
    let shared_name = format!("DELAUNAY_PACHNER_STRESS_{field}");
    configured_usize(&case_name, configured_usize(&shared_name, default))
}

/// Reads a case-specific positive override as a proof-bearing nonzero count.
fn configured_case_nonzero_usize(label: &str, field: &str, default: NonZeroUsize) -> NonZeroUsize {
    let case_name = format!(
        "DELAUNAY_PACHNER_STRESS_{field}_{}",
        label.to_ascii_uppercase()
    );
    let shared_name = format!("DELAUNAY_PACHNER_STRESS_{field}");
    configured_nonzero_usize(&case_name, configured_nonzero_usize(&shared_name, default))
}

/// Reads a case-specific seed override before the shared Monte Carlo seed.
fn configured_case_seed(label: &str, default: u64) -> u64 {
    let case_name = format!(
        "DELAUNAY_PACHNER_STRESS_SEED_{}",
        label.to_ascii_uppercase()
    );
    configured_u64(
        &case_name,
        configured_u64("DELAUNAY_PACHNER_STRESS_SEED", default),
    )
}

/// Returns whether Monte Carlo source and metric lines should be printed.
fn monte_carlo_report_enabled() -> bool {
    env::var_os(MONTE_CARLO_REPORT_ENV).is_some()
}

/// Builds the dimension-specific Monte Carlo stress configuration.
fn monte_carlo_config<const D: usize>(
    label: &'static str,
    default_vertices: usize,
    default_seed: u64,
) -> MonteCarloConfig {
    let vertex_count =
        configured_case_usize(label, "VERTICES", default_vertices).max(D.saturating_add(1));
    let move_attempts = configured_case_nonzero_usize(label, "ATTEMPTS", *MONTE_CARLO_ATTEMPTS);
    let validate_every =
        configured_case_nonzero_usize(label, "VALIDATE_EVERY", *MONTE_CARLO_VALIDATE_EVERY);
    let validate_every = NonZeroUsize::new(validate_every.get().min(move_attempts.get()))
        .or_abort(format_args!("clamped validation interval must be non-zero"));
    let key_refresh_every =
        configured_case_nonzero_usize(label, "KEY_REFRESH_EVERY", *MONTE_CARLO_KEY_REFRESH_EVERY);
    let growth_slack = (vertex_count / (*MONTE_CARLO_VERTEX_GROWTH_DIVISOR).get()).max(D + 1);
    let shrink_slack = vertex_count / (*MONTE_CARLO_VERTEX_SHRINK_DIVISOR).get();

    MonteCarloConfig {
        label,
        vertex_count,
        move_attempts,
        validate_every,
        key_refresh_every,
        min_vertex_count: vertex_count.saturating_sub(shrink_slack).max(D + 1),
        max_vertex_count: vertex_count.saturating_add(growth_slack),
        seed: configured_case_seed(label, default_seed),
    }
}

/// Returns the coordinate range used for Monte Carlo point clouds.
fn monte_carlo_bounds() -> CoordinateRange<f64> {
    CoordinateRange::try_new(0.0_f64, 1.0).or_abort()
}

/// Builds the initial randomized triangulation for one Monte Carlo stress case.
fn build_monte_carlo_dt<const D: usize>(
    config: MonteCarloConfig,
    emit_report: bool,
) -> MonteCarloTriangulation<D> {
    let points = generate_random_points_in_range_seeded::<D>(
        config.vertex_count,
        monte_carlo_bounds(),
        config.seed,
    )
    .or_abort();
    let vertices = try_vertices_from_points(&points).or_abort();
    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
        attempts: *MONTE_CARLO_RETRY_ATTEMPTS,
        base_seed: Some(config.seed ^ 0xC0DE_0253_C0DE_0253),
    });

    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .construction_options(options)
        .build_with_kernel(&RobustKernel::new())
        .or_abort();
    let tri = dt.into_triangulation();
    validate_monte_carlo_state(
        &tri,
        format_args!(
            "initial Monte Carlo state dimension={D} label={} seed={}",
            config.label, config.seed
        ),
    );
    if emit_report {
        println!(
            "pachner_stress_source dimension={D} label={} vertices={} simplices={} seed={}",
            config.label,
            tri.number_of_vertices(),
            tri.number_of_simplices(),
            config.seed
        );
    }
    tri
}

/// Validates the invariants Pachner moves are expected to preserve.
fn validate_monte_carlo_state<const D: usize>(
    dt: &MonteCarloTriangulation<D>,
    context: impl Display,
) {
    if let Err(error) = dt.validate() {
        abort_benchmark(format_args!(
            "{context}: topology validation failed: {error}"
        ));
    }
    if let Err(error) = dt.is_valid_embedding() {
        abort_benchmark(format_args!(
            "{context}: embedding validation failed: {error}"
        ));
    }
}

/// Return current process memory usage in KiB.
fn memory_usage_kib() -> u64 {
    let pid = get_current_pid().or_abort();
    let mut system = System::new_with_specifics(
        RefreshKind::nothing().with_processes(ProcessRefreshKind::nothing().with_memory()),
    );
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    system
        .process(pid)
        .map_or(0, |process| process.memory() / 1024)
}

/// Converts bounded diagnostic counters into trace-observable values.
fn trace_value(value: usize) -> f64 {
    f64::from(u32::try_from(value).or_abort())
}

/// Numeric observables recorded for each completed MCMC step.
fn monte_carlo_observables<const D: usize>(
    dt: &MonteCarloTriangulation<D>,
    proposal: &PachnerProposalKernel<D>,
) -> [f64; 4] {
    [
        trace_value(dt.number_of_vertices()),
        trace_value(dt.number_of_simplices()),
        trace_value(proposal.candidate_misses),
        trace_value(proposal.proposal_rejections),
    ]
}

/// Records a completed MCMC step in the shared trace format.
fn record_monte_carlo_step<const D: usize>(
    recorder: &mut TraceRecorder,
    chain: &Chain<MonteCarloTriangulation<D>>,
    proposal: &PachnerProposalKernel<D>,
    step: &DelayedStep<PachnerStepInfo<D>>,
) {
    recorder
        .record(
            chain,
            TraceStepOutcome::from(step),
            monte_carlo_observables(chain.state(), proposal),
        )
        .or_abort();
}

/// Formats the short proposal metadata attached to the last delayed step.
fn describe_step_info<const D: usize>(info: &PachnerStepInfo<D>) -> String {
    match info {
        PachnerStepInfo::Proposed { request } => format!("proposed request={request:?}"),
        PachnerStepInfo::CandidateMiss => String::from("candidate_miss"),
        PachnerStepInfo::ProposalRejected { request, rejection } => {
            format!("proposal_rejected request={request:?} rejection={rejection}")
        }
    }
}

/// Formats the tail of the MCMC trace for invariant-failure diagnostics.
fn trace_tail(trace: &Trace) -> String {
    let records = trace.records();
    let start = records.len().saturating_sub(MONTE_CARLO_TRACE_TAIL);
    let mut output = String::new();
    for record in &records[start..] {
        let values = record.observable_values();
        let vertices = values.first().copied().unwrap_or_default();
        let simplices = values.get(1).copied().unwrap_or_default();
        let candidate_misses = values.get(2).copied().unwrap_or_default();
        let proposal_rejections = values.get(3).copied().unwrap_or_default();
        let outcome = record.outcome();
        let _ = write!(
            &mut output,
            "step={} accepted={} proposed={} vertices={} simplices={} \
             candidate_misses={} proposal_rejections={}; ",
            record.step(),
            outcome.is_accepted(),
            outcome.had_proposal(),
            vertices,
            simplices,
            candidate_misses,
            proposal_rejections
        );
    }
    output
}

/// Builds a diagnostic validation context from chain and trace state.
fn monte_carlo_validation_context<const D: usize>(
    config: MonteCarloConfig,
    step: usize,
    chain: &Chain<MonteCarloTriangulation<D>>,
    proposal: &PachnerProposalKernel<D>,
    last_step: Option<&DelayedStep<PachnerStepInfo<D>>>,
    trace: &Trace,
) -> String {
    let chain_id = ChainId::new(0);
    let mut context = format!(
        "Monte Carlo validation dimension={D} label={} step={} attempts={} accepted={} \
         rejected={} candidate_misses={} proposal_rejections={} acceptance_rate={:.6} \
         last_request={:?} last_result={:?}",
        config.label,
        step,
        config.move_attempts(),
        chain.accepted(),
        chain.rejected(),
        proposal.candidate_misses,
        proposal.proposal_rejections,
        trace.acceptance_rate(chain_id),
        proposal.last_request,
        proposal.last_result
    );
    if let Some(step) = last_step {
        let info = step
            .info
            .as_ref()
            .map_or_else(|| String::from("none"), describe_step_info);
        let _ = write!(
            &mut context,
            " last_step_outcome={:?} last_step_info={} last_log_alpha={:?}",
            step.outcome, info, step.log_alpha
        );
    }
    let _ = write!(&mut context, " trace_tail=[{}]", trace_tail(trace));
    context
}

/// Aborts with chain context when a delayed Pachner step fails exceptionally.
fn abort_monte_carlo_step_error<const D: usize>(
    config: MonteCarloConfig,
    step: usize,
    chain: &Chain<MonteCarloTriangulation<D>>,
    proposal: &PachnerProposalKernel<D>,
    trace: &Trace,
    error: &DelayedStepError<FlipError>,
) -> ! {
    let context = monte_carlo_validation_context(config, step, chain, proposal, None, trace);
    abort_benchmark(format_args!("{context}: MCMC Pachner step failed: {error}"));
}

/// Chooses one raw Pachner request from the current cached topology frontier.
fn random_pachner_move<R: Rng + ?Sized, const D: usize>(
    dt: &MonteCarloTriangulation<D>,
    sampler: &MoveSampler,
    rng: &mut R,
    config: MonteCarloConfig,
) -> Option<PachnerMove<(), D>> {
    let move_kind_count = if D >= 4 { 6 } else { 5 };
    let mut move_kind = rng.random_range(0..move_kind_count);
    let vertex_count = dt.number_of_vertices();
    if vertex_count >= config.max_vertex_count && move_kind == 0 {
        move_kind = 1;
    } else if vertex_count <= config.min_vertex_count && move_kind == 1 {
        move_kind = 0;
    }

    match move_kind {
        0 => random_k1_insert(dt, sampler, rng),
        1 => sampler
            .random_vertex_key(rng)
            .map(|vertex_key| PachnerMove::K1Remove { vertex_key }),
        2 => random_k2(sampler, rng),
        3 => random_k2_inverse(sampler, rng),
        4 => random_k3(sampler, rng),
        5 => random_k3_inverse(dt, sampler, rng),
        _ => None,
    }
}

/// Chooses a random simplex and inserts a vertex at its centroid.
fn random_k1_insert<const D: usize>(
    dt: &MonteCarloTriangulation<D>,
    sampler: &MoveSampler,
    rng: &mut (impl Rng + ?Sized),
) -> Option<PachnerMove<(), D>> {
    let simplex_key = sampler.random_simplex_key(rng)?;
    let coords = random_simplex_centroid(dt, simplex_key)?.or_abort();
    let vertex: Vertex<(), D> = vertex!(coords).or_abort();
    Some(PachnerMove::K1Insert {
        simplex_key,
        vertex,
    })
}

/// Chooses a random simplex facet for a k=2 move.
fn random_k2<const D: usize>(
    sampler: &MoveSampler,
    rng: &mut (impl Rng + ?Sized),
) -> Option<PachnerMove<(), D>> {
    let facet = sampler.random_facet(rng)?;
    Some(PachnerMove::K2 { facet })
}

/// Chooses two vertices from a random simplex as an inverse k=2 edge candidate.
fn random_k2_inverse<const D: usize>(
    sampler: &MoveSampler,
    rng: &mut (impl Rng + ?Sized),
) -> Option<PachnerMove<(), D>> {
    let edge = sampler.random_edge(rng)?;
    Some(PachnerMove::K2Inverse { edge })
}

/// Chooses a random ridge from a random simplex for a k=3 move.
fn random_k3<const D: usize>(
    sampler: &MoveSampler,
    rng: &mut (impl Rng + ?Sized),
) -> Option<PachnerMove<(), D>> {
    let ridge = sampler.random_ridge(rng)?;
    Some(PachnerMove::K3 { ridge })
}

/// Chooses three vertices from a random simplex as an inverse k=3 triangle candidate.
fn random_k3_inverse<const D: usize>(
    dt: &MonteCarloTriangulation<D>,
    sampler: &MoveSampler,
    rng: &mut (impl Rng + ?Sized),
) -> Option<PachnerMove<(), D>> {
    let simplex_key = sampler.random_simplex_key(rng)?;
    let vertices = dt.simplex_vertices(simplex_key).ok()?;
    let [a, b, c] = three_distinct_indices(rng, vertices.len())?;
    let triangle = TriangleHandle::try_new(vertices[a], vertices[b], vertices[c]).or_abort();
    Some(PachnerMove::K3Inverse { triangle })
}

/// Computes a live simplex centroid when the cached key still exists.
fn random_simplex_centroid<const D: usize>(
    dt: &MonteCarloTriangulation<D>,
    simplex_key: SimplexKey,
) -> Option<Result<[f64; D], TryFromIntError>> {
    let vertices = dt.simplex_vertices(simplex_key).ok()?;
    let mut coords = [0.0; D];
    for &vertex_key in vertices {
        let vertex_coords = dt.vertex_coords(vertex_key)?;
        for (coord, value) in coords.iter_mut().zip(vertex_coords) {
            *coord += *value;
        }
    }

    let vertex_count = match u32::try_from(vertices.len()) {
        Ok(value) => f64::from(value),
        Err(error) => return Some(Err(error)),
    };
    for coord in &mut coords {
        *coord /= vertex_count;
    }
    Some(Ok(coords))
}

/// Chooses three distinct indices from a collection length.
fn three_distinct_indices(rng: &mut (impl Rng + ?Sized), len: usize) -> Option<[usize; 3]> {
    if len < 3 {
        return None;
    }
    let first = rng.random_range(0..len);
    let mut second = rng.random_range(0..len);
    while second == first {
        second = rng.random_range(0..len);
    }
    let mut third = rng.random_range(0..len);
    while third == first || third == second {
        third = rng.random_range(0..len);
    }
    Some([first, second, third])
}

/// Executes one long randomized Pachner sequence and validates periodically.
fn run_monte_carlo_sequence<const D: usize>(
    dt: MonteCarloTriangulation<D>,
    config: MonteCarloConfig,
) -> MonteCarloReport {
    let mut rng = StdRng::seed_from_u64(config.seed ^ 0x0253_0253_0253_0253);
    let target = FlatPachnerTarget;
    let mut chain = Chain::new(dt, &target).or_abort();
    let mut proposal = PachnerProposalKernel::new(chain.state(), config);
    let mut recorder = TraceRecorder::new(
        ChainId::new(0),
        [
            "vertices",
            "simplices",
            "candidate_misses",
            "proposal_rejections",
        ],
    )
    .or_abort();
    let start_rss_kib = memory_usage_kib();
    let mut max_rss_kib = start_rss_kib;
    let mut validations = 0;
    let mut validation_nanos = 0;
    let mut last_step = None;

    for step in 1..=config.move_attempts() {
        let mcmc_step = chain
            .step_delayed(&target, &mut proposal, &mut rng)
            .unwrap_or_else(|error| {
                abort_monte_carlo_step_error(
                    config,
                    step,
                    &chain,
                    &proposal,
                    recorder.trace(),
                    &error,
                );
            });
        record_monte_carlo_step(&mut recorder, &chain, &proposal, &mcmc_step);
        last_step = Some(mcmc_step);

        if step.is_multiple_of(config.validate_every()) {
            let validation_start = Instant::now();
            let context = monte_carlo_validation_context(
                config,
                step,
                &chain,
                &proposal,
                last_step.as_ref(),
                recorder.trace(),
            );
            validate_monte_carlo_state(chain.state(), context);
            validation_nanos += validation_start.elapsed().as_nanos();
            validations += 1;
            max_rss_kib = max_rss_kib.max(memory_usage_kib());
            proposal.sampler.refresh(chain.state());
        }
    }

    if !config
        .move_attempts()
        .is_multiple_of(config.validate_every())
    {
        let validation_start = Instant::now();
        let context = monte_carlo_validation_context(
            config,
            config.move_attempts(),
            &chain,
            &proposal,
            last_step.as_ref(),
            recorder.trace(),
        );
        validate_monte_carlo_state(chain.state(), context);
        validation_nanos += validation_start.elapsed().as_nanos();
        validations += 1;
    }

    let final_rss_kib = memory_usage_kib();
    max_rss_kib = max_rss_kib.max(final_rss_kib);
    MonteCarloReport {
        attempts: config.move_attempts(),
        accepted: chain.accepted(),
        rejected: chain.rejected(),
        candidate_misses: proposal.candidate_misses,
        proposal_rejections: proposal.proposal_rejections,
        validations,
        validation_nanos,
        elapsed_nanos: 0,
        attempts_per_second: 0,
        final_vertices: chain.state().number_of_vertices(),
        final_simplices: chain.state().number_of_simplices(),
        start_rss_kib,
        max_rss_kib,
        final_rss_kib,
    }
}

/// Records the Monte Carlo sequence counters in a parseable one-line format.
fn emit_monte_carlo_report<const D: usize>(config: MonteCarloConfig, report: MonteCarloReport) {
    println!(
        "pachner_stress_metric dimension={D} label={} attempts={} accepted={} rejected={} \
         candidate_misses={} proposal_rejections={} validations={} validation_nanos={} \
         elapsed_nanos={} attempts_per_second={} final_vertices={} final_simplices={} \
         start_rss_kib={} max_rss_kib={} final_rss_kib={}",
        config.label,
        report.attempts,
        report.accepted,
        report.rejected,
        report.candidate_misses,
        report.proposal_rejections,
        report.validations,
        report.validation_nanos,
        report.elapsed_nanos,
        report.attempts_per_second,
        report.final_vertices,
        report.final_simplices,
        report.start_rss_kib,
        report.max_rss_kib,
        report.final_rss_kib
    );
}

/// Runs one Monte Carlo sequence per Criterion iteration and excludes reporting overhead.
fn bench_monte_carlo_case<const D: usize>(
    c: &mut Criterion,
    label: &'static str,
    default_vertices: usize,
    default_seed: u64,
) {
    let config = monte_carlo_config::<D>(label, default_vertices, default_seed);
    let mut group = c.benchmark_group(format!("pachner_stress/monte_carlo/{label}"));
    group.sample_size((*MONTE_CARLO_SAMPLE_SIZE).get());
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(30));
    group.throughput(Throughput::Elements(
        u64::try_from(config.move_attempts()).or_abort(),
    ));

    let bench_name = format!(
        "{}v_{}attempts_validate{}",
        config.vertex_count,
        config.move_attempts(),
        config.validate_every()
    );
    let emit_reports = monte_carlo_report_enabled();
    group.bench_function(bench_name, |b| {
        let mut source: Option<MonteCarloTriangulation<D>> = None;
        b.iter_custom(|iters| {
            let mut total = Duration::new(0, 0);
            for _ in 0..iters {
                let dt = source
                    .get_or_insert_with(|| build_monte_carlo_dt::<D>(config, emit_reports))
                    .clone();
                let start = Instant::now();
                let mut report = run_monte_carlo_sequence(dt, config);
                let elapsed = start.elapsed();
                total += elapsed;
                report.elapsed_nanos = elapsed.as_nanos();
                let attempts = u128::try_from(report.attempts).or_abort();
                report.attempts_per_second =
                    attempts.saturating_mul(1_000_000_000) / report.elapsed_nanos.max(1);
                if emit_reports {
                    emit_monte_carlo_report::<D>(config, report);
                }
                black_box(report);
            }
            total
        });
    });

    group.finish();
}

/// Registers the dimension-scaled Monte Carlo Pachner stress cases.
fn pachner_monte_carlo_stress(c: &mut Criterion) {
    bench_monte_carlo_case::<3>(c, "3d", MONTE_CARLO_3D_VERTICES, 0x0253_0000_0000_0003);
    bench_monte_carlo_case::<4>(c, "4d", MONTE_CARLO_4D_VERTICES, 0x0253_0000_0000_0004);
}

/// Builds one stable 4D fixture and selects deterministic accepted move supports.
fn stress_setup() -> PachnerStressSetup {
    let base_dt = flip_workflows::build_flip_dt(STABLE_POINTS_4D).or_abort();
    let simplex_key =
        flip_workflows::largest_volume_simplex(&base_dt, CandidateFilter::Any).or_abort();
    let k1_coords = simplex_centroid(&base_dt, simplex_key).or_abort();
    let k1_vertex: Vertex<(), 4> = vertex!(k1_coords).or_abort();
    let facet = flip_workflows::flippable_k2_facet(&base_dt, true, CandidateFilter::Any).or_abort();
    let ridge = flip_workflows::flippable_k3_ridge(&base_dt, true, CandidateFilter::Any).or_abort();
    let (k1_remove_dt, k1_remove_vertex_key) = k1_remove_fixture(&base_dt, simplex_key, k1_vertex);
    let (k2_inverse_dt, k2_inverse_edge) = k2_inverse_fixture(&base_dt, facet);
    let (k3_inverse_dt, k3_inverse_triangle) = k3_inverse_fixture(&base_dt, ridge);

    PachnerStressSetup {
        base_dt,
        simplex_key,
        k1_vertex,
        k1_remove_dt,
        k1_remove_vertex_key,
        facet,
        k2_inverse_dt,
        k2_inverse_edge,
        ridge,
        k3_inverse_dt,
        k3_inverse_triangle,
    }
}

/// Computes a simplex centroid for the k=1 insertion stress move.
fn simplex_centroid(
    dt: &FlipTriangulation<4>,
    simplex_key: SimplexKey,
) -> Result<[f64; 4], TryFromIntError> {
    let simplex = dt
        .simplex(simplex_key)
        .or_abort(format_args!("missing selected simplex {simplex_key:?}"));
    let mut coords = [0.0; 4];
    for &vertex_key in simplex.vertices() {
        let vertex = dt
            .vertex(vertex_key)
            .or_abort(format_args!("missing simplex vertex {vertex_key:?}"));
        for (coord, value) in coords.iter_mut().zip(vertex.point().coords()) {
            *coord += *value;
        }
    }

    let vertex_count = u32::try_from(simplex.vertices().len()).map(f64::from)?;
    for coord in &mut coords {
        *coord /= vertex_count;
    }
    Ok(coords)
}

/// Builds a fixture where a k=1 inverse move is accepted.
fn k1_remove_fixture(
    base_dt: &FlipTriangulation<4>,
    simplex_key: SimplexKey,
    vertex: Vertex<(), 4>,
) -> (FlipTriangulation<4>, VertexKey) {
    let mut dt = base_dt.clone();
    let vertex_uuid = vertex.uuid();
    let inserted = attempt_pachner_move(
        &mut dt,
        PachnerMove::K1Insert {
            simplex_key,
            vertex,
        },
    );
    let vertex_key = dt
        .vertices()
        .find_map(|(vertex_key, vertex)| (vertex.uuid() == vertex_uuid).then_some(vertex_key))
        .or_abort(format_args!("missing inserted k=1 vertex {vertex_uuid}"));
    assert_eq!(inserted.inserted_face_vertices.as_slice(), &[vertex_key]);
    assert!(!inserted.new_simplices.is_empty());

    (dt, vertex_key)
}

/// Builds a fixture where a k=2 inverse move is accepted.
fn k2_inverse_fixture(
    base_dt: &FlipTriangulation<4>,
    facet: FacetHandle,
) -> (FlipTriangulation<4>, EdgeKey) {
    let mut dt = base_dt.clone();
    let info = attempt_pachner_move(&mut dt, PachnerMove::K2 { facet });
    let edge = inserted_edge(&dt, &info.inserted_face_vertices);

    (dt, edge)
}

/// Builds a fixture where a k=3 inverse move is accepted.
fn k3_inverse_fixture(
    base_dt: &FlipTriangulation<4>,
    ridge: RidgeHandle,
) -> (FlipTriangulation<4>, TriangleHandle) {
    let mut dt = base_dt.clone();
    let info = attempt_pachner_move(&mut dt, PachnerMove::K3 { ridge });
    let triangle = inserted_triangle(&info.inserted_face_vertices);

    (dt, triangle)
}

/// Parses and commits one Pachner request on the same topology owner.
fn attempt_pachner_move(
    dt: &mut FlipTriangulation<4>,
    pachner_move: PachnerMove<(), 4>,
) -> PachnerMoveResult<4> {
    dt.propose_pachner(pachner_move)
        .or_abort()
        .attempt_on(dt)
        .or_abort()
}

/// Converts a reported inserted face into an inverse k=2 edge handle.
fn inserted_edge(dt: &FlipTriangulation<4>, vertices: &[VertexKey]) -> EdgeKey {
    let [a, b] = vertices else {
        return Option::<EdgeKey>::None.or_abort(format_args!(
            "k=2 flip reported {} inserted-face vertices",
            vertices.len()
        ));
    };
    dt.edges()
        .find(|edge| {
            let (first, second) = edge.endpoints();
            (first == *a && second == *b) || (first == *b && second == *a)
        })
        .or_abort(format_args!("inserted k=2 edge {a:?}-{b:?} is missing"))
}

/// Converts a reported inserted face into an inverse k=3 triangle handle.
fn inserted_triangle(vertices: &[VertexKey]) -> TriangleHandle {
    let [a, b, c] = vertices else {
        return Option::<TriangleHandle>::None.or_abort(format_args!(
            "k=3 flip reported {} inserted-face vertices",
            vertices.len()
        ));
    };
    TriangleHandle::try_new(*a, *b, *c).or_abort()
}

/// Creates one batch of independent triangulation clones for repeated move attempts.
fn clone_batch(base_dt: &FlipTriangulation<4>) -> Vec<FlipTriangulation<4>> {
    vec![base_dt.clone(); (*MOVES_PER_SAMPLE).get()]
}

/// Registers one stress case that repeats the same raw Pachner request.
///
/// `bench_pachner_move` measures `attempt_pachner_move`, so each sample now
/// includes `propose_pachner` feasibility work and `attempt_on` revalidation
/// before mutation, not only the raw flip mutation cost.
fn bench_pachner_move(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &'static str,
    base_dt: &FlipTriangulation<4>,
    pachner_move: PachnerMove<(), 4>,
) {
    group.bench_function(name, move |b| {
        b.iter_batched(
            || clone_batch(base_dt),
            |mut triangulations| {
                for dt in &mut triangulations {
                    let result = attempt_pachner_move(dt, pachner_move);
                    black_box(&result);
                }
                black_box(triangulations);
            },
            BatchSize::LargeInput,
        );
    });
}

/// Runs the unified Pachner move stress benchmark group.
fn pachner_stress(c: &mut Criterion) {
    let setup = stress_setup();
    let mut group = c.benchmark_group("pachner_stress");
    group.throughput(Throughput::Elements(
        u64::try_from((*MOVES_PER_SAMPLE).get()).or_abort(),
    ));

    bench_pachner_move(
        &mut group,
        "k1_insert",
        &setup.base_dt,
        PachnerMove::K1Insert {
            simplex_key: setup.simplex_key,
            vertex: setup.k1_vertex,
        },
    );
    bench_pachner_move(
        &mut group,
        "k1_remove",
        &setup.k1_remove_dt,
        PachnerMove::K1Remove {
            vertex_key: setup.k1_remove_vertex_key,
        },
    );
    bench_pachner_move(
        &mut group,
        "k2",
        &setup.base_dt,
        PachnerMove::K2 { facet: setup.facet },
    );
    bench_pachner_move(
        &mut group,
        "k2_inverse",
        &setup.k2_inverse_dt,
        PachnerMove::K2Inverse {
            edge: setup.k2_inverse_edge,
        },
    );
    bench_pachner_move(
        &mut group,
        "k3",
        &setup.base_dt,
        PachnerMove::K3 { ridge: setup.ridge },
    );
    bench_pachner_move(
        &mut group,
        "k3_inverse",
        &setup.k3_inverse_dt,
        PachnerMove::K3Inverse {
            triangle: setup.k3_inverse_triangle,
        },
    );

    group.finish();
}

criterion_group!(benches, pachner_stress, pachner_monte_carlo_stress);
criterion_main!(benches);
