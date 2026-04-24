//! Large-scale triangulation debug harness.
//!
//! This module is intended as a **manual debugging tool** for issues that only show up
//! at large vertex counts (slow cases, rare geometric degeneracies, topology/Delaunay
//! validation failures, etc.).
//!
//! The tests are `#[ignore]` by default.
//!
//! ## Usage
//!
//! Run one dimension with full output:
//! ```bash
//! cargo test --release --test large_scale_debug debug_large_scale_3d -- --ignored --nocapture
//! ```
//!
//! Override defaults via environment variables:
//! ```bash
//! # Base RNG seed (decimal or 0x-hex)
//! DELAUNAY_LARGE_DEBUG_SEED=0xDEADBEEF \
//! # Optional explicit case seed (overrides derived seed_for_case)
//! DELAUNAY_LARGE_DEBUG_CASE_SEED=0x12345678 \
//! # Override point count for the selected test
//! DELAUNAY_LARGE_DEBUG_N=10000 \
//! # Point distribution: "ball" (default) or "box"
//! DELAUNAY_LARGE_DEBUG_DISTRIBUTION=ball \
//! # Ball radius (default: 100) [used when distribution=ball]
//! DELAUNAY_LARGE_DEBUG_BALL_RADIUS=100 \
//! # Box half-width (default: 100) [used when distribution=box]
//! DELAUNAY_LARGE_DEBUG_BOX_HALF_WIDTH=100 \
//! # Construction mode:
//! # - "new" (default): build via DelaunayTriangulation::new() which applies Hilbert ordering
//! # - "incremental": manual insert loop (debug/profiling)
//! DELAUNAY_LARGE_DEBUG_CONSTRUCTION_MODE=new \
//! # Debug mode: "cadenced" (default, repair/validate on a cadence) or "strict" (per-insertion)
//! DELAUNAY_LARGE_DEBUG_DEBUG_MODE=cadenced \
//! # Deterministically shuffle insertion order (incremental mode only)
//! DELAUNAY_LARGE_DEBUG_SHUFFLE_SEED=123 \
//! # Print progress every N insertions (incremental mode only)
//! DELAUNAY_LARGE_DEBUG_PROGRESS_EVERY=1000 \
//! # (Optional) validate topology every N insertions once cells exist (incremental mode only; can be expensive)
//! DELAUNAY_LARGE_DEBUG_VALIDATE_EVERY=2000 \
//! # Allow skipped vertices (otherwise the test fails if any are skipped)
//! DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1 \
//! # Skip the final flip-based repair pass (faster, but may leave Delaunay violations)
//! DELAUNAY_LARGE_DEBUG_SKIP_FINAL_REPAIR=1 \
//! # Run bounded incremental flip repair every N successful insertions (incremental mode only; 0 disables; default: 128)
//! DELAUNAY_LARGE_DEBUG_REPAIR_EVERY=128 \
//! # Hard wall-clock cap in seconds before the harness aborts (0 = no cap; default: 600)
//! DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS=600 \
//! # Optional: emit periodic batch-construction summaries for new()/Hilbert runs
//! DELAUNAY_BULK_PROGRESS_EVERY=100 \
//! # Optional: dump the first cavity reduction chain once per run
//! DELAUNAY_DEBUG_CAVITY_REDUCTION_ONCE=1 \
//! # Optional: trace retryable conflict-region skips with attempt/rollback details
//! DELAUNAY_DEBUG_RETRYABLE_SKIP=1 \
//! # Optional: dump the first detected ridge-fan cavity snapshot once per run
//! DELAUNAY_DEBUG_RIDGE_FAN_ONCE=1 \
//! cargo test --release --test large_scale_debug debug_large_scale_4d -- --ignored --nocapture
//! ```

#![forbid(unsafe_code)]

use delaunay::geometry::kernel::RobustKernel;
use delaunay::geometry::util::{
    generate_random_points_in_ball_seeded, generate_random_points_seeded,
};
use delaunay::prelude::triangulation::*;
use delaunay::triangulation::delaunay::{
    ConstructionOptions, ConstructionStatistics, DelaunayRepairHeuristicConfig,
    DelaunayTriangulationConstructionErrorWithStatistics,
};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::env;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

/// Installs a per-test wall-clock cap.
///
/// Spawns a watchdog thread that calls [`std::process::abort`] if `max_secs` elapses.
/// Returns a [`std::sync::mpsc::SyncSender`] whose **drop** cancels the watchdog: when
/// the sender is dropped (i.e. the test completes normally), the channel disconnects and
/// the watchdog thread exits without aborting.  This prevents a stale watchdog installed
/// for one test from firing during a subsequent test.
fn install_runtime_cap(max_secs: u64) -> std::sync::mpsc::SyncSender<()> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<()>(0);
    std::thread::spawn(move || {
        match rx.recv_timeout(Duration::from_secs(max_secs)) {
            // Sender dropped (test finished) or explicit send — exit cleanly.
            Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {}
            // Deadline exceeded — hard abort.
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                tracing::error!(
                    "=== TIMEOUT: wall time exceeded {max_secs} seconds — aborting ==="
                );
                std::process::abort();
            }
        }
    });
    tx
}

#[derive(Debug, Clone)]
struct SkipSample<const D: usize> {
    index: usize,
    uuid: uuid::Uuid,
    coords: [f64; D],
    attempts: usize,
    error: String,
}

#[derive(Debug, Default, Clone)]
struct InsertionSummary<const D: usize> {
    inserted: usize,
    skipped_duplicate: usize,
    skipped_degeneracy: usize,

    total_attempts: usize,
    max_attempts: usize,
    attempts_histogram: Vec<usize>,

    used_perturbation: usize,

    cells_removed_total: usize,
    cells_removed_max: usize,

    skip_samples: Vec<SkipSample<D>>,
}

impl<const D: usize> InsertionSummary<D> {
    fn record(&mut self, stats: InsertionStatistics) {
        self.total_attempts = self.total_attempts.saturating_add(stats.attempts);
        self.max_attempts = self.max_attempts.max(stats.attempts);

        if self.attempts_histogram.len() <= stats.attempts {
            self.attempts_histogram.resize(stats.attempts + 1, 0);
        }
        self.attempts_histogram[stats.attempts] =
            self.attempts_histogram[stats.attempts].saturating_add(1);

        if stats.used_perturbation() {
            self.used_perturbation = self.used_perturbation.saturating_add(1);
        }

        self.cells_removed_total = self
            .cells_removed_total
            .saturating_add(stats.cells_removed_during_repair);
        self.cells_removed_max = self
            .cells_removed_max
            .max(stats.cells_removed_during_repair);
    }

    fn record_inserted(&mut self, stats: InsertionStatistics) {
        self.inserted = self.inserted.saturating_add(1);
        self.record(stats);
    }

    fn record_skipped(&mut self, sample: SkipSample<D>, stats: InsertionStatistics) {
        if stats.skipped_duplicate() {
            self.skipped_duplicate = self.skipped_duplicate.saturating_add(1);
        } else {
            self.skipped_degeneracy = self.skipped_degeneracy.saturating_add(1);
        }

        // Keep the first few skip samples so we have concrete reproduction anchors.
        if self.skip_samples.len() < 8 {
            self.skip_samples.push(sample);
        }

        self.record(stats);
    }

    const fn total_skipped(&self) -> usize {
        self.skipped_duplicate + self.skipped_degeneracy
    }
}

impl<const D: usize> From<ConstructionStatistics> for InsertionSummary<D> {
    fn from(stats: ConstructionStatistics) -> Self {
        let skip_samples: Vec<SkipSample<D>> = stats
            .skip_samples
            .iter()
            .filter_map(|s| {
                let coords: [f64; D] = if let Ok(coords) = s.coords.as_slice().try_into() {
                    coords
                } else {
                    tracing::warn!(
                        index = s.index,
                        uuid = %s.uuid,
                        coords_len = s.coords.len(),
                        expected_dim = D,
                        "dropping skip sample due to coordinate dimension mismatch"
                    );
                    return None;
                };
                Some(SkipSample {
                    index: s.index,
                    uuid: s.uuid,
                    coords,
                    attempts: s.attempts,
                    error: s.error.clone(),
                })
            })
            .collect();

        Self {
            inserted: stats.inserted,
            skipped_duplicate: stats.skipped_duplicate,
            skipped_degeneracy: stats.skipped_degeneracy,
            total_attempts: stats.total_attempts,
            max_attempts: stats.max_attempts,
            attempts_histogram: stats.attempts_histogram,
            used_perturbation: stats.used_perturbation,
            cells_removed_total: stats.cells_removed_total,
            cells_removed_max: stats.cells_removed_max,
            skip_samples,
        }
    }
}

fn parse_u64(s: &str) -> Option<u64> {
    let s = s.trim();
    s.strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .map_or_else(|| s.parse().ok(), |hex| u64::from_str_radix(hex, 16).ok())
}

fn env_u64(name: &str) -> Option<u64> {
    env::var(name).ok().and_then(|v| parse_u64(&v))
}

fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().and_then(|v| {
        let trimmed = v.trim();
        trimmed.parse().ok().or_else(|| {
            trimmed
                .split_once('=')
                .and_then(|(_, rhs)| rhs.trim().parse().ok())
        })
    })
}

fn env_flag(name: &str) -> bool {
    env::var(name).ok().is_some_and(|v| {
        let v = v.trim();
        !v.is_empty() && v != "0" && v != "false"
    })
}

fn init_tracing() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // Debug-level tracing is needed to surface the release-visible diagnostic hooks
        // (retryable-skip, cavity-reduction, ridge-fan-dump, bulk-progress, bulk-retry)
        // that are emitted through `tracing::debug!` inside the library.
        let debug_env_vars = [
            "DELAUNAY_INSERT_TRACE",
            "DELAUNAY_DEBUG_RETRYABLE_SKIP",
            "DELAUNAY_DEBUG_CAVITY_REDUCTION_ONCE",
            "DELAUNAY_DEBUG_RIDGE_FAN_ONCE",
            "DELAUNAY_BULK_PROGRESS_EVERY",
            "DELAUNAY_DEBUG_SHUFFLE",
        ];
        let default_filter = if debug_env_vars
            .iter()
            .any(|name| env::var_os(name).is_some())
        {
            "debug"
        } else {
            "warn"
        };
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_filter));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PointDistribution {
    Ball,
    Box,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstructionMode {
    /// Build via `DelaunayTriangulation::new()` (batch construction + Hilbert ordering).
    New,
    /// Insert vertices one by one (manual incremental construction).
    Incremental,
}

impl ConstructionMode {
    const fn name(self) -> &'static str {
        match self {
            Self::New => "new",
            Self::Incremental => "incremental",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DebugMode {
    /// Faster default: repair/check in cadence, with suspicion-driven automatic validation.
    Cadenced,
    /// Maximal diagnostics: strict guarantee + per-insertion automatic validation.
    Strict,
}

impl DebugMode {
    const fn name(self) -> &'static str {
        match self {
            Self::Cadenced => "cadenced",
            Self::Strict => "strict",
        }
    }
}

impl PointDistribution {
    const fn name(self) -> &'static str {
        match self {
            Self::Ball => "ball",
            Self::Box => "box",
        }
    }
}

fn env_f64(name: &str) -> Option<f64> {
    let Ok(raw) = env::var(name) else {
        return None;
    };

    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }

    Some(
        raw.parse()
            .unwrap_or_else(|e| panic!("invalid {name}={raw:?}: {e}")),
    )
}

fn point_distribution_from_env() -> PointDistribution {
    let Ok(raw) = env::var("DELAUNAY_LARGE_DEBUG_DISTRIBUTION") else {
        return PointDistribution::Ball;
    };

    let raw = raw.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("ball") {
        return PointDistribution::Ball;
    }

    if raw.eq_ignore_ascii_case("box") {
        return PointDistribution::Box;
    }

    panic!("invalid DELAUNAY_LARGE_DEBUG_DISTRIBUTION={raw:?} (expected 'ball' or 'box')");
}

fn construction_mode_from_env() -> ConstructionMode {
    let Ok(raw) = env::var("DELAUNAY_LARGE_DEBUG_CONSTRUCTION_MODE") else {
        return ConstructionMode::New;
    };

    let raw = raw.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("new") {
        return ConstructionMode::New;
    }

    if raw.eq_ignore_ascii_case("incremental") {
        return ConstructionMode::Incremental;
    }

    panic!(
        "invalid DELAUNAY_LARGE_DEBUG_CONSTRUCTION_MODE={raw:?} (expected 'new' or 'incremental')"
    );
}

fn debug_mode_from_env() -> DebugMode {
    let Ok(raw) = env::var("DELAUNAY_LARGE_DEBUG_DEBUG_MODE") else {
        return DebugMode::Cadenced;
    };

    let raw = raw.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("cadenced") {
        return DebugMode::Cadenced;
    }

    if raw.eq_ignore_ascii_case("strict") {
        return DebugMode::Strict;
    }

    panic!("invalid DELAUNAY_LARGE_DEBUG_DEBUG_MODE={raw:?} (expected 'cadenced' or 'strict')");
}

fn seed_for_case<const D: usize>(base_seed: u64, n_points: usize) -> u64 {
    const SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
    base_seed
        .wrapping_add((n_points as u64).wrapping_mul(SEED_SALT))
        .wrapping_add((D as u64).wrapping_mul(SEED_SALT.rotate_left(17)))
}
/// Classifies debug harness outcomes into distinct categories for structured reporting.
///
/// The harness still prints full diagnostic output during the run; this enum captures
/// the final outcome so callers can emit a single summary line and fail tests explicitly.
#[derive(Debug, Clone)]
enum DebugOutcome {
    Success,
    ConstructionFailure {
        error: String,
    },
    SkippedVertices {
        skipped: usize,
        total: usize,
    },
    RepairNonConvergence {
        error: String,
    },
    ValidationFailure {
        kind: InvariantKind,
        details: String,
    },
}

impl std::fmt::Display for DebugOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success => write!(f, "Success"),
            Self::ConstructionFailure { error } => {
                write!(f, "ConstructionFailure | {error}")
            }
            Self::SkippedVertices { skipped, total } => {
                write!(f, "SkippedVertices | {skipped}/{total} skipped")
            }
            Self::RepairNonConvergence { error } => {
                write!(f, "RepairNonConvergence | {error}")
            }
            Self::ValidationFailure { kind, details } => {
                write!(f, "ValidationFailure({kind:?}) | {details}")
            }
        }
    }
}

fn print_abort_summary<const D: usize>(
    outcome: &DebugOutcome,
    seed: u64,
    n_points: usize,
    phase: &str,
) {
    println!();
    println!("OUTCOME: {outcome}");
    println!("Phase:   {phase}");
    println!();
    println!("Replay command:");
    println!(
        "  DELAUNAY_LARGE_DEBUG_N_{D}D={n_points} DELAUNAY_LARGE_DEBUG_CASE_SEED_{D}D=0x{seed:X} DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1 cargo test --test large_scale_debug debug_large_scale_{D}d -- --ignored --nocapture"
    );
}

fn classify_validation_report(report: &TriangulationValidationReport) -> DebugOutcome {
    let first = report.violations.first();
    let (kind, details) = first.map_or_else(
        || {
            (
                InvariantKind::Topology,
                "no violations captured".to_string(),
            )
        },
        |violation| (violation.kind, format!("{}", violation.error)),
    );
    DebugOutcome::ValidationFailure { kind, details }
}

fn print_validation_report(report: &TriangulationValidationReport) {
    println!(
        "Validation report: {} violation(s)",
        report.violations.len()
    );
    for (idx, violation) in report.violations.iter().enumerate() {
        println!("  {}. {:?}: {}", idx + 1, violation.kind, violation.error);
    }
}

fn print_insertion_summary<const D: usize>(summary: &InsertionSummary<D>, elapsed: Duration) {
    println!("Insertion summary:");
    println!("  inserted:          {}", summary.inserted);
    println!("  skipped_duplicate: {}", summary.skipped_duplicate);
    println!("  skipped_degeneracy:{}", summary.skipped_degeneracy);
    println!("  total_skipped:     {}", summary.total_skipped());
    println!();

    println!("  total_attempts:    {}", summary.total_attempts);
    println!("  max_attempts:      {}", summary.max_attempts);
    println!("  used_perturbation: {}", summary.used_perturbation);
    println!(
        "  cells_removed_during_repair: total={}, max={} (insertion safety-net / repair bookkeeping)",
        summary.cells_removed_total, summary.cells_removed_max
    );

    if !summary.attempts_histogram.is_empty() {
        println!("  attempts_histogram (attempts -> count):");
        for (attempts, count) in summary.attempts_histogram.iter().enumerate().skip(1) {
            if *count > 0 {
                println!("    {attempts} -> {count}");
            }
        }
    }

    if !summary.skip_samples.is_empty() {
        println!();
        println!("  skip_samples (first {}):", summary.skip_samples.len());
        for s in &summary.skip_samples {
            println!(
                "    idx={} uuid={} attempts={} coords={:?} error={}",
                s.index, s.uuid, s.attempts, s.coords, s.error
            );
        }
    }

    println!();
    println!("Insertion wall time: {elapsed:?}");
}

#[expect(
    clippy::too_many_lines,
    reason = "Intentional debug harness; kept as a single flow for manual inspection"
)]
fn debug_large_case<const D: usize>(dimension_name: &str, default_n_points: usize) -> DebugOutcome {
    init_tracing();

    // Install a hard wall-clock cap so the harness doesn't hang indefinitely.
    // Override with DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS (0 = no cap).
    let max_runtime_secs = env_usize("DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS").unwrap_or(600);
    // Hold the sender for the lifetime of this function; dropping it at return
    // cancels the watchdog thread so it does not outlive this test.
    let _watchdog = (max_runtime_secs > 0).then(|| install_runtime_cap(max_runtime_secs as u64));

    let base_seed = env_u64("DELAUNAY_LARGE_DEBUG_SEED").unwrap_or(42);

    let n_points = env_usize(&format!("DELAUNAY_LARGE_DEBUG_N_{D}D"))
        .or_else(|| env_usize("DELAUNAY_LARGE_DEBUG_N"))
        .unwrap_or(default_n_points)
        .max(D + 1);

    let seed = env_u64(&format!("DELAUNAY_LARGE_DEBUG_CASE_SEED_{D}D"))
        .or_else(|| env_u64("DELAUNAY_LARGE_DEBUG_CASE_SEED"))
        .unwrap_or_else(|| seed_for_case::<D>(base_seed, n_points));
    let distribution = point_distribution_from_env();
    let ball_radius = env_f64("DELAUNAY_LARGE_DEBUG_BALL_RADIUS").unwrap_or(100.0);
    let box_half_width = env_f64("DELAUNAY_LARGE_DEBUG_BOX_HALF_WIDTH").unwrap_or(100.0);

    let mode = construction_mode_from_env();
    let debug_mode = debug_mode_from_env();

    let shuffle_seed = env_u64("DELAUNAY_LARGE_DEBUG_SHUFFLE_SEED");
    let progress_every = env_usize("DELAUNAY_LARGE_DEBUG_PROGRESS_EVERY")
        .unwrap_or(1000)
        .max(1);

    let allow_skips = env_flag("DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS");
    let skip_final_repair = env_flag("DELAUNAY_LARGE_DEBUG_SKIP_FINAL_REPAIR");

    // Delaunay repair scheduling
    // - 0 disables incremental repair
    // - 1 runs repair after every insertion
    // - N>1 runs repair after every N successful insertions
    let repair_every = env_usize("DELAUNAY_LARGE_DEBUG_REPAIR_EVERY").unwrap_or(128);
    let repair_max_flips = env_usize("DELAUNAY_LARGE_DEBUG_REPAIR_MAX_FLIPS");
    let validate_every = env_usize("DELAUNAY_LARGE_DEBUG_VALIDATE_EVERY").or_else(|| {
        if matches!(debug_mode, DebugMode::Cadenced) {
            (repair_every != 0).then_some(repair_every)
        } else {
            None
        }
    });

    println!("=============================================");
    println!("Large-scale triangulation debug: {dimension_name}");
    println!("=============================================");
    println!("Config:");
    println!("  D:             {D}");
    println!("  n_points:      {n_points}");
    println!("  base_seed:     0x{base_seed:X} ({base_seed})");
    println!("  case_seed:     0x{seed:X} ({seed})");
    println!(
        "  distribution: {distribution}",
        distribution = distribution.name()
    );
    match distribution {
        PointDistribution::Ball => println!("  ball_radius:  {ball_radius}"),
        PointDistribution::Box => println!("  box_half_width:{box_half_width}"),
    }
    println!("  construction_mode: {}", mode.name());
    println!("  debug_mode:    {}", debug_mode.name());
    println!("  shuffle_seed:  {shuffle_seed:?}");
    println!("  progress_every:{progress_every}");
    println!("  validate_every:{validate_every:?}");
    println!("  allow_skips:   {allow_skips}");
    println!("  skip_final_repair: {skip_final_repair}");
    println!("  repair_every:  {repair_every}");
    println!("  repair_max_flips: {repair_max_flips:?}");
    if max_runtime_secs > 0 {
        println!("  max_runtime_secs: {max_runtime_secs}");
    }
    println!();

    println!("Generating points...");
    let t_gen = Instant::now();
    let points = match distribution {
        PointDistribution::Ball => {
            generate_random_points_in_ball_seeded::<f64, D>(n_points, ball_radius, seed)
                .unwrap_or_else(|e| {
                    panic!(
                        "failed to generate deterministic ball points (radius={ball_radius}): {e}"
                    )
                })
        }
        PointDistribution::Box => {
            let range = (-box_half_width, box_half_width);
            generate_random_points_seeded::<f64, D>(n_points, range, seed).unwrap_or_else(|e| {
                panic!("failed to generate deterministic box points (range={range:?}): {e}")
            })
        }
    };
    println!("Generated {} points in {:?}", points.len(), t_gen.elapsed());

    println!("Building vertices...");
    let t_vertices = Instant::now();
    let mut vertices: Vec<Vertex<f64, (), D>> = points.into_iter().map(|p| vertex!(p)).collect();
    println!(
        "Built {} vertices in {:?}",
        vertices.len(),
        t_vertices.elapsed()
    );

    let t_insert = Instant::now();

    let mut dt: DelaunayTriangulation<_, (), (), D> = match mode {
        ConstructionMode::New => {
            // `DelaunayTriangulation::new()` applies Hilbert ordering during batch construction.
            // Use the statistics-returning variant so we can report aggregate insertion telemetry.
            //
            // Use PLManifoldStrict during batch construction to ensure vertex-link invariants are
            // maintained on each insertion.
            let kernel = RobustKernel::<f64>::new();
            println!("Starting batch construction (new)...");
            let t_batch = Instant::now();
            match DelaunayTriangulation::with_topology_guarantee_and_options_with_construction_statistics(
                &kernel,
                &vertices,
                TopologyGuarantee::PLManifoldStrict,
                ConstructionOptions::default(),
            ) {
                Ok((dt, stats)) => {
                    let summary: InsertionSummary<D> = stats.into();
                    print_insertion_summary(&summary, t_insert.elapsed());
                    println!("Batch construction completed in {:?}", t_batch.elapsed());
                    dt
                }
                Err(e) => {
                    let DelaunayTriangulationConstructionErrorWithStatistics {
                        error,
                        statistics,
                        ..
                    } = e;
                    let summary: InsertionSummary<D> = statistics.into();
                    print_insertion_summary(&summary, t_insert.elapsed());
                    println!("Batch construction failed after {:?}", t_batch.elapsed());
                    println!("construction failed: {error}");
                    let outcome = DebugOutcome::ConstructionFailure {
                        error: format!("{error}"),
                    };
                    print_abort_summary::<D>(&outcome, seed, n_points, "batch construction");
                    return outcome;
                }
            }
        }
        ConstructionMode::Incremental => {
            if let Some(shuffle_seed) = shuffle_seed {
                let mut rng = StdRng::seed_from_u64(shuffle_seed);
                vertices.shuffle(&mut rng);
                println!("Shuffled insertion order with seed {shuffle_seed}");
            }

            let (topology_guarantee, validation_policy) = match debug_mode {
                DebugMode::Cadenced => {
                    (TopologyGuarantee::PLManifold, ValidationPolicy::OnSuspicion)
                }
                DebugMode::Strict => (
                    TopologyGuarantee::PLManifoldStrict,
                    ValidationPolicy::Always,
                ),
            };

            let mut dt: DelaunayTriangulation<_, (), (), D> =
                DelaunayTriangulation::with_empty_kernel_and_topology_guarantee(
                    RobustKernel::<f64>::new(),
                    topology_guarantee,
                );

            // Delaunay policies:
            // - Enable bounded incremental repair (local flip queue) every N successful insertions.
            // - Keep global Delaunay checks off during insertion; the harness can optionally run a final
            //   global repair pass at the end.
            let repair_policy = match NonZeroUsize::new(repair_every) {
                None => DelaunayRepairPolicy::Never,
                Some(n) if n.get() == 1 => DelaunayRepairPolicy::EveryInsertion,
                Some(n) => DelaunayRepairPolicy::EveryN(n),
            };
            dt.set_delaunay_repair_policy(repair_policy);
            dt.set_delaunay_check_policy(DelaunayCheckPolicy::EndOnly);

            // Debug-mode-dependent topology validation strategy:
            // - cadenced: suspicion-driven automatic checks + periodic explicit checks
            // - strict: per-insertion automatic checks
            dt.set_validation_policy(validation_policy);

            println!("Policies:");
            println!("  topology_guarantee:   {:?}", dt.topology_guarantee());
            println!("  validation_policy:    {:?}", dt.validation_policy());
            println!("  delaunay_repair_policy:{:?}", dt.delaunay_repair_policy());
            println!("  delaunay_check_policy: {:?}", dt.delaunay_check_policy());
            println!();

            let mut summary: InsertionSummary<D> = InsertionSummary::default();
            let mut had_cells = false;
            let mut t_last_progress = Instant::now();

            for (idx, vertex) in vertices.iter().copied().enumerate() {
                let coords = *vertex.point().coords();
                let uuid = vertex.uuid();

                match dt.insert_with_statistics(vertex) {
                    Ok((InsertionOutcome::Inserted { .. }, stats)) => {
                        summary.record_inserted(stats);
                    }
                    Ok((InsertionOutcome::Skipped { error }, stats)) => {
                        let sample = SkipSample {
                            index: idx,
                            uuid,
                            coords,
                            attempts: stats.attempts,
                            error: error.to_string(),
                        };
                        summary.record_skipped(sample, stats);
                    }
                    Err(err) => {
                        println!(
                            "Non-retryable insertion error at idx={idx} uuid={uuid} coords={coords:?}"
                        );
                        println!("  error: {err}");

                        if let Err(report) = dt.validation_report() {
                            print_validation_report(&report);
                        } else {
                            println!("validation_report: OK (after rollback)");
                        }
                        let outcome = DebugOutcome::ConstructionFailure {
                            error: format!("{err}"),
                        };
                        print_abort_summary::<D>(&outcome, seed, n_points, "incremental insertion");
                        return outcome;
                    }
                }

                if !had_cells && dt.number_of_cells() > 0 {
                    had_cells = true;
                    println!("Initial simplex created at insertion {}", idx + 1);
                }

                if let Some(every) = validate_every
                    && every > 0
                    && had_cells
                    && (idx + 1) % every == 0
                    && let Err(e) = dt.as_triangulation().is_valid()
                {
                    println!("Topology validation failed at idx={idx}: {e}");
                    let outcome = if let Err(report) = dt.validation_report() {
                        print_validation_report(&report);
                        classify_validation_report(&report)
                    } else {
                        DebugOutcome::ValidationFailure {
                            kind: InvariantKind::Topology,
                            details: format!("{e}"),
                        }
                    };
                    print_abort_summary::<D>(&outcome, seed, n_points, "periodic validation");
                    return outcome;
                }

                if (idx + 1) % progress_every == 0 {
                    let chunk_elapsed = t_last_progress.elapsed();
                    let progress_f64: f64 =
                        delaunay::geometry::util::safe_usize_to_scalar(progress_every)
                            .unwrap_or(f64::NAN);
                    let rate = progress_f64 / chunk_elapsed.as_secs_f64().max(1e-9);
                    println!(
                        "progress: {}/{} inserted={} skipped={} cells={} elapsed={:?} ({:.1} pts/s last {})",
                        idx + 1,
                        n_points,
                        summary.inserted,
                        summary.total_skipped(),
                        dt.number_of_cells(),
                        t_insert.elapsed(),
                        rate,
                        progress_every,
                    );
                    t_last_progress = Instant::now();
                }
            }

            print_insertion_summary(&summary, t_insert.elapsed());

            dt
        }
    };

    // Infer skipped count from the resulting triangulation size.
    let skipped_total = n_points.saturating_sub(dt.number_of_vertices());

    println!(
        "Triangulation size: vertices={} (skipped={skipped_total}) cells={} dim={}",
        dt.number_of_vertices(),
        dt.number_of_cells(),
        dt.dim()
    );

    if !allow_skips && skipped_total > 0 {
        let outcome = DebugOutcome::SkippedVertices {
            skipped: skipped_total,
            total: n_points,
        };
        print_abort_summary::<D>(&outcome, seed, n_points, "skip check");
        return outcome;
    }

    let mut repair_failure: Option<String> = None;
    if !skip_final_repair && dt.number_of_cells() > 0 {
        println!();
        println!("Running final flip-based repair (advanced)...");
        let t_repair = Instant::now();
        let mut repair_config = DelaunayRepairHeuristicConfig::default();
        repair_config.max_flips = repair_max_flips;
        match dt.repair_delaunay_with_flips_advanced(repair_config) {
            Ok(outcome) => {
                println!(
                    "repair: checked={} flips={} max_queue={} used_heuristic={}",
                    outcome.stats.facets_checked,
                    outcome.stats.flips_performed,
                    outcome.stats.max_queue_len,
                    outcome.used_heuristic()
                );
                if let Some(seeds) = outcome.heuristic {
                    println!(
                        "repair heuristic seeds: shuffle_seed={} perturbation_seed={}",
                        seeds.shuffle_seed, seeds.perturbation_seed
                    );
                }
            }
            Err(e) => {
                println!("repair failed: {e}");
                repair_failure = Some(format!("{e}"));
            }
        }
        println!("repair wall time: {:?}", t_repair.elapsed());
    }

    println!();
    println!("Running validation_report (Levels 1–4)...");
    let t_validate = Instant::now();
    let validation_result = dt.validation_report();
    println!("validation_report wall time: {:?}", t_validate.elapsed());
    match validation_result {
        Ok(()) => println!("validation_report: OK"),
        Err(report) => {
            print_validation_report(&report);
            let outcome = classify_validation_report(&report);
            print_abort_summary::<D>(&outcome, seed, n_points, "final validation");
            return outcome;
        }
    }

    // If repair failed but validation passed, surface the repair failure as the outcome.
    // This ensures operators see repair non-convergence even when the triangulation
    // happens to pass L1-L4 validation (e.g. the violations are below the flip budget).
    if let Some(error) = repair_failure {
        let outcome = DebugOutcome::RepairNonConvergence { error };
        print_abort_summary::<D>(&outcome, seed, n_points, "final repair");
        return outcome;
    }

    println!();
    println!("Total wall time: {:?}", t_gen.elapsed());
    DebugOutcome::Success
}

/// Regression test for issue #228: 3D 1000-point flip-repair non-convergence.
///
/// Before the fix, `AdaptiveKernel`'s exact+SoS predicates were overridden by
/// `use_robust_on_ambiguous` in the flip repair code, causing tolerance-based
/// predicates to return wrong signs for near-degenerate cases and triggering
/// flip cycles.
///
/// This test uses the exact seed that reproduced the original failure:
/// `seed_for_case::<3>(42, 1000)` with ball distribution (radius=100).
///
/// Uses `PLManifold` topology for full PL-manifold validation including
/// geometric cell orientation (#258 fixed the negative-orientation gap that
/// previously required the `Pseudomanifold` workaround).
///
/// Gated behind `slow-tests` and `#[ignore]` because 1000-point 3D
/// construction takes minutes in debug mode, exceeding CI timeout.
/// Run manually with:
/// ```bash
/// cargo test --test large_scale_debug --features slow-tests regression_issue_228 -- --ignored --nocapture
/// ```
#[cfg(feature = "slow-tests")]
#[test]
#[ignore = "1000-point 3D construction exceeds CI timeout (~30min debug)"]
fn regression_issue_228_3d_1000_flip_repair_convergence() {
    use delaunay::core::triangulation::TopologyGuarantee;

    let seed = seed_for_case::<3>(42, 1000);
    let points = generate_random_points_in_ball_seeded::<f64, 3>(1000, 100.0, seed)
        .expect("point generation should succeed");
    let vertices: Vec<Vertex<f64, (), 3>> = points.into_iter().map(|p| vertex!(p)).collect();

    // Use the default kernel (AdaptiveKernel — exact+SoS predicates) to match the
    // actual regression scenario.  PLManifold enables full orientation validation
    // now that #258 is fixed.
    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .expect("construction must not fail (#228 regression)");

    assert!(
        dt.as_triangulation().validate().is_ok(),
        "Topology validation (L1-L3) must pass (#228 regression, seed=0x{seed:X})"
    );
    assert!(
        dt.is_delaunay_via_flips().is_ok(),
        "Delaunay property must hold (#228 regression, seed=0x{seed:X})"
    );
}

/// Regression test for issue #230: 4D 100-point orientation failure path.
///
/// This locks in the 4D seed used by the debug harness:
/// `seed_for_case::<4>(42, 100)` with ball distribution (radius=100).
///
/// Current expected behavior:
/// - Batch construction succeeds
/// - Some vertices may be skipped due to degeneracy handling/retries
/// - Resulting triangulation passes topology validation (L1-L3)
///
/// Gated behind `slow-tests` and `#[ignore]` because 4D construction can
/// take multiple minutes in debug mode.
#[cfg(feature = "slow-tests")]
#[test]
#[ignore = "4D 100-point construction can take minutes in debug mode"]
fn regression_issue_230_4d_100_orientation() {
    use delaunay::core::triangulation::TopologyGuarantee;

    let seed = seed_for_case::<4>(42, 100);
    let points = generate_random_points_in_ball_seeded::<f64, 4>(100, 100.0, seed)
        .expect("point generation should succeed");
    let vertices: Vec<Vertex<f64, (), 4>> = points.into_iter().map(|p| vertex!(p)).collect();

    let kernel = RobustKernel::<f64>::new();
    let (dt, stats) = DelaunayTriangulation::<RobustKernel<f64>, (), (), 4>::with_topology_guarantee_and_options_with_construction_statistics(
        &kernel,
        &vertices,
        TopologyGuarantee::PLManifoldStrict,
        ConstructionOptions::default(),
    )
    .expect("construction must not fail (#230 regression)");

    println!(
        "regression_issue_230: inserted={} skipped={} (duplicate={} degeneracy={}) seed=0x{seed:X}",
        stats.inserted,
        stats.total_skipped(),
        stats.skipped_duplicate,
        stats.skipped_degeneracy
    );

    assert!(
        dt.as_triangulation().validate().is_ok(),
        "Topology validation (L1-L3) must pass (#230 regression, seed=0x{seed:X})"
    );
}

#[test]
#[ignore = "large-scale debug harness (manual run)"]
fn debug_large_scale_3d() {
    let outcome = debug_large_case::<3>("3D", 10_000);
    assert!(matches!(outcome, DebugOutcome::Success), "{outcome}");
}

#[test]
#[ignore = "large-scale debug harness (manual run)"]
fn debug_large_scale_4d() {
    let outcome = debug_large_case::<4>("4D", 3_000);
    assert!(matches!(outcome, DebugOutcome::Success), "{outcome}");
}

#[test]
#[ignore = "large-scale debug harness (manual run)"]
fn debug_large_scale_5d() {
    let outcome = debug_large_case::<5>("5D", 1_000);
    assert!(matches!(outcome, DebugOutcome::Success), "{outcome}");
}
