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
//! cargo test --test large_scale_debug debug_large_scale_3d -- --ignored --nocapture
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
//! # Cap final repair at N flips (useful for bounding 4D+ repair cost; 0 = no cap; default: no cap)
//! DELAUNAY_LARGE_DEBUG_REPAIR_MAX_FLIPS=500 \
//! cargo test --test large_scale_debug debug_large_scale_4d -- --ignored --nocapture
//! ```

#![forbid(unsafe_code)]

use delaunay::core::delaunay_triangulation::{
    ConstructionOptions, ConstructionStatistics, DelaunayRepairHeuristicConfig,
    DelaunayTriangulationConstructionErrorWithStatistics,
};
use delaunay::geometry::kernel::RobustKernel;
use delaunay::geometry::util::safe_usize_to_scalar;
use delaunay::geometry::util::{
    generate_random_points_in_ball_seeded, generate_random_points_seeded,
};
use delaunay::prelude::triangulation::*;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

/// Installs a per-test wall-clock cap.
///
/// Spawns a watchdog thread that prints a structured timeout outcome and calls
/// [`std::process::abort`] if `max_secs` elapses. Returns a
/// [`std::sync::mpsc::SyncSender`] whose **drop** cancels the watchdog: when
/// the sender is dropped (i.e. the test completes normally), the channel disconnects and
/// the watchdog thread exits without aborting. This prevents a stale watchdog installed
/// for one test from firing during a subsequent test.
fn install_runtime_cap(max_secs: u64, details: String) -> std::sync::mpsc::SyncSender<()> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<()>(0);
    std::thread::spawn(move || {
        match rx.recv_timeout(Duration::from_secs(max_secs)) {
            // Sender dropped (test finished) or explicit send — exit cleanly.
            Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {}
            // Deadline exceeded — hard abort.
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                let outcome = DebugOutcome::Timeout { details };
                eprintln!("OUTCOME: {outcome}");
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
    std::env::var(name).ok().and_then(|v| parse_u64(&v))
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| {
        let trimmed = v.trim();
        trimmed.parse().ok().or_else(|| {
            trimmed
                .split_once('=')
                .and_then(|(_, rhs)| rhs.trim().parse().ok())
        })
    })
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|v| {
        let v = v.trim();
        !v.is_empty() && v != "0" && v != "false"
    })
}

fn init_tracing() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
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
    let Ok(raw) = std::env::var(name) else {
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
    let Ok(raw) = std::env::var("DELAUNAY_LARGE_DEBUG_DISTRIBUTION") else {
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
    let Ok(raw) = std::env::var("DELAUNAY_LARGE_DEBUG_CONSTRUCTION_MODE") else {
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
    let Ok(raw) = std::env::var("DELAUNAY_LARGE_DEBUG_DEBUG_MODE") else {
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

/// Classifies the outcome of a debug harness run for structured reporting.
#[derive(Debug)]
enum DebugOutcome {
    /// Construction and validation both succeeded.
    Success {
        inserted: usize,
        skipped: usize,
        cells: usize,
    },
    /// Batch or incremental construction failed before completion.
    ConstructionFailure { error: String },
    /// Watchdog timeout aborted the run before the harness completed.
    Timeout { details: String },
    /// Validation (Levels 1-4) failed after construction.
    ValidationFailure { details: String },
}

impl std::fmt::Display for DebugOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success {
                inserted,
                skipped,
                cells,
            } => write!(
                f,
                "SUCCESS | inserted={inserted} skipped={skipped} cells={cells}"
            ),
            Self::ConstructionFailure { error } => {
                write!(f, "CONSTRUCTION_FAILURE | {error}")
            }
            Self::Timeout { details } => write!(f, "TIMEOUT | {details}"),
            Self::ValidationFailure { details } => {
                write!(f, "VALIDATION_FAILURE | {details}")
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ReplayConfig {
    n_points: usize,
    mode: ConstructionMode,
    debug_mode: DebugMode,
    distribution: PointDistribution,
    ball_radius: f64,
    box_half_width: f64,
    allow_skips: bool,
    skip_final_repair: bool,
    shuffle_seed: Option<u64>,
    progress_every: usize,
    validate_every: Option<usize>,
    repair_every: usize,
    repair_max_flips: Option<usize>,
    max_runtime_secs: usize,
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

/// Print a deterministic replay command that reproduces this exact run.
fn print_replay_command<const D: usize>(seed: u64, replay: &ReplayConfig) {
    let skips = if replay.allow_skips { "1" } else { "0" };
    let skip_final_repair = if replay.skip_final_repair { "1" } else { "0" };
    let cargo_test = if cfg!(debug_assertions) {
        "cargo test"
    } else {
        "cargo test --release"
    };
    println!("Replay command:");
    println!(
        "  DELAUNAY_LARGE_DEBUG_CONSTRUCTION_MODE={} \\",
        replay.mode.name()
    );
    println!(
        "  DELAUNAY_LARGE_DEBUG_DEBUG_MODE={} \\",
        replay.debug_mode.name()
    );
    println!("  DELAUNAY_LARGE_DEBUG_N_{D}D={} \\", replay.n_points);
    println!("  DELAUNAY_LARGE_DEBUG_CASE_SEED_{D}D=0x{seed:X} \\");
    println!(
        "  DELAUNAY_LARGE_DEBUG_DISTRIBUTION={} \\",
        replay.distribution.name()
    );
    println!(
        "  DELAUNAY_LARGE_DEBUG_BALL_RADIUS={} \\",
        replay.ball_radius
    );
    println!(
        "  DELAUNAY_LARGE_DEBUG_BOX_HALF_WIDTH={} \\",
        replay.box_half_width
    );
    if let Some(shuffle_seed) = replay.shuffle_seed {
        println!("  DELAUNAY_LARGE_DEBUG_SHUFFLE_SEED=0x{shuffle_seed:X} \\");
    }
    println!(
        "  DELAUNAY_LARGE_DEBUG_PROGRESS_EVERY={} \\",
        replay.progress_every
    );
    println!(
        "  DELAUNAY_LARGE_DEBUG_VALIDATE_EVERY={} \\",
        replay.validate_every.unwrap_or(0)
    );
    println!("  DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS={skips} \\");
    println!("  DELAUNAY_LARGE_DEBUG_SKIP_FINAL_REPAIR={skip_final_repair} \\");
    println!(
        "  DELAUNAY_LARGE_DEBUG_REPAIR_EVERY={} \\",
        replay.repair_every
    );
    println!(
        "  DELAUNAY_LARGE_DEBUG_REPAIR_MAX_FLIPS={} \\",
        replay.repair_max_flips.unwrap_or(0)
    );
    println!(
        "  DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS={} \\",
        replay.max_runtime_secs
    );
    println!(
        "  {cargo_test} --test large_scale_debug debug_large_scale_{D}d -- --ignored --nocapture"
    );
}

fn emit_outcome<const D: usize>(outcome: &DebugOutcome, seed: u64, replay: &ReplayConfig) {
    println!();
    println!("OUTCOME: {outcome}");
    println!();
    print_replay_command::<D>(seed, replay);
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
fn debug_large_case<const D: usize>(dimension_name: &str, default_n_points: usize) {
    init_tracing();

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
    let repair_max_flips =
        env_usize("DELAUNAY_LARGE_DEBUG_REPAIR_MAX_FLIPS").and_then(|v| (v > 0).then_some(v));

    // Delaunay repair scheduling
    // - 0 disables incremental repair
    // - 1 runs repair after every insertion
    // - N>1 runs repair after every N successful insertions
    let repair_every = env_usize("DELAUNAY_LARGE_DEBUG_REPAIR_EVERY").unwrap_or(128);
    let max_runtime_secs = env_usize("DELAUNAY_LARGE_DEBUG_MAX_RUNTIME_SECS").unwrap_or(600);
    let validate_every = env_usize("DELAUNAY_LARGE_DEBUG_VALIDATE_EVERY").or_else(|| {
        if matches!(debug_mode, DebugMode::Cadenced) {
            (repair_every != 0).then_some(repair_every)
        } else {
            None
        }
    });
    let replay_config = ReplayConfig {
        n_points,
        mode,
        debug_mode,
        distribution,
        ball_radius,
        box_half_width,
        allow_skips,
        skip_final_repair,
        shuffle_seed,
        progress_every,
        validate_every,
        repair_every,
        repair_max_flips,
        max_runtime_secs,
    };
    let timeout_details = format!(
        "wall time exceeded {max_runtime_secs} seconds (profile={}, D={D}, n_points={n_points}, mode={}, debug_mode={})",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        mode.name(),
        debug_mode.name(),
    );
    // Hold the sender for the lifetime of this function; dropping it at return
    // cancels the watchdog thread so it does not outlive this test.
    let _watchdog = (max_runtime_secs > 0)
        .then(|| install_runtime_cap(max_runtime_secs as u64, timeout_details));

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
                    let outcome = DebugOutcome::ConstructionFailure {
                        error: error.to_string(),
                    };
                    emit_outcome::<D>(&outcome, seed, &replay_config);
                    return;
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
            let mut last_progress_time = t_insert;
            let mut last_progress_inserted: usize = 0;

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
                        println!();
                        println!("Phase: construction (incremental insertion)");
                        print_insertion_summary(&summary, t_insert.elapsed());

                        if let Err(report) = dt.validation_report() {
                            print_validation_report(&report);
                        } else {
                            println!("validation_report: OK (after rollback)");
                        }

                        let outcome = DebugOutcome::ConstructionFailure {
                            error: format!(
                                "non-retryable insertion error at idx={idx} uuid={uuid} coords={coords:?}: {err}"
                            ),
                        };
                        emit_outcome::<D>(&outcome, seed, &replay_config);
                        return;
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
                    if let Err(report) = dt.validation_report() {
                        print_validation_report(&report);
                    }
                    let outcome = DebugOutcome::ConstructionFailure {
                        error: format!("topology validation failed at idx={idx}: {e}"),
                    };
                    emit_outcome::<D>(&outcome, seed, &replay_config);
                    return;
                }

                if (idx + 1) % progress_every == 0 {
                    let now = Instant::now();
                    let batch_elapsed = now.duration_since(last_progress_time);
                    let batch_inserted = summary.inserted.saturating_sub(last_progress_inserted);
                    let batch_f64: f64 = safe_usize_to_scalar(batch_inserted).unwrap_or(0.0);
                    let rate = if batch_elapsed.as_secs_f64() > 0.0 {
                        batch_f64 / batch_elapsed.as_secs_f64()
                    } else {
                        0.0
                    };
                    println!(
                        "progress: {}/{} inserted={} skipped={} cells={} elapsed={:?} [{batch_inserted} in {batch_elapsed:.1?}, {rate:.0} ins/s]",
                        idx + 1,
                        n_points,
                        summary.inserted,
                        summary.total_skipped(),
                        dt.number_of_cells(),
                        t_insert.elapsed()
                    );
                    last_progress_time = now;
                    last_progress_inserted = summary.inserted;
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
        let outcome = DebugOutcome::ConstructionFailure {
            error: format!(
                "{skipped_total} vertices were skipped (set DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1 to allow)"
            ),
        };
        emit_outcome::<D>(&outcome, seed, &replay_config);
        return;
    }

    if !skip_final_repair && dt.number_of_cells() > 0 {
        println!();
        println!("Running final flip-based repair (advanced)...");
        let t_repair = Instant::now();
        let repair_config = DelaunayRepairHeuristicConfig {
            max_flips: repair_max_flips,
            ..Default::default()
        };
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
            }
        }
        println!("repair wall time: {:?}", t_repair.elapsed());
    }

    let construction_wall = t_insert.elapsed();

    println!();
    println!("Running validation_report (Levels 1–4)...");
    let t_validate = Instant::now();
    let validation_result = dt.validation_report();
    let validate_wall = t_validate.elapsed();
    println!("validation_report wall time: {validate_wall:?}");

    let outcome = match validation_result {
        Ok(()) => {
            println!("validation_report: OK");
            DebugOutcome::Success {
                inserted: dt.number_of_vertices(),
                skipped: skipped_total,
                cells: dt.number_of_cells(),
            }
        }
        Err(report) => {
            print_validation_report(&report);
            let details = report.violations.first().map_or_else(
                || "no violations captured".to_string(),
                |v| format!("{:?}: {}", v.kind, v.error),
            );
            DebugOutcome::ValidationFailure { details }
        }
    };

    // Phase timing summary
    let total_wall = t_gen.elapsed();
    let gen_wall = total_wall
        .saturating_sub(construction_wall)
        .saturating_sub(validate_wall);
    println!();
    println!("Phase timing:");
    println!("  point_generation: {gen_wall:?}");
    println!("  construction:     {construction_wall:?}");
    println!("  validation:       {validate_wall:?}");
    println!("  total:            {total_wall:?}");

    println!();
    emit_outcome::<D>(&outcome, seed, &replay_config);
}

#[derive(Debug, Clone)]
struct IncrementalPrefixFailure<const D: usize> {
    prefix_len: usize,
    index: usize,
    uuid: uuid::Uuid,
    coords: [f64; D],
    error: String,
}

fn run_incremental_prefix<const D: usize>(
    vertices: &[Vertex<f64, (), D>],
    prefix_len: usize,
    _repair_every: usize,
) -> Result<(), IncrementalPrefixFailure<D>> {
    let kernel = RobustKernel::<f64>::new();
    let prefix = &vertices[..prefix_len];
    let mut dt = match DelaunayTriangulation::<RobustKernel<f64>, (), (), D>::with_topology_guarantee_and_options_with_construction_statistics(
        &kernel,
        prefix,
        TopologyGuarantee::PLManifoldStrict,
        ConstructionOptions::default(),
    ) {
        Ok((dt, _stats)) => dt,
        Err(err) => {
            let DelaunayTriangulationConstructionErrorWithStatistics {
                error, statistics, ..
            } = err;
            let idx = statistics
                .inserted
                .saturating_sub(1)
                .min(prefix_len.saturating_sub(1));
            let (uuid, coords) = prefix.get(idx).copied().map_or_else(
                || (uuid::Uuid::nil(), [0.0; D]),
                |vertex| (vertex.uuid(), *vertex.point().coords()),
            );
            return Err(IncrementalPrefixFailure {
                prefix_len,
                index: idx,
                uuid,
                coords,
                error: format!(
                    "{} [inserted={} skipped_duplicate={} skipped_degeneracy={}]",
                    error,
                    statistics.inserted,
                    statistics.skipped_duplicate,
                    statistics.skipped_degeneracy
                ),
            })
        }
    };

    let skipped_total = prefix_len.saturating_sub(dt.number_of_vertices());
    if !env_flag("DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS") && skipped_total > 0 {
        let idx = prefix_len.saturating_sub(1);
        let (uuid, coords) = prefix.get(idx).copied().map_or_else(
            || (uuid::Uuid::nil(), [0.0; D]),
            |vertex| (vertex.uuid(), *vertex.point().coords()),
        );
        return Err(IncrementalPrefixFailure {
            prefix_len,
            index: idx,
            uuid,
            coords,
            error: format!(
                "{skipped_total} vertices were skipped (set DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1 to allow)"
            ),
        });
    }

    if !env_flag("DELAUNAY_LARGE_DEBUG_SKIP_FINAL_REPAIR") && dt.number_of_cells() > 0 {
        let _ = dt.repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default());
    }

    if let Err(report) = dt.validation_report() {
        let idx = prefix_len.saturating_sub(1);
        let (uuid, coords) = prefix.get(idx).copied().map_or_else(
            || (uuid::Uuid::nil(), [0.0; D]),
            |vertex| (vertex.uuid(), *vertex.point().coords()),
        );
        let detail = report.violations.first().map_or_else(
            || "no violations captured".to_string(),
            |violation| format!("{:?}: {}", violation.kind, violation.error),
        );
        return Err(IncrementalPrefixFailure {
            prefix_len,
            index: idx,
            uuid,
            coords,
            error: format!("validation_report failed: {detail}"),
        });
    }

    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "Debug harness intentionally verbose for reproducibility and operator guidance"
)]
fn debug_large_scale_incremental_prefix_bisect<const D: usize>(
    dimension_name: &str,
    default_total_n: usize,
) {
    let total_n = env_usize("DELAUNAY_LARGE_DEBUG_PREFIX_TOTAL")
        .unwrap_or(default_total_n)
        .max(D + 1);
    let base_seed = env_u64("DELAUNAY_LARGE_DEBUG_SEED").unwrap_or(42);
    let case_seed = env_u64(&format!("DELAUNAY_LARGE_DEBUG_CASE_SEED_{D}D"))
        .or_else(|| env_u64("DELAUNAY_LARGE_DEBUG_CASE_SEED"))
        .unwrap_or_else(|| seed_for_case::<D>(base_seed, total_n));
    let ball_radius = env_f64("DELAUNAY_LARGE_DEBUG_BALL_RADIUS").unwrap_or(100.0);
    let repair_every = env_usize("DELAUNAY_LARGE_DEBUG_REPAIR_EVERY").unwrap_or(128);
    let max_probes = env_usize("DELAUNAY_LARGE_DEBUG_PREFIX_MAX_PROBES");
    let max_runtime_secs = env_usize("DELAUNAY_LARGE_DEBUG_PREFIX_MAX_RUNTIME_SECS").unwrap_or(0);

    println!("=============================================");
    println!("{dimension_name} incremental prefix bisect");
    println!("=============================================");
    println!("Config:");
    println!("  total_n:        {total_n}");
    println!("  base_seed:      0x{base_seed:X} ({base_seed})");
    println!("  case_seed:      0x{case_seed:X} ({case_seed})");
    println!("  ball_radius:    {ball_radius}");
    println!("  repair_every:   {repair_every}");
    println!("  probe_mode:     new (batch, matches debug_large_scale_3d default)");
    println!("  max_probes:     {max_probes:?}");
    println!("  max_runtime_secs:{max_runtime_secs}");
    println!();

    let points = generate_random_points_in_ball_seeded::<f64, D>(total_n, ball_radius, case_seed)
        .unwrap_or_else(|e| {
            panic!("failed to generate deterministic {dimension_name} ball points for bisect: {e}")
        });
    let vertices: Vec<Vertex<f64, (), D>> = points.into_iter().map(|p| vertex!(p)).collect();

    let t_bisect = Instant::now();
    let mut probe_count = 0usize;

    let mut run_probe = |prefix_len: usize| -> Option<Result<(), IncrementalPrefixFailure<D>>> {
        if let Some(limit) = max_probes
            && probe_count >= limit
        {
            println!(
                "Stopping early: reached DELAUNAY_LARGE_DEBUG_PREFIX_MAX_PROBES={limit} (elapsed {:?})",
                t_bisect.elapsed()
            );
            return None;
        }

        if max_runtime_secs > 0 && t_bisect.elapsed().as_secs() >= max_runtime_secs as u64 {
            println!(
                "Stopping early: reached DELAUNAY_LARGE_DEBUG_PREFIX_MAX_RUNTIME_SECS={} (probes={probe_count}, elapsed {:?})",
                max_runtime_secs,
                t_bisect.elapsed()
            );
            return None;
        }

        probe_count = probe_count.saturating_add(1);
        let t_probe = Instant::now();
        let result = run_incremental_prefix(&vertices, prefix_len, repair_every);
        println!(
            "  probe #{probe_count}: prefix_len={prefix_len} -> {} ({:?})",
            if result.is_err() { "FAIL" } else { "PASS" },
            t_probe.elapsed()
        );
        Some(result)
    };

    let first_failure = match run_probe(total_n) {
        None => return,
        Some(Ok(())) => {
            if let Err(mismatch) = run_incremental_prefix(&vertices, total_n, repair_every) {
                println!(
                    "HARNESS MISMATCH: bisect full-prefix probe passed but canonical full-prefix recheck failed."
                );
                println!(
                    "  mismatch details: idx={} uuid={} coords={:?} error={}",
                    mismatch.index, mismatch.uuid, mismatch.coords, mismatch.error
                );
                panic!("aborting: harness mismatch (bisect PASS vs canonical FAIL)");
            }
            println!("Canonical full-prefix recheck: PASS");
            println!(
                "No failure observed for full prefix total_n={total_n}; bisect skipped (likely fixed or total too small)."
            );
            println!(
                "Config recap: base_seed=0x{base_seed:X} case_seed=0x{case_seed:X} ball_radius={ball_radius} repair_every={repair_every} mode=new"
            );
            println!(
                "To force a failure, increase DELAUNAY_LARGE_DEBUG_PREFIX_TOTAL or adjust DELAUNAY_LARGE_DEBUG_CASE_SEED_{D}D."
            );
            return;
        }
        Some(Err(err)) => err,
    };

    let mut lo = D + 1;
    let mut hi = first_failure.prefix_len.max(lo);
    println!(
        "Full-run first failure: idx={} (prefix_len={})",
        first_failure.index, first_failure.prefix_len
    );
    println!("Initial binary-search range: [{lo}, {hi}]");

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let Some(result) = run_probe(mid) else {
            return;
        };
        let failed = result.is_err();

        if failed {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    let minimal_prefix = lo;
    let minimal_failure = match run_probe(minimal_prefix) {
        None => return,
        Some(Ok(())) => {
            panic!(
                "internal bisect inconsistency: expected failure at minimal_prefix={minimal_prefix}"
            )
        }
        Some(Err(err)) => err,
    };

    if minimal_prefix > D + 1 {
        assert!(
            run_probe(minimal_prefix - 1).is_some_and(|result| result.is_ok()),
            "internal bisect inconsistency: prefix {} should pass",
            minimal_prefix - 1
        );
    }

    println!();
    println!("Minimal failing prefix: {minimal_prefix}");
    println!(
        "Failure details: idx={} uuid={} coords={:?}",
        minimal_failure.index, minimal_failure.uuid, minimal_failure.coords
    );
    println!("Error: {}", minimal_failure.error);
    println!();
    println!("Replay command:");
    println!(
        "  DELAUNAY_LARGE_DEBUG_CONSTRUCTION_MODE=new DELAUNAY_LARGE_DEBUG_N_{D}D={minimal_prefix} DELAUNAY_LARGE_DEBUG_CASE_SEED_{D}D=0x{case_seed:X} DELAUNAY_REPAIR_DEBUG_FACETS=1 cargo test --test large_scale_debug debug_large_scale_{D}d -- --ignored --nocapture"
    );
}

#[test]
#[ignore = "large-scale debug harness (manual run)"]
fn debug_large_scale_3d_incremental_prefix_bisect() {
    init_tracing();
    debug_large_scale_incremental_prefix_bisect::<3>("3D", 1_000);
}

#[test]
#[ignore = "large-scale debug harness (manual run)"]
fn debug_large_scale_4d_incremental_prefix_bisect() {
    init_tracing();
    debug_large_scale_incremental_prefix_bisect::<4>("4D", 100);
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

/// Regression test for issue #230: 4D 100-point post-construction manifold failure.
///
/// At this seed, construction now inserts all 100 vertices without skips, but the
/// full PL-manifold / orientation validation can still fail after construction due to
/// a disconnected ridge link. This test keeps the exact seed under manual coverage
/// until the full 100-point case validates cleanly end to end.
///
/// Gated behind `slow-tests` and `#[ignore]` because 4D construction is expensive.
/// Run manually with:
/// ```bash
/// cargo test --release --test large_scale_debug --features slow-tests regression_issue_230 -- --ignored --nocapture
/// ```
#[cfg(feature = "slow-tests")]
#[test]
#[ignore = "4D 100-point construction is expensive (~30s release, minutes debug)"]
fn regression_issue_230_4d_100_orientation() {
    use delaunay::core::delaunay_triangulation::{
        ConstructionOptions, DelaunayTriangulationConstructionErrorWithStatistics,
    };
    use delaunay::core::triangulation::TopologyGuarantee;

    init_tracing();

    let seed = seed_for_case::<4>(0x2A, 100);
    let points = generate_random_points_in_ball_seeded::<f64, 4>(100, 100.0, seed)
        .expect("point generation should succeed");
    let vertices: Vec<Vertex<f64, (), 4>> = points.into_iter().map(|p| vertex!(p)).collect();

    let kernel = RobustKernel::<f64>::new();
    let result = DelaunayTriangulation::<_, (), (), 4>::with_topology_guarantee_and_options_with_construction_statistics(
        &kernel,
        &vertices,
        TopologyGuarantee::PLManifoldStrict,
        ConstructionOptions::default(),
    );

    match result {
        Ok((dt, stats)) => {
            assert!(
                dt.as_triangulation().validate().is_ok(),
                "Topology validation (L1-L3) must pass (#230 regression, seed=0x{seed:X})"
            );
            assert_eq!(
                stats.total_skipped(),
                0,
                "#230 regression must not skip vertices (inserted={} skipped={}, seed=0x{seed:X})",
                stats.inserted,
                stats.total_skipped()
            );
            assert_eq!(
                stats.inserted,
                vertices.len(),
                "#230 regression must insert all vertices (inserted={} expected={}, seed=0x{seed:X})",
                stats.inserted,
                vertices.len()
            );
            assert!(
                dt.validation_report().is_ok(),
                "Full validation report (Levels 1-4) must pass (#230 regression, seed=0x{seed:X})"
            );
            println!(
                "#230 regression: inserted={} skipped={} (seed=0x{seed:X})",
                stats.inserted,
                stats.total_skipped()
            );
        }
        Err(e) => {
            let DelaunayTriangulationConstructionErrorWithStatistics {
                error, statistics, ..
            } = e;
            panic!(
                "#230 regression: unexpected construction failure (inserted={} skipped={}): {error}",
                statistics.inserted,
                statistics.total_skipped()
            );
        }
    }
}

#[test]
#[ignore = "large-scale debug harness (manual run)"]
fn debug_large_scale_2d() {
    debug_large_case::<2>("2D", 10_000);
}

#[test]
#[ignore = "large-scale debug harness (manual run)"]
fn debug_large_scale_3d() {
    debug_large_case::<3>("3D", 10_000);
}

#[test]
#[ignore = "large-scale debug harness (manual run)"]
fn debug_large_scale_4d() {
    debug_large_case::<4>("4D", 3_000);
}

#[test]
#[ignore = "large-scale debug harness (manual run)"]
fn debug_large_scale_5d() {
    debug_large_case::<5>("5D", 1_000);
}
