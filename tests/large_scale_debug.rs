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
//! cargo test --test large_scale_debug debug_large_scale_4d -- --ignored --nocapture
//! ```

#![forbid(unsafe_code)]

use delaunay::core::delaunay_triangulation::{
    ConstructionOptions, ConstructionStatistics, DelaunayRepairHeuristicConfig,
    DelaunayTriangulationConstructionErrorWithStatistics,
};
use delaunay::geometry::kernel::FastKernel;
use delaunay::geometry::util::{
    generate_random_points_in_ball_seeded, generate_random_points_seeded,
};
use delaunay::prelude::triangulation::*;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

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
                let coords: [f64; D] = s.coords.as_slice().try_into().ok()?;
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
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok())
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

fn seed_for_case<const D: usize>(base_seed: u64, n_points: usize) -> u64 {
    const SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
    base_seed
        .wrapping_add((n_points as u64).wrapping_mul(SEED_SALT))
        .wrapping_add((D as u64).wrapping_mul(SEED_SALT.rotate_left(17)))
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

    let shuffle_seed = env_u64("DELAUNAY_LARGE_DEBUG_SHUFFLE_SEED");
    let progress_every = env_usize("DELAUNAY_LARGE_DEBUG_PROGRESS_EVERY")
        .unwrap_or(1000)
        .max(1);
    let validate_every = env_usize("DELAUNAY_LARGE_DEBUG_VALIDATE_EVERY");

    let allow_skips = env_flag("DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS");
    let skip_final_repair = env_flag("DELAUNAY_LARGE_DEBUG_SKIP_FINAL_REPAIR");

    // Delaunay repair scheduling: run a bounded local flip-repair pass every N successful insertions.
    // - 0 disables incremental repair
    // - 1 runs repair after every insertion
    // - N>1 runs repair after every N successful insertions
    let repair_every = env_usize("DELAUNAY_LARGE_DEBUG_REPAIR_EVERY").unwrap_or(128);

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
    println!("  shuffle_seed:  {shuffle_seed:?}");
    println!("  progress_every:{progress_every}");
    println!("  validate_every:{validate_every:?}");
    println!("  allow_skips:   {allow_skips}");
    println!("  skip_final_repair: {skip_final_repair}");
    println!("  repair_every:  {repair_every}");
    println!();

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

    let mut vertices: Vec<Vertex<f64, (), D>> = points.into_iter().map(|p| vertex!(p)).collect();

    let t_insert = Instant::now();

    let mut dt: DelaunayTriangulation<_, (), (), D> = match mode {
        ConstructionMode::New => {
            // `DelaunayTriangulation::new()` applies Hilbert ordering during batch construction.
            // Use the statistics-returning variant so we can report aggregate insertion telemetry.
            //
            // Use PLManifoldStrict during batch construction to ensure vertex-link invariants are
            // maintained on each insertion.
            let kernel = FastKernel::<f64>::new();
            match DelaunayTriangulation::with_topology_guarantee_and_options_with_construction_statistics(
                &kernel,
                &vertices,
                TopologyGuarantee::PLManifoldStrict,
                ConstructionOptions::default(),
            ) {
                Ok((dt, stats)) => {
                    let summary: InsertionSummary<D> = stats.into();
                    print_insertion_summary(&summary, t_insert.elapsed());
                    dt
                }
                Err(e) => {
                    let DelaunayTriangulationConstructionErrorWithStatistics {
                        error,
                        statistics,
                        ..
                    } = e;
                    let summary: InsertionSummary<D> = (*statistics).into();
                    print_insertion_summary(&summary, t_insert.elapsed());
                    println!("construction failed: {error}");
                    panic!("aborting: batch construction failed");
                }
            }
        }
        ConstructionMode::Incremental => {
            if let Some(shuffle_seed) = shuffle_seed {
                let mut rng = StdRng::seed_from_u64(shuffle_seed);
                vertices.shuffle(&mut rng);
                println!("Shuffled insertion order with seed {shuffle_seed}");
            }

            let mut dt: DelaunayTriangulation<_, (), (), D> =
                DelaunayTriangulation::empty_with_topology_guarantee(
                    TopologyGuarantee::PLManifoldStrict,
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

            // In incremental mode we want strict topology guarantees to be enforced immediately,
            // so force validation to run after each insertion.
            dt.set_validation_policy(ValidationPolicy::Always);

            println!("Policies:");
            println!("  topology_guarantee:   {:?}", dt.topology_guarantee());
            println!("  validation_policy:    {:?}", dt.validation_policy());
            println!("  delaunay_repair_policy:{:?}", dt.delaunay_repair_policy());
            println!("  delaunay_check_policy: {:?}", dt.delaunay_check_policy());
            println!();

            let mut summary: InsertionSummary<D> = InsertionSummary::default();
            let mut had_cells = false;

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

                        panic!("aborting: non-retryable insertion error");
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
                    panic!("aborting: topology validation failure");
                }

                if (idx + 1) % progress_every == 0 {
                    println!(
                        "progress: {}/{} inserted={} skipped={} cells={} elapsed={:?}",
                        idx + 1,
                        n_points,
                        summary.inserted,
                        summary.total_skipped(),
                        dt.number_of_cells(),
                        t_insert.elapsed()
                    );
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

    assert!(
        allow_skips || skipped_total == 0,
        "{skipped_total} vertices were skipped (set DELAUNAY_LARGE_DEBUG_ALLOW_SKIPS=1 to allow)"
    );

    if !skip_final_repair && dt.number_of_cells() > 0 {
        println!();
        println!("Running final flip-based repair (advanced)...");
        let t_repair = Instant::now();
        match dt.repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default()) {
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

    println!();
    println!("Running validation_report (Levels 1–4)...");
    match dt.validation_report() {
        Ok(()) => println!("validation_report: OK"),
        Err(report) => {
            print_validation_report(&report);
            panic!("aborting: validation_report failed");
        }
    }

    println!();
    println!("Total wall time: {:?}", t_gen.elapsed());
}
#[derive(Debug, Clone)]
struct IncrementalFailure3d {
    prefix_len: usize,
    index: usize,
    uuid: uuid::Uuid,
    coords: [f64; 3],
    error: String,
}

fn run_incremental_prefix_3d(
    vertices: &[Vertex<f64, (), 3>],
    prefix_len: usize,
    repair_every: usize,
) -> Result<(), IncrementalFailure3d> {
    let mut dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::empty_with_topology_guarantee(TopologyGuarantee::PLManifoldStrict);

    let repair_policy = match NonZeroUsize::new(repair_every) {
        None => DelaunayRepairPolicy::Never,
        Some(n) if n.get() == 1 => DelaunayRepairPolicy::EveryInsertion,
        Some(n) => DelaunayRepairPolicy::EveryN(n),
    };
    dt.set_delaunay_repair_policy(repair_policy);
    dt.set_delaunay_check_policy(DelaunayCheckPolicy::EndOnly);
    dt.set_validation_policy(ValidationPolicy::Always);

    for (idx, vertex) in vertices.iter().copied().take(prefix_len).enumerate() {
        let uuid = vertex.uuid();
        let coords = *vertex.point().coords();

        match dt.insert_with_statistics(vertex) {
            Ok((InsertionOutcome::Inserted { .. } | InsertionOutcome::Skipped { .. }, _)) => {}
            Err(err) => {
                return Err(IncrementalFailure3d {
                    prefix_len,
                    index: idx,
                    uuid,
                    coords,
                    error: err.to_string(),
                });
            }
        }
    }

    Ok(())
}

#[test]
#[ignore = "large-scale debug harness (manual run)"]
fn debug_large_scale_3d_incremental_prefix_bisect() {
    init_tracing();

    let total_n = env_usize("DELAUNAY_LARGE_DEBUG_PREFIX_TOTAL")
        .unwrap_or(1000)
        .max(4);
    let base_seed = env_u64("DELAUNAY_LARGE_DEBUG_SEED").unwrap_or(42);
    let case_seed = env_u64("DELAUNAY_LARGE_DEBUG_CASE_SEED_3D")
        .or_else(|| env_u64("DELAUNAY_LARGE_DEBUG_CASE_SEED"))
        .unwrap_or_else(|| seed_for_case::<3>(base_seed, total_n));
    let ball_radius = env_f64("DELAUNAY_LARGE_DEBUG_BALL_RADIUS").unwrap_or(100.0);
    let repair_every = env_usize("DELAUNAY_LARGE_DEBUG_REPAIR_EVERY").unwrap_or(128);

    println!("=============================================");
    println!("3D incremental prefix bisect");
    println!("=============================================");
    println!("Config:");
    println!("  total_n:        {total_n}");
    println!("  base_seed:      0x{base_seed:X} ({base_seed})");
    println!("  case_seed:      0x{case_seed:X} ({case_seed})");
    println!("  ball_radius:    {ball_radius}");
    println!("  repair_every:   {repair_every}");
    println!();

    let points = generate_random_points_in_ball_seeded::<f64, 3>(total_n, ball_radius, case_seed)
        .unwrap_or_else(|e| {
            panic!("failed to generate deterministic 3D ball points for bisect: {e}")
        });
    let vertices: Vec<Vertex<f64, (), 3>> = points.into_iter().map(|p| vertex!(p)).collect();

    let first_failure = match run_incremental_prefix_3d(&vertices, total_n, repair_every) {
        Ok(()) => {
            println!(
                "No failure observed for full prefix total_n={total_n}; bisect skipped (likely fixed or total too small)."
            );
            println!(
                "Config recap: base_seed=0x{base_seed:X} case_seed=0x{case_seed:X} ball_radius={ball_radius} repair_every={repair_every}"
            );
            println!(
                "To force a failure, increase DELAUNAY_LARGE_DEBUG_PREFIX_TOTAL or adjust DELAUNAY_LARGE_DEBUG_CASE_SEED_3D."
            );
            return;
        }
        Err(err) => err,
    };

    let mut lo = 4;
    let mut hi = (first_failure.index + 1).max(lo);
    println!(
        "Full-run first failure: idx={} (prefix_len={})",
        first_failure.index, first_failure.prefix_len
    );
    println!("Initial binary-search range: [{lo}, {hi}]");

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let failed = run_incremental_prefix_3d(&vertices, mid, repair_every).is_err();
        println!(
            "  bisect probe: prefix_len={mid} -> {}",
            if failed { "FAIL" } else { "PASS" }
        );

        if failed {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    let minimal_prefix = lo;
    let minimal_failure = run_incremental_prefix_3d(&vertices, minimal_prefix, repair_every)
        .err()
        .unwrap_or_else(|| {
            panic!(
                "internal bisect inconsistency: expected failure at minimal_prefix={minimal_prefix}"
            )
        });

    if minimal_prefix > 4 {
        assert!(
            run_incremental_prefix_3d(&vertices, minimal_prefix - 1, repair_every).is_ok(),
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
        "  DELAUNAY_LARGE_DEBUG_CONSTRUCTION_MODE=incremental DELAUNAY_LARGE_DEBUG_N_3D={minimal_prefix} DELAUNAY_LARGE_DEBUG_CASE_SEED_3D=0x{case_seed:X} DELAUNAY_REPAIR_DEBUG_FACETS=1 cargo test --test large_scale_debug debug_large_scale_3d -- --ignored --nocapture"
    );
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
