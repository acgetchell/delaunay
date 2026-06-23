//! Comprehensive profiling suite for optimization work.
//!
//! This benchmark suite provides manual performance diagnostics that are too
//! expensive or too specialized for the regular CI regression contract:
//!
//! 1. **Large-scale triangulation performance** across 2D through 5D
//! 2. **RSS memory deltas** during construction
//! 3. **Allocation profiling** with the `count-allocations` feature
//! 4. **Query and iteration latency** on constructed triangulations
//! 5. **Algorithmic bottleneck identification** for focused operations
//! 6. **Validation layer diagnostics** separating topology and Delaunay costs
//!
//! ## Usage
//!
//! Run the comprehensive profiling harness:
//! ```bash
//! cargo bench --profile perf --bench profiling_suite
//! ```
//!
//! Run specific profiling categories:
//! ```bash
//! # Large-scale construction only
//! cargo bench --profile perf --bench profiling_suite -- construction
//!
//! # Allocation profiling only
//! cargo bench --profile perf --bench profiling_suite -- memory_profiling
//!
//! # Query and iteration profiling only
//! cargo bench --profile perf --bench profiling_suite -- queries
//! cargo bench --profile perf --bench profiling_suite -- query_latency
//! ```
//!
//! ## Scale Modes
//!
//! The default large-scale counts are 1K/5K/10K for 2D and 3D, 1K/3K for
//! 4D, and 500/1K for 5D. Enable larger 4D runs on dedicated hardware:
//! ```bash
//! BENCH_LARGE_SCALE=1 cargo bench --profile perf --bench profiling_suite
//! ```
//!
//! `PROFILING_DEV_MODE=1` still reduces the auxiliary profiling groups for
//! faster samply/flamegraph iteration.
//!
//! ## Environment Variable Configuration
//!
//! The benchmark suite supports several environment variables for customization:
//!
//! - `BENCH_LARGE_SCALE`: Set to "1", "true", "yes", or "on" for larger
//!   4D large-scale counts.
//! - `PROFILING_DEV_MODE`: Set to "1", "true", "yes", or "on" for reduced
//!   auxiliary profiling groups.
//! - `BENCH_MEASUREMENT_TIME`: Override measurement time in seconds (minimum: 1, guards against invalid values)
//! - `BENCH_PERCENTILE`: Configure percentile for memory analysis (1-100, default: 95)
//! - `BENCH_SAMPLE_SIZE`: Override Criterion sample size (default: 10; values below 10 are clamped to 10)
//! - `BENCH_WARMUP_SECS`: Override Criterion warm-up time in seconds (default: 10)
//! - `DELAUNAY_BENCH_SEED`: Override the deterministic random seed
//! - `DELAUNAY_BENCH_RETRY_ATTEMPTS`: Override shuffled construction retries
//!
//! Example with custom configuration:
//! ```bash
//! BENCH_SAMPLE_SIZE=10 BENCH_WARMUP_SECS=5 BENCH_PERCENTILE=90 cargo bench --profile perf --bench profiling_suite
//! ```

use std::env;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::{Mutex, Once, OnceLock};
use std::time::{Duration, Instant};

#[cfg(all(feature = "count-allocations", feature = "bench-logging"))]
use allocation_counter::AllocationInfo;
#[cfg(feature = "count-allocations")]
use allocation_counter::measure;
use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use delaunay::prelude::collections::SmallBuffer;
use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayTriangulation, DelaunayTriangulationBuilder, RetryPolicy, Vertex,
};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateRange, Point};
use delaunay::prelude::query::*;
use delaunay::try_vertices_from_points;
use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System, get_current_pid};

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, abort_benchmark};

/// Builds a benchmark point from hard-coded finite coordinates.
fn finite_point<const D: usize>(coords: [f64; D]) -> Point<D> {
    Point::try_new(coords).unwrap_or_else(|_| std::process::abort())
}

/// Parses a non-zero retry count used by fixed benchmark policies.
fn retry_attempts(value: usize) -> NonZeroUsize {
    let Some(attempts) = NonZeroUsize::new(value) else {
        unreachable!("hard-coded retry attempt count must be non-zero");
    };
    attempts
}

/// Parses benchmark coordinate bounds and aborts if the fixture is invalid.
fn coordinate_range(min: f64, max: f64) -> CoordinateRange<f64> {
    CoordinateRange::try_new(min, max).or_abort()
}

/// Returns broad bounds used by general random benchmark point clouds.
fn wide_bounds() -> CoordinateRange<f64> {
    coordinate_range(-100.0, 100.0)
}

/// Returns compact bounds used by adversarial benchmark point clouds.
fn adversarial_bounds() -> CoordinateRange<f64> {
    coordinate_range(-1.0, 1.0)
}

#[cfg(feature = "bench-logging")]
fn init_tracing() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    });
}

#[cfg(not(feature = "bench-logging"))]
const fn init_tracing() {}

macro_rules! bench_info {
    ($($arg:tt)*) => {{
        #[cfg(feature = "bench-logging")]
        {
            init_tracing();
            tracing::info!($($arg)*);
        }
    }};
}

// SmallBuffer size constants for different use cases
#[cfg(all(feature = "count-allocations", feature = "bench-logging"))]
const BENCHMARK_ITERATION_BUFFER_SIZE: usize = 8; // For tracking allocation info across benchmark iterations
const SIMPLEX_VERTICES_BUFFER_SIZE: usize = 4; // 3D simplex = 4 vertices
const QUERY_RESULTS_BUFFER_SIZE: usize = 1024; // For bounded query result collections (max 1000 in code)

// Reusable seeds and caps
const DEFAULT_SEED: u64 = 42;
const QUERY_SEED: u64 = 123;
const MAX_QUERY_RESULTS: usize = 1_000;
const VALIDATION_SEED_SEARCH_LIMIT: u64 = 64;

/// Memory usage information for benchmarking, in KiB.
#[cfg_attr(
    not(feature = "bench-logging"),
    expect(
        dead_code,
        reason = "Memory fields are unpacked by optional bench diagnostics"
    )
)]
#[derive(Debug, Clone)]
struct MemoryInfo {
    before: u64,
    after: u64,
    delta: i64,
    tds_delta: i64,
}

#[cfg(not(feature = "count-allocations"))]
#[derive(Debug, Default)]
struct AllocationInfo;

#[cfg(not(feature = "count-allocations"))]
fn measure(f: impl FnOnce()) -> AllocationInfo {
    f();
    AllocationInfo
}

#[cfg(not(feature = "count-allocations"))]
fn print_alloc_banner_once() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        #[cfg(feature = "bench-logging")]
        println!("allocation stats unavailable: count-allocations feature disabled");
    });
}

/// Parse `PROFILING_DEV_MODE` environment variable as boolean-like value
/// Returns true for: "1", "true", "TRUE", "yes", "on" (case-insensitive)
/// Returns false for anything else (including "0", "false", empty, or unset)
fn is_dev_mode() -> bool {
    let dev = env::var("PROFILING_DEV_MODE").ok();
    dev.as_deref().is_some_and(|s| {
        s == "1"
            || s.eq_ignore_ascii_case("true")
            || s.eq_ignore_ascii_case("yes")
            || s.eq_ignore_ascii_case("on")
    })
}

/// Parse `BENCH_LARGE_SCALE` as a boolean-like value.
fn is_large_scale_mode() -> bool {
    let large = env::var("BENCH_LARGE_SCALE").ok();
    large.as_deref().is_some_and(|s| {
        s == "1"
            || s.eq_ignore_ascii_case("true")
            || s.eq_ignore_ascii_case("yes")
            || s.eq_ignore_ascii_case("on")
    })
}

/// Return current process memory usage in KiB.
fn memory_usage_kib() -> u64 {
    static SYS: OnceLock<Mutex<System>> = OnceLock::new();
    static UNIT_LOGGED: Once = Once::new();

    UNIT_LOGGED.call_once(|| {
        bench_info!("Memory measurements in KiB (sysinfo::Process::memory() / 1024)");
    });

    let pid = get_current_pid().or_abort();
    let sys = SYS.get_or_init(|| {
        Mutex::new(System::new_with_specifics(
            RefreshKind::nothing().with_processes(ProcessRefreshKind::nothing().with_memory()),
        ))
    });
    let mut system = sys.lock().or_abort();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    system
        .process(pid)
        .map_or(0, |process| process.memory() / 1024)
}

/// Return the deterministic base seed for random point generation.
fn benchmark_seed() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| {
        let seed = env::var("DELAUNAY_BENCH_SEED")
            .ok()
            .and_then(|s| {
                let s = s.trim();
                s.strip_prefix("0x")
                    .or_else(|| s.strip_prefix("0X"))
                    .map_or_else(|| s.parse().ok(), |hex| u64::from_str_radix(hex, 16).ok())
            })
            .unwrap_or(DEFAULT_SEED);

        if env::var("PRINT_BENCH_SEED").is_ok() {
            bench_info!("Benchmark seed: 0x{seed:X} ({seed})");
        }

        seed
    })
}

fn benchmark_retry_attempts() -> NonZeroUsize {
    static ATTEMPTS: OnceLock<NonZeroUsize> = OnceLock::new();
    *ATTEMPTS.get_or_init(|| {
        let attempts = env::var("DELAUNAY_BENCH_RETRY_ATTEMPTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(6)
            .max(1);

        let Some(attempts) = NonZeroUsize::new(attempts) else {
            unreachable!("attempts clamped to >= 1");
        };
        attempts
    })
}

fn seed_for_case<const D: usize>(n_points: usize) -> u64 {
    const SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

    let base_seed = benchmark_seed();
    base_seed
        .wrapping_add((n_points as u64).wrapping_mul(SEED_SALT))
        .wrapping_add((D as u64).wrapping_mul(SEED_SALT.rotate_left(17)))
}

fn construction_options(seed: u64) -> ConstructionOptions {
    ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
        attempts: benchmark_retry_attempts(),
        base_seed: Some(seed),
    })
}

fn construct_triangulation<const D: usize>(
    vertices: &[Vertex<(), D>],
    seed: u64,
) -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D> {
    DelaunayTriangulation::try_new_with_options(vertices, construction_options(seed)).or_abort()
}

/// Converts generated point fixtures into benchmark vertices without attaching data.
fn benchmark_vertices_from_generated_points<const D: usize>(
    points: &[Point<D>],
) -> Vec<Vertex<(), D>> {
    try_vertices_from_points(points).or_abort()
}

/// Generates deterministic benchmark points inside validated bounds.
fn generated_points_in_range<const D: usize>(
    count: usize,
    bounds: CoordinateRange<f64>,
    seed: u64,
) -> Vec<Point<D>> {
    generate_random_points_in_range_seeded::<D>(count, bounds, seed).or_abort()
}

/// Measure memory delta during triangulation construction.
fn measure_construction_with_memory<const D: usize>(n_points: usize, seed: u64) -> MemoryInfo {
    let mem_before = memory_usage_kib();
    let points = generated_points_in_range::<D>(n_points, wide_bounds(), seed);
    let vertices = benchmark_vertices_from_generated_points(&points);

    let mem_before_tds = memory_usage_kib();
    let dt = construct_triangulation::<D>(&vertices, seed);
    let mem_after = memory_usage_kib();
    black_box(&dt);

    let delta_i128 = i128::from(mem_after) - i128::from(mem_before);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped to i64 range before casting"
    )]
    let delta = delta_i128.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;

    let tds_delta_i128 = i128::from(mem_after) - i128::from(mem_before_tds);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped to i64 range before casting"
    )]
    let tds_delta = tds_delta_i128.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;

    MemoryInfo {
        before: mem_before,
        after: mem_after,
        delta,
        tds_delta,
    }
}

/// Helper function to parse benchmark measurement time from environment
/// Guards against zero/invalid values by ensuring minimum of 1 second
fn bench_time(default_secs: u64) -> Duration {
    let secs = env::var("BENCH_MEASUREMENT_TIME")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map_or_else(|| default_secs.max(1), |parsed| parsed.max(1));
    Duration::from_secs(secs)
}

/// Point distribution types for comprehensive testing
#[derive(Debug, Clone, Copy)]
enum PointDistribution {
    Random,
    Adversarial,
}

/// Generate points according to the specified distribution
fn gen_points<const D: usize>(
    count: usize,
    distribution: PointDistribution,
    seed: u64,
) -> Vec<Point<D>> {
    match distribution {
        PointDistribution::Random => generated_points_in_range(count, wide_bounds(), seed),
        PointDistribution::Adversarial => generated_points_in_range::<D>(
            count,
            adversarial_bounds(),
            seed ^ 0xA5A5_A5A5_A5A5_A5A5,
        )
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let index = u32::try_from(index).or_abort();
            let mut coords = [0.0_f64; D];
            for (axis, coord) in coords.iter_mut().enumerate() {
                let axis_number = u32::try_from(axis + 1).or_abort();
                let base: f64 = point.coords()[axis];
                let cluster_offset = f64::from(index % 7) * 1.0e-3;
                let axis_offset = f64::from(axis_number) * 0.25;
                let perturbation = f64::from((index + axis_number) % 11) * 1.0e-6;
                *coord = base.mul_add(1.0e3, 1.0e9 + axis_offset + cluster_offset + perturbation);
            }
            finite_point(coords)
        })
        .collect(),
    }
}

// ============================================================================
// Large-Scale Triangulation Performance Profiling
// ============================================================================

/// Benchmark triangulation construction time for one dimension/count pair.
fn bench_construction<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize) {
    let bench_name = format!("construction/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.throughput(Throughput::Elements(n_points as u64));
    group.measurement_time(bench_time(5));

    let seed = seed_for_case::<D>(n_points);

    if (D == 4 && n_points >= 5000) || D == 5 {
        group.sample_size(10);
        group.measurement_time(bench_time(120));
    }

    group.bench_function("construct", |b| {
        b.iter_batched(
            || {
                let points = generated_points_in_range::<D>(n_points, wide_bounds(), seed);
                benchmark_vertices_from_generated_points(&points)
            },
            |vertices| {
                let dt = construct_triangulation::<D>(black_box(&vertices), seed);
                black_box(dt)
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Benchmark process RSS deltas during construction.
fn bench_rss_memory_usage<const D: usize>(
    c: &mut Criterion,
    dimension_name: &str,
    n_points: usize,
) {
    let bench_name = format!("memory/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.sample_size(10);
    group.measurement_time(bench_time(5));

    let seed = seed_for_case::<D>(n_points);

    #[cfg(feature = "bench-logging")]
    if env::var_os("BENCH_PRINT_MEM").is_some() {
        let mem_info = measure_construction_with_memory::<D>(n_points, seed);
        bench_info!(
            "Memory sample: before={} KiB, after={} KiB, delta={} KiB (TDS-only: {} KiB)",
            mem_info.before,
            mem_info.after,
            mem_info.delta,
            mem_info.tds_delta
        );
    }

    let _ = memory_usage_kib();

    group.bench_function("construction_memory_delta", |b| {
        b.iter(|| {
            let mem_info = measure_construction_with_memory::<D>(n_points, seed);
            black_box(mem_info)
        });
    });

    group.finish();
}

/// Benchmark topology validation over a prebuilt triangulation.
fn bench_validation<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize) {
    let bench_name = format!("validation/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.measurement_time(bench_time(5));

    if n_points >= 5000 || D == 5 {
        group.sample_size(10);
        group.measurement_time(bench_time(120));
    }

    let seed = seed_for_case::<D>(n_points);
    let points = generated_points_in_range::<D>(n_points, wide_bounds(), seed);
    let vertices = benchmark_vertices_from_generated_points(&points);
    let dt = construct_triangulation::<D>(&vertices, seed);
    let tri = dt.as_triangulation();

    group.throughput(Throughput::Elements(tri.number_of_simplices() as u64));

    group.bench_function("validate_topology", |b| {
        b.iter(|| {
            if let Err(error) = tri.is_valid() {
                abort_benchmark(format_args!(
                    "triangulation should be structurally valid during validation benchmark: {error}"
                ));
            }
        });
    });

    group.finish();
}

/// Benchmark neighbor lookup over all simplices.
fn bench_neighbor_queries<const D: usize>(
    c: &mut Criterion,
    dimension_name: &str,
    n_points: usize,
) {
    let bench_name = format!("queries/neighbors/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.measurement_time(bench_time(5));

    if D == 5 || (D == 4 && n_points >= 5000) {
        group.sample_size(10);
        group.measurement_time(bench_time(120));
    }

    let seed = seed_for_case::<D>(n_points);
    let points = generated_points_in_range::<D>(n_points, wide_bounds(), seed);
    let vertices = benchmark_vertices_from_generated_points(&points);
    let dt = construct_triangulation::<D>(&vertices, seed);
    let tds = dt.tds();
    let simplex_keys: Vec<_> = tds.simplex_keys().collect();

    group.throughput(Throughput::Elements(simplex_keys.len() as u64));

    group.bench_function("find_neighbors_all_simplices", |b| {
        b.iter(|| {
            for &simplex_key in &simplex_keys {
                let neighbors = tds.find_neighbors_by_key(simplex_key);
                black_box(neighbors);
            }
        });
    });

    group.finish();
}

/// Benchmark vertex iteration over a prebuilt triangulation.
fn bench_vertex_iteration<const D: usize>(
    c: &mut Criterion,
    dimension_name: &str,
    n_points: usize,
) {
    let bench_name = format!("queries/vertices/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.throughput(Throughput::Elements(n_points as u64));
    group.measurement_time(bench_time(5));

    if D == 5 || (D == 4 && n_points >= 5000) {
        group.sample_size(10);
        group.measurement_time(bench_time(120));
    }

    let seed = seed_for_case::<D>(n_points);
    let points = generated_points_in_range::<D>(n_points, wide_bounds(), seed);
    let vertices = benchmark_vertices_from_generated_points(&points);
    let dt = construct_triangulation::<D>(&vertices, seed);
    let tds = dt.tds();

    group.bench_function("iterate_all_vertices", |b| {
        b.iter(|| {
            let mut count = 0;
            for (_, vertex) in tds.vertices() {
                black_box(vertex);
                count += 1;
            }
            black_box(count)
        });
    });

    group.finish();
}

/// Benchmark simplex-key iteration over a prebuilt triangulation.
fn bench_simplex_iteration<const D: usize>(
    c: &mut Criterion,
    dimension_name: &str,
    n_points: usize,
) {
    let bench_name = format!("queries/simplices/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.measurement_time(bench_time(5));

    if D == 5 || (D == 4 && n_points >= 5000) {
        group.sample_size(10);
        group.measurement_time(bench_time(120));
    }

    let seed = seed_for_case::<D>(n_points);
    let points = generated_points_in_range::<D>(n_points, wide_bounds(), seed);
    let vertices = benchmark_vertices_from_generated_points(&points);
    let dt = construct_triangulation::<D>(&vertices, seed);
    let tds = dt.tds();

    group.throughput(Throughput::Elements(tds.number_of_simplices() as u64));

    group.bench_function("iterate_all_simplices", |b| {
        b.iter(|| {
            let mut count = 0;
            for simplex_key in tds.simplex_keys() {
                black_box(simplex_key);
                count += 1;
            }
            black_box(count)
        });
    });

    group.finish();
}

fn bench_2d_suite(c: &mut Criterion) {
    for &n_points in &[1000, 5000, 10_000] {
        bench_construction::<2>(c, "2D", n_points);
        bench_rss_memory_usage::<2>(c, "2D", n_points);
        bench_validation::<2>(c, "2D", n_points);
        bench_neighbor_queries::<2>(c, "2D", n_points);
        bench_vertex_iteration::<2>(c, "2D", n_points);
        bench_simplex_iteration::<2>(c, "2D", n_points);
    }
}

fn bench_3d_suite(c: &mut Criterion) {
    for &n_points in &[1000, 5000, 10_000] {
        bench_construction::<3>(c, "3D", n_points);
        bench_rss_memory_usage::<3>(c, "3D", n_points);
        bench_validation::<3>(c, "3D", n_points);
        bench_neighbor_queries::<3>(c, "3D", n_points);
        bench_vertex_iteration::<3>(c, "3D", n_points);
        bench_simplex_iteration::<3>(c, "3D", n_points);
    }
}

fn bench_4d_suite(c: &mut Criterion) {
    let point_counts: &[usize] = if is_large_scale_mode() {
        &[1000, 5000, 10_000]
    } else {
        &[1000, 3000]
    };

    for &n_points in point_counts {
        bench_construction::<4>(c, "4D", n_points);
        bench_rss_memory_usage::<4>(c, "4D", n_points);
        bench_validation::<4>(c, "4D", n_points);
        bench_neighbor_queries::<4>(c, "4D", n_points);
        bench_vertex_iteration::<4>(c, "4D", n_points);
        bench_simplex_iteration::<4>(c, "4D", n_points);
    }
}

fn bench_5d_suite(c: &mut Criterion) {
    for &n_points in &[500, 1000] {
        bench_construction::<5>(c, "5D", n_points);
        bench_rss_memory_usage::<5>(c, "5D", n_points);
        bench_validation::<5>(c, "5D", n_points);
        bench_neighbor_queries::<5>(c, "5D", n_points);
        bench_vertex_iteration::<5>(c, "5D", n_points);
        bench_simplex_iteration::<5>(c, "5D", n_points);
    }
}

// ============================================================================
// Memory Usage Profiling
// ============================================================================

/// Read the memory summary percentile from `BENCH_PERCENTILE` (default: 95).
#[cfg(all(feature = "count-allocations", feature = "bench-logging"))]
fn configured_percentile() -> usize {
    env::var("BENCH_PERCENTILE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map_or(95, |p| p.clamp(1, 100))
}

/// Format a percentile as an ordinal label for the memory summary.
#[cfg(all(feature = "count-allocations", feature = "bench-logging"))]
fn percentile_label(percentile: usize) -> String {
    let suffix = match percentile % 100 {
        11..=13 => "th",
        _ => match percentile % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        },
    };
    format!("{percentile}{suffix}")
}

/// Return percentile from a slice of values using nearest-rank method.
#[cfg(all(feature = "count-allocations", feature = "bench-logging"))]
fn percentile_nearest_rank(values: &mut [u64], percentile: usize) -> u64 {
    if values.is_empty() {
        return 0;
    }

    values.sort_unstable();
    let n = values.len();
    // nearest-rank: ceil(p/100 * n), clamped to [1, n]
    let rank = ((percentile.saturating_mul(n)).saturating_add(99))
        .saturating_div(100)
        .clamp(1, n);
    let index = rank - 1; // safe: rank in [1, n]
    values[index]
}

/// Print memory allocation summary
#[cfg(all(feature = "count-allocations", feature = "bench-logging"))]
#[expect(
    clippy::cast_precision_loss,
    reason = "allocation summary reports byte ratios where f64 precision is sufficient"
)]
fn print_alloc_summary(
    info: &AllocationInfo,
    description: &str,
    actual_point_count: usize,
    percentile: usize,
    percentile_value: u64,
) {
    println!("\n=== Memory Allocation Summary for {description} ({actual_point_count} points) ===");
    println!("Total allocations: {}", info.count_total);
    println!("Current allocations: {}", info.count_current);
    println!("Max allocations: {}", info.count_max);
    println!("Total bytes allocated: {}", info.bytes_total);
    println!("Current bytes allocated: {}", info.bytes_current);
    println!(
        "Max bytes allocated: {} ({:.2} MB)",
        info.bytes_max,
        info.bytes_max as f64 / (1024.0 * 1024.0)
    );
    println!(
        "{} percentile bytes: {} ({:.2} MB)",
        percentile_label(percentile),
        percentile_value,
        percentile_value as f64 / (1024.0 * 1024.0)
    );
    if actual_point_count > 0 {
        println!(
            "Bytes per point (peak): {:.1}",
            info.bytes_max as f64 / actual_point_count as f64
        );
    } else {
        println!("Bytes per point (peak): n/a (0 points)");
    }
    println!("=====================================\n");
}

#[cfg(all(feature = "count-allocations", feature = "bench-logging"))]
#[expect(
    clippy::cast_possible_wrap,
    reason = "sample counts fit signed indexing for percentile diagnostics"
)]
fn print_alloc_summary_from_samples<const D: usize>(
    allocation_infos: &SmallBuffer<AllocationInfo, BENCHMARK_ITERATION_BUFFER_SIZE>,
    actual_point_counts: &SmallBuffer<usize, BENCHMARK_ITERATION_BUFFER_SIZE>,
) {
    if allocation_infos.is_empty() {
        return;
    }

    // Safe cast for division: Criterion sample buffers here are small and non-empty.
    let divisor_unsigned = allocation_infos.len() as u64;
    let divisor_signed = allocation_infos.len() as i64;
    let avg_info = AllocationInfo {
        count_total: allocation_infos.iter().map(|i| i.count_total).sum::<u64>() / divisor_unsigned,
        count_current: allocation_infos
            .iter()
            .map(|i| i.count_current)
            .sum::<i64>()
            / divisor_signed,
        count_max: allocation_infos
            .iter()
            .map(|i| i.count_max)
            .max()
            .unwrap_or(0),
        bytes_total: allocation_infos.iter().map(|i| i.bytes_total).sum::<u64>() / divisor_unsigned,
        bytes_current: allocation_infos
            .iter()
            .map(|i| i.bytes_current)
            .sum::<i64>()
            / divisor_signed,
        bytes_max: allocation_infos
            .iter()
            .map(|i| i.bytes_max)
            .max()
            .unwrap_or(0),
    };
    let avg_actual_count = if actual_point_counts.is_empty() {
        0
    } else {
        actual_point_counts.iter().sum::<usize>() / actual_point_counts.len()
    };

    let mut bytes_max_values: Vec<u64> = allocation_infos.iter().map(|i| i.bytes_max).collect();
    let percentile = configured_percentile();
    let percentile_value = percentile_nearest_rank(&mut bytes_max_values, percentile);

    print_alloc_summary(
        &avg_info,
        &format!("{D}D Triangulation"),
        avg_actual_count,
        percentile,
        percentile_value,
    );
}

/// Generic helper to benchmark memory usage for a specific dimension D
fn bench_memory_usage<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    bench_id_prefix: &str,
    count: usize,
) {
    #[cfg(all(feature = "count-allocations", feature = "bench-logging"))]
    let mut allocation_infos: SmallBuffer<AllocationInfo, BENCHMARK_ITERATION_BUFFER_SIZE> =
        SmallBuffer::new();

    #[cfg(all(feature = "count-allocations", feature = "bench-logging"))]
    let mut actual_point_counts: SmallBuffer<usize, BENCHMARK_ITERATION_BUFFER_SIZE> =
        SmallBuffer::new();

    group.bench_with_input(
        BenchmarkId::new(bench_id_prefix, count),
        &count,
        |b, &count| {
            b.iter_custom(|iters| {
                let mut total_time = Duration::new(0, 0);

                for _ in 0..iters {
                    let points = gen_points::<D>(count, PointDistribution::Random, DEFAULT_SEED);
                    #[cfg(all(feature = "count-allocations", feature = "bench-logging"))]
                    let pts_len = points.len();
                    let vertices = benchmark_vertices_from_generated_points(&points);
                    let start_time = Instant::now();

                    let alloc_info = measure(|| {
                        let dt = DelaunayTriangulationBuilder::new(&vertices)
                            .build::<()>()
                            .or_abort();
                        black_box(dt);
                    });

                    total_time += start_time.elapsed();

                    #[cfg(all(feature = "count-allocations", feature = "bench-logging"))]
                    {
                        allocation_infos.push(alloc_info);
                        actual_point_counts.push(pts_len);
                    }

                    #[cfg(not(all(feature = "count-allocations", feature = "bench-logging")))]
                    let _ = alloc_info;
                }

                total_time
            });
        },
    );

    #[cfg(all(feature = "count-allocations", feature = "bench-logging"))]
    print_alloc_summary_from_samples::<D>(&allocation_infos, &actual_point_counts);
}

/// Memory usage profiling across different scales and dimensions using allocation counter
fn benchmark_memory_profiling(c: &mut Criterion) {
    #[cfg(not(feature = "count-allocations"))]
    print_alloc_banner_once();

    let counts = if is_dev_mode() {
        &[1_000, 10_000][..]
    } else {
        &[1_000, 10_000, 100_000][..]
    };

    let mut group = c.benchmark_group("memory_profiling");
    group.measurement_time(bench_time(30));

    for &count in counts {
        // 2D Memory Profiling
        bench_memory_usage::<2>(&mut group, "memory_usage_2d", count);

        // 3D Memory Profiling (smaller counts due to complexity)
        if count <= 10_000 {
            bench_memory_usage::<3>(&mut group, "memory_usage_3d", count);
        }

        // 4D Memory Profiling (even smaller counts due to exponential complexity)
        if count <= 3_000 {
            bench_memory_usage::<4>(&mut group, "memory_usage_4d", count);
        }

        // 5D Memory Profiling (very small counts due to very high complexity)
        if count <= 1_000 {
            bench_memory_usage::<5>(&mut group, "memory_usage_5d", count);
        }
    }

    group.finish();
}

// ============================================================================
// Query Latency Analysis
// ============================================================================

/// Query latency profiling for circumsphere containment tests
fn benchmark_query_latency(c: &mut Criterion) {
    const MAX_PRECOMPUTED_SIMPLICES: usize = 1000;
    let counts = if is_dev_mode() {
        &[1_000, 3_000][..]
    } else {
        &[1_000, 10_000, 30_000][..]
    };

    let mut group = c.benchmark_group("query_latency");
    group.measurement_time(bench_time(90));
    group.throughput(Throughput::Elements(MAX_QUERY_RESULTS as u64));

    for &count in counts {
        // Create triangulation and test circumsphere queries
        group.bench_with_input(
            BenchmarkId::new("circumsphere_queries_3d", count),
            &count,
            |b, &count| {
                // Setup: Create triangulation and query points
                let points = gen_points::<3>(count, PointDistribution::Random, DEFAULT_SEED);
                let vertices = benchmark_vertices_from_generated_points(&points);
                let Ok(dt) = DelaunayTriangulationBuilder::new(&vertices).build::<()>() else {
                    // Construction hit a geometric degeneracy; skip this benchmark entry
                    b.iter(|| {});
                    return;
                };
                let tds = dt.tds();

                // Generate query points
                let query_points = gen_points::<3>(100, PointDistribution::Random, QUERY_SEED);

                // Precompute all valid simplex vertices outside the benchmark loop
                let mut precomputed_simplices: Vec<
                    SmallBuffer<Point<3>, SIMPLEX_VERTICES_BUFFER_SIZE>,
                > = Vec::with_capacity(MAX_PRECOMPUTED_SIMPLICES);
                let mut sampled_count = 0;
                for simplex in tds.simplices() {
                    if sampled_count >= MAX_PRECOMPUTED_SIMPLICES {
                        break;
                    }

                    // Get vertex points for this simplex by looking up each vertex key
                    let vertex_keys = simplex.1.vertices();
                    if vertex_keys.len() == 4 {
                        // Valid 3D simplex - collect points
                        let mut vertex_points: SmallBuffer<Point<3>, SIMPLEX_VERTICES_BUFFER_SIZE> =
                            SmallBuffer::new();
                        for vkey in vertex_keys {
                            if let Some(vertex) = tds.vertex(*vkey) {
                                vertex_points.push(*vertex.point());
                            }
                        }
                        if vertex_points.len() == 4 {
                            precomputed_simplices.push(vertex_points);
                            sampled_count += 1;
                        }
                    }
                }

                b.iter(|| {
                    // Perform circumsphere containment queries using precomputed data
                    let mut query_results: SmallBuffer<_, QUERY_RESULTS_BUFFER_SIZE> =
                        SmallBuffer::new();

                    for points_for_test in &precomputed_simplices {
                        for query_point in &query_points {
                            let query_point_obj = *query_point;

                            // Use the fastest circumsphere method (based on benchmark results)
                            let result = insphere_lifted(points_for_test, query_point_obj);
                            query_results.push(result);

                            // Limit total queries to prevent extremely long benchmarks
                            if query_results.len() >= MAX_QUERY_RESULTS {
                                break;
                            }
                        }

                        if query_results.len() >= MAX_QUERY_RESULTS {
                            break;
                        }
                    }

                    black_box(query_results);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Validation Layer Diagnostics
// ============================================================================

macro_rules! benchmark_validation_components_dimension {
    ($dim:literal, $func_name:ident, $count:expr) => {
        fn $func_name(c: &mut Criterion) {
            let is_adversarial = stringify!($func_name).ends_with("_adversarial");
            let distribution = if is_adversarial {
                PointDistribution::Adversarial
            } else {
                PointDistribution::Random
            };
            let suffix = if is_adversarial { "_adversarial" } else { "" };
            let mut last_error = None;
            let dt = (0..VALIDATION_SEED_SEARCH_LIMIT)
                .find_map(|offset| {
                    let seed = DEFAULT_SEED.wrapping_add(offset);
                    let points = gen_points::<$dim>($count, distribution, seed);
                    let vertices = benchmark_vertices_from_generated_points(&points);
                    let builder = DelaunayTriangulationBuilder::new(&vertices);
                    let builder = if is_adversarial {
                        let attempts = retry_attempts(8);
                        builder.construction_options(
                            ConstructionOptions::default().with_retry_policy(
                                RetryPolicy::Shuffled {
                                    attempts,
                                    base_seed: Some(seed),
                                },
                            ),
                        )
                    } else {
                        builder
                    };

                    match builder.build::<()>() {
                        Ok(dt) => Some(dt),
                        Err(err) => {
                            last_error = Some(format!("{err}"));
                            None
                        }
                    }
                })
                .unwrap_or_else(|| {
                    abort_benchmark(format_args!(
                        "failed to build {}D validation component benchmark triangulation \
                         after {} seeds (last error: {})",
                        $dim,
                        VALIDATION_SEED_SEARCH_LIMIT,
                        last_error.unwrap_or_else(|| "none".to_string())
                    ));
                });

            let mut group = c.benchmark_group(format!("validation_components_{}d{}", $dim, suffix));
            group.measurement_time(bench_time(15));
            group.throughput(Throughput::Elements($count as u64));

            group.bench_function("tds_is_valid", |b| {
                b.iter(|| {
                    if let Err(error) = black_box(dt.tds().is_valid()) {
                        abort_benchmark(format_args!(
                            "TDS validation should pass for benchmark triangulation: {error}"
                        ));
                    }
                });
            });

            group.bench_function("tri_is_valid", |b| {
                b.iter(|| {
                    if let Err(error) = black_box(dt.as_triangulation().is_valid()) {
                        abort_benchmark(format_args!(
                            "triangulation validation should pass for benchmark triangulation: {error}"
                        ));
                    }
                });
            });

            group.bench_function("is_valid_delaunay", |b| {
                b.iter(|| {
                    if let Err(error) = black_box(dt.is_valid()) {
                        abort_benchmark(format_args!(
                            "Delaunay validation should pass for benchmark triangulation: {error}"
                        ));
                    }
                });
            });

            group.bench_function("validate", |b| {
                b.iter(|| {
                    if let Err(error) = black_box(dt.validate()) {
                        abort_benchmark(format_args!(
                            "full validation should pass for benchmark triangulation: {error}"
                        ));
                    }
                });
            });

            group.finish();
        }
    };
}

benchmark_validation_components_dimension!(2, benchmark_validation_components_2d, 50);
benchmark_validation_components_dimension!(3, benchmark_validation_components_3d, 50);
benchmark_validation_components_dimension!(4, benchmark_validation_components_4d, 25);
benchmark_validation_components_dimension!(5, benchmark_validation_components_5d, 25);
benchmark_validation_components_dimension!(2, benchmark_validation_components_2d_adversarial, 50);
benchmark_validation_components_dimension!(3, benchmark_validation_components_3d_adversarial, 50);
benchmark_validation_components_dimension!(4, benchmark_validation_components_4d_adversarial, 25);
benchmark_validation_components_dimension!(5, benchmark_validation_components_5d_adversarial, 25);

// ============================================================================
// Algorithmic Bottleneck Identification
// ============================================================================

/// Profile specific algorithmic components to identify bottlenecks
fn bench_bottlenecks(c: &mut Criterion) {
    let counts = if is_dev_mode() {
        &[3_000][..]
    } else {
        &[10_000][..]
    };

    let mut group = c.benchmark_group("algorithmic_bottlenecks");
    group.measurement_time(bench_time(15));

    for &count in counts {
        // Profile boundary facet computation
        group.bench_with_input(
            BenchmarkId::new("boundary_facets_3d", count),
            &count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let points =
                            gen_points::<3>(count, PointDistribution::Random, DEFAULT_SEED);
                        let vertices = benchmark_vertices_from_generated_points(&points);
                        DelaunayTriangulationBuilder::new(&vertices)
                            .build::<()>()
                            .ok()
                    },
                    |dt| {
                        if let Some(dt) = dt {
                            let boundary_facets = match dt.tds().one_sided_facets() {
                                Ok(value) => value,
                                Err(error) => {
                                    abort_benchmark(format_args!(
                                        "boundary_facets failed: {error}"
                                    ));
                                }
                            };
                            black_box(boundary_facets.len());
                        }
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // Profile convex hull extraction
        group.bench_with_input(
            BenchmarkId::new("convex_hull_3d", count),
            &count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let points =
                            gen_points::<3>(count, PointDistribution::Random, DEFAULT_SEED);
                        let vertices = benchmark_vertices_from_generated_points(&points);
                        DelaunayTriangulationBuilder::new(&vertices)
                            .build::<()>()
                            .ok()
                    },
                    |dt| {
                        if let Some(dt) = dt {
                            let hull =
                                match ConvexHull::try_from_triangulation(dt.as_triangulation()) {
                                    Ok(value) => value,
                                    Err(error) => abort_benchmark(format_args!(
                                        "convex hull extraction failed: {error}"
                                    )),
                                };
                            let _ = black_box(hull);
                        }
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Main Criterion Configuration
// ============================================================================

criterion_group!(
    name = profiling_benches;
    config = {
        init_tracing();
        // Allow configuration via environment variables for CI stability
        let sample_size = env::var("BENCH_SAMPLE_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .map_or(10, |size: usize| size.max(10));
        let warm_up_secs = env::var("BENCH_WARMUP_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);

        Criterion::default()
            .sample_size(sample_size)  // Configurable samples (default: 10 for long-running benchmarks)
            .warm_up_time(Duration::from_secs(warm_up_secs))  // Configurable warm-up (default: 10s)
            .measurement_time(bench_time(60))
    };
    targets =
        bench_2d_suite,
        bench_3d_suite,
        bench_4d_suite,
        bench_5d_suite,
        benchmark_memory_profiling,
        benchmark_query_latency,
        benchmark_validation_components_2d,
        benchmark_validation_components_3d,
        benchmark_validation_components_4d,
        benchmark_validation_components_5d,
        benchmark_validation_components_2d_adversarial,
        benchmark_validation_components_3d_adversarial,
        benchmark_validation_components_4d_adversarial,
        benchmark_validation_components_5d_adversarial,
        bench_bottlenecks
);

criterion_main!(profiling_benches);
