//! Comprehensive Profiling Suite - Extended performance analysis for optimization work
//!
//! This benchmark suite provides extensive profiling capabilities for deep performance
//! analysis and optimization work that would be too time-consuming for regular CI/CD:
//!
//! 1. **Large-scale triangulation performance** (10³ to 10⁶ points)
//! 2. **Multiple point distributions** (random, grid, Poisson disk)
//! 3. **Memory usage profiling** (allocation tracking)
//! 4. **Query latency analysis** (circumsphere tests, neighbor queries)
//! 5. **Multi-dimensional scaling** (2D through 5D)
//! 6. **Algorithmic bottleneck identification** (specific operation profiling)
//! 7. **Validation layer diagnostics** (Level 1-3 vs Level 4 cost separation)
//!
//! ## Usage
//!
//! Run comprehensive profiling (expect 1-2 hours):
//! ```bash
//! cargo bench --profile perf --bench profiling_suite
//! ```
//!
//! Run specific profiling categories:
//! ```bash
//! # Large-scale triangulation only
//! cargo bench --profile perf --bench profiling_suite -- triangulation_scaling
//!
//! # Memory profiling only  
//! cargo bench --profile perf --bench profiling_suite -- memory_profiling
//!
//! # Query latency only
//! cargo bench --profile perf --bench profiling_suite -- query_latency
//! ```
//!
//! ## Development vs Production Mode
//!
//! For faster iteration during optimization work:
//! ```bash
//! # Development mode - reduced scale for quick feedback
//! PROFILING_DEV_MODE=1 cargo bench --profile perf --bench profiling_suite
//! ```
//!
//! ## Environment Variable Configuration
//!
//! The benchmark suite supports several environment variables for customization:
//!
//! - `PROFILING_DEV_MODE`: Set to "1", "true", "yes", or "on" for reduced scale (faster iteration)
//! - `BENCH_MEASUREMENT_TIME`: Override measurement time in seconds (minimum: 1, guards against invalid values)
//! - `BENCH_PERCENTILE`: Configure percentile for memory analysis (1-100, default: 95)
//! - `BENCH_SAMPLE_SIZE`: Override Criterion sample size (default: 10; values below 10 are clamped to 10, so
//!   `BENCH_SAMPLE_SIZE=5` still runs 10 samples)
//! - `BENCH_WARMUP_SECS`: Override Criterion warm-up time in seconds (default: 10)
//!
//! Example with custom configuration:
//! ```bash
//! BENCH_SAMPLE_SIZE=10 BENCH_WARMUP_SECS=5 BENCH_PERCENTILE=90 cargo bench --profile perf --bench profiling_suite
//! ```

use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use delaunay::core::collections::SmallBuffer;
use delaunay::geometry::traits::coordinate::Coordinate;
use delaunay::geometry::util::{
    generate_grid_points, generate_poisson_points, generate_random_points_seeded,
    safe_usize_to_scalar,
};
use delaunay::prelude::query::*;
use delaunay::prelude::triangulation::{
    ConstructionOptions, DelaunayTriangulationBuilder, RetryPolicy,
};
use delaunay::vertex;
use num_traits::cast;
use std::env;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Once;
use std::time::{Duration, Instant};

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

// Memory allocation counting support
#[cfg(feature = "count-allocations")]
use allocation_counter::{AllocationInfo, measure};

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

/// Large-scale point counts for comprehensive profiling
/// Production mode: 10³ to 10⁶ points across multiple decades
/// Development mode: Reduced scale for faster iteration
const PROFILING_COUNTS_PRODUCTION: &[usize] = &[
    1_000,     // 10³
    3_000,     // ~3×10³
    10_000,    // 10⁴
    30_000,    // ~3×10⁴
    100_000,   // 10⁵
    300_000,   // ~3×10⁵
    1_000_000, // 10⁶
];

const PROFILING_COUNTS_DEVELOPMENT: &[usize] = &[
    1_000,  // 10³
    3_000,  // ~3×10³
    10_000, // 10⁴
];

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

/// Get appropriate point counts based on environment
fn get_profiling_counts() -> &'static [usize] {
    if is_dev_mode() {
        PROFILING_COUNTS_DEVELOPMENT
    } else {
        PROFILING_COUNTS_PRODUCTION
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
    Grid,
    PoissonDisk,
    Adversarial,
}

impl PointDistribution {
    const fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Grid => "grid",
            Self::PoissonDisk => "poisson",
            Self::Adversarial => "adversarial",
        }
    }
}

/// Generate points according to the specified distribution
fn gen_points<const D: usize>(
    count: usize,
    distribution: PointDistribution,
    seed: u64,
) -> Vec<Point<f64, D>> {
    match distribution {
        PointDistribution::Random => generate_random_points_seeded(count, (-100.0, 100.0), seed)
            .expect("random point generation failed"),
        PointDistribution::Adversarial => generate_random_points_seeded::<f64, D>(
            count,
            (-1.0, 1.0),
            seed ^ 0xA5A5_A5A5_A5A5_A5A5,
        )
        .expect("adversarial base point generation failed")
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let index = u32::try_from(index).expect("benchmark point index should fit in u32");
            let mut coords = [0.0_f64; D];
            for (axis, coord) in coords.iter_mut().enumerate() {
                let axis_number = u32::try_from(axis + 1).expect("axis should fit in u32");
                let base: f64 = point.coords()[axis];
                let cluster_offset = f64::from(index % 7) * 1.0e-3;
                let axis_offset = f64::from(axis_number) * 0.25;
                let perturbation = f64::from((index + axis_number) % 11) * 1.0e-6;
                *coord = base.mul_add(1.0e3, 1.0e9 + axis_offset + cluster_offset + perturbation);
            }
            Point::new(coords)
        })
        .collect(),
        PointDistribution::Grid => {
            // Calculate points per dimension to get approximately `count` points total
            let count_f64 = safe_usize_to_scalar::<f64>(count).unwrap_or(2.0);
            let d_f64 = safe_usize_to_scalar::<f64>(D).unwrap_or(1.0);
            let raw = count_f64.powf(1.0 / d_f64).ceil();

            let points_per_dim = if raw.is_finite() && raw >= 2.0 {
                // Saturate instead of shrinking to 2 on cast failure
                cast::<f64, usize>(raw).unwrap_or(usize::MAX).max(2)
            } else {
                2
            };
            match generate_grid_points(points_per_dim, 10.0, [0.0; D]) {
                Ok(pts) => pts,
                Err(e) => {
                    // Grid generation failed - this indicates a configuration issue
                    // Rather than silently falling back and producing misleading benchmarks,
                    // we should fail fast to alert developers to adjust parameters
                    panic!(
                        "Grid generation failed for D={D}: count={count}, points_per_dim={points_per_dim}, err={e:?}. \
                         Adjust grid parameters or use smaller point counts for high-dimensional grid benchmarks."
                    );
                }
            }
        }
        PointDistribution::PoissonDisk => {
            let min_distance = match D {
                2 => 5.0,  // 2D: reasonable spacing for [-100, 100] bounds
                3 => 8.0,  // 3D: slightly larger spacing
                4 => 12.0, // 4D: larger spacing for higher dimensions
                5 => 15.0, // 5D: even larger spacing
                _ => 20.0, // Higher dimensions: very large spacing
            };
            generate_poisson_points(count, (-100.0, 100.0), min_distance, seed)
                .expect("poisson point generation failed")
        }
    }
}

// ============================================================================
// Large-Scale Triangulation Performance Profiling
// ============================================================================

/// Comprehensive triangulation scaling analysis across dimensions and distributions
#[expect(clippy::significant_drop_tightening, clippy::too_many_lines)]
fn bench_scaling(c: &mut Criterion) {
    let counts = get_profiling_counts();
    let distributions = [
        PointDistribution::Random,
        PointDistribution::Grid,
        PointDistribution::PoissonDisk,
    ];

    // 2D Triangulation Scaling
    let mut group = c.benchmark_group("triangulation_scaling_2d");
    group.measurement_time(bench_time(120));

    for &count in counts {
        for &distribution in &distributions {
            // Pre-generate sample points to calculate actual count and avoid double-generation
            let sample_points = gen_points::<2>(count, distribution, DEFAULT_SEED);
            let actual_count = sample_points.len();
            group.throughput(Throughput::Elements(actual_count as u64));

            let bench_id = format!("{}_2d_{}", distribution.name(), count);
            group.bench_with_input(
                BenchmarkId::new("tds_new", bench_id),
                &(count, distribution, actual_count),
                |b, &(count, distribution, _actual_count)| {
                    b.iter_batched(
                        || {
                            // Reuse same generation logic to ensure consistent point count
                            let points = gen_points::<2>(count, distribution, DEFAULT_SEED);
                            points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                        },
                        |vertices| {
                            if let Ok(dt) =
                                DelaunayTriangulationBuilder::new(&vertices).build::<()>()
                            {
                                black_box(dt);
                            }
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
    group.finish();

    // 3D Triangulation Scaling
    let mut group = c.benchmark_group("triangulation_scaling_3d");
    group.measurement_time(bench_time(180));

    // 3D: cap to 100_000 in production to avoid runaway memory/time
    for count in counts
        .iter()
        .copied()
        .filter(|c| is_dev_mode() || *c <= 100_000)
    {
        for &distribution in &distributions {
            // Skip very large counts for 3D in development mode to prevent timeouts
            if is_dev_mode() && count > 10_000 {
                continue;
            }

            // Pre-generate sample points to calculate actual count and avoid double-generation
            let sample_points = gen_points::<3>(count, distribution, DEFAULT_SEED);
            let actual_count = sample_points.len();
            group.throughput(Throughput::Elements(actual_count as u64));

            let bench_id = format!("{}_3d_{}", distribution.name(), count);
            group.bench_with_input(
                BenchmarkId::new("tds_new", bench_id),
                &(count, distribution, actual_count),
                |b, &(count, distribution, _actual_count)| {
                    b.iter_batched(
                        || {
                            let points = gen_points::<3>(count, distribution, DEFAULT_SEED);
                            points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                        },
                        |vertices| {
                            if let Ok(dt) =
                                DelaunayTriangulationBuilder::new(&vertices).build::<()>()
                            {
                                black_box(dt);
                            }
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
    group.finish();

    // 4D and 5D Triangulation Scaling (smaller counts due to exponential complexity)
    let high_dim_counts = if is_dev_mode() {
        &[1_000, 3_000][..]
    } else {
        &[1_000, 3_000, 10_000][..]
    };

    // 4D Triangulation Scaling
    let mut group = c.benchmark_group("triangulation_scaling_4d");
    group.measurement_time(bench_time(240));

    // 4D: cap to 3_000 in production to avoid runaway memory/time
    for count in high_dim_counts
        .iter()
        .copied()
        .filter(|c| is_dev_mode() || *c <= 3_000)
    {
        for &distribution in &distributions {
            // Pre-generate sample points to calculate actual count and avoid double-generation
            let sample_points = gen_points::<4>(count, distribution, DEFAULT_SEED);
            let actual_count = sample_points.len();
            group.throughput(Throughput::Elements(actual_count as u64));

            let bench_id = format!("{}_4d_{}", distribution.name(), count);
            group.bench_with_input(
                BenchmarkId::new("tds_new", bench_id),
                &(count, distribution, actual_count),
                |b, &(count, distribution, _actual_count)| {
                    b.iter_batched(
                        || {
                            let points = gen_points::<4>(count, distribution, DEFAULT_SEED);
                            points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                        },
                        |vertices| {
                            if let Ok(dt) =
                                DelaunayTriangulationBuilder::new(&vertices).build::<()>()
                            {
                                black_box(dt);
                            }
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
    group.finish();

    // 5D Triangulation Scaling (even smaller counts due to very high complexity)
    let ultra_high_dim_counts = if is_dev_mode() {
        &[1_000][..]
    } else {
        &[1_000, 3_000][..]
    };

    let mut group = c.benchmark_group("triangulation_scaling_5d");
    group.measurement_time(bench_time(300));

    // 5D: cap to 1_000 in production to avoid runaway memory/time
    for count in ultra_high_dim_counts
        .iter()
        .copied()
        .filter(|c| is_dev_mode() || *c <= 1_000)
    {
        for &distribution in &distributions {
            // Pre-generate sample points to calculate actual count and avoid double-generation
            let sample_points = gen_points::<5>(count, distribution, DEFAULT_SEED);
            let actual_count = sample_points.len();
            group.throughput(Throughput::Elements(actual_count as u64));

            let bench_id = format!("{}_5d_{}", distribution.name(), count);
            group.bench_with_input(
                BenchmarkId::new("tds_new", bench_id),
                &(count, distribution, actual_count),
                |b, &(count, distribution, _actual_count)| {
                    b.iter_batched(
                        || {
                            let points = gen_points::<5>(count, distribution, DEFAULT_SEED);
                            points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                        },
                        |vertices| {
                            if let Ok(dt) =
                                DelaunayTriangulationBuilder::new(&vertices).build::<()>()
                            {
                                black_box(dt);
                            }
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
    group.finish();
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

/// Calculate percentile from a slice of values using nearest-rank method.
#[cfg(all(feature = "count-allocations", feature = "bench-logging"))]
fn calculate_percentile(values: &mut [u64], percentile: usize) -> u64 {
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
#[expect(clippy::cast_precision_loss)]
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
#[expect(clippy::cast_possible_wrap)]
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
    let percentile_value = calculate_percentile(&mut bytes_max_values, percentile);

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
                    let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                    let start_time = Instant::now();

                    let alloc_info = measure(|| {
                        if let Ok(dt) = DelaunayTriangulationBuilder::new(&vertices).build::<()>() {
                            black_box(dt);
                        }
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
                let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
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
                    SmallBuffer<Point<f64, 3>, SIMPLEX_VERTICES_BUFFER_SIZE>,
                > = Vec::with_capacity(MAX_PRECOMPUTED_SIMPLICES);
                let mut sampled_count = 0;
                for cell in tds.cells() {
                    if sampled_count >= MAX_PRECOMPUTED_SIMPLICES {
                        break;
                    }

                    // Get vertex points for this cell by looking up each vertex key
                    let vertex_keys = cell.1.vertices();
                    if vertex_keys.len() == 4 {
                        // Valid 3D simplex - collect points
                        let mut vertex_points: SmallBuffer<
                            Point<f64, 3>,
                            SIMPLEX_VERTICES_BUFFER_SIZE,
                        > = SmallBuffer::new();
                        for vkey in vertex_keys {
                            if let Some(vertex) = tds.get_vertex_by_key(*vkey) {
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
                    let vertices: Vec<_> = points.iter().map(|point| vertex!(*point)).collect();
                    let builder = DelaunayTriangulationBuilder::new(&vertices);
                    let builder = if is_adversarial {
                        let attempts =
                            NonZeroUsize::new(8).expect("retry attempts must be non-zero");
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
                    panic!(
                        "failed to build {}D validation component benchmark triangulation \
                         after {} seeds (last error: {})",
                        $dim,
                        VALIDATION_SEED_SEARCH_LIMIT,
                        last_error.unwrap_or_else(|| "none".to_string())
                    );
                });

            let mut group = c.benchmark_group(format!("validation_components_{}d{}", $dim, suffix));
            group.measurement_time(bench_time(15));
            group.throughput(Throughput::Elements($count as u64));

            group.bench_function("tds_is_valid", |b| {
                b.iter(|| {
                    black_box(dt.tds().is_valid())
                        .expect("TDS validation should pass for benchmark triangulation");
                });
            });

            group.bench_function("tri_is_valid", |b| {
                b.iter(|| {
                    black_box(dt.as_triangulation().is_valid())
                        .expect("triangulation validation should pass for benchmark triangulation");
                });
            });

            group.bench_function("is_valid_delaunay", |b| {
                b.iter(|| {
                    black_box(dt.is_valid())
                        .expect("Delaunay validation should pass for benchmark triangulation");
                });
            });

            group.bench_function("validate", |b| {
                b.iter(|| {
                    black_box(dt.validate())
                        .expect("full validation should pass for benchmark triangulation");
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
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        DelaunayTriangulationBuilder::new(&vertices)
                            .build::<()>()
                            .ok()
                    },
                    |dt| {
                        if let Some(dt) = dt {
                            let boundary_facets =
                                dt.tds().boundary_facets().expect("boundary_facets failed");
                            black_box(boundary_facets);
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
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        DelaunayTriangulationBuilder::new(&vertices)
                            .build::<()>()
                            .ok()
                    },
                    |dt| {
                        if let Some(dt) = dt {
                            let hull =
                                ConvexHull::from_triangulation(dt.as_triangulation()).unwrap();
                            black_box(hull);
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
        bench_scaling,
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
