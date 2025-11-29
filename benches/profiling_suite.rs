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
//!
//! ## Usage
//!
//! Run comprehensive profiling (expect 1-2 hours):
//! ```bash
//! cargo bench --bench profiling_suite
//! ```
//!
//! Run specific profiling categories:
//! ```bash
//! # Large-scale triangulation only
//! cargo bench --bench profiling_suite -- triangulation_scaling
//!
//! # Memory profiling only  
//! cargo bench --bench profiling_suite -- memory_profiling
//!
//! # Query latency only
//! cargo bench --bench profiling_suite -- query_latency
//! ```
//!
//! ## Development vs Production Mode
//!
//! For faster iteration during optimization work:
//! ```bash
//! # Development mode - reduced scale for quick feedback
//! PROFILING_DEV_MODE=1 cargo bench --bench profiling_suite
//! ```
//!
//! ## Environment Variable Configuration
//!
//! The benchmark suite supports several environment variables for customization:
//!
//! - `PROFILING_DEV_MODE`: Set to "1", "true", "yes", or "on" for reduced scale (faster iteration)
//! - `BENCH_MEASUREMENT_TIME`: Override measurement time in seconds (minimum: 1, guards against invalid values)
//! - `BENCH_PERCENTILE`: Configure percentile for memory analysis (1-100, default: 95)
//! - `BENCH_SAMPLE_SIZE`: Override Criterion sample size (default: 10)
//! - `BENCH_WARMUP_SECS`: Override Criterion warm-up time in seconds (default: 10)
//!
//! Example with custom configuration:
//! ```bash
//! BENCH_SAMPLE_SIZE=5 BENCH_WARMUP_SECS=5 BENCH_PERCENTILE=90 cargo bench --bench profiling_suite
//! ```

#![allow(missing_docs)]
#![expect(deprecated)] // Benchmark uses deprecated Tds::new() until migration to DelaunayTriangulation

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::core::collections::SmallBuffer;
use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
use delaunay::geometry::util::{
    generate_grid_points, generate_poisson_points, generate_random_points_seeded,
    safe_usize_to_scalar,
};
use delaunay::prelude::*;
use delaunay::vertex;
use num_traits::cast;
use serde::{Serialize, de::DeserializeOwned};
use std::hint::black_box;
use std::time::{Duration, Instant};

// SmallBuffer size constants for different use cases
const BENCHMARK_ITERATION_BUFFER_SIZE: usize = 8; // For tracking allocation info across benchmark iterations
const SIMPLEX_VERTICES_BUFFER_SIZE: usize = 4; // 3D simplex = 4 vertices
const QUERY_RESULTS_BUFFER_SIZE: usize = 1024; // For bounded query result collections (max 1000 in code)

// Reusable seeds and caps
const DEFAULT_SEED: u64 = 42;
const QUERY_SEED: u64 = 123;
const MAX_QUERY_RESULTS: usize = 1_000;

// Memory allocation counting support
#[cfg(feature = "count-allocations")]
use allocation_counter::{AllocationInfo, measure};

#[cfg(not(feature = "count-allocations"))]
#[derive(Debug, Default)]
struct AllocationInfo {
    count_total: u64,
    count_current: i64,
    count_max: u64,
    bytes_total: u64,
    bytes_current: i64,
    bytes_max: u64,
}

#[cfg(not(feature = "count-allocations"))]
fn measure<F: FnOnce()>(f: F) -> AllocationInfo {
    f();
    AllocationInfo::default()
}

#[cfg(not(feature = "count-allocations"))]
fn print_count_allocations_banner_once() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        eprintln!("count-allocations feature not enabled; memory stats are placeholders.");
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
    let dev = std::env::var("PROFILING_DEV_MODE").ok();
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
    let secs = std::env::var("BENCH_MEASUREMENT_TIME")
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
}

impl PointDistribution {
    const fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Grid => "grid",
            Self::PoissonDisk => "poisson",
        }
    }
}

/// Generate points according to the specified distribution
fn generate_points_by_distribution<const D: usize>(
    count: usize,
    distribution: PointDistribution,
    seed: u64,
) -> Vec<Point<f64, D>> {
    match distribution {
        PointDistribution::Random => generate_random_points_seeded(count, (-100.0, 100.0), seed)
            .expect("random point generation failed"),
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
fn benchmark_triangulation_scaling(c: &mut Criterion) {
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
            let sample_points =
                generate_points_by_distribution::<2>(count, distribution, DEFAULT_SEED);
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
                            let points = generate_points_by_distribution::<2>(
                                count,
                                distribution,
                                DEFAULT_SEED,
                            );
                            points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                        },
                        |vertices| {
                            black_box(Tds::<f64, (), (), 2>::new(&vertices).unwrap());
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
            let sample_points =
                generate_points_by_distribution::<3>(count, distribution, DEFAULT_SEED);
            let actual_count = sample_points.len();
            group.throughput(Throughput::Elements(actual_count as u64));

            let bench_id = format!("{}_3d_{}", distribution.name(), count);
            group.bench_with_input(
                BenchmarkId::new("tds_new", bench_id),
                &(count, distribution, actual_count),
                |b, &(count, distribution, _actual_count)| {
                    b.iter_batched(
                        || {
                            let points = generate_points_by_distribution::<3>(
                                count,
                                distribution,
                                DEFAULT_SEED,
                            );
                            points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                        },
                        |vertices| {
                            black_box(Tds::<f64, (), (), 3>::new(&vertices).unwrap());
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
            let sample_points =
                generate_points_by_distribution::<4>(count, distribution, DEFAULT_SEED);
            let actual_count = sample_points.len();
            group.throughput(Throughput::Elements(actual_count as u64));

            let bench_id = format!("{}_4d_{}", distribution.name(), count);
            group.bench_with_input(
                BenchmarkId::new("tds_new", bench_id),
                &(count, distribution, actual_count),
                |b, &(count, distribution, _actual_count)| {
                    b.iter_batched(
                        || {
                            let points = generate_points_by_distribution::<4>(
                                count,
                                distribution,
                                DEFAULT_SEED,
                            );
                            points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                        },
                        |vertices| {
                            black_box(Tds::<f64, (), (), 4>::new(&vertices).unwrap());
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
            let sample_points =
                generate_points_by_distribution::<5>(count, distribution, DEFAULT_SEED);
            let actual_count = sample_points.len();
            group.throughput(Throughput::Elements(actual_count as u64));

            let bench_id = format!("{}_5d_{}", distribution.name(), count);
            group.bench_with_input(
                BenchmarkId::new("tds_new", bench_id),
                &(count, distribution, actual_count),
                |b, &(count, distribution, _actual_count)| {
                    b.iter_batched(
                        || {
                            let points = generate_points_by_distribution::<5>(
                                count,
                                distribution,
                                DEFAULT_SEED,
                            );
                            points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                        },
                        |vertices| {
                            black_box(Tds::<f64, (), (), 5>::new(&vertices).unwrap());
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

/// Calculate percentile from a slice of values using nearest-rank method
/// Supports configurable percentile via environment variable `BENCH_PERCENTILE` (default: 95)
fn calculate_percentile(values: &mut [u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }

    // Parse percentile from environment, defaulting to 95
    let percentile = std::env::var("BENCH_PERCENTILE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .map_or(95, |p| p.clamp(1, 100)); // Clamp to valid percentile range

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
#[expect(clippy::cast_precision_loss)]
fn print_alloc_summary(
    info: &AllocationInfo,
    description: &str,
    actual_point_count: usize,
    percentile_95: u64,
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
        "95th percentile bytes: {} ({:.2} MB)",
        percentile_95,
        percentile_95 as f64 / (1024.0 * 1024.0)
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

/// Generic helper to benchmark memory usage for a specific dimension D
#[expect(clippy::cast_possible_wrap)]
fn bench_memory_usage<const D: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_id_prefix: &str,
    count: usize,
) where
    [f64; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    group.bench_with_input(
        BenchmarkId::new(bench_id_prefix, count),
        &count,
        |b, &count| {
            b.iter_custom(|iters| {
                let mut total_time = Duration::new(0, 0);
                let mut allocation_infos: SmallBuffer<
                    AllocationInfo,
                    BENCHMARK_ITERATION_BUFFER_SIZE,
                > = SmallBuffer::new();

                let mut actual_point_counts: SmallBuffer<usize, BENCHMARK_ITERATION_BUFFER_SIZE> =
                    SmallBuffer::new();

                for _ in 0..iters {
                    let start_time = Instant::now();

                    let alloc_info = measure(|| {
                        let points = generate_points_by_distribution::<D>(
                            count,
                            PointDistribution::Random,
                            DEFAULT_SEED,
                        );
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        actual_point_counts.push(points.len()); // Track actual count
                        black_box(Tds::<f64, (), (), D>::new(&vertices).unwrap());
                    });

                    total_time += start_time.elapsed();
                    allocation_infos.push(alloc_info);
                }

                // Report memory usage summary if available
                if !allocation_infos.is_empty() {
                    // Safe cast for division - allocation_infos.len() is guaranteed to be small and non-zero
                    let divisor_unsigned = allocation_infos.len() as u64;
                    let divisor_signed = allocation_infos.len() as i64;
                    let avg_info = AllocationInfo {
                        count_total: allocation_infos.iter().map(|i| i.count_total).sum::<u64>()
                            / divisor_unsigned,
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
                        bytes_total: allocation_infos.iter().map(|i| i.bytes_total).sum::<u64>()
                            / divisor_unsigned,
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

                    // Calculate percentile of bytes_max (configurable via BENCH_PERCENTILE, default 95th)
                    let mut bytes_max_values: Vec<u64> =
                        allocation_infos.iter().map(|i| i.bytes_max).collect();
                    let percentile_value = calculate_percentile(&mut bytes_max_values);

                    print_alloc_summary(
                        &avg_info,
                        &format!("{D}D Triangulation"),
                        avg_actual_count,
                        percentile_value,
                    );
                }

                total_time
            });
        },
    );
}

/// Memory usage profiling across different scales and dimensions using allocation counter
fn benchmark_memory_profiling(c: &mut Criterion) {
    #[cfg(not(feature = "count-allocations"))]
    print_count_allocations_banner_once();

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
                let points = generate_points_by_distribution::<3>(
                    count,
                    PointDistribution::Random,
                    DEFAULT_SEED,
                );
                let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                let tds = Tds::<f64, (), (), 3>::new(&vertices).unwrap();

                // Generate query points
                let query_points = generate_points_by_distribution::<3>(
                    100,
                    PointDistribution::Random,
                    QUERY_SEED,
                );

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
                            {
                                use delaunay::geometry::predicates::insphere_lifted;
                                let result = insphere_lifted(points_for_test, query_point_obj);
                                query_results.push(result);
                            }

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
// Algorithmic Bottleneck Identification
// ============================================================================

/// Profile specific algorithmic components to identify bottlenecks
fn benchmark_algorithmic_bottlenecks(c: &mut Criterion) {
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
                        let points = generate_points_by_distribution::<3>(
                            count,
                            PointDistribution::Random,
                            DEFAULT_SEED,
                        );
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        DelaunayTriangulation::<_, (), (), 3>::new(&vertices).unwrap()
                    },
                    |dt| {
                        let boundary_facets =
                            dt.tds().boundary_facets().expect("boundary_facets failed");
                        black_box(boundary_facets);
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
                        let points = generate_points_by_distribution::<3>(
                            count,
                            PointDistribution::Random,
                            DEFAULT_SEED,
                        );
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        DelaunayTriangulation::<_, (), (), 3>::new(&vertices).unwrap()
                    },
                    |dt| {
                        let hull = delaunay::geometry::algorithms::convex_hull::ConvexHull::from_triangulation(dt.triangulation()).unwrap();
                        black_box(hull);
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
        // Allow configuration via environment variables for CI stability
        let sample_size = std::env::var("BENCH_SAMPLE_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let warm_up_secs = std::env::var("BENCH_WARMUP_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);

        Criterion::default()
            .sample_size(sample_size)  // Configurable samples (default: 10 for long-running benchmarks)
            .warm_up_time(Duration::from_secs(warm_up_secs))  // Configurable warm-up (default: 10s)
            .measurement_time(bench_time(60))
    };
    targets =
        benchmark_triangulation_scaling,
        benchmark_memory_profiling,
        benchmark_query_latency,
        benchmark_algorithmic_bottlenecks
);

criterion_main!(profiling_benches);
