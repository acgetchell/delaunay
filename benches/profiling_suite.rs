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

#![allow(missing_docs)]

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::core::collections::SmallBuffer;
use delaunay::geometry::util::{
    generate_grid_points, generate_poisson_points, generate_random_points_seeded,
};
use delaunay::prelude::*;
use delaunay::vertex;
use std::hint::black_box;
use std::time::{Duration, Instant};

// SmallBuffer size constants for different use cases
const BENCHMARK_ITERATION_BUFFER_SIZE: usize = 8; // For tracking allocation info across benchmark iterations
const SIMPLEX_VERTICES_BUFFER_SIZE: usize = 8; // For 3D simplex vertices (4) with some headroom
const QUERY_RESULTS_BUFFER_SIZE: usize = 1024; // For bounded query result collections (max 1000 in code)

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

/// Get appropriate point counts based on environment
fn get_profiling_counts() -> &'static [usize] {
    if std::env::var("PROFILING_DEV_MODE").is_ok() {
        PROFILING_COUNTS_DEVELOPMENT
    } else {
        PROFILING_COUNTS_PRODUCTION
    }
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
        PointDistribution::Random => {
            generate_random_points_seeded(count, (-100.0, 100.0), seed).unwrap()
        }
        PointDistribution::Grid => {
            // Calculate points per dimension to get approximately `count` points total
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let points_per_dim = ((count as f64).powf(1.0 / D as f64).ceil() as usize).max(2);
            generate_grid_points(points_per_dim, 10.0, [0.0; D]).unwrap()
        }
        PointDistribution::PoissonDisk => {
            let min_distance = match D {
                2 => 5.0,  // 2D: reasonable spacing for [-100, 100] bounds
                3 => 8.0,  // 3D: slightly larger spacing
                4 => 12.0, // 4D: larger spacing for higher dimensions
                5 => 15.0, // 5D: even larger spacing
                _ => 20.0, // Higher dimensions: very large spacing
            };
            generate_poisson_points(count, (-100.0, 100.0), min_distance, seed).unwrap()
        }
    }
}

// ============================================================================
// Large-Scale Triangulation Performance Profiling
// ============================================================================

/// Comprehensive triangulation scaling analysis across dimensions and distributions
#[allow(clippy::significant_drop_tightening)]
fn benchmark_triangulation_scaling(c: &mut Criterion) {
    let counts = get_profiling_counts();
    let distributions = [
        PointDistribution::Random,
        PointDistribution::Grid,
        PointDistribution::PoissonDisk,
    ];

    // 2D Triangulation Scaling
    let mut group = c.benchmark_group("triangulation_scaling_2d");
    group.measurement_time(Duration::from_secs(120)); // Allow longer measurement time

    for &count in counts {
        for &distribution in &distributions {
            // Calculate actual point count for accurate throughput metrics
            let sample_points = generate_points_by_distribution::<2>(count, distribution, 42);
            let actual_count = sample_points.len();
            group.throughput(Throughput::Elements(actual_count as u64));

            let bench_id = format!("{}_2d_{}", distribution.name(), count);
            group.bench_with_input(
                BenchmarkId::new("tds_new", bench_id),
                &(count, distribution),
                |b, &(count, distribution)| {
                    b.iter_batched(
                        || {
                            let points =
                                generate_points_by_distribution::<2>(count, distribution, 42);
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
    group.measurement_time(Duration::from_secs(180)); // Even longer for 3D

    for &count in counts {
        for &distribution in &distributions {
            // Skip very large counts for 3D in development mode to prevent timeouts
            if std::env::var("PROFILING_DEV_MODE").is_ok() && count > 10_000 {
                continue;
            }

            // Calculate actual point count for accurate throughput metrics
            let sample_points = generate_points_by_distribution::<3>(count, distribution, 42);
            let actual_count = sample_points.len();
            group.throughput(Throughput::Elements(actual_count as u64));

            let bench_id = format!("{}_3d_{}", distribution.name(), count);
            group.bench_with_input(
                BenchmarkId::new("tds_new", bench_id),
                &(count, distribution),
                |b, &(count, distribution)| {
                    b.iter_batched(
                        || {
                            let points =
                                generate_points_by_distribution::<3>(count, distribution, 42);
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
    let high_dim_counts = if std::env::var("PROFILING_DEV_MODE").is_ok() {
        &[1_000, 3_000][..]
    } else {
        &[1_000, 3_000, 10_000][..]
    };

    // 4D Triangulation Scaling
    let mut group = c.benchmark_group("triangulation_scaling_4d");
    group.measurement_time(Duration::from_secs(240));

    for &count in high_dim_counts {
        for &distribution in &distributions {
            // Calculate actual point count for accurate throughput metrics
            let sample_points = generate_points_by_distribution::<4>(count, distribution, 42);
            let actual_count = sample_points.len();
            group.throughput(Throughput::Elements(actual_count as u64));

            let bench_id = format!("{}_4d_{}", distribution.name(), count);
            group.bench_with_input(
                BenchmarkId::new("tds_new", bench_id),
                &(count, distribution),
                |b, &(count, distribution)| {
                    b.iter_batched(
                        || {
                            let points =
                                generate_points_by_distribution::<4>(count, distribution, 42);
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
}

// ============================================================================
// Memory Usage Profiling
// ============================================================================

/// Print memory allocation summary
#[allow(clippy::cast_precision_loss)]
fn print_alloc_summary(info: &AllocationInfo, description: &str, actual_point_count: usize) {
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
        "Bytes per point (peak): {:.1}",
        info.bytes_max as f64 / actual_point_count as f64
    );
    println!("=====================================\n");
}

/// Memory usage profiling across different scales and dimensions using allocation counter
#[allow(clippy::too_many_lines, clippy::cast_possible_wrap)]
fn benchmark_memory_profiling(c: &mut Criterion) {
    let counts = if std::env::var("PROFILING_DEV_MODE").is_ok() {
        &[1_000, 10_000][..]
    } else {
        &[1_000, 10_000, 100_000][..]
    };

    let mut group = c.benchmark_group("memory_profiling");
    group.measurement_time(Duration::from_secs(60));

    for &count in counts {
        // 2D Memory Profiling
        group.bench_with_input(
            BenchmarkId::new("memory_usage_2d", count),
            &count,
            |b, &count| {
                b.iter_custom(|iters| {
                    let mut total_time = Duration::new(0, 0);
                    let mut allocation_infos: SmallBuffer<
                        AllocationInfo,
                        BENCHMARK_ITERATION_BUFFER_SIZE,
                    > = SmallBuffer::new();

                    let mut actual_point_counts: SmallBuffer<
                        usize,
                        BENCHMARK_ITERATION_BUFFER_SIZE,
                    > = SmallBuffer::new();

                    for _ in 0..iters {
                        let start_time = Instant::now();

                        let alloc_info = measure(|| {
                            let points = generate_points_by_distribution::<2>(
                                count,
                                PointDistribution::Random,
                                42,
                            );
                            let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                            actual_point_counts.push(points.len()); // Track actual count
                            black_box(Tds::<f64, (), (), 2>::new(&vertices).unwrap());
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
                            count_total: allocation_infos
                                .iter()
                                .map(|i| i.count_total)
                                .sum::<u64>()
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
                            bytes_total: allocation_infos
                                .iter()
                                .map(|i| i.bytes_total)
                                .sum::<u64>()
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
                        let avg_actual_count =
                            actual_point_counts.iter().sum::<usize>() / actual_point_counts.len();
                        print_alloc_summary(&avg_info, "2D Triangulation", avg_actual_count);
                    }

                    total_time
                });
            },
        );

        // 3D Memory Profiling (smaller counts due to complexity)
        if count <= 10_000 {
            group.bench_with_input(
                BenchmarkId::new("memory_usage_3d", count),
                &count,
                |b, &count| {
                    b.iter_custom(|iters| {
                        let mut total_time = Duration::new(0, 0);
                        let mut allocation_infos: SmallBuffer<
                            AllocationInfo,
                            BENCHMARK_ITERATION_BUFFER_SIZE,
                        > = SmallBuffer::new();

                        let mut actual_point_counts: SmallBuffer<
                            usize,
                            BENCHMARK_ITERATION_BUFFER_SIZE,
                        > = SmallBuffer::new();

                        for _ in 0..iters {
                            let start_time = Instant::now();

                            let alloc_info = measure(|| {
                                let points = generate_points_by_distribution::<3>(
                                    count,
                                    PointDistribution::Random,
                                    42,
                                );
                                let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                                actual_point_counts.push(points.len()); // Track actual count
                                black_box(Tds::<f64, (), (), 3>::new(&vertices).unwrap());
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
                                count_total: allocation_infos
                                    .iter()
                                    .map(|i| i.count_total)
                                    .sum::<u64>()
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
                                bytes_total: allocation_infos
                                    .iter()
                                    .map(|i| i.bytes_total)
                                    .sum::<u64>()
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
                            let avg_actual_count = actual_point_counts.iter().sum::<usize>()
                                / actual_point_counts.len();
                            print_alloc_summary(&avg_info, "3D Triangulation", avg_actual_count);
                        }

                        total_time
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Query Latency Analysis
// ============================================================================

/// Query latency profiling for circumsphere containment tests
fn benchmark_query_latency(c: &mut Criterion) {
    let counts = if std::env::var("PROFILING_DEV_MODE").is_ok() {
        &[1_000, 3_000][..]
    } else {
        &[1_000, 10_000, 30_000][..]
    };

    let mut group = c.benchmark_group("query_latency");
    group.measurement_time(Duration::from_secs(90));

    for &count in counts {
        // Create triangulation and test circumsphere queries
        group.bench_with_input(
            BenchmarkId::new("circumsphere_queries_3d", count),
            &count,
            |b, &count| {
                // Setup: Create triangulation and query points
                let points =
                    generate_points_by_distribution::<3>(count, PointDistribution::Random, 42);
                let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                let tds = Tds::<f64, (), (), 3>::new(&vertices).unwrap();

                // Generate query points
                let query_points =
                    generate_points_by_distribution::<3>(100, PointDistribution::Random, 123);

                b.iter(|| {
                    // Perform circumsphere containment queries
                    // Using SmallBuffer since we limit to max 1000 results
                    let mut query_results: SmallBuffer<_, QUERY_RESULTS_BUFFER_SIZE> =
                        SmallBuffer::new();

                    for cell in tds.cells() {
                        // Precompute once per cell to avoid redundant allocations
                        // Using SmallBuffer since 3D cells have exactly 4 vertices
                        let vertex_coords: SmallBuffer<[f64; 3], SIMPLEX_VERTICES_BUFFER_SIZE> =
                            cell.1
                                .vertices()
                                .iter()
                                .map(std::convert::Into::into)
                                .collect();

                        if vertex_coords.len() == 4 {
                            // Valid 3D simplex - convert coordinates once per cell
                            // Using SmallBuffer since 3D simplex has exactly 4 vertices
                            let points_for_test: SmallBuffer<
                                Point<f64, 3>,
                                SIMPLEX_VERTICES_BUFFER_SIZE,
                            > = vertex_coords.iter().copied().map(Point::new).collect();

                            for query_point in &query_points {
                                let query_coords: [f64; 3] = (*query_point).into();
                                let query_point_obj = Point::new(query_coords);

                                // Use the fastest circumsphere method (based on benchmark results)
                                #[allow(clippy::items_after_statements)]
                                {
                                    use delaunay::geometry::predicates::insphere_lifted;
                                    let result = insphere_lifted(&points_for_test, query_point_obj);
                                    query_results.push(result);
                                }

                                // Limit total queries to prevent extremely long benchmarks
                                if query_results.len() >= 1000 {
                                    break;
                                }
                            }
                        }

                        if query_results.len() >= 1000 {
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
    let counts = if std::env::var("PROFILING_DEV_MODE").is_ok() {
        &[3_000][..]
    } else {
        &[10_000][..]
    };

    let mut group = c.benchmark_group("algorithmic_bottlenecks");
    group.measurement_time(Duration::from_secs(120));

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
                            42,
                        );
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        Tds::<f64, (), (), 3>::new(&vertices).unwrap()
                    },
                    |tds| {
                        let boundary_facets =
                            tds.boundary_facets().expect("boundary_facets failed");
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
                            42,
                        );
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        Tds::<f64, (), (), 3>::new(&vertices).unwrap()
                    },
                    |tds| {
                        let hull = delaunay::geometry::algorithms::convex_hull::ConvexHull::from_triangulation(&tds).unwrap();
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
    config = Criterion::default()
        .sample_size(10)  // Fewer samples due to long-running nature
        .warm_up_time(Duration::from_secs(10))
        .measurement_time(Duration::from_secs(60));
    targets =
        benchmark_triangulation_scaling,
        benchmark_memory_profiling,
        benchmark_query_latency,
        benchmark_algorithmic_bottlenecks
);

criterion_main!(profiling_benches);
