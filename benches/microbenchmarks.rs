//! Microbenchmarks for key delaunay methods
//!
//! This benchmark suite focuses on measuring the performance of individual key methods
//! in the delaunay triangulation library, particularly those that are performance-critical:
//!
//! 1. **`DelaunayTriangulation::with_kernel`**: Complete triangulation creation
//! 2. **Layered validation**: `dt.tds().is_valid()/validate()`, `dt.as_triangulation().is_valid()/validate()`, `dt.is_valid()`, `dt.validate()`
//! 3. **Incremental construction**: Performance of `insert()` method for vertex insertion
//! 4. **Memory usage patterns**: Allocation and deallocation patterns
//!
//! These benchmarks measure the effectiveness of the optimization implementations
//! completed as part of the Pure Incremental Delaunay Triangulation refactoring project.

#![allow(missing_docs)] // Criterion macros generate undocumented functions

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::core::delaunay_triangulation::DelaunayTriangulation;
use delaunay::geometry::kernel::RobustKernel;
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::prelude::query::*;
use delaunay::vertex;
use std::hint::black_box;
use std::sync::OnceLock;

/// Get the deterministic seed for random point generation.
/// Reads `DELAUNAY_BENCH_SEED` (decimal or 0x-hex). Defaults to 0xD1EA.
/// Prints the resolved seed once on first use if `PRINT_BENCH_SEED` is set.
fn get_benchmark_seed() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| {
        let seed = std::env::var("DELAUNAY_BENCH_SEED")
            .ok()
            .and_then(|s| {
                let s = s.trim();
                s.strip_prefix("0x")
                    .or_else(|| s.strip_prefix("0X"))
                    .map_or_else(|| s.parse().ok(), |hex| u64::from_str_radix(hex, 16).ok())
            })
            .unwrap_or(0xD1EA);
        if std::env::var("PRINT_BENCH_SEED").is_ok() {
            eprintln!("Benchmark seed: 0x{seed:X} ({seed})");
        }
        seed
    })
}

/// Macro to generate comprehensive dimensional benchmarks for core algorithms
macro_rules! generate_dimensional_benchmarks {
    ($dim:literal) => {
        pastey::paste! {
            /// Benchmark incremental Delaunay triangulation for [<$dim>]D
            fn [<benchmark_delaunay_triangulation_ $dim d>](c: &mut Criterion) {
                let point_counts = [10, 25, 50, 100, 250];
                let seed = get_benchmark_seed(); // Cache seed locally for consistency across iterations

                let mut group = c.benchmark_group(concat!("delaunay_triangulation_", stringify!([<$dim>]), "d"));

                for &n_points in &point_counts {
                    let throughput = n_points as u64;
                    group.throughput(Throughput::Elements(throughput));

                    group.bench_with_input(
                        BenchmarkId::new("with_kernel", n_points),
                        &n_points,
                        |b, &n_points| {
                            b.iter_batched(
                                || {
                                    let points: Vec<Point<f64, $dim>> = generate_random_points_seeded(n_points, (-100.0, 100.0), seed).unwrap();
                                    points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                                },
                                |vertices| black_box(DelaunayTriangulation::<RobustKernel<f64>, (), (), $dim>::with_kernel(&RobustKernel::new(), &vertices).unwrap()),
                                BatchSize::LargeInput,
                            );
                        },
                    );
                }

                group.finish();
            }
        }
    };
}

// Generate comprehensive benchmarks for dimensions 2-5
generate_dimensional_benchmarks!(2);
generate_dimensional_benchmarks!(3);
generate_dimensional_benchmarks!(4);
generate_dimensional_benchmarks!(5);

/// Macro to generate memory usage benchmarks for all dimensions
macro_rules! generate_memory_usage_benchmarks {
    ($dim:literal) => {
        pastey::paste! {
            /// Benchmark memory allocation patterns for [<$dim>]D
            fn [<benchmark_memory_usage_ $dim d>](c: &mut Criterion) {
                let point_counts: &[usize] = if $dim <= 3 { &[50, 100, 200] } else { &[20, 50, 100] };
                let seed = get_benchmark_seed(); // Cache seed locally for consistency across iterations

                let mut group = c.benchmark_group(&format!("memory_usage_{}d", $dim));

                for &n_points in point_counts {
                    group.bench_with_input(
                        BenchmarkId::new("triangulation_memory", n_points),
                        &n_points,
                        |b, &n_points| {
                            b.iter(|| {
                                // Measure complete triangulation creation and destruction
                                let points: Vec<Point<f64, $dim>> = generate_random_points_seeded(n_points, (-100.0, 100.0), seed).unwrap();
                                let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                                let dt = DelaunayTriangulation::<RobustKernel<f64>, (), (), $dim>::with_kernel(&RobustKernel::new(), &vertices).unwrap();
                                black_box((dt.tds().number_of_vertices(), dt.tds().number_of_cells()))
                            });
                        },
                    );
                }

                group.finish();
            }
        }
    };
}

// Generate memory usage benchmarks for dimensions 2-5
generate_memory_usage_benchmarks!(2);
generate_memory_usage_benchmarks!(3);
generate_memory_usage_benchmarks!(4);
generate_memory_usage_benchmarks!(5);

/// Macro to generate validation method benchmarks for all dimensions
macro_rules! generate_validation_benchmarks {
    ($dim:literal) => {
        pastey::paste! {
            /// Benchmark validation methods performance for [<$dim>]D
            fn [<benchmark_validation_methods_ $dim d>](c: &mut Criterion) {
                let point_counts: &[usize] = if $dim <= 3 { &[10, 25, 50, 100] } else { &[10, 25, 50] };
                let seed = get_benchmark_seed(); // Cache seed locally for consistency across iterations

                let mut group = c.benchmark_group(&format!("validation_methods_{}d", $dim));

                for &n_points in point_counts {
                    let throughput = n_points as u64;
                    group.throughput(Throughput::Elements(throughput));

                    group.bench_with_input(
                        BenchmarkId::new("validate", n_points),
                        &n_points,
                        |b, &n_points| {
                            b.iter_batched(
                                || {
                                    let points: Vec<Point<f64, $dim>> = generate_random_points_seeded(n_points, (-100.0, 100.0), seed).unwrap();
                                    let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                                    DelaunayTriangulation::<RobustKernel<f64>, (), (), $dim>::with_kernel(&RobustKernel::new(), &vertices).unwrap()

                                },
                                |dt| {
                                    dt.validate().unwrap();
                                    black_box(dt);
                                },
                                BatchSize::LargeInput,
                            );
                        },
                    );

                    group.bench_with_input(
                        BenchmarkId::new("is_valid_delaunay", n_points),
                        &n_points,
                        |b, &n_points| {
                            b.iter_batched(
                                || {
                                    let points: Vec<Point<f64, $dim>> = generate_random_points_seeded(n_points, (-100.0, 100.0), seed).unwrap();
                                    let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                                    DelaunayTriangulation::<RobustKernel<f64>, (), (), $dim>::with_kernel(&RobustKernel::new(), &vertices).unwrap()
                                },
                                |dt| {
                                    dt.is_valid().unwrap();
                                    black_box(dt);
                                },
                                BatchSize::LargeInput,
                            );
                        },
                    );
                }

                group.finish();
            }

            /// Benchmark individual validation components for [<$dim>]D
            fn [<benchmark_validation_components_ $dim d>](c: &mut Criterion) {
                let seed = get_benchmark_seed(); // Cache seed locally for consistency across iterations
                let n_points = if $dim <= 3 { 50 } else { 25 }; // Fixed size for component benchmarks
                let points: Vec<Point<f64, $dim>> = generate_random_points_seeded(n_points, (-100.0, 100.0), seed).unwrap();
                let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                let dt = DelaunayTriangulation::<RobustKernel<f64>, (), (), $dim>::with_kernel(&RobustKernel::new(), &vertices).unwrap();

                let mut group = c.benchmark_group(&format!("validation_components_{}d", $dim));

                group.bench_function("tds_is_valid", |b| {
                    b.iter(|| {
                        dt.tds().is_valid().unwrap();
                        // Black box to prevent dead code elimination
                        black_box(());
                    });
                });

                group.bench_function("tri_is_valid", |b| {
                    b.iter(|| {
                        dt.as_triangulation().is_valid().unwrap();
                        // Black box to prevent dead code elimination
                        black_box(());
                    });
                });

                group.bench_function("is_valid_delaunay", |b| {
                    b.iter(|| {
                        dt.is_valid().unwrap();
                        // Black box to prevent dead code elimination
                        black_box(());
                    });
                });

                group.bench_function("validate", |b| {
                    b.iter(|| {
                        dt.validate().unwrap();
                        // Black box to prevent dead code elimination
                        black_box(());
                    });
                });

                group.finish();
            }
        }
    };
}

// Generate validation benchmarks for dimensions 2-5
generate_validation_benchmarks!(2);
generate_validation_benchmarks!(3);
generate_validation_benchmarks!(4);
generate_validation_benchmarks!(5);

/// Macro to generate incremental construction benchmarks for all dimensions
macro_rules! generate_incremental_construction_benchmarks {
    ($dim:literal) => {
        pastey::paste! {
            /// Benchmark incremental vertex addition for [<$dim>]D
            fn [<benchmark_incremental_construction_ $dim d>](c: &mut Criterion) {
                let seed = get_benchmark_seed(); // Cache seed locally for consistency across iterations
                let mut group = c.benchmark_group(&format!("incremental_construction_{}d", $dim));

                // Generate initial simplex for the given dimension
                let mut initial_coords = Vec::new();
                for i in 0..=$dim {
                    let mut coords = vec![0.0; $dim];
                    if i < $dim {
                        coords[i] = 1.0;
                    }
                    initial_coords.push(coords);
                }
                let initial_vertices: Vec<_> = initial_coords
                    .into_iter()
                    .map(|coords| {
                        let mut array = [0.0; $dim];
                        array.copy_from_slice(&coords);
                        vertex!(array)
                    })
                    .collect();

                // Test single vertex addition
                let additional_coords = vec![0.5; $dim];
                let mut additional_array = [0.0; $dim];
                additional_array.copy_from_slice(&additional_coords);
                // Note: additional_vertex is Copy, so we can use the same value in each benchmark iteration
                let additional_vertex = vertex!(additional_array);

                group.bench_function("single_vertex_addition", |b| {
                    b.iter_batched(
                        || DelaunayTriangulation::<RobustKernel<f64>, (), (), $dim>::with_kernel(&RobustKernel::new(), &initial_vertices).unwrap(),
                        |mut dt| {
                            dt.insert(additional_vertex).unwrap();
                            black_box(dt);
                        },
                        BatchSize::SmallInput,
                    );
                });

                // Test multiple vertex additions with dimension-appropriate counts
                let counts: &[usize] = if $dim <= 3 { &[2, 5, 10] } else { &[2, 4, 6] };
                for &count in counts {
                    group.bench_with_input(
                        BenchmarkId::new("multiple_vertex_addition", count),
                        &count,
                        |b, &count| {
                            b.iter_batched(
                                || {
                                    let dt = DelaunayTriangulation::<RobustKernel<f64>, (), (), $dim>::with_kernel(&RobustKernel::new(), &initial_vertices).unwrap();
                                    let additional_points: Vec<Point<f64, $dim>> = generate_random_points_seeded(count, (-100.0, 100.0), seed).unwrap();
                                    let additional_vertices: Vec<_> =
                                        additional_points.iter().map(|p| vertex!(*p)).collect();
                                    (dt, additional_vertices)
                                },
                                |(mut dt, additional_vertices)| {
                                    for vertex in additional_vertices {
                                        dt.insert(vertex).unwrap();
                                    }
                                    black_box(dt);
                                },
                                BatchSize::SmallInput,
                            );
                        },
                    );
                }

                group.finish();
            }
        }
    };
}

// Generate incremental construction benchmarks for dimensions 2-5
generate_incremental_construction_benchmarks!(2);
generate_incremental_construction_benchmarks!(3);
generate_incremental_construction_benchmarks!(4);
generate_incremental_construction_benchmarks!(5);

/// Build Criterion configuration with optional environment variable overrides.
///
/// Supports:
/// - `CRIT_SAMPLE_SIZE`: Number of samples per benchmark (default: Criterion's default)
/// - `CRIT_MEASUREMENT_MS`: Measurement time in milliseconds (default: Criterion's default)
/// - `CRIT_WARMUP_MS`: Warm-up time in milliseconds (default: Criterion's default)
///
/// This allows CI and local tuning without code changes.
fn bench_config() -> Criterion {
    use std::time::Duration;
    let mut c = Criterion::default();

    if let Some(v) = std::env::var("CRIT_SAMPLE_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
    {
        c = c.sample_size(v);
    } else if std::env::var("CRIT_SAMPLE_SIZE").is_ok() {
        eprintln!("Warning: Failed to parse CRIT_SAMPLE_SIZE, using default");
    }

    if let Some(v) = std::env::var("CRIT_MEASUREMENT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        c = c.measurement_time(Duration::from_millis(v));
    } else if std::env::var("CRIT_MEASUREMENT_MS").is_ok() {
        eprintln!("Warning: Failed to parse CRIT_MEASUREMENT_MS, using default");
    }

    if let Some(v) = std::env::var("CRIT_WARMUP_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        c = c.warm_up_time(Duration::from_millis(v));
    } else if std::env::var("CRIT_WARMUP_MS").is_ok() {
        eprintln!("Warning: Failed to parse CRIT_WARMUP_MS, using default");
    }

    c
}

criterion_group!(
    name = benches;
    config = bench_config();
    targets =
        // Core triangulation benchmarks (2D-5D)
        benchmark_delaunay_triangulation_2d,
        benchmark_delaunay_triangulation_3d,
        benchmark_delaunay_triangulation_4d,
        benchmark_delaunay_triangulation_5d,

        // Memory usage benchmarks (2D-5D)
        benchmark_memory_usage_2d,
        benchmark_memory_usage_3d,
        benchmark_memory_usage_4d,
        benchmark_memory_usage_5d,

        // Validation benchmarks (2D-5D)
        benchmark_validation_methods_2d,
        benchmark_validation_methods_3d,
        benchmark_validation_methods_4d,
        benchmark_validation_methods_5d,
        benchmark_validation_components_2d,
        benchmark_validation_components_3d,
        benchmark_validation_components_4d,
        benchmark_validation_components_5d,

        // Incremental construction benchmarks (2D-5D)
        benchmark_incremental_construction_2d,
        benchmark_incremental_construction_3d,
        benchmark_incremental_construction_4d,
        benchmark_incremental_construction_5d
);
criterion_main!(benches);
