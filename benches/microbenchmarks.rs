//! Microbenchmarks for key delaunay methods
//!
//! This benchmark suite focuses on measuring the performance of individual key methods
//! in the delaunay triangulation library, particularly those that are performance-critical:
//!
//! 1. **`Tds::new` (Bowyer-Watson triangulation)**: Complete triangulation creation
//! 2. **`remove_duplicate_cells`**: Duplicate cell removal and cleanup
//! 3. **`is_valid`**: Complete triangulation validation performance
//! 4. **Individual validation components**: Mapping validation, duplicate detection, etc.
//! 5. **Incremental construction**: Performance of `add()` method for vertex insertion
//! 6. **Memory usage patterns**: Allocation and deallocation patterns
//!
//! **Note:** `assign_neighbors` benchmarks have been moved to `assign_neighbors_performance.rs`
//! for more comprehensive testing with multiple distributions (random, grid, spherical) and
//! scaling analysis. Use that benchmark file for `assign_neighbors` performance evaluation.
//!
//! These benchmarks measure the effectiveness of the optimization implementations
//! completed as part of the Pure Incremental Delaunay Triangulation refactoring project.
//!
//! # Safety and Invariant Violations
//!
//! **WARNING**: Some benchmarks in this file intentionally violate TDS invariants for
//! performance testing purposes. Specifically:
//!
//! - `remove_duplicate_cells` benchmarks directly insert duplicate cells without updating
//!   UUID mappings to create test scenarios for the cleanup algorithm.
//!
//! **THESE PATTERNS MUST NEVER BE USED IN**:
//! - Production code
//! - Correctness tests
//! - Example code
//! - Library documentation
//!
//! They exist solely for microbenchmarking internal cleanup performance.

#![allow(missing_docs)] // Criterion macros generate undocumented functions

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::prelude::*;
use delaunay::vertex;
use std::hint::black_box;

/// Get the seed for deterministic random point generation.
/// Checks `DELAUNAY_BENCH_SEED` environment variable, defaults to 0xD1EA ("DEEA" - Delaunay).
/// Supports both decimal and hexadecimal (0x-prefixed) seeds.
fn get_benchmark_seed() -> u64 {
    std::env::var("DELAUNAY_BENCH_SEED")
        .ok()
        .map(|s| s.trim().to_string())
        .and_then(|s| {
            s.strip_prefix("0x")
                .or_else(|| s.strip_prefix("0X"))
                .map_or_else(|| s.parse().ok(), |hex| u64::from_str_radix(hex, 16).ok())
        })
        .unwrap_or(0xD1EA)
}

/// Macro to generate comprehensive dimensional benchmarks for core algorithms
macro_rules! generate_dimensional_benchmarks {
    ($dim:literal) => {
        pastey::paste! {
            /// Benchmark Bowyer-Watson triangulation for [<$dim>]D
            fn [<benchmark_bowyer_watson_triangulation_ $dim d>](c: &mut Criterion) {
                let point_counts = [10, 25, 50, 100, 250];
                let seed = get_benchmark_seed(); // Cache seed to avoid repeated env var parsing

                let mut group = c.benchmark_group(concat!("bowyer_watson_triangulation_", stringify!([<$dim>]), "d"));

                for &n_points in &point_counts {
                    #[allow(clippy::cast_sign_loss)]
                    let throughput = n_points as u64;
                    group.throughput(Throughput::Elements(throughput));

                    group.bench_with_input(
                        BenchmarkId::new("tds_new", n_points),
                        &n_points,
                        |b, &n_points| {
                            b.iter_batched(
                                || {
                                    let points: Vec<Point<f64, $dim>> = generate_random_points_seeded(n_points, (-100.0, 100.0), seed).unwrap();
                                    points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                                },
                                |vertices| black_box(Tds::<f64, (), (), $dim>::new(&vertices).unwrap()),
                                BatchSize::LargeInput,
                            );
                        },
                    );
                }

                group.finish();
            }

            /// Benchmark `remove_duplicate_cells` for [<$dim>]D
            fn [<benchmark_remove_duplicate_cells_ $dim d>](c: &mut Criterion) {
                let point_counts = [10, 25, 50, 100];
                let seed = get_benchmark_seed(); // Cache seed to avoid repeated env var parsing

                let mut group = c.benchmark_group(concat!("remove_duplicate_cells_", stringify!([<$dim>]), "d"));

                for &n_points in &point_counts {
                    #[allow(clippy::cast_sign_loss)]
                    let throughput = n_points as u64;
                    group.throughput(Throughput::Elements(throughput));

                    group.bench_with_input(
                        BenchmarkId::new("remove_duplicate_cells", n_points),
                        &n_points,
                        |b, &n_points| {
                            b.iter_batched(
                                || {
                                    let points: Vec<Point<f64, $dim>> = generate_random_points_seeded(n_points, (-100.0, 100.0), seed).unwrap();
                                    let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                                    // Note: tds must be mutable for insert_cell_unchecked() calls below
                                    let mut tds = Tds::<f64, (), (), $dim>::new(&vertices).unwrap();

                                    // ============================================================
                                    // BENCH-ONLY INVARIANT VIOLATION ZONE - DO NOT COPY
                                    // ============================================================
                                    // WARNING: This code intentionally violates TDS invariants by
                                    // directly inserting duplicate cells without updating UUID mappings.
                                    // This is ONLY for performance testing of `remove_duplicate_cells`.
                                    // DO NOT use this pattern in:
                                    // - Production code
                                    // - Correctness tests
                                    // - Examples
                                    // - Documentation
                                    // Note: This code only runs in benchmarks and is clearly documented as
                                    // bench-only invariant violation. No additional cfg guard is needed.
                                    #[allow(deprecated)]
                                    {
                                        // Scoped import to avoid items_after_statements warning
                                        use delaunay::cell;
                                        let cell_vertices: Vec<_> = tds.vertices().map(|(_, v)| *v).collect();
                                        if cell_vertices.len() >= ($dim + 1) {
                                            // SAFETY(BENCH-ONLY): Deliberately create duplicates for perf testing
                                            for _ in 0..3 {
                                                let duplicate_cell = cell!(cell_vertices[0..($dim + 1)].to_vec());
                                                let _cell_key = tds.insert_cell_unchecked(duplicate_cell);
                                                // Intentionally not updating UUID mappings to create true duplicates
                                            }
                                        }
                                    }
                                    // ============================================================
                                    // END INVARIANT VIOLATION ZONE
                                    // ============================================================
                                    tds
                                },
                                |mut tds| {
                                    let removed = tds.remove_duplicate_cells().expect("remove_duplicate_cells failed");
                                    black_box((tds, removed));
                                },
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

// Legacy 3D benchmark function for backward compatibility
fn benchmark_bowyer_watson_triangulation(c: &mut Criterion) {
    benchmark_bowyer_watson_triangulation_3d(c);
}

// Legacy 3D benchmark function for backward compatibility
fn benchmark_remove_duplicate_cells(c: &mut Criterion) {
    benchmark_remove_duplicate_cells_3d(c);
}

// Legacy dimensional benchmark functions for backward compatibility
fn benchmark_2d_triangulation(c: &mut Criterion) {
    benchmark_bowyer_watson_triangulation_2d(c);
}

fn benchmark_4d_triangulation(c: &mut Criterion) {
    benchmark_bowyer_watson_triangulation_4d(c);
}

fn benchmark_5d_triangulation(c: &mut Criterion) {
    benchmark_bowyer_watson_triangulation_5d(c);
}

/// Macro to generate memory usage benchmarks for all dimensions
macro_rules! generate_memory_usage_benchmarks {
    ($dim:literal) => {
        pastey::paste! {
            /// Benchmark memory allocation patterns for [<$dim>]D
            fn [<benchmark_memory_usage_ $dim d>](c: &mut Criterion) {
                let point_counts: &[usize] = if $dim <= 3 { &[50, 100, 200] } else { &[20, 50, 100] };
                let seed = get_benchmark_seed(); // Cache seed to avoid repeated env var parsing

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
                                let tds = Tds::<f64, (), (), $dim>::new(&vertices).unwrap();
                                black_box((tds.number_of_vertices(), tds.number_of_cells()))
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

// Legacy wrapper for backward compatibility
fn benchmark_memory_usage(c: &mut Criterion) {
    benchmark_memory_usage_3d(c);
}

/// Macro to generate validation method benchmarks for all dimensions
macro_rules! generate_validation_benchmarks {
    ($dim:literal) => {
        pastey::paste! {
            /// Benchmark validation methods performance for [<$dim>]D
            fn [<benchmark_validation_methods_ $dim d>](c: &mut Criterion) {
                let point_counts: &[usize] = if $dim <= 3 { &[10, 25, 50, 100] } else { &[10, 25, 50] };
                let seed = get_benchmark_seed(); // Cache seed to avoid repeated env var parsing

                let mut group = c.benchmark_group(&format!("validation_methods_{}d", $dim));

                for &n_points in point_counts {
                    #[allow(clippy::cast_sign_loss)]
                    let throughput = n_points as u64;
                    group.throughput(Throughput::Elements(throughput));

                    group.bench_with_input(
                        BenchmarkId::new("is_valid", n_points),
                        &n_points,
                        |b, &n_points| {
                            b.iter_batched(
                                || {
                                    let points: Vec<Point<f64, $dim>> = generate_random_points_seeded(n_points, (-100.0, 100.0), seed).unwrap();
                                    let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                                    Tds::<f64, (), (), $dim>::new(&vertices).unwrap()
                                },
                                |tds| {
                                    tds.is_valid().unwrap();
                                    black_box(tds);
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
                let seed = get_benchmark_seed(); // Cache seed to avoid repeated env var parsing
                let n_points = if $dim <= 3 { 50 } else { 25 }; // Fixed size for component benchmarks
                let points: Vec<Point<f64, $dim>> = generate_random_points_seeded(n_points, (-100.0, 100.0), seed).unwrap();
                let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                let tds = Tds::<f64, (), (), $dim>::new(&vertices).unwrap();

                let mut group = c.benchmark_group(&format!("validation_components_{}d", $dim));

                group.bench_function("validate_vertex_mappings", |b| {
                    b.iter(|| {
                        tds.validate_vertex_mappings().unwrap();
                        black_box(&tds);
                    });
                });

                group.bench_function("validate_cell_mappings", |b| {
                    b.iter(|| {
                        tds.validate_cell_mappings().unwrap();
                        black_box(&tds);
                    });
                });

                // Note: validate_no_duplicate_cells and validate_facet_sharing are private methods
                // They are tested indirectly through the full is_valid() benchmark

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

// Legacy wrappers for backward compatibility
fn benchmark_validation_methods(c: &mut Criterion) {
    benchmark_validation_methods_3d(c);
}

fn benchmark_validation_components(c: &mut Criterion) {
    benchmark_validation_components_3d(c);
}

/// Macro to generate incremental construction benchmarks for all dimensions
macro_rules! generate_incremental_construction_benchmarks {
    ($dim:literal) => {
        pastey::paste! {
            /// Benchmark incremental vertex addition for [<$dim>]D
            fn [<benchmark_incremental_construction_ $dim d>](c: &mut Criterion) {
                let seed = get_benchmark_seed(); // Cache seed to avoid repeated env var parsing
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
                let additional_vertex = vertex!(additional_array);

                group.bench_function("single_vertex_addition", |b| {
                    b.iter_batched(
                        || Tds::<f64, (), (), $dim>::new(&initial_vertices).unwrap(),
                        |mut tds| {
                            tds.add(additional_vertex).unwrap();
                            black_box(tds);
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
                                    let tds = Tds::<f64, (), (), $dim>::new(&initial_vertices).unwrap();
                                    let additional_points: Vec<Point<f64, $dim>> = generate_random_points_seeded(count, (-100.0, 100.0), seed).unwrap();
                                    let additional_vertices: Vec<_> =
                                        additional_points.iter().map(|p| vertex!(*p)).collect();
                                    (tds, additional_vertices)
                                },
                                |(mut tds, additional_vertices)| {
                                    for vertex in additional_vertices {
                                        tds.add(vertex).unwrap();
                                    }
                                    black_box(tds);
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

// Legacy wrapper for backward compatibility
fn benchmark_incremental_construction(c: &mut Criterion) {
    benchmark_incremental_construction_3d(c);
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        // Core triangulation benchmarks (2D-5D)
        benchmark_bowyer_watson_triangulation_2d,
        benchmark_bowyer_watson_triangulation_3d,
        benchmark_bowyer_watson_triangulation_4d,
        benchmark_bowyer_watson_triangulation_5d,
        benchmark_remove_duplicate_cells_2d,
        benchmark_remove_duplicate_cells_3d,
        benchmark_remove_duplicate_cells_4d,
        benchmark_remove_duplicate_cells_5d,

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
        benchmark_incremental_construction_5d,

        // Legacy wrappers for backward compatibility
        benchmark_bowyer_watson_triangulation,
        benchmark_remove_duplicate_cells,
        benchmark_2d_triangulation,
        benchmark_4d_triangulation,
        benchmark_5d_triangulation,
        benchmark_validation_methods,
        benchmark_validation_components,
        benchmark_incremental_construction,
        benchmark_memory_usage
);
criterion_main!(benches);
