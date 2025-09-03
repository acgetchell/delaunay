//! Performance benchmark for `assign_neighbors` method
//!
//! This benchmark measures the runtime of the `assign_neighbors` method before and after
//! optimizations to confirm reduced overhead on representative triangulations.

#![allow(missing_docs, unused_doc_comments, unused_attributes)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::*;
use delaunay::vertex;
use pastey::paste;
use rand::Rng;
use std::hint::black_box;

mod util;
use util::{
    clear_all_neighbors, generate_random_points_2d, generate_random_points_3d,
    generate_random_points_4d, generate_random_points_5d,
};

/// Creates a regular grid of points for consistent benchmarking
fn generate_grid_points_3d(n_side: usize) -> Vec<Point<f64, 3>> {
    let mut points = Vec::new();
    let spacing = 1.0;

    for i in 0..n_side {
        for j in 0..n_side {
            for k in 0..n_side {
                #[allow(clippy::cast_precision_loss)]
                let point =
                    Point::new([i as f64 * spacing, j as f64 * spacing, k as f64 * spacing]);
                points.push(point);
            }
        }
    }
    points
}

/// Creates points in a spherical distribution
fn generate_spherical_points_3d(n_points: usize) -> Vec<Point<f64, 3>> {
    let mut rng = rand::rng();
    (0..n_points)
        .map(|_| {
            let theta = rng.random_range(0.0..std::f64::consts::TAU);
            let phi = rng.random_range(0.0..std::f64::consts::PI);
            let r = rng.random_range(10.0..50.0);

            let x = r * phi.sin() * theta.cos();
            let y = r * phi.sin() * theta.sin();
            let z = r * phi.cos();

            Point::new([x, y, z])
        })
        .collect()
}

/// Macro to generate `assign_neighbors` benchmarks for all dimensions
macro_rules! generate_assign_neighbors_benchmarks {
    ($($dim:literal),* $(,)?) => {
        $(
            paste! {
                /// Benchmark `assign_neighbors` with random point distributions for [<$dim>]D
                fn [<benchmark_assign_neighbors_ $dim d_random>](c: &mut Criterion) {
                    let point_counts = [10, 20, 30, 40, 50];

                    let mut group = c.benchmark_group(&format!("assign_neighbors_{}d_random", $dim));

                    for &n_points in &point_counts {
                        group.throughput(Throughput::Elements(n_points as u64));

                        group.bench_with_input(
                            BenchmarkId::new("random_points", n_points),
                            &n_points,
                            |b, &n_points| {
                                b.iter_with_setup(
                                    || {
                                        let points = [<generate_random_points_ $dim d>](n_points);
                                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                                        let mut tds = Tds::<f64, (), (), $dim>::new(&vertices).unwrap();

                                        // Clear existing neighbors to benchmark the assignment process
                                        clear_all_neighbors(&mut tds);
                                        tds
                                    },
                                    |mut tds| {
                                        tds.assign_neighbors().unwrap();
                                        black_box(tds);
                                    },
                                );
                            },
                        );
                    }

                    group.finish();
                }

                /// Benchmark `assign_neighbors` scaling with different triangulation sizes for [<$dim>]D
                fn [<benchmark_assign_neighbors_ $dim d_scaling>](c: &mut Criterion) {
                    let point_counts = [8, 16, 24, 32];

                    let mut group = c.benchmark_group(&format!("assign_neighbors_{}d_scaling", $dim));
                    group.sample_size(20); // Reduce sample size for longer tests

                    // Pre-compute and print scaling information outside of benchmark timing
                    println!("\n=== {}D Scaling Analysis Pre-computation ===", $dim);
                    for &n_points in &point_counts {
                        let points = [<generate_random_points_ $dim d>](n_points);
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        let tds = Tds::<f64, (), (), $dim>::new(&vertices).unwrap();

                        let num_cells = tds.number_of_cells();
                        let num_vertices = tds.number_of_vertices();

                        #[allow(clippy::cast_precision_loss)]
                        let ratio = num_cells as f64 / n_points as f64;

                        println!(
                            "Points: {n_points}, Cells: {num_cells}, Vertices: {num_vertices} (ratio: {ratio:.2} cells/point)"
                        );
                    }
                    println!("==========================================\n");

                    for &n_points in &point_counts {
                        group.throughput(Throughput::Elements(n_points as u64));

                        group.bench_with_input(
                            BenchmarkId::new("scaling", n_points),
                            &n_points,
                            |b, &n_points| {
                                b.iter_with_setup(
                                    || {
                                        let points = [<generate_random_points_ $dim d>](n_points);
                                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                                        let mut tds = Tds::<f64, (), (), $dim>::new(&vertices).unwrap();

                                        // Clear existing neighbors to benchmark the assignment process
                                        clear_all_neighbors(&mut tds);
                                        tds
                                    },
                                    |mut tds| {
                                        tds.assign_neighbors().unwrap();
                                        black_box(tds);
                                    },
                                );
                            },
                        );
                    }

                    group.finish();
                }
            }
        )*
    };
}

// Generate benchmarks for dimensions 2, 3, 4, 5
generate_assign_neighbors_benchmarks!(2, 3, 4, 5);

// Legacy wrapper for backward compatibility
fn benchmark_assign_neighbors_random(c: &mut Criterion) {
    benchmark_assign_neighbors_3d_random(c);
}

/// Benchmark `assign_neighbors` with grid point distributions (3D only)
fn benchmark_assign_neighbors_grid(c: &mut Criterion) {
    let grid_sizes = [2, 3, 4]; // 2^3=8, 3^3=27, 4^3=64 points

    let mut group = c.benchmark_group("assign_neighbors_grid");

    for &grid_size in &grid_sizes {
        let n_points = grid_size * grid_size * grid_size;
        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("grid_points", n_points),
            &grid_size,
            |b, &grid_size| {
                b.iter_with_setup(
                    || {
                        let points = generate_grid_points_3d(grid_size);
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        let mut tds = Tds::<f64, (), (), 3>::new(&vertices).unwrap();

                        // Clear existing neighbors to benchmark the assignment process
                        clear_all_neighbors(&mut tds);
                        tds
                    },
                    |mut tds| {
                        tds.assign_neighbors().unwrap();
                        black_box(tds);
                    },
                );
            },
        );
    }

    group.finish();
}

/// Benchmark `assign_neighbors` with spherical point distributions (3D only)
fn benchmark_assign_neighbors_spherical(c: &mut Criterion) {
    let point_counts = [15, 25, 35, 45];

    let mut group = c.benchmark_group("assign_neighbors_spherical");

    for &n_points in &point_counts {
        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("spherical_points", n_points),
            &n_points,
            |b, &n_points| {
                b.iter_with_setup(
                    || {
                        let points = generate_spherical_points_3d(n_points);
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        let mut tds = Tds::<f64, (), (), 3>::new(&vertices).unwrap();

                        // Clear existing neighbors to benchmark the assignment process
                        clear_all_neighbors(&mut tds);
                        tds
                    },
                    |mut tds| {
                        tds.assign_neighbors().unwrap();
                        black_box(tds);
                    },
                );
            },
        );
    }

    group.finish();
}

// Legacy wrapper for backward compatibility
fn benchmark_assign_neighbors_scaling(c: &mut Criterion) {
    benchmark_assign_neighbors_3d_scaling(c);
}

/// Compare `assign_neighbors` performance across dimensions 2D through 5D
fn benchmark_assign_neighbors_multi_dimensional(c: &mut Criterion) {
    let point_counts = [10, 20, 30];

    let mut group = c.benchmark_group("assign_neighbors_multi_dimensional");

    for &n_points in &point_counts {
        group.throughput(Throughput::Elements(n_points as u64));

        // 2D benchmarks
        group.bench_with_input(
            BenchmarkId::new("2d", n_points),
            &n_points,
            |b, &n_points| {
                b.iter_with_setup(
                    || {
                        let points = generate_random_points_2d(n_points);
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        let mut tds = Tds::<f64, (), (), 2>::new(&vertices).unwrap();

                        // Clear existing neighbors
                        clear_all_neighbors(&mut tds);
                        tds
                    },
                    |mut tds| {
                        tds.assign_neighbors().unwrap();
                        black_box(tds);
                    },
                );
            },
        );

        // 3D benchmarks
        group.bench_with_input(
            BenchmarkId::new("3d", n_points),
            &n_points,
            |b, &n_points| {
                b.iter_with_setup(
                    || {
                        let points = generate_random_points_3d(n_points);
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        let mut tds = Tds::<f64, (), (), 3>::new(&vertices).unwrap();

                        // Clear existing neighbors
                        clear_all_neighbors(&mut tds);
                        tds
                    },
                    |mut tds| {
                        tds.assign_neighbors().unwrap();
                        black_box(tds);
                    },
                );
            },
        );

        // 4D benchmarks
        group.bench_with_input(
            BenchmarkId::new("4d", n_points),
            &n_points,
            |b, &n_points| {
                b.iter_with_setup(
                    || {
                        let points = generate_random_points_4d(n_points);
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        let mut tds = Tds::<f64, (), (), 4>::new(&vertices).unwrap();

                        // Clear existing neighbors
                        clear_all_neighbors(&mut tds);
                        tds
                    },
                    |mut tds| {
                        tds.assign_neighbors().unwrap();
                        black_box(tds);
                    },
                );
            },
        );

        // 5D benchmarks
        group.bench_with_input(
            BenchmarkId::new("5d", n_points),
            &n_points,
            |b, &n_points| {
                b.iter_with_setup(
                    || {
                        let points = generate_random_points_5d(n_points);
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        let mut tds = Tds::<f64, (), (), 5>::new(&vertices).unwrap();

                        // Clear existing neighbors
                        clear_all_neighbors(&mut tds);
                        tds
                    },
                    |mut tds| {
                        tds.assign_neighbors().unwrap();
                        black_box(tds);
                    },
                );
            },
        );
    }

    group.finish();
}

// Legacy wrapper for backward compatibility
fn benchmark_assign_neighbors_2d_vs_3d(c: &mut Criterion) {
    benchmark_assign_neighbors_multi_dimensional(c);
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        // Legacy wrappers
        benchmark_assign_neighbors_random,
        benchmark_assign_neighbors_grid,
        benchmark_assign_neighbors_spherical,
        benchmark_assign_neighbors_scaling,
        benchmark_assign_neighbors_2d_vs_3d,
        // New comprehensive dimension-specific benchmarks
        benchmark_assign_neighbors_2d_random,
        benchmark_assign_neighbors_3d_random,
        benchmark_assign_neighbors_4d_random,
        benchmark_assign_neighbors_5d_random,
        benchmark_assign_neighbors_2d_scaling,
        benchmark_assign_neighbors_3d_scaling,
        benchmark_assign_neighbors_4d_scaling,
        benchmark_assign_neighbors_5d_scaling,
        benchmark_assign_neighbors_multi_dimensional
);
criterion_main!(benches);
