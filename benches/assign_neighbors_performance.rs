//! Performance benchmark for `assign_neighbors` method
//!
//! This benchmark measures the runtime of the `assign_neighbors` method before and after
//! optimizations to confirm reduced overhead on representative triangulations.

#![allow(missing_docs, unused_doc_comments, unused_attributes)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::*;
use delaunay::vertex;
use rand::Rng;
use std::hint::black_box;

mod helpers;
use helpers::clear_all_neighbors;

/// Creates random points for benchmarking
fn generate_random_points_3d(n_points: usize) -> Vec<Point<f64, 3>> {
    let mut rng = rand::rng();
    (0..n_points)
        .map(|_| {
            Point::new([
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
            ])
        })
        .collect()
}

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

/// Benchmark `assign_neighbors` with random point distributions
fn benchmark_assign_neighbors_random(c: &mut Criterion) {
    let point_counts = [10, 20, 30, 40, 50];

    let mut group = c.benchmark_group("assign_neighbors_random");

    for &n_points in &point_counts {
        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("random_points", n_points),
            &n_points,
            |b, &n_points| {
                b.iter_with_setup(
                    || {
                        let points = generate_random_points_3d(n_points);
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

/// Benchmark `assign_neighbors` with grid point distributions
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

/// Benchmark `assign_neighbors` with spherical point distributions
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

/// Benchmark the scaling behavior with different triangulation sizes
fn benchmark_assign_neighbors_scaling(c: &mut Criterion) {
    let point_counts = [8, 16, 24, 32];

    let mut group = c.benchmark_group("assign_neighbors_scaling");
    group.sample_size(20); // Reduce sample size for longer tests

    // Pre-compute and print scaling information outside of benchmark timing
    println!("\n=== Scaling Analysis Pre-computation ===");
    for &n_points in &point_counts {
        let points = generate_random_points_3d(n_points);
        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
        let tds = Tds::<f64, (), (), 3>::new(&vertices).unwrap();

        let num_cells = tds.number_of_cells();
        let num_vertices = tds.number_of_vertices();

        // Calculate ratio with explicit precision handling
        // For benchmark display purposes, f64 precision is sufficient
        // Alternative approaches shown in comments below

        // Approach 1: Simple with clippy allow (most practical)
        #[allow(clippy::cast_precision_loss)]
        let ratio = num_cells as f64 / n_points as f64;

        // Approach 2: Check for precision loss (more defensive)
        // let ratio = if num_cells < (1_u64 << 53) && n_points < (1_u64 << 53) {
        //     num_cells as f64 / n_points as f64
        // } else {
        //     // For very large numbers, use integer arithmetic with rounding
        //     (num_cells * 100 / n_points) as f64 / 100.0
        // };

        // Approach 3: Using ordered-float for guaranteed precision (overkill for this case)
        // use ordered_float::OrderedFloat;
        // let ratio_of = OrderedFloat(num_cells as f64) / OrderedFloat(n_points as f64);
        // let ratio = ratio_of.into_inner();

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
                        let points = generate_random_points_3d(n_points);
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

/// Compare `assign_neighbors` performance across different dimensions
fn benchmark_assign_neighbors_2d_vs_3d(c: &mut Criterion) {
    let point_counts = [10, 20, 30];

    let mut group = c.benchmark_group("assign_neighbors_2d_vs_3d");

    // 2D benchmarks
    for &n_points in &point_counts {
        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("2d", n_points),
            &n_points,
            |b, &n_points| {
                b.iter_with_setup(
                    || {
                        let mut rng = rand::rng();
                        let points: Vec<Point<f64, 2>> = (0..n_points)
                            .map(|_| {
                                Point::new([
                                    rng.random_range(-100.0..100.0),
                                    rng.random_range(-100.0..100.0),
                                ])
                            })
                            .collect();
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
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        benchmark_assign_neighbors_random,
        benchmark_assign_neighbors_grid,
        benchmark_assign_neighbors_spherical,
        benchmark_assign_neighbors_scaling,
        benchmark_assign_neighbors_2d_vs_3d
);
criterion_main!(benches);
