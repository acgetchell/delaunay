//! Small-scale benchmarks for d-dimensional Delaunay triangulation operations.
//!
//! This benchmark measures the performance of core triangulation operations
//! across different dimensions (2D, 3D, 4D) and small point counts (10-50 points).
//! Memory allocations are tracked at each iteration with detailed logging and bounds checking.

#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::*;
use delaunay::vertex;
use rand::Rng;
use std::hint::black_box;

/// Generate random points for a given dimension and count
fn generate_points_2d(count: usize) -> Vec<Point<f64, 2>> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| {
            Point::new([
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
            ])
        })
        .collect()
}

fn generate_points_3d(count: usize) -> Vec<Point<f64, 3>> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| {
            Point::new([
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
            ])
        })
        .collect()
}

fn generate_points_4d(count: usize) -> Vec<Point<f64, 4>> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| {
            Point::new([
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
            ])
        })
        .collect()
}

/// Benchmark `Tds::new` for 2D triangulations
fn benchmark_tds_new_2d(c: &mut Criterion) {
    let counts = [10, 20, 30, 40, 50];
    let mut group = c.benchmark_group("tds_new_2d");

    for &count in &counts {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::new("tds_new", count), &count, |b, &count| {
            b.iter_with_setup(
                || {
                    let points = generate_points_2d(count);
                    points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                },
                |vertices| {
                    #[cfg(feature = "count-allocations")]
                    {
                        let result = allocation_counter::measure(|| {
                            let tds = Tds::<f64, (), (), 2>::new(&vertices).unwrap();
                            black_box(tds);
                        });
                        println!(
                            "TDS 2D creation - Points: {}, Allocation info: {result:?}",
                            vertices.len()
                        );
                    }
                    #[cfg(not(feature = "count-allocations"))]
                    {
                        black_box(Tds::<f64, (), (), 2>::new(&vertices).unwrap());
                    }
                },
            );
        });
    }

    group.finish();
}

/// Benchmark `Tds::new` for 3D triangulations
fn benchmark_tds_new_3d(c: &mut Criterion) {
    let counts = [10, 20, 30, 40, 50];
    let mut group = c.benchmark_group("tds_new_3d");

    for &count in &counts {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::new("tds_new", count), &count, |b, &count| {
            b.iter_with_setup(
                || {
                    let points = generate_points_3d(count);
                    points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                },
                |vertices| {
                    #[cfg(feature = "count-allocations")]
                    {
                        let result = allocation_counter::measure(|| {
                            let tds = Tds::<f64, (), (), 3>::new(&vertices).unwrap();
                            black_box(tds);
                        });
                        println!(
                            "TDS 3D creation - Points: {}, Allocation info: {result:?}",
                            vertices.len()
                        );
                    }
                    #[cfg(not(feature = "count-allocations"))]
                    {
                        black_box(Tds::<f64, (), (), 3>::new(&vertices).unwrap());
                    }
                },
            );
        });
    }

    group.finish();
}

/// Benchmark `Tds::new` for 4D triangulations
fn benchmark_tds_new_4d(c: &mut Criterion) {
    let counts = [10, 20, 30, 40, 50];
    let mut group = c.benchmark_group("tds_new_4d");

    for &count in &counts {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::new("tds_new", count), &count, |b, &count| {
            b.iter_with_setup(
                || {
                    let points = generate_points_4d(count);
                    points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                },
                |vertices| {
                    #[cfg(feature = "count-allocations")]
                    {
                        let result = allocation_counter::measure(|| {
                            let tds = Tds::<f64, (), (), 4>::new(&vertices).unwrap();
                            black_box(tds);
                        });
                        println!(
                            "TDS 4D creation - Points: {}, Allocation info: {result:?}",
                            vertices.len()
                        );
                    }
                    #[cfg(not(feature = "count-allocations"))]
                    {
                        black_box(Tds::<f64, (), (), 4>::new(&vertices).unwrap());
                    }
                },
            );
        });
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        benchmark_tds_new_2d,
        benchmark_tds_new_3d,
        benchmark_tds_new_4d
);
criterion_main!(benches);
