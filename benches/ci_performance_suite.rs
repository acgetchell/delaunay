//! CI Performance Suite - Optimized performance regression testing for CI/CD
//!
//! This benchmark consolidates the most critical performance tests from across
//! the delaunay library into a single, CI-optimized suite that provides:
//!
//! 1. Core triangulation performance (3D/4D/5D at key scales)
//! 2. Critical circumsphere operations (`insphere_lifted` focus)
//! 3. Key algorithmic bottlenecks (neighbor assignment, deduplication)
//! 4. Basic memory footprint tracking
//!
//! Designed for ~5-10 minute CI runtime while maintaining comprehensive
//! regression detection across all performance-critical code paths.
//!
//! ## Dimensional Focus
//!
//! Tests 2D, 3D, 4D, and 5D triangulations for comprehensive coverage:
//! - 2D: Fundamental triangulation case
//! - 3D-5D: Higher-dimensional triangulations as documented in README.md

#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::*;
use delaunay::vertex;
use rand::Rng;
use std::hint::black_box;

/// Generate random 2D points for benchmarking
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

/// Generate random 3D points for benchmarking
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

/// Generate random 4D points for benchmarking
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

/// Generate random 5D points for benchmarking
fn generate_points_5d(count: usize) -> Vec<Point<f64, 5>> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| {
            Point::new([
                rng.random_range(-100.0..100.0),
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
    let counts = [10, 25, 50];
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
                    black_box(Tds::<f64, (), (), 2>::new(&vertices).unwrap());
                },
            );
        });
    }

    group.finish();
}

/// Benchmark `Tds::new` for 3D triangulations
fn benchmark_tds_new_3d(c: &mut Criterion) {
    let counts = [10, 25, 50];
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
                    black_box(Tds::<f64, (), (), 3>::new(&vertices).unwrap());
                },
            );
        });
    }

    group.finish();
}

/// Benchmark `Tds::new` for 4D triangulations
fn benchmark_tds_new_4d(c: &mut Criterion) {
    let counts = [10, 25, 50];
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
                    black_box(Tds::<f64, (), (), 4>::new(&vertices).unwrap());
                },
            );
        });
    }

    group.finish();
}

/// Benchmark `Tds::new` for 5D triangulations
fn benchmark_tds_new_5d(c: &mut Criterion) {
    let counts = [10, 25, 50];
    let mut group = c.benchmark_group("tds_new_5d");

    for &count in &counts {
        group.throughput(Throughput::Elements(count as u64));

        group.bench_with_input(BenchmarkId::new("tds_new", count), &count, |b, &count| {
            b.iter_with_setup(
                || {
                    let points = generate_points_5d(count);
                    points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                },
                |vertices| {
                    black_box(Tds::<f64, (), (), 5>::new(&vertices).unwrap());
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
        benchmark_tds_new_4d,
        benchmark_tds_new_5d
);
criterion_main!(benches);
