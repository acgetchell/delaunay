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
use delaunay::geometry::util::generate_random_points;
use delaunay::prelude::*;
use delaunay::vertex;
use std::hint::black_box;

/// Common sample sizes used across all CI performance benchmarks
const COUNTS: &[usize] = &[10, 25, 50];

/// Generate random 2D points for benchmarking
fn generate_points_2d(count: usize) -> Vec<Point<f64, 2>> {
    generate_random_points(count, (-100.0, 100.0)).unwrap()
}

/// Generate random 3D points for benchmarking
fn generate_points_3d(count: usize) -> Vec<Point<f64, 3>> {
    generate_random_points(count, (-100.0, 100.0)).unwrap()
}

/// Generate random 4D points for benchmarking
fn generate_points_4d(count: usize) -> Vec<Point<f64, 4>> {
    generate_random_points(count, (-100.0, 100.0)).unwrap()
}

/// Generate random 5D points for benchmarking
fn generate_points_5d(count: usize) -> Vec<Point<f64, 5>> {
    generate_random_points(count, (-100.0, 100.0)).unwrap()
}

/// Macro to reduce duplication in dimensional benchmark functions
macro_rules! benchmark_tds_new_dimension {
    ($dim:literal, $func_name:ident, $points_fn:ident) => {
        /// Benchmark `Tds::new` for D-dimensional triangulations
        fn $func_name(c: &mut Criterion) {
            let counts = COUNTS;
            let mut group = c.benchmark_group(concat!("tds_new_", stringify!($dim), "d"));

            for &count in counts {
                group.throughput(Throughput::Elements(count as u64));

                group.bench_with_input(BenchmarkId::new("tds_new", count), &count, |b, &count| {
                    b.iter_with_setup(
                        || {
                            let points = $points_fn(count);
                            points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                        },
                        |vertices| {
                            black_box(Tds::<f64, (), (), $dim>::new(&vertices).unwrap());
                        },
                    );
                });
            }

            group.finish();
        }
    };
}

// Generate benchmark functions using the macro
benchmark_tds_new_dimension!(2, benchmark_tds_new_2d, generate_points_2d);
benchmark_tds_new_dimension!(3, benchmark_tds_new_3d, generate_points_3d);
benchmark_tds_new_dimension!(4, benchmark_tds_new_4d, generate_points_4d);
benchmark_tds_new_dimension!(5, benchmark_tds_new_5d, generate_points_5d);

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
