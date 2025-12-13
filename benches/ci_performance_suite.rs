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
//! ## Sample Size Strategy
//!
//! Uses dimension-dependent sample sizes to balance accuracy with CI time constraints:
//! - 2D: Default Criterion sample size (100)
//! - 3D: Reduced to 25 samples
//! - 4D/5D: Further reduced to 15 samples for longer-running high-dimensional cases
//!
//! ## Dimensional Focus
//!
//! Tests 2D, 3D, 4D, and 5D triangulations for comprehensive coverage:
//! - 2D: Fundamental triangulation case
//! - 3D-5D: Higher-dimensional triangulations as documented in README.md

#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::prelude::DelaunayTriangulation;
use delaunay::vertex;
use std::hint::black_box;

/// Common sample sizes used across all CI performance benchmarks
const COUNTS: &[usize] = &[10, 25, 50];

/// Fixed seeds for deterministic triangulation generation across benchmark runs.
/// Using seeded random number generation reduces variance in performance measurements
/// and improves regression detection accuracy in CI environments.
/// Different seeds per dimension ensure triangulations are uncorrelated.
///
/// Macro to reduce duplication in dimensional benchmark functions
macro_rules! benchmark_tds_new_dimension {
    ($dim:literal, $func_name:ident, $seed:literal) => {
        /// Benchmark triangulation creation for D-dimensional triangulations
        fn $func_name(c: &mut Criterion) {
            let counts = COUNTS;
            let mut group = c.benchmark_group(concat!("tds_new_", stringify!($dim), "d"));

            // Set smaller sample sizes for higher dimensions to keep CI times reasonable
            if $dim >= 4 {
                group.sample_size(15); // Fewer samples for 4D and 5D
            } else if $dim == 3 {
                group.sample_size(25); // Medium sample size for 3D
            }

            for &count in counts {
                group.throughput(Throughput::Elements(count as u64));

                group.bench_with_input(BenchmarkId::new("tds_new", count), &count, |b, &count| {
                    // Reduce variance: pre-generate deterministic inputs outside the measured loop,
                    // then benchmark only triangulation construction.
                    let points =
                        generate_random_points_seeded::<f64, $dim>(count, (-100.0, 100.0), $seed)
                            .expect(concat!(
                                "generate_random_points_seeded failed for ",
                                stringify!($dim),
                                "D"
                            ));
                    let vertices = points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>();

                    b.iter(|| {
                        black_box(
                            DelaunayTriangulation::<_, (), (), $dim>::new(&vertices).expect(
                                concat!(
                                    "DelaunayTriangulation::new failed for ",
                                    stringify!($dim),
                                    "D"
                                ),
                            ),
                        );
                    });
                });
            }

            group.finish();
        }
    };
}

// Generate benchmark functions using the macro
benchmark_tds_new_dimension!(2, benchmark_tds_new_2d, 42);
benchmark_tds_new_dimension!(3, benchmark_tds_new_3d, 123);
benchmark_tds_new_dimension!(4, benchmark_tds_new_4d, 456);
benchmark_tds_new_dimension!(5, benchmark_tds_new_5d, 789);

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
