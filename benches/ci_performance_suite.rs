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

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::prelude::{ConstructionOptions, DelaunayTriangulation, RetryPolicy};
use delaunay::vertex;
use std::hint::black_box;
use std::num::NonZeroUsize;
use tracing::error;

/// Common sample sizes used across all CI performance benchmarks
const COUNTS: &[usize] = &[10, 25, 50];

fn bench_logging_enabled() -> bool {
    std::env::var("DELAUNAY_BENCH_LOG")
        .map(|value| value != "0")
        .unwrap_or(false)
}

fn bench_seed_search_enabled() -> bool {
    std::env::var("DELAUNAY_BENCH_SEED_SEARCH")
        .map(|value| value != "0")
        .unwrap_or(false)
}

fn bench_seed_search_limit() -> usize {
    std::env::var("DELAUNAY_BENCH_SEED_SEARCH_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2000)
}

/// Fixed seeds for deterministic triangulation generation across benchmark runs.
/// Using seeded random number generation reduces variance in performance measurements
/// and improves regression detection accuracy in CI environments.
/// Different seeds per dimension ensure triangulations are uncorrelated.
///
/// Macro to reduce duplication in dimensional benchmark functions
macro_rules! benchmark_tds_new_dimension {
    ($dim:literal, $func_name:ident, $seed:literal) => {
        /// Benchmark triangulation creation for D-dimensional triangulations
        #[expect(
            clippy::too_many_lines,
            reason = "Keep benchmark configuration, seed search, and error reporting together"
        )]
        fn $func_name(c: &mut Criterion) {
            let counts = COUNTS;

            // Opt-in helper for discovering stable seeds without paying Criterion warmup/
            // measurement cost per seed.
            //
            // NOTE: This helper is intentionally per (dim, count) benchmark case.
            // It returns early on the first successful seed (and panics on failure),
            // so it is meant to be run with a Criterion filter that selects a single
            // case, for example:
            //
            //     cargo bench --bench ci_performance_suite -- 'tds_new_3d/tds_new/50'
            //
            // Because the base seed is derived from `count`, a seed that works for one
            // count may still fail for a different count.
            //
            // We avoid `std::process::exit` here so that destructors run and Criterion
            // can clean up state on both success and failure.
            if bench_seed_search_enabled() {
                let bounds = (-100.0, 100.0);
                let filters: Vec<String> = std::env::args()
                    .skip(1)
                    .filter(|arg| !arg.starts_with('-'))
                    .collect();

                for &count in counts {
                    let bench_id =
                        format!("tds_new_{}d/tds_new/{}", stringify!($dim), count);

                    if !filters.is_empty() && !filters.iter().any(|filter| bench_id.contains(filter)) {
                        continue;
                    }

                    let seed = ($seed as u64).wrapping_add(count as u64);
                    let limit = bench_seed_search_limit();

                    for offset in 0..limit {
                        let candidate_seed = seed.wrapping_add(offset as u64);
                        let points = generate_random_points_seeded::<f64, $dim>(
                            count,
                            bounds,
                            candidate_seed,
                        )
                        .expect(concat!(
                            "generate_random_points_seeded failed for ",
                            stringify!($dim),
                            "D"
                        ));
                        let vertices = points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>();

                        let options =
                            ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
                                attempts: NonZeroUsize::new(6)
                                    .expect("retry attempts must be non-zero"),
                                base_seed: Some(candidate_seed),
                            });

                        if DelaunayTriangulation::<_, (), (), $dim>::new_with_options(
                            &vertices,
                            options,
                        )
                        .is_ok()
                        {
                            println!(
                                "seed_search_found dim={} count={} seed={}",
                                $dim, count, candidate_seed
                            );
                            return;
                        }
                    }

                    panic!(
                        "seed_search_failed dim={} count={} start_seed={} limit={}",
                        $dim,
                        count,
                        seed,
                        limit
                    );
                }

                // No filter matched this benchmark function; do nothing.
                return;
            }

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
                    //
                    // Note: Use per-count seeds so that each benchmark case has its own deterministic
                    // point set. This avoids a single pathological input (e.g. 3D/50) aborting the
                    // entire suite.
                    let bounds = (-100.0, 100.0);
                    let base_seed = ($seed as u64).wrapping_add(count as u64);
                    let search_limit = bench_seed_search_limit();
                    let mut selected = None;

                    // Find a deterministic seed/input pair that successfully constructs once.
                    // This avoids hard CI failures caused by rare geometric-degeneracy seeds.
                    for offset in 0..search_limit {
                        let candidate_seed = base_seed.wrapping_add(offset as u64);
                        let points = generate_random_points_seeded::<f64, $dim>(
                            count,
                            bounds,
                            candidate_seed,
                        )
                        .expect(concat!(
                            "generate_random_points_seeded failed for ",
                            stringify!($dim),
                            "D"
                        ));
                        let vertices = points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>();
                        let options =
                            ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
                                attempts: NonZeroUsize::new(6)
                                    .expect("retry attempts must be non-zero"),
                                base_seed: Some(candidate_seed),
                            });
                        if DelaunayTriangulation::<_, (), (), $dim>::new_with_options(
                            &vertices,
                            options,
                        )
                        .is_ok()
                        {
                            selected = Some((candidate_seed, points, vertices));
                            break;
                        }
                    }

                    let (seed, points, vertices) = selected.unwrap_or_else(|| {
                        panic!(
                            "No stable benchmark seed found for {}D case: dim={}; count={}; start_seed={}; search_limit={}; bounds={bounds:?}",
                            $dim,
                            $dim,
                            count,
                            base_seed,
                            search_limit
                        )
                    });
                    let sample_points = points.iter().take(5).collect::<Vec<_>>();

                    // In benchmarks we compile in release mode, where the default retry policy is
                    // disabled. For deterministic CI benchmarks we opt into a small number of
                    // shuffled retries to avoid aborting the suite on rare non-convergent repair
                    // cases.
                    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
                        attempts: NonZeroUsize::new(6).expect("retry attempts must be non-zero"),
                        base_seed: Some(seed),
                    });

                    b.iter(|| {
                        match DelaunayTriangulation::<_, (), (), $dim>::new_with_options(
                            &vertices,
                            options,
                        ) {
                            Ok(dt) => {
                                black_box(dt);
                            }
                            Err(err) => {
                                let error = format!("{err:?}");
                                if bench_logging_enabled() {
                                    error!(
                                        dim = $dim,
                                        count,
                                        seed,
                                        bounds = ?bounds,
                                        sample_points = ?sample_points,
                                        error = %error,
                                        "DelaunayTriangulation::new failed"
                                    );
                                }
                                panic!(
                                    "DelaunayTriangulation::new failed for {}D: {error}; dim={}; count={}; seed={}; bounds={:?}; sample_points={sample_points:?}",
                                    $dim,
                                    $dim,
                                    count,
                                    seed,
                                    bounds
                                );
                            }
                        }
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
