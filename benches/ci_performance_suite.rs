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
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::prelude::{ConstructionOptions, DelaunayTriangulation, RetryPolicy};
use delaunay::vertex;
use std::hint::black_box;
use std::num::NonZeroUsize;
use tracing::{error, warn};

/// Default point counts for 2D–4D benchmarks.
const COUNTS: &[usize] = &[10, 25, 50];
/// Reduced point counts for 5D (50-point construction is prohibitively slow).
const COUNTS_5D: &[usize] = &[10, 25];
type SeedSearchResult<const D: usize> = Option<(u64, Vec<Point<f64, D>>, Vec<Vertex<f64, (), D>>)>;

/// Pre-computed seeds for each (dimension, count) pair.
///
/// These were discovered using `DELAUNAY_BENCH_DISCOVER_SEEDS=1` and eliminate
/// the expensive runtime seed search from the benchmark hot path.
///
/// To refresh seeds (e.g. after algorithm changes that invalidate them), run:
///
/// ```bash
/// DELAUNAY_BENCH_DISCOVER_SEEDS=1 cargo bench --bench ci_performance_suite
/// ```
///
/// (Use a Criterion filter for individual cases, e.g.
///  `-- "tds_new_5d/tds_new/25"`)
const KNOWN_SEEDS: &[(usize, usize, u64)] = &[
    // 2D
    (2, 10, 52),
    (2, 25, 67),
    (2, 50, 92),
    // 3D
    (3, 10, 133),
    (3, 25, 148),
    (3, 50, 173),
    // 4D
    (4, 10, 466),
    (4, 25, 481),
    (4, 50, 506),
    // 5D (50-point case excluded — too slow for CI)
    (5, 10, 799),
    (5, 25, 816),
];

fn known_seed(dim: usize, count: usize) -> Option<u64> {
    KNOWN_SEEDS
        .iter()
        .find(|&&(d, c, _)| d == dim && c == count)
        .map(|&(_, _, seed)| seed)
}

/// Prepare benchmark inputs by looking up a pre-computed seed, falling back
/// to a runtime search only if the known seed is missing or invalid.
fn prepare_benchmark_data<const D: usize>(
    dim_seed: u64,
    count: usize,
    bounds: (f64, f64),
    attempts: NonZeroUsize,
) -> (u64, Vec<Point<f64, D>>, Vec<Vertex<f64, (), D>>) {
    // Fast path: use the pre-computed seed (single verification construction)
    if let Some(seed) = known_seed(D, count) {
        if let Some(result) = find_seed_and_vertices::<D>(seed, count, bounds, 1, attempts) {
            return result;
        }
        warn!(
            known_seed = seed,
            dim = D,
            count,
            "known seed failed, falling back to runtime search"
        );
    }

    // Slow fallback: runtime search from the base seed
    let base_seed = dim_seed.wrapping_add(count as u64);
    let search_limit = bench_seed_search_limit();
    find_seed_and_vertices::<D>(base_seed, count, bounds, search_limit, attempts).unwrap_or_else(
        || {
            panic!(
                "No stable benchmark seed found for {D}D/{count}: \
                 start_seed={base_seed}; search_limit={search_limit}; bounds={bounds:?}"
            )
        },
    )
}

fn find_seed_and_vertices<const D: usize>(
    start_seed: u64,
    count: usize,
    bounds: (f64, f64),
    limit: usize,
    attempts: NonZeroUsize,
) -> SeedSearchResult<D> {
    for offset in 0..limit {
        let candidate_seed = start_seed.wrapping_add(offset as u64);
        let points = generate_random_points_seeded::<f64, D>(count, bounds, candidate_seed)
            .unwrap_or_else(|error| {
                panic!("generate_random_points_seeded failed for {D}D: {error}");
            });
        let vertices = points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>();

        let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
            attempts,
            base_seed: Some(candidate_seed),
        });

        if DelaunayTriangulation::<_, (), (), D>::new_with_options(&vertices, options).is_ok() {
            return Some((candidate_seed, points, vertices));
        }
    }

    None
}

fn bench_logging_enabled() -> bool {
    std::env::var("DELAUNAY_BENCH_LOG").is_ok_and(|value| value != "0")
}

fn bench_discover_seeds_enabled() -> bool {
    std::env::var("DELAUNAY_BENCH_DISCOVER_SEEDS").is_ok_and(|value| value != "0")
}

fn bench_seed_search_limit() -> usize {
    std::env::var("DELAUNAY_BENCH_DISCOVER_SEEDS_LIMIT")
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
    ($dim:literal, $func_name:ident, $seed:literal, $counts:expr) => {
        /// Benchmark triangulation creation for D-dimensional triangulations
        fn $func_name(c: &mut Criterion) {
            let counts = $counts;

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
            if bench_discover_seeds_enabled() {
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
                    let attempts =
                        NonZeroUsize::new(6).expect("retry attempts must be non-zero");

                    if let Some((candidate_seed, _, _)) =
                        find_seed_and_vertices::<$dim>(seed, count, bounds, limit, attempts)
                    {
                        println!(
                            "seed_search_found dim={} count={} seed={}",
                            $dim, count, candidate_seed
                        );
                        return;
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
                    let bounds = (-100.0, 100.0);
                    let attempts =
                        NonZeroUsize::new(6).expect("retry attempts must be non-zero");
                    let (seed, points, vertices) =
                        prepare_benchmark_data::<$dim>($seed, count, bounds, attempts);
                    let sample_points = points.iter().take(5).collect::<Vec<_>>();

                    // In benchmarks we compile in release mode, where the default retry policy is
                    // disabled. For deterministic CI benchmarks we opt into a small number of
                    // shuffled retries to avoid aborting the suite on rare non-convergent repair
                    // cases.
                    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
                        attempts,
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
benchmark_tds_new_dimension!(2, benchmark_tds_new_2d, 42, COUNTS);
benchmark_tds_new_dimension!(3, benchmark_tds_new_3d, 123, COUNTS);
benchmark_tds_new_dimension!(4, benchmark_tds_new_4d, 456, COUNTS);
benchmark_tds_new_dimension!(5, benchmark_tds_new_5d, 789, COUNTS_5D);

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
