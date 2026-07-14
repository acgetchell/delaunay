#![forbid(unsafe_code)]

//! Microbenchmark for `core::hint::cold_path` adoption in geometric predicates.
//!
//! This benchmark exercises the hot Stage-1 path of the [`insphere`] /
//! [`insphere_lifted`] predicates (and implicitly the orientation predicate they
//! both invoke) in 2D–4D, plus a Stage-2-dominant 5D reference. A secondary
//! centered-query group retains its historical `near_boundary` benchmark
//! identifier for saved-baseline compatibility, but it does not independently
//! establish Stage-2 entry. The `exact_fallback` group uses known cospherical
//! points and validates their boundary classification before timing, so it is
//! the correctness-certified Stage-2 signal for 2D-4D.
//!
//! ## Stage anatomy
//!
//! Both `insphere_from_matrix` and `orientation_from_matrix` are structured as:
//!
//! 1. Stage 1 (hot): f64 fast filter with Shewchuk-style errbound.
//! 2. Stage 2 (cold): exact sign via Bareiss — only reached for ambiguous f64
//!    results or D ≥ 5.
//! 3. Stage 3 (very cold): non-finite fallback.
//!
//! Random, well-separated inputs hit Stage 1 almost exclusively. The
//! `near_boundary` group samples a small cube around the standard simplex's
//! circumsphere center. It measures a distinct centered-query workload, not a
//! proven Stage-2 workload.
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --profile perf --bench cold_path_predicates
//! ```
//!
//! To save a baseline before applying `cold_path()` and compare afterwards:
//!
//! ```bash
//! cargo bench --profile perf --bench cold_path_predicates -- --save-baseline pre
//! # (edit src/geometry/predicates.rs)
//! cargo bench --profile perf --bench cold_path_predicates -- --baseline pre
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{CoordinateRange, InSphere};
use delaunay::prelude::query::*;
use std::hint::black_box;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, abort_benchmark};

fn finite_point<const D: usize>(coords: [f64; D]) -> Point<D> {
    Point::try_new(coords).unwrap_or_else(|_| std::process::abort())
}

fn coordinate_range(min: f64, max: f64) -> CoordinateRange<f64> {
    CoordinateRange::try_new(min, max).or_abort()
}

/// Deterministic seed for query-point generation in the hot path.
const HOT_SEED: u64 = 0xC01D_BEEF_0000_CAFE_u64;
/// Deterministic seed for query-point generation in the centered-query group.
const CENTERED_QUERY_SEED: u64 = 0x0000_0000_0000_BEEF_u64;
/// Number of queries per hot-path benchmark.  Keep above the size where
/// per-iteration overhead dominates so Stage-1 improvements are visible.
const HOT_QUERIES: usize = 10_000;
/// Number of queries per centered-query benchmark.
const CENTERED_QUERIES: usize = 1_000;

/// Standard D-dimensional simplex: origin + unit basis vectors.
fn standard_simplex<const D: usize>() -> Vec<Point<D>> {
    let mut pts = Vec::with_capacity(D + 1);
    pts.push(finite_point([0.0; D]));
    for i in 0..D {
        let mut coords = [0.0; D];
        coords[i] = 1.0;
        pts.push(finite_point(coords));
    }
    pts
}

/// Generate well-separated hot-path query points for dimension `D`.
///
/// Uses the range `[-10, 10]` against a unit simplex so that the Shewchuk
/// errbound comfortably resolves the sign in Stage 1.
fn hot_queries<const D: usize>() -> Vec<Point<D>> {
    generate_random_points_in_range_seeded(HOT_QUERIES, coordinate_range(-10.0, 10.0), HOT_SEED)
        .or_abort()
}

/// Generate centered query points for dimension `D`.
///
/// Uses a narrow range around the standard simplex's circumsphere center. For
/// the standard simplex this cube lies well inside the circumsphere, so this
/// fixture must not be used as evidence that Stage 2 is exercised.
fn centered_queries<const D: usize>() -> Vec<Point<D>> {
    // Centered on the circumsphere center of the standard simplex. Preserve the
    // existing distribution so saved performance baselines remain comparable.
    generate_random_points_in_range_seeded(
        CENTERED_QUERIES,
        coordinate_range(0.40, 0.60),
        CENTERED_QUERY_SEED,
    )
    .or_abort()
}

/// Run `insphere` across `queries` against `simplex`, black-boxing each result.
fn run_insphere<const D: usize>(simplex: &[Point<D>], queries: &[Point<D>]) {
    for q in queries {
        let result = match insphere(black_box(simplex), black_box(*q)) {
            Ok(value) => value,
            Err(error) => abort_benchmark(format_args!("insphere query failed: {error}")),
        };
        black_box(result);
    }
}

/// Run `insphere_lifted` across `queries` against `simplex`, black-boxing each result.
fn run_insphere_lifted<const D: usize>(simplex: &[Point<D>], queries: &[Point<D>]) {
    for q in queries {
        let result = match insphere_lifted(black_box(simplex), black_box(*q)) {
            Ok(value) => value,
            Err(error) => abort_benchmark(format_args!("insphere_lifted query failed: {error}")),
        };
        black_box(result);
    }
}

/// Benchmark a known cospherical query that forces exact sign resolution.
fn bench_exact_fallback_case<const D: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    let simplex = standard_simplex::<D>();
    let boundary = finite_point([1.0; D]);

    let insphere_result = insphere(&simplex, boundary).or_abort();
    let lifted_result = insphere_lifted(&simplex, boundary).or_abort();
    if insphere_result != InSphere::BOUNDARY || lifted_result != InSphere::BOUNDARY {
        abort_benchmark(format_args!(
            "{D}D exact-fallback fixture must be cospherical: insphere={insphere_result}, lifted={lifted_result}"
        ));
    }

    group.bench_function(format!("insphere_{D}d"), |b| {
        b.iter(|| {
            let result = insphere(black_box(&simplex), black_box(boundary)).or_abort();
            black_box(result)
        });
    });
    group.bench_function(format!("insphere_lifted_{D}d"), |b| {
        b.iter(|| {
            let result = insphere_lifted(black_box(&simplex), black_box(boundary)).or_abort();
            black_box(result)
        });
    });
}

/// Benchmark correctness-certified Stage-2 exact fallback in 2D-4D.
fn bench_exact_fallback(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicates/exact_fallback");
    bench_exact_fallback_case::<2>(&mut group);
    bench_exact_fallback_case::<3>(&mut group);
    bench_exact_fallback_case::<4>(&mut group);
    group.finish();
}

/// Benchmark the Stage-1 hot path for `insphere` and `insphere_lifted`.
fn bench_hot_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicates/hot");
    group.throughput(Throughput::Elements(HOT_QUERIES as u64));

    // 2D
    {
        let simplex = standard_simplex::<2>();
        let queries = hot_queries::<2>();
        group.bench_with_input(
            BenchmarkId::new("insphere_2d", HOT_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere(black_box(&simplex), black_box(&queries)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("insphere_lifted_2d", HOT_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere_lifted(black_box(&simplex), black_box(&queries)));
            },
        );
    }

    // 3D
    {
        let simplex = standard_simplex::<3>();
        let queries = hot_queries::<3>();
        group.bench_with_input(
            BenchmarkId::new("insphere_3d", HOT_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere(black_box(&simplex), black_box(&queries)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("insphere_lifted_3d", HOT_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere_lifted(black_box(&simplex), black_box(&queries)));
            },
        );
    }

    // 4D
    {
        let simplex = standard_simplex::<4>();
        let queries = hot_queries::<4>();
        group.bench_with_input(
            BenchmarkId::new("insphere_4d", HOT_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere(black_box(&simplex), black_box(&queries)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("insphere_lifted_4d", HOT_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere_lifted(black_box(&simplex), black_box(&queries)));
            },
        );
    }

    // 5D — Stage 1 in `insphere_from_matrix` is skipped because `det_errbound()`
    // returns None for D ≥ 5; all queries go straight to Stage 2.  This is kept
    // here as a Stage-2-dominant reference group rather than a hot path.
    {
        let simplex = standard_simplex::<5>();
        let queries = hot_queries::<5>();
        group.bench_with_input(
            BenchmarkId::new("insphere_5d", HOT_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere(black_box(&simplex), black_box(&queries)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("insphere_lifted_5d", HOT_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere_lifted(black_box(&simplex), black_box(&queries)));
            },
        );
    }

    group.finish();
}

/// Benchmark the historical `near_boundary` centered-query workload (2D–4D).
fn bench_centered_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicates/near_boundary");
    group.throughput(Throughput::Elements(CENTERED_QUERIES as u64));

    {
        let simplex = standard_simplex::<2>();
        let queries = centered_queries::<2>();
        group.bench_with_input(
            BenchmarkId::new("insphere_2d", CENTERED_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere(black_box(&simplex), black_box(&queries)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("insphere_lifted_2d", CENTERED_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere_lifted(black_box(&simplex), black_box(&queries)));
            },
        );
    }

    {
        let simplex = standard_simplex::<3>();
        let queries = centered_queries::<3>();
        group.bench_with_input(
            BenchmarkId::new("insphere_3d", CENTERED_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere(black_box(&simplex), black_box(&queries)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("insphere_lifted_3d", CENTERED_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere_lifted(black_box(&simplex), black_box(&queries)));
            },
        );
    }

    {
        let simplex = standard_simplex::<4>();
        let queries = centered_queries::<4>();
        group.bench_with_input(
            BenchmarkId::new("insphere_4d", CENTERED_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere(black_box(&simplex), black_box(&queries)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("insphere_lifted_4d", CENTERED_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere_lifted(black_box(&simplex), black_box(&queries)));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hot_path,
    bench_centered_queries,
    bench_exact_fallback
);
criterion_main!(benches);
