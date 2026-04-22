//! Microbenchmark for `core::hint::cold_path` adoption in geometric predicates.
//!
//! This benchmark exercises the hot Stage-1 path of the [`insphere`] / [`insphere_lifted`]
//! predicates (and implicitly the orientation predicate they both invoke) across
//! 2D–5D.  A small "near-boundary" group is included to guard against regressions
//! when `cold_path` is added to Stage-2 / Stage-3 branches.
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
//! `near_boundary` group constructs test points very close to the circumsphere
//! boundary to exercise Stage 2.
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
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::prelude::query::*;
use std::hint::black_box;

/// Deterministic seed for query-point generation in the hot path.
const HOT_SEED: u64 = 0xC01D_BEEF_0000_CAFE_u64;
/// Deterministic seed for query-point generation in the near-boundary group.
const NEAR_BOUNDARY_SEED: u64 = 0x0000_0000_0000_BEEF_u64;
/// Number of queries per hot-path benchmark.  Keep above the size where
/// per-iteration overhead dominates so Stage-1 improvements are visible.
const HOT_QUERIES: usize = 10_000;
/// Number of queries per near-boundary benchmark.  Kept smaller because
/// Stage 2 is slower per call.
const NEAR_BOUNDARY_QUERIES: usize = 1_000;

/// Standard D-dimensional simplex: origin + unit basis vectors.
fn standard_simplex<const D: usize>() -> Vec<Point<f64, D>> {
    let mut pts = Vec::with_capacity(D + 1);
    pts.push(Point::new([0.0; D]));
    for i in 0..D {
        let mut coords = [0.0; D];
        coords[i] = 1.0;
        pts.push(Point::new(coords));
    }
    pts
}

/// Generate well-separated hot-path query points for dimension `D`.
///
/// Uses the range `[-10, 10]` against a unit simplex so that the Shewchuk
/// errbound comfortably resolves the sign in Stage 1.
fn hot_queries<const D: usize>() -> Vec<Point<f64, D>> {
    generate_random_points_seeded(HOT_QUERIES, (-10.0, 10.0), HOT_SEED)
        .expect("failed to generate hot-path query points")
}

/// Generate near-boundary query points for dimension `D`.
///
/// Uses a narrow range centered on the standard simplex so many queries land
/// within the Stage-1 errbound window and spill into Stage 2.
fn near_boundary_queries<const D: usize>() -> Vec<Point<f64, D>> {
    // Centered near the circumsphere radius of the standard simplex (~0.5 for
    // the D = 3 unit case); the exact value is unimportant — we just want a
    // high rate of errbound-ambiguous inputs.
    generate_random_points_seeded(NEAR_BOUNDARY_QUERIES, (0.40, 0.60), NEAR_BOUNDARY_SEED)
        .expect("failed to generate near-boundary query points")
}

/// Run `insphere` across `queries` against `simplex`, black-boxing each result.
fn run_insphere<const D: usize>(simplex: &[Point<f64, D>], queries: &[Point<f64, D>]) {
    for q in queries {
        black_box(insphere(black_box(simplex), black_box(*q)).unwrap());
    }
}

/// Run `insphere_lifted` across `queries` against `simplex`, black-boxing each result.
fn run_insphere_lifted<const D: usize>(simplex: &[Point<f64, D>], queries: &[Point<f64, D>]) {
    for q in queries {
        black_box(insphere_lifted(black_box(simplex), black_box(*q)).unwrap());
    }
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

/// Benchmark the Stage-2 cold path via near-boundary queries (2D–4D).
fn bench_near_boundary(c: &mut Criterion) {
    let mut group = c.benchmark_group("predicates/near_boundary");
    group.throughput(Throughput::Elements(NEAR_BOUNDARY_QUERIES as u64));

    {
        let simplex = standard_simplex::<2>();
        let queries = near_boundary_queries::<2>();
        group.bench_with_input(
            BenchmarkId::new("insphere_2d", NEAR_BOUNDARY_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere(black_box(&simplex), black_box(&queries)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("insphere_lifted_2d", NEAR_BOUNDARY_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere_lifted(black_box(&simplex), black_box(&queries)));
            },
        );
    }

    {
        let simplex = standard_simplex::<3>();
        let queries = near_boundary_queries::<3>();
        group.bench_with_input(
            BenchmarkId::new("insphere_3d", NEAR_BOUNDARY_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere(black_box(&simplex), black_box(&queries)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("insphere_lifted_3d", NEAR_BOUNDARY_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere_lifted(black_box(&simplex), black_box(&queries)));
            },
        );
    }

    {
        let simplex = standard_simplex::<4>();
        let queries = near_boundary_queries::<4>();
        group.bench_with_input(
            BenchmarkId::new("insphere_4d", NEAR_BOUNDARY_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere(black_box(&simplex), black_box(&queries)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("insphere_lifted_4d", NEAR_BOUNDARY_QUERIES),
            &(),
            |b, ()| {
                b.iter(|| run_insphere_lifted(black_box(&simplex), black_box(&queries)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_hot_path, bench_near_boundary);
criterion_main!(benches);
