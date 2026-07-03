#![forbid(unsafe_code)]

//! Benchmark: point-location (`locate`) facet-walk latency.
//!
//! This benchmark isolates locate cost from insertion cost by building
//! deterministic triangulations once, precomputing interior query batches in
//! setup, and timing only the `locate` calls. Two cases are tracked per
//! dimension and fixture size:
//!
//! - `no_hint`: every query starts from the triangulation's first simplex,
//!   measuring full facet-walk cost.
//! - `exact_hint`: every query passes its own containing simplex as the hint,
//!   measuring the hint fast path used by locality-aware callers such as
//!   incremental insertion.
//!
//! Intended for **manual** runs (not part of the CI performance suite).
//!
//! Run with:
//! ```bash
//! cargo bench --profile perf --bench locate
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::algorithms::{LocateResult, locate};
use delaunay::prelude::construction::{
    DelaunayTriangulation, DelaunayTriangulationBuilder, Vertex,
};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateRange, Point};
use delaunay::prelude::tds::SimplexKey;
use delaunay::try_vertices_from_points;
use std::hint::black_box;
use std::time::Duration;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, abort_benchmark};

const SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
const SEED_SEARCH_ATTEMPTS: usize = 16;
const QUERY_COUNT: usize = 64;
const QUERY_CANDIDATE_FACTOR: usize = 8;
const SAMPLE_SIZE: usize = 10;
const WARM_UP_TIME: Duration = Duration::from_millis(500);
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);

type BenchTriangulation<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;

struct LocateSource<const D: usize> {
    vertex_count: usize,
    simplex_count: usize,
    triangulation: BenchTriangulation<D>,
    /// Interior queries paired with their containing simplex for the hint fast path.
    hinted_queries: Vec<(Point<D>, SimplexKey)>,
}

/// Derive a deterministic, dimension-specific seed for one benchmark case.
fn seed_for_case<const D: usize>(requested_vertices: usize, seed_base: u64) -> u64 {
    let vertices = u64::try_from(requested_vertices).or_abort();
    let dimension = u64::try_from(D).or_abort();
    seed_base ^ vertices.wrapping_mul(SEED_SALT) ^ dimension.rotate_left(32)
}

/// Coordinate range used for triangulation points.
fn point_bounds() -> CoordinateRange<f64> {
    CoordinateRange::try_new(0.0_f64, 1.0).or_abort()
}

/// Coordinate range used for query candidates, interior to the point cloud.
fn query_bounds() -> CoordinateRange<f64> {
    CoordinateRange::try_new(0.25_f64, 0.75).or_abort()
}

/// Build one deterministic triangulation plus an inside-the-hull query batch.
fn build_source<const D: usize>(requested_vertices: usize, seed_base: u64) -> LocateSource<D> {
    let kernel = AdaptiveKernel::<f64>::new();

    for attempt in 0..SEED_SEARCH_ATTEMPTS {
        let attempt_seed = u64::try_from(attempt).or_abort();
        let seed = seed_for_case::<D>(requested_vertices, seed_base)
            ^ attempt_seed.wrapping_mul(SEED_SALT.rotate_left(17));
        let points =
            generate_random_points_in_range_seeded::<D>(requested_vertices, point_bounds(), seed)
                .or_abort();
        let vertices: Vec<Vertex<(), D>> = try_vertices_from_points(&points).or_abort();
        let Ok(triangulation) = DelaunayTriangulationBuilder::new(&vertices).build() else {
            continue;
        };

        let candidates = generate_random_points_in_range_seeded::<D>(
            QUERY_COUNT * QUERY_CANDIDATE_FACTOR,
            query_bounds(),
            seed ^ SEED_SALT,
        )
        .or_abort();

        let mut hinted_queries = Vec::with_capacity(QUERY_COUNT);
        for query in candidates {
            if hinted_queries.len() == QUERY_COUNT {
                break;
            }
            let located = locate(triangulation.tds(), &kernel, &query, None).or_abort();
            if let LocateResult::InsideSimplex(simplex_key) = located {
                hinted_queries.push((query, simplex_key));
            }
        }
        if hinted_queries.len() < QUERY_COUNT {
            continue;
        }

        return LocateSource {
            vertex_count: triangulation.number_of_vertices(),
            simplex_count: triangulation.number_of_simplices(),
            triangulation,
            hinted_queries,
        };
    }

    abort_benchmark(format!(
        "no {D}D locate fixture with {QUERY_COUNT} interior queries found for \
         {requested_vertices} vertices after {SEED_SEARCH_ATTEMPTS} seeds"
    ))
}

/// Register the no-hint and exact-hint locate cases for one dimension.
fn bench_locate_dimension<const D: usize>(
    c: &mut Criterion,
    dim_label: &str,
    counts: &[usize],
    seed_base: u64,
) {
    let kernel = AdaptiveKernel::<f64>::new();
    let sources: Vec<LocateSource<D>> = counts
        .iter()
        .map(|&requested_vertices| build_source::<D>(requested_vertices, seed_base))
        .collect();
    let queries_per_iteration = u64::try_from(QUERY_COUNT).or_abort();

    {
        let mut group = c.benchmark_group(format!("locate/no_hint/{dim_label}"));
        group.sample_size(SAMPLE_SIZE);
        group.warm_up_time(WARM_UP_TIME);
        group.measurement_time(MEASUREMENT_TIME);
        for source in &sources {
            group.throughput(Throughput::Elements(queries_per_iteration));
            group.bench_with_input(
                BenchmarkId::new(
                    "locate",
                    format!(
                        "vertices_{}_simplices_{}",
                        source.vertex_count, source.simplex_count
                    ),
                ),
                source,
                |b, source| {
                    b.iter(|| {
                        for (query, _) in &source.hinted_queries {
                            black_box(
                                locate(source.triangulation.tds(), &kernel, query, None).or_abort(),
                            );
                        }
                    });
                },
            );
        }
        group.finish();
    }

    {
        let mut group = c.benchmark_group(format!("locate/exact_hint/{dim_label}"));
        group.sample_size(SAMPLE_SIZE);
        group.warm_up_time(WARM_UP_TIME);
        group.measurement_time(MEASUREMENT_TIME);
        for source in &sources {
            group.throughput(Throughput::Elements(queries_per_iteration));
            group.bench_with_input(
                BenchmarkId::new(
                    "locate",
                    format!(
                        "vertices_{}_simplices_{}",
                        source.vertex_count, source.simplex_count
                    ),
                ),
                source,
                |b, source| {
                    b.iter(|| {
                        for (query, hint) in &source.hinted_queries {
                            black_box(
                                locate(source.triangulation.tds(), &kernel, query, Some(*hint))
                                    .or_abort(),
                            );
                        }
                    });
                },
            );
        }
        group.finish();
    }
}

/// Benchmark 2D point location.
fn bench_locate_2d(c: &mut Criterion) {
    bench_locate_dimension::<2>(c, "2d", &[100, 500, 2_000], 0x10CA_0000_0000_0002);
}

/// Benchmark 3D point location.
fn bench_locate_3d(c: &mut Criterion) {
    bench_locate_dimension::<3>(c, "3d", &[50, 150, 500], 0x10CA_0000_0000_0003);
}

/// Benchmark 4D point location.
fn bench_locate_4d(c: &mut Criterion) {
    bench_locate_dimension::<4>(c, "4d", &[50, 100], 0x10CA_0000_0000_0004);
}

/// Benchmark 5D point location.
fn bench_locate_5d(c: &mut Criterion) {
    bench_locate_dimension::<5>(c, "5d", &[25, 40], 0x10CA_0000_0000_0005);
}

criterion_group!(
    benches,
    bench_locate_2d,
    bench_locate_3d,
    bench_locate_4d,
    bench_locate_5d
);
criterion_main!(benches);
