//! Circumsphere query batches over large 3D triangulations.
//!
//! Manual diagnostics extracted from `profiling_suite`; excluded from release signals.
//! Fixtures are initialized inside selected Criterion callbacks, before timing.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::collections::SmallBuffer;
use delaunay::prelude::construction::DelaunayTriangulationBuilder;
use delaunay::prelude::geometry::Point;
use delaunay::prelude::query::{insphere, insphere_lifted};
use std::{hint::black_box, sync::OnceLock};
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
#[path = "common/profiling_fixtures.rs"]
mod profiling_fixtures;
use bench_utils::{OrAbort, OrAbortWithContext};
use profiling_fixtures::{
    DEFAULT_SEED, PointDistribution, bench_time, criterion_config, gen_points, is_dev_mode,
    vertices_from_points,
};

const SIMPLEX_VERTICES_BUFFER_SIZE: usize = 4;
const QUERY_RESULTS_BUFFER_SIZE: usize = 1024;
const QUERY_SEED: u64 = 123;
const MAX_QUERY_RESULTS: usize = 1_000;

/// Query latency profiling for circumsphere containment tests
fn benchmark_query_latency(c: &mut Criterion) {
    const MAX_PRECOMPUTED_SIMPLICES: usize = 1000;
    let counts = if is_dev_mode() {
        &[1_000, 3_000][..]
    } else {
        &[1_000, 10_000, 30_000][..]
    };

    let mut group = c.benchmark_group("query_latency");
    group.measurement_time(bench_time(90));
    group.throughput(Throughput::Elements(MAX_QUERY_RESULTS as u64));

    for &count in counts {
        let fixture = OnceLock::new();
        // Create triangulation and test circumsphere queries
        group.bench_with_input(
            BenchmarkId::new("circumsphere_queries_3d", count),
            &count,
            |b, &count| {
                let (precomputed_simplices, query_points) = fixture.get_or_init(|| {
                    // Setup: Create triangulation and query points
                    let points = gen_points::<3>(count, PointDistribution::Random, DEFAULT_SEED);
                    let vertices = vertices_from_points(&points);
                    let dt = DelaunayTriangulationBuilder::new(&vertices)
                        .build()
                        .or_abort();
                    assert_eq!(dt.number_of_vertices(), count);
                    dt.validate().or_abort();

                    // Generate query points
                    let query_points = gen_points::<3>(100, PointDistribution::Random, QUERY_SEED);

                    // Precompute all valid simplex vertices outside the benchmark loop
                    let mut precomputed_simplices: Vec<
                        SmallBuffer<Point<3>, SIMPLEX_VERTICES_BUFFER_SIZE>,
                    > = Vec::with_capacity(MAX_PRECOMPUTED_SIMPLICES);
                    for (_, simplex) in dt.simplices().take(MAX_PRECOMPUTED_SIMPLICES) {
                        let vertex_keys = simplex.vertices();
                        assert_eq!(vertex_keys.len(), 4);
                        let mut vertex_points: SmallBuffer<Point<3>, SIMPLEX_VERTICES_BUFFER_SIZE> =
                            SmallBuffer::new();
                        for vkey in vertex_keys {
                            let vertex = dt
                                .vertex(*vkey)
                                .or_abort("validated simplex vertex must exist");
                            vertex_points.push(*vertex.point());
                        }
                        precomputed_simplices.push(vertex_points);
                    }
                    assert!(precomputed_simplices.len() * query_points.len() >= MAX_QUERY_RESULTS);
                    for (simplex, query) in precomputed_simplices
                        .iter()
                        .flat_map(|simplex| query_points.iter().map(move |query| (simplex, query)))
                        .take(MAX_QUERY_RESULTS)
                    {
                        let lifted = insphere_lifted(simplex, *query).or_abort();
                        assert_eq!(lifted, insphere(simplex, *query).or_abort());
                    }
                    (precomputed_simplices, query_points)
                });

                b.iter(|| {
                    // Perform circumsphere containment queries using precomputed data
                    let mut query_results: SmallBuffer<_, QUERY_RESULTS_BUFFER_SIZE> =
                        SmallBuffer::new();

                    for points_for_test in precomputed_simplices {
                        for query_point in query_points {
                            let query_point_obj = *query_point;

                            // Use the fastest circumsphere method (based on benchmark results)
                            let result =
                                insphere_lifted(points_for_test, query_point_obj).or_abort();
                            query_results.push(result);

                            // Limit total queries to prevent extremely long benchmarks
                            if query_results.len() >= MAX_QUERY_RESULTS {
                                break;
                            }
                        }

                        if query_results.len() >= MAX_QUERY_RESULTS {
                            break;
                        }
                    }

                    black_box(query_results);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(name = benches; config = criterion_config(); targets = benchmark_query_latency);
criterion_main!(benches);
