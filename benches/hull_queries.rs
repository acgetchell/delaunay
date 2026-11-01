//! Boundary enumeration and convex hull extraction on large immutable 3D fixtures.
//! Manual diagnostics; excluded from the curated release-signal experiment.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use delaunay::prelude::construction::DelaunayTriangulationBuilder;
use delaunay::prelude::query::ConvexHull;
use std::{hint::black_box, sync::OnceLock};
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
#[path = "common/profiling_fixtures.rs"]
mod profiling_fixtures;
use bench_utils::OrAbort;
use profiling_fixtures::{
    DEFAULT_SEED, PointDistribution, bench_time, criterion_config, gen_points, is_dev_mode,
    vertices_from_points,
};

/// Shares validated immutable geometry across the selected read-only operations.
fn bench_hull_queries(c: &mut Criterion) {
    let count = if is_dev_mode() { 3_000 } else { 10_000 };
    let fixture = OnceLock::new();
    let mut group = c.benchmark_group("algorithmic_bottlenecks");
    group.measurement_time(bench_time(15));
    let prepare = || {
        let points = gen_points::<3>(count, PointDistribution::Random, DEFAULT_SEED);
        let vertices = vertices_from_points(&points);
        let dt = DelaunayTriangulationBuilder::new(&vertices)
            .build()
            .or_abort();
        assert_eq!(dt.number_of_vertices(), count);
        dt.validate().or_abort();
        dt
    };
    group.bench_function(BenchmarkId::new("boundary_facets_3d", count), |b| {
        let dt = fixture.get_or_init(prepare);
        let expected = dt
            .boundary_facets()
            .or_abort()
            .collect::<Result<Vec<_>, _>>()
            .or_abort()
            .len();
        assert!(expected > 0);
        b.iter(|| {
            black_box(
                dt.boundary_facets()
                    .or_abort()
                    .try_fold(0_usize, |n, facet| facet.map(|_| n + 1))
                    .or_abort(),
            )
        });
    });
    group.bench_function(BenchmarkId::new("convex_hull_3d", count), |b| {
        let dt = fixture.get_or_init(prepare);
        let hull = ConvexHull::try_from_triangulation(dt.as_triangulation()).or_abort();
        black_box(hull.facets().count());
        b.iter(|| {
            black_box(
                ConvexHull::try_from_triangulation(black_box(dt.as_triangulation())).or_abort(),
            )
        });
    });
    group.finish();
}

criterion_group!(name = benches; config = criterion_config(); targets = bench_hull_queries);
criterion_main!(benches);
