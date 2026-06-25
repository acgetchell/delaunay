#![forbid(unsafe_code)]

//! Focused microbenchmarks for public edge-key query construction.
//!
//! Run with:
//!
//! ```bash
//! cargo bench --profile perf --bench edge_key_queries -- --noplot
//! ```

#[path = "common/bench_utils.rs"]
mod bench_utils;

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use delaunay::prelude::construction::{DelaunayTriangulation, DelaunayTriangulationBuilder};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateRange};
use delaunay::prelude::tds::{EdgeKey, VertexKey};
use delaunay::try_vertices_from_points;
use std::{collections::BTreeSet, hint::black_box, time::Duration};

use bench_utils::{OrAbort, OrAbortWithContext, abort_benchmark};

const EDGE_PAIR_COUNT: usize = 128;
const COUNT_2D: usize = 1_000;
const COUNT_3D: usize = 350;
const COUNT_4D: usize = 80;
const COUNT_5D: usize = 35;
const SEED_2D: u64 = 9_321;
const SEED_3D: u64 = 9_322;
const SEED_4D: u64 = 9_323;
const SEED_5D: u64 = 9_324;
const SAMPLE_SIZE: usize = 32;

type BenchTriangulation<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;

/// Holds a prebuilt triangulation and live edge endpoints selected outside the measured path.
struct EdgeKeyFixture<const D: usize> {
    dt: BenchTriangulation<D>,
    endpoint_pairs: Vec<(VertexKey, VertexKey)>,
    vertex_count: usize,
    simplex_count: usize,
}

/// Returns a deterministic coordinate range for benchmark point clouds.
fn benchmark_bounds() -> CoordinateRange<f64> {
    CoordinateRange::try_new(-100.0_f64, 100.0).or_abort()
}

/// Builds a deterministic triangulation and live edge endpoint pairs for one dimension.
fn prepare_fixture<const D: usize>(count: usize, seed: u64) -> EdgeKeyFixture<D> {
    let points =
        generate_random_points_in_range_seeded::<D>(count, benchmark_bounds(), seed).or_abort();
    let vertices = try_vertices_from_points(&points).or_abort();
    let dt = DelaunayTriangulationBuilder::new(&vertices)
        .build::<()>()
        .or_abort();
    let endpoint_pairs = late_simplex_edge_pairs(&dt);
    if endpoint_pairs.len() < EDGE_PAIR_COUNT {
        abort_benchmark(format_args!(
            "{D}D EdgeKey benchmark found {} unique edge pairs, expected at least {EDGE_PAIR_COUNT}",
            endpoint_pairs.len()
        ));
    }
    let _first_pair = endpoint_pairs.first().or_abort(format_args!(
        "{D}D EdgeKey benchmark should contain at least one endpoint pair"
    ));

    EdgeKeyFixture {
        vertex_count: dt.number_of_vertices(),
        simplex_count: dt.number_of_simplices(),
        dt,
        endpoint_pairs,
    }
}

/// Selects unique live edges from the end of simplex storage to exercise lookup cost.
fn late_simplex_edge_pairs<const D: usize>(
    dt: &BenchTriangulation<D>,
) -> Vec<(VertexKey, VertexKey)> {
    let simplex_vertices: Vec<_> = dt
        .simplices()
        .map(|(_simplex_key, simplex)| simplex.vertices().to_vec())
        .collect();
    let mut pairs = BTreeSet::new();

    for vertices in simplex_vertices.iter().rev() {
        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                let first = vertices[i];
                let second = vertices[j];
                pairs.insert(canonical_pair(first, second));
                if pairs.len() >= EDGE_PAIR_COUNT {
                    return pairs.into_iter().collect();
                }
            }
        }
    }

    pairs.into_iter().collect()
}

/// Canonicalizes an endpoint pair without constructing the `EdgeKey` being benchmarked.
fn canonical_pair(first: VertexKey, second: VertexKey) -> (VertexKey, VertexKey) {
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

/// Measures successful public `EdgeKey::try_new` construction for one dimension.
fn bench_edge_key_try_new<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    fixture: &EdgeKeyFixture<D>,
) {
    let tds = fixture.dt.tds();
    let endpoint_pairs = &fixture.endpoint_pairs;

    group.bench_function(
        BenchmarkId::new(
            format!("edge_key_try_new_{D}d"),
            format!(
                "vertices_{}_simplices_{}_edges_{}",
                fixture.vertex_count,
                fixture.simplex_count,
                endpoint_pairs.len()
            ),
        ),
        |b| {
            b.iter(|| {
                let mut edge_count = 0_usize;
                for &(first, second) in endpoint_pairs {
                    let edge =
                        EdgeKey::try_new(tds, black_box(first), black_box(second)).or_abort();
                    black_box(edge);
                    edge_count += 1;
                }
                assert_eq!(edge_count, endpoint_pairs.len());
                black_box(edge_count);
            });
        },
    );
}

/// Runs edge-key construction benchmarks across supported practical dimensions.
fn edge_key_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_key_queries");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    let fixture_2d = prepare_fixture::<2>(COUNT_2D, SEED_2D);
    let fixture_3d = prepare_fixture::<3>(COUNT_3D, SEED_3D);
    let fixture_4d = prepare_fixture::<4>(COUNT_4D, SEED_4D);
    let fixture_5d = prepare_fixture::<5>(COUNT_5D, SEED_5D);

    bench_edge_key_try_new(&mut group, &fixture_2d);
    bench_edge_key_try_new(&mut group, &fixture_3d);
    bench_edge_key_try_new(&mut group, &fixture_4d);
    bench_edge_key_try_new(&mut group, &fixture_5d);
    group.finish();
}

criterion_group!(benches, edge_key_queries);
criterion_main!(benches);
