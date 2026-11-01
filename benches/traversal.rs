//! Neighbor queries and vertex/simplex iteration in 2D–5D.
//!
//! Manual diagnostics extracted from `profiling_suite`; excluded from release signals.
//! Fixtures are initialized inside selected Criterion callbacks, before timing.

use bench_utils::OrAbortWithContext;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::construction::DelaunayTriangulation;
use delaunay::prelude::geometry::{AdaptiveKernel, ExactPredicates};
use delaunay::prelude::tds::SimplexKey;
use std::{hint::black_box, sync::OnceLock};
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
#[path = "common/profiling_fixtures.rs"]
mod profiling_fixtures;
use profiling_fixtures::{
    bench_time, criterion_config, scale_cases, seed_for_case, validated_fixture,
};

/// Benchmark neighbor lookup over all simplices.
fn bench_neighbor_queries<const D: usize>(
    c: &mut Criterion,
    dimension_name: &str,
    n_points: usize,
    fixture: &OnceLock<DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>>,
) where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let bench_name = format!("queries/neighbors/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.measurement_time(bench_time(5));

    if D == 5 || (D == 4 && n_points >= 5000) {
        group.sample_size(10);
        group.measurement_time(bench_time(120));
    }

    group.bench_function("find_neighbors_all_simplices", |b| {
        let dt =
            fixture.get_or_init(|| validated_fixture::<D>(n_points, seed_for_case::<D>(n_points)));
        let simplex_keys: Vec<SimplexKey> = dt.simplices().map(|(key, _)| key).collect();
        for &key in &simplex_keys {
            for neighbor in dt.simplex_neighbors(key) {
                dt.simplex(neighbor)
                    .or_abort("neighbor must exist in the validated owner");
            }
        }
        b.iter(|| {
            for &simplex_key in &simplex_keys {
                for neighbor in dt.simplex_neighbors(simplex_key) {
                    black_box(neighbor);
                }
            }
        });
    });

    group.finish();
}

/// Benchmark vertex iteration over a prebuilt triangulation.
fn bench_vertex_iteration<const D: usize>(
    c: &mut Criterion,
    dimension_name: &str,
    n_points: usize,
    fixture: &OnceLock<DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>>,
) where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let bench_name = format!("queries/vertices/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.throughput(Throughput::Elements(n_points as u64));
    group.measurement_time(bench_time(5));

    if D == 5 || (D == 4 && n_points >= 5000) {
        group.sample_size(10);
        group.measurement_time(bench_time(120));
    }

    group.bench_function("iterate_all_vertices", |b| {
        let dt =
            fixture.get_or_init(|| validated_fixture::<D>(n_points, seed_for_case::<D>(n_points)));
        b.iter(|| {
            let mut count = 0;
            for (_, vertex) in dt.vertices() {
                black_box(vertex);
                count += 1;
            }
            black_box(count)
        });
    });

    group.finish();
}

/// Benchmark simplex-key iteration over a prebuilt triangulation.
fn bench_simplex_iteration<const D: usize>(
    c: &mut Criterion,
    dimension_name: &str,
    n_points: usize,
    fixture: &OnceLock<DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>>,
) where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let bench_name = format!("queries/simplices/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.measurement_time(bench_time(5));

    if D == 5 || (D == 4 && n_points >= 5000) {
        group.sample_size(10);
        group.measurement_time(bench_time(120));
    }

    group.bench_function("iterate_all_simplices", |b| {
        let dt =
            fixture.get_or_init(|| validated_fixture::<D>(n_points, seed_for_case::<D>(n_points)));
        b.iter(|| {
            let mut count = 0;
            for (simplex_key, _) in dt.simplices() {
                black_box(simplex_key);
                count += 1;
            }
            black_box(count)
        });
    });

    group.finish();
}

/// Registers traversal diagnostics; filtering selects fixtures lazily.
fn bench_traversal(c: &mut Criterion) {
    scale_cases!(c, bench_dimension);
}

/// Shares the immutable fixture across selected traversal cases at one scale.
fn bench_dimension<const D: usize>(c: &mut Criterion, name: &str, count: usize)
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let fixture = OnceLock::new();
    bench_neighbor_queries::<D>(c, name, count, &fixture);
    bench_vertex_iteration::<D>(c, name, count, &fixture);
    bench_simplex_iteration::<D>(c, name, count, &fixture);
}

criterion_group!(name = benches; config = criterion_config(); targets = bench_traversal);
criterion_main!(benches);
