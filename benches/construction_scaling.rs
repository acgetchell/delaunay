//! Large-scale Delaunay construction in 2D–5D.
//!
//! Manual diagnostics extracted from `profiling_suite`; excluded from release signals.
//! Fixtures are initialized inside selected Criterion callbacks, before timing.

use criterion::{BatchSize, Throughput};
use criterion::{Criterion, criterion_group, criterion_main};
use delaunay::prelude::geometry::{AdaptiveKernel, ExactPredicates};
use std::{hint::black_box, sync::OnceLock};
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
#[path = "common/profiling_fixtures.rs"]
mod profiling_fixtures;
use profiling_fixtures::{
    bench_time, benchmark_vertices_from_generated_points, construct_triangulation,
    criterion_config, generated_points_in_range, scale_cases, seed_for_case, validated_fixture,
    wide_bounds,
};

/// Benchmark triangulation construction time for one dimension/count pair.
fn bench_construction<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize)
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let bench_name = format!("construction/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.throughput(Throughput::Elements(n_points as u64));
    group.measurement_time(bench_time(5));

    let seed = seed_for_case::<D>(n_points);

    if (D == 4 && n_points >= 5000) || D == 5 {
        group.sample_size(10);
        group.measurement_time(bench_time(120));
    }

    let preflight = OnceLock::new();
    group.bench_function("construct", |b| {
        // The same seeded input must succeed and preserve all five layers.
        preflight.get_or_init(|| {
            black_box(validated_fixture::<D>(n_points, seed));
        });
        b.iter_batched_ref(
            || {
                let points = generated_points_in_range::<D>(n_points, wide_bounds(), seed);
                benchmark_vertices_from_generated_points(&points)
            },
            |vertices| {
                let dt = construct_triangulation::<D>(black_box(vertices.as_slice()), seed);
                black_box(dt)
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Registers only large-scale construction, without query or validation fixtures.
fn bench_scaling(c: &mut Criterion) {
    scale_cases!(c, bench_construction);
}

criterion_group!(name = benches; config = criterion_config(); targets = bench_scaling);
criterion_main!(benches);
