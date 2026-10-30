//! Layer-local and cumulative validation costs in 2D–5D.
//!
//! Manual diagnostics extracted from `profiling_suite`; excluded from release signals.
//! Fixtures are initialized inside selected Criterion callbacks, before timing.

use bench_utils::OrAbort;
use bench_utils::abort_benchmark;
use criterion::Throughput;
use criterion::{Criterion, criterion_group, criterion_main};
use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayTriangulationBuilder, RetryPolicy,
};
use delaunay::prelude::geometry::{AdaptiveKernel, ExactPredicates};
use std::{hint::black_box, sync::OnceLock};
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
#[path = "common/profiling_fixtures.rs"]
mod profiling_fixtures;
use profiling_fixtures::{
    DEFAULT_SEED, PointDistribution, bench_time, benchmark_vertices_from_generated_points,
    criterion_config, gen_points, retry_attempts, scale_cases, seed_for_case, validated_fixture,
};

/// Benchmark topology validation over a prebuilt triangulation.
fn bench_validation<const D: usize>(c: &mut Criterion, dimension_name: &str, n_points: usize)
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let bench_name = format!("validation/{dimension_name}/{n_points}v");
    let mut group = c.benchmark_group(&bench_name);
    group.measurement_time(bench_time(5));

    if n_points >= 5000 || D == 5 {
        group.sample_size(10);
        group.measurement_time(bench_time(120));
    }

    let fixture = OnceLock::new();
    group.bench_function("validate_topology", |b| {
        let dt =
            fixture.get_or_init(|| validated_fixture::<D>(n_points, seed_for_case::<D>(n_points)));
        let tri = dt.as_triangulation();
        b.iter(|| {
            if let Err(error) = tri.is_valid_topology() {
                abort_benchmark(format_args!(
                    "triangulation should be structurally valid during validation benchmark: {error}"
                ));
            }
        });
    });

    group.finish();
}

macro_rules! benchmark_validation_components_dimension {
    ($dim:literal, $func_name:ident, $count:expr) => {
        fn $func_name(c: &mut Criterion) {
            let is_adversarial = stringify!($func_name).ends_with("_adversarial");
            let distribution = if is_adversarial {
                PointDistribution::Adversarial
            } else {
                PointDistribution::Random
            };
            let suffix = if is_adversarial { "_adversarial" } else { "" };
            let seed = DEFAULT_SEED;
            let fixture = OnceLock::new();
            let prepare = || {
                let points = gen_points::<$dim>($count, distribution, seed);
                let vertices = benchmark_vertices_from_generated_points(&points);
                let builder = DelaunayTriangulationBuilder::new(&vertices);
                let builder = if is_adversarial {
                    let attempts = retry_attempts(8);
                    builder.construction_options(
                        ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
                            attempts,
                            base_seed: Some(seed),
                        }),
                    )
                } else {
                    builder
                };
                let dt = builder.build().unwrap_or_else(|error| {
                    abort_benchmark(format_args!(
                        "{}D validation component fixture{suffix} seed {seed:#x} failed: {error}",
                        $dim,
                    ));
                });
                assert_eq!(dt.number_of_vertices(), $count);
                dt.validate().or_abort();
                dt
            };

            let mut group = c.benchmark_group(format!(
                "validation_components_{}d{}_seed_{seed:016x}", $dim, suffix
            ));
            group.measurement_time(bench_time(15));
            group.throughput(Throughput::Elements($count as u64));

            group.bench_function("is_valid_structure", |b| {
                let dt = fixture.get_or_init(prepare);
                b.iter(|| {
                    if let Err(error) = black_box(dt.is_valid_structure()) {
                        abort_benchmark(format_args!(
                            "structure validation should pass for benchmark triangulation: {error}"
                        ));
                    }
                });
            });

            group.bench_function("is_valid_topology", |b| {
                let dt = fixture.get_or_init(prepare);
                b.iter(|| {
                    if let Err(error) = black_box(dt.as_triangulation().is_valid_topology()) {
                        abort_benchmark(format_args!(
                            "triangulation validation should pass for benchmark triangulation: {error}"
                        ));
                    }
                });
            });

            group.bench_function("validate_realization", |b| {
                let dt = fixture.get_or_init(prepare);
                b.iter(|| {
                    if let Err(error) = black_box(dt.as_triangulation().validate_realization()) {
                        abort_benchmark(format_args!(
                            "realization validation should pass for benchmark triangulation: {error}"
                        ));
                    }
                });
            });

            group.bench_function("is_valid_delaunay", |b| {
                let dt = fixture.get_or_init(prepare);
                b.iter(|| {
                    if let Err(error) = black_box(dt.is_valid_delaunay()) {
                        abort_benchmark(format_args!(
                            "Delaunay validation should pass for benchmark triangulation: {error}"
                        ));
                    }
                });
            });

            group.bench_function("validate", |b| {
                let dt = fixture.get_or_init(prepare);
                b.iter(|| {
                    if let Err(error) = black_box(dt.validate()) {
                        abort_benchmark(format_args!(
                            "full validation should pass for benchmark triangulation: {error}"
                        ));
                    }
                });
            });

            group.finish();
        }
    };
}

benchmark_validation_components_dimension!(2, benchmark_validation_components_2d, 50);
benchmark_validation_components_dimension!(3, benchmark_validation_components_3d, 50);
benchmark_validation_components_dimension!(4, benchmark_validation_components_4d, 25);
benchmark_validation_components_dimension!(5, benchmark_validation_components_5d, 25);
benchmark_validation_components_dimension!(2, benchmark_validation_components_2d_adversarial, 50);
benchmark_validation_components_dimension!(3, benchmark_validation_components_3d_adversarial, 50);
benchmark_validation_components_dimension!(4, benchmark_validation_components_4d_adversarial, 25);
benchmark_validation_components_dimension!(5, benchmark_validation_components_5d_adversarial, 25);

/// Registers the retained large-fixture Level 3 scaling cases.
fn bench_validation_scaling(c: &mut Criterion) {
    scale_cases!(c, bench_validation);
}

criterion_group!(name = benches; config = criterion_config(); targets = bench_validation_scaling, benchmark_validation_components_2d, benchmark_validation_components_3d, benchmark_validation_components_4d, benchmark_validation_components_5d, benchmark_validation_components_2d_adversarial, benchmark_validation_components_3d_adversarial, benchmark_validation_components_4d_adversarial, benchmark_validation_components_5d_adversarial);
criterion_main!(benches);
