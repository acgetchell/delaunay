#![forbid(unsafe_code)]

//! Benchmark: PL-manifold construction cost vs audit cadence (2D–5D)
//!
//! This benchmark compares PL-manifold construction with explicit-only and
//! always-on full global audits. Pseudomanifold construction is excluded because
//! its weaker proof is not sufficient for every production Delaunay repair.
//!
//! Intended for release-artifact and manual runs (not part of the PR CI
//! performance suite).
//!
//! Run with:
//! ```bash
//! cargo bench --locked --profile perf --bench topology_guarantee_construction
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::construction::{
    DelaunayTriangulation, DelaunayTriangulationDraft, TopologyGuarantee, Vertex, vertex,
};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateRange};
use delaunay::prelude::validation::ValidationPolicy;
use std::hint::black_box;
use std::time::Duration;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, abort_benchmark};

const SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

fn benchmark_bounds() -> CoordinateRange<f64> {
    CoordinateRange::try_new(-100.0_f64, 100.0).or_abort()
}

fn construct_with_policy<const D: usize>(
    vertices: &[Vertex<(), D>],
    validation_policy: ValidationPolicy,
) -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D> {
    let mut dt = DelaunayTriangulationDraft::with_topology_guarantee(TopologyGuarantee::PLManifold);
    dt.try_set_validation_policy(validation_policy).or_abort();
    for (vertex_index, vertex) in vertices.iter().enumerate() {
        if let Err(error) = dt.insert_with_statistics(*vertex) {
            abort_benchmark(format_args!(
                "insertion {vertex_index} failed for PLManifold with {validation_policy:?}: \
                 {error}"
            ));
        }
    }
    dt.finish().or_abort()
}

fn validate_preflight<const D: usize>(
    vertices: &[Vertex<(), D>],
    validation_policy: ValidationPolicy,
) {
    construct_with_policy(vertices, validation_policy)
        .as_triangulation()
        .validate()
        .or_abort();
}

fn bench_dimension<const D: usize>(
    c: &mut Criterion,
    dim_label: &str,
    counts: &[usize],
    seed_base: u64,
    sample_size: usize,
    measurement_time: Duration,
) {
    let mut group = c.benchmark_group(format!("topology_guarantee_construction/{dim_label}"));
    group.sample_size(sample_size);
    group.measurement_time(measurement_time);

    for &n_points in counts {
        group.throughput(Throughput::Elements(n_points as u64));

        // Deterministic input per (dimension, count).
        let seed = seed_base ^ (n_points as u64).wrapping_mul(SEED_SALT);
        let points =
            generate_random_points_in_range_seeded::<D>(n_points, benchmark_bounds(), seed)
                .or_abort();
        let vertices = points
            .into_iter()
            .map(|p| vertex!(p.into()).or_abort())
            .collect::<Vec<_>>();

        group.bench_with_input(
            BenchmarkId::new("pl_manifold_always", n_points),
            &vertices,
            |b, vertices| {
                // Prove the measured mode produces a Levels 1-3-valid result
                // without charging certification to Criterion's timed loop.
                validate_preflight(vertices, ValidationPolicy::Always);
                b.iter(|| black_box(construct_with_policy(vertices, ValidationPolicy::Always)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pl_manifold_explicit_only", n_points),
            &vertices,
            |b, vertices| {
                validate_preflight(vertices, ValidationPolicy::ExplicitOnly);
                b.iter(|| {
                    black_box(construct_with_policy(
                        vertices,
                        ValidationPolicy::ExplicitOnly,
                    ))
                });
            },
        );
    }

    group.finish();
}

fn topology_guarantee_construction_2d(c: &mut Criterion) {
    // 2D can scale higher than 3D+, but PL-manifold validation is significantly more expensive,
    // so keep the upper end moderate for repeatable local runs.
    let counts: &[usize] = &[250, 1000];
    bench_dimension::<2>(c, "2d", counts, 12_345, 10, Duration::from_secs(20));
}

fn topology_guarantee_construction_3d(c: &mut Criterion) {
    let counts: &[usize] = &[50, 100, 250];
    bench_dimension::<3>(c, "3d", counts, 23_456, 15, Duration::from_secs(20));
}

fn topology_guarantee_construction_4d(c: &mut Criterion) {
    // Keep 4D counts moderate to bound runtime and memory.
    let counts: &[usize] = &[25, 50];
    bench_dimension::<4>(c, "4d", counts, 34_567, 12, Duration::from_secs(25));
}

fn topology_guarantee_construction_5d(c: &mut Criterion) {
    // 5D gets expensive quickly; keep counts low.
    let counts: &[usize] = &[15, 25];
    bench_dimension::<5>(c, "5d", counts, 45_678, 10, Duration::from_secs(30));
}

criterion_group!(
    benches,
    topology_guarantee_construction_2d,
    topology_guarantee_construction_3d,
    topology_guarantee_construction_4d,
    topology_guarantee_construction_5d
);
criterion_main!(benches);
