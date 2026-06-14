#![forbid(unsafe_code)]

//! Benchmark: construction cost vs topology guarantee (2D–5D)
//!
//! This benchmark compares `TopologyGuarantee::Pseudomanifold`, `TopologyGuarantee::PLManifold`
//! (incremental: ridge-link during insertion-link at completion), and
//! `TopologyGuarantee::PLManifoldStrict` (vertex-link after every insertion) for Delaunay
//! triangulation construction.
//!
//! Intended for **manual** runs (not part of the CI performance suite).
//!
//! Run with:
//! ```bash
//! cargo bench --profile perf --bench topology_guarantee_construction
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::construction::{DelaunayTriangulation, TopologyGuarantee};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::CoordinateRange;
use delaunay::prelude::repair::DelaunayRepairPolicy;
use delaunay::prelude::validation::ValidationPolicy;
use std::hint::black_box;
use std::time::Duration;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{abort_benchmark, bench_result};

const SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

fn benchmark_bounds() -> CoordinateRange<f64> {
    bench_result(
        CoordinateRange::try_new(-100.0_f64, 100.0),
        "topology-guarantee benchmark bounds must be valid",
    )
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
            generate_random_points_in_range_seeded::<D>(n_points, benchmark_bounds(), seed);
        let vertices = points
            .into_iter()
            .map(|p| {
                bench_result(
                    delaunay::prelude::Vertex::<(), _>::try_new(p.into()),
                    "finite benchmark vertex coordinates",
                )
            })
            .collect::<Vec<_>>();

        group.bench_with_input(
            BenchmarkId::new("pseudomanifold", n_points),
            &vertices,
            |b, vertices| {
                b.iter(|| {
                    let mut dt: DelaunayTriangulation<_, (), (), D> =
                        DelaunayTriangulation::empty_with_topology_guarantee(
                            TopologyGuarantee::Pseudomanifold,
                        );

                    // Exercise topology validation cost under each guarantee and disable
                    // flip-based Delaunay repair for consistent comparison.
                    dt.set_validation_policy(ValidationPolicy::Always);
                    dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);

                    for v in vertices {
                        // Use the statistics API so retryable degeneracies can be skipped
                        // (transactional rollback) instead of aborting the benchmark.
                        if let Err(error) = dt.insert_with_statistics(*v) {
                            abort_benchmark(format_args!("non-retryable insertion error: {error}"));
                        }
                    }
                    // Completion-time PL-manifold certification when required.
                    let _ = dt.as_triangulation().validate_at_completion();

                    black_box(dt)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pl_manifold_strict", n_points),
            &vertices,
            |b, vertices| {
                b.iter(|| {
                    let mut dt: DelaunayTriangulation<_, (), (), D> =
                        DelaunayTriangulation::empty_with_topology_guarantee(
                            TopologyGuarantee::PLManifoldStrict,
                        );

                    dt.set_validation_policy(ValidationPolicy::Always);
                    dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);

                    for v in vertices {
                        if let Err(error) = dt.insert_with_statistics(*v) {
                            abort_benchmark(format_args!("non-retryable insertion error: {error}"));
                        }
                    }

                    // Completion-time PL-manifold certification when required.
                    let _ = dt.as_triangulation().validate_at_completion();

                    black_box(dt)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pl_manifold", n_points),
            &vertices,
            |b, vertices| {
                b.iter(|| {
                    let mut dt: DelaunayTriangulation<_, (), (), D> =
                        DelaunayTriangulation::empty_with_topology_guarantee(
                            TopologyGuarantee::PLManifold,
                        );

                    dt.set_validation_policy(ValidationPolicy::Always);
                    dt.set_delaunay_repair_policy(DelaunayRepairPolicy::Never);

                    for v in vertices {
                        if let Err(error) = dt.insert_with_statistics(*v) {
                            abort_benchmark(format_args!("non-retryable insertion error: {error}"));
                        }
                    }
                    // Completion-time PL-manifold certification when required.
                    let _ = dt.as_triangulation().validate_at_completion();

                    black_box(dt)
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
