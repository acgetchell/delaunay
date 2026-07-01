//! Criterion benchmarks for PL-manifold repair paths.

#![forbid(unsafe_code)]

use std::{hint::black_box, time::Duration};

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::bench_fixtures::pl_manifold::{
    TargetedTopologyRepairFixture, repair_overshared_facet_orphan_cleanup_3d,
    repair_targeted_pl_manifold_topology, validated_boundary_ridge_multiplicity_repair_3d,
    validated_overshared_facet_orphan_cleanup_3d, validated_ridge_link_repair_2d,
    validated_vertex_link_repair_3d,
};

#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::OrAbort;

const SAMPLE_SIZE: usize = 10;
const WARM_UP_TIME: Duration = Duration::from_millis(500);
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);
const CLUSTER_COUNTS: [usize; 4] = [1, 8, 32, 128];
const TARGETED_CLUSTER_COUNTS: [usize; 3] = [1, 4, 16];
const VERTEX_LINK_CLUSTER_COUNTS: [usize; 2] = [1, 4];

/// Benchmarks repair of over-shared facets where each removed simplex creates
/// one orphaned vertex.
fn bench_overshared_facets_orphan_cleanup(c: &mut Criterion) {
    let mut group = c.benchmark_group("pl_manifold_repair/overshared_facets_orphan_cleanup");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);

    for cluster_count in CLUSTER_COUNTS {
        let fixture = validated_overshared_facet_orphan_cleanup_3d(cluster_count).or_abort();
        group.throughput(Throughput::Elements(
            u64::try_from(fixture.cluster_count).or_abort(),
        ));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_clusters", fixture.cluster_count)),
            &fixture,
            |b, fixture| {
                b.iter_batched(
                    || fixture.tds.clone(),
                    |mut tds| {
                        let stats = repair_overshared_facet_orphan_cleanup_3d(&mut tds).or_abort();
                        let _ = black_box(stats);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmarks the targeted boundary-ridge, ridge-link, and vertex-link repair stages.
fn bench_targeted_topology_repair(c: &mut Criterion) {
    let mut group = c.benchmark_group("pl_manifold_repair/targeted_topology");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);

    for cluster_count in TARGETED_CLUSTER_COUNTS {
        let boundary_fixture =
            validated_boundary_ridge_multiplicity_repair_3d(cluster_count).or_abort();
        bench_targeted_fixture(
            &mut group,
            "boundary_ridge_multiplicity_3d",
            &boundary_fixture,
        );

        let ridge_fixture = validated_ridge_link_repair_2d(cluster_count).or_abort();
        bench_targeted_fixture(&mut group, "ridge_link_2d", &ridge_fixture);
    }

    for cluster_count in VERTEX_LINK_CLUSTER_COUNTS {
        let vertex_fixture = validated_vertex_link_repair_3d(cluster_count).or_abort();
        bench_targeted_fixture(&mut group, "vertex_link_3d", &vertex_fixture);
    }

    group.finish();
}

/// Registers one targeted topology repair fixture with Criterion.
fn bench_targeted_fixture<const D: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    stage_name: &str,
    fixture: &TargetedTopologyRepairFixture<D>,
) {
    group.throughput(Throughput::Elements(
        u64::try_from(fixture.cluster_count).or_abort(),
    ));

    group.bench_with_input(
        BenchmarkId::new(stage_name, format!("{}_clusters", fixture.cluster_count)),
        fixture,
        |b, fixture| {
            b.iter_batched(
                || fixture.tds.clone(),
                |mut tds| {
                    let stats = repair_targeted_pl_manifold_topology(&mut tds).or_abort();
                    let _ = black_box(stats);
                },
                BatchSize::SmallInput,
            );
        },
    );
}

criterion_group!(
    benches,
    bench_overshared_facets_orphan_cleanup,
    bench_targeted_topology_repair
);
criterion_main!(benches);
