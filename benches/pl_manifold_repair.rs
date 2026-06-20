//! Criterion benchmarks for PL-manifold repair paths.

#![forbid(unsafe_code)]

use std::{hint::black_box, time::Duration};

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::bench_fixtures::pl_manifold::{
    repair_overshared_facet_orphan_cleanup_3d, validated_overshared_facet_orphan_cleanup_3d,
};

#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::OrAbort;

const SAMPLE_SIZE: usize = 10;
const WARM_UP_TIME: Duration = Duration::from_millis(500);
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);
const CLUSTER_COUNTS: [usize; 4] = [1, 8, 32, 128];

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

criterion_group!(benches, bench_overshared_facets_orphan_cleanup);
criterion_main!(benches);
