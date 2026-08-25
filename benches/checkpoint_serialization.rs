//! Throughput evidence for scientific checkpoint manifest and wire serialization.

#![forbid(unsafe_code)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::checkpoint::DelaunayCheckpoint;
use delaunay::prelude::construction::{
    DelaunayTriangulation, DelaunayTriangulationBuilder, Vertex,
};
use delaunay::prelude::geometry::AdaptiveKernel;
use serde_json::{from_slice, to_vec};
use std::hint::black_box;

#[path = "common/bench_utils.rs"]
mod bench_utils;
use bench_utils::{OrAbort, OrAbortWithContext};

fn representative_triangulation() -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 2> {
    let mut vertices = Vec::with_capacity(64);
    for row in 0_u32..8 {
        for column in 0_u32..8 {
            let jitter = f64::from((row * 17 + column * 31) % 11) * 1.0e-4;
            vertices.push(
                Vertex::try_new([f64::from(column) + jitter, f64::from(row) - jitter]).or_abort(),
            );
        }
    }
    DelaunayTriangulationBuilder::new(&vertices)
        .build()
        .or_abort()
}

fn checkpoint_benches(criterion: &mut Criterion) {
    let triangulation = representative_triangulation();
    let checkpoint_json = to_vec(&triangulation).or_abort();
    let expected_manifest = triangulation.checkpoint_manifest().or_abort();
    let checkpoint: DelaunayCheckpoint<(), (), 2> = from_slice(&checkpoint_json).or_abort();
    let restored = checkpoint
        .try_into_delaunay_with_kernel(AdaptiveKernel::new())
        .or_abort();
    assert_eq!(restored.checkpoint_manifest().or_abort(), expected_manifest);
    restored.validate().or_abort();
    restored
        .simplices()
        .next()
        .or_abort("restored checkpoint benchmark must contain a simplex");

    let mut group = criterion.benchmark_group("checkpoint_serialization/64_vertices");
    group.throughput(Throughput::Elements(64));

    group.bench_function("manifest", |bencher| {
        bencher.iter(|| black_box(triangulation.checkpoint_manifest()).or_abort());
    });
    group.bench_function("json", |bencher| {
        bencher.iter(|| black_box(to_vec(black_box(&triangulation))).or_abort());
    });
    group.bench_function("json_load", |bencher| {
        bencher.iter(|| {
            let checkpoint: DelaunayCheckpoint<(), (), 2> =
                from_slice(black_box(&checkpoint_json)).or_abort();
            black_box(
                checkpoint
                    .try_into_delaunay_with_kernel(AdaptiveKernel::new())
                    .or_abort(),
            );
        });
    });
    group.finish();
}

criterion_group!(benches, checkpoint_benches);
criterion_main!(benches);
