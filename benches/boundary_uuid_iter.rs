#![forbid(unsafe_code)]

//! Focused microbenchmarks for boundary-facet traversal and simplex UUID iteration.
//!
//! These cases used to live as `#[cfg(feature = "bench")]` unit tests. Keeping
//! them in a Criterion harness avoids creating a third test bucket while
//! preserving the quick performance probes.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::construction::{DelaunayTriangulation, Vertex};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::CoordinateRange;
use delaunay::prelude::query::BoundaryAnalysis;
use uuid::Uuid;

use std::collections::HashSet;
use std::hint::black_box;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{bench_option, bench_result};

const BOUNDARY_COUNTS_3D: &[usize] = &[20, 40, 60, 80];

fn benchmark_bounds() -> CoordinateRange<f64> {
    bench_result(
        CoordinateRange::try_new(-100.0_f64, 100.0),
        "boundary benchmark bounds must be valid",
    )
}

fn boundary_triangulation_3d(
    requested_vertices: usize,
) -> DelaunayTriangulation<delaunay::prelude::geometry::AdaptiveKernel<f64>, (), (), 3> {
    let points = generate_random_points_in_range_seeded::<3>(
        requested_vertices,
        benchmark_bounds(),
        0xB0DA_FACE_0000_0000 ^ requested_vertices as u64,
    );
    let points = bench_result(points, "failed to generate boundary benchmark points");
    let vertices = Vertex::from_validated_points(&points);
    bench_result(
        DelaunayTriangulation::try_new(&vertices),
        "failed to build 3D boundary benchmark triangulation",
    )
}

fn bench_boundary_facets_micro(c: &mut Criterion) {
    let mut group = c.benchmark_group("boundary_facets_micro");

    for &requested_vertices in BOUNDARY_COUNTS_3D {
        let dt = boundary_triangulation_3d(requested_vertices);
        let boundary_count = bench_result(
            bench_result(dt.boundary_facets(), "boundary facets should be available")
                .try_fold(0_usize, |count, facet| facet.map(|_| count + 1)),
            "boundary facets should be valid",
        );
        group.throughput(Throughput::Elements(bench_result(
            u64::try_from(boundary_count),
            "boundary facet count fits in u64",
        )));

        group.bench_with_input(
            BenchmarkId::new("boundary_facets_count_3d", requested_vertices),
            &dt,
            |b, dt| {
                b.iter(|| {
                    black_box(bench_result(
                        bench_result(dt.boundary_facets(), "boundary facets should be available")
                            .try_fold(0_usize, |count, facet| facet.map(|_| count + 1)),
                        "boundary facets should be valid",
                    ));
                });
            },
        );

        let boundary_facets = bench_result(
            bench_result(dt.boundary_facets(), "boundary facets should be available")
                .collect::<Result<Vec<_>, _>>(),
            "boundary facets should be valid",
        );
        group.bench_with_input(
            BenchmarkId::new("is_boundary_facet_3d", requested_vertices),
            &(&dt, boundary_facets),
            |b, (dt, facets)| {
                b.iter(|| {
                    let confirmed = facets
                        .iter()
                        .filter(|facet| {
                            bench_result(
                                dt.tds().is_boundary_facet(facet),
                                "boundary facet check should succeed",
                            )
                        })
                        .count();
                    black_box(confirmed);
                });
            },
        );
    }

    group.finish();
}

fn uuid_iter_source()
-> DelaunayTriangulation<delaunay::prelude::geometry::AdaptiveKernel<f64>, (), (), 3> {
    let vertices = vec![
        bench_result(
            delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0]),
            "finite benchmark vertex coordinates",
        ),
        bench_result(
            delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0]),
            "finite benchmark vertex coordinates",
        ),
        bench_result(
            delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0]),
            "finite benchmark vertex coordinates",
        ),
        bench_result(
            delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0]),
            "finite benchmark vertex coordinates",
        ),
    ];
    bench_result(
        DelaunayTriangulation::try_new(&vertices),
        "failed to build UUID iterator benchmark triangulation",
    )
}

fn bench_vertex_uuid_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertex_uuid_iter");
    let dt = uuid_iter_source();
    let (_simplex_key, simplex) = bench_option(
        dt.simplices().next(),
        "simplex should exist for UUID iterator benchmark",
    );

    group.throughput(Throughput::Elements(bench_result(
        u64::try_from(simplex.vertices().len()),
        "vertex count fits in u64",
    )));

    group.bench_function("by_value", |b| {
        b.iter(|| {
            let unique_uuids = simplex
                .vertex_uuid_iter(dt.tds())
                .collect::<Result<HashSet<_>, _>>();
            let unique_uuids = bench_result(unique_uuids, "UUID iteration should succeed");
            black_box(unique_uuids);
        });
    });

    group.bench_function("simulated_by_reference", |b| {
        b.iter(|| {
            let uuid_values: Vec<Uuid> = simplex
                .vertices()
                .iter()
                .map(|&vkey| bench_option(dt.tds().vertex(vkey), "vertex should exist").uuid())
                .collect();
            let uuid_refs: Vec<&Uuid> = uuid_values.iter().collect();
            black_box(uuid_refs);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_boundary_facets_micro, bench_vertex_uuid_iter);
criterion_main!(benches);
