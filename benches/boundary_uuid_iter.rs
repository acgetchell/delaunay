#![forbid(unsafe_code)]

//! Focused microbenchmarks for boundary-facet traversal and simplex UUID iteration.
//!
//! These cases used to live as `#[cfg(feature = "bench")]` unit tests. Keeping
//! them in a Criterion harness avoids creating a third test bucket while
//! preserving the quick performance probes.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::construction::{DelaunayTriangulation, vertex};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateRange};
use delaunay::prelude::query::FacetIncidenceAnalysis;
use delaunay::try_vertices_from_points;
use uuid::Uuid;

use std::collections::HashSet;
use std::hint::black_box;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, OrAbortWithContext};

const BOUNDARY_COUNTS_3D: &[usize] = &[20, 40, 60, 80];

fn benchmark_bounds() -> CoordinateRange<f64> {
    CoordinateRange::try_new(-100.0_f64, 100.0).or_abort()
}

fn boundary_triangulation_3d(
    requested_vertices: usize,
) -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3> {
    let points = generate_random_points_in_range_seeded::<3>(
        requested_vertices,
        benchmark_bounds(),
        0xB0DA_FACE_0000_0000 ^ requested_vertices as u64,
    );
    let points = points.or_abort();
    let vertices = try_vertices_from_points(&points).or_abort();
    DelaunayTriangulation::try_new(&vertices).or_abort()
}

fn bench_boundary_facets_micro(c: &mut Criterion) {
    let mut group = c.benchmark_group("boundary_facets_micro");

    for &requested_vertices in BOUNDARY_COUNTS_3D {
        let dt = boundary_triangulation_3d(requested_vertices);
        let boundary_count = dt
            .boundary_facets()
            .or_abort()
            .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
            .or_abort();
        group.throughput(Throughput::Elements(
            u64::try_from(boundary_count).or_abort(),
        ));

        group.bench_with_input(
            BenchmarkId::new("boundary_facets_count_3d", requested_vertices),
            &dt,
            |b, dt| {
                b.iter(|| {
                    black_box(
                        dt.boundary_facets()
                            .or_abort()
                            .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                            .or_abort(),
                    );
                });
            },
        );

        let boundary_facets = dt
            .boundary_facets()
            .or_abort()
            .collect::<Result<Vec<_>, _>>()
            .or_abort();
        group.bench_with_input(
            BenchmarkId::new("is_one_sided_facet_3d", requested_vertices),
            &(&dt, boundary_facets),
            |b, (dt, facets)| {
                b.iter(|| {
                    let confirmed = facets
                        .iter()
                        .filter(|facet| dt.tds().is_one_sided_facet(facet).or_abort())
                        .count();
                    black_box(confirmed);
                });
            },
        );
    }

    group.finish();
}

fn uuid_iter_source() -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), 3> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0].or_abort(),
        vertex![1.0, 0.0, 0.0].or_abort(),
        vertex![0.0, 1.0, 0.0].or_abort(),
        vertex![0.0, 0.0, 1.0].or_abort(),
    ];
    DelaunayTriangulation::try_new(&vertices).or_abort()
}

fn bench_vertex_uuid_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertex_uuid_iter");
    let dt = uuid_iter_source();
    let (_simplex_key, simplex) = dt
        .simplices()
        .next()
        .or_abort("simplex should exist for UUID iterator benchmark");

    group.throughput(Throughput::Elements(
        u64::try_from(simplex.vertices().len()).or_abort(),
    ));

    group.bench_function("by_value", |b| {
        b.iter(|| {
            let unique_uuids = simplex
                .vertex_uuid_iter(dt.tds())
                .collect::<Result<HashSet<_>, _>>();
            let unique_uuids = unique_uuids.or_abort();
            black_box(unique_uuids);
        });
    });

    group.bench_function("simulated_by_reference", |b| {
        b.iter(|| {
            let uuid_values: Vec<Uuid> = simplex
                .vertices()
                .iter()
                .map(|&vkey| dt.tds().vertex(vkey).or_abort("vertex should exist").uuid())
                .collect();
            let uuid_refs: Vec<&Uuid> = uuid_values.iter().collect();
            black_box(uuid_refs);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_boundary_facets_micro, bench_vertex_uuid_iter);
criterion_main!(benches);
