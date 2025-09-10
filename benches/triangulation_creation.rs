//! Benchmarks for d-dimensional Delaunay triangulation creation.
#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use delaunay::geometry::util::generate_random_points;
use delaunay::prelude::*;
use delaunay::vertex;
use std::hint::black_box;

// =============================================================================
// TRIANGULATION BENCHMARKS
// =============================================================================

/// Macro to reduce duplication in triangulation creation benchmarks
macro_rules! bench_triangulation_creation {
    ($dim:literal, $func_name:ident) => {
        /// Benchmarks the creation of a D-dimensional Delaunay triangulation with 1,000 vertices.
        fn $func_name(c: &mut Criterion) {
            let points: Vec<Point<f64, $dim>> =
                generate_random_points(1_000, (-100.0, 100.0)).unwrap();
            let mut vertices: Vec<Vertex<f64, (), $dim>> = Vec::with_capacity(points.len());
            vertices.extend(points.iter().map(|p| vertex!(*p)));

            let mut group =
                c.benchmark_group(concat!(stringify!($dim), "d_triangulation_creation"));
            group.throughput(Throughput::Elements(points.len() as u64));
            group.bench_function("triangulation", |b| {
                b.iter(|| {
                    Tds::<f64, (), (), $dim>::new(black_box(&vertices)).unwrap();
                });
            });
            group.finish();
        }
    };
}

// Generate benchmark functions using the macro
bench_triangulation_creation!(2, bench_triangulation_creation_2d);
bench_triangulation_creation!(3, bench_triangulation_creation_3d);
bench_triangulation_creation!(4, bench_triangulation_creation_4d);
bench_triangulation_creation!(5, bench_triangulation_creation_5d);

criterion_group!(
    benches,
    bench_triangulation_creation_2d,
    bench_triangulation_creation_3d,
    bench_triangulation_creation_4d,
    bench_triangulation_creation_5d
);
criterion_main!(benches);
