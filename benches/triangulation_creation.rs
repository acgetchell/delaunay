//! Benchmarks for d-dimensional Delaunay triangulation creation.
#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use delaunay::prelude::*;
use delaunay::vertex;
use rand::Rng;
use std::hint::black_box;

// =============================================================================
// TRIANGULATION BENCHMARKS
// =============================================================================

/// Benchmarks the creation of a 2D Delaunay triangulation with 1,000 vertices.
fn bench_triangulation_creation_2d(c: &mut Criterion) {
    let mut rng = rand::rng();
    let points: Vec<Point<f64, 2>> = (0..1_000)
        .map(|_| {
            Point::new([
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
            ])
        })
        .collect();
    let vertices: Vec<Vertex<f64, (), 2>> = points.iter().map(|p| vertex!(*p)).collect();

    c.bench_function("2d_triangulation_creation", |b| {
        b.iter(|| {
            Tds::<f64, (), (), 2>::new(black_box(&vertices)).unwrap();
        });
    });
}

/// Benchmarks the creation of a 3D Delaunay triangulation with 1,000 vertices.
fn bench_triangulation_creation_3d(c: &mut Criterion) {
    let mut rng = rand::rng();
    let points: Vec<Point<f64, 3>> = (0..1_000)
        .map(|_| {
            Point::new([
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
            ])
        })
        .collect();
    let vertices: Vec<Vertex<f64, (), 3>> = points.iter().map(|p| vertex!(*p)).collect();

    c.bench_function("3d_triangulation_creation", |b| {
        b.iter(|| {
            Tds::<f64, (), (), 3>::new(black_box(&vertices)).unwrap();
        });
    });
}

/// Benchmarks the creation of a 4D Delaunay triangulation with 1,000 vertices.
fn bench_triangulation_creation_4d(c: &mut Criterion) {
    let mut rng = rand::rng();
    let points: Vec<Point<f64, 4>> = (0..1_000)
        .map(|_| {
            Point::new([
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
            ])
        })
        .collect();
    let vertices: Vec<Vertex<f64, (), 4>> = points.iter().map(|p| vertex!(*p)).collect();

    c.bench_function("4d_triangulation_creation", |b| {
        b.iter(|| {
            Tds::<f64, (), (), 4>::new(black_box(&vertices)).unwrap();
        });
    });
}

criterion_group!(
    benches,
    bench_triangulation_creation_2d,
    bench_triangulation_creation_3d,
    bench_triangulation_creation_4d
);
criterion_main!(benches);
