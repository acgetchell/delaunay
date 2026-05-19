#![forbid(unsafe_code)]

//! Benchmark: `DelaunayTriangulation::remove_vertex` mutation and rollback cost.
//!
//! This benchmark separates vertex-removal cost from construction cost by building
//! deterministic source triangulations once, then cloning them in Criterion setup
//! before timing the removal call itself. The timed path still includes the
//! operation's own transactional snapshot and invariant validation, which is the
//! behavior this benchmark is meant to track.
//!
//! Intended for **manual** runs (not part of the CI performance suite).
//!
//! Run with:
//! ```bash
//! cargo bench --profile perf --bench remove_vertex
//! ```

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::construction::{DelaunayTriangulation, Vertex};
use delaunay::prelude::generators::generate_random_points_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, Coordinate, Point};
use delaunay::prelude::tds::VertexKey;
use std::hint::black_box;
use std::time::Duration;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{bench_option, bench_result};

const INTERIOR_BOUNDS: (f64, f64) = (0.0, 1.0);
const INTERIOR_RADIUS_MIN: f64 = 0.15;
const INTERIOR_RADIUS_SPAN: f64 = 0.70;
const SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
const SEED_SEARCH_ATTEMPTS: usize = 64;
const SAMPLE_SIZE: usize = 10;
const WARM_UP_TIME: Duration = Duration::from_millis(500);
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);

type BenchTriangulation<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;

struct RemovalSource<const D: usize> {
    vertex_count: usize,
    simplex_count: usize,
    triangulation: BenchTriangulation<D>,
    vertex_key: VertexKey,
}

/// Derive a deterministic, dimension-specific seed for one benchmark case.
fn seed_for_case<const D: usize>(requested_vertices: usize, seed_base: u64) -> u64 {
    let vertices = bench_result(
        u64::try_from(requested_vertices),
        "vertex count does not fit in u64",
    );
    let dimension = bench_result(u64::try_from(D), "dimension does not fit in u64");
    seed_base ^ vertices.wrapping_mul(SEED_SALT) ^ dimension.rotate_left(32)
}

/// Generate a reproducible canonical simplex with interior points.
fn generate_vertices<const D: usize>(
    requested_vertices: usize,
    seed: u64,
) -> Vec<Vertex<f64, (), D>> {
    let interior_count = requested_vertices.saturating_sub(D + 1);
    let mut points = simplex_points::<D>();
    let raw_points = bench_result(
        generate_random_points_seeded::<f64, D>(interior_count, INTERIOR_BOUNDS, seed),
        format!("failed to generate {D}D interior benchmark points"),
    );

    for (index, raw_point) in raw_points.iter().enumerate() {
        let radius = interior_radius(index);
        let direction = normalized_positive_direction(raw_point);
        let mut coords = [0.0; D];
        for (coord, direction_coord) in coords.iter_mut().zip(direction) {
            *coord = radius * direction_coord;
        }
        points.push(Point::new(coords));
    }

    Vertex::from_points(&points)
}

/// Generate the minimal full-dimensional simplex points.
fn simplex_points<const D: usize>() -> Vec<Point<f64, D>> {
    let mut points = Vec::with_capacity(D + 1);
    points.push(Point::new([0.0; D]));

    for axis in 0..D {
        let mut coords = [0.0; D];
        coords[axis] = 1.0;
        points.push(Point::new(coords));
    }

    points
}

/// Generate the minimal full-dimensional simplex for the rollback benchmark.
fn simplex_vertices<const D: usize>() -> Vec<Vertex<f64, (), D>> {
    Vertex::from_points(&simplex_points::<D>())
}

/// Deterministic radial coordinate for a point inside the canonical simplex.
fn interior_radius(index: usize) -> f64 {
    let numerator = bench_result(
        u32::try_from(index.wrapping_mul(37) % 997),
        "interior radius numerator does not fit in u32",
    );
    INTERIOR_RADIUS_MIN + INTERIOR_RADIUS_SPAN * f64::from(numerator) / 997.0
}

/// Convert a random point in `[0, 1]^D` into a positive simplex direction.
fn normalized_positive_direction<const D: usize>(point: &Point<f64, D>) -> [f64; D] {
    let mut weights = [0.0; D];
    let mut weight_sum = 0.0;

    for (weight, coordinate) in weights.iter_mut().zip(point.coords()) {
        *weight = coordinate + f64::EPSILON;
        weight_sum += *weight;
    }

    for weight in &mut weights {
        *weight /= weight_sum;
    }

    weights
}

/// Find a vertex whose removal succeeds for the prepared triangulation.
fn successful_removal_vertex<const D: usize>(
    triangulation: &BenchTriangulation<D>,
) -> Option<VertexKey> {
    for (vertex_key, _) in triangulation.vertices() {
        let mut candidate = triangulation.clone();
        if candidate.remove_vertex(vertex_key).is_ok() {
            return Some(vertex_key);
        }
    }

    None
}

/// Build the source triangulation for successful interior-removal measurements.
fn build_success_source<const D: usize>(
    requested_vertices: usize,
    seed_base: u64,
) -> RemovalSource<D> {
    for attempt in 0..SEED_SEARCH_ATTEMPTS {
        let attempt_seed = bench_result(u64::try_from(attempt), "seed attempt does not fit in u64");
        let seed = seed_for_case::<D>(requested_vertices, seed_base)
            ^ attempt_seed.wrapping_mul(SEED_SALT.rotate_left(17));
        let vertices = generate_vertices::<D>(requested_vertices, seed);
        let Ok(triangulation) = DelaunayTriangulation::new(&vertices) else {
            continue;
        };
        let Some(vertex_key) = successful_removal_vertex(&triangulation) else {
            continue;
        };

        return RemovalSource {
            vertex_count: triangulation.number_of_vertices(),
            simplex_count: triangulation.number_of_simplices(),
            triangulation,
            vertex_key,
        };
    }

    bench_option(
        None,
        format!(
            "no successful {D}D remove_vertex fixture found for {requested_vertices} vertices \
             after {SEED_SEARCH_ATTEMPTS} seeds"
        ),
    )
}

/// Build the source triangulation for invalid-removal rollback measurements.
fn build_rollback_source<const D: usize>() -> RemovalSource<D> {
    let vertices = simplex_vertices::<D>();
    let triangulation: BenchTriangulation<D> = bench_result(
        DelaunayTriangulation::new(&vertices),
        format!("failed to build {D}D rollback benchmark simplex"),
    );
    let vertex_key = bench_option(
        triangulation.vertices().next().map(|(key, _)| key),
        format!("rollback benchmark simplex has no {D}D vertices"),
    );

    RemovalSource {
        vertex_count: triangulation.number_of_vertices(),
        simplex_count: triangulation.number_of_simplices(),
        triangulation,
        vertex_key,
    }
}

/// Report benchmark throughput in total stored vertices plus simplices.
fn triangulation_element_count<const D: usize>(source: &RemovalSource<D>) -> u64 {
    let total_elements = source.vertex_count + source.simplex_count;
    bench_result(
        u64::try_from(total_elements),
        "triangulation element count does not fit in u64",
    )
}

/// Register the successful-removal cases for one dimension and input-size schedule.
fn bench_success_dimension<const D: usize>(
    c: &mut Criterion,
    dim_label: &str,
    counts: &[usize],
    seed_base: u64,
) {
    let mut group = c.benchmark_group(format!("remove_vertex/success/{dim_label}"));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);

    for &requested_vertices in counts {
        let source = build_success_source::<D>(requested_vertices, seed_base);
        group.throughput(Throughput::Elements(triangulation_element_count(&source)));

        group.bench_with_input(
            BenchmarkId::new(
                "remove_vertex",
                format!(
                    "vertices_{}_simplices_{}",
                    source.vertex_count, source.simplex_count
                ),
            ),
            &source,
            |b, source| {
                b.iter_batched(
                    || source.triangulation.clone(),
                    |mut triangulation| {
                        black_box(bench_result(
                            triangulation.remove_vertex(source.vertex_key),
                            "successful remove_vertex benchmark unexpectedly failed",
                        ));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Register the minimal-simplex rollback case for one dimension.
fn bench_rollback_dimension<const D: usize>(c: &mut Criterion, dim_label: &str) {
    let source = build_rollback_source::<D>();
    let mut group = c.benchmark_group(format!("remove_vertex/rollback/{dim_label}"));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.throughput(Throughput::Elements(triangulation_element_count(&source)));

    group.bench_with_input(
        BenchmarkId::new(
            "remove_vertex_invalid_remnant",
            format!(
                "vertices_{}_simplices_{}",
                source.vertex_count, source.simplex_count
            ),
        ),
        &source,
        |b, source| {
            b.iter_batched(
                || source.triangulation.clone(),
                |mut triangulation| {
                    black_box(triangulation.remove_vertex(source.vertex_key).unwrap_err());
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

/// Benchmark successful 2D vertex removal.
fn bench_remove_vertex_success_2d(c: &mut Criterion) {
    bench_success_dimension::<2>(c, "2d", &[100, 500, 2_000], 0xD2AA_0000_0000_0001);
}

/// Benchmark successful 3D vertex removal.
fn bench_remove_vertex_success_3d(c: &mut Criterion) {
    bench_success_dimension::<3>(c, "3d", &[50, 150, 500], 0xD3AA_0000_0000_0002);
}

/// Benchmark successful 4D vertex removal.
fn bench_remove_vertex_success_4d(c: &mut Criterion) {
    bench_success_dimension::<4>(c, "4d", &[20, 50, 100], 0xD4AA_0000_0000_0003);
}

/// Benchmark successful 5D vertex removal.
fn bench_remove_vertex_success_5d(c: &mut Criterion) {
    bench_success_dimension::<5>(c, "5d", &[12, 25, 40], 0xD5AA_0000_0000_0004);
}

/// Benchmark rollback for invalid lower-dimensional remnants.
fn bench_remove_vertex_rollback(c: &mut Criterion) {
    bench_rollback_dimension::<2>(c, "2d");
    bench_rollback_dimension::<3>(c, "3d");
    bench_rollback_dimension::<4>(c, "4d");
    bench_rollback_dimension::<5>(c, "5d");
}

criterion_group!(
    benches,
    bench_remove_vertex_success_2d,
    bench_remove_vertex_success_3d,
    bench_remove_vertex_success_4d,
    bench_remove_vertex_success_5d,
    bench_remove_vertex_rollback
);
criterion_main!(benches);
