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
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateRange, Point};
use delaunay::prelude::tds::VertexKey;
use delaunay::try_vertices_from_points;
use std::hint::black_box;
use std::time::Duration;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, OrAbortWithContext, abort_benchmark};

const INTERIOR_RADIUS_MIN: f64 = 0.15;
const INTERIOR_RADIUS_SPAN: f64 = 0.70;
const NEAR_BOUNDARY_EPSILON: f64 = 1.0e-9;
const NEAR_DEGENERATE_EPSILON: f64 = 1.0e-10;
const COSPHERICAL_CENTER: f64 = 0.5;
const COSPHERICAL_RADIUS: f64 = 0.25;
const LARGE_COORDINATE_SCALE: f64 = 1.0e6;

fn finite_point<const D: usize>(coords: [f64; D]) -> Point<D> {
    Point::try_new(coords).unwrap_or_else(|_| std::process::abort())
}

fn interior_bounds() -> CoordinateRange<f64> {
    CoordinateRange::try_new(0.0_f64, 1.0).or_abort()
}
const LARGE_COORDINATE_JITTER: f64 = 1.0e3;
const SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
const SEED_SEARCH_ATTEMPTS: usize = 64;
const SAMPLE_SIZE: usize = 10;
const WARM_UP_TIME: Duration = Duration::from_millis(500);
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);

type BenchTriangulation<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;

struct RemovalSource<const D: usize> {
    vertex_count: usize,
    simplex_count: usize,
    fixture_kind: FixtureKind,
    triangulation: BenchTriangulation<D>,
    vertex_key: VertexKey,
}

#[derive(Clone, Copy)]
enum FixtureKind {
    Interior,
    NearBoundary,
    Cospherical,
    NearDegenerate,
    LargeCoordinate,
}

const FIXTURE_KINDS: [FixtureKind; 5] = [
    FixtureKind::Interior,
    FixtureKind::NearBoundary,
    FixtureKind::Cospherical,
    FixtureKind::NearDegenerate,
    FixtureKind::LargeCoordinate,
];

impl FixtureKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Interior => "interior",
            Self::NearBoundary => "near_boundary",
            Self::Cospherical => "cospherical",
            Self::NearDegenerate => "near_degenerate",
            Self::LargeCoordinate => "large_coordinate",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Interior => 0,
            Self::NearBoundary => 1,
            Self::Cospherical => 2,
            Self::NearDegenerate => 3,
            Self::LargeCoordinate => 4,
        }
    }
}

/// Derive a deterministic, dimension-specific seed for one benchmark case.
fn seed_for_case<const D: usize>(requested_vertices: usize, seed_base: u64) -> u64 {
    let vertices = u64::try_from(requested_vertices).or_abort();
    let dimension = u64::try_from(D).or_abort();
    seed_base ^ vertices.wrapping_mul(SEED_SALT) ^ dimension.rotate_left(32)
}

/// Pick the preferred geometric fixture for one benchmark case.
fn preferred_fixture_kind<const D: usize>(case_index: usize) -> FixtureKind {
    FIXTURE_KINDS[(D + case_index) % FIXTURE_KINDS.len()]
}

/// Cycle through all fixture kinds during seed search, starting with the preferred kind.
const fn fixture_kind_for_attempt(preferred_kind: FixtureKind, attempt: usize) -> FixtureKind {
    FIXTURE_KINDS[(preferred_kind.index() + attempt) % FIXTURE_KINDS.len()]
}

/// Convert a bounded benchmark index to `f64` without unchecked casts.
fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).or_abort())
}

/// Generate a reproducible canonical simplex with selected adversarial points.
fn generate_vertices<const D: usize>(
    requested_vertices: usize,
    seed: u64,
    fixture_kind: FixtureKind,
) -> Vec<Vertex<(), D>> {
    let generated_count = requested_vertices.saturating_sub(D + 1);
    let mut points = simplex_points::<D>();
    let generated_points = match fixture_kind {
        FixtureKind::Interior => generate_interior_points(generated_count, seed),
        FixtureKind::NearBoundary => generate_near_boundary_points(generated_count, seed),
        FixtureKind::Cospherical => generate_cospherical_points(generated_count, seed),
        FixtureKind::NearDegenerate => generate_near_degenerate_simplex(generated_count, seed),
        FixtureKind::LargeCoordinate => generate_large_coordinate_points(generated_count, seed),
    };

    points.extend(generated_points);
    try_vertices_from_points(&points).or_abort()
}

/// Generate well-conditioned interior points inside the canonical simplex.
fn generate_interior_points<const D: usize>(count: usize, seed: u64) -> Vec<Point<D>> {
    let raw_points =
        generate_random_points_in_range_seeded::<D>(count, interior_bounds(), seed).or_abort();
    let mut points = Vec::with_capacity(count);

    for (index, raw_point) in raw_points.iter().enumerate() {
        let radius = interior_radius(index);
        let direction = normalized_positive_direction(raw_point);
        let mut coords = [0.0; D];
        for (coord, direction_coord) in coords.iter_mut().zip(direction) {
            *coord = radius * direction_coord;
        }
        points.push(finite_point(coords));
    }

    points
}

/// Generate points close to coordinate-boundary facets of the canonical simplex.
fn generate_near_boundary_points<const D: usize>(count: usize, seed: u64) -> Vec<Point<D>> {
    let raw_points =
        generate_random_points_in_range_seeded::<D>(count, interior_bounds(), seed).or_abort();
    let mut points = Vec::with_capacity(count);

    for (index, raw_point) in raw_points.iter().enumerate() {
        let mut coords = [0.0; D];
        let direction = normalized_positive_direction(raw_point);
        let near_boundary_axis = index % D;
        for (coord, direction_coord) in coords.iter_mut().zip(direction) {
            *coord = 0.98 * direction_coord;
        }
        coords[near_boundary_axis] = NEAR_BOUNDARY_EPSILON * usize_to_f64(index + 1);
        points.push(finite_point(coords));
    }

    points
}

/// Generate points on a shared sphere to stress cospherical predicates.
fn generate_cospherical_points<const D: usize>(count: usize, seed: u64) -> Vec<Point<D>> {
    let raw_points =
        generate_random_points_in_range_seeded::<D>(count, interior_bounds(), seed).or_abort();
    let mut points = Vec::with_capacity(count);

    for raw_point in &raw_points {
        let direction = centered_unit_direction(raw_point);
        let mut coords = [0.0; D];
        for (coord, direction_coord) in coords.iter_mut().zip(direction) {
            *coord = COSPHERICAL_RADIUS.mul_add(direction_coord, COSPHERICAL_CENTER);
        }
        points.push(finite_point(coords));
    }

    points
}

/// Generate points close to a lower-dimensional diagonal simplex.
fn generate_near_degenerate_simplex<const D: usize>(count: usize, seed: u64) -> Vec<Point<D>> {
    let seed_offset = f64::from(u32::try_from(seed % 997).or_abort()) * 1.0e-14;
    let denominator = usize_to_f64(count + 1);
    let mut points = Vec::with_capacity(count);

    for index in 0..count {
        let index_factor = usize_to_f64(index + 1);
        let diagonal = index_factor / denominator;
        let mut coords = [0.0; D];
        for (axis, coord) in coords.iter_mut().enumerate() {
            let axis_factor = usize_to_f64(axis + 1);
            *coord = (NEAR_DEGENERATE_EPSILON * axis_factor)
                .mul_add(index_factor, diagonal + seed_offset);
        }
        points.push(finite_point(coords));
    }

    points
}

/// Generate finite points with large coordinates to stress scale-sensitive paths.
fn generate_large_coordinate_points<const D: usize>(count: usize, seed: u64) -> Vec<Point<D>> {
    let raw_points =
        generate_random_points_in_range_seeded::<D>(count, interior_bounds(), seed).or_abort();
    let mut points = Vec::with_capacity(count);

    for (index, raw_point) in raw_points.iter().enumerate() {
        let index_offset = usize_to_f64(index + 1);
        let mut coords = [0.0; D];
        for (axis, (coord, raw_coord)) in coords.iter_mut().zip(raw_point.coords()).enumerate() {
            let axis_factor = usize_to_f64(axis + 1);
            *coord = LARGE_COORDINATE_SCALE.mul_add(
                axis_factor,
                LARGE_COORDINATE_JITTER.mul_add(*raw_coord, index_offset),
            );
        }
        points.push(finite_point(coords));
    }

    points
}

/// Generate the minimal full-dimensional simplex points.
fn simplex_points<const D: usize>() -> Vec<Point<D>> {
    let mut points = Vec::with_capacity(D + 1);
    points.push(finite_point([0.0; D]));

    for axis in 0..D {
        let mut coords = [0.0; D];
        coords[axis] = 1.0;
        points.push(finite_point(coords));
    }

    points
}

/// Generate the minimal full-dimensional simplex for the rollback benchmark.
fn simplex_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
    try_vertices_from_points(&simplex_points::<D>()).or_abort()
}

/// Deterministic radial coordinate for a point inside the canonical simplex.
fn interior_radius(index: usize) -> f64 {
    let numerator = u32::try_from(index.wrapping_mul(37) % 997).or_abort();
    INTERIOR_RADIUS_MIN + INTERIOR_RADIUS_SPAN * f64::from(numerator) / 997.0
}

/// Convert a random point in `[0, 1]^D` into a positive simplex direction.
fn normalized_positive_direction<const D: usize>(point: &Point<D>) -> [f64; D] {
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

/// Convert a random point in `[0, 1]^D` into a unit direction around the origin.
fn centered_unit_direction<const D: usize>(point: &Point<D>) -> [f64; D] {
    let mut direction = [0.0; D];
    let mut norm_squared = 0.0;

    for (direction_coord, coordinate) in direction.iter_mut().zip(point.coords()) {
        *direction_coord = coordinate - COSPHERICAL_CENTER;
        norm_squared = (*direction_coord).mul_add(*direction_coord, norm_squared);
    }

    if norm_squared <= f64::EPSILON {
        direction[0] = 1.0;
        return direction;
    }

    let norm = norm_squared.sqrt();
    for direction_coord in &mut direction {
        *direction_coord /= norm;
    }

    direction
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
    preferred_kind: FixtureKind,
) -> RemovalSource<D> {
    for attempt in 0..SEED_SEARCH_ATTEMPTS {
        let attempt_seed = u64::try_from(attempt).or_abort();
        let seed = seed_for_case::<D>(requested_vertices, seed_base)
            ^ attempt_seed.wrapping_mul(SEED_SALT.rotate_left(17));
        let fixture_kind = fixture_kind_for_attempt(preferred_kind, attempt);
        let vertices = generate_vertices::<D>(requested_vertices, seed, fixture_kind);
        let Ok(triangulation) = DelaunayTriangulation::try_new(&vertices) else {
            continue;
        };
        let Some(vertex_key) = successful_removal_vertex(&triangulation) else {
            continue;
        };

        return RemovalSource {
            vertex_count: triangulation.number_of_vertices(),
            simplex_count: triangulation.number_of_simplices(),
            fixture_kind,
            triangulation,
            vertex_key,
        };
    }

    abort_benchmark(format!(
        "no successful {D}D remove_vertex fixture found for {requested_vertices} vertices \
             after {SEED_SEARCH_ATTEMPTS} seeds across all fixture kinds"
    ))
}

/// Build the source triangulation for invalid-removal rollback measurements.
fn build_rollback_source<const D: usize>() -> RemovalSource<D> {
    let vertices = simplex_vertices::<D>();
    let triangulation: BenchTriangulation<D> = DelaunayTriangulation::try_new(&vertices).or_abort();
    let vertex_key = triangulation
        .vertices()
        .next()
        .map(|(key, _)| key)
        .or_abort(format!("rollback benchmark simplex has no {D}D vertices"));

    RemovalSource {
        vertex_count: triangulation.number_of_vertices(),
        simplex_count: triangulation.number_of_simplices(),
        fixture_kind: FixtureKind::Interior,
        triangulation,
        vertex_key,
    }
}

/// Report benchmark throughput in total stored vertices plus simplices.
fn triangulation_element_count<const D: usize>(source: &RemovalSource<D>) -> u64 {
    let total_elements = source.vertex_count + source.simplex_count;
    u64::try_from(total_elements).or_abort()
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

    for (case_index, &requested_vertices) in counts.iter().enumerate() {
        let source = build_success_source::<D>(
            requested_vertices,
            seed_base,
            preferred_fixture_kind::<D>(case_index),
        );
        group.throughput(Throughput::Elements(triangulation_element_count(&source)));

        group.bench_with_input(
            BenchmarkId::new(
                "remove_vertex",
                format!(
                    "{}_vertices_{}_simplices_{}",
                    source.fixture_kind.label(),
                    source.vertex_count,
                    source.simplex_count
                ),
            ),
            &source,
            |b, source| {
                b.iter_batched(
                    || source.triangulation.clone(),
                    |mut triangulation| {
                        black_box(triangulation.remove_vertex(source.vertex_key).or_abort());
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
