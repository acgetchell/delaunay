#![forbid(unsafe_code)]

//! Benchmark: `DelaunayTriangulation::delete_vertex` mutation and rollback cost.
//!
//! This benchmark separates vertex-deletion cost from construction cost by building
//! deterministic source triangulations once, then cloning them in Criterion setup
//! before timing the deletion call itself. The timed path still includes the
//! operation's own transactional snapshot and invariant validation, which is the
//! behavior this benchmark is meant to track.
//!
//! The 2D/3D success cases retain fixed seeded workloads and explicit input
//! targets. The 4D/5D success cases measure only inverse-k1 deletion of the
//! interior vertex in a canonical simplex subdivision, using the default public
//! builder, deletion policy, and validation through Levels 1-5. General fan
//! deletion revokes the construction provenance needed to certify larger
//! high-dimensional PL vertex links. The inverse-k1 IDs name this narrower
//! workload and its actual size; they do not represent large-mesh deletion.
//! Minimal-simplex rollback cases remain separate in every dimension.
//!
//! Intended for **manual** runs (not part of the CI performance suite).
//!
//! Run with:
//! ```bash
//! cargo bench --profile perf --bench delete_vertex
//! ```

use std::{hint::black_box, time::Duration};

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::TopologyGuarantee;
use delaunay::prelude::construction::{DelaunayTriangulation, Vertex};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateRange, ExactPredicates, Point};
use delaunay::prelude::tds::VertexKey;
use delaunay::try_vertices_from_points;
use la_stack::Vector;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, OrAbortWithContext, abort_benchmark};

const INTERIOR_RADIUS_MIN: f64 = 0.15;
const INTERIOR_RADIUS_SPAN: f64 = 0.70;
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
const SAMPLE_SIZE: usize = 10;
const WARM_UP_TIME: Duration = Duration::from_millis(500);
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);

type BenchTriangulation<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;

struct DeletionSource<const D: usize> {
    vertex_count: usize,
    simplex_count: usize,
    triangulation: BenchTriangulation<D>,
    vertex_key: VertexKey,
}

#[derive(Clone, Copy)]
enum FixtureKind {
    Interior,
    Cospherical,
    NearDegenerate,
    LargeCoordinate,
}

#[derive(Clone, Copy)]
struct DeletionFixture {
    requested_vertices: usize,
    seed: u64,
    fixture_kind: FixtureKind,
    input_index: usize,
}

impl FixtureKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Interior => "interior",
            Self::Cospherical => "cospherical",
            Self::NearDegenerate => "near_degenerate",
            Self::LargeCoordinate => "large_coordinate",
        }
    }
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

/// Generate points on an affine line in real arithmetic, perturbed by rounding.
///
/// The axis-dependent epsilon tilts the line, and the seed translates it; neither
/// supplies transverse thickness. The full fixture also includes a canonical
/// simplex, so its input set still spans the ambient dimension.
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

    for (direction_coord, coordinate) in direction.iter_mut().zip(point.coords()) {
        *direction_coord = coordinate - COSPHERICAL_CENTER;
    }
    // Coordinates lie in [-0.5, 0.5], so this squared epsilon comparison and
    // its reused square root are representable for the fixture dimensions.
    let norm_squared = Vector::try_new(direction)
        .or_abort()
        .norm_squared()
        .or_abort();

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

/// Build the source triangulation for a fixed successful vertex deletion measurement.
fn build_success_source<const D: usize>(fixture: DeletionFixture) -> DeletionSource<D>
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let DeletionFixture {
        requested_vertices,
        seed,
        fixture_kind,
        input_index,
    } = fixture;
    let vertices = generate_vertices::<D>(requested_vertices, seed, fixture_kind);
    // Pin the input vertex independently of storage order and of which
    // deletions the current implementation accepts.
    let selected_vertex = vertices.get(input_index).or_abort(format_args!(
        "{D}D deletion fixture seed {seed:#x} has no input vertex {input_index}"
    ));
    let triangulation = DelaunayTriangulation::builder(&vertices)
        .build()
        .unwrap_or_else(|error| {
            abort_benchmark(format_args!(
                "{D}D {} deletion fixture seed {seed:#x} failed construction: {error}",
                fixture_kind.label()
            ))
        });
    let vertex_key = triangulation
        .vertex_key_from_uuid(&selected_vertex.uuid())
        .or_abort(format_args!(
            "{D}D deletion fixture seed {seed:#x} lost input vertex {input_index}"
        ));
    let mut probe = triangulation.clone();
    probe.delete_vertex(vertex_key).unwrap_or_else(|error| {
        abort_benchmark(format_args!(
            "{D}D {} deletion fixture seed {seed:#x}, input vertex {input_index} failed deletion: {error}",
            fixture_kind.label()
        ))
    });

    DeletionSource {
        vertex_count: triangulation.number_of_vertices(),
        simplex_count: triangulation.number_of_simplices(),
        triangulation,
        vertex_key,
    }
}

/// Build and validate the exact star used by the 4D/5D inverse-k1 measurements.
fn build_inverse_k1_source<const D: usize>() -> DeletionSource<D>
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let mut points = simplex_points::<D>();
    // Positive coordinates with sum 0.4 in 4D or 0.5 in 5D establish a strictly
    // interior point, with a large margin relative to floating-point rounding.
    points.push(finite_point([0.1; D]));
    let vertices = try_vertices_from_points(&points).or_abort();
    let triangulation: BenchTriangulation<D> =
        DelaunayTriangulation::builder(&vertices).build().or_abort();
    assert_eq!(
        triangulation.topology_guarantee(),
        TopologyGuarantee::PLManifold
    );
    triangulation.validate().or_abort();
    assert_eq!(triangulation.number_of_vertices(), D + 2);
    assert_eq!(triangulation.number_of_simplices(), D + 1);

    let removed_uuid = vertices[D + 1].uuid();
    let vertex_key = triangulation
        .vertex_key_from_uuid(&removed_uuid)
        .or_abort("inverse-k1 fixture lost the selected interior input");
    let canonical_keys: Vec<_> = vertices[..=D]
        .iter()
        .map(|vertex| {
            triangulation
                .vertex_key_from_uuid(&vertex.uuid())
                .or_abort("inverse-k1 fixture lost a canonical input")
        })
        .collect();
    let mut omitted_counts = vec![0_usize; D + 1];
    for (_, simplex) in triangulation.simplices() {
        assert!(simplex.contains_vertex(vertex_key));
        assert_eq!(simplex.number_of_vertices(), D + 1);
        let omitted: Vec<_> = canonical_keys
            .iter()
            .enumerate()
            .filter_map(|(index, &key)| (!simplex.contains_vertex(key)).then_some(index))
            .collect();
        assert_eq!(
            omitted.len(),
            1,
            "inverse-k1 star must cover each outer facet"
        );
        omitted_counts[omitted[0]] += 1;
    }
    assert!(omitted_counts.iter().all(|&count| count == 1));

    let mut probe = triangulation.clone();
    assert_eq!(probe.delete_vertex(vertex_key).or_abort(), D + 1);
    probe.validate().or_abort();
    assert_eq!(probe.number_of_vertices(), D + 1);
    assert_eq!(probe.number_of_simplices(), 1);
    assert!(probe.vertex_key_from_uuid(&removed_uuid).is_none());
    for vertex in &vertices[..=D] {
        let key = probe
            .vertex_key_from_uuid(&vertex.uuid())
            .or_abort("inverse-k1 deletion lost a surviving canonical input");
        assert!(
            probe
                .simplices()
                .all(|(_, simplex)| simplex.contains_vertex(key))
        );
    }

    DeletionSource {
        vertex_count: triangulation.number_of_vertices(),
        simplex_count: triangulation.number_of_simplices(),
        triangulation,
        vertex_key,
    }
}

/// Build the source triangulation for invalid-deletion rollback measurements.
fn build_rollback_source<const D: usize>() -> DeletionSource<D>
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let vertices = simplex_vertices::<D>();
    let triangulation: BenchTriangulation<D> =
        DelaunayTriangulation::builder(&vertices).build().or_abort();
    let vertex_key = triangulation
        .vertices()
        .next()
        .map(|(key, _)| key)
        .or_abort(format!("rollback benchmark simplex has no {D}D vertices"));

    DeletionSource {
        vertex_count: triangulation.number_of_vertices(),
        simplex_count: triangulation.number_of_simplices(),
        triangulation,
        vertex_key,
    }
}

/// Report benchmark throughput in total stored vertices plus simplices.
fn triangulation_element_count<const D: usize>(source: &DeletionSource<D>) -> u64 {
    let total_elements = source.vertex_count + source.simplex_count;
    u64::try_from(total_elements).or_abort()
}

/// Register the successful-deletion cases for one dimension and input-size schedule.
fn bench_success_dimension<const D: usize>(
    c: &mut Criterion,
    dim_label: &str,
    fixtures: &[DeletionFixture],
) where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let mut group = c.benchmark_group(format!("delete_vertex/success/{dim_label}"));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);

    for &fixture in fixtures {
        let source = build_success_source::<D>(fixture);
        group.throughput(Throughput::Elements(triangulation_element_count(&source)));

        group.bench_with_input(
            BenchmarkId::new(
                "delete_vertex",
                format!(
                    "{}_vertices_{}_simplices_{}_seed_{:016x}_input_vertex_{}",
                    fixture.fixture_kind.label(),
                    source.vertex_count,
                    source.simplex_count,
                    fixture.seed,
                    fixture.input_index
                ),
            ),
            &source,
            |b, source| {
                b.iter_batched(
                    || source.triangulation.clone(),
                    |mut triangulation| {
                        black_box(triangulation.delete_vertex(source.vertex_key).or_abort());
                        black_box(triangulation)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Register one fixed inverse-k1 success case with its actual owner size.
fn bench_inverse_k1_dimension<const D: usize>(c: &mut Criterion, dim_label: &str)
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let source = build_inverse_k1_source::<D>();
    let mut group = c.benchmark_group(format!("delete_vertex/success/{dim_label}"));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.throughput(Throughput::Elements(triangulation_element_count(&source)));

    group.bench_with_input(
        BenchmarkId::new(
            "delete_vertex",
            format!(
                "inverse_k1_vertices_{}_simplices_{}_input_vertex_{}",
                source.vertex_count,
                source.simplex_count,
                D + 1
            ),
        ),
        &source,
        |b, source| {
            b.iter_batched(
                || source.triangulation.clone(),
                |mut triangulation| {
                    black_box(triangulation.delete_vertex(source.vertex_key).or_abort());
                    black_box(triangulation)
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

/// Register the minimal-simplex rollback case for one dimension.
fn bench_rollback_dimension<const D: usize>(c: &mut Criterion, dim_label: &str)
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let source = build_rollback_source::<D>();
    let mut group = c.benchmark_group(format!("delete_vertex/rollback/{dim_label}"));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);
    group.throughput(Throughput::Elements(triangulation_element_count(&source)));

    group.bench_with_input(
        BenchmarkId::new(
            "delete_vertex_invalid_remnant",
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
                    black_box(triangulation.delete_vertex(source.vertex_key).unwrap_err());
                    black_box(triangulation)
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

/// Benchmark successful 2D vertex deletion.
fn bench_delete_vertex_success_2d(c: &mut Criterion) {
    bench_success_dimension::<2>(
        c,
        "2d",
        &[
            DeletionFixture {
                requested_vertices: 100,
                seed: 0x1F01_8C77_B918_7835,
                fixture_kind: FixtureKind::Cospherical,
                input_index: 3,
            },
            DeletionFixture {
                requested_vertices: 500,
                seed: 0xD6F3_BE4E_9D7A_5905,
                fixture_kind: FixtureKind::NearDegenerate,
                input_index: 4,
            },
            DeletionFixture {
                requested_vertices: 2_000,
                seed: 0xC3CC_F930_75E9_6411,
                fixture_kind: FixtureKind::LargeCoordinate,
                input_index: 3,
            },
        ],
    );
}

/// Benchmark successful 3D vertex deletion.
fn bench_delete_vertex_success_3d(c: &mut Criterion) {
    bench_success_dimension::<3>(
        c,
        "3d",
        &[
            DeletionFixture {
                requested_vertices: 50,
                seed: 0x357F_C639_DC8C_3C18,
                fixture_kind: FixtureKind::NearDegenerate,
                input_index: 0,
            },
            DeletionFixture {
                requested_vertices: 150,
                seed: 0x672B_52B3_95A4_B44C,
                fixture_kind: FixtureKind::LargeCoordinate,
                input_index: 5,
            },
            DeletionFixture {
                requested_vertices: 500,
                seed: 0xD7F3_BE4F_9D7A_5906,
                fixture_kind: FixtureKind::Interior,
                input_index: 5,
            },
        ],
    );
}

/// Benchmark 4D inverse-k1 vertex deletion.
fn bench_delete_vertex_success_4d(c: &mut Criterion) {
    bench_inverse_k1_dimension::<4>(c, "4d");
}

/// Benchmark 5D inverse-k1 vertex deletion.
fn bench_delete_vertex_success_5d(c: &mut Criterion) {
    bench_inverse_k1_dimension::<5>(c, "5d");
}

/// Benchmark rollback for invalid lower-dimensional remnants.
fn bench_delete_vertex_rollback(c: &mut Criterion) {
    bench_rollback_dimension::<2>(c, "2d");
    bench_rollback_dimension::<3>(c, "3d");
    bench_rollback_dimension::<4>(c, "4d");
    bench_rollback_dimension::<5>(c, "5d");
}

criterion_group!(
    benches,
    bench_delete_vertex_success_2d,
    bench_delete_vertex_success_3d,
    bench_delete_vertex_success_4d,
    bench_delete_vertex_success_5d,
    bench_delete_vertex_rollback
);
criterion_main!(benches);
