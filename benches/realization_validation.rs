//! Criterion benchmarks for Level 4 realization validation.
//!
//! The narrow-phase cases isolate exact shared-face intersection decisions,
//! while the whole-triangulation cases measure the public Level 4 validator on
//! deterministic Euclidean Delaunay triangulations. Fixture construction and
//! invariant checks stay outside Criterion's measured closures.

use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use delaunay::prelude::construction::{DelaunayTriangulation, DelaunayTriangulationBuilder};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{
    AdaptiveKernel, CoordinateRange, ExactPredicates, LabeledSimplexRealization,
    validate_simplex_intersection,
};
use delaunay::try_vertices_from_points;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, abort_benchmark};

const REALISTIC_SEED: u64 = 0x482;
const SEED_SEARCH_LIMIT: u64 = 64;

/// Builds the standard D-simplex with vertices at the origin and coordinate axes.
fn standard_simplex_coordinates<const D: usize>() -> Vec<[f64; D]> {
    let mut coordinates = Vec::with_capacity(D + 1);
    coordinates.push([0.0; D]);
    for axis in 0..D {
        let mut vertex = [0.0; D];
        vertex[axis] = 1.0;
        coordinates.push(vertex);
    }
    coordinates
}

/// Builds simplices whose exact intersection is a non-facet shared face.
fn shared_face_pair<const D: usize>(
    transverse_scale: f64,
) -> (
    LabeledSimplexRealization<usize, D>,
    LabeledSimplexRealization<usize, D>,
) {
    let shared_count = D.saturating_sub(1).max(1);
    let first_coordinates = standard_simplex_coordinates::<D>();
    let mut second_coordinates = first_coordinates[..shared_count].to_vec();
    for axis in shared_count.saturating_sub(1)..D {
        let mut coordinates = [0.0; D];
        coordinates[axis] = -transverse_scale;
        second_coordinates.push(coordinates);
    }

    let first = LabeledSimplexRealization::try_new(0..=D, first_coordinates).or_abort();
    let second_labels = (0..shared_count).chain(D + 1..2 * D + 2 - shared_count);
    let second = LabeledSimplexRealization::try_new(second_labels, second_coordinates).or_abort();
    (first, second)
}

/// Registers one certified shared-face narrow-phase case.
fn register_valid_narrow_phase<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    case: &str,
    transverse_scale: f64,
) {
    let (first, second) = shared_face_pair::<D>(transverse_scale);
    if let Err(error) = validate_simplex_intersection(&first, &second) {
        abort_benchmark(format_args!(
            "{D}D {case} fixture must meet only in its shared face: {error}"
        ));
    }

    group.bench_function(BenchmarkId::new(case, format!("{D}d")), |b| {
        b.iter(|| {
            let _ = black_box(validate_simplex_intersection(
                black_box(&first),
                black_box(&second),
            ));
        });
    });
}

/// Measures ordinary and near-degenerate shared-face decisions in 2D-5D.
fn bench_realization_narrow_phase(c: &mut Criterion) {
    let mut group = c.benchmark_group("realization_narrow_phase");
    group.throughput(Throughput::Elements(1));

    register_valid_narrow_phase::<2>(&mut group, "shared_face_boundary", 1.0);
    register_valid_narrow_phase::<3>(&mut group, "shared_face_boundary", 1.0);
    register_valid_narrow_phase::<4>(&mut group, "shared_face_boundary", 1.0);
    register_valid_narrow_phase::<5>(&mut group, "shared_face_boundary", 1.0);
    register_valid_narrow_phase::<2>(&mut group, "near_degenerate_shared_face", 2.0_f64.powi(-40));
    register_valid_narrow_phase::<3>(&mut group, "near_degenerate_shared_face", 2.0_f64.powi(-40));
    register_valid_narrow_phase::<4>(&mut group, "near_degenerate_shared_face", 2.0_f64.powi(-40));
    register_valid_narrow_phase::<5>(&mut group, "near_degenerate_shared_face", 2.0_f64.powi(-40));

    group.finish();
}

/// Builds one deterministic triangulation that passes Level 4 validation.
fn realistic_triangulation<const D: usize>(
    vertex_count: usize,
) -> DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let bounds = CoordinateRange::try_new(-100.0, 100.0).or_abort();
    for offset in 0..SEED_SEARCH_LIMIT {
        let seed = REALISTIC_SEED.wrapping_add(offset);
        let points =
            generate_random_points_in_range_seeded::<D>(vertex_count, bounds, seed).or_abort();
        let vertices = try_vertices_from_points(&points).or_abort();
        if let Ok(triangulation) = DelaunayTriangulationBuilder::new(&vertices).build()
            && triangulation
                .as_triangulation()
                .validate_realization()
                .is_ok()
        {
            return triangulation;
        }
    }

    abort_benchmark(format_args!(
        "failed to build a valid {D}D realization benchmark fixture after {SEED_SEARCH_LIMIT} seeds"
    ));
}

/// Registers one whole-triangulation Level 4 validation case.
fn register_realistic_validation<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    vertex_count: usize,
) where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let triangulation = realistic_triangulation::<D>(vertex_count);
    let simplex_count = triangulation.number_of_simplices();
    group.throughput(Throughput::Elements(
        u64::try_from(simplex_count).or_abort(),
    ));
    group.bench_function(
        BenchmarkId::new(format!("{D}d"), format!("{vertex_count}v")),
        |b| {
            b.iter(|| {
                let _ =
                    black_box(black_box(triangulation.as_triangulation()).is_valid_realization());
            });
        },
    );
}

/// Measures the public Level 4 validator on realistic 2D-5D triangulations.
fn bench_realization_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("realization_validation");
    register_realistic_validation::<2>(&mut group, 500);
    register_realistic_validation::<3>(&mut group, 20);
    register_realistic_validation::<4>(&mut group, 10);
    register_realistic_validation::<5>(&mut group, 8);
    group.finish();
}

criterion_group!(
    benches,
    bench_realization_narrow_phase,
    bench_realization_validation
);
criterion_main!(benches);
