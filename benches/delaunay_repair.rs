#![forbid(unsafe_code)]

//! Benchmark: flip-based Delaunay repair (`repair_delaunay_with_flips_advanced`).
//!
//! Fixtures are built with
//! [`ConstructionOptions::without_final_delaunay_enforcement`], which returns a
//! valid Levels 1-4 triangulation without running batch or final Level 5
//! repair, so well-conditioned random fixtures typically still carry Delaunay
//! violations. The benchmark then times the public repair entry point on a
//! clone of each prepared fixture.
//!
//! Setup verifies on a throwaway clone that repair converges for the chosen
//! seed, so the measured closure never times a repair failure chain. Case
//! labels record whether the prepared fixture was `violating` or already
//! `delaunay`, so baseline comparisons notice when a fixture changes meaning.
//! Exactly-cospherical adversarial fixtures are deliberately excluded: strict
//! flip repair can legitimately fail to converge on them, which is a
//! correctness scenario rather than a stable performance contract.
//!
//! Intended for **manual** runs (not part of the CI performance suite).
//!
//! Run with:
//! ```bash
//! cargo bench --profile perf --bench delaunay_repair
//! ```

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayTriangulation, DelaunayTriangulationBuilder, Vertex,
};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{
    AdaptiveKernel, CoordinateRange, ExactPredicates, Kernel, Point,
};
use delaunay::prelude::repair::DelaunayRepairHeuristicConfig;
use delaunay::try_vertices_from_points;
use std::hint::black_box;
use std::process;
use std::time::Duration;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, abort_benchmark};

const SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
const SEED_SEARCH_ATTEMPTS: usize = 16;
const SAMPLE_SIZE: usize = 10;
const WARM_UP_TIME: Duration = Duration::from_millis(500);
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);

type BenchTriangulation<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;

struct RepairSource<const D: usize> {
    vertex_count: usize,
    simplex_count: usize,
    violating: bool,
    triangulation: BenchTriangulation<D>,
}

/// Construct a finite point or abort benchmark setup.
fn finite_point<const D: usize>(coords: [f64; D]) -> Point<D> {
    Point::try_new(coords).unwrap_or_else(|_| process::abort())
}

/// Coordinate range used for raw generated points.
fn point_bounds() -> CoordinateRange<f64> {
    CoordinateRange::try_new(0.0_f64, 1.0).or_abort()
}

/// Derive a deterministic, dimension-specific seed for one benchmark case.
fn seed_for_case<const D: usize>(requested_vertices: usize, seed_base: u64) -> u64 {
    let vertices = u64::try_from(requested_vertices).or_abort();
    let dimension = u64::try_from(D).or_abort();
    seed_base ^ vertices.wrapping_mul(SEED_SALT) ^ dimension.rotate_left(32)
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

/// Generate a reproducible fixture vertex set.
fn generate_vertices<const D: usize>(requested_vertices: usize, seed: u64) -> Vec<Vertex<(), D>> {
    let generated_count = requested_vertices.saturating_sub(D + 1);
    let mut points = simplex_points::<D>();
    points.extend(
        generate_random_points_in_range_seeded::<D>(generated_count, point_bounds(), seed)
            .or_abort(),
    );
    try_vertices_from_points(&points).or_abort()
}

/// Build one repair fixture whose repair is verified to converge.
///
/// Seeds are searched until construction succeeds and a throwaway clone
/// completes `repair_delaunay_with_flips_advanced`, so the measured closure
/// never times a repair failure chain.
fn build_source<const D: usize>(requested_vertices: usize, seed_base: u64) -> RepairSource<D>
where
    AdaptiveKernel<f64>: ExactPredicates<D> + Kernel<D, Scalar = f64>,
{
    let options = ConstructionOptions::default().without_final_delaunay_enforcement();

    for attempt in 0..SEED_SEARCH_ATTEMPTS {
        let attempt_seed = u64::try_from(attempt).or_abort();
        let seed = seed_for_case::<D>(requested_vertices, seed_base)
            ^ attempt_seed.wrapping_mul(SEED_SALT.rotate_left(17));
        let vertices = generate_vertices::<D>(requested_vertices, seed);
        let Ok(triangulation) = DelaunayTriangulationBuilder::new(&vertices)
            .construction_options(options)
            .build()
        else {
            continue;
        };

        let mut probe = triangulation.clone();
        if probe
            .repair_delaunay_with_flips_advanced(DelaunayRepairHeuristicConfig::default())
            .is_err()
        {
            continue;
        }

        let violating = triangulation.is_delaunay_via_flips().is_err();
        return RepairSource {
            vertex_count: triangulation.number_of_vertices(),
            simplex_count: triangulation.number_of_simplices(),
            violating,
            triangulation,
        };
    }

    abort_benchmark(format!(
        "no repair-convergent {D}D fixture built for {requested_vertices} vertices \
         after {SEED_SEARCH_ATTEMPTS} seeds"
    ))
}

/// Report benchmark throughput in total stored vertices plus simplices.
fn triangulation_element_count<const D: usize>(source: &RepairSource<D>) -> u64 {
    let total_elements = source.vertex_count + source.simplex_count;
    u64::try_from(total_elements).or_abort()
}

/// Register the repair cases for one dimension and input-size schedule.
fn bench_repair_dimension<const D: usize>(
    c: &mut Criterion,
    dim_label: &str,
    counts: &[usize],
    seed_base: u64,
) where
    AdaptiveKernel<f64>: ExactPredicates<D> + Kernel<D, Scalar = f64>,
{
    let mut group = c.benchmark_group(format!("delaunay_repair/{dim_label}"));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);

    for &requested_vertices in counts {
        let source = build_source::<D>(requested_vertices, seed_base);
        let delaunay_state = if source.violating {
            "violating"
        } else {
            "delaunay"
        };
        group.throughput(Throughput::Elements(triangulation_element_count(&source)));

        group.bench_with_input(
            BenchmarkId::new(
                "repair_advanced",
                format!(
                    "{delaunay_state}_vertices_{}_simplices_{}",
                    source.vertex_count, source.simplex_count
                ),
            ),
            &source,
            |b, source| {
                b.iter_batched(
                    || source.triangulation.clone(),
                    |mut triangulation| {
                        black_box(
                            triangulation
                                .repair_delaunay_with_flips_advanced(
                                    DelaunayRepairHeuristicConfig::default(),
                                )
                                .or_abort(),
                        );
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark 2D flip-based Delaunay repair.
fn bench_delaunay_repair_2d(c: &mut Criterion) {
    bench_repair_dimension::<2>(c, "2d", &[500, 2_000], 0x2EFA_0000_0000_0002);
}

/// Benchmark 3D flip-based Delaunay repair.
fn bench_delaunay_repair_3d(c: &mut Criterion) {
    bench_repair_dimension::<3>(c, "3d", &[150, 500], 0x2EFA_0000_0000_0003);
}

/// Benchmark 4D flip-based Delaunay repair.
fn bench_delaunay_repair_4d(c: &mut Criterion) {
    bench_repair_dimension::<4>(c, "4d", &[50, 100], 0x2EFA_0000_0000_0004);
}

/// Benchmark 5D flip-based Delaunay repair.
fn bench_delaunay_repair_5d(c: &mut Criterion) {
    bench_repair_dimension::<5>(c, "5d", &[25, 40], 0x2EFA_0000_0000_0005);
}

criterion_group!(
    benches,
    bench_delaunay_repair_2d,
    bench_delaunay_repair_3d,
    bench_delaunay_repair_4d,
    bench_delaunay_repair_5d
);
criterion_main!(benches);
