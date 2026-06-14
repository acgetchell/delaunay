#![forbid(unsafe_code)]

//! Benchmark: `Tds::clone` snapshot cost vs triangulation size (2D-5D)
//!
//! This benchmark measures the full topology/data snapshot cost that currently
//! dominates transactional rollback designs based on whole-`Tds` cloning. It is
//! intended as a baseline for comparing future journaled or localized rollback
//! designs.
//!
//! Intended for **manual** runs (not part of the CI performance suite).
//!
//! Run with:
//! ```bash
//! cargo bench --profile perf --bench tds_clone
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::prelude::construction::{DelaunayTriangulation, Vertex};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateRange};
use delaunay::prelude::tds::Tds;
use std::hint::black_box;
use std::time::Duration;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::bench_result;

const SEED_SALT: u64 = 0x9E37_79B9_7F4A_7C15;
const SAMPLE_SIZE: usize = 10;
const WARM_UP_TIME: Duration = Duration::from_millis(500);
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);

type BenchTriangulation<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;

fn benchmark_bounds() -> CoordinateRange<f64> {
    bench_result(
        CoordinateRange::try_new(-100.0_f64, 100.0),
        "clone benchmark bounds must be valid",
    )
}

struct CloneSource<const D: usize> {
    vertex_count: usize,
    simplex_count: usize,
    tds: Tds<(), (), D>,
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

/// Generate a reproducible vertex set for one clone-cost benchmark fixture.
fn generate_vertices<const D: usize>(requested_vertices: usize, seed: u64) -> Vec<Vertex<(), D>> {
    let points =
        generate_random_points_in_range_seeded::<D>(requested_vertices, benchmark_bounds(), seed);
    Vertex::from_points(&points)
}

/// Build the triangulation snapshot that each benchmark iteration clones.
fn build_clone_source<const D: usize>(requested_vertices: usize, seed_base: u64) -> CloneSource<D> {
    let seed = seed_for_case::<D>(requested_vertices, seed_base);
    let vertices = generate_vertices::<D>(requested_vertices, seed);
    let triangulation: BenchTriangulation<D> = bench_result(
        DelaunayTriangulation::new(&vertices),
        format!("failed to build {D}D benchmark triangulation"),
    );
    let tds = triangulation.tds().clone();

    CloneSource {
        vertex_count: tds.number_of_vertices(),
        simplex_count: tds.number_of_simplices(),
        tds,
    }
}

/// Report benchmark throughput in total stored vertices plus simplices.
fn tds_element_count<const D: usize>(source: &CloneSource<D>) -> u64 {
    let total_elements = source.vertex_count + source.simplex_count;
    bench_result(
        u64::try_from(total_elements),
        "TDS element count does not fit in u64",
    )
}

/// Register the clone-cost cases for one dimension and input-size schedule.
fn bench_dimension<const D: usize>(
    c: &mut Criterion,
    dim_label: &str,
    counts: &[usize],
    seed_base: u64,
) {
    let mut group = c.benchmark_group(format!("tds_clone/{dim_label}"));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);

    for &requested_vertices in counts {
        let source = build_clone_source::<D>(requested_vertices, seed_base);
        group.throughput(Throughput::Elements(tds_element_count(&source)));

        group.bench_with_input(
            BenchmarkId::new(
                "tds_clone",
                format!(
                    "vertices_{}_simplices_{}",
                    source.vertex_count, source.simplex_count
                ),
            ),
            &source,
            |b, source| {
                b.iter(|| black_box(source.tds.clone()));
            },
        );
    }

    group.finish();
}

/// Benchmark `Tds::clone` for representative 2D triangulations.
fn bench_tds_clone_2d(c: &mut Criterion) {
    bench_dimension::<2>(c, "2d", &[25, 100, 500], 0xD2C1_0000_0000_0001);
}

/// Benchmark `Tds::clone` for representative 3D triangulations.
fn bench_tds_clone_3d(c: &mut Criterion) {
    bench_dimension::<3>(c, "3d", &[25, 75, 150], 0xD3C1_0000_0000_0002);
}

/// Benchmark `Tds::clone` for representative 4D triangulations.
fn bench_tds_clone_4d(c: &mut Criterion) {
    bench_dimension::<4>(c, "4d", &[15, 30, 60], 0xD4C1_0000_0000_0003);
}

/// Benchmark `Tds::clone` for representative 5D triangulations.
fn bench_tds_clone_5d(c: &mut Criterion) {
    bench_dimension::<5>(c, "5d", &[10, 20, 35], 0xD5C1_0000_0000_0004);
}

criterion_group!(
    benches,
    bench_tds_clone_2d,
    bench_tds_clone_3d,
    bench_tds_clone_4d,
    bench_tds_clone_5d
);
criterion_main!(benches);
