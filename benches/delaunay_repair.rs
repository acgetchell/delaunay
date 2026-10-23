#![forbid(unsafe_code)]

//! Benchmark: consuming `Triangulation` to `DelaunayTriangulation` conversion.
//!
//! Fixtures are built with a `build_triangulation*` terminal, which returns a
//! valid Levels 1-4 triangulation without running batch or final Level 5 repair.
//! Construction may already produce a Delaunay triangulation.
//! The benchmark then times the public `delaunayize` entry point on a clone of
//! each prepared fixture.
//!
//! Setup verifies on a throwaway clone that repair converges for the chosen
//! seed, so the measured closure never times a repair failure chain. Case
//! labels record whether the prepared fixture was `violating` or already
//! `delaunay`, so baseline comparisons notice when a fixture changes meaning.
//! A separate `repair_transaction_pressure` case applies a fixed geometric flip
//! during setup and requires the probe to repair it without fallback rebuild,
//! isolating the public repair path most sensitive to per-flip rollback
//! snapshots. Compare that case with `tds_clone.rs` when evaluating journaled or
//! reusable rollback designs.
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

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use delaunay::flips::BistellarFlips;
use delaunay::prelude::construction::{DelaunayTriangulationBuilder, Vertex};
use delaunay::prelude::delaunayize::{DelaunayRefinementBuilder, DelaunayizeOutcome};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateRange, ExactPredicates, Point};
use delaunay::prelude::triangulation::Triangulation;
use delaunay::try_vertices_from_points;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::process;
use std::time::Duration;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, abort_benchmark};

const SAMPLE_SIZE: usize = 10;
const WARM_UP_TIME: Duration = Duration::from_millis(500);
const MEASUREMENT_TIME: Duration = Duration::from_secs(2);
const TRANSACTION_PRESSURE_MIN_FLIPS: NonZeroUsize = NonZeroUsize::MIN;

type BenchTriangulation<const D: usize> = Triangulation<AdaptiveKernel<f64>, (), (), D>;

#[derive(Clone, Copy)]
struct RepairFixture {
    requested_vertices: usize,
    seed: u64,
    flipped_facet_vertices: &'static [usize],
}

#[derive(Clone, Copy)]
enum RepairFixtureRequirement {
    AnyConvergent,
    DirectFlipRepair { min_flips: NonZeroUsize },
}

impl RepairFixtureRequirement {
    /// Returns whether a probe run has the repair behavior this benchmark case needs.
    const fn accepts(self, outcome: &DelaunayizeOutcome) -> bool {
        match self {
            Self::AnyConvergent => true,
            Self::DirectFlipRepair { min_flips } => {
                !outcome.used_fallback_rebuild
                    && outcome.delaunay_repair.flips_performed >= min_flips.get()
            }
        }
    }

    /// Describes the fixture predicate for benchmark setup failure messages.
    fn description(self) -> String {
        match self {
            Self::AnyConvergent => String::from("repair-convergent"),
            Self::DirectFlipRepair { min_flips } => format!(
                "direct flip-repair fixture with at least {} performed flips",
                min_flips.get()
            ),
        }
    }
}

struct RepairSource<const D: usize> {
    vertex_count: usize,
    simplex_count: usize,
    violating: bool,
    probe_flips_performed: usize,
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

/// Introduce a fixed violation without relying on incidental construction defects.
fn flip_fixture_facet<const D: usize>(
    triangulation: &mut BenchTriangulation<D>,
    vertices: &[Vertex<(), D>],
    input_indices: &[usize],
) where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    if input_indices.is_empty() {
        return;
    }
    if input_indices.len() != D {
        abort_benchmark(format_args!("{D}D fixture facet has incorrect arity"));
    }
    let vertex_ids: Vec<_> = input_indices
        .iter()
        .map(|&index| {
            vertices
                .get(index)
                .unwrap_or_else(|| {
                    abort_benchmark(format_args!(
                        "{D}D fixture facet references missing input vertex {index}"
                    ))
                })
                .uuid()
        })
        .collect();
    let facet = triangulation
        .facets()
        .map(OrAbort::or_abort)
        .find(|facet| {
            facet
                .vertices()
                .all(|vertex| vertex_ids.contains(&vertex.uuid()))
        })
        .map_or_else(
            || {
                abort_benchmark(format_args!(
                    "{D}D fixture no longer contains fixed facet {input_indices:?}"
                ))
            },
            |facet| facet.handle(),
        );
    triangulation.flip_k2(facet).or_abort();
}

/// Build one repair fixture whose repair is verified to converge.
///
/// A fixed seed must construct successfully, and a throwaway clone must complete
/// the consuming `delaunayize` workflow before any timings are collected.
fn build_source<const D: usize>(fixture: RepairFixture) -> RepairSource<D>
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    build_source_with_requirement(fixture, RepairFixtureRequirement::AnyConvergent)
}

/// Build one repair fixture whose probe satisfies the requested repair behavior.
///
/// The transaction-pressure case uses this to require successful direct
/// flip-based repair with performed flips, keeping fallback rebuild setup out
/// of the benchmark case that is meant to expose per-flip snapshot cost.
fn build_source_with_requirement<const D: usize>(
    fixture: RepairFixture,
    requirement: RepairFixtureRequirement,
) -> RepairSource<D>
where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let RepairFixture {
        requested_vertices,
        seed,
        flipped_facet_vertices,
    } = fixture;
    let vertices = generate_vertices::<D>(requested_vertices, seed);
    let mut triangulation = DelaunayTriangulationBuilder::new(&vertices)
        .build_triangulation()
        .unwrap_or_else(|error| {
            abort_benchmark(format_args!(
                "{D}D repair fixture seed {seed:#x} failed construction: {error}"
            ))
        });
    flip_fixture_facet(&mut triangulation, &vertices, flipped_facet_vertices);
    let probe = DelaunayRefinementBuilder::new(triangulation.clone())
        .repair_by_flips()
        .build()
        .unwrap_or_else(|error| {
            abort_benchmark(format_args!(
                "{D}D repair fixture seed {seed:#x} failed its repair probe: {error}"
            ))
        });
    if !requirement.accepts(&probe.outcome) {
        abort_benchmark(format_args!(
            "{D}D repair fixture seed {seed:#x} no longer satisfies {}",
            requirement.description()
        ));
    }

    let violating = !triangulation
        .delaunay_violation_report(None)
        .or_abort()
        .is_valid();
    RepairSource {
        vertex_count: triangulation.number_of_vertices(),
        simplex_count: triangulation.number_of_simplices(),
        violating,
        probe_flips_performed: probe.outcome.delaunay_repair.flips_performed,
        triangulation,
    }
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
    fixtures: &[RepairFixture],
    transaction_pressure_fixture: RepairFixture,
) where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let mut group = c.benchmark_group(format!("delaunay_repair/{dim_label}"));
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(WARM_UP_TIME);
    group.measurement_time(MEASUREMENT_TIME);

    for &fixture in fixtures {
        let source = build_source::<D>(fixture);
        let delaunay_state = if source.violating {
            "violating"
        } else {
            "delaunay"
        };
        group.throughput(Throughput::Elements(triangulation_element_count(&source)));

        group.bench_with_input(
            BenchmarkId::new(
                "delaunayize",
                format!(
                    "{delaunay_state}_vertices_{}_simplices_{}_seed_{:016x}",
                    source.vertex_count, source.simplex_count, fixture.seed
                ),
            ),
            &source,
            |b, source| {
                b.iter_batched(
                    || source.triangulation.clone(),
                    |triangulation| {
                        black_box(
                            DelaunayRefinementBuilder::new(triangulation)
                                .repair_by_flips()
                                .build()
                                .or_abort(),
                        )
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    bench_transaction_pressure_case::<D>(&mut group, transaction_pressure_fixture);

    group.finish();
}

/// Register the public repair case selected to exercise transactional rollback journals.
fn bench_transaction_pressure_case<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    fixture: RepairFixture,
) where
    AdaptiveKernel<f64>: ExactPredicates<D>,
{
    let source = build_source_with_requirement::<D>(
        fixture,
        RepairFixtureRequirement::DirectFlipRepair {
            min_flips: TRANSACTION_PRESSURE_MIN_FLIPS,
        },
    );
    group.throughput(Throughput::Elements(triangulation_element_count(&source)));

    group.bench_with_input(
        BenchmarkId::new(
            "repair_transaction_pressure",
            format!(
                "flip_repair_vertices_{}_simplices_{}_probe_flips_{}_seed_{:016x}_facet_{:?}",
                source.vertex_count,
                source.simplex_count,
                source.probe_flips_performed,
                fixture.seed,
                fixture.flipped_facet_vertices,
            ),
        ),
        &source,
        |b, source| {
            b.iter_batched(
                || source.triangulation.clone(),
                |triangulation| {
                    black_box(
                        DelaunayRefinementBuilder::new(triangulation)
                            .repair_by_flips()
                            .build()
                            .or_abort(),
                    )
                },
                BatchSize::SmallInput,
            );
        },
    );
}

/// Benchmark 2D flip-based Delaunay repair.
fn bench_delaunay_repair_2d(c: &mut Criterion) {
    bench_repair_dimension::<2>(
        c,
        "2d",
        &[
            RepairFixture {
                requested_vertices: 500,
                seed: 0x2AA3_BE4E_9D7A_5906,
                flipped_facet_vertices: &[],
            },
            RepairFixture {
                requested_vertices: 2_000,
                seed: 0x3F9C_F930_75E9_6412,
                flipped_facet_vertices: &[],
            },
        ],
        RepairFixture {
            requested_vertices: 2_000,
            seed: 0x9A8D_1083_19A4_43E3,
            flipped_facet_vertices: &[0, 223],
        },
    );
}

/// Benchmark 3D flip-based Delaunay repair.
fn bench_delaunay_repair_3d(c: &mut Criterion) {
    bench_repair_dimension::<3>(
        c,
        "3d",
        &[
            RepairFixture {
                requested_vertices: 150,
                seed: 0x9A7B_52B3_95A4_B44D,
                flipped_facet_vertices: &[],
            },
            RepairFixture {
                requested_vertices: 500,
                seed: 0x2AA3_BE4F_9D7A_5907,
                flipped_facet_vertices: &[],
            },
        ],
        RepairFixture {
            requested_vertices: 500,
            seed: 0x8FB2_57FC_F137_7EF6,
            flipped_facet_vertices: &[0, 23, 204],
        },
    );
}

/// Benchmark 4D flip-based Delaunay repair.
fn bench_delaunay_repair_4d(c: &mut Criterion) {
    bench_repair_dimension::<4>(
        c,
        "4d",
        &[
            RepairFixture {
                requested_vertices: 50,
                seed: 0xC82F_C63E_DC8C_3C20,
                flipped_facet_vertices: &[],
            },
            RepairFixture {
                requested_vertices: 100,
                seed: 0xE351_8C71_B918_7837,
                flipped_facet_vertices: &[],
            },
        ],
        RepairFixture {
            requested_vertices: 100,
            seed: 0x4640_65C2_D555_5FC8,
            flipped_facet_vertices: &[0, 2, 35, 98],
        },
    );
}

/// Benchmark 5D flip-based Delaunay repair.
fn bench_delaunay_repair_5d(c: &mut Criterion) {
    bench_repair_dimension::<5>(
        c,
        "5d",
        &[
            RepairFixture {
                requested_vertices: 25,
                seed: 0x5D90_E318_6E46_1E09,
                flipped_facet_vertices: &[],
            },
            RepairFixture {
                requested_vertices: 40,
                seed: 0x9651_04FE_E3A3_6365,
                flipped_facet_vertices: &[],
            },
        ],
        RepairFixture {
            requested_vertices: 40,
            seed: 0x3340_ED4D_8FEE_44D0,
            flipped_facet_vertices: &[0, 2, 11, 24, 25],
        },
    );
}

criterion_group!(
    benches,
    bench_delaunay_repair_2d,
    bench_delaunay_repair_3d,
    bench_delaunay_repair_4d,
    bench_delaunay_repair_5d
);
criterion_main!(benches);
