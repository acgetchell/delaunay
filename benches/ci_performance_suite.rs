#![forbid(unsafe_code)]

//! CI Performance Suite - optimized performance regression testing for CI/CD
//!
//! This benchmark is the small, durable performance contract for the delaunay
//! library. It covers the user-facing workflows that should stay fast across
//! releases without duplicating every specialized microbenchmark:
//!
//! 1. Delaunay construction across 2D-5D at calibrated canary scales
//! 2. Convex hull extraction from completed triangulations
//! 3. Convex hull visibility/containment queries
//! 4. Boundary facet traversal
//! 5. Full validation (Levels 1-4)
//! 6. Incremental vertex insertion
//! 7. Explicit bistellar flip workflows on stable and adversarial 2D-5D
//!    PL-manifold cases
//!
//! The roundtrip flip cases are an n=1 ergodicity check for the public
//! Pachner/bistellar move API: one admissible move followed immediately by its
//! inverse must recover the same valid triangulation, including the same vertex
//! identities and simplex incidence. This is the local reversibility contract
//! behind the Pachner-move connectedness results cited in `REFERENCES.md` under
//! "Bistellar (Pachner) Moves and Delaunay Repair"; it is not a finite proof of
//! global ergodicity across all triangulations.
//!
//! Predicate microbenchmarks, allocation-focused measurements, and large-scale
//! stress tests live in the dedicated benchmark targets under `benches/`.
//!
//! ## Sample Size Strategy
//!
//! Uses dimension-dependent sample sizes to balance accuracy with CI time constraints:
//! - Construction canaries: 10 samples, with per-dimension counts selected to
//!   keep the repeated Criterion workload practical while exceeding toy inputs
//! - Operation benchmarks: reduced samples where setup or dimensionality is heavier
//!
//! ## Dimensional Focus
//!
//! Tests 2D, 3D, 4D, and 5D triangulations for comprehensive coverage:
//! - 2D: Fundamental triangulation case
//! - 3D-5D: Higher-dimensional triangulations as documented in README.md

use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayTriangulation, RetryPolicy, Vertex,
};
use delaunay::prelude::flips::{FacetHandle, RidgeHandle, SimplexKey};
use delaunay::prelude::generators::generate_random_points_in_range_seeded;
use delaunay::prelude::geometry::{AdaptiveKernel, CoordinateRange, Point};
use delaunay::prelude::query::ConvexHull;
use delaunay::try_vertices_from_points;
use std::{env, hint::black_box, num::NonZeroUsize, sync::Once};
#[cfg(feature = "bench-logging")]
use tracing::warn;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{abort_benchmark, bench_option, bench_result};

#[path = "common/flip_fixtures.rs"]
mod flip_fixtures;
use flip_fixtures::{
    ADVERSARIAL_POINTS_2D, ADVERSARIAL_POINTS_3D, ADVERSARIAL_POINTS_4D, ADVERSARIAL_POINTS_5D,
    STABLE_POINTS_2D, STABLE_POINTS_3D, STABLE_POINTS_4D, STABLE_POINTS_5D,
};

#[path = "common/flip_workflows.rs"]
mod flip_workflows;
use flip_workflows::{CandidateFilter, FlipTriangulation};

/// Calibrated fixture sizes for repeated Criterion performance checks.
///
/// These deliberately differ from the `just debug-large-scale-*` defaults. The
/// debug helpers are one-off, roughly one-minute acceptance/profiling runs.
/// Keep these canaries large enough to be release-comparison signals while
/// keeping each normal construction benchmark around one second on release
/// hardware. The adversarial variants use the same vertex counts and may take
/// longer.
const CANARY_COUNT_2D: usize = 4_000;
const CANARY_COUNT_3D: usize = 750;
const CANARY_COUNT_4D: usize = 75;
const CANARY_COUNT_5D: usize = 25;
/// Incremental insertion batch sizes are deliberately separate from fixture
/// sizes: these benchmarks measure adding a small batch into an existing
/// calibrated triangulation, not rebuilding the full fixture.
const INSERT_BATCH_COUNT_2D_3D: usize = 10;
const INSERT_BATCH_COUNT_4D: usize = 6;
const INSERT_BATCH_COUNT_5D: usize = 4;
type SeedSearchResult<const D: usize> = Option<(u64, Vec<Point<D>>, Vec<Vertex<(), D>>)>;
type BenchTriangulation<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;

fn finite_point<const D: usize>(coords: [f64; D]) -> Point<D> {
    Point::try_new(coords).unwrap_or_else(|_| std::process::abort())
}

fn retry_attempts(value: usize) -> NonZeroUsize {
    let Some(attempts) = NonZeroUsize::new(value) else {
        unreachable!("hard-coded retry attempt count must be non-zero");
    };
    attempts
}

#[derive(Clone, Copy)]
enum Dataset {
    WellConditioned,
    Adversarial,
}

impl Dataset {
    const fn suffix(self) -> &'static str {
        match self {
            Self::WellConditioned => "",
            Self::Adversarial => "_adversarial",
        }
    }
}

struct ApiBenchmarkEntry {
    group: &'static str,
    public_api: &'static str,
    dimensions: &'static str,
    benchmark_ids: String,
    note: &'static str,
}

static API_BENCHMARK_MANIFEST: Once = Once::new();

fn construction_benchmark_ids() -> String {
    [
        format!("tds_new_2d/{{tds_new,tds_new_adversarial}}/{CANARY_COUNT_2D}"),
        format!("tds_new_3d/{{tds_new,tds_new_adversarial}}/{CANARY_COUNT_3D}"),
        format!("tds_new_4d/{{tds_new,tds_new_adversarial}}/{CANARY_COUNT_4D}"),
        format!("tds_new_5d/{{tds_new,tds_new_adversarial}}/{CANARY_COUNT_5D}"),
    ]
    .join(";")
}

fn operation_benchmark_ids(group: &str, prefix: &str) -> String {
    [
        format!("{group}/{{{prefix}_2d,{prefix}_2d_adversarial}}/{CANARY_COUNT_2D}"),
        format!("{group}/{{{prefix}_3d,{prefix}_3d_adversarial}}/{CANARY_COUNT_3D}"),
        format!("{group}/{{{prefix}_4d,{prefix}_4d_adversarial}}/{CANARY_COUNT_4D}"),
        format!("{group}/{{{prefix}_5d,{prefix}_5d_adversarial}}/{CANARY_COUNT_5D}"),
    ]
    .join(";")
}

fn validation_benchmark_ids() -> String {
    [
        format!("validation/{{validate_3d,validate_3d_adversarial}}/{CANARY_COUNT_3D}"),
        format!("validation/{{validate_4d,validate_4d_adversarial}}/{CANARY_COUNT_4D}"),
        format!("validation/{{validate_5d,validate_5d_adversarial}}/{CANARY_COUNT_5D}"),
    ]
    .join(";")
}

fn hull_query_benchmark_ids() -> String {
    [
        format!(
            "convex_hull_queries/{{is_point_outside_3d,is_point_outside_3d_adversarial}}/{CANARY_COUNT_3D}"
        ),
        format!(
            "convex_hull_queries/{{find_visible_facets_3d,find_visible_facets_3d_adversarial}}/{CANARY_COUNT_3D}"
        ),
        format!(
            "convex_hull_queries/{{find_nearest_visible_facet_3d,find_nearest_visible_facet_3d_adversarial}}/{CANARY_COUNT_3D}"
        ),
    ]
    .join(";")
}

fn insert_benchmark_ids() -> String {
    [
        format!(
            "incremental_insert/{{insert_2d,insert_2d_adversarial}}/{INSERT_BATCH_COUNT_2D_3D}"
        ),
        format!(
            "incremental_insert/{{insert_3d,insert_3d_adversarial}}/{INSERT_BATCH_COUNT_2D_3D}"
        ),
        format!("incremental_insert/{{insert_4d,insert_4d_adversarial}}/{INSERT_BATCH_COUNT_4D}"),
        format!("incremental_insert/{{insert_5d,insert_5d_adversarial}}/{INSERT_BATCH_COUNT_5D}"),
    ]
    .join(";")
}

fn api_benchmark_entries() -> Vec<ApiBenchmarkEntry> {
    vec![
        ApiBenchmarkEntry {
            group: "construction",
            public_api: "DelaunayTriangulation::try_new_with_options",
            dimensions: "2,3,4,5",
            benchmark_ids: construction_benchmark_ids(),
            note: "construct_from_calibrated_seeded_and_adversarial_inputs",
        },
        ApiBenchmarkEntry {
            group: "boundary_facets",
            public_api: "DelaunayTriangulation::boundary_facets",
            dimensions: "2,3,4,5",
            benchmark_ids: operation_benchmark_ids("boundary_facets", "boundary_facets"),
            note: "iterate_boundary_facets_on_well_conditioned_and_adversarial_inputs",
        },
        ApiBenchmarkEntry {
            group: "convex_hull",
            public_api: "ConvexHull::try_from_triangulation",
            dimensions: "2,3,4,5",
            benchmark_ids: operation_benchmark_ids("convex_hull", "from_triangulation"),
            note: "extract_hull_from_well_conditioned_and_adversarial_triangulations",
        },
        ApiBenchmarkEntry {
            group: "convex_hull_queries",
            public_api: "ConvexHull::{is_point_outside,find_visible_facets,find_nearest_visible_facet}",
            dimensions: "3",
            benchmark_ids: hull_query_benchmark_ids(),
            note: "query_prebuilt_3d_hulls_from_well_conditioned_and_adversarial_inputs",
        },
        ApiBenchmarkEntry {
            group: "validation",
            public_api: "DelaunayTriangulation::validate",
            dimensions: "3,4,5",
            benchmark_ids: validation_benchmark_ids(),
            note: "levels_1_through_4_on_well_conditioned_and_adversarial_inputs",
        },
        ApiBenchmarkEntry {
            group: "incremental_insert",
            public_api: "DelaunayTriangulation::insert",
            dimensions: "2,3,4,5",
            benchmark_ids: insert_benchmark_ids(),
            note: "insert_batches_into_calibrated_well_conditioned_and_adversarial_triangulations",
        },
        ApiBenchmarkEntry {
            group: "bistellar_flips",
            public_api: "BistellarFlips::{flip_k1_insert,flip_k1_remove,flip_k2,flip_k2_inverse_from_edge,flip_k3,flip_k3_inverse_from_triangle}",
            dimensions: "2,3,4,5",
            benchmark_ids: [
                "bistellar_flips_2d/k1_roundtrip",
                "bistellar_flips_2d/k1_roundtrip_adversarial",
                "bistellar_flips_2d/k2_edge_flip",
                "bistellar_flips_2d/k2_edge_flip_adversarial",
                "bistellar_flips_3d/k1_roundtrip",
                "bistellar_flips_3d/k1_roundtrip_adversarial",
                "bistellar_flips_3d/k2_roundtrip",
                "bistellar_flips_3d/k2_roundtrip_adversarial",
                "bistellar_flips_3d/k3_forward",
                "bistellar_flips_3d/k3_forward_adversarial",
                "bistellar_flips_4d/k1_roundtrip",
                "bistellar_flips_4d/k1_roundtrip_adversarial",
                "bistellar_flips_4d/k2_roundtrip",
                "bistellar_flips_4d/k2_roundtrip_adversarial",
                "bistellar_flips_4d/k3_roundtrip",
                "bistellar_flips_4d/k3_roundtrip_adversarial",
                "bistellar_flips_5d/k1_roundtrip",
                "bistellar_flips_5d/k1_roundtrip_adversarial",
                "bistellar_flips_5d/k2_roundtrip",
                "bistellar_flips_5d/k2_roundtrip_adversarial",
                "bistellar_flips_5d/k3_roundtrip",
                "bistellar_flips_5d/k3_roundtrip_adversarial",
            ]
            .join(";"),
            note: "stable_and_adversarial_pl_manifold_public_flip_workflows",
        },
    ]
}

/// Pre-computed seeds for each calibrated (dimension, count) pair.
///
/// These were discovered using `DELAUNAY_BENCH_DISCOVER_SEEDS=1` and eliminate
/// the expensive runtime seed search from the benchmark hot path.
///
/// To refresh seeds (e.g. after algorithm changes that invalidate them), run:
///
/// ```bash
/// DELAUNAY_BENCH_DISCOVER_SEEDS=1 cargo bench --profile perf --bench ci_performance_suite
/// ```
///
/// (Use a Criterion filter for individual cases, e.g.
///  `-- "tds_new_5d/tds_new/25"`)
const KNOWN_SEEDS: &[(usize, usize, u64)] = &[
    (2, CANARY_COUNT_2D, 4042),
    (3, CANARY_COUNT_3D, 873),
    (4, CANARY_COUNT_4D, 531),
    (5, CANARY_COUNT_5D, 816),
];

const KNOWN_ADV_SEEDS: &[(usize, usize, u64)] = &[
    (2, CANARY_COUNT_2D, 2_779_101_199),
    (3, CANARY_COUNT_3D, 2_779_099_326),
    (4, CANARY_COUNT_4D, 2_779_104_312),
    (5, CANARY_COUNT_5D, 2_779_109_924),
];

fn known_seed(dim: usize, count: usize) -> Option<u64> {
    KNOWN_SEEDS
        .iter()
        .find(|&&(d, c, _)| d == dim && c == count)
        .map(|&(_, _, seed)| seed)
}

fn known_adv_seed(dim: usize, count: usize) -> Option<u64> {
    KNOWN_ADV_SEEDS
        .iter()
        .find(|&&(d, c, _)| d == dim && c == count)
        .map(|&(_, _, seed)| seed)
}

fn print_manifest_once() {
    API_BENCHMARK_MANIFEST.call_once(|| {
        println!(
            "api_benchmark_manifest crate=delaunay version={} benchmark=ci_performance_suite schema=1",
            env!("CARGO_PKG_VERSION")
        );
        for entry in api_benchmark_entries() {
            println!(
                "api_benchmark group={} public_api={} dimensions={} benchmark_ids={} note={}",
                entry.group, entry.public_api, entry.dimensions, entry.benchmark_ids, entry.note
            );
        }
    });
}

/// Prepare benchmark inputs by looking up a pre-computed seed, falling back
/// to a runtime search only if the known seed is missing or invalid.
fn prepare_data<const D: usize>(
    dim_seed: u64,
    count: usize,
    bounds: CoordinateRange<f64>,
    attempts: NonZeroUsize,
) -> (u64, Vec<Point<D>>, Vec<Vertex<(), D>>) {
    // Fast path: use the pre-computed seed (single verification construction)
    if let Some(seed) = known_seed(D, count) {
        if let Some(result) = find_seed_vertices::<D>(seed, count, bounds, 1, attempts) {
            return result;
        }

        warn_known_seed_failed::<D>(seed, count, Dataset::WellConditioned);
    }

    // Slow fallback: runtime search from the base seed
    let base_seed = dim_seed.wrapping_add(count as u64);
    let search_limit = seed_search_limit();
    bench_option(
        find_seed_vertices::<D>(base_seed, count, bounds, search_limit, attempts),
        format_args!(
            "No stable benchmark seed found for {D}D/{count}: \
                 start_seed={base_seed}; search_limit={search_limit}; bounds={bounds}"
        ),
    )
}

fn warn_known_seed_failed<const D: usize>(seed: u64, count: usize, dataset: Dataset) {
    let dataset_label = match dataset {
        Dataset::WellConditioned => "well_conditioned",
        Dataset::Adversarial => "adversarial",
    };

    #[cfg(not(feature = "bench-logging"))]
    let _ = (seed, count, dataset_label);
    #[cfg(feature = "bench-logging")]
    {
        warn!(
            known_seed = seed,
            dim = D,
            count,
            dataset = dataset_label,
            "known seed failed, falling back to runtime search"
        );
    }
}

fn prepare_dt<const D: usize>(dim_seed: u64, count: usize) -> BenchTriangulation<D> {
    let bounds = bench_result(
        CoordinateRange::try_new(-100.0_f64, 100.0),
        "well-conditioned benchmark bounds must be valid",
    );
    let attempts = retry_attempts(6);
    let (seed, _, vertices) = prepare_data::<D>(dim_seed, count, bounds, attempts);
    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
        attempts,
        base_seed: Some(seed),
    });

    bench_result(
        BenchTriangulation::<D>::try_new_with_options(&vertices, options),
        format!("failed to prepare {D}D benchmark triangulation with {count} vertices"),
    )
}

fn prepare_adv_dt<const D: usize>(dim_seed: u64, count: usize) -> BenchTriangulation<D> {
    let attempts = retry_attempts(8);
    let (seed, _, vertices) = prepare_adv_data::<D>(dim_seed, count, attempts);
    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
        attempts,
        base_seed: Some(seed),
    });

    bench_result(
        BenchTriangulation::<D>::try_new_with_options(&vertices, options),
        format!("failed to prepare adversarial {D}D benchmark triangulation with {count} vertices"),
    )
}

fn prepare_inserts<const D: usize>(
    dim_seed: u64,
    count: usize,
    dataset: Dataset,
) -> Vec<Vertex<(), D>> {
    let mut seed = dim_seed.wrapping_add(0x5151_5151);
    if matches!(dataset, Dataset::Adversarial) {
        seed ^= 0xA5A5_A5A5;
    }
    let points = match dataset {
        Dataset::WellConditioned => bench_result(
            generate_random_points_in_range_seeded::<D>(
                count,
                bench_result(
                    CoordinateRange::try_new(-50.0_f64, 50.0),
                    "insert benchmark bounds must be valid",
                ),
                seed,
            ),
            "failed to generate insert benchmark points",
        ),
        Dataset::Adversarial => generate_adv_points::<D>(count, seed),
    };
    bench_result(
        try_vertices_from_points(&points),
        "failed to create insert benchmark vertices",
    )
}

fn find_seed_vertices<const D: usize>(
    start_seed: u64,
    count: usize,
    bounds: CoordinateRange<f64>,
    limit: usize,
    attempts: NonZeroUsize,
) -> SeedSearchResult<D> {
    for offset in 0..limit {
        let candidate_seed = start_seed.wrapping_add(offset as u64);
        let points = bench_result(
            generate_random_points_in_range_seeded::<D>(count, bounds, candidate_seed),
            "failed to generate candidate benchmark points",
        );
        let vertices = bench_result(
            try_vertices_from_points(&points),
            "failed to create candidate benchmark vertices",
        );

        let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
            attempts,
            base_seed: Some(candidate_seed),
        });

        if DelaunayTriangulation::<_, (), (), D>::try_new_with_options(&vertices, options).is_ok() {
            return Some((candidate_seed, points, vertices));
        }
    }

    None
}

fn stable_adv_points<const D: usize>(
    seed: u64,
    count: usize,
    attempts: NonZeroUsize,
) -> SeedSearchResult<D> {
    let points = generate_adv_points::<D>(count, seed);
    let vertices = bench_result(
        try_vertices_from_points(&points),
        "failed to create adversarial benchmark vertices",
    );
    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
        attempts,
        base_seed: Some(seed),
    });

    BenchTriangulation::<D>::try_new_with_options(&vertices, options)
        .is_ok()
        .then_some((seed, points, vertices))
}

fn prepare_adv_data<const D: usize>(
    dim_seed: u64,
    count: usize,
    attempts: NonZeroUsize,
) -> (u64, Vec<Point<D>>, Vec<Vertex<(), D>>) {
    if !discover_seeds_enabled()
        && let Some(seed) = known_adv_seed(D, count)
    {
        if let Some(result) = stable_adv_points::<D>(seed, count, attempts) {
            return result;
        }

        warn_known_seed_failed::<D>(seed, count, Dataset::Adversarial);
    }

    let start_seed = dim_seed
        .wrapping_mul(17)
        .wrapping_add(count as u64)
        .wrapping_add(0xA5A5_A5A5);
    let search_limit = seed_search_limit();

    for offset in 0..search_limit {
        let candidate_seed = start_seed.wrapping_add(offset as u64);
        if let Some(result) = stable_adv_points::<D>(candidate_seed, count, attempts) {
            if discover_seeds_enabled() {
                println!("ADV_SEED {D} {count} {candidate_seed}");
            }
            return result;
        }
    }

    abort_benchmark(format_args!(
        "No stable adversarial benchmark seed found for {D}D/{count}: \
         start_seed={start_seed}; search_limit={search_limit}"
    ));
}

fn generate_adv_points<const D: usize>(count: usize, seed: u64) -> Vec<Point<D>> {
    let base_points = bench_result(
        generate_random_points_in_range_seeded::<D>(
            count,
            bench_result(
                CoordinateRange::try_new(-1.0_f64, 1.0),
                "adversarial benchmark bounds must be valid",
            ),
            seed,
        ),
        "failed to generate adversarial benchmark base points",
    );

    base_points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let index = bench_result(
                u32::try_from(index),
                "benchmark point index should fit in u32",
            );
            let mut coords = [0.0_f64; D];
            for (axis, coord) in coords.iter_mut().enumerate() {
                let axis_number = bench_result(u32::try_from(axis + 1), "axis should fit in u32");
                let base = point.coords()[axis];
                let cluster_offset = f64::from(index % 7) * 1.0e-3;
                let axis_offset = f64::from(axis_number) * 0.25;
                let perturbation = f64::from((index + axis_number) % 11) * 1.0e-6;
                *coord = base.mul_add(1.0e3, 1.0e9 + axis_offset + cluster_offset + perturbation);
            }
            finite_point(coords)
        })
        .collect()
}

/// Builds a deterministic PL-manifold triangulation for bistellar flip coverage.
///
/// The benchmark uses input ordering so preselected public flip candidates from
/// both stable and adversarial fixtures stay deterministic across runs and
/// Criterion measures only the public flip operation.
fn build_flip_dt<const D: usize>(points: &[[f64; D]]) -> FlipTriangulation<D> {
    bench_result(
        flip_workflows::build_flip_dt(points),
        format!("failed to build {D}D flip fixture triangulation"),
    )
}

/// Selects a non-degenerate simplex for deterministic k=1 benchmark setup.
fn largest_volume_simplex<const D: usize>(dt: &FlipTriangulation<D>) -> SimplexKey {
    largest_volume_simplex_matching(dt, CandidateFilter::Any)
}

/// Selects a deterministic simplex touching an adversarial fixture feature.
fn adversarial_largest_volume_simplex<const D: usize>(dt: &FlipTriangulation<D>) -> SimplexKey {
    let simplex_key =
        largest_volume_simplex_matching(dt, CandidateFilter::TouchesAdversarialFeature);
    let touches_feature = bench_result(
        flip_workflows::simplex_touches_adversarial_feature(dt, simplex_key),
        format!("failed to inspect adversarial k=1 simplex support for {D}D"),
    );
    if !touches_feature {
        abort_benchmark(format_args!(
            "selected adversarial {D}D k=1 simplex does not touch an adversarial fixture feature"
        ));
    }
    simplex_key
}

/// Selects a largest-volume simplex using a caller-provided filter.
fn largest_volume_simplex_matching<const D: usize>(
    dt: &FlipTriangulation<D>,
    filter: CandidateFilter,
) -> SimplexKey {
    bench_result(
        flip_workflows::largest_volume_simplex(dt, filter),
        format!("failed to select {filter:?} k=1 simplex for {D}D flip benchmark"),
    )
}

/// Exercises the public k=1 insert and remove APIs as one benchmark workflow.
fn roundtrip_k1<const D: usize>(dt: &mut FlipTriangulation<D>, simplex_key: SimplexKey) {
    bench_result(
        flip_workflows::roundtrip_k1(dt, simplex_key),
        format!("k=1 roundtrip should succeed in {D}D"),
    );
}

/// Finds a deterministic k=2 facet candidate before Criterion opens the timed group.
///
/// Some dimensions benchmark only the forward public flip because the inverse
/// move is not part of that dimension's coverage contract. `require_inverse`
/// controls whether candidate discovery also proves the inverse public API is
/// available for the selected facet.
fn flippable_k2_facet<const D: usize>(
    dt: &FlipTriangulation<D>,
    require_inverse: bool,
) -> FacetHandle {
    flippable_k2_facet_matching(dt, require_inverse, CandidateFilter::Any)
}

/// Finds a deterministic adversarial k=2 facet candidate before timing.
fn adversarial_flippable_k2_facet<const D: usize>(
    dt: &FlipTriangulation<D>,
    require_inverse: bool,
) -> FacetHandle {
    let facet = flippable_k2_facet_matching(
        dt,
        require_inverse,
        CandidateFilter::TouchesAdversarialFeature,
    );
    let touches_feature = bench_result(
        flip_workflows::facet_support_touches_adversarial_feature(dt, facet),
        format!("failed to inspect adversarial k=2 facet support for {D}D"),
    );
    if !touches_feature {
        abort_benchmark(format_args!(
            "selected adversarial {D}D k=2 facet does not touch an adversarial fixture feature"
        ));
    }
    facet
}

/// Selects a k=2 facet candidate using a caller-provided filter.
fn flippable_k2_facet_matching<const D: usize>(
    dt: &FlipTriangulation<D>,
    require_inverse: bool,
    filter: CandidateFilter,
) -> FacetHandle {
    bench_result(
        flip_workflows::flippable_k2_facet(dt, require_inverse, filter),
        format!("failed to select {filter:?} k=2 facet for {D}D flip benchmark"),
    )
}

/// Exercises the public k=2 forward flip API for dimensions without inversion.
fn forward_k2<const D: usize>(dt: &mut FlipTriangulation<D>, facet: FacetHandle) {
    bench_result(
        flip_workflows::forward_k2(dt, facet),
        format!("k=2 flip should succeed for preselected {D}D benchmark facet"),
    );
}

/// Exercises the public k=2 flip and inverse APIs as one benchmark workflow.
fn roundtrip_k2<const D: usize>(dt: &mut FlipTriangulation<D>, facet: FacetHandle) {
    bench_result(
        flip_workflows::roundtrip_k2(dt, facet),
        format!("k=2 roundtrip should succeed in {D}D"),
    );
}

/// Finds a deterministic k=3 ridge candidate before Criterion opens the timed group.
///
/// Some dimensions benchmark only the forward public flip because the inverse
/// move is not part of that dimension's coverage contract. `require_inverse`
/// controls whether candidate discovery also proves the inverse public API is
/// available for the selected ridge.
fn flippable_k3_ridge<const D: usize>(
    dt: &FlipTriangulation<D>,
    require_inverse: bool,
) -> RidgeHandle {
    flippable_k3_ridge_matching(dt, require_inverse, CandidateFilter::Any)
}

/// Finds a deterministic adversarial k=3 ridge candidate before timing.
fn adversarial_flippable_k3_ridge<const D: usize>(
    dt: &FlipTriangulation<D>,
    require_inverse: bool,
) -> RidgeHandle {
    let ridge = flippable_k3_ridge_matching(
        dt,
        require_inverse,
        CandidateFilter::TouchesAdversarialFeature,
    );
    let touches_feature = bench_result(
        flip_workflows::ridge_support_touches_adversarial_feature(dt, ridge),
        format!("failed to inspect adversarial k=3 ridge support for {D}D"),
    );
    if !touches_feature {
        abort_benchmark(format_args!(
            "selected adversarial {D}D k=3 ridge does not touch an adversarial fixture feature"
        ));
    }
    ridge
}

/// Selects a k=3 ridge candidate using a caller-provided filter.
fn flippable_k3_ridge_matching<const D: usize>(
    dt: &FlipTriangulation<D>,
    require_inverse: bool,
    filter: CandidateFilter,
) -> RidgeHandle {
    bench_result(
        flip_workflows::flippable_k3_ridge(dt, require_inverse, filter),
        format!("failed to select {filter:?} k=3 ridge for {D}D flip benchmark"),
    )
}

/// Exercises the public k=3 forward flip API for dimensions without inversion.
fn forward_k3<const D: usize>(dt: &mut FlipTriangulation<D>, ridge: RidgeHandle) {
    bench_result(
        flip_workflows::forward_k3(dt, ridge),
        format!("k=3 flip should succeed for preselected {D}D benchmark ridge"),
    );
}

/// Exercises the public k=3 flip and inverse APIs as one benchmark workflow.
fn roundtrip_k3<const D: usize>(dt: &mut FlipTriangulation<D>, ridge: RidgeHandle) {
    bench_result(
        flip_workflows::roundtrip_k3(dt, ridge),
        format!("k=3 roundtrip should succeed in {D}D"),
    );
}

/// Registers one k=1 insert/remove roundtrip flip benchmark case.
fn bench_k1_roundtrip_case<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &'static str,
    base_dt: &FlipTriangulation<D>,
    simplex_key: SimplexKey,
) {
    bench_result(
        flip_workflows::verify_k1_roundtrip(base_dt, simplex_key, name),
        format!("k=1 setup roundtrip should recover exact topology for {name}"),
    );
    group.bench_function(name, |b| {
        b.iter_batched(
            || base_dt.clone(),
            |mut dt| {
                roundtrip_k1(&mut dt, simplex_key);
                black_box(dt);
            },
            BatchSize::LargeInput,
        );
    });
}

/// Registers one forward-only k=2 flip benchmark case.
fn bench_k2_forward_case<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &'static str,
    base_dt: &FlipTriangulation<D>,
    facet: FacetHandle,
) {
    group.bench_function(name, |b| {
        b.iter_batched(
            || base_dt.clone(),
            |mut dt| {
                forward_k2(&mut dt, facet);
                black_box(dt);
            },
            BatchSize::LargeInput,
        );
    });
}

/// Registers one k=2 flip/inverse roundtrip benchmark case.
fn bench_k2_roundtrip_case<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &'static str,
    base_dt: &FlipTriangulation<D>,
    facet: FacetHandle,
) {
    bench_result(
        flip_workflows::verify_k2_roundtrip(base_dt, facet, name),
        format!("k=2 setup roundtrip should recover exact topology for {name}"),
    );
    group.bench_function(name, |b| {
        b.iter_batched(
            || base_dt.clone(),
            |mut dt| {
                roundtrip_k2(&mut dt, facet);
                black_box(dt);
            },
            BatchSize::LargeInput,
        );
    });
}

/// Registers one forward-only k=3 flip benchmark case.
fn bench_k3_forward_case<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &'static str,
    base_dt: &FlipTriangulation<D>,
    ridge: RidgeHandle,
) {
    group.bench_function(name, |b| {
        b.iter_batched(
            || base_dt.clone(),
            |mut dt| {
                forward_k3(&mut dt, ridge);
                black_box(dt);
            },
            BatchSize::LargeInput,
        );
    });
}

/// Registers one k=3 flip/inverse roundtrip benchmark case.
fn bench_k3_roundtrip_case<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &'static str,
    base_dt: &FlipTriangulation<D>,
    ridge: RidgeHandle,
) {
    bench_result(
        flip_workflows::verify_k3_roundtrip(base_dt, ridge, name),
        format!("k=3 setup roundtrip should recover exact topology for {name}"),
    );
    group.bench_function(name, |b| {
        b.iter_batched(
            || base_dt.clone(),
            |mut dt| {
                roundtrip_k3(&mut dt, ridge);
                black_box(dt);
            },
            BatchSize::LargeInput,
        );
    });
}

fn discover_seeds_enabled() -> bool {
    env::var("DELAUNAY_BENCH_DISCOVER_SEEDS").is_ok_and(|value| value != "0")
}

/// Return positional Criterion filter arguments from the current benchmark invocation.
fn criterion_filters() -> Vec<String> {
    env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with('-'))
        .collect()
}

/// Return whether a benchmark ID matches at least one Criterion-style substring filter.
fn filter_matches_benchmark(filters: &[String], benchmark_id: &str) -> bool {
    filters.iter().any(|filter| benchmark_id.contains(filter))
}

/// Return whether a benchmark ID should run for the provided filters.
fn benchmark_selected(filters: &[String], benchmark_id: &str) -> bool {
    filters.is_empty() || filter_matches_benchmark(filters, benchmark_id)
}

/// Return whether the benchmark should emit construction metrics without sampling.
///
/// This is a release-tooling escape hatch for refreshing generated simplex
/// counts with a filtered `tds_new` run. Any present value except `0` enables
/// the metric-only path.
fn export_metrics_enabled() -> bool {
    env::var("DELAUNAY_BENCH_EXPORT_METRICS").is_ok_and(|value| value != "0")
}

fn seed_search_limit() -> usize {
    env::var("DELAUNAY_BENCH_DISCOVER_SEEDS_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2000)
}

/// Emit one parseable construction metric line for the generated report.
///
/// The summary generator parses these `api_benchmark_metric` lines to display
/// vertex and simplex counts in `benches/PERFORMANCE_RESULTS.md`. Construction
/// failures abort the benchmark because stale or missing simplex counts would
/// make the generated release data misleading.
fn emit_construction_metric<const D: usize>(
    benchmark_id: &str,
    vertices: &[Vertex<(), D>],
    options: ConstructionOptions,
) {
    let dt = bench_result(
        BenchTriangulation::<D>::try_new_with_options(vertices, options),
        format!("failed to collect construction metrics for {benchmark_id}"),
    );
    println!(
        "api_benchmark_metric benchmark_id={benchmark_id} vertices={} simplices={}",
        vertices.len(),
        dt.number_of_simplices()
    );
}

/// Fixed seeds for deterministic triangulation generation across benchmark runs.
/// Using seeded random number generation reduces variance in performance measurements
/// and improves regression detection accuracy in CI environments.
/// Different seeds per dimension ensure triangulations are uncorrelated.
///
/// Macro to reduce duplication in dimensional benchmark functions
macro_rules! benchmark_tds_new_dimension {
    ($dim:literal, $func_name:ident, $seed:literal, $count:expr) => {
        /// Benchmark triangulation creation for D-dimensional triangulations
        #[expect(
            clippy::too_many_lines,
            reason = "dimension-specific benchmark macro keeps setup and measurements together"
        )]
        fn $func_name(c: &mut Criterion) {
            print_manifest_once();
            let count = $count;

            // Opt-in helper for discovering stable seeds without paying Criterion warmup/
            // measurement cost per seed.
            //
            // NOTE: This helper is intentionally per calibrated benchmark case.
            // It returns early on the first successful seed (and panics on failure),
            // so it is meant to be run with a Criterion filter, for example:
            //
            //     cargo bench --profile perf --bench ci_performance_suite -- 'tds_new_3d/tds_new/750'
            //
            // The seed table should contain exactly this calibrated count for each
            // dimension/dataset pair; runtime search is only a refresh fallback.
            //
            // We avoid `std::process::exit` here so that destructors run and Criterion
            // can clean up state on both success and failure.
            if discover_seeds_enabled() {
                let bounds = bench_result(
                    CoordinateRange::try_new(-100.0_f64, 100.0),
                    "well-conditioned benchmark bounds must be valid",
                );
                let filters = criterion_filters();

                let bench_id = format!("tds_new_{}d/tds_new/{count}", stringify!($dim));
                let adv_bench_id =
                    format!("tds_new_{}d/tds_new_adversarial/{count}", stringify!($dim));

                if !filters.is_empty() && filter_matches_benchmark(&filters, &adv_bench_id) {
                    let attempts = retry_attempts(8);
                    let _ = prepare_adv_data::<$dim>($seed, count, attempts);
                    return;
                }

                if benchmark_selected(&filters, &bench_id) {
                    let seed = ($seed as u64).wrapping_add(count as u64);
                    let limit = seed_search_limit();
                    let attempts = retry_attempts(6);

                    if let Some((candidate_seed, _, _)) =
                        find_seed_vertices::<$dim>(seed, count, bounds, limit, attempts)
                    {
                        println!(
                            "seed_search_found dim={} count={} seed={}",
                            $dim, count, candidate_seed
                        );
                        return;
                    }

                    abort_benchmark(format_args!(
                        "seed_search_failed dim={} count={} start_seed={} limit={}",
                        $dim, count, seed, limit
                    ));
                }

                // No filter matched this benchmark function; do nothing.
                return;
            }

            if export_metrics_enabled() {
                let filters = criterion_filters();
                let bench_id = format!("tds_new_{}d/tds_new/{count}", stringify!($dim));
                let adv_bench_id =
                    format!("tds_new_{}d/tds_new_adversarial/{count}", stringify!($dim));

                if benchmark_selected(&filters, &bench_id) {
                    let bounds = bench_result(
                        CoordinateRange::try_new(-100.0_f64, 100.0),
                        "well-conditioned benchmark bounds must be valid",
                    );
                    let attempts = retry_attempts(6);
                    let (seed, _, vertices) = prepare_data::<$dim>($seed, count, bounds, attempts);
                    let options = ConstructionOptions::default().with_retry_policy(
                        RetryPolicy::Shuffled {
                            attempts,
                            base_seed: Some(seed),
                        },
                    );
                    emit_construction_metric::<$dim>(&bench_id, &vertices, options);
                }

                if benchmark_selected(&filters, &adv_bench_id) {
                    let attempts = retry_attempts(8);
                    let (seed, _, vertices) = prepare_adv_data::<$dim>($seed, count, attempts);
                    let options = ConstructionOptions::default().with_retry_policy(
                        RetryPolicy::Shuffled {
                            attempts,
                            base_seed: Some(seed),
                        },
                    );
                    emit_construction_metric::<$dim>(&adv_bench_id, &vertices, options);
                }
                return;
            }

            let bench_id = format!("tds_new_{}d/tds_new/{count}", stringify!($dim));
            let adv_bench_id = format!("tds_new_{}d/tds_new_adversarial/{count}", stringify!($dim));
            let mut group = c.benchmark_group(concat!("tds_new_", stringify!($dim), "d"));

            // The calibrated construction cases are intentionally larger than
            // the historical toy counts. Use Criterion's practical floor so
            // the repeated samples stay suitable for PR regression checks.
            group.sample_size(10);

            group.throughput(Throughput::Elements(count as u64));

            group.bench_with_input(BenchmarkId::new("tds_new", count), &count, |b, &count| {
                    // Reduce variance: pre-generate deterministic inputs outside the measured loop,
                    // then benchmark only triangulation construction.
                    let bounds = bench_result(
                        CoordinateRange::try_new(-100.0_f64, 100.0),
                        "well-conditioned benchmark bounds must be valid",
                    );
                    let attempts = retry_attempts(6);
                    let (seed, points, vertices) =
                        prepare_data::<$dim>($seed, count, bounds, attempts);
                    let sample_points = points.iter().take(5).collect::<Vec<_>>();

                    // In benchmarks we compile in release mode, where the default retry policy is
                    // disabled. For deterministic CI benchmarks we opt into a small number of
                    // shuffled retries to avoid aborting the suite on rare non-convergent repair
                    // cases.
                    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
                        attempts,
                        base_seed: Some(seed),
                    });
                    emit_construction_metric::<$dim>(&bench_id, &vertices, options);

                    b.iter(|| {
                        match DelaunayTriangulation::<_, (), (), $dim>::try_new_with_options(
                            &vertices,
                            options,
                        ) {
                            Ok(dt) => {
                                black_box(dt);
                            }
                            Err(err) => {
                                let error = format!("{err:?}");
                                abort_benchmark(format_args!(
                                    "DelaunayTriangulation::try_new_with_options failed for {}D: {error}; dim={}; count={}; seed={}; bounds={:?}; sample_points={sample_points:?}",
                                    $dim,
                                    $dim,
                                    count,
                                    seed,
                                    bounds
                                ));
                            }
                        }
                    });
            });

            group.bench_with_input(
                BenchmarkId::new("tds_new_adversarial", count),
                &count,
                |b, &count| {
                        let attempts = retry_attempts(8);
                        let (seed, points, vertices) =
                            prepare_adv_data::<$dim>($seed, count, attempts);
                        let sample_points = points.iter().take(5).collect::<Vec<_>>();
                        let options = ConstructionOptions::default().with_retry_policy(
                            RetryPolicy::Shuffled {
                                attempts,
                                base_seed: Some(seed),
                            },
                        );
                        emit_construction_metric::<$dim>(&adv_bench_id, &vertices, options);

                        b.iter(|| {
                            match DelaunayTriangulation::<_, (), (), $dim>::try_new_with_options(
                                &vertices,
                                options,
                            ) {
                                Ok(dt) => {
                                    black_box(dt);
                                }
                                Err(err) => {
                                    let error = format!("{err:?}");
                                    abort_benchmark(format_args!(
                                        "adversarial DelaunayTriangulation::try_new_with_options failed for {}D: {error}; dim={}; count={}; seed={}; sample_points={sample_points:?}",
                                        $dim,
                                        $dim,
                                        count,
                                        seed
                                    ));
                                }
                            }
                        });
                },
            );

            group.finish();
        }
    };
}

// Generate benchmark functions using the macro
benchmark_tds_new_dimension!(2, benchmark_tds_new_2d, 42, CANARY_COUNT_2D);
benchmark_tds_new_dimension!(3, benchmark_tds_new_3d, 123, CANARY_COUNT_3D);
benchmark_tds_new_dimension!(4, benchmark_tds_new_4d, 456, CANARY_COUNT_4D);
benchmark_tds_new_dimension!(5, benchmark_tds_new_5d, 789, CANARY_COUNT_5D);

fn bench_boundary_case<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    dimension: usize,
    dataset: Dataset,
    count: usize,
    dt: &BenchTriangulation<D>,
) {
    group.throughput(Throughput::Elements(count as u64));
    group.bench_function(
        BenchmarkId::new(
            format!("boundary_facets_{dimension}d{}", dataset.suffix()),
            count,
        ),
        |b| {
            b.iter(|| {
                black_box(match dt.boundary_facets() {
                    Ok(facets) => facets.count(),
                    Err(error) => unreachable!(
                        "validated benchmark triangulation should build boundary facets: {error}"
                    ),
                })
            });
        },
    );
}

fn bench_hull_case<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    dimension: usize,
    dataset: Dataset,
    count: usize,
    dt: &BenchTriangulation<D>,
) {
    group.throughput(Throughput::Elements(count as u64));
    group.bench_function(
        BenchmarkId::new(
            format!("from_triangulation_{dimension}d{}", dataset.suffix()),
            count,
        ),
        |b| {
            b.iter(|| {
                let hull = match ConvexHull::try_from_triangulation(dt.as_triangulation()) {
                    Ok(value) => value,
                    Err(error) => abort_benchmark(format_args!(
                        "convex hull extraction should succeed: {error}"
                    )),
                };
                let _ = black_box(hull);
            });
        },
    );
}

fn exterior_hull_query_point<const D: usize>(dt: &BenchTriangulation<D>) -> Point<D> {
    let mut mins = [f64::INFINITY; D];
    let mut maxs = [f64::NEG_INFINITY; D];
    let mut has_vertices = false;

    for (_, vertex) in dt.as_triangulation().vertices() {
        has_vertices = true;
        for (axis, coord) in vertex.point().coords().iter().copied().enumerate() {
            mins[axis] = mins[axis].min(coord);
            maxs[axis] = maxs[axis].max(coord);
        }
    }

    if !has_vertices {
        abort_benchmark("hull query benchmark triangulation should contain vertices");
    }

    let mut coords = [0.0; D];
    for (axis, coord) in coords.iter_mut().enumerate() {
        let span = maxs[axis] - mins[axis];
        *coord = maxs[axis] + span.max(1.0);
    }
    finite_point(coords)
}

fn bench_hull_query_case<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    dimension: usize,
    dataset: Dataset,
    count: usize,
    dt: &BenchTriangulation<D>,
) {
    let hull = match ConvexHull::try_from_triangulation(dt.as_triangulation()) {
        Ok(value) => value,
        Err(error) => abort_benchmark(format_args!(
            "convex hull extraction should succeed before query benchmarks: {error}"
        )),
    };
    let outside_point = exterior_hull_query_point(dt);
    match hull.is_point_outside(&outside_point, dt.as_triangulation()) {
        Ok(true) => {}
        Ok(false) => abort_benchmark(
            "computed exterior hull query point should be outside the benchmark hull",
        ),
        Err(error) => abort_benchmark(format_args!(
            "ConvexHull::is_point_outside should validate the exterior query point: {error}"
        )),
    }

    group.throughput(Throughput::Elements(count as u64));
    group.bench_function(
        BenchmarkId::new(
            format!("is_point_outside_{dimension}d{}", dataset.suffix()),
            count,
        ),
        |b| {
            b.iter(
                || match hull.is_point_outside(&outside_point, dt.as_triangulation()) {
                    Ok(value) => {
                        let _ = black_box(value);
                    }
                    Err(error) => abort_benchmark(format_args!(
                        "ConvexHull::is_point_outside should succeed: {error}"
                    )),
                },
            );
        },
    );

    group.bench_function(
        BenchmarkId::new(
            format!("find_visible_facets_{dimension}d{}", dataset.suffix()),
            count,
        ),
        |b| {
            b.iter(
                || match hull.find_visible_facets(&outside_point, dt.as_triangulation()) {
                    Ok(value) => {
                        let _ = black_box(value);
                    }
                    Err(error) => abort_benchmark(format_args!(
                        "ConvexHull::find_visible_facets should succeed: {error}"
                    )),
                },
            );
        },
    );

    group.bench_function(
        BenchmarkId::new(
            format!(
                "find_nearest_visible_facet_{dimension}d{}",
                dataset.suffix()
            ),
            count,
        ),
        |b| {
            b.iter(|| {
                match hull.find_nearest_visible_facet(&outside_point, dt.as_triangulation()) {
                    Ok(value) => {
                        let _ = black_box(value);
                    }
                    Err(error) => abort_benchmark(format_args!(
                        "ConvexHull::find_nearest_visible_facet should succeed: {error}"
                    )),
                }
            });
        },
    );
}

fn bench_validate_case<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    dimension: usize,
    dataset: Dataset,
    count: usize,
    dt: &BenchTriangulation<D>,
) {
    group.throughput(Throughput::Elements(count as u64));
    group.bench_function(
        BenchmarkId::new(format!("validate_{dimension}d{}", dataset.suffix()), count),
        |b| {
            b.iter(|| match black_box(dt.validate()) {
                Ok(()) => {}
                Err(error) => {
                    abort_benchmark(format_args!(
                        "{dimension}D benchmark triangulation should validate: {error}"
                    ));
                }
            });
        },
    );
}

fn bench_insert_case<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    dimension: usize,
    dataset: Dataset,
    count: usize,
    base_dt: &BenchTriangulation<D>,
    insert_vertices: &[Vertex<(), D>],
) {
    group.throughput(Throughput::Elements(count as u64));
    group.bench_function(
        BenchmarkId::new(format!("insert_{dimension}d{}", dataset.suffix()), count),
        |b| {
            b.iter_batched(
                || (base_dt.clone(), insert_vertices.to_vec()),
                |(mut dt, vertices)| {
                    for vertex in vertices {
                        match black_box(dt.insert(vertex)) {
                            Ok(_) => {}
                            Err(error) => {
                                abort_benchmark(format_args!(
                                    "{dimension}D incremental insert should succeed: {error}"
                                ));
                            }
                        }
                    }
                    black_box(dt);
                },
                BatchSize::LargeInput,
            );
        },
    );
}

fn benchmark_boundary_facets(c: &mut Criterion) {
    print_manifest_once();
    if discover_seeds_enabled() {
        return;
    }
    let mut group = c.benchmark_group("boundary_facets");
    group.sample_size(25);

    let dt_2d = prepare_dt::<2>(42, CANARY_COUNT_2D);
    bench_boundary_case(
        &mut group,
        2,
        Dataset::WellConditioned,
        CANARY_COUNT_2D,
        &dt_2d,
    );
    let dt_2d_adversarial = prepare_adv_dt::<2>(42, CANARY_COUNT_2D);
    bench_boundary_case(
        &mut group,
        2,
        Dataset::Adversarial,
        CANARY_COUNT_2D,
        &dt_2d_adversarial,
    );

    let dt_3d = prepare_dt::<3>(123, CANARY_COUNT_3D);
    bench_boundary_case(
        &mut group,
        3,
        Dataset::WellConditioned,
        CANARY_COUNT_3D,
        &dt_3d,
    );
    let dt_3d_adversarial = prepare_adv_dt::<3>(123, CANARY_COUNT_3D);
    bench_boundary_case(
        &mut group,
        3,
        Dataset::Adversarial,
        CANARY_COUNT_3D,
        &dt_3d_adversarial,
    );

    let dt_4d = prepare_dt::<4>(456, CANARY_COUNT_4D);
    bench_boundary_case(
        &mut group,
        4,
        Dataset::WellConditioned,
        CANARY_COUNT_4D,
        &dt_4d,
    );
    let dt_4d_adversarial = prepare_adv_dt::<4>(456, CANARY_COUNT_4D);
    bench_boundary_case(
        &mut group,
        4,
        Dataset::Adversarial,
        CANARY_COUNT_4D,
        &dt_4d_adversarial,
    );

    let dt_5d = prepare_dt::<5>(789, CANARY_COUNT_5D);
    bench_boundary_case(
        &mut group,
        5,
        Dataset::WellConditioned,
        CANARY_COUNT_5D,
        &dt_5d,
    );
    let dt_5d_adversarial = prepare_adv_dt::<5>(789, CANARY_COUNT_5D);
    bench_boundary_case(
        &mut group,
        5,
        Dataset::Adversarial,
        CANARY_COUNT_5D,
        &dt_5d_adversarial,
    );

    group.finish();
}

fn benchmark_convex_hull(c: &mut Criterion) {
    print_manifest_once();
    if discover_seeds_enabled() {
        return;
    }
    let mut group = c.benchmark_group("convex_hull");
    group.sample_size(20);

    let dt_2d = prepare_dt::<2>(42, CANARY_COUNT_2D);
    bench_hull_case(
        &mut group,
        2,
        Dataset::WellConditioned,
        CANARY_COUNT_2D,
        &dt_2d,
    );
    let dt_2d_adversarial = prepare_adv_dt::<2>(42, CANARY_COUNT_2D);
    bench_hull_case(
        &mut group,
        2,
        Dataset::Adversarial,
        CANARY_COUNT_2D,
        &dt_2d_adversarial,
    );

    let dt_3d = prepare_dt::<3>(123, CANARY_COUNT_3D);
    bench_hull_case(
        &mut group,
        3,
        Dataset::WellConditioned,
        CANARY_COUNT_3D,
        &dt_3d,
    );
    let dt_3d_adversarial = prepare_adv_dt::<3>(123, CANARY_COUNT_3D);
    bench_hull_case(
        &mut group,
        3,
        Dataset::Adversarial,
        CANARY_COUNT_3D,
        &dt_3d_adversarial,
    );

    let dt_4d = prepare_dt::<4>(456, CANARY_COUNT_4D);
    bench_hull_case(
        &mut group,
        4,
        Dataset::WellConditioned,
        CANARY_COUNT_4D,
        &dt_4d,
    );
    let dt_4d_adversarial = prepare_adv_dt::<4>(456, CANARY_COUNT_4D);
    bench_hull_case(
        &mut group,
        4,
        Dataset::Adversarial,
        CANARY_COUNT_4D,
        &dt_4d_adversarial,
    );

    let dt_5d = prepare_dt::<5>(789, CANARY_COUNT_5D);
    bench_hull_case(
        &mut group,
        5,
        Dataset::WellConditioned,
        CANARY_COUNT_5D,
        &dt_5d,
    );
    let dt_5d_adversarial = prepare_adv_dt::<5>(789, CANARY_COUNT_5D);
    bench_hull_case(
        &mut group,
        5,
        Dataset::Adversarial,
        CANARY_COUNT_5D,
        &dt_5d_adversarial,
    );

    group.finish();
}

fn benchmark_convex_hull_queries(c: &mut Criterion) {
    print_manifest_once();
    if discover_seeds_enabled() {
        return;
    }
    let mut group = c.benchmark_group("convex_hull_queries");
    group.sample_size(20);

    let dt_3d = prepare_dt::<3>(123, CANARY_COUNT_3D);
    bench_hull_query_case(
        &mut group,
        3,
        Dataset::WellConditioned,
        CANARY_COUNT_3D,
        &dt_3d,
    );
    let dt_3d_adversarial = prepare_adv_dt::<3>(123, CANARY_COUNT_3D);
    bench_hull_query_case(
        &mut group,
        3,
        Dataset::Adversarial,
        CANARY_COUNT_3D,
        &dt_3d_adversarial,
    );

    group.finish();
}

fn benchmark_validation(c: &mut Criterion) {
    print_manifest_once();
    if discover_seeds_enabled() {
        return;
    }
    let mut group = c.benchmark_group("validation");
    group.sample_size(15);

    let dt_3d = prepare_dt::<3>(123, CANARY_COUNT_3D);
    bench_validate_case(
        &mut group,
        3,
        Dataset::WellConditioned,
        CANARY_COUNT_3D,
        &dt_3d,
    );
    let dt_3d_adversarial = prepare_adv_dt::<3>(123, CANARY_COUNT_3D);
    bench_validate_case(
        &mut group,
        3,
        Dataset::Adversarial,
        CANARY_COUNT_3D,
        &dt_3d_adversarial,
    );

    let dt_4d = prepare_dt::<4>(456, CANARY_COUNT_4D);
    bench_validate_case(
        &mut group,
        4,
        Dataset::WellConditioned,
        CANARY_COUNT_4D,
        &dt_4d,
    );
    let dt_4d_adversarial = prepare_adv_dt::<4>(456, CANARY_COUNT_4D);
    bench_validate_case(
        &mut group,
        4,
        Dataset::Adversarial,
        CANARY_COUNT_4D,
        &dt_4d_adversarial,
    );

    let dt_5d = prepare_dt::<5>(789, CANARY_COUNT_5D);
    bench_validate_case(
        &mut group,
        5,
        Dataset::WellConditioned,
        CANARY_COUNT_5D,
        &dt_5d,
    );
    let dt_5d_adversarial = prepare_adv_dt::<5>(789, CANARY_COUNT_5D);
    bench_validate_case(
        &mut group,
        5,
        Dataset::Adversarial,
        CANARY_COUNT_5D,
        &dt_5d_adversarial,
    );

    group.finish();
}

fn benchmark_insert(c: &mut Criterion) {
    print_manifest_once();
    if discover_seeds_enabled() {
        return;
    }
    let mut group = c.benchmark_group("incremental_insert");
    group.sample_size(15);

    let dt_2d = prepare_dt::<2>(42, CANARY_COUNT_2D);
    let insert_2d = prepare_inserts::<2>(42, INSERT_BATCH_COUNT_2D_3D, Dataset::WellConditioned);
    bench_insert_case(
        &mut group,
        2,
        Dataset::WellConditioned,
        INSERT_BATCH_COUNT_2D_3D,
        &dt_2d,
        &insert_2d,
    );
    let dt_2d_adversarial = prepare_adv_dt::<2>(42, CANARY_COUNT_2D);
    let insert_2d_adversarial =
        prepare_inserts::<2>(42, INSERT_BATCH_COUNT_2D_3D, Dataset::Adversarial);
    bench_insert_case(
        &mut group,
        2,
        Dataset::Adversarial,
        INSERT_BATCH_COUNT_2D_3D,
        &dt_2d_adversarial,
        &insert_2d_adversarial,
    );

    let dt_3d = prepare_dt::<3>(123, CANARY_COUNT_3D);
    let insert_3d = prepare_inserts::<3>(123, INSERT_BATCH_COUNT_2D_3D, Dataset::WellConditioned);
    bench_insert_case(
        &mut group,
        3,
        Dataset::WellConditioned,
        INSERT_BATCH_COUNT_2D_3D,
        &dt_3d,
        &insert_3d,
    );
    let dt_3d_adversarial = prepare_adv_dt::<3>(123, CANARY_COUNT_3D);
    let insert_3d_adversarial =
        prepare_inserts::<3>(123, INSERT_BATCH_COUNT_2D_3D, Dataset::Adversarial);
    bench_insert_case(
        &mut group,
        3,
        Dataset::Adversarial,
        INSERT_BATCH_COUNT_2D_3D,
        &dt_3d_adversarial,
        &insert_3d_adversarial,
    );

    let dt_4d = prepare_dt::<4>(456, CANARY_COUNT_4D);
    let insert_4d = prepare_inserts::<4>(456, INSERT_BATCH_COUNT_4D, Dataset::WellConditioned);
    bench_insert_case(
        &mut group,
        4,
        Dataset::WellConditioned,
        INSERT_BATCH_COUNT_4D,
        &dt_4d,
        &insert_4d,
    );
    let dt_4d_adversarial = prepare_adv_dt::<4>(456, CANARY_COUNT_4D);
    let insert_4d_adversarial =
        prepare_inserts::<4>(456, INSERT_BATCH_COUNT_4D, Dataset::Adversarial);
    bench_insert_case(
        &mut group,
        4,
        Dataset::Adversarial,
        INSERT_BATCH_COUNT_4D,
        &dt_4d_adversarial,
        &insert_4d_adversarial,
    );

    let dt_5d = prepare_dt::<5>(789, CANARY_COUNT_5D);
    let insert_5d = prepare_inserts::<5>(789, INSERT_BATCH_COUNT_5D, Dataset::WellConditioned);
    bench_insert_case(
        &mut group,
        5,
        Dataset::WellConditioned,
        INSERT_BATCH_COUNT_5D,
        &dt_5d,
        &insert_5d,
    );
    let dt_5d_adversarial = prepare_adv_dt::<5>(789, CANARY_COUNT_5D);
    let insert_5d_adversarial =
        prepare_inserts::<5>(789, INSERT_BATCH_COUNT_5D, Dataset::Adversarial);
    bench_insert_case(
        &mut group,
        5,
        Dataset::Adversarial,
        INSERT_BATCH_COUNT_5D,
        &dt_5d_adversarial,
        &insert_5d_adversarial,
    );

    group.finish();
}

/// Registers the complete 2D-5D public bistellar flip benchmark matrix.
fn benchmark_bistellar_flips(c: &mut Criterion) {
    print_manifest_once();
    if discover_seeds_enabled() {
        return;
    }

    bench_bistellar_flips_2d(c);
    bench_bistellar_flips_3d(c);
    bench_bistellar_flips_4d(c);
    bench_bistellar_flips_5d(c);
}

/// Benchmarks 2D public flip coverage: k=1 roundtrip and k=2 edge flip.
fn bench_bistellar_flips_2d(c: &mut Criterion) {
    let base_dt_2d = build_flip_dt(STABLE_POINTS_2D);
    let k1_simplex_2d = largest_volume_simplex(&base_dt_2d);
    let k2_facet_2d = flippable_k2_facet(&base_dt_2d, false);
    let adv_dt_2d = build_flip_dt(ADVERSARIAL_POINTS_2D);
    let adv_k1_simplex_2d = adversarial_largest_volume_simplex(&adv_dt_2d);
    let adv_k2_facet_2d = adversarial_flippable_k2_facet(&adv_dt_2d, false);

    let mut group_2d = c.benchmark_group("bistellar_flips_2d");
    group_2d.sample_size(10);

    bench_k1_roundtrip_case(&mut group_2d, "k1_roundtrip", &base_dt_2d, k1_simplex_2d);
    bench_k1_roundtrip_case(
        &mut group_2d,
        "k1_roundtrip_adversarial",
        &adv_dt_2d,
        adv_k1_simplex_2d,
    );
    bench_k2_forward_case(&mut group_2d, "k2_edge_flip", &base_dt_2d, k2_facet_2d);
    bench_k2_forward_case(
        &mut group_2d,
        "k2_edge_flip_adversarial",
        &adv_dt_2d,
        adv_k2_facet_2d,
    );
    group_2d.finish();
}

/// Benchmarks 3D public flip coverage: k=1/k=2 roundtrips and k=3 forward.
fn bench_bistellar_flips_3d(c: &mut Criterion) {
    let base_dt_3d = build_flip_dt(STABLE_POINTS_3D);
    let k1_simplex_3d = largest_volume_simplex(&base_dt_3d);
    let k2_facet_3d = flippable_k2_facet(&base_dt_3d, true);
    let k3_ridge_3d = flippable_k3_ridge(&base_dt_3d, false);
    let adv_dt_3d = build_flip_dt(ADVERSARIAL_POINTS_3D);
    let adv_k1_simplex_3d = adversarial_largest_volume_simplex(&adv_dt_3d);
    let adv_k2_facet_3d = adversarial_flippable_k2_facet(&adv_dt_3d, true);
    let adv_k3_ridge_3d = adversarial_flippable_k3_ridge(&adv_dt_3d, false);

    let mut group_3d = c.benchmark_group("bistellar_flips_3d");
    group_3d.sample_size(10);

    bench_k1_roundtrip_case(&mut group_3d, "k1_roundtrip", &base_dt_3d, k1_simplex_3d);
    bench_k1_roundtrip_case(
        &mut group_3d,
        "k1_roundtrip_adversarial",
        &adv_dt_3d,
        adv_k1_simplex_3d,
    );
    bench_k2_roundtrip_case(&mut group_3d, "k2_roundtrip", &base_dt_3d, k2_facet_3d);
    bench_k2_roundtrip_case(
        &mut group_3d,
        "k2_roundtrip_adversarial",
        &adv_dt_3d,
        adv_k2_facet_3d,
    );
    bench_k3_forward_case(&mut group_3d, "k3_forward", &base_dt_3d, k3_ridge_3d);
    bench_k3_forward_case(
        &mut group_3d,
        "k3_forward_adversarial",
        &adv_dt_3d,
        adv_k3_ridge_3d,
    );
    group_3d.finish();
}

/// Benchmarks 4D public flip coverage: k=1/k=2/k=3 roundtrips.
fn bench_bistellar_flips_4d(c: &mut Criterion) {
    let base_dt_4d = build_flip_dt(STABLE_POINTS_4D);
    let k1_simplex_4d = largest_volume_simplex(&base_dt_4d);
    let k2_facet_4d = flippable_k2_facet(&base_dt_4d, true);
    let k3_ridge_4d = flippable_k3_ridge(&base_dt_4d, true);
    let adv_dt_4d = build_flip_dt(ADVERSARIAL_POINTS_4D);
    let adv_k1_simplex_4d = adversarial_largest_volume_simplex(&adv_dt_4d);
    let adv_k2_facet_4d = adversarial_flippable_k2_facet(&adv_dt_4d, true);
    let adv_k3_ridge_4d = adversarial_flippable_k3_ridge(&adv_dt_4d, true);

    let mut group_4d = c.benchmark_group("bistellar_flips_4d");
    group_4d.sample_size(10);

    bench_k1_roundtrip_case(&mut group_4d, "k1_roundtrip", &base_dt_4d, k1_simplex_4d);
    bench_k1_roundtrip_case(
        &mut group_4d,
        "k1_roundtrip_adversarial",
        &adv_dt_4d,
        adv_k1_simplex_4d,
    );
    bench_k2_roundtrip_case(&mut group_4d, "k2_roundtrip", &base_dt_4d, k2_facet_4d);
    bench_k2_roundtrip_case(
        &mut group_4d,
        "k2_roundtrip_adversarial",
        &adv_dt_4d,
        adv_k2_facet_4d,
    );
    bench_k3_roundtrip_case(&mut group_4d, "k3_roundtrip", &base_dt_4d, k3_ridge_4d);
    bench_k3_roundtrip_case(
        &mut group_4d,
        "k3_roundtrip_adversarial",
        &adv_dt_4d,
        adv_k3_ridge_4d,
    );
    group_4d.finish();
}

/// Benchmarks 5D public flip coverage: k=1/k=2/k=3 roundtrips.
fn bench_bistellar_flips_5d(c: &mut Criterion) {
    let base_dt_5d = build_flip_dt(STABLE_POINTS_5D);
    let k1_simplex_5d = largest_volume_simplex(&base_dt_5d);
    let k2_facet_5d = flippable_k2_facet(&base_dt_5d, true);
    let k3_ridge_5d = flippable_k3_ridge(&base_dt_5d, true);
    let adv_dt_5d = build_flip_dt(ADVERSARIAL_POINTS_5D);
    let adv_k1_simplex_5d = adversarial_largest_volume_simplex(&adv_dt_5d);
    let adv_k2_facet_5d = adversarial_flippable_k2_facet(&adv_dt_5d, true);
    let adv_k3_ridge_5d = adversarial_flippable_k3_ridge(&adv_dt_5d, true);

    let mut group_5d = c.benchmark_group("bistellar_flips_5d");
    group_5d.sample_size(10);

    bench_k1_roundtrip_case(&mut group_5d, "k1_roundtrip", &base_dt_5d, k1_simplex_5d);
    bench_k1_roundtrip_case(
        &mut group_5d,
        "k1_roundtrip_adversarial",
        &adv_dt_5d,
        adv_k1_simplex_5d,
    );
    bench_k2_roundtrip_case(&mut group_5d, "k2_roundtrip", &base_dt_5d, k2_facet_5d);
    bench_k2_roundtrip_case(
        &mut group_5d,
        "k2_roundtrip_adversarial",
        &adv_dt_5d,
        adv_k2_facet_5d,
    );
    bench_k3_roundtrip_case(&mut group_5d, "k3_roundtrip", &base_dt_5d, k3_ridge_5d);
    bench_k3_roundtrip_case(
        &mut group_5d,
        "k3_roundtrip_adversarial",
        &adv_dt_5d,
        adv_k3_ridge_5d,
    );
    group_5d.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        benchmark_tds_new_2d,
        benchmark_tds_new_3d,
        benchmark_tds_new_4d,
        benchmark_tds_new_5d,
        benchmark_boundary_facets,
        benchmark_convex_hull,
        benchmark_convex_hull_queries,
        benchmark_validation,
        benchmark_insert,
        benchmark_bistellar_flips
);
criterion_main!(benches);
