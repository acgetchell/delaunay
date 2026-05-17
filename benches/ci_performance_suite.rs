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
//! 7. Explicit bistellar flip roundtrips on a stable 4D PL-manifold case
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
use delaunay::prelude::generators::generate_random_points_seeded;
use delaunay::prelude::geometry::{
    AdaptiveKernel, Coordinate, Point, RobustKernel, simplex_volume,
};
use delaunay::prelude::query::ConvexHull;
use delaunay::prelude::triangulation::construction::{
    ConstructionOptions, DelaunayTriangulation, InsertionOrderStrategy, RetryPolicy,
    TopologyGuarantee, Vertex,
};
use delaunay::prelude::triangulation::flips::{
    BistellarFlips, EdgeKey, FacetHandle, RidgeHandle, SimplexKey, TriangleHandle,
};
use delaunay::vertex;
use std::{env, hint::black_box, num::NonZeroUsize, sync::Once};
#[cfg(feature = "bench-logging")]
use tracing::warn;

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{abort_benchmark, bench_option, bench_result};

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
type SeedSearchResult<const D: usize> = Option<(u64, Vec<Point<f64, D>>, Vec<Vertex<f64, (), D>>)>;
type BenchTriangulation<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;
type FlipTriangulation4 = DelaunayTriangulation<RobustKernel<f64>, (), (), 4>;

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
            public_api: "DelaunayTriangulation::new_with_options",
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
            public_api: "ConvexHull::from_triangulation",
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
            dimensions: "4",
            benchmark_ids: "bistellar_flips_4d/k1_roundtrip;bistellar_flips_4d/k2_roundtrip;bistellar_flips_4d/k3_roundtrip".to_string(),
            note: "stable_pl_manifold_roundtrips",
        },
    ]
}

/// Stable 4D PL-manifold configuration used for explicit bistellar flips.
const STABLE_POINTS_4D: &[[f64; 4]] = &[
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.10, 0.10, 0.10, 0.10],
    [0.15, 0.10, 0.10, 0.10],
    [0.10, 0.15, 0.10, 0.10],
    [0.10, 0.10, 0.15, 0.10],
    [0.12, 0.12, 0.12, 0.12],
    [0.20, 0.15, 0.10, 0.05],
    [0.08, 0.18, 0.12, 0.14],
];

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
    bounds: (f64, f64),
    attempts: NonZeroUsize,
) -> (u64, Vec<Point<f64, D>>, Vec<Vertex<f64, (), D>>) {
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
                 start_seed={base_seed}; search_limit={search_limit}; bounds={bounds:?}"
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
    let bounds = (-100.0, 100.0);
    let attempts = retry_attempts(6);
    let (seed, _, vertices) = prepare_data::<D>(dim_seed, count, bounds, attempts);
    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
        attempts,
        base_seed: Some(seed),
    });

    bench_result(
        BenchTriangulation::<D>::new_with_options(&vertices, options),
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
        BenchTriangulation::<D>::new_with_options(&vertices, options),
        format!("failed to prepare adversarial {D}D benchmark triangulation with {count} vertices"),
    )
}

fn prepare_inserts<const D: usize>(
    dim_seed: u64,
    count: usize,
    dataset: Dataset,
) -> Vec<Vertex<f64, (), D>> {
    let mut seed = dim_seed.wrapping_add(0x5151_5151);
    if matches!(dataset, Dataset::Adversarial) {
        seed ^= 0xA5A5_A5A5;
    }
    let points = match dataset {
        Dataset::WellConditioned => bench_result(
            generate_random_points_seeded::<f64, D>(count, (-50.0, 50.0), seed),
            format!("insert point generation failed for {D}D"),
        ),
        Dataset::Adversarial => generate_adv_points::<D>(count, seed),
    };
    points.iter().map(|point| vertex!(*point)).collect()
}

fn find_seed_vertices<const D: usize>(
    start_seed: u64,
    count: usize,
    bounds: (f64, f64),
    limit: usize,
    attempts: NonZeroUsize,
) -> SeedSearchResult<D> {
    for offset in 0..limit {
        let candidate_seed = start_seed.wrapping_add(offset as u64);
        let points = bench_result(
            generate_random_points_seeded::<f64, D>(count, bounds, candidate_seed),
            format!("generate_random_points_seeded failed for {D}D"),
        );
        let vertices = points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>();

        let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
            attempts,
            base_seed: Some(candidate_seed),
        });

        if DelaunayTriangulation::<_, (), (), D>::new_with_options(&vertices, options).is_ok() {
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
    let vertices = points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>();
    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
        attempts,
        base_seed: Some(seed),
    });

    BenchTriangulation::<D>::new_with_options(&vertices, options)
        .is_ok()
        .then_some((seed, points, vertices))
}

fn prepare_adv_data<const D: usize>(
    dim_seed: u64,
    count: usize,
    attempts: NonZeroUsize,
) -> (u64, Vec<Point<f64, D>>, Vec<Vertex<f64, (), D>>) {
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

fn generate_adv_points<const D: usize>(count: usize, seed: u64) -> Vec<Point<f64, D>> {
    let base_points = bench_result(
        generate_random_points_seeded::<f64, D>(count, (-1.0, 1.0), seed),
        format!("generate_random_points_seeded failed for adversarial {D}D"),
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
            Point::new(coords)
        })
        .collect()
}

fn stable_vertices_4d() -> Vec<Vertex<f64, (), 4>> {
    STABLE_POINTS_4D
        .iter()
        .map(|coords| vertex!(*coords))
        .collect()
}

fn build_flip_dt_4d() -> FlipTriangulation4 {
    let vertices = stable_vertices_4d();
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    DelaunayTriangulation::with_topology_guarantee_and_options(
        &RobustKernel::new(),
        &vertices,
        TopologyGuarantee::PLManifold,
        options,
    )
    .unwrap_or_else(|err| {
        abort_benchmark(format_args!(
            "failed to build stable 4D flip triangulation: {err}"
        ))
    })
}

fn simplex_centroid_4d(dt: &FlipTriangulation4, simplex_key: SimplexKey) -> [f64; 4] {
    let simplex = bench_option(
        dt.tds().simplex(simplex_key),
        "simplex key should exist in benchmark triangulation",
    );

    let mut coords = [0.0_f64; 4];
    for &vkey in simplex.vertices() {
        let vertex = bench_option(
            dt.tds().vertex(vkey),
            "vertex key should exist in benchmark triangulation",
        );
        let vcoords = vertex.point().coords();
        for i in 0..4 {
            coords[i] += vcoords[i];
        }
    }

    let vertex_count = bench_result(
        u32::try_from(simplex.vertices().len()),
        "simplex vertex count should fit in u32",
    );
    let inv = 1.0_f64 / f64::from(vertex_count);
    for coord in &mut coords {
        *coord *= inv;
    }
    coords
}

fn simplex_points_4d(dt: &FlipTriangulation4, simplex_key: SimplexKey) -> Vec<Point<f64, 4>> {
    let simplex = bench_option(
        dt.tds().simplex(simplex_key),
        "simplex key should exist in benchmark triangulation",
    );

    simplex
        .vertices()
        .iter()
        .map(|vertex_key| {
            *bench_option(
                dt.tds().vertex(*vertex_key),
                "vertex key should exist in benchmark triangulation",
            )
            .point()
        })
        .collect()
}

fn largest_volume_simplex_4d(dt: &FlipTriangulation4) -> SimplexKey {
    dt.simplices()
        .filter_map(|(simplex_key, _)| {
            simplex_volume(&simplex_points_4d(dt, simplex_key))
                .ok()
                .map(|volume| (simplex_key, volume))
        })
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or_else(
            || {
                abort_benchmark(
                    "stable 4D benchmark triangulation should have a non-degenerate simplex",
                )
            },
            |(simplex_key, _)| simplex_key,
        )
}

fn roundtrip_k1_4d(dt: &mut FlipTriangulation4, simplex_key: SimplexKey) {
    let centroid = simplex_centroid_4d(dt, simplex_key);
    let new_vertex = vertex!(centroid);
    let new_uuid = new_vertex.uuid();

    bench_result(
        dt.flip_k1_insert(simplex_key, new_vertex),
        "k=1 insert should succeed on stable 4D benchmark triangulation",
    );

    let new_key = bench_option(
        dt.tds().vertex_key_from_uuid(&new_uuid),
        "inserted vertex should be present after k=1 insert",
    );

    bench_result(
        dt.flip_k1_remove(new_key),
        "k=1 remove should invert k=1 insert",
    );
}

fn interior_facets_4d(dt: &FlipTriangulation4) -> Vec<FacetHandle> {
    let mut facets = Vec::new();
    for (simplex_key, simplex) in dt.simplices() {
        if let Some(neighbors) = simplex.neighbors() {
            for (facet_index, neighbor) in neighbors.enumerate() {
                if neighbor.is_some() {
                    let Ok(facet_index) = u8::try_from(facet_index) else {
                        continue;
                    };
                    facets.push(FacetHandle::new(simplex_key, facet_index));
                }
            }
        }
    }
    facets
}

fn flippable_k2_facet_4d(dt: &FlipTriangulation4) -> FacetHandle {
    let mut last_error = None;
    for facet in interior_facets_4d(dt) {
        let mut trial = dt.clone();
        match trial.flip_k2(facet) {
            Ok(info) => {
                assert_eq!(
                    info.inserted_face_vertices.len(),
                    2,
                    "k=2 flip should insert an edge"
                );
                let edge = EdgeKey::new(
                    info.inserted_face_vertices[0],
                    info.inserted_face_vertices[1],
                );
                bench_result(
                    trial.flip_k2_inverse_from_edge(edge),
                    "k=2 inverse should succeed after k=2 flip",
                );
                return facet;
            }
            Err(err) => last_error = Some(format!("{err}")),
        }
    }

    abort_benchmark(format_args!(
        "no flippable interior facet found for k=2 benchmark (last error: {})",
        last_error.unwrap_or_else(|| "none".to_string())
    ));
}

fn roundtrip_k2_4d(dt: &mut FlipTriangulation4, facet: FacetHandle) {
    let info = bench_result(
        dt.flip_k2(facet),
        "k=2 flip should succeed for preselected 4D benchmark facet",
    );
    assert_eq!(
        info.inserted_face_vertices.len(),
        2,
        "k=2 flip should insert an edge"
    );
    let edge = EdgeKey::new(
        info.inserted_face_vertices[0],
        info.inserted_face_vertices[1],
    );
    bench_result(
        dt.flip_k2_inverse_from_edge(edge),
        "k=2 inverse should succeed after k=2 flip",
    );
}

fn ridges_4d(dt: &FlipTriangulation4) -> Vec<RidgeHandle> {
    let mut ridges = Vec::new();
    for (simplex_key, simplex) in dt.simplices() {
        let vertex_count = simplex.number_of_vertices();
        for i in 0..vertex_count {
            for j in (i + 1)..vertex_count {
                let Ok(omit_a) = u8::try_from(i) else {
                    continue;
                };
                let Ok(omit_b) = u8::try_from(j) else {
                    continue;
                };
                ridges.push(RidgeHandle::new(simplex_key, omit_a, omit_b));
            }
        }
    }
    ridges
}

fn flippable_k3_ridge_4d(dt: &FlipTriangulation4) -> RidgeHandle {
    let mut last_error = None;
    for ridge in ridges_4d(dt) {
        let mut trial = dt.clone();
        match trial.flip_k3(ridge) {
            Ok(info) => {
                assert_eq!(
                    info.inserted_face_vertices.len(),
                    3,
                    "k=3 flip should insert a triangle"
                );
                let triangle = TriangleHandle::new(
                    info.inserted_face_vertices[0],
                    info.inserted_face_vertices[1],
                    info.inserted_face_vertices[2],
                );
                bench_result(
                    trial.flip_k3_inverse_from_triangle(triangle),
                    "k=3 inverse should succeed after k=3 flip",
                );
                return ridge;
            }
            Err(err) => last_error = Some(format!("{err}")),
        }
    }

    abort_benchmark(format_args!(
        "no flippable ridge found for k=3 benchmark (last error: {})",
        last_error.unwrap_or_else(|| "none".to_string())
    ));
}

fn roundtrip_k3_4d(dt: &mut FlipTriangulation4, ridge: RidgeHandle) {
    let info = bench_result(
        dt.flip_k3(ridge),
        "k=3 flip should succeed for preselected 4D benchmark ridge",
    );
    assert_eq!(
        info.inserted_face_vertices.len(),
        3,
        "k=3 flip should insert a triangle"
    );
    let triangle = TriangleHandle::new(
        info.inserted_face_vertices[0],
        info.inserted_face_vertices[1],
        info.inserted_face_vertices[2],
    );
    bench_result(
        dt.flip_k3_inverse_from_triangle(triangle),
        "k=3 inverse should succeed after k=3 flip",
    );
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
    vertices: &[Vertex<f64, (), D>],
    options: ConstructionOptions,
) {
    let dt = bench_result(
        BenchTriangulation::<D>::new_with_options(vertices, options),
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
                let bounds = (-100.0, 100.0);
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
                    let bounds = (-100.0, 100.0);
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
                    let bounds = (-100.0, 100.0);
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
                        match DelaunayTriangulation::<_, (), (), $dim>::new_with_options(
                            &vertices,
                            options,
                        ) {
                            Ok(dt) => {
                                black_box(dt);
                            }
                            Err(err) => {
                                let error = format!("{err:?}");
                                abort_benchmark(format_args!(
                                    "DelaunayTriangulation::new failed for {}D: {error}; dim={}; count={}; seed={}; bounds={:?}; sample_points={sample_points:?}",
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
                            match DelaunayTriangulation::<_, (), (), $dim>::new_with_options(
                                &vertices,
                                options,
                            ) {
                                Ok(dt) => {
                                    black_box(dt);
                                }
                                Err(err) => {
                                    let error = format!("{err:?}");
                                    abort_benchmark(format_args!(
                                        "adversarial DelaunayTriangulation::new failed for {}D: {error}; dim={}; count={}; seed={}; sample_points={sample_points:?}",
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
            b.iter(|| black_box(dt.boundary_facets().count()));
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
                let hull = match ConvexHull::from_triangulation(dt.as_triangulation()) {
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

fn exterior_hull_query_point<const D: usize>(dt: &BenchTriangulation<D>) -> Point<f64, D> {
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
    Point::new(coords)
}

fn bench_hull_query_case<const D: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    dimension: usize,
    dataset: Dataset,
    count: usize,
    dt: &BenchTriangulation<D>,
) {
    let hull = match ConvexHull::from_triangulation(dt.as_triangulation()) {
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
    insert_vertices: &[Vertex<f64, (), D>],
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

fn benchmark_bistellar_flips(c: &mut Criterion) {
    print_manifest_once();
    if discover_seeds_enabled() {
        return;
    }
    let mut group = c.benchmark_group("bistellar_flips_4d");
    group.sample_size(10);
    let base_dt = build_flip_dt_4d();
    let k1_simplex = largest_volume_simplex_4d(&base_dt);
    let k2_facet = flippable_k2_facet_4d(&base_dt);
    let k3_ridge = flippable_k3_ridge_4d(&base_dt);

    group.bench_function("k1_roundtrip", |b| {
        b.iter_batched(
            || base_dt.clone(),
            |mut dt| {
                roundtrip_k1_4d(&mut dt, k1_simplex);
                black_box(dt);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("k2_roundtrip", |b| {
        b.iter_batched(
            || base_dt.clone(),
            |mut dt| {
                roundtrip_k2_4d(&mut dt, k2_facet);
                black_box(dt);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("k3_roundtrip", |b| {
        b.iter_batched(
            || base_dt.clone(),
            |mut dt| {
                roundtrip_k3_4d(&mut dt, k3_ridge);
                black_box(dt);
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
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
