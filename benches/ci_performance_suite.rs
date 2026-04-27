//! CI Performance Suite - optimized performance regression testing for CI/CD
//!
//! This benchmark is the small, durable performance contract for the delaunay
//! library. It covers the user-facing workflows that should stay fast across
//! releases without duplicating every specialized microbenchmark:
//!
//! 1. Delaunay construction across 2D-5D at CI-sized scales
//! 2. Convex hull extraction from completed triangulations
//! 3. Boundary facet traversal
//! 4. Full validation (Levels 1-4)
//! 5. Explicit bistellar flip roundtrips on a stable 4D PL-manifold case
//!
//! Predicate microbenchmarks, allocation-focused measurements, and large-scale
//! stress tests live in the dedicated benchmark targets under `benches/`.
//!
//! ## Sample Size Strategy
//!
//! Uses dimension-dependent sample sizes to balance accuracy with CI time constraints:
//! - 2D: Default Criterion sample size (100)
//! - 3D: Reduced to 25 samples
//! - 4D/5D: Further reduced to 15 samples for longer-running high-dimensional cases
//!
//! ## Dimensional Focus
//!
//! Tests 2D, 3D, 4D, and 5D triangulations for comprehensive coverage:
//! - 2D: Fundamental triangulation case
//! - 3D-5D: Higher-dimensional triangulations as documented in README.md

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use delaunay::core::vertex::Vertex;
use delaunay::geometry::algorithms::convex_hull::ConvexHull;
use delaunay::geometry::kernel::{AdaptiveKernel, RobustKernel};
use delaunay::geometry::point::Point;
use delaunay::geometry::util::generate_random_points_seeded;
use delaunay::prelude::triangulation::flips::{
    BistellarFlips, CellKey, EdgeKey, FacetHandle, RidgeHandle, TopologyGuarantee, TriangleHandle,
};
use delaunay::prelude::{
    ConstructionOptions, DelaunayTriangulation, InsertionOrderStrategy, RetryPolicy,
};
use delaunay::vertex;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Once;
use tracing::{error, warn};

/// Default point counts for 2D–4D benchmarks.
const COUNTS: &[usize] = &[10, 25, 50];
/// Reduced point counts for 5D (50-point construction is prohibitively slow).
const COUNTS_5D: &[usize] = &[10, 25];
/// Representative operation count for 2D-4D non-construction workflows.
const OPERATION_COUNT: usize = 50;
/// Representative operation count for 5D non-construction workflows.
const OPERATION_COUNT_5D: usize = 25;
type SeedSearchResult<const D: usize> = Option<(u64, Vec<Point<f64, D>>, Vec<Vertex<f64, (), D>>)>;
type BenchTriangulation<const D: usize> = DelaunayTriangulation<AdaptiveKernel<f64>, (), (), D>;
type FlipTriangulation4 = DelaunayTriangulation<RobustKernel<f64>, (), (), 4>;

struct ApiBenchmarkEntry {
    group: &'static str,
    public_api: &'static str,
    dimensions: &'static str,
    benchmark_ids: &'static str,
    note: &'static str,
}

static API_BENCHMARK_MANIFEST: Once = Once::new();

const API_BENCHMARK_ENTRIES: &[ApiBenchmarkEntry] = &[
    ApiBenchmarkEntry {
        group: "construction",
        public_api: "DelaunayTriangulation::new_with_options",
        dimensions: "2,3,4,5",
        benchmark_ids: "tds_new_2d/tds_new/{10,25,50};tds_new_3d/tds_new/{10,25,50};tds_new_4d/tds_new/{10,25,50};tds_new_5d/tds_new/{10,25}",
        note: "construct_from_seeded_vertices",
    },
    ApiBenchmarkEntry {
        group: "boundary_facets",
        public_api: "DelaunayTriangulation::boundary_facets",
        dimensions: "2,3,4,5",
        benchmark_ids: "boundary_facets/boundary_facets_2d/50;boundary_facets/boundary_facets_3d/50;boundary_facets/boundary_facets_4d/50;boundary_facets/boundary_facets_5d/25",
        note: "iterate_boundary_facets",
    },
    ApiBenchmarkEntry {
        group: "convex_hull",
        public_api: "ConvexHull::from_triangulation",
        dimensions: "2,3,4,5",
        benchmark_ids: "convex_hull/from_triangulation_2d/50;convex_hull/from_triangulation_3d/50;convex_hull/from_triangulation_4d/50;convex_hull/from_triangulation_5d/25",
        note: "extract_hull_from_completed_triangulation",
    },
    ApiBenchmarkEntry {
        group: "validation",
        public_api: "DelaunayTriangulation::validate",
        dimensions: "3,4,5",
        benchmark_ids: "validation/validate_3d/50;validation/validate_4d/50;validation/validate_5d/25",
        note: "levels_1_through_4",
    },
    ApiBenchmarkEntry {
        group: "bistellar_flips",
        public_api: "BistellarFlips::{flip_k1_insert,flip_k1_remove,flip_k2,flip_k2_inverse_from_edge,flip_k3,flip_k3_inverse_from_triangle}",
        dimensions: "4",
        benchmark_ids: "bistellar_flips_4d/k1_roundtrip;bistellar_flips_4d/k2_roundtrip;bistellar_flips_4d/k3_roundtrip",
        note: "stable_pl_manifold_roundtrips",
    },
];

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

/// Pre-computed seeds for each (dimension, count) pair.
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
    // 2D
    (2, 10, 52),
    (2, 25, 67),
    (2, 50, 92),
    // 3D
    (3, 10, 133),
    (3, 25, 148),
    (3, 50, 173),
    // 4D
    (4, 10, 466),
    (4, 25, 481),
    (4, 50, 506),
    // 5D (50-point case excluded — too slow for CI)
    (5, 10, 799),
    (5, 25, 816),
];

fn known_seed(dim: usize, count: usize) -> Option<u64> {
    KNOWN_SEEDS
        .iter()
        .find(|&&(d, c, _)| d == dim && c == count)
        .map(|&(_, _, seed)| seed)
}

fn print_api_benchmark_manifest_once() {
    API_BENCHMARK_MANIFEST.call_once(|| {
        println!(
            "api_benchmark_manifest crate=delaunay version={} benchmark=ci_performance_suite schema=1",
            env!("CARGO_PKG_VERSION")
        );
        for entry in API_BENCHMARK_ENTRIES {
            println!(
                "api_benchmark group={} public_api={} dimensions={} benchmark_ids={} note={}",
                entry.group, entry.public_api, entry.dimensions, entry.benchmark_ids, entry.note
            );
        }
    });
}

/// Prepare benchmark inputs by looking up a pre-computed seed, falling back
/// to a runtime search only if the known seed is missing or invalid.
fn prepare_benchmark_data<const D: usize>(
    dim_seed: u64,
    count: usize,
    bounds: (f64, f64),
    attempts: NonZeroUsize,
) -> (u64, Vec<Point<f64, D>>, Vec<Vertex<f64, (), D>>) {
    // Fast path: use the pre-computed seed (single verification construction)
    if let Some(seed) = known_seed(D, count) {
        if let Some(result) = find_seed_and_vertices::<D>(seed, count, bounds, 1, attempts) {
            return result;
        }
        warn!(
            known_seed = seed,
            dim = D,
            count,
            "known seed failed, falling back to runtime search"
        );
    }

    // Slow fallback: runtime search from the base seed
    let base_seed = dim_seed.wrapping_add(count as u64);
    let search_limit = bench_seed_search_limit();
    find_seed_and_vertices::<D>(base_seed, count, bounds, search_limit, attempts).unwrap_or_else(
        || {
            panic!(
                "No stable benchmark seed found for {D}D/{count}: \
                 start_seed={base_seed}; search_limit={search_limit}; bounds={bounds:?}"
            )
        },
    )
}

fn prepare_triangulation<const D: usize>(dim_seed: u64, count: usize) -> BenchTriangulation<D> {
    let bounds = (-100.0, 100.0);
    let attempts = NonZeroUsize::new(6).expect("retry attempts must be non-zero");
    let (seed, _, vertices) = prepare_benchmark_data::<D>(dim_seed, count, bounds, attempts);
    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
        attempts,
        base_seed: Some(seed),
    });

    BenchTriangulation::<D>::new_with_options(&vertices, options).unwrap_or_else(|err| {
        panic!("failed to prepare {D}D benchmark triangulation with {count} vertices: {err}");
    })
}

fn find_seed_and_vertices<const D: usize>(
    start_seed: u64,
    count: usize,
    bounds: (f64, f64),
    limit: usize,
    attempts: NonZeroUsize,
) -> SeedSearchResult<D> {
    for offset in 0..limit {
        let candidate_seed = start_seed.wrapping_add(offset as u64);
        let points = generate_random_points_seeded::<f64, D>(count, bounds, candidate_seed)
            .unwrap_or_else(|error| {
                panic!("generate_random_points_seeded failed for {D}D: {error}");
            });
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

fn stable_vertices_4d() -> Vec<Vertex<f64, (), 4>> {
    STABLE_POINTS_4D
        .iter()
        .map(|coords| vertex!(*coords))
        .collect()
}

fn build_flip_triangulation_4d() -> FlipTriangulation4 {
    let vertices = stable_vertices_4d();
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    DelaunayTriangulation::with_topology_guarantee_and_options(
        &RobustKernel::new(),
        &vertices,
        TopologyGuarantee::PLManifold,
        options,
    )
    .unwrap_or_else(|err| panic!("failed to build stable 4D flip triangulation: {err}"))
}

fn cell_centroid_4d(dt: &FlipTriangulation4, cell_key: CellKey) -> [f64; 4] {
    let cell = dt
        .tds()
        .get_cell(cell_key)
        .expect("cell key should exist in benchmark triangulation");

    let mut coords = [0.0_f64; 4];
    for &vkey in cell.vertices() {
        let vertex = dt
            .tds()
            .get_vertex_by_key(vkey)
            .expect("vertex key should exist in benchmark triangulation");
        let vcoords = vertex.point().coords();
        for i in 0..4 {
            coords[i] += vcoords[i];
        }
    }

    let vertex_count =
        u32::try_from(cell.vertices().len()).expect("cell vertex count should fit in u32");
    let inv = 1.0_f64 / f64::from(vertex_count);
    for coord in &mut coords {
        *coord *= inv;
    }
    coords
}

fn roundtrip_k1_4d(dt: &mut FlipTriangulation4) {
    let cell_key = dt
        .cells()
        .next()
        .map(|(cell_key, _)| cell_key)
        .expect("benchmark triangulation should have cells");
    let centroid = cell_centroid_4d(dt, cell_key);
    let new_vertex = vertex!(centroid);
    let new_uuid = new_vertex.uuid();

    dt.flip_k1_insert(cell_key, new_vertex)
        .expect("k=1 insert should succeed on stable 4D benchmark triangulation");

    let new_key = dt
        .tds()
        .vertex_key_from_uuid(&new_uuid)
        .expect("inserted vertex should be present after k=1 insert");

    dt.flip_k1_remove(new_key)
        .expect("k=1 remove should invert k=1 insert");
}

fn collect_interior_facets_4d(dt: &FlipTriangulation4) -> Vec<FacetHandle> {
    let mut facets = Vec::new();
    for (cell_key, cell) in dt.cells() {
        if let Some(neighbors) = cell.neighbors() {
            for (facet_index, neighbor) in neighbors.iter().enumerate() {
                if neighbor.is_some() {
                    let facet_index = u8::try_from(facet_index).expect("facet index fits in u8");
                    facets.push(FacetHandle::new(cell_key, facet_index));
                }
            }
        }
    }
    facets
}

fn roundtrip_k2_4d(dt: &mut FlipTriangulation4) {
    let mut last_error = None;
    for facet in collect_interior_facets_4d(dt) {
        match dt.flip_k2(facet) {
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
                dt.flip_k2_inverse_from_edge(edge)
                    .expect("k=2 inverse should succeed after k=2 flip");
                return;
            }
            Err(err) => last_error = Some(format!("{err}")),
        }
    }

    panic!(
        "no flippable interior facet found for k=2 benchmark (last error: {})",
        last_error.unwrap_or_else(|| "none".to_string())
    );
}

fn collect_ridges_4d(dt: &FlipTriangulation4) -> Vec<RidgeHandle> {
    let mut ridges = Vec::new();
    for (cell_key, cell) in dt.cells() {
        let vertex_count = cell.number_of_vertices();
        for i in 0..vertex_count {
            for j in (i + 1)..vertex_count {
                let omit_a = u8::try_from(i).expect("ridge index fits in u8");
                let omit_b = u8::try_from(j).expect("ridge index fits in u8");
                ridges.push(RidgeHandle::new(cell_key, omit_a, omit_b));
            }
        }
    }
    ridges
}

fn roundtrip_k3_4d(dt: &mut FlipTriangulation4) {
    let mut last_error = None;
    for ridge in collect_ridges_4d(dt) {
        match dt.flip_k3(ridge) {
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
                dt.flip_k3_inverse_from_triangle(triangle)
                    .expect("k=3 inverse should succeed after k=3 flip");
                return;
            }
            Err(err) => last_error = Some(format!("{err}")),
        }
    }

    panic!(
        "no flippable ridge found for k=3 benchmark (last error: {})",
        last_error.unwrap_or_else(|| "none".to_string())
    );
}

fn bench_logging_enabled() -> bool {
    std::env::var("DELAUNAY_BENCH_LOG").is_ok_and(|value| value != "0")
}

fn bench_discover_seeds_enabled() -> bool {
    std::env::var("DELAUNAY_BENCH_DISCOVER_SEEDS").is_ok_and(|value| value != "0")
}

fn bench_seed_search_limit() -> usize {
    std::env::var("DELAUNAY_BENCH_DISCOVER_SEEDS_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2000)
}

/// Fixed seeds for deterministic triangulation generation across benchmark runs.
/// Using seeded random number generation reduces variance in performance measurements
/// and improves regression detection accuracy in CI environments.
/// Different seeds per dimension ensure triangulations are uncorrelated.
///
/// Macro to reduce duplication in dimensional benchmark functions
macro_rules! benchmark_tds_new_dimension {
    ($dim:literal, $func_name:ident, $seed:literal, $counts:expr) => {
        /// Benchmark triangulation creation for D-dimensional triangulations
        fn $func_name(c: &mut Criterion) {
            print_api_benchmark_manifest_once();
            let counts = $counts;

            // Opt-in helper for discovering stable seeds without paying Criterion warmup/
            // measurement cost per seed.
            //
            // NOTE: This helper is intentionally per (dim, count) benchmark case.
            // It returns early on the first successful seed (and panics on failure),
            // so it is meant to be run with a Criterion filter that selects a single
            // case, for example:
            //
            //     cargo bench --profile perf --bench ci_performance_suite -- 'tds_new_3d/tds_new/50'
            //
            // Because the base seed is derived from `count`, a seed that works for one
            // count may still fail for a different count.
            //
            // We avoid `std::process::exit` here so that destructors run and Criterion
            // can clean up state on both success and failure.
            if bench_discover_seeds_enabled() {
                let bounds = (-100.0, 100.0);
                let filters: Vec<String> = std::env::args()
                    .skip(1)
                    .filter(|arg| !arg.starts_with('-'))
                    .collect();

                for &count in counts {
                    let bench_id =
                        format!("tds_new_{}d/tds_new/{}", stringify!($dim), count);

                    if !filters.is_empty() && !filters.iter().any(|filter| bench_id.contains(filter)) {
                        continue;
                    }

                    let seed = ($seed as u64).wrapping_add(count as u64);
                    let limit = bench_seed_search_limit();
                    let attempts =
                        NonZeroUsize::new(6).expect("retry attempts must be non-zero");

                    if let Some((candidate_seed, _, _)) =
                        find_seed_and_vertices::<$dim>(seed, count, bounds, limit, attempts)
                    {
                        println!(
                            "seed_search_found dim={} count={} seed={}",
                            $dim, count, candidate_seed
                        );
                        return;
                    }

                    panic!(
                        "seed_search_failed dim={} count={} start_seed={} limit={}",
                        $dim,
                        count,
                        seed,
                        limit
                    );
                }

                // No filter matched this benchmark function; do nothing.
                return;
            }

            let mut group = c.benchmark_group(concat!("tds_new_", stringify!($dim), "d"));

            // Set smaller sample sizes for higher dimensions to keep CI times reasonable
            if $dim >= 4 {
                group.sample_size(15); // Fewer samples for 4D and 5D
            } else if $dim == 3 {
                group.sample_size(25); // Medium sample size for 3D
            }

            for &count in counts {
                group.throughput(Throughput::Elements(count as u64));

                group.bench_with_input(BenchmarkId::new("tds_new", count), &count, |b, &count| {
                    // Reduce variance: pre-generate deterministic inputs outside the measured loop,
                    // then benchmark only triangulation construction.
                    let bounds = (-100.0, 100.0);
                    let attempts =
                        NonZeroUsize::new(6).expect("retry attempts must be non-zero");
                    let (seed, points, vertices) =
                        prepare_benchmark_data::<$dim>($seed, count, bounds, attempts);
                    let sample_points = points.iter().take(5).collect::<Vec<_>>();

                    // In benchmarks we compile in release mode, where the default retry policy is
                    // disabled. For deterministic CI benchmarks we opt into a small number of
                    // shuffled retries to avoid aborting the suite on rare non-convergent repair
                    // cases.
                    let options = ConstructionOptions::default().with_retry_policy(RetryPolicy::Shuffled {
                        attempts,
                        base_seed: Some(seed),
                    });

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
                                if bench_logging_enabled() {
                                    error!(
                                        dim = $dim,
                                        count,
                                        seed,
                                        bounds = ?bounds,
                                        sample_points = ?sample_points,
                                        error = %error,
                                        "DelaunayTriangulation::new failed"
                                    );
                                }
                                panic!(
                                    "DelaunayTriangulation::new failed for {}D: {error}; dim={}; count={}; seed={}; bounds={:?}; sample_points={sample_points:?}",
                                    $dim,
                                    $dim,
                                    count,
                                    seed,
                                    bounds
                                );
                            }
                        }
                    });
                });
            }

            group.finish();
        }
    };
}

// Generate benchmark functions using the macro
benchmark_tds_new_dimension!(2, benchmark_tds_new_2d, 42, COUNTS);
benchmark_tds_new_dimension!(3, benchmark_tds_new_3d, 123, COUNTS);
benchmark_tds_new_dimension!(4, benchmark_tds_new_4d, 456, COUNTS);
benchmark_tds_new_dimension!(5, benchmark_tds_new_5d, 789, COUNTS_5D);

fn benchmark_boundary_facets(c: &mut Criterion) {
    print_api_benchmark_manifest_once();
    let mut group = c.benchmark_group("boundary_facets");
    group.sample_size(25);

    let dt_2d = prepare_triangulation::<2>(42, OPERATION_COUNT);
    group.throughput(Throughput::Elements(OPERATION_COUNT as u64));
    group.bench_function(
        BenchmarkId::new("boundary_facets_2d", OPERATION_COUNT),
        |b| {
            b.iter(|| black_box(dt_2d.boundary_facets().count()));
        },
    );

    let dt_3d = prepare_triangulation::<3>(123, OPERATION_COUNT);
    group.throughput(Throughput::Elements(OPERATION_COUNT as u64));
    group.bench_function(
        BenchmarkId::new("boundary_facets_3d", OPERATION_COUNT),
        |b| {
            b.iter(|| black_box(dt_3d.boundary_facets().count()));
        },
    );

    let dt_4d = prepare_triangulation::<4>(456, OPERATION_COUNT);
    group.throughput(Throughput::Elements(OPERATION_COUNT as u64));
    group.bench_function(
        BenchmarkId::new("boundary_facets_4d", OPERATION_COUNT),
        |b| {
            b.iter(|| black_box(dt_4d.boundary_facets().count()));
        },
    );

    let dt_5d = prepare_triangulation::<5>(789, OPERATION_COUNT_5D);
    group.throughput(Throughput::Elements(OPERATION_COUNT_5D as u64));
    group.bench_function(
        BenchmarkId::new("boundary_facets_5d", OPERATION_COUNT_5D),
        |b| {
            b.iter(|| black_box(dt_5d.boundary_facets().count()));
        },
    );

    group.finish();
}

fn benchmark_convex_hull(c: &mut Criterion) {
    print_api_benchmark_manifest_once();
    let mut group = c.benchmark_group("convex_hull");
    group.sample_size(20);

    let dt_2d = prepare_triangulation::<2>(42, OPERATION_COUNT);
    group.throughput(Throughput::Elements(OPERATION_COUNT as u64));
    group.bench_function(
        BenchmarkId::new("from_triangulation_2d", OPERATION_COUNT),
        |b| {
            b.iter(|| {
                black_box(
                    ConvexHull::from_triangulation(dt_2d.as_triangulation())
                        .expect("2D convex hull extraction should succeed"),
                );
            });
        },
    );

    let dt_3d = prepare_triangulation::<3>(123, OPERATION_COUNT);
    group.throughput(Throughput::Elements(OPERATION_COUNT as u64));
    group.bench_function(
        BenchmarkId::new("from_triangulation_3d", OPERATION_COUNT),
        |b| {
            b.iter(|| {
                black_box(
                    ConvexHull::from_triangulation(dt_3d.as_triangulation())
                        .expect("3D convex hull extraction should succeed"),
                );
            });
        },
    );

    let dt_4d = prepare_triangulation::<4>(456, OPERATION_COUNT);
    group.throughput(Throughput::Elements(OPERATION_COUNT as u64));
    group.bench_function(
        BenchmarkId::new("from_triangulation_4d", OPERATION_COUNT),
        |b| {
            b.iter(|| {
                black_box(
                    ConvexHull::from_triangulation(dt_4d.as_triangulation())
                        .expect("4D convex hull extraction should succeed"),
                );
            });
        },
    );

    let dt_5d = prepare_triangulation::<5>(789, OPERATION_COUNT_5D);
    group.throughput(Throughput::Elements(OPERATION_COUNT_5D as u64));
    group.bench_function(
        BenchmarkId::new("from_triangulation_5d", OPERATION_COUNT_5D),
        |b| {
            b.iter(|| {
                black_box(
                    ConvexHull::from_triangulation(dt_5d.as_triangulation())
                        .expect("5D convex hull extraction should succeed"),
                );
            });
        },
    );

    group.finish();
}

fn benchmark_validation(c: &mut Criterion) {
    print_api_benchmark_manifest_once();
    let mut group = c.benchmark_group("validation");
    group.sample_size(15);

    let dt_3d = prepare_triangulation::<3>(123, OPERATION_COUNT);
    group.throughput(Throughput::Elements(OPERATION_COUNT as u64));
    group.bench_function(BenchmarkId::new("validate_3d", OPERATION_COUNT), |b| {
        b.iter(|| {
            black_box(dt_3d.validate()).expect("3D benchmark triangulation should validate");
        });
    });

    let dt_4d = prepare_triangulation::<4>(456, OPERATION_COUNT);
    group.throughput(Throughput::Elements(OPERATION_COUNT as u64));
    group.bench_function(BenchmarkId::new("validate_4d", OPERATION_COUNT), |b| {
        b.iter(|| {
            black_box(dt_4d.validate()).expect("4D benchmark triangulation should validate");
        });
    });

    let dt_5d = prepare_triangulation::<5>(789, OPERATION_COUNT_5D);
    group.throughput(Throughput::Elements(OPERATION_COUNT_5D as u64));
    group.bench_function(BenchmarkId::new("validate_5d", OPERATION_COUNT_5D), |b| {
        b.iter(|| {
            black_box(dt_5d.validate()).expect("5D benchmark triangulation should validate");
        });
    });

    group.finish();
}

fn benchmark_bistellar_flips(c: &mut Criterion) {
    print_api_benchmark_manifest_once();
    let mut group = c.benchmark_group("bistellar_flips_4d");
    group.sample_size(10);
    let base_dt = build_flip_triangulation_4d();

    group.bench_function("k1_roundtrip", |b| {
        b.iter_batched(
            || base_dt.clone(),
            |mut dt| {
                roundtrip_k1_4d(&mut dt);
                black_box(dt);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("k2_roundtrip", |b| {
        b.iter_batched(
            || base_dt.clone(),
            |mut dt| {
                roundtrip_k2_4d(&mut dt);
                black_box(dt);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("k3_roundtrip", |b| {
        b.iter_batched(
            || base_dt.clone(),
            |mut dt| {
                roundtrip_k3_4d(&mut dt);
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
        benchmark_validation,
        benchmark_bistellar_flips
);
criterion_main!(benches);
