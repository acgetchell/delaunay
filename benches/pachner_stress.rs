#![forbid(unsafe_code)]

//! Stress benchmarks for the unified Pachner move API.
//!
//! This harness measures repeated accepted [`PachnerMove`] attempts on stable
//! PL-manifold fixtures. It complements `ci_performance_suite` by focusing on
//! the unified dispatch facade rather than the individual flip primitives.

use std::{
    env,
    hint::black_box,
    num::{NonZeroUsize, TryFromIntError},
    sync::LazyLock,
};

use criterion::{
    BatchSize, BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use delaunay::prelude::construction::{Vertex, vertex};
use delaunay::prelude::pachner::{
    EdgeKey, FacetHandle, PachnerMove, PachnerMoveResult, PachnerMoves, RidgeHandle, SimplexKey,
    TriangleHandle, VertexKey,
};

/// Shared benchmark setup error helpers.
#[path = "common/bench_utils.rs"]
pub mod bench_utils;
use bench_utils::{OrAbort, OrAbortWithContext};

#[path = "common/flip_fixtures.rs"]
#[expect(
    dead_code,
    reason = "shared fixture catalog is intentionally broader than this 4D stress target"
)]
mod flip_fixtures;
use flip_fixtures::STABLE_POINTS_4D;

#[path = "common/flip_workflows.rs"]
#[expect(
    dead_code,
    reason = "shared flip workflow module is reused by broader benchmark and test targets"
)]
mod flip_workflows;
use flip_workflows::{CandidateFilter, FlipTriangulation};

static MOVES_PER_SAMPLE: LazyLock<NonZeroUsize> =
    LazyLock::new(|| nonzero_usize("MOVES_PER_SAMPLE", 256));

struct PachnerStressSetup {
    base_dt: FlipTriangulation<4>,
    simplex_key: SimplexKey,
    k1_vertex: Vertex<(), 4>,
    k1_remove_dt: FlipTriangulation<4>,
    k1_remove_vertex_key: VertexKey,
    facet: FacetHandle,
    k2_inverse_dt: FlipTriangulation<4>,
    k2_inverse_edge: EdgeKey,
    ridge: RidgeHandle,
    k3_inverse_dt: FlipTriangulation<4>,
    k3_inverse_triangle: TriangleHandle,
}

/// Reads a positive `usize` override, preserving the non-zero proof.
fn nonzero_usize(name: &str, default: usize) -> NonZeroUsize {
    let value = env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default);
    NonZeroUsize::new(value).or_abort(format_args!("{name} must be non-zero"))
}

/// Builds one stable 4D fixture and selects deterministic accepted move supports.
fn stress_setup() -> PachnerStressSetup {
    let base_dt = flip_workflows::build_flip_dt(STABLE_POINTS_4D).or_abort();
    let simplex_key =
        flip_workflows::largest_volume_simplex(&base_dt, CandidateFilter::Any).or_abort();
    let k1_coords = simplex_centroid(&base_dt, simplex_key).or_abort();
    let k1_vertex: Vertex<(), 4> = vertex!(k1_coords).or_abort();
    let facet = flip_workflows::flippable_k2_facet(&base_dt, true, CandidateFilter::Any).or_abort();
    let ridge = flip_workflows::flippable_k3_ridge(&base_dt, true, CandidateFilter::Any).or_abort();
    let (k1_remove_dt, k1_remove_vertex_key) = k1_remove_fixture(&base_dt, simplex_key, k1_vertex);
    let (k2_inverse_dt, k2_inverse_edge) = k2_inverse_fixture(&base_dt, facet);
    let (k3_inverse_dt, k3_inverse_triangle) = k3_inverse_fixture(&base_dt, ridge);

    PachnerStressSetup {
        base_dt,
        simplex_key,
        k1_vertex,
        k1_remove_dt,
        k1_remove_vertex_key,
        facet,
        k2_inverse_dt,
        k2_inverse_edge,
        ridge,
        k3_inverse_dt,
        k3_inverse_triangle,
    }
}

/// Computes a simplex centroid for the k=1 insertion stress move.
fn simplex_centroid(
    dt: &FlipTriangulation<4>,
    simplex_key: SimplexKey,
) -> Result<[f64; 4], TryFromIntError> {
    let simplex = dt
        .simplex(simplex_key)
        .or_abort(format_args!("missing selected simplex {simplex_key:?}"));
    let mut coords = [0.0; 4];
    for &vertex_key in simplex.vertices() {
        let vertex = dt
            .vertex(vertex_key)
            .or_abort(format_args!("missing simplex vertex {vertex_key:?}"));
        for (coord, value) in coords.iter_mut().zip(vertex.point().coords()) {
            *coord += *value;
        }
    }

    let vertex_count = u32::try_from(simplex.vertices().len()).map(f64::from)?;
    for coord in &mut coords {
        *coord /= vertex_count;
    }
    Ok(coords)
}

/// Builds a fixture where a k=1 inverse move is accepted.
fn k1_remove_fixture(
    base_dt: &FlipTriangulation<4>,
    simplex_key: SimplexKey,
    vertex: Vertex<(), 4>,
) -> (FlipTriangulation<4>, VertexKey) {
    let mut dt = base_dt.clone();
    let vertex_uuid = vertex.uuid();
    let inserted = attempt_pachner_move(
        &mut dt,
        PachnerMove::K1Insert {
            simplex_key,
            vertex,
        },
    );
    let vertex_key = dt
        .vertices()
        .find_map(|(vertex_key, vertex)| (vertex.uuid() == vertex_uuid).then_some(vertex_key))
        .or_abort(format_args!("missing inserted k=1 vertex {vertex_uuid}"));
    assert_eq!(inserted.inserted_face_vertices.as_slice(), &[vertex_key]);
    assert!(!inserted.new_simplices.is_empty());

    (dt, vertex_key)
}

/// Builds a fixture where a k=2 inverse move is accepted.
fn k2_inverse_fixture(
    base_dt: &FlipTriangulation<4>,
    facet: FacetHandle,
) -> (FlipTriangulation<4>, EdgeKey) {
    let mut dt = base_dt.clone();
    let info = attempt_pachner_move(&mut dt, PachnerMove::K2 { facet });
    let edge = inserted_edge(&dt, &info.inserted_face_vertices);

    (dt, edge)
}

/// Builds a fixture where a k=3 inverse move is accepted.
fn k3_inverse_fixture(
    base_dt: &FlipTriangulation<4>,
    ridge: RidgeHandle,
) -> (FlipTriangulation<4>, TriangleHandle) {
    let mut dt = base_dt.clone();
    let info = attempt_pachner_move(&mut dt, PachnerMove::K3 { ridge });
    let triangle = inserted_triangle(&info.inserted_face_vertices);

    (dt, triangle)
}

/// Parses and commits one Pachner request on the same topology owner.
fn attempt_pachner_move(
    dt: &mut FlipTriangulation<4>,
    pachner_move: PachnerMove<(), 4>,
) -> PachnerMoveResult<4> {
    dt.propose_pachner(pachner_move)
        .or_abort()
        .attempt_on(dt)
        .or_abort()
}

/// Converts a forward Pachner result into the complementary inverse request.
fn inverse_move_from_forward_result(
    dt: &FlipTriangulation<4>,
    result: &PachnerMoveResult<4>,
) -> PachnerMove<(), 4> {
    match result.inserted_face_vertices.as_slice() {
        [vertex_key] => PachnerMove::K1Remove {
            vertex_key: *vertex_key,
        },
        vertices @ [_, _] => PachnerMove::K2Inverse {
            edge: inserted_edge(dt, vertices),
        },
        vertices @ [_, _, _] => PachnerMove::K3Inverse {
            triangle: inserted_triangle(vertices),
        },
        vertices => Option::<PachnerMove<(), 4>>::None.or_abort(format_args!(
            "forward Pachner move reported {} inserted-face vertices",
            vertices.len()
        )),
    }
}

/// Converts a reported inserted face into an inverse k=2 edge handle.
fn inserted_edge(dt: &FlipTriangulation<4>, vertices: &[VertexKey]) -> EdgeKey {
    let [a, b] = vertices else {
        return Option::<EdgeKey>::None.or_abort(format_args!(
            "k=2 flip reported {} inserted-face vertices",
            vertices.len()
        ));
    };
    dt.edges()
        .find(|edge| {
            let (first, second) = edge.endpoints();
            (first == *a && second == *b) || (first == *b && second == *a)
        })
        .or_abort(format_args!("inserted k=2 edge {a:?}-{b:?} is missing"))
}

/// Converts a reported inserted face into an inverse k=3 triangle handle.
fn inserted_triangle(vertices: &[VertexKey]) -> TriangleHandle {
    let [a, b, c] = vertices else {
        return Option::<TriangleHandle>::None.or_abort(format_args!(
            "k=3 flip reported {} inserted-face vertices",
            vertices.len()
        ));
    };
    TriangleHandle::try_new(*a, *b, *c).or_abort()
}

/// Creates one batch of independent triangulation clones for repeated move attempts.
fn clone_batch(base_dt: &FlipTriangulation<4>) -> Vec<FlipTriangulation<4>> {
    vec![base_dt.clone(); (*MOVES_PER_SAMPLE).get()]
}

/// Registers one stress case that repeats the same raw Pachner request.
///
/// `bench_pachner_move` measures `attempt_pachner_move`, so each sample includes
/// `propose_pachner` feasibility work and `attempt_on` revalidation before
/// mutation, not only the raw flip mutation cost.
fn bench_pachner_move(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &'static str,
    base_dt: &FlipTriangulation<4>,
    pachner_move: PachnerMove<(), 4>,
) {
    group.bench_function(name, move |b| {
        b.iter_batched(
            || clone_batch(base_dt),
            |mut triangulations| {
                for dt in &mut triangulations {
                    let result = attempt_pachner_move(dt, pachner_move);
                    black_box(&result);
                }
                black_box(triangulations);
            },
            BatchSize::LargeInput,
        );
    });
}

/// Registers one stress case that commits a forward move and its inverse.
fn bench_pachner_round_trip(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &'static str,
    base_dt: &FlipTriangulation<4>,
    pachner_move: PachnerMove<(), 4>,
) {
    group.bench_function(name, move |b| {
        b.iter_batched(
            || clone_batch(base_dt),
            |mut triangulations| {
                for dt in &mut triangulations {
                    let forward = attempt_pachner_move(dt, pachner_move);
                    let inverse = inverse_move_from_forward_result(dt, &forward);
                    let backward = attempt_pachner_move(dt, inverse);
                    black_box((&forward, &backward));
                }
                black_box(triangulations);
            },
            BatchSize::LargeInput,
        );
    });
}

/// Runs the unified Pachner move stress benchmark group.
fn pachner_stress(c: &mut Criterion) {
    let setup = stress_setup();
    let mut group = c.benchmark_group("pachner_stress");
    group.throughput(Throughput::Elements(
        u64::try_from((*MOVES_PER_SAMPLE).get()).or_abort(),
    ));

    bench_pachner_move(
        &mut group,
        "k1_insert",
        &setup.base_dt,
        PachnerMove::K1Insert {
            simplex_key: setup.simplex_key,
            vertex: setup.k1_vertex,
        },
    );
    bench_pachner_move(
        &mut group,
        "k1_remove",
        &setup.k1_remove_dt,
        PachnerMove::K1Remove {
            vertex_key: setup.k1_remove_vertex_key,
        },
    );
    bench_pachner_move(
        &mut group,
        "k2",
        &setup.base_dt,
        PachnerMove::K2 { facet: setup.facet },
    );
    bench_pachner_move(
        &mut group,
        "k2_inverse",
        &setup.k2_inverse_dt,
        PachnerMove::K2Inverse {
            edge: setup.k2_inverse_edge,
        },
    );
    bench_pachner_move(
        &mut group,
        "k3",
        &setup.base_dt,
        PachnerMove::K3 { ridge: setup.ridge },
    );
    bench_pachner_move(
        &mut group,
        "k3_inverse",
        &setup.k3_inverse_dt,
        PachnerMove::K3Inverse {
            triangle: setup.k3_inverse_triangle,
        },
    );
    bench_pachner_round_trip(
        &mut group,
        "round_trip_k1",
        &setup.base_dt,
        PachnerMove::K1Insert {
            simplex_key: setup.simplex_key,
            vertex: setup.k1_vertex,
        },
    );
    bench_pachner_round_trip(
        &mut group,
        "round_trip_k2",
        &setup.base_dt,
        PachnerMove::K2 { facet: setup.facet },
    );
    bench_pachner_round_trip(
        &mut group,
        "round_trip_k3",
        &setup.base_dt,
        PachnerMove::K3 { ridge: setup.ridge },
    );

    group.finish();
}

criterion_group!(benches, pachner_stress);
criterion_main!(benches);
