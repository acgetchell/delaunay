#![forbid(unsafe_code)]

//! Public API roundtrip tests for Pachner/bistellar flips.

use delaunay::flips::{BistellarFlips, FlipFailureKind, FlipFeasibility, FlipMutationError};
use delaunay::{TdsConstructionFailure, vertex};
use std::assert_matches;

use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayError, DelaunayResult, DelaunayTriangulation,
    DelaunayTriangulationBuilder, InsertionOrderStrategy, TopologyGuarantee, Vertex,
};
use delaunay::prelude::geometry::RobustKernel;
#[cfg(feature = "slow-tests")]
use delaunay::prelude::pachner::RidgeHandle;
use delaunay::prelude::pachner::{
    BistellarFlipKind, EdgeKey, EdgeKeyError, FacetHandle, FlipDirection, FlipError, PachnerMove,
    PachnerMoveFeasibility, PachnerMoveResult, PachnerMoves, PachnerProposal, SimplexKey,
    TopologyOwner, TriangleHandle, VertexKey,
};
use uuid::Uuid;

type Dt4 = DelaunayTriangulation<RobustKernel<f64>, (), (), 4>;
type Dt2 = DelaunayTriangulation<RobustKernel<f64>, (), (), 2>;
type Dt<const D: usize> = DelaunayTriangulation<RobustKernel<f64>, (), (), D>;

const FLIPPABLE_POINTS_2D: &[[f64; 2]] = &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

const MINIMAL_POINTS_4D: &[[f64; 4]] = &[
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn vertex_key_by_uuid<const D: usize>(dt: &Dt<D>, uuid: Uuid) -> Option<VertexKey> {
    dt.vertices()
        .find_map(|(vertex_key, vertex)| (vertex.uuid() == uuid).then_some(vertex_key))
}

fn find_live_edge<const D: usize>(
    dt: &Dt<D>,
    a: VertexKey,
    b: VertexKey,
) -> Result<EdgeKey, EdgeKeyError> {
    if a == b {
        return Err(EdgeKeyError::DuplicateEndpoint { endpoint: a });
    }
    if !dt.contains_vertex_key(a) {
        return Err(EdgeKeyError::MissingEndpoint { endpoint: a });
    }
    if !dt.contains_vertex_key(b) {
        return Err(EdgeKeyError::MissingEndpoint { endpoint: b });
    }
    dt.edges()
        .find(|edge| {
            let (first, second) = edge.endpoints();
            (first == a && second == b) || (first == b && second == a)
        })
        .ok_or(EdgeKeyError::EdgeNotFound { v0: a, v1: b })
}

#[cfg(feature = "slow-tests")]
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct TopologySnapshot {
    vertex_uuids: Vec<Uuid>,
    simplex_vertex_uuids: Vec<Vec<Uuid>>,
}

fn attempt_pachner_move<const D: usize>(
    dt: &mut Dt<D>,
    pachner_move: PachnerMove<(), D>,
) -> Result<PachnerMoveResult<D>, FlipError> {
    dt.propose_pachner(pachner_move)?.attempt_on(dt)
}

fn topology_and_delaunay_valid<const D: usize>(
    dt: &DelaunayTriangulation<RobustKernel<f64>, (), (), D>,
) -> bool {
    dt.as_triangulation().validate().is_ok()
        && dt.as_triangulation().is_valid_realization().is_ok()
        && dt.is_valid_delaunay().is_ok()
}

fn assert_topology_and_delaunay_valid<const D: usize>(
    dt: &DelaunayTriangulation<RobustKernel<f64>, (), (), D>,
    context: &str,
) {
    dt.as_triangulation()
        .validate()
        .unwrap_or_else(|err| panic!("{context} should pass Levels 1-3: {err}"));
    dt.as_triangulation()
        .is_valid_realization()
        .unwrap_or_else(|err| panic!("{context} should pass Level 4 realization: {err}"));
    dt.is_valid_delaunay()
        .unwrap_or_else(|err| panic!("{context} should pass Level 5: {err}"));
}

#[test]
#[cfg(feature = "slow-tests")]
fn public_pachner_roundtrips_preserve_stable_4d_topology() {
    let base = build_stable_dt_4d();
    assert_topology_and_delaunay_valid(&base, "stable 4D fixture");
    let before = snapshot_topology(&base);

    let mut k1 = base.clone();
    roundtrip_k1(&mut k1);
    assert_topology_and_delaunay_valid(&k1, "k=1 roundtrip");
    assert_eq!(snapshot_topology(&k1), before);

    let k2_facet = flippable_k2_facet(&base);
    let mut k2 = base.clone();
    roundtrip_k2(&mut k2, k2_facet);
    assert_topology_and_delaunay_valid(&k2, "k=2 roundtrip");
    assert_eq!(snapshot_topology(&k2), before);

    let k3_ridge = flippable_k3_ridge(&base);
    let mut k3 = base;
    roundtrip_k3(&mut k3, k3_ridge);
    assert_topology_and_delaunay_valid(&k3, "k=3 roundtrip");
    assert_eq!(snapshot_topology(&k3), before);
}

#[test]
fn stale_k1_insert_request_fails_without_mutating_topology() {
    let base = build_minimal_dt_4d();
    assert_stale_k1_insert_preserves_topology(base);
}

#[test]
fn stale_k1_remove_request_fails_without_mutating_topology() {
    let base = build_minimal_dt_4d();
    assert_stale_k1_remove_preserves_topology(base);
}

#[test]
fn pachner_proposal_rejects_different_topology_owner_without_mutating() {
    let dt = build_minimal_simplex_dt::<2>();
    let simplex_key = first_simplex_generic(&dt);
    let vertex: Vertex<(), 2> = vertex!(simplex_centroid_generic(&dt, simplex_key))
        .expect("simplex centroid should be valid");
    let proposal = dt
        .propose_pachner(PachnerMove::K1Insert {
            simplex_key,
            vertex,
        })
        .expect("source proposal should be valid");

    let mut target = build_minimal_simplex_dt::<2>();
    let before = snapshot_topology_2d(&target);
    let expected_owner = target.topology_owner_id();
    let found_owner = proposal.owner_id().clone();
    let feasibility = proposal
        .can_attempt_on(&target)
        .expect_err("independent triangulation should reject proposal from original owner");
    assert_matches!(
        &feasibility,
        FlipError::WrongTopologyOwner { expected, found }
            if expected == &expected_owner && found == &found_owner
    );
    assert_eq!(
        FlipFailureKind::from(&feasibility),
        FlipFailureKind::WrongTopologyOwner
    );
    assert_eq!(snapshot_topology_2d(&target), before);

    let attempted = proposal
        .attempt_on(&mut target)
        .expect_err("independent triangulation should not mutate from original proposal");
    assert_matches!(
        &attempted,
        FlipError::WrongTopologyOwner { expected, found }
            if expected == &expected_owner && found == &found_owner
    );
    assert_eq!(
        FlipFailureKind::from(&attempted),
        FlipFailureKind::WrongTopologyOwner
    );
    assert_eq!(snapshot_topology_2d(&target), before);
}

#[test]
#[cfg(feature = "slow-tests")]
fn stale_k2_request_fails_without_mutating_topology() {
    let base = build_stable_dt_4d();
    assert_stale_k2_preserves_topology(base);
}

#[test]
#[cfg(feature = "slow-tests")]
fn stale_k2_inverse_request_fails_without_mutating_topology() {
    let base = build_stable_dt_4d();
    assert_stale_k2_inverse_preserves_topology(base);
}

#[test]
#[cfg(feature = "slow-tests")]
fn stale_k3_request_fails_without_mutating_topology() {
    let base = build_stable_dt_4d();
    assert_stale_k3_preserves_topology(base);
}

#[test]
#[cfg(feature = "slow-tests")]
fn stale_k3_inverse_request_fails_without_mutating_topology() {
    let base = build_stable_dt_4d();
    assert_stale_k3_inverse_preserves_topology(base);
}

#[test]
fn stale_pachner_error_propagates_through_delaunay_result() {
    let mut dt = build_minimal_dt_4d();
    let stale_simplex = first_simplex(&dt);
    let vertex_coords = simplex_centroid(&dt, stale_simplex);
    let vertex: Vertex<(), 4> =
        vertex!(vertex_coords).expect("centroid of a stable simplex should be a valid vertex");
    let vertex_uuid = vertex.uuid();
    let inserted = attempt_pachner_move(
        &mut dt,
        PachnerMove::K1Insert {
            simplex_key: stale_simplex,
            vertex,
        },
    )
    .expect("initial k=1 insert should make the simplex key stale");
    let inserted_vertex = vertex_key_by_uuid(&dt, vertex_uuid)
        .expect("initial k=1 insert should create the requested vertex");
    assert_k1_insert_result(&inserted, inserted_vertex);
    let before_failed_attempt = snapshot_topology(&dt);

    let err = try_stale_k1_insert(&mut dt, stale_simplex, vertex_coords)
        .expect_err("stale Pachner failure should propagate through DelaunayResult");
    assert_matches!(
        &err,
        DelaunayError::Flip { source }
            if matches!(
                source.as_ref(),
                FlipError::MissingSimplex { simplex_key } if *simplex_key == stale_simplex
            ),
        "unexpected DelaunayResult error for stale Pachner move: {err:?}"
    );
    dt.is_valid_structure()
        .expect("failed Pachner attempt should preserve TDS validity");
    assert_eq!(snapshot_topology(&dt), before_failed_attempt);
}

#[test]
fn edge_to_facet_query_tracks_2d_k2_mutation_freshness() {
    let mut dt = build_flippable_dt_2d();
    let facet = flippable_k2_facet_2d(&dt);
    let old_edge = edge_for_facet_2d(&dt, facet);

    let old_incident_facets: Vec<_> = dt
        .try_incident_facets_to_edge_2d(old_edge)
        .unwrap()
        .collect();
    assert_eq!(old_incident_facets.len(), 2);
    assert!(old_incident_facets.contains(&facet));
    assert!(
        old_incident_facets
            .iter()
            .all(|&incident_facet| edge_for_facet_2d(&dt, incident_facet) == old_edge)
    );
    assert!(
        dt.try_interior_facet_for_edge_2d(old_edge)
            .unwrap()
            .is_some()
    );

    let info = attempt_pachner_move(&mut dt, PachnerMove::K2 { facet })
        .expect("2D k=2 flip should succeed on selected fixture facet");
    assert_eq!(info.inserted_face_vertices.len(), 2);

    match dt.try_incident_facets_to_edge_2d(old_edge) {
        Err(EdgeKeyError::EdgeNotFound { .. }) => {}
        Err(err) => panic!("expected old edge to be absent after k=2 flip, got {err:?}"),
        Ok(_) => panic!("expected old edge to be absent after k=2 flip, got facets"),
    }
    assert_matches!(
        dt.try_interior_facet_for_edge_2d(old_edge),
        Err(EdgeKeyError::EdgeNotFound { .. })
    );

    let new_edge = inserted_edge_2d(&dt, &info.inserted_face_vertices);
    let new_incident_facets: Vec<_> = dt
        .try_incident_facets_to_edge_2d(new_edge)
        .unwrap()
        .collect();
    assert_eq!(new_incident_facets.len(), 2);
    assert!(
        new_incident_facets
            .iter()
            .all(|&incident_facet| edge_for_facet_2d(&dt, incident_facet) == new_edge)
    );
    assert!(
        dt.try_interior_facet_for_edge_2d(new_edge)
            .unwrap()
            .is_some()
    );
    assert_topology_and_delaunay_valid(&dt, "2D k=2 mutation-freshness fixture");
}

#[test]
fn pachner_feasibility_agrees_with_successful_2d_k2_attempt() {
    let dt = build_flippable_dt_2d();
    let facet = flippable_k2_facet_2d(&dt);
    let pachner_move = PachnerMove::K2 { facet };

    let proposal = dt
        .propose_pachner(pachner_move)
        .expect("2D k=2 proposal parsing should accept selected fixture facet");
    let feasibility = proposal
        .can_attempt_on(&dt)
        .expect("2D k=2 feasibility should accept selected fixture facet")
        .clone();
    assert_pachner_feasibility_contract(
        &feasibility,
        BistellarFlipKind::k2(2),
        FlipDirection::Forward,
    );

    let mut trial = dt;
    let result = proposal
        .attempt_on(&mut trial)
        .expect("2D k=2 attempt should agree with feasibility");
    assert_eq!(feasibility.kind, result.kind);
    assert_eq!(feasibility.direction, result.direction);
    assert_eq!(feasibility.removed_simplices, result.removed_simplices);
    assert_eq!(
        feasibility.removed_face_vertices,
        result.removed_face_vertices
    );
    assert_eq!(
        feasibility.inserted_face_vertices.as_ref(),
        Some(&result.inserted_face_vertices)
    );
}

#[test]
fn pachner_feasibility_rejects_unsupported_2d_k2_inverse_without_mutating() {
    let mut dt = build_flippable_dt_2d();
    let facet = flippable_k2_facet_2d(&dt);
    let forward = attempt_pachner_move(&mut dt, PachnerMove::K2 { facet })
        .expect("2D k=2 attempt should create an inverse edge candidate");
    let edge = inserted_edge_2d(&dt, &forward.inserted_face_vertices);
    let pachner_move = PachnerMove::K2Inverse { edge };
    let before = snapshot_topology_2d(&dt);

    assert_matches!(
        dt.can_flip_k2_inverse_from_edge(edge),
        Err(FlipError::UnsupportedDimension { dimension: 2 })
    );
    assert_eq!(snapshot_topology_2d(&dt), before);
    assert_pachner_rejection_preserves_topology(dt, pachner_move, |err| {
        assert_matches!(err, FlipError::UnsupportedDimension { dimension: 2 });
    });
}

#[test]
fn pachner_feasibility_rejects_boundary_facet_like_attempt_2d() {
    let dt = build_single_triangle_dt_2d();
    let facet = dt
        .boundary_facets()
        .expect("single-triangle fixture should classify boundary facets")
        .next()
        .expect("single-triangle fixture should expose a boundary facet")
        .expect("boundary facet should reborrow as a live view")
        .handle();
    let pachner_move = PachnerMove::K2 { facet };

    let feasibility = dt.propose_pachner(pachner_move);
    assert_matches!(feasibility, Err(FlipError::BoundaryFacet { .. }));

    let trial = dt;
    let attempted = trial.propose_pachner(pachner_move);
    assert_matches!(attempted, Err(FlipError::BoundaryFacet { .. }));
    assert_topology_and_delaunay_valid(&trial, "failed boundary feasibility agreement");
}

#[test]
fn pachner_feasibility_rejects_stale_facet_like_attempt_2d() {
    let mut dt = build_flippable_dt_2d();
    let facet = flippable_k2_facet_2d(&dt);
    let first_flip = attempt_pachner_move(&mut dt, PachnerMove::K2 { facet })
        .expect("first k=2 attempt should stale the original facet");
    assert!(!first_flip.new_simplices.is_empty());
    let stale_move = PachnerMove::K2 { facet };

    let feasibility = dt.propose_pachner(stale_move);
    assert_matches!(
        feasibility,
        Err(FlipError::MissingSimplex { simplex_key }) if simplex_key == facet.simplex_key()
    );

    let before_failed_attempt = snapshot_topology_2d(&dt);
    let attempted = dt.propose_pachner(stale_move);
    assert_matches!(
        attempted,
        Err(FlipError::MissingSimplex { simplex_key }) if simplex_key == facet.simplex_key()
    );
    assert_eq!(snapshot_topology_2d(&dt), before_failed_attempt);
}

#[test]
fn pachner_feasibility_agrees_with_toroidal_2d_k1_insert() {
    let dt = build_canonicalized_toroidal_dt_2d();
    let simplex_key = first_simplex_generic(&dt);
    let vertex: Vertex<(), 2> = vertex!(simplex_centroid_generic(&dt, simplex_key))
        .expect("toroidal simplex centroid should be a finite vertex");
    let vertex_uuid = vertex.uuid();
    let pachner_move = PachnerMove::K1Insert {
        simplex_key,
        vertex,
    };

    let proposal = dt
        .propose_pachner(pachner_move)
        .expect("toroidal k=1 proposal parsing should accept a simplex centroid");
    let feasibility = proposal
        .can_attempt_on(&dt)
        .expect("toroidal k=1 feasibility should accept a simplex centroid")
        .clone();
    assert_pachner_feasibility_contract(
        &feasibility,
        BistellarFlipKind::k1(2),
        FlipDirection::Forward,
    );
    assert!(feasibility.inserted_face_vertices.is_none());

    let mut trial = dt;
    let result = proposal
        .attempt_on(&mut trial)
        .expect("toroidal k=1 attempt should agree with feasibility");
    let inserted_vertex = vertex_key_by_uuid(&trial, vertex_uuid)
        .expect("successful k=1 attempt should allocate the inserted vertex key");
    assert_eq!(result.kind, feasibility.kind);
    assert_eq!(result.direction, feasibility.direction);
    assert_eq!(result.removed_simplices, feasibility.removed_simplices);
    assert_eq!(
        result.removed_face_vertices,
        feasibility.removed_face_vertices
    );
    assert_eq!(result.inserted_face_vertices.as_slice(), &[inserted_vertex]);
    trial
        .as_triangulation()
        .validate()
        .expect("toroidal k=1 attempt should preserve topology validity");
}

#[test]
fn pachner_feasibility_rejects_duplicate_k1_insert_uuid_without_mutating() {
    let dt = build_single_triangle_dt_2d();
    let simplex_key = first_simplex_generic(&dt);
    let duplicate_vertex = dt
        .vertices()
        .next()
        .map(|(_, vertex)| *vertex)
        .expect("single-triangle fixture should contain a live vertex");
    let duplicate_uuid = duplicate_vertex.uuid();
    let pachner_move = PachnerMove::K1Insert {
        simplex_key,
        vertex: duplicate_vertex,
    };

    assert_duplicate_vertex_uuid_error(dt.propose_pachner(pachner_move), duplicate_uuid);

    let trial = dt;
    let before_failed_attempt = snapshot_topology_2d(&trial);
    assert_duplicate_vertex_uuid_error(trial.propose_pachner(pachner_move), duplicate_uuid);
    assert_eq!(snapshot_topology_2d(&trial), before_failed_attempt);
}

#[test]
fn pachner_feasibility_rejects_invalid_3d_inverse_k2_without_mutating() {
    let dt = build_minimal_simplex_dt::<3>();
    let simplex = dt
        .simplices()
        .next()
        .map(|(_, simplex)| simplex)
        .expect("minimal 3D fixture should contain one simplex");
    let [a, b, ..] = simplex.vertices() else {
        panic!("3D simplex should contain at least two vertices");
    };
    let edge = find_live_edge(&dt, *a, *b).expect("simplex vertices should form an edge");
    let pachner_move = PachnerMove::K2Inverse { edge };

    assert_pachner_rejection_preserves_topology(dt, pachner_move, |err| {
        assert_matches!(
            err,
            FlipError::InvalidEdgeMultiplicity {
                found: 1,
                expected: 3
            }
        );
    });
}

#[test]
fn pachner_feasibility_rejects_invalid_3d_k3_without_mutating() {
    let dt = build_minimal_simplex_dt::<3>();
    let ridge = dt
        .ridge_handles()
        .next()
        .expect("minimal 3D fixture should expose a ridge handle")
        .expect("minimal 3D fixture should expose a ridge handle");
    let pachner_move = PachnerMove::K3 { ridge };

    assert_pachner_rejection_preserves_topology(dt, pachner_move, |err| {
        assert_matches!(err, FlipError::InvalidRidgeMultiplicity { found: 1 });
    });
}

#[test]
fn pachner_feasibility_rejects_invalid_4d_k3_inverse_without_mutating() {
    let dt = build_minimal_simplex_dt::<4>();
    let simplex = dt
        .simplices()
        .next()
        .map(|(_, simplex)| simplex)
        .expect("minimal 4D fixture should contain one simplex");
    let [a, b, c, ..] = simplex.vertices() else {
        panic!("4D simplex should contain at least three vertices");
    };
    let triangle =
        TriangleHandle::try_new(*a, *b, *c).expect("simplex vertices should form a triangle");
    let pachner_move = PachnerMove::K3Inverse { triangle };

    assert_pachner_rejection_preserves_topology(dt, pachner_move, |err| {
        assert_matches!(
            err,
            FlipError::InvalidTriangleMultiplicity {
                found: 1,
                expected: 3
            }
        );
    });
}

#[test]
fn pachner_feasibility_rejects_unsupported_3d_k3_inverse_without_mutating() {
    let dt = build_minimal_simplex_dt::<3>();
    let simplex = dt
        .simplices()
        .next()
        .map(|(_, simplex)| simplex)
        .expect("minimal 3D fixture should contain one simplex");
    let [a, b, c, ..] = simplex.vertices() else {
        panic!("3D simplex should contain at least three vertices");
    };
    let triangle =
        TriangleHandle::try_new(*a, *b, *c).expect("simplex vertices should form a triangle");
    let pachner_move = PachnerMove::K3Inverse { triangle };

    assert_pachner_rejection_preserves_topology(dt, pachner_move, |err| {
        assert_matches!(err, FlipError::UnsupportedDimension { dimension: 3 });
    });
}

#[test]
fn pachner_feasibility_public_k1_insert_smoke_2d_to_5d() {
    assert_public_k1_insert_feasibility_smoke::<2>();
    assert_public_k1_insert_feasibility_smoke::<3>();
    assert_public_k1_insert_feasibility_smoke::<4>();
    assert_public_k1_insert_feasibility_smoke::<5>();
}

/// Attempts a stale k=1 insert through the public `DelaunayResult` alias.
fn try_stale_k1_insert(
    dt: &mut Dt4,
    stale_simplex: SimplexKey,
    vertex_coords: [f64; 4],
) -> DelaunayResult<()> {
    let vertex: Vertex<(), 4> = vertex!(vertex_coords)?;
    let vertex_uuid = vertex.uuid();
    let inserted = attempt_pachner_move(
        dt,
        PachnerMove::K1Insert {
            simplex_key: stale_simplex,
            vertex,
        },
    )?;
    let inserted_vertex = vertex_key_by_uuid(dt, vertex_uuid)
        .expect("unexpected successful stale insert should create the requested vertex");
    assert_k1_insert_result(&inserted, inserted_vertex);
    Ok(())
}

/// Builds the deterministic 4D fixture used to find reversible public Pachner moves.
#[cfg(feature = "slow-tests")]
fn build_stable_dt_4d() -> Dt4 {
    build_dt_4d(STABLE_POINTS_4D, "stable")
}

/// Builds the smallest 4D fixture needed by stale-handle atomicity checks.
fn build_minimal_dt_4d() -> Dt4 {
    build_dt_4d(MINIMAL_POINTS_4D, "minimal")
}

/// Builds a deterministic 4D fixture with input-order construction.
fn build_dt_4d(points: &[[f64; 4]], fixture_name: &str) -> Dt4 {
    let vertices = points
        .iter()
        .map(|coords| vertex!(*coords).unwrap())
        .collect::<Vec<_>>();
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);

    DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .construction_options(options)
        .build_with_kernel(&RobustKernel::new())
        .unwrap_or_else(|err| panic!("{fixture_name} 4D fixture should build: {err}"))
}

/// Builds a minimal Euclidean D-simplex fixture for dimension smoke tests.
fn build_minimal_simplex_dt<const D: usize>() -> Dt<D> {
    let vertices = minimal_simplex_vertices::<D>();
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);

    DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .construction_options(options)
        .build_with_kernel(&RobustKernel::new())
        .unwrap_or_else(|err| panic!("{D}D minimal simplex fixture should build: {err}"))
}

/// Returns the origin plus coordinate unit vectors as a nondegenerate D-simplex.
fn minimal_simplex_vertices<const D: usize>() -> Vec<Vertex<(), D>> {
    let mut vertices = Vec::with_capacity(D + 1);
    vertices.push(vertex!([0.0; D]).expect("origin vertex should be finite"));
    for axis in 0..D {
        let mut coords = [0.0; D];
        coords[axis] = 1.0;
        vertices.push(vertex!(coords).expect("unit simplex vertex should be finite"));
    }
    vertices
}

/// Builds a deterministic 2D fixture with at least one public k=2 move.
fn build_flippable_dt_2d() -> Dt2 {
    let vertices = FLIPPABLE_POINTS_2D
        .iter()
        .map(|coords| vertex!(*coords).expect("stable 2D fixture coordinates"))
        .collect::<Vec<_>>();
    let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];

    let dt = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .expect("explicit 2D fixture connectivity should parse")
        .build_with_kernel(&RobustKernel::new())
        .expect("stable 2D fixture should build");
    assert_topology_and_delaunay_valid(&dt, "stable 2D fixture before local edits");
    dt
}

/// Builds a single-triangle 2D fixture for boundary-facet rejection checks.
fn build_single_triangle_dt_2d() -> Dt2 {
    let vertices = vec![
        vertex!([0.0, 0.0]).expect("single-triangle fixture coordinate"),
        vertex!([1.0, 0.0]).expect("single-triangle fixture coordinate"),
        vertex!([0.0, 1.0]).expect("single-triangle fixture coordinate"),
    ];

    DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .build_with_kernel(&RobustKernel::new())
        .expect("single-triangle fixture should build")
}

/// Builds a canonicalized 2D toroidal fixture with live periodic topology metadata.
fn build_canonicalized_toroidal_dt_2d() -> Dt2 {
    let vertices = vec![
        vertex!([0.2, 0.3]).expect("toroidal fixture coordinate"),
        vertex!([1.8, 0.1]).expect("toroidal fixture coordinate"),
        vertex!([0.5, 0.7]).expect("toroidal fixture coordinate"),
        vertex!([-0.4, 0.9]).expect("toroidal fixture coordinate"),
    ];

    DelaunayTriangulationBuilder::new(&vertices)
        .try_canonicalized_toroidal([1.0, 1.0])
        .expect("canonicalized toroidal domain should parse")
        .build_with_kernel(&RobustKernel::new())
        .expect("canonicalized toroidal fixture should build")
}

/// Searches the 2D fixture for an edge facet whose public k=2 move succeeds.
fn flippable_k2_facet_2d(dt: &Dt2) -> FacetHandle {
    for facet in dt.facets() {
        let facet = facet.expect("2D fixture facets should reborrow as live views");
        let facet = facet.handle();
        let mut trial = dt.clone();
        if attempt_pachner_move(&mut trial, PachnerMove::K2 { facet }).is_ok()
            && topology_and_delaunay_valid(&trial)
        {
            return facet;
        }
    }
    panic!("stable 2D fixture should contain a public k=2 candidate");
}

/// Captures 2D topology by stable UUIDs so failed attempts can prove non-mutation.
fn snapshot_topology_2d(dt: &Dt2) -> TopologySnapshot {
    snapshot_topology_generic(dt)
}

/// Verifies the primitive flip feasibility and unified Pachner report agree.
fn assert_flip_and_pachner_feasibility_match<const D: usize>(
    primitive: &FlipFeasibility<D>,
    pachner: &PachnerMoveFeasibility<D>,
) {
    assert_eq!(primitive.kind, pachner.kind);
    assert_eq!(primitive.direction, pachner.direction);
    assert_eq!(primitive.removed_simplices, pachner.removed_simplices);
    assert_eq!(
        primitive.removed_face_vertices,
        pachner.removed_face_vertices
    );
    assert_eq!(
        primitive.inserted_face_vertices,
        pachner.inserted_face_vertices
    );
}

/// Captures topology by stable UUIDs for any deterministic test fixture.
fn snapshot_topology_generic<const D: usize>(dt: &Dt<D>) -> TopologySnapshot {
    let mut vertex_uuids = dt
        .vertices()
        .map(|(_, vertex)| vertex.uuid())
        .collect::<Vec<_>>();
    vertex_uuids.sort();

    let mut simplex_vertex_uuids = dt
        .simplices()
        .map(|(_, simplex)| {
            let mut uuids = simplex
                .vertices()
                .iter()
                .map(|vertex_key| {
                    dt.vertex(*vertex_key)
                        .expect("simplex should reference live vertices")
                        .uuid()
                })
                .collect::<Vec<_>>();
            uuids.sort();
            uuids
        })
        .collect::<Vec<_>>();
    simplex_vertex_uuids.sort();

    TopologySnapshot {
        vertex_uuids,
        simplex_vertex_uuids,
    }
}

/// Verifies an invalid raw Pachner request reports the same error without mutation.
fn assert_pachner_rejection_preserves_topology<const D: usize>(
    dt: Dt<D>,
    pachner_move: PachnerMove<(), D>,
    assert_error: impl Fn(&FlipError),
) {
    let before = snapshot_topology_generic(&dt);
    let feasibility = dt
        .propose_pachner(pachner_move)
        .expect_err("Pachner proposal parsing should reject invalid request");
    assert_error(&feasibility);
    assert_eq!(snapshot_topology_generic(&dt), before);

    let trial = dt;
    let attempted = trial
        .propose_pachner(pachner_move)
        .expect_err("Pachner proposal parsing should reject invalid request");
    assert_error(&attempted);
    assert_eq!(snapshot_topology_generic(&trial), before);
}

/// Verifies the generic Pachner feasibility arities implied by the reported move kind.
fn assert_pachner_feasibility_contract<const D: usize>(
    feasibility: &PachnerMoveFeasibility<D>,
    kind: BistellarFlipKind,
    direction: FlipDirection,
) {
    assert_eq!(feasibility.kind, kind);
    assert_eq!(feasibility.direction, direction);
    assert_eq!(feasibility.removed_simplices.len(), kind.k());
    assert_eq!(feasibility.removed_face_vertices.len(), D + 2 - kind.k());
    if let Some(inserted_face_vertices) = &feasibility.inserted_face_vertices {
        assert_eq!(inserted_face_vertices.len(), kind.k());
    }
}

/// Verifies duplicate inserted-vertex UUIDs surface through the typed flip error.
fn assert_duplicate_vertex_uuid_error<T>(result: Result<T, FlipError>, duplicate_uuid: Uuid) {
    match result {
        Err(FlipError::TdsMutation { reason })
            if matches!(
                reason.as_ref(),
                FlipMutationError::VertexInsertion {
                    source: TdsConstructionFailure::DuplicateUuid { uuid, .. },
                } if *uuid == duplicate_uuid
            ) => {}
        Err(err) => {
            panic!("expected duplicate vertex UUID error for {duplicate_uuid}, got {err:?}")
        }
        Ok(_) => panic!("expected duplicate vertex UUID error for {duplicate_uuid}"),
    }
}

/// Exercises public `PachnerMove::K1Insert` feasibility and mutation in one dimension.
fn assert_public_k1_insert_feasibility_smoke<const D: usize>() {
    let dt = build_minimal_simplex_dt::<D>();
    let simplex_key = first_simplex_generic(&dt);
    let vertex: Vertex<(), D> = vertex!(simplex_centroid_generic(&dt, simplex_key))
        .unwrap_or_else(|err| panic!("{D}D simplex centroid should be finite: {err}"));
    let vertex_uuid = vertex.uuid();
    let pachner_move = PachnerMove::K1Insert {
        simplex_key,
        vertex,
    };

    let proposal = dt
        .propose_pachner(pachner_move)
        .unwrap_or_else(|err| panic!("{D}D public k=1 proposal should succeed: {err:?}"));
    let feasibility = proposal
        .can_attempt_on(&dt)
        .unwrap_or_else(|err| panic!("{D}D public k=1 feasibility should succeed: {err:?}"))
        .clone();
    assert_pachner_feasibility_contract(
        &feasibility,
        BistellarFlipKind::k1(D),
        FlipDirection::Forward,
    );
    assert!(feasibility.inserted_face_vertices.is_none());

    let mut trial = dt;
    let result = proposal
        .attempt_on(&mut trial)
        .unwrap_or_else(|err| panic!("{D}D public k=1 mutation should succeed: {err:?}"));
    let inserted_vertex = vertex_key_by_uuid(&trial, vertex_uuid)
        .expect("successful k=1 mutation should allocate the requested vertex");
    assert_eq!(feasibility.kind, result.kind);
    assert_eq!(feasibility.direction, result.direction);
    assert_eq!(feasibility.removed_simplices, result.removed_simplices);
    assert_eq!(
        feasibility.removed_face_vertices,
        result.removed_face_vertices
    );
    assert_eq!(result.inserted_face_vertices.as_slice(), &[inserted_vertex]);

    let remove_move = PachnerMove::K1Remove {
        vertex_key: inserted_vertex,
    };
    let primitive_remove_feasibility = trial
        .can_flip_k1_remove(inserted_vertex)
        .unwrap_or_else(|err| panic!("{D}D public k=1 remove feasibility should succeed: {err:?}"));
    let remove_proposal = trial.propose_pachner(remove_move).unwrap_or_else(|err| {
        panic!("{D}D public Pachner k=1 remove proposal should succeed: {err:?}")
    });
    let remove_feasibility = remove_proposal
        .can_attempt_on(&trial)
        .unwrap_or_else(|err| panic!("{D}D public Pachner k=1 remove should succeed: {err:?}"));
    assert_flip_and_pachner_feasibility_match(&primitive_remove_feasibility, remove_feasibility);
    assert_pachner_feasibility_contract(
        remove_feasibility,
        BistellarFlipKind::k1(D).inverse(),
        FlipDirection::Inverse,
    );

    let removed = remove_proposal
        .attempt_on(&mut trial)
        .unwrap_or_else(|err| panic!("{D}D public k=1 remove mutation should succeed: {err:?}"));
    assert_pachner_result_contract(
        &removed,
        BistellarFlipKind::k1(D).inverse(),
        FlipDirection::Inverse,
    );
    assert_topology_and_delaunay_valid(&trial, &format!("{D}D k=1 mutation"));
}

/// Returns any live simplex key from a generic D-dimensional fixture.
fn first_simplex_generic<const D: usize>(dt: &Dt<D>) -> SimplexKey {
    dt.simplices()
        .next()
        .map(|(simplex_key, _)| simplex_key)
        .expect("fixture should contain simplices")
}

/// Computes a simplex centroid for generic dimension smoke tests.
fn simplex_centroid_generic<const D: usize>(dt: &Dt<D>, simplex_key: SimplexKey) -> [f64; D] {
    *dt.simplex_barycenter(simplex_key)
        .expect("simplex key should have a finite barycenter")
        .coords()
}

/// Converts a 2D facet handle into the edge key represented by that facet.
fn edge_for_facet_2d(dt: &Dt2, facet: FacetHandle) -> EdgeKey {
    let view = dt
        .facets()
        .find_map(|candidate| {
            let candidate = candidate.expect("2D fixture facets should reborrow as live views");
            (candidate.handle() == facet).then_some(candidate)
        })
        .expect("facet handle should still be live");
    let vertices = view.simplex().vertices();
    let endpoints = match usize::from(view.facet_index()) {
        0 => [vertices[1], vertices[2]],
        1 => [vertices[0], vertices[2]],
        2 => [vertices[0], vertices[1]],
        index => {
            panic!("invalid 2D facet index {index}");
        }
    };
    let [a, b] = endpoints;
    find_live_edge(dt, a, b).expect("facet endpoints should form a live edge")
}

/// Parses the inserted edge reported by a 2D k=2 move.
fn inserted_edge_2d(dt: &Dt2, vertices: &[VertexKey]) -> EdgeKey {
    let [a, b] = vertices else {
        panic!(
            "2D k=2 move should report exactly two inserted edge vertices, got {}",
            vertices.len()
        );
    };
    find_live_edge(dt, *a, *b).expect("reported inserted vertices should form a live edge")
}

/// Checks that a rejected detached proposal leaves the live topology byte-for-byte equivalent.
fn assert_failed_attempt_preserves_topology(
    dt: &mut Dt4,
    proposal: PachnerProposal<(), 4>,
    assert_error: impl Fn(&FlipError),
) {
    let before = snapshot_topology(dt);
    let proposal_generation = proposal.topology_generation();
    let current_generation = dt.topology_generation();

    let feasibility_err = proposal
        .can_attempt_on(dt)
        .expect_err("stale Pachner proposal feasibility should fail");
    assert_error(&feasibility_err);
    assert_stale_proposal_generation(&feasibility_err, proposal_generation, current_generation);
    assert_eq!(snapshot_topology(dt), before);

    let err = proposal
        .attempt_on(dt)
        .expect_err("stale Pachner proposal should fail");
    assert_error(&err);
    assert_stale_proposal_generation(&err, proposal_generation, current_generation);
    dt.is_valid_structure()
        .expect("failed Pachner attempt should preserve TDS validity");
    assert_eq!(snapshot_topology(dt), before);
}

/// Verifies stale proposal diagnostics report both the parsed and current generations.
fn assert_stale_proposal_generation(
    err: &FlipError,
    proposal_generation: u64,
    current_generation: u64,
) {
    assert_matches!(
        err,
        FlipError::StaleTopologyProposal {
            proposal_generation: reported_proposal_generation,
            current_generation: reported_current_generation,
        } if *reported_proposal_generation == proposal_generation
            && *reported_current_generation == current_generation
    );
    assert_eq!(
        FlipFailureKind::from(err),
        FlipFailureKind::StaleTopologyProposal
    );
}

/// Verifies the extra k=1 insert contract: the inserted face is exactly the new vertex.
fn assert_k1_insert_result(result: &PachnerMoveResult<4>, inserted_vertex: VertexKey) {
    assert_pachner_result_contract(result, BistellarFlipKind::k1(4), FlipDirection::Forward);
    assert_eq!(result.inserted_face_vertices.as_slice(), &[inserted_vertex]);
}

/// Verifies the generic Pachner result arities implied by the reported move kind.
fn assert_pachner_result_contract<const D: usize>(
    result: &PachnerMoveResult<D>,
    kind: BistellarFlipKind,
    direction: FlipDirection,
) {
    assert_eq!(result.kind, kind);
    assert_eq!(result.direction, direction);
    assert_eq!(result.removed_simplices.len(), kind.k());
    assert_eq!(result.new_simplices.len(), D + 2 - kind.k());
    assert_eq!(result.removed_face_vertices.len(), D + 2 - kind.k());
    assert_eq!(result.inserted_face_vertices.len(), kind.k());
}

/// Makes a k=1 insert proposal stale, then proves retrying it is failure-atomic.
fn assert_stale_k1_insert_preserves_topology(mut dt: Dt4) {
    let stale_simplex = first_simplex(&dt);
    let vertex_coords = simplex_centroid(&dt, stale_simplex);
    let vertex: Vertex<(), 4> = vertex!(vertex_coords).unwrap();
    let vertex_uuid = vertex.uuid();
    let stale_proposal = dt
        .propose_pachner(PachnerMove::K1Insert {
            simplex_key: stale_simplex,
            vertex,
        })
        .expect("initial k=1 insert proposal should be valid");
    let inserted = stale_proposal
        .clone()
        .attempt_on(&mut dt)
        .expect("initial k=1 insert should make the proposal stale");
    let inserted_vertex = vertex_key_by_uuid(&dt, vertex_uuid)
        .expect("initial k=1 insert should create the requested vertex");
    assert_k1_insert_result(&inserted, inserted_vertex);
    assert_failed_attempt_preserves_topology(&mut dt, stale_proposal, |err| {
        assert_matches!(
            err,
            FlipError::StaleTopologyProposal { .. },
            "unexpected stale k=1 insert error: {err:?}"
        );
    });
}

/// Makes a k=1 remove proposal stale, then proves retrying it is failure-atomic.
fn assert_stale_k1_remove_preserves_topology(mut dt: Dt4) {
    let simplex_key = first_simplex(&dt);
    let vertex: Vertex<(), 4> = vertex!(simplex_centroid(&dt, simplex_key)).unwrap();
    let vertex_uuid = vertex.uuid();
    let insert_proposal = dt
        .propose_pachner(PachnerMove::K1Insert {
            simplex_key,
            vertex,
        })
        .expect("k=1 insert proposal should be valid");
    let inserted = insert_proposal
        .attempt_on(&mut dt)
        .expect("k=1 insert should create a removable vertex");
    let vertex_key =
        vertex_key_by_uuid(&dt, vertex_uuid).expect("inserted k=1 vertex should be present");
    assert_k1_insert_result(&inserted, vertex_key);
    let stale_proposal = dt
        .propose_pachner(PachnerMove::K1Remove { vertex_key })
        .expect("k=1 remove proposal should be valid");
    let removed = stale_proposal
        .clone()
        .attempt_on(&mut dt)
        .expect("k=1 remove should make the proposal stale");
    assert!(!removed.removed_simplices.is_empty());

    assert_failed_attempt_preserves_topology(&mut dt, stale_proposal, |err| {
        assert_matches!(
            err,
            FlipError::StaleTopologyProposal { .. },
            "unexpected stale k=1 remove error: {err:?}"
        );
    });
}

/// Makes a k=2 facet proposal stale, then proves retrying it is failure-atomic.
#[cfg(feature = "slow-tests")]
fn assert_stale_k2_preserves_topology(mut dt: Dt4) {
    let facet = flippable_k2_facet(&dt);
    let stale_proposal = dt
        .propose_pachner(PachnerMove::K2 { facet })
        .expect("k=2 proposal should be valid");
    let flipped = stale_proposal
        .clone()
        .attempt_on(&mut dt)
        .expect("k=2 flip should make its proposal stale");
    assert_eq!(flipped.inserted_face_vertices.len(), 2);
    assert!(!flipped.new_simplices.is_empty());

    assert_failed_attempt_preserves_topology(&mut dt, stale_proposal, |err| {
        assert_matches!(
            err,
            FlipError::StaleTopologyProposal { .. },
            "unexpected stale k=2 error: {err:?}"
        );
    });
}

/// Makes an inverse k=2 edge proposal stale, then proves retrying it is failure-atomic.
#[cfg(feature = "slow-tests")]
fn assert_stale_k2_inverse_preserves_topology(mut dt: Dt4) {
    let facet = flippable_k2_facet(&dt);
    let info = attempt_pachner_move(&mut dt, PachnerMove::K2 { facet })
        .expect("k=2 flip should create an inverse edge");
    let edge = inserted_edge(&dt, &info.inserted_face_vertices);
    let stale_proposal = dt
        .propose_pachner(PachnerMove::K2Inverse { edge })
        .expect("k=2 inverse proposal should be valid");
    let inverse = stale_proposal
        .clone()
        .attempt_on(&mut dt)
        .expect("k=2 inverse should make its proposal stale");
    assert!(!inverse.removed_simplices.is_empty());

    assert_failed_attempt_preserves_topology(&mut dt, stale_proposal, |err| {
        assert_matches!(
            err,
            FlipError::StaleTopologyProposal { .. },
            "unexpected stale inverse k=2 error: {err:?}"
        );
    });
}

/// Makes a k=3 ridge proposal stale, then proves retrying it is failure-atomic.
#[cfg(feature = "slow-tests")]
fn assert_stale_k3_preserves_topology(mut dt: Dt4) {
    let ridge = flippable_k3_ridge(&dt);
    let stale_proposal = dt
        .propose_pachner(PachnerMove::K3 { ridge })
        .expect("k=3 proposal should be valid");
    let flipped = stale_proposal
        .clone()
        .attempt_on(&mut dt)
        .expect("k=3 flip should make its proposal stale");
    assert_eq!(flipped.inserted_face_vertices.len(), 3);
    assert!(!flipped.new_simplices.is_empty());

    assert_failed_attempt_preserves_topology(&mut dt, stale_proposal, |err| {
        assert_matches!(
            err,
            FlipError::StaleTopologyProposal { .. },
            "unexpected stale k=3 error: {err:?}"
        );
    });
}

/// Makes an inverse k=3 triangle proposal stale, then proves retrying it is failure-atomic.
#[cfg(feature = "slow-tests")]
fn assert_stale_k3_inverse_preserves_topology(mut dt: Dt4) {
    let ridge = flippable_k3_ridge(&dt);
    let info = attempt_pachner_move(&mut dt, PachnerMove::K3 { ridge })
        .expect("k=3 flip should create an inverse triangle");
    let triangle = inserted_triangle(&info.inserted_face_vertices);
    let stale_proposal = dt
        .propose_pachner(PachnerMove::K3Inverse { triangle })
        .expect("k=3 inverse proposal should be valid");
    let inverse = stale_proposal
        .clone()
        .attempt_on(&mut dt)
        .expect("k=3 inverse should make its proposal stale");
    assert!(!inverse.removed_simplices.is_empty());

    assert_failed_attempt_preserves_topology(&mut dt, stale_proposal, |err| {
        assert_matches!(
            err,
            FlipError::StaleTopologyProposal { .. },
            "unexpected stale inverse k=3 error: {err:?}"
        );
    });
}

/// Captures topology by stable UUIDs so slotmap key reuse cannot hide mutations.
fn snapshot_topology(dt: &Dt4) -> TopologySnapshot {
    snapshot_topology_generic(dt)
}

/// Returns a simplex from the stable fixture for tests that only need any live simplex.
fn first_simplex(dt: &Dt4) -> SimplexKey {
    dt.simplices()
        .next()
        .map(|(simplex_key, _)| simplex_key)
        .expect("stable fixture should contain simplices")
}

/// Computes an interior-ish point for k=1 insertion into a known simplex.
fn simplex_centroid(dt: &Dt4, simplex_key: SimplexKey) -> [f64; 4] {
    *dt.simplex_barycenter(simplex_key)
        .expect("simplex key should have a finite barycenter")
        .coords()
}

/// Applies a k=1 insert/remove pair and checks the reported move metadata.
#[cfg(feature = "slow-tests")]
fn roundtrip_k1(dt: &mut Dt4) {
    let simplex_key = first_simplex(dt);
    let new_vertex: Vertex<(), 4> = vertex!(simplex_centroid(dt, simplex_key)).unwrap();
    let new_uuid = new_vertex.uuid();
    let inserted = attempt_pachner_move(
        dt,
        PachnerMove::K1Insert {
            simplex_key,
            vertex: new_vertex,
        },
    )
    .expect("k=1 insert should succeed on stable 4D fixture");
    assert_eq!(inserted.inserted_face_vertices.len(), 1);

    let inserted_key =
        vertex_key_by_uuid(dt, new_uuid).expect("inserted k=1 vertex should be present");
    let removed = attempt_pachner_move(
        dt,
        PachnerMove::K1Remove {
            vertex_key: inserted_key,
        },
    )
    .expect("k=1 remove should invert insert");
    assert_pachner_result_contract(
        &removed,
        BistellarFlipKind::k1(4).inverse(),
        FlipDirection::Inverse,
    );
}

/// Searches the fixture for a k=2 facet that also supports the public inverse API.
#[cfg(feature = "slow-tests")]
fn flippable_k2_facet(dt: &Dt4) -> FacetHandle {
    for facet in dt.facets() {
        let facet = facet.expect("4D fixture facets should reborrow as live views");
        let facet = facet.handle();
        let mut trial = dt.clone();
        let Ok(info) = attempt_pachner_move(&mut trial, PachnerMove::K2 { facet }) else {
            continue;
        };
        let edge = inserted_edge(&trial, &info.inserted_face_vertices);
        if attempt_pachner_move(&mut trial, PachnerMove::K2Inverse { edge }).is_ok()
            && topology_and_delaunay_valid(&trial)
        {
            return facet;
        }
    }
    panic!("stable 4D fixture should contain a public k=2 roundtrip candidate");
}

/// Applies a k=2 forward/inverse pair and checks both move reports.
#[cfg(feature = "slow-tests")]
fn roundtrip_k2(dt: &mut Dt4, facet: FacetHandle) {
    let info: PachnerMoveResult<4> = attempt_pachner_move(dt, PachnerMove::K2 { facet })
        .expect("k=2 flip should succeed on selected stable 4D facet");
    assert_pachner_result_contract(&info, BistellarFlipKind::k2(4), FlipDirection::Forward);
    let edge = inserted_edge(dt, &info.inserted_face_vertices);
    let inverse = attempt_pachner_move(dt, PachnerMove::K2Inverse { edge })
        .expect("k=2 inverse should succeed after k=2 flip");
    assert_pachner_result_contract(
        &inverse,
        BistellarFlipKind::k2(4).inverse(),
        FlipDirection::Inverse,
    );
}

/// Parses the inserted face of a k=2 move into the edge expected by the inverse API.
#[cfg(feature = "slow-tests")]
fn inserted_edge(dt: &Dt4, vertices: &[VertexKey]) -> EdgeKey {
    let [a, b] = vertices else {
        panic!(
            "k=2 flip should report an inserted edge, got {} vertices",
            vertices.len()
        );
    };
    find_live_edge(dt, *a, *b).expect("k=2 flip should report a real inserted edge")
}

/// Searches the fixture for a k=3 ridge that also supports the public inverse API.
#[cfg(feature = "slow-tests")]
fn flippable_k3_ridge(dt: &Dt4) -> RidgeHandle {
    for ridge in dt.ridge_handles() {
        let ridge = ridge.expect("4D fixture ridges should produce live handles");
        let mut trial = dt.clone();
        let Ok(info) = attempt_pachner_move(&mut trial, PachnerMove::K3 { ridge }) else {
            continue;
        };
        let triangle = inserted_triangle(&info.inserted_face_vertices);
        if attempt_pachner_move(&mut trial, PachnerMove::K3Inverse { triangle }).is_ok()
            && topology_and_delaunay_valid(&trial)
        {
            return ridge;
        }
    }
    panic!("stable 4D fixture should contain a public k=3 roundtrip candidate");
}

/// Applies a k=3 forward/inverse pair and checks both move reports.
#[cfg(feature = "slow-tests")]
fn roundtrip_k3(dt: &mut Dt4, ridge: RidgeHandle) {
    let info: PachnerMoveResult<4> = attempt_pachner_move(dt, PachnerMove::K3 { ridge })
        .expect("k=3 flip should succeed on selected stable 4D ridge");
    assert_pachner_result_contract(&info, BistellarFlipKind::k3(4), FlipDirection::Forward);
    let inverse = attempt_pachner_move(
        dt,
        PachnerMove::K3Inverse {
            triangle: inserted_triangle(&info.inserted_face_vertices),
        },
    )
    .expect("k=3 inverse should succeed after k=3 flip");
    assert_pachner_result_contract(
        &inverse,
        BistellarFlipKind::k3(4).inverse(),
        FlipDirection::Inverse,
    );
}

/// Parses the inserted face of a k=3 move into the triangle expected by the inverse API.
#[cfg(feature = "slow-tests")]
fn inserted_triangle(vertices: &[VertexKey]) -> TriangleHandle {
    let [a, b, c] = vertices else {
        panic!(
            "k=3 flip should report an inserted triangle, got {} vertices",
            vertices.len()
        );
    };
    TriangleHandle::try_new(*a, *b, *c).expect("k=3 flip should report a valid inserted triangle")
}
