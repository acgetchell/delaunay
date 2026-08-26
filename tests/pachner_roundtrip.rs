#![forbid(unsafe_code)]

//! Public API roundtrip tests for Pachner/bistellar flips.

use delaunay::flips::{BistellarFlips, FlipFailureKind, FlipFeasibility, FlipMutationError};
use delaunay::{DelaunayRefinementBuilder, TdsConstructionFailure, vertex};
use std::assert_matches;

use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayError, DelaunayResult, DelaunayTriangulationBuilder,
    InsertionOrderStrategy, TopologyGuarantee, Vertex,
};
use delaunay::prelude::geometry::RobustKernel;
#[cfg(feature = "slow-tests")]
use delaunay::prelude::pachner::RidgeHandle;
use delaunay::prelude::pachner::{
    BistellarFlipKind, EdgeKey, EdgeKeyError, FacetHandle, FlipDirection, FlipError, PachnerMove,
    PachnerMoveFeasibility, PachnerMoveResult, PachnerMoves, PachnerProposal, SimplexKey,
    TopologyOwner, TriangleHandle, VertexKey,
};
use delaunay::prelude::triangulation::Triangulation;
use uuid::Uuid;

type Tri4 = Triangulation<RobustKernel<f64>, (), (), 4>;
type Tri2 = Triangulation<RobustKernel<f64>, (), (), 2>;
type Tri<const D: usize> = Triangulation<RobustKernel<f64>, (), (), D>;

const FLIPPABLE_POINTS_2D: &[[f64; 2]] = &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

const MINIMAL_POINTS_4D: &[[f64; 4]] = &[
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn vertex_key_by_uuid<const D: usize>(tri: &Tri<D>, uuid: Uuid) -> Option<VertexKey> {
    tri.vertices()
        .find_map(|(vertex_key, vertex)| (vertex.uuid() == uuid).then_some(vertex_key))
}

fn find_live_edge<const D: usize>(
    tri: &Tri<D>,
    a: VertexKey,
    b: VertexKey,
) -> Result<EdgeKey, EdgeKeyError> {
    if a == b {
        return Err(EdgeKeyError::DuplicateEndpoint { endpoint: a });
    }
    if !tri.contains_vertex_key(a) {
        return Err(EdgeKeyError::MissingEndpoint { endpoint: a });
    }
    if !tri.contains_vertex_key(b) {
        return Err(EdgeKeyError::MissingEndpoint { endpoint: b });
    }
    tri.edges()
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

fn topology_and_delaunay_valid<const D: usize>(
    tri: &Triangulation<RobustKernel<f64>, (), (), D>,
) -> bool {
    tri.validate_realization().is_ok()
        && DelaunayRefinementBuilder::new(tri.clone()).build().is_ok()
}

fn assert_topology_and_delaunay_valid<const D: usize>(
    tri: &Triangulation<RobustKernel<f64>, (), (), D>,
    context: &str,
) {
    tri.validate()
        .unwrap_or_else(|err| panic!("{context} should pass Levels 1-3: {err}"));
    tri.is_valid_realization()
        .unwrap_or_else(|err| panic!("{context} should pass Level 4 realization: {err}"));
    DelaunayRefinementBuilder::new(tri.clone())
        .build()
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
    let tri = build_minimal_simplex_dt::<2>();
    let simplex_key = first_simplex_generic(&tri);
    let vertex: Vertex<(), 2> = vertex!(simplex_centroid_generic(&tri, simplex_key))
        .expect("simplex centroid should be valid");
    let proposal = tri
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
    let mut tri = build_minimal_dt_4d();
    let stale_simplex = first_simplex(&tri);
    let vertex_coords = simplex_centroid(&tri, stale_simplex);
    let vertex: Vertex<(), 4> =
        vertex!(vertex_coords).expect("centroid of a stable simplex should be a valid vertex");
    let vertex_uuid = vertex.uuid();
    let inserted = tri
        .propose_pachner(PachnerMove::K1Insert {
            simplex_key: stale_simplex,
            vertex,
        })
        .expect("initial k=1 proposal should be valid")
        .attempt_on(&mut tri)
        .expect("initial k=1 insert should make the simplex key stale");
    let inserted_vertex = vertex_key_by_uuid(&tri, vertex_uuid)
        .expect("initial k=1 insert should create the requested vertex");
    assert_k1_insert_result(&inserted, inserted_vertex);
    let before_failed_attempt = snapshot_topology(&tri);

    let err = try_stale_k1_insert(&mut tri, stale_simplex, vertex_coords)
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
    tri.is_valid_structure()
        .expect("failed Pachner attempt should preserve TDS validity");
    assert_eq!(snapshot_topology(&tri), before_failed_attempt);
}

#[test]
fn edge_to_facet_query_tracks_2d_k2_mutation_freshness() {
    let mut tri = build_flippable_dt_2d();
    let facet = flippable_k2_facet_2d(&tri);
    let old_edge = edge_for_facet_2d(&tri, facet);

    let old_incident_facets: Vec<_> = tri
        .try_incident_facets_to_edge_2d(old_edge)
        .unwrap()
        .collect();
    assert_eq!(old_incident_facets.len(), 2);
    assert!(old_incident_facets.contains(&facet));
    assert!(
        old_incident_facets
            .iter()
            .all(|&incident_facet| edge_for_facet_2d(&tri, incident_facet) == old_edge)
    );
    assert!(
        tri.try_interior_facet_for_edge_2d(old_edge)
            .unwrap()
            .is_some()
    );

    let info = tri
        .propose_pachner(PachnerMove::K2 { facet })
        .expect("selected fixture facet should produce a 2D k=2 proposal")
        .attempt_on(&mut tri)
        .expect("2D k=2 flip should succeed on selected fixture facet");
    assert_eq!(info.inserted_face_vertices.len(), 2);

    match tri.try_incident_facets_to_edge_2d(old_edge) {
        Err(EdgeKeyError::EdgeNotFound { .. }) => {}
        Err(err) => panic!("expected old edge to be absent after k=2 flip, got {err:?}"),
        Ok(_) => panic!("expected old edge to be absent after k=2 flip, got facets"),
    }
    assert_matches!(
        tri.try_interior_facet_for_edge_2d(old_edge),
        Err(EdgeKeyError::EdgeNotFound { .. })
    );

    let new_edge = inserted_edge_2d(&tri, &info.inserted_face_vertices);
    let new_incident_facets: Vec<_> = tri
        .try_incident_facets_to_edge_2d(new_edge)
        .unwrap()
        .collect();
    assert_eq!(new_incident_facets.len(), 2);
    assert!(
        new_incident_facets
            .iter()
            .all(|&incident_facet| edge_for_facet_2d(&tri, incident_facet) == new_edge)
    );
    assert!(
        tri.try_interior_facet_for_edge_2d(new_edge)
            .unwrap()
            .is_some()
    );
    assert_topology_and_delaunay_valid(&tri, "2D k=2 mutation-freshness fixture");
}

#[test]
fn pachner_feasibility_agrees_with_successful_2d_k2_attempt() {
    let tri = build_flippable_dt_2d();
    let facet = flippable_k2_facet_2d(&tri);
    let pachner_move = PachnerMove::K2 { facet };

    let proposal = tri
        .propose_pachner(pachner_move)
        .expect("2D k=2 proposal parsing should accept selected fixture facet");
    let feasibility = proposal
        .can_attempt_on(&tri)
        .expect("2D k=2 feasibility should accept selected fixture facet")
        .clone();
    assert_pachner_feasibility_contract(
        &feasibility,
        BistellarFlipKind::try_k2(2).expect("2D k=2 move kind should be valid"),
        FlipDirection::Forward,
    );

    let mut trial = tri;
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
    let mut tri = build_flippable_dt_2d();
    let facet = flippable_k2_facet_2d(&tri);
    let forward = tri
        .propose_pachner(PachnerMove::K2 { facet })
        .expect("selected fixture facet should produce a 2D k=2 proposal")
        .attempt_on(&mut tri)
        .expect("2D k=2 attempt should create an inverse edge candidate");
    let edge = inserted_edge_2d(&tri, &forward.inserted_face_vertices);
    let pachner_move = PachnerMove::K2Inverse { edge };
    let before = snapshot_topology_2d(&tri);

    assert_matches!(
        tri.can_flip_k2_inverse_from_edge(edge),
        Err(FlipError::UnsupportedDimension { dimension: 2 })
    );
    assert_eq!(snapshot_topology_2d(&tri), before);
    assert_pachner_rejection_preserves_topology(tri, pachner_move, |err| {
        assert_matches!(err, FlipError::UnsupportedDimension { dimension: 2 });
    });
}

#[test]
fn pachner_feasibility_rejects_boundary_facet_like_attempt_2d() {
    let tri = build_single_triangle_dt_2d();
    let facet = tri
        .boundary_facets()
        .expect("single-triangle fixture should classify boundary facets")
        .next()
        .expect("single-triangle fixture should expose a boundary facet")
        .expect("boundary facet should reborrow as a live view")
        .handle();
    let pachner_move = PachnerMove::K2 { facet };

    let feasibility = tri.propose_pachner(pachner_move);
    assert_matches!(feasibility, Err(FlipError::BoundaryFacet { .. }));

    let trial = tri;
    let attempted = trial.propose_pachner(pachner_move);
    assert_matches!(attempted, Err(FlipError::BoundaryFacet { .. }));
    assert_topology_and_delaunay_valid(&trial, "failed boundary feasibility agreement");
}

#[test]
fn pachner_feasibility_rejects_stale_facet_like_attempt_2d() {
    let mut tri = build_flippable_dt_2d();
    let facet = flippable_k2_facet_2d(&tri);
    let first_flip = tri
        .propose_pachner(PachnerMove::K2 { facet })
        .expect("fresh fixture facet should produce a k=2 proposal")
        .attempt_on(&mut tri)
        .expect("first k=2 attempt should stale the original facet");
    assert!(!first_flip.new_simplices.is_empty());
    let stale_move = PachnerMove::K2 { facet };

    let feasibility = tri.propose_pachner(stale_move);
    assert_matches!(
        feasibility,
        Err(FlipError::MissingSimplex { simplex_key }) if simplex_key == facet.simplex_key()
    );

    let before_failed_attempt = snapshot_topology_2d(&tri);
    let attempted = tri.propose_pachner(stale_move);
    assert_matches!(
        attempted,
        Err(FlipError::MissingSimplex { simplex_key }) if simplex_key == facet.simplex_key()
    );
    assert_eq!(snapshot_topology_2d(&tri), before_failed_attempt);
}

#[test]
fn pachner_feasibility_rejects_duplicate_k1_insert_uuid_without_mutating() {
    let tri = build_single_triangle_dt_2d();
    let simplex_key = first_simplex_generic(&tri);
    let duplicate_vertex = tri
        .vertices()
        .next()
        .map(|(_, vertex)| *vertex)
        .expect("single-triangle fixture should contain a live vertex");
    let duplicate_uuid = duplicate_vertex.uuid();
    let pachner_move = PachnerMove::K1Insert {
        simplex_key,
        vertex: duplicate_vertex,
    };

    assert_duplicate_vertex_uuid_error(tri.propose_pachner(pachner_move), duplicate_uuid);

    let trial = tri;
    let before_failed_attempt = snapshot_topology_2d(&trial);
    assert_duplicate_vertex_uuid_error(trial.propose_pachner(pachner_move), duplicate_uuid);
    assert_eq!(snapshot_topology_2d(&trial), before_failed_attempt);
}

#[test]
fn pachner_feasibility_rejects_invalid_3d_inverse_k2_without_mutating() {
    let tri = build_minimal_simplex_dt::<3>();
    let simplex = tri
        .simplices()
        .next()
        .map(|(_, simplex)| simplex)
        .expect("minimal 3D fixture should contain one simplex");
    let [a, b, ..] = simplex.vertices() else {
        panic!("3D simplex should contain at least two vertices");
    };
    let edge = find_live_edge(&tri, *a, *b).expect("simplex vertices should form an edge");
    let pachner_move = PachnerMove::K2Inverse { edge };

    assert_pachner_rejection_preserves_topology(tri, pachner_move, |err| {
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
    let tri = build_minimal_simplex_dt::<3>();
    let ridge = tri
        .ridge_handles()
        .next()
        .expect("minimal 3D fixture should expose a ridge handle")
        .expect("minimal 3D fixture should expose a ridge handle");
    let pachner_move = PachnerMove::K3 { ridge };

    assert_pachner_rejection_preserves_topology(tri, pachner_move, |err| {
        assert_matches!(err, FlipError::InvalidRidgeMultiplicity { found: 1 });
    });
}

#[test]
fn pachner_feasibility_rejects_invalid_4d_k3_inverse_without_mutating() {
    let tri = build_minimal_simplex_dt::<4>();
    let simplex = tri
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

    assert_pachner_rejection_preserves_topology(tri, pachner_move, |err| {
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
    let tri = build_minimal_simplex_dt::<3>();
    let simplex = tri
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

    assert_pachner_rejection_preserves_topology(tri, pachner_move, |err| {
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
    tri: &mut Tri4,
    stale_simplex: SimplexKey,
    vertex_coords: [f64; 4],
) -> DelaunayResult<()> {
    let vertex: Vertex<(), 4> = vertex!(vertex_coords)?;
    let vertex_uuid = vertex.uuid();
    let inserted = tri
        .propose_pachner(PachnerMove::K1Insert {
            simplex_key: stale_simplex,
            vertex,
        })?
        .attempt_on(tri)?;
    let inserted_vertex = vertex_key_by_uuid(tri, vertex_uuid)
        .expect("unexpected successful stale insert should create the requested vertex");
    assert_k1_insert_result(&inserted, inserted_vertex);
    Ok(())
}

/// Builds the deterministic 4D fixture used to find reversible public Pachner moves.
#[cfg(feature = "slow-tests")]
fn build_stable_dt_4d() -> Tri4 {
    build_dt_4d(STABLE_POINTS_4D, "stable")
}

/// Builds the smallest 4D fixture needed by stale-handle atomicity checks.
fn build_minimal_dt_4d() -> Tri4 {
    build_dt_4d(MINIMAL_POINTS_4D, "minimal")
}

/// Builds a deterministic 4D fixture with input-order construction.
fn build_dt_4d(points: &[[f64; 4]], fixture_name: &str) -> Tri4 {
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
        .into_triangulation()
}

/// Builds a minimal Euclidean D-simplex fixture for dimension smoke tests.
fn build_minimal_simplex_dt<const D: usize>() -> Tri<D>
where
    RobustKernel<f64>: delaunay::prelude::geometry::ExactPredicates<D>,
{
    let vertices = minimal_simplex_vertices::<D>();
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);

    DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .construction_options(options)
        .build_with_kernel(&RobustKernel::new())
        .unwrap_or_else(|err| panic!("{D}D minimal simplex fixture should build: {err}"))
        .into_triangulation()
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
fn build_flippable_dt_2d() -> Tri2 {
    let vertices = FLIPPABLE_POINTS_2D
        .iter()
        .map(|coords| vertex!(*coords).expect("stable 2D fixture coordinates"))
        .collect::<Vec<_>>();
    let simplices = vec![vec![0, 1, 2], vec![0, 2, 3]];

    let tri = DelaunayTriangulationBuilder::try_from_vertices_and_simplices(&vertices, &simplices)
        .expect("explicit 2D fixture connectivity should parse")
        .build_with_kernel(&RobustKernel::new())
        .expect("stable 2D fixture should build")
        .into_triangulation();
    assert_topology_and_delaunay_valid(&tri, "stable 2D fixture before local edits");
    tri
}

/// Builds a single-triangle 2D fixture for boundary-facet rejection checks.
fn build_single_triangle_dt_2d() -> Tri2 {
    let vertices = vec![
        vertex!([0.0, 0.0]).expect("single-triangle fixture coordinate"),
        vertex!([1.0, 0.0]).expect("single-triangle fixture coordinate"),
        vertex!([0.0, 1.0]).expect("single-triangle fixture coordinate"),
    ];

    DelaunayTriangulationBuilder::new(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .build_with_kernel(&RobustKernel::new())
        .expect("single-triangle fixture should build")
        .into_triangulation()
}

/// Searches the 2D fixture for an edge facet whose public k=2 move succeeds.
fn flippable_k2_facet_2d(tri: &Tri2) -> FacetHandle {
    for facet in tri.facets() {
        let facet = facet.expect("2D fixture facets should reborrow as live views");
        let facet = facet.handle();
        let mut trial = tri.clone();
        let attempt = match trial.propose_pachner(PachnerMove::K2 { facet }) {
            Ok(proposal) => proposal.attempt_on(&mut trial),
            Err(error) => Err(error),
        };
        if attempt.is_ok() && topology_and_delaunay_valid(&trial) {
            return facet;
        }
    }
    panic!("stable 2D fixture should contain a public k=2 candidate");
}

/// Captures 2D topology by stable UUIDs so failed attempts can prove non-mutation.
fn snapshot_topology_2d(tri: &Tri2) -> TopologySnapshot {
    snapshot_topology_generic(tri)
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
fn snapshot_topology_generic<const D: usize>(tri: &Tri<D>) -> TopologySnapshot {
    let mut vertex_uuids = tri
        .vertices()
        .map(|(_, vertex)| vertex.uuid())
        .collect::<Vec<_>>();
    vertex_uuids.sort();

    let mut simplex_vertex_uuids = tri
        .simplices()
        .map(|(_, simplex)| {
            let mut uuids = simplex
                .vertices()
                .iter()
                .map(|vertex_key| {
                    tri.vertex(*vertex_key)
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
    tri: Tri<D>,
    pachner_move: PachnerMove<(), D>,
    assert_error: impl Fn(&FlipError),
) {
    let before = snapshot_topology_generic(&tri);
    let feasibility = tri
        .propose_pachner(pachner_move)
        .expect_err("Pachner proposal parsing should reject invalid request");
    assert_error(&feasibility);
    assert_eq!(snapshot_topology_generic(&tri), before);

    let trial = tri;
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
fn assert_public_k1_insert_feasibility_smoke<const D: usize>()
where
    RobustKernel<f64>: delaunay::prelude::geometry::ExactPredicates<D>,
{
    let tri = build_minimal_simplex_dt::<D>();
    let simplex_key = first_simplex_generic(&tri);
    let vertex: Vertex<(), D> = vertex!(simplex_centroid_generic(&tri, simplex_key))
        .unwrap_or_else(|err| panic!("{D}D simplex centroid should be finite: {err}"));
    let vertex_uuid = vertex.uuid();
    let pachner_move = PachnerMove::K1Insert {
        simplex_key,
        vertex,
    };

    let proposal = tri
        .propose_pachner(pachner_move)
        .unwrap_or_else(|err| panic!("{D}D public k=1 proposal should succeed: {err:?}"));
    let feasibility = proposal
        .can_attempt_on(&tri)
        .unwrap_or_else(|err| panic!("{D}D public k=1 feasibility should succeed: {err:?}"))
        .clone();
    assert_pachner_feasibility_contract(
        &feasibility,
        BistellarFlipKind::try_k1(D).expect("fixture dimension should support k=1"),
        FlipDirection::Forward,
    );
    assert!(feasibility.inserted_face_vertices.is_none());

    let mut trial = tri;
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
        BistellarFlipKind::try_k1(D)
            .expect("fixture dimension should support k=1")
            .inverse(),
        FlipDirection::Inverse,
    );

    let removed = remove_proposal
        .attempt_on(&mut trial)
        .unwrap_or_else(|err| panic!("{D}D public k=1 remove mutation should succeed: {err:?}"));
    assert_pachner_result_contract(
        &removed,
        BistellarFlipKind::try_k1(D)
            .expect("fixture dimension should support k=1")
            .inverse(),
        FlipDirection::Inverse,
    );
    assert_topology_and_delaunay_valid(&trial, &format!("{D}D k=1 mutation"));
}

/// Returns any live simplex key from a generic D-dimensional fixture.
fn first_simplex_generic<const D: usize>(tri: &Tri<D>) -> SimplexKey {
    tri.simplices()
        .next()
        .map(|(simplex_key, _)| simplex_key)
        .expect("fixture should contain simplices")
}

/// Computes a simplex centroid for generic dimension smoke tests.
fn simplex_centroid_generic<const D: usize>(tri: &Tri<D>, simplex_key: SimplexKey) -> [f64; D] {
    *tri.simplex_barycenter(simplex_key)
        .expect("simplex key should have a finite barycenter")
        .coords()
}

/// Converts a 2D facet handle into the edge key represented by that facet.
fn edge_for_facet_2d(tri: &Tri2, facet: FacetHandle) -> EdgeKey {
    let view = tri
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
    find_live_edge(tri, a, b).expect("facet endpoints should form a live edge")
}

/// Parses the inserted edge reported by a 2D k=2 move.
fn inserted_edge_2d(tri: &Tri2, vertices: &[VertexKey]) -> EdgeKey {
    let [a, b] = vertices else {
        panic!(
            "2D k=2 move should report exactly two inserted edge vertices, got {}",
            vertices.len()
        );
    };
    find_live_edge(tri, *a, *b).expect("reported inserted vertices should form a live edge")
}

/// Checks that a rejected detached proposal leaves the live topology byte-for-byte equivalent.
fn assert_failed_attempt_preserves_topology(
    tri: &mut Tri4,
    proposal: PachnerProposal<(), 4>,
    assert_error: impl Fn(&FlipError),
) {
    let before = snapshot_topology(tri);
    let proposal_generation = proposal.topology_generation();
    let current_generation = tri.topology_generation();

    let feasibility_err = proposal
        .can_attempt_on(tri)
        .expect_err("stale Pachner proposal feasibility should fail");
    assert_error(&feasibility_err);
    assert_stale_proposal_generation(&feasibility_err, proposal_generation, current_generation);
    assert_eq!(snapshot_topology(tri), before);

    let err = proposal
        .attempt_on(tri)
        .expect_err("stale Pachner proposal should fail");
    assert_error(&err);
    assert_stale_proposal_generation(&err, proposal_generation, current_generation);
    tri.is_valid_structure()
        .expect("failed Pachner attempt should preserve TDS validity");
    assert_eq!(snapshot_topology(tri), before);
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
    assert_pachner_result_contract(
        result,
        BistellarFlipKind::try_k1(4).expect("4D k=1 move kind should be valid"),
        FlipDirection::Forward,
    );
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
fn assert_stale_k1_insert_preserves_topology(mut tri: Tri4) {
    let stale_simplex = first_simplex(&tri);
    let vertex_coords = simplex_centroid(&tri, stale_simplex);
    let vertex: Vertex<(), 4> = vertex!(vertex_coords).unwrap();
    let vertex_uuid = vertex.uuid();
    let stale_proposal = tri
        .propose_pachner(PachnerMove::K1Insert {
            simplex_key: stale_simplex,
            vertex,
        })
        .expect("initial k=1 insert proposal should be valid");
    let inserted = stale_proposal
        .clone()
        .attempt_on(&mut tri)
        .expect("initial k=1 insert should make the proposal stale");
    let inserted_vertex = vertex_key_by_uuid(&tri, vertex_uuid)
        .expect("initial k=1 insert should create the requested vertex");
    assert_k1_insert_result(&inserted, inserted_vertex);
    assert_failed_attempt_preserves_topology(&mut tri, stale_proposal, |err| {
        assert_matches!(
            err,
            FlipError::StaleTopologyProposal { .. },
            "unexpected stale k=1 insert error: {err:?}"
        );
    });
}

/// Makes a k=1 remove proposal stale, then proves retrying it is failure-atomic.
fn assert_stale_k1_remove_preserves_topology(mut tri: Tri4) {
    let simplex_key = first_simplex(&tri);
    let vertex: Vertex<(), 4> = vertex!(simplex_centroid(&tri, simplex_key)).unwrap();
    let vertex_uuid = vertex.uuid();
    let insert_proposal = tri
        .propose_pachner(PachnerMove::K1Insert {
            simplex_key,
            vertex,
        })
        .expect("k=1 insert proposal should be valid");
    let inserted = insert_proposal
        .attempt_on(&mut tri)
        .expect("k=1 insert should create a removable vertex");
    let vertex_key =
        vertex_key_by_uuid(&tri, vertex_uuid).expect("inserted k=1 vertex should be present");
    assert_k1_insert_result(&inserted, vertex_key);
    let stale_proposal = tri
        .propose_pachner(PachnerMove::K1Remove { vertex_key })
        .expect("k=1 remove proposal should be valid");
    let removed = stale_proposal
        .clone()
        .attempt_on(&mut tri)
        .expect("k=1 remove should make the proposal stale");
    assert!(!removed.removed_simplices.is_empty());

    assert_failed_attempt_preserves_topology(&mut tri, stale_proposal, |err| {
        assert_matches!(
            err,
            FlipError::StaleTopologyProposal { .. },
            "unexpected stale k=1 remove error: {err:?}"
        );
    });
}

/// Makes a k=2 facet proposal stale, then proves retrying it is failure-atomic.
#[cfg(feature = "slow-tests")]
fn assert_stale_k2_preserves_topology(mut tri: Tri4) {
    let facet = flippable_k2_facet(&tri);
    let stale_proposal = tri
        .propose_pachner(PachnerMove::K2 { facet })
        .expect("k=2 proposal should be valid");
    let flipped = stale_proposal
        .clone()
        .attempt_on(&mut tri)
        .expect("k=2 flip should make its proposal stale");
    assert_eq!(flipped.inserted_face_vertices.len(), 2);
    assert!(!flipped.new_simplices.is_empty());

    assert_failed_attempt_preserves_topology(&mut tri, stale_proposal, |err| {
        assert_matches!(
            err,
            FlipError::StaleTopologyProposal { .. },
            "unexpected stale k=2 error: {err:?}"
        );
    });
}

/// Makes an inverse k=2 edge proposal stale, then proves retrying it is failure-atomic.
#[cfg(feature = "slow-tests")]
fn assert_stale_k2_inverse_preserves_topology(mut tri: Tri4) {
    let facet = flippable_k2_facet(&tri);
    let info = tri
        .propose_pachner(PachnerMove::K2 { facet })
        .expect("selected fixture facet should produce a k=2 proposal")
        .attempt_on(&mut tri)
        .expect("k=2 flip should create an inverse edge");
    let edge = inserted_edge(&tri, &info.inserted_face_vertices);
    let stale_proposal = tri
        .propose_pachner(PachnerMove::K2Inverse { edge })
        .expect("k=2 inverse proposal should be valid");
    let inverse = stale_proposal
        .clone()
        .attempt_on(&mut tri)
        .expect("k=2 inverse should make its proposal stale");
    assert!(!inverse.removed_simplices.is_empty());

    assert_failed_attempt_preserves_topology(&mut tri, stale_proposal, |err| {
        assert_matches!(
            err,
            FlipError::StaleTopologyProposal { .. },
            "unexpected stale inverse k=2 error: {err:?}"
        );
    });
}

/// Makes a k=3 ridge proposal stale, then proves retrying it is failure-atomic.
#[cfg(feature = "slow-tests")]
fn assert_stale_k3_preserves_topology(mut tri: Tri4) {
    let ridge = flippable_k3_ridge(&tri);
    let stale_proposal = tri
        .propose_pachner(PachnerMove::K3 { ridge })
        .expect("k=3 proposal should be valid");
    let flipped = stale_proposal
        .clone()
        .attempt_on(&mut tri)
        .expect("k=3 flip should make its proposal stale");
    assert_eq!(flipped.inserted_face_vertices.len(), 3);
    assert!(!flipped.new_simplices.is_empty());

    assert_failed_attempt_preserves_topology(&mut tri, stale_proposal, |err| {
        assert_matches!(
            err,
            FlipError::StaleTopologyProposal { .. },
            "unexpected stale k=3 error: {err:?}"
        );
    });
}

/// Makes an inverse k=3 triangle proposal stale, then proves retrying it is failure-atomic.
#[cfg(feature = "slow-tests")]
fn assert_stale_k3_inverse_preserves_topology(mut tri: Tri4) {
    let ridge = flippable_k3_ridge(&tri);
    let info = tri
        .propose_pachner(PachnerMove::K3 { ridge })
        .expect("selected fixture ridge should produce a k=3 proposal")
        .attempt_on(&mut tri)
        .expect("k=3 flip should create an inverse triangle");
    let triangle = inserted_triangle(&info.inserted_face_vertices);
    let stale_proposal = tri
        .propose_pachner(PachnerMove::K3Inverse { triangle })
        .expect("k=3 inverse proposal should be valid");
    let inverse = stale_proposal
        .clone()
        .attempt_on(&mut tri)
        .expect("k=3 inverse should make its proposal stale");
    assert!(!inverse.removed_simplices.is_empty());

    assert_failed_attempt_preserves_topology(&mut tri, stale_proposal, |err| {
        assert_matches!(
            err,
            FlipError::StaleTopologyProposal { .. },
            "unexpected stale inverse k=3 error: {err:?}"
        );
    });
}

/// Captures topology by stable UUIDs so slotmap key reuse cannot hide mutations.
fn snapshot_topology(tri: &Tri4) -> TopologySnapshot {
    snapshot_topology_generic(tri)
}

/// Returns a simplex from the stable fixture for tests that only need any live simplex.
fn first_simplex(tri: &Tri4) -> SimplexKey {
    tri.simplices()
        .next()
        .map(|(simplex_key, _)| simplex_key)
        .expect("stable fixture should contain simplices")
}

/// Computes an interior-ish point for k=1 insertion into a known simplex.
fn simplex_centroid(tri: &Tri4, simplex_key: SimplexKey) -> [f64; 4] {
    *tri.simplex_barycenter(simplex_key)
        .expect("simplex key should have a finite barycenter")
        .coords()
}

/// Applies a k=1 insert/remove pair and checks the reported move metadata.
#[cfg(feature = "slow-tests")]
fn roundtrip_k1(tri: &mut Tri4) {
    let simplex_key = first_simplex(tri);
    let new_vertex: Vertex<(), 4> = vertex!(simplex_centroid(tri, simplex_key)).unwrap();
    let new_uuid = new_vertex.uuid();
    let inserted = tri
        .propose_pachner(PachnerMove::K1Insert {
            simplex_key,
            vertex: new_vertex,
        })
        .expect("stable simplex should produce a k=1 insertion proposal")
        .attempt_on(tri)
        .expect("k=1 insert should succeed on stable 4D fixture");
    assert_eq!(inserted.inserted_face_vertices.len(), 1);

    let inserted_key =
        vertex_key_by_uuid(tri, new_uuid).expect("inserted k=1 vertex should be present");
    let removed = tri
        .propose_pachner(PachnerMove::K1Remove {
            vertex_key: inserted_key,
        })
        .expect("inserted vertex should produce a k=1 removal proposal")
        .attempt_on(tri)
        .expect("k=1 remove should invert insert");
    assert_pachner_result_contract(
        &removed,
        BistellarFlipKind::try_k1(4)
            .expect("4D k=1 move kind should be valid")
            .inverse(),
        FlipDirection::Inverse,
    );
}

/// Searches the fixture for a k=2 facet that also supports the public inverse API.
#[cfg(feature = "slow-tests")]
fn flippable_k2_facet(tri: &Tri4) -> FacetHandle {
    for facet in tri.facets() {
        let facet = facet.expect("4D fixture facets should reborrow as live views");
        let facet = facet.handle();
        let mut trial = tri.clone();
        let Ok(proposal) = trial.propose_pachner(PachnerMove::K2 { facet }) else {
            continue;
        };
        let Ok(info) = proposal.attempt_on(&mut trial) else {
            continue;
        };
        let edge = inserted_edge(&trial, &info.inserted_face_vertices);
        let Ok(inverse) = trial.propose_pachner(PachnerMove::K2Inverse { edge }) else {
            continue;
        };
        if inverse.attempt_on(&mut trial).is_ok() && topology_and_delaunay_valid(&trial) {
            return facet;
        }
    }
    panic!("stable 4D fixture should contain a public k=2 roundtrip candidate");
}

/// Applies a k=2 forward/inverse pair and checks both move reports.
#[cfg(feature = "slow-tests")]
fn roundtrip_k2(tri: &mut Tri4, facet: FacetHandle) {
    let info: PachnerMoveResult<4> = tri
        .propose_pachner(PachnerMove::K2 { facet })
        .expect("selected facet should produce a k=2 proposal")
        .attempt_on(tri)
        .expect("k=2 flip should succeed on selected stable 4D facet");
    assert_pachner_result_contract(
        &info,
        BistellarFlipKind::try_k2(4).expect("4D k=2 move kind should be valid"),
        FlipDirection::Forward,
    );
    let edge = inserted_edge(tri, &info.inserted_face_vertices);
    let inverse = tri
        .propose_pachner(PachnerMove::K2Inverse { edge })
        .expect("inserted edge should produce a k=2 inverse proposal")
        .attempt_on(tri)
        .expect("k=2 inverse should succeed after k=2 flip");
    assert_pachner_result_contract(
        &inverse,
        BistellarFlipKind::try_k2(4)
            .expect("4D k=2 move kind should be valid")
            .inverse(),
        FlipDirection::Inverse,
    );
}

/// Parses the inserted face of a k=2 move into the edge expected by the inverse API.
#[cfg(feature = "slow-tests")]
fn inserted_edge(tri: &Tri4, vertices: &[VertexKey]) -> EdgeKey {
    let [a, b] = vertices else {
        panic!(
            "k=2 flip should report an inserted edge, got {} vertices",
            vertices.len()
        );
    };
    find_live_edge(tri, *a, *b).expect("k=2 flip should report a real inserted edge")
}

/// Searches the fixture for a k=3 ridge that also supports the public inverse API.
#[cfg(feature = "slow-tests")]
fn flippable_k3_ridge(tri: &Tri4) -> RidgeHandle {
    for ridge in tri.ridge_handles() {
        let ridge = ridge.expect("4D fixture ridges should produce live handles");
        let mut trial = tri.clone();
        let Ok(proposal) = trial.propose_pachner(PachnerMove::K3 { ridge }) else {
            continue;
        };
        let Ok(info) = proposal.attempt_on(&mut trial) else {
            continue;
        };
        let triangle = inserted_triangle(&info.inserted_face_vertices);
        let Ok(inverse) = trial.propose_pachner(PachnerMove::K3Inverse { triangle }) else {
            continue;
        };
        if inverse.attempt_on(&mut trial).is_ok() && topology_and_delaunay_valid(&trial) {
            return ridge;
        }
    }
    panic!("stable 4D fixture should contain a public k=3 roundtrip candidate");
}

/// Applies a k=3 forward/inverse pair and checks both move reports.
#[cfg(feature = "slow-tests")]
fn roundtrip_k3(tri: &mut Tri4, ridge: RidgeHandle) {
    let info: PachnerMoveResult<4> = tri
        .propose_pachner(PachnerMove::K3 { ridge })
        .expect("selected ridge should produce a k=3 proposal")
        .attempt_on(tri)
        .expect("k=3 flip should succeed on selected stable 4D ridge");
    assert_pachner_result_contract(
        &info,
        BistellarFlipKind::try_k3(4).expect("4D k=3 move kind should be valid"),
        FlipDirection::Forward,
    );
    let inverse = tri
        .propose_pachner(PachnerMove::K3Inverse {
            triangle: inserted_triangle(&info.inserted_face_vertices),
        })
        .expect("inserted triangle should produce a k=3 inverse proposal")
        .attempt_on(tri)
        .expect("k=3 inverse should succeed after k=3 flip");
    assert_pachner_result_contract(
        &inverse,
        BistellarFlipKind::try_k3(4)
            .expect("4D k=3 move kind should be valid")
            .inverse(),
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
