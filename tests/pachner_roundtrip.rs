#![forbid(unsafe_code)]

//! Public API roundtrip tests for Pachner/bistellar flips.

use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayError, DelaunayResult, DelaunayTriangulation,
    InsertionOrderStrategy, TopologyGuarantee, Vertex, vertex,
};
use delaunay::prelude::geometry::RobustKernel;
use delaunay::prelude::pachner::{
    BistellarFlipKind, EdgeKey, FacetHandle, FlipDirection, FlipError, PachnerMove,
    PachnerMoveResult, PachnerMoves, RidgeHandle, SimplexKey, TriangleHandle, VertexKey,
};
use uuid::Uuid;

type Dt4 = DelaunayTriangulation<RobustKernel<f64>, (), (), 4>;

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

#[test]
fn public_pachner_roundtrips_preserve_stable_4d_topology() {
    let base = build_stable_dt_4d();
    base.validate().expect("stable 4D fixture should validate");
    let before = snapshot_topology(&base);

    let mut k1 = base.clone();
    roundtrip_k1(&mut k1);
    k1.validate().expect("k=1 roundtrip should validate");
    assert_eq!(snapshot_topology(&k1), before);

    let k2_facet = flippable_k2_facet(&base);
    let mut k2 = base.clone();
    roundtrip_k2(&mut k2, k2_facet);
    k2.validate().expect("k=2 roundtrip should validate");
    assert_eq!(snapshot_topology(&k2), before);

    let k3_ridge = flippable_k3_ridge(&base);
    let mut k3 = base;
    roundtrip_k3(&mut k3, k3_ridge);
    k3.validate().expect("k=3 roundtrip should validate");
    assert_eq!(snapshot_topology(&k3), before);
}

#[test]
fn stale_pachner_requests_fail_without_mutating_topology() {
    let base = build_stable_dt_4d();

    assert_stale_k1_insert_preserves_topology(base.clone());
    assert_stale_k1_remove_preserves_topology(base.clone());
    assert_stale_k2_preserves_topology(base.clone());
    assert_stale_k2_inverse_preserves_topology(base.clone());
    assert_stale_k3_preserves_topology(base.clone());
    assert_stale_k3_inverse_preserves_topology(base);
}

#[test]
fn stale_pachner_error_propagates_through_delaunay_result() {
    let mut dt = build_stable_dt_4d();
    let stale_simplex = first_simplex(&dt);
    let vertex_coords = simplex_centroid(&dt, stale_simplex);
    let vertex: Vertex<(), 4> =
        vertex!(vertex_coords).expect("centroid of a stable simplex should be a valid vertex");
    let vertex_uuid = vertex.uuid();
    let inserted = dt
        .attempt_pachner(PachnerMove::K1Insert {
            simplex_key: stale_simplex,
            vertex,
        })
        .expect("initial k=1 insert should make the simplex key stale");
    let inserted_vertex = dt
        .tds()
        .vertex_key_from_uuid(&vertex_uuid)
        .expect("initial k=1 insert should create the requested vertex");
    assert_k1_insert_result(&inserted, inserted_vertex);
    let before_failed_attempt = snapshot_topology(&dt);

    let err = try_stale_k1_insert(&mut dt, stale_simplex, vertex_coords)
        .expect_err("stale Pachner failure should propagate through DelaunayResult");
    assert!(
        matches!(
            &err,
            DelaunayError::Flip {
                source: FlipError::MissingSimplex { simplex_key },
            } if *simplex_key == stale_simplex
        ),
        "unexpected DelaunayResult error for stale Pachner move: {err:?}"
    );
    dt.tds()
        .is_valid()
        .expect("failed Pachner attempt should preserve TDS validity");
    assert_eq!(snapshot_topology(&dt), before_failed_attempt);
}

/// Attempts a stale k=1 insert through the public `DelaunayResult` alias.
#[expect(
    clippy::result_large_err,
    reason = "test intentionally exercises DelaunayResult error propagation"
)]
fn try_stale_k1_insert(
    dt: &mut Dt4,
    stale_simplex: SimplexKey,
    vertex_coords: [f64; 4],
) -> DelaunayResult<()> {
    let vertex: Vertex<(), 4> = vertex!(vertex_coords)?;
    let vertex_uuid = vertex.uuid();
    let inserted = dt.attempt_pachner(PachnerMove::K1Insert {
        simplex_key: stale_simplex,
        vertex,
    })?;
    let inserted_vertex = dt
        .tds()
        .vertex_key_from_uuid(&vertex_uuid)
        .expect("unexpected successful stale insert should create the requested vertex");
    assert_k1_insert_result(&inserted, inserted_vertex);
    Ok(())
}

/// Builds the deterministic 4D fixture used to find reversible public Pachner moves.
fn build_stable_dt_4d() -> Dt4 {
    let vertices = STABLE_POINTS_4D
        .iter()
        .map(|coords| vertex!(*coords).unwrap())
        .collect::<Vec<_>>();
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);

    DelaunayTriangulation::try_with_topology_guarantee_and_options(
        &RobustKernel::new(),
        &vertices,
        TopologyGuarantee::PLManifold,
        options,
    )
    .expect("stable 4D fixture should build")
}

/// Checks that a rejected detached move leaves the live topology byte-for-byte equivalent.
fn assert_failed_attempt_preserves_topology(
    dt: &mut Dt4,
    pachner_move: PachnerMove<(), 4>,
    assert_error: impl FnOnce(&FlipError),
) {
    let before = snapshot_topology(dt);
    let err = dt
        .attempt_pachner(pachner_move)
        .expect_err("stale Pachner move should fail");
    assert_error(&err);
    dt.tds()
        .is_valid()
        .expect("failed Pachner attempt should preserve TDS validity");
    assert_eq!(snapshot_topology(dt), before);
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
    let inserted = dt
        .attempt_pachner(PachnerMove::K1Insert {
            simplex_key: stale_simplex,
            vertex,
        })
        .expect("initial k=1 insert should make the simplex key stale");
    let inserted_vertex = dt
        .tds()
        .vertex_key_from_uuid(&vertex_uuid)
        .expect("initial k=1 insert should create the requested vertex");
    assert_k1_insert_result(&inserted, inserted_vertex);
    let stale_attempt_vertex: Vertex<(), 4> = vertex!(vertex_coords).unwrap();

    assert_failed_attempt_preserves_topology(
        &mut dt,
        PachnerMove::K1Insert {
            simplex_key: stale_simplex,
            vertex: stale_attempt_vertex,
        },
        |err| {
            assert!(
                matches!(
                    err,
                    FlipError::MissingSimplex { simplex_key } if *simplex_key == stale_simplex
                ),
                "unexpected stale k=1 insert error: {err:?}"
            );
        },
    );
}

/// Makes a k=1 remove proposal stale, then proves retrying it is failure-atomic.
fn assert_stale_k1_remove_preserves_topology(mut dt: Dt4) {
    let simplex_key = first_simplex(&dt);
    let vertex: Vertex<(), 4> = vertex!(simplex_centroid(&dt, simplex_key)).unwrap();
    let vertex_uuid = vertex.uuid();
    let inserted = dt
        .attempt_pachner(PachnerMove::K1Insert {
            simplex_key,
            vertex,
        })
        .expect("k=1 insert should create a removable vertex");
    let vertex_key = dt
        .tds()
        .vertex_key_from_uuid(&vertex_uuid)
        .expect("inserted k=1 vertex should be present");
    assert_k1_insert_result(&inserted, vertex_key);
    let removed = dt
        .attempt_pachner(PachnerMove::K1Remove { vertex_key })
        .expect("k=1 remove should make the vertex key stale");
    assert!(!removed.removed_simplices.is_empty());

    assert_failed_attempt_preserves_topology(
        &mut dt,
        PachnerMove::K1Remove { vertex_key },
        |err| {
            assert!(
                matches!(
                    err,
                    FlipError::MissingVertex { vertex_key: missing } if *missing == vertex_key
                ),
                "unexpected stale k=1 remove error: {err:?}"
            );
        },
    );
}

/// Makes a k=2 facet proposal stale, then proves retrying it is failure-atomic.
fn assert_stale_k2_preserves_topology(mut dt: Dt4) {
    let facet = flippable_k2_facet(&dt);
    let flipped = dt
        .attempt_pachner(PachnerMove::K2 { facet })
        .expect("k=2 flip should make its source facet stale");
    assert_eq!(flipped.inserted_face_vertices.len(), 2);
    assert!(!flipped.new_simplices.is_empty());
    let stale_simplex = facet.simplex_key();

    assert_failed_attempt_preserves_topology(&mut dt, PachnerMove::K2 { facet }, |err| {
        assert!(
            matches!(
                err,
                FlipError::MissingSimplex { simplex_key } if *simplex_key == stale_simplex
            ),
            "unexpected stale k=2 error: {err:?}"
        );
    });
}

/// Makes an inverse k=2 edge proposal stale, then proves retrying it is failure-atomic.
fn assert_stale_k2_inverse_preserves_topology(mut dt: Dt4) {
    let facet = flippable_k2_facet(&dt);
    let info = dt
        .attempt_pachner(PachnerMove::K2 { facet })
        .expect("k=2 flip should create an inverse edge");
    let edge = inserted_edge(&dt, &info.inserted_face_vertices);
    let inverse = dt
        .attempt_pachner(PachnerMove::K2Inverse { edge })
        .expect("k=2 inverse should make its edge stale");
    assert!(!inverse.removed_simplices.is_empty());

    assert_failed_attempt_preserves_topology(&mut dt, PachnerMove::K2Inverse { edge }, |err| {
        assert!(
            matches!(
                err,
                FlipError::InvalidEdgeMultiplicity {
                    found: 0,
                    expected: 4
                }
            ),
            "unexpected stale inverse k=2 error: {err:?}"
        );
    });
}

/// Makes a k=3 ridge proposal stale, then proves retrying it is failure-atomic.
fn assert_stale_k3_preserves_topology(mut dt: Dt4) {
    let ridge = flippable_k3_ridge(&dt);
    let flipped = dt
        .attempt_pachner(PachnerMove::K3 { ridge })
        .expect("k=3 flip should make its source ridge stale");
    assert_eq!(flipped.inserted_face_vertices.len(), 3);
    assert!(!flipped.new_simplices.is_empty());
    let stale_simplex = ridge.simplex_key();

    assert_failed_attempt_preserves_topology(&mut dt, PachnerMove::K3 { ridge }, |err| {
        assert!(
            matches!(
                err,
                FlipError::MissingSimplex { simplex_key } if *simplex_key == stale_simplex
            ),
            "unexpected stale k=3 error: {err:?}"
        );
    });
}

/// Makes an inverse k=3 triangle proposal stale, then proves retrying it is failure-atomic.
fn assert_stale_k3_inverse_preserves_topology(mut dt: Dt4) {
    let ridge = flippable_k3_ridge(&dt);
    let info = dt
        .attempt_pachner(PachnerMove::K3 { ridge })
        .expect("k=3 flip should create an inverse triangle");
    let triangle = inserted_triangle(&info.inserted_face_vertices);
    let inverse = dt
        .attempt_pachner(PachnerMove::K3Inverse { triangle })
        .expect("k=3 inverse should make its triangle stale");
    assert!(!inverse.removed_simplices.is_empty());

    assert_failed_attempt_preserves_topology(&mut dt, PachnerMove::K3Inverse { triangle }, |err| {
        assert!(
            matches!(
                err,
                FlipError::InvalidTriangleMultiplicity {
                    found: 0,
                    expected: 3
                }
            ),
            "unexpected stale inverse k=3 error: {err:?}"
        );
    });
}

/// Captures topology by stable UUIDs so slotmap key reuse cannot hide mutations.
fn snapshot_topology(dt: &Dt4) -> TopologySnapshot {
    let tds = dt.tds();
    let mut vertex_uuids = tds
        .vertices()
        .map(|(_, vertex)| vertex.uuid())
        .collect::<Vec<_>>();
    vertex_uuids.sort();

    let mut simplex_vertex_uuids = tds
        .simplices()
        .map(|(_, simplex)| {
            let mut uuids = simplex
                .vertices()
                .iter()
                .map(|&vkey| {
                    tds.vertex(vkey)
                        .expect("simplex vertex key should exist")
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

/// Returns a simplex from the stable fixture for tests that only need any live simplex.
fn first_simplex(dt: &Dt4) -> SimplexKey {
    dt.simplices()
        .next()
        .map(|(simplex_key, _)| simplex_key)
        .expect("stable fixture should contain simplices")
}

/// Computes an interior-ish point for k=1 insertion into a known simplex.
fn simplex_centroid(dt: &Dt4, simplex_key: SimplexKey) -> [f64; 4] {
    let simplex = dt
        .tds()
        .simplex(simplex_key)
        .expect("simplex key should exist");
    let mut coords = [0.0; 4];
    for &vkey in simplex.vertices() {
        let vertex = dt.tds().vertex(vkey).expect("vertex key should exist");
        for (coord, value) in coords.iter_mut().zip(vertex.point().coords()) {
            *coord += *value;
        }
    }

    let vertex_count = u32::try_from(simplex.vertices().len())
        .map(f64::from)
        .expect("simplex vertex count should fit in u32");
    for coord in &mut coords {
        *coord /= vertex_count;
    }
    coords
}

/// Applies a k=1 insert/remove pair and checks the reported move metadata.
fn roundtrip_k1(dt: &mut Dt4) {
    let simplex_key = first_simplex(dt);
    let new_vertex: Vertex<(), 4> = vertex!(simplex_centroid(dt, simplex_key)).unwrap();
    let new_uuid = new_vertex.uuid();
    let inserted = dt
        .attempt_pachner(PachnerMove::K1Insert {
            simplex_key,
            vertex: new_vertex,
        })
        .expect("k=1 insert should succeed on stable 4D fixture");
    assert_eq!(inserted.inserted_face_vertices.len(), 1);

    let inserted_key = dt
        .tds()
        .vertex_key_from_uuid(&new_uuid)
        .expect("inserted k=1 vertex should be present");
    let removed = dt
        .attempt_pachner(PachnerMove::K1Remove {
            vertex_key: inserted_key,
        })
        .expect("k=1 remove should invert insert");
    assert_pachner_result_contract(
        &removed,
        BistellarFlipKind::k1(4).inverse(),
        FlipDirection::Inverse,
    );
}

/// Searches the fixture for a k=2 facet that also supports the public inverse API.
fn flippable_k2_facet(dt: &Dt4) -> FacetHandle {
    for (simplex_key, simplex) in dt.simplices() {
        let Some(neighbors) = simplex.neighbors() else {
            continue;
        };
        for (facet_index, neighbor) in neighbors.enumerate() {
            if neighbor.is_none() {
                continue;
            }
            let facet = FacetHandle::try_new(
                dt.tds(),
                simplex_key,
                u8::try_from(facet_index).expect("facet index should fit in u8"),
            )
            .expect("interior facet index should be valid");
            let mut trial = dt.clone();
            let Ok(info) = trial.attempt_pachner(PachnerMove::K2 { facet }) else {
                continue;
            };
            let edge = inserted_edge(&trial, &info.inserted_face_vertices);
            if trial
                .attempt_pachner(PachnerMove::K2Inverse { edge })
                .is_ok()
                && trial.validate().is_ok()
            {
                return facet;
            }
        }
    }
    panic!("stable 4D fixture should contain a public k=2 roundtrip candidate");
}

/// Applies a k=2 forward/inverse pair and checks both move reports.
fn roundtrip_k2(dt: &mut Dt4, facet: FacetHandle) {
    let info: PachnerMoveResult<4> = dt
        .attempt_pachner(PachnerMove::K2 { facet })
        .expect("k=2 flip should succeed on selected stable 4D facet");
    assert_pachner_result_contract(&info, BistellarFlipKind::k2(4), FlipDirection::Forward);
    let edge = inserted_edge(dt, &info.inserted_face_vertices);
    let inverse = dt
        .attempt_pachner(PachnerMove::K2Inverse { edge })
        .expect("k=2 inverse should succeed after k=2 flip");
    assert_pachner_result_contract(
        &inverse,
        BistellarFlipKind::k2(4).inverse(),
        FlipDirection::Inverse,
    );
}

/// Parses the inserted face of a k=2 move into the edge expected by the inverse API.
fn inserted_edge(dt: &Dt4, vertices: &[VertexKey]) -> EdgeKey {
    let [a, b] = vertices else {
        panic!(
            "k=2 flip should report an inserted edge, got {} vertices",
            vertices.len()
        );
    };
    EdgeKey::try_new(dt.tds(), *a, *b).expect("k=2 flip should report a real inserted edge")
}

/// Searches the fixture for a k=3 ridge that also supports the public inverse API.
fn flippable_k3_ridge(dt: &Dt4) -> RidgeHandle {
    for (simplex_key, simplex) in dt.simplices() {
        for i in 0..simplex.number_of_vertices() {
            for j in (i + 1)..simplex.number_of_vertices() {
                let ridge = RidgeHandle::try_new(
                    dt.tds(),
                    simplex_key,
                    u8::try_from(i).expect("ridge index should fit in u8"),
                    u8::try_from(j).expect("ridge index should fit in u8"),
                )
                .expect("ridge indices should be valid");
                let mut trial = dt.clone();
                let Ok(info) = trial.attempt_pachner(PachnerMove::K3 { ridge }) else {
                    continue;
                };
                let triangle = inserted_triangle(&info.inserted_face_vertices);
                if trial
                    .attempt_pachner(PachnerMove::K3Inverse { triangle })
                    .is_ok()
                    && trial.validate().is_ok()
                {
                    return ridge;
                }
            }
        }
    }
    panic!("stable 4D fixture should contain a public k=3 roundtrip candidate");
}

/// Applies a k=3 forward/inverse pair and checks both move reports.
fn roundtrip_k3(dt: &mut Dt4, ridge: RidgeHandle) {
    let info: PachnerMoveResult<4> = dt
        .attempt_pachner(PachnerMove::K3 { ridge })
        .expect("k=3 flip should succeed on selected stable 4D ridge");
    assert_pachner_result_contract(&info, BistellarFlipKind::k3(4), FlipDirection::Forward);
    let inverse = dt
        .attempt_pachner(PachnerMove::K3Inverse {
            triangle: inserted_triangle(&info.inserted_face_vertices),
        })
        .expect("k=3 inverse should succeed after k=3 flip");
    assert_pachner_result_contract(
        &inverse,
        BistellarFlipKind::k3(4).inverse(),
        FlipDirection::Inverse,
    );
}

/// Parses the inserted face of a k=3 move into the triangle expected by the inverse API.
fn inserted_triangle(vertices: &[VertexKey]) -> TriangleHandle {
    let [a, b, c] = vertices else {
        panic!(
            "k=3 flip should report an inserted triangle, got {} vertices",
            vertices.len()
        );
    };
    TriangleHandle::try_new(*a, *b, *c).expect("k=3 flip should report a valid inserted triangle")
}
