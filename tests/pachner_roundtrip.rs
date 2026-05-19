#![forbid(unsafe_code)]

//! Public API roundtrip tests for Pachner/bistellar flips.

use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayTriangulation, InsertionOrderStrategy, TopologyGuarantee, vertex,
};
use delaunay::prelude::flips::{
    BistellarFlips, EdgeKey, FacetHandle, RidgeHandle, SimplexKey, TriangleHandle, VertexKey,
};
use delaunay::prelude::geometry::RobustKernel;
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

fn build_stable_dt_4d() -> Dt4 {
    let vertices = STABLE_POINTS_4D
        .iter()
        .map(|coords| vertex!(*coords))
        .collect::<Vec<_>>();
    let options =
        ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);

    DelaunayTriangulation::with_topology_guarantee_and_options(
        &RobustKernel::new(),
        &vertices,
        TopologyGuarantee::PLManifold,
        options,
    )
    .expect("stable 4D fixture should build")
}

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

fn first_simplex(dt: &Dt4) -> SimplexKey {
    dt.simplices()
        .next()
        .map(|(simplex_key, _)| simplex_key)
        .expect("stable fixture should contain simplices")
}

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

fn roundtrip_k1(dt: &mut Dt4) {
    let simplex_key = first_simplex(dt);
    let new_vertex = vertex!(simplex_centroid(dt, simplex_key));
    let new_uuid = new_vertex.uuid();
    dt.flip_k1_insert(simplex_key, new_vertex)
        .expect("k=1 insert should succeed on stable 4D fixture");

    let inserted_key = dt
        .tds()
        .vertex_key_from_uuid(&new_uuid)
        .expect("inserted k=1 vertex should be present");
    dt.flip_k1_remove(inserted_key)
        .expect("k=1 remove should invert insert");
}

fn interior_facets(dt: &Dt4) -> Vec<FacetHandle> {
    let mut facets = Vec::new();
    for (simplex_key, simplex) in dt.simplices() {
        let Some(neighbors) = simplex.neighbors() else {
            continue;
        };
        for (facet_index, neighbor) in neighbors.enumerate() {
            if neighbor.is_some() {
                facets.push(FacetHandle::new(
                    simplex_key,
                    u8::try_from(facet_index).expect("facet index should fit in u8"),
                ));
            }
        }
    }
    facets
}

fn flippable_k2_facet(dt: &Dt4) -> FacetHandle {
    for facet in interior_facets(dt) {
        let mut trial = dt.clone();
        if let Ok(info) = trial.flip_k2(facet) {
            let edge = inserted_edge(&info.inserted_face_vertices);
            if trial.flip_k2_inverse_from_edge(edge).is_ok() && trial.validate().is_ok() {
                return facet;
            }
        }
    }
    panic!("stable 4D fixture should contain a public k=2 roundtrip candidate");
}

fn roundtrip_k2(dt: &mut Dt4, facet: FacetHandle) {
    let info = dt
        .flip_k2(facet)
        .expect("k=2 flip should succeed on selected stable 4D facet");
    dt.flip_k2_inverse_from_edge(inserted_edge(&info.inserted_face_vertices))
        .expect("k=2 inverse should succeed after k=2 flip");
}

fn inserted_edge(vertices: &[VertexKey]) -> EdgeKey {
    match vertices {
        [a, b] => EdgeKey::new(*a, *b),
        _ => panic!("k=2 flip should report an inserted edge"),
    }
}

fn ridges(dt: &Dt4) -> Vec<RidgeHandle> {
    let mut ridges = Vec::new();
    for (simplex_key, simplex) in dt.simplices() {
        for i in 0..simplex.number_of_vertices() {
            for j in (i + 1)..simplex.number_of_vertices() {
                ridges.push(RidgeHandle::new(
                    simplex_key,
                    u8::try_from(i).expect("ridge index should fit in u8"),
                    u8::try_from(j).expect("ridge index should fit in u8"),
                ));
            }
        }
    }
    ridges
}

fn flippable_k3_ridge(dt: &Dt4) -> RidgeHandle {
    for ridge in ridges(dt) {
        let mut trial = dt.clone();
        if let Ok(info) = trial.flip_k3(ridge) {
            let triangle = inserted_triangle(&info.inserted_face_vertices);
            if trial.flip_k3_inverse_from_triangle(triangle).is_ok() && trial.validate().is_ok() {
                return ridge;
            }
        }
    }
    panic!("stable 4D fixture should contain a public k=3 roundtrip candidate");
}

fn roundtrip_k3(dt: &mut Dt4, ridge: RidgeHandle) {
    let info = dt
        .flip_k3(ridge)
        .expect("k=3 flip should succeed on selected stable 4D ridge");
    dt.flip_k3_inverse_from_triangle(inserted_triangle(&info.inserted_face_vertices))
        .expect("k=3 inverse should succeed after k=3 flip");
}

fn inserted_triangle(vertices: &[VertexKey]) -> TriangleHandle {
    match vertices {
        [a, b, c] => TriangleHandle::new(*a, *b, *c),
        _ => panic!("k=3 flip should report an inserted triangle"),
    }
}
