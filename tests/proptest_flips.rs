//! Property-based tests for public bistellar flip invariants.
//!
//! Lower-level k=2/k=3 property fixtures live with `core::algorithms::flips`,
//! where the TDS construction helpers are available. This integration test keeps
//! the public editing API under property pressure and verifies that k=1
//! subdivision round-trips preserve topology and Euler characteristic across
//! dimensions 2D-5D.

use ::uuid::Uuid;
use delaunay::prelude::geometry::{AdaptiveKernel, Coordinate, Point};
use delaunay::prelude::triangulation::flips::BistellarFlips;
use delaunay::prelude::triangulation::{
    DelaunayTriangulation, TopologyGuarantee, Triangulation, Vertex,
};
use proptest::prelude::*;
use std::collections::{BTreeSet, HashMap};

#[derive(Debug, PartialEq, Eq)]
struct TopologySnapshot {
    vertices: Vec<Uuid>,
    cell_vertices: Vec<Vec<Uuid>>,
}

/// Generates bounded finite coordinates for stable simplex placement.
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-10.0..10.0).prop_filter("must be finite", |value: &f64| value.is_finite())
}

/// Generates positive edge lengths that keep test simplices well-conditioned.
fn well_conditioned_edge_length() -> impl Strategy<Value = f64> {
    (0.25_f64..10.0).prop_filter("must be finite and positive", |value: &f64| {
        value.is_finite() && *value > 0.0
    })
}

/// Builds an axis-aligned D-simplex translated by `origin`.
fn axis_aligned_simplex_vertices<const D: usize>(
    origin: [f64; D],
    edge_lengths: [f64; D],
) -> Vec<Vertex<f64, (), D>> {
    let mut points = Vec::with_capacity(D + 1);
    points.push(Point::new(origin));

    for (axis, edge_length) in edge_lengths.iter().copied().enumerate() {
        let mut coordinates = origin;
        coordinates[axis] += edge_length;
        points.push(Point::new(coordinates));
    }

    Vertex::from_points(&points)
}

/// Places a vertex strictly inside the generated axis-aligned simplex.
fn interior_simplex_vertex<const D: usize>(
    origin: [f64; D],
    edge_lengths: [f64; D],
) -> Vertex<f64, (), D> {
    let denominator = f64::from(u32::try_from(D + 1).expect("test dimension fits in u32"));
    let mut coordinates = origin;
    for (axis, edge_length) in edge_lengths.iter().copied().enumerate() {
        coordinates[axis] += edge_length / denominator;
    }
    Vertex::from_point(Point::new(coordinates))
}

/// Captures the public vertex/cell incidence needed to check round-trips.
fn snapshot_topology<const D: usize>(
    triangulation: &Triangulation<AdaptiveKernel<f64>, (), (), D>,
) -> Result<TopologySnapshot, TestCaseError> {
    let key_to_uuid: HashMap<_, _> = triangulation
        .vertices()
        .map(|(key, vertex)| (key, vertex.uuid()))
        .collect();
    let mut vertices: Vec<Uuid> = key_to_uuid.values().copied().collect();
    vertices.sort();

    let mut cell_vertices = Vec::new();
    for (_, cell) in triangulation.cells() {
        let mut uuids = Vec::with_capacity(D + 1);
        for vertex_key in cell.vertices() {
            let uuid = key_to_uuid.get(vertex_key).copied().ok_or_else(|| {
                TestCaseError::fail(format!(
                    "cell references missing vertex key: {vertex_key:?}"
                ))
            })?;
            uuids.push(uuid);
        }
        uuids.sort();
        cell_vertices.push(uuids);
    }
    cell_vertices.sort();

    Ok(TopologySnapshot {
        vertices,
        cell_vertices,
    })
}

/// Recursively records all UUID-simplex combinations of a fixed size.
fn record_uuid_combinations(
    vertices: &[Uuid],
    simplex_size: usize,
    start: usize,
    current: &mut Vec<Uuid>,
    seen: &mut BTreeSet<Vec<Uuid>>,
) {
    if current.len() == simplex_size {
        let mut simplex = current.clone();
        simplex.sort();
        seen.insert(simplex);
        return;
    }

    for index in start..vertices.len() {
        current.push(vertices[index]);
        record_uuid_combinations(vertices, simplex_size, index + 1, current, seen);
        current.pop();
    }
}

/// Computes Euler characteristic from public cell/vertex iterators.
fn euler_characteristic<const D: usize>(
    triangulation: &Triangulation<AdaptiveKernel<f64>, (), (), D>,
) -> Result<isize, TestCaseError> {
    let cell_vertices = snapshot_topology(triangulation)?.cell_vertices;
    let mut simplices_by_dim: Vec<BTreeSet<Vec<Uuid>>> = (0..=D).map(|_| BTreeSet::new()).collect();

    for vertices in &cell_vertices {
        for (simplex_dimension, simplices) in simplices_by_dim.iter_mut().enumerate().take(D + 1) {
            let mut current = Vec::with_capacity(simplex_dimension + 1);
            record_uuid_combinations(vertices, simplex_dimension + 1, 0, &mut current, simplices);
        }
    }

    let mut chi = 0_isize;
    for (simplex_dimension, simplices) in simplices_by_dim.iter().enumerate() {
        let count = isize::try_from(simplices.len())
            .map_err(|err| TestCaseError::fail(format!("simplex count overflowed: {err:?}")))?;
        if simplex_dimension % 2 == 0 {
            chi += count;
        } else {
            chi -= count;
        }
    }

    Ok(chi)
}

/// Checks both public topology validation entry points after each edit.
fn assert_valid<const D: usize>(
    triangulation: &Triangulation<AdaptiveKernel<f64>, (), (), D>,
    context: &str,
) -> Result<(), TestCaseError> {
    triangulation
        .is_valid()
        .map_err(|err| TestCaseError::fail(format!("{context} invariant check failed: {err:?}")))?;
    triangulation
        .validate()
        .map_err(|err| TestCaseError::fail(format!("{context} validation failed: {err:?}")))?;
    Ok(())
}

/// Verifies a public k=1 flip followed by its inverse is topology-preserving.
fn check_k1_roundtrip<const D: usize>(
    origin: [f64; D],
    edge_lengths: [f64; D],
) -> Result<(), TestCaseError> {
    let vertices = axis_aligned_simplex_vertices::<D>(origin, edge_lengths);
    let simplex =
        DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), D>::new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )
        .map_err(|err| {
            TestCaseError::fail(format!(
                "{D}D axis-aligned simplex construction failed: {err:?}"
            ))
        })?;

    let mut triangulation = simplex.as_triangulation().clone();
    assert_valid(&triangulation, "initial")?;
    let before = snapshot_topology(&triangulation)?;
    let before_chi = euler_characteristic(&triangulation)?;

    let cell_key = triangulation
        .cells()
        .next()
        .map(|(key, _)| key)
        .ok_or_else(|| TestCaseError::fail("constructed simplex should contain one cell"))?;
    let inserted = triangulation
        .flip_k1_insert(cell_key, interior_simplex_vertex::<D>(origin, edge_lengths))
        .map_err(|err| TestCaseError::fail(format!("k=1 insertion failed: {err:?}")))?;
    prop_assert!(!inserted.new_cells.is_empty());
    prop_assert_eq!(euler_characteristic(&triangulation)?, before_chi);
    assert_valid(&triangulation, "after k=1 insertion")?;

    let inserted_vertex = inserted
        .inserted_face_vertices
        .first()
        .copied()
        .ok_or_else(|| TestCaseError::fail("k=1 insertion did not report an inserted vertex"))?;
    let removed = triangulation
        .flip_k1_remove(inserted_vertex)
        .map_err(|err| TestCaseError::fail(format!("inverse k=1 removal failed: {err:?}")))?;
    prop_assert!(!removed.removed_cells.is_empty());
    prop_assert_eq!(euler_characteristic(&triangulation)?, before_chi);
    assert_valid(&triangulation, "after inverse k=1 removal")?;

    let after = snapshot_topology(&triangulation)?;
    prop_assert_eq!(after, before);
    Ok(())
}

macro_rules! gen_k1_roundtrip_properties {
    ($($dim:literal),* $(,)?) => {
        pastey::paste! {
            $(
                proptest! {
                    #![proptest_config(ProptestConfig::with_cases(32))]

                    #[test]
                    fn [<prop_k1_flip_roundtrip_preserves_topology_and_euler_ $dim d>](
                        origin in prop::array::[<uniform $dim>](finite_coordinate()),
                        edge_lengths in prop::array::[<uniform $dim>](well_conditioned_edge_length())
                    ) {
                        check_k1_roundtrip::<$dim>(origin, edge_lengths)?;
                    }
                }
            )*
        }
    };
}

gen_k1_roundtrip_properties!(2, 3, 4, 5);
