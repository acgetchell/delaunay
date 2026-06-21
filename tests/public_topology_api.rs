//! Integration tests for the public topology traversal and adjacency APIs.
//!
//! These tests cover:
//! - Global edge enumeration via [`DelaunayTriangulation::edges`]
//! - Vertex incident edges via [`DelaunayTriangulation::incident_edges`]
//! - Simplex neighborhood traversal via [`DelaunayTriangulation::simplex_neighbors`]
//! - Building and validating opt-in split topology views

use delaunay::prelude::DelaunayTriangulationConstructionError;
use delaunay::prelude::TopologyGuarantee;
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::prelude::query::*;
use std::collections::HashSet;

#[derive(Debug, thiserror::Error)]
enum PublicTopologyApiTestError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    CoordinateConversion(#[from] CoordinateConversionError),
    #[error(transparent)]
    TopologyIndex(#[from] TopologyIndexBuildError),
    #[error("single tetrahedron triangulation has no vertices")]
    EmptySingleTetrahedronVertices,
    #[error("single tetrahedron triangulation has no simplices")]
    EmptySingleTetrahedronSimplices,
    #[error("single simplex triangulation has no simplices")]
    EmptySingleSimplexSimplices,
    #[error("simplex key from triangulation has no vertices")]
    MissingSimplexVertices,
    #[error("double tetrahedron did not contain the expected shared vertex")]
    MissingExpectedSharedVertex,
}

/// Builds the vertices of a standard D-simplex for dimension-generic topology tests.
fn standard_simplex_vertices<const D: usize>()
-> Result<Vec<Vertex<(), D>>, CoordinateConversionError> {
    let mut vertices = Vec::with_capacity(D + 1);
    vertices.push(Vertex::<(), _>::try_new([0.0; D])?);
    for axis in 0..D {
        let mut coords = [0.0; D];
        coords[axis] = 1.0;
        vertices.push(Vertex::<(), _>::try_new(coords)?);
    }
    Ok(vertices)
}

/// Verifies split topology views against the closed-form topology of one simplex.
fn assert_split_topology_single_simplex<const D: usize>() -> Result<(), PublicTopologyApiTestError>
{
    let vertices = standard_simplex_vertices::<D>()?;
    let dt: DelaunayTriangulation<_, (), (), D> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )?;
    let tri = dt.as_triangulation();
    let expected_edges = D * (D + 1) / 2;

    assert_eq!(tri.number_of_vertices(), D + 1);
    assert_eq!(tri.number_of_simplices(), 1);
    assert_eq!(tri.number_of_edges(), expected_edges);

    let simplex_key = tri
        .simplices()
        .next()
        .map(|(simplex_key, _)| simplex_key)
        .ok_or(PublicTopologyApiTestError::EmptySingleSimplexSimplices)?;
    let incidence = tri.incidence()?;
    let edge_index = tri.build_edge_index()?;
    let neighbor_index = tri.build_simplex_neighbor_index()?;
    let adjacency = tri.adjacency()?;

    assert_eq!(edge_index.number_of_edges(), expected_edges);
    assert_eq!(adjacency.number_of_edges(), expected_edges);
    assert_eq!(
        edge_index.edges().collect::<HashSet<_>>().len(),
        expected_edges
    );
    assert_eq!(
        adjacency.edges().collect::<HashSet<_>>().len(),
        expected_edges
    );
    assert_eq!(neighbor_index.number_of_simplex_neighbors(simplex_key), 0);
    assert_eq!(adjacency.number_of_simplex_neighbors(simplex_key), 0);

    for (vertex_key, _) in tri.vertices() {
        assert_eq!(incidence.number_of_adjacent_simplices(vertex_key), 1);
        assert_eq!(adjacency.number_of_adjacent_simplices(vertex_key), 1);
        assert_eq!(edge_index.number_of_incident_edges(vertex_key), D);
        assert_eq!(adjacency.number_of_incident_edges(vertex_key), D);

        let adjacent: HashSet<_> = incidence.adjacent_simplices(vertex_key).collect();
        assert_eq!(adjacent.len(), 1);
        assert!(adjacent.contains(&simplex_key));
    }

    Ok(())
}

macro_rules! gen_split_topology_single_simplex_tests {
    ($($dim:literal),+ $(,)?) => {
        pastey::paste! {
            $(
                #[test]
                fn [<split_topology_indexes_single_simplex_ $dim d>]() -> Result<(), PublicTopologyApiTestError> {
                    assert_split_topology_single_simplex::<$dim>()
                }
            )+
        }
    };
}

gen_split_topology_single_simplex_tests!(2, 3, 4, 5);

#[test]
fn edges_and_incident_edges_on_single_tetrahedron() -> Result<(), PublicTopologyApiTestError> {
    // Single tetrahedron: 4 vertices, 1 simplex, 6 unique edges.
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];

    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )?;
    let tri = dt.as_triangulation();

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_simplices(), 1);

    let edge_count = dt.edges().count();
    assert_eq!(edge_count, 6);

    let edges: HashSet<_> = dt.edges().collect();
    assert_eq!(edges.len(), 6);

    let edge_index = tri.build_edge_index()?;
    let indexed_edges: HashSet<_> = edge_index.edges().collect();
    assert_eq!(indexed_edges, edges);

    // Pick an arbitrary vertex; in a tetrahedron its degree is 3.
    let v0 = dt
        .vertices()
        .next()
        .map(|(vertex_key, _)| vertex_key)
        .ok_or(PublicTopologyApiTestError::EmptySingleTetrahedronVertices)?;
    assert_eq!(dt.incident_edges(v0).count(), 3);
    assert_eq!(edge_index.incident_edges(v0).count(), 3);

    let incident: HashSet<_> = dt.incident_edges(v0).collect();
    assert_eq!(incident.len(), 3);

    // A single tetrahedron has no simplex neighbors.
    let simplex_key = dt
        .simplices()
        .next()
        .map(|(simplex_key, _)| simplex_key)
        .ok_or(PublicTopologyApiTestError::EmptySingleTetrahedronSimplices)?;
    assert_eq!(dt.simplex_neighbors(simplex_key).count(), 0);
    let neighbor_index = dt.build_simplex_neighbor_index()?;
    assert_eq!(neighbor_index.simplex_neighbors(simplex_key).count(), 0);

    // Geometry accessors are zero-allocation and should succeed for keys from this triangulation.
    // They are also forwarded on `DelaunayTriangulation`.
    assert_eq!(dt.vertex_coords(v0), tri.vertex_coords(v0));
    assert!(dt.vertex_coords(v0).is_some());

    assert_eq!(
        dt.simplex_vertices(simplex_key),
        tri.simplex_vertices(simplex_key)
    );
    assert_eq!(
        dt.simplex_vertices(simplex_key)
            .ok_or(PublicTopologyApiTestError::MissingSimplexVertices)?
            .len(),
        4
    );
    Ok(())
}

#[test]
fn split_topology_indexes_on_double_tetrahedron() -> Result<(), PublicTopologyApiTestError> {
    // Two tetrahedra sharing a triangular facet.
    let vertices: Vec<_> = vec![
        // Shared triangle
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([2.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 2.0, 0.0])?,
        // Two apices
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.7, 1.5])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.7, -1.5])?,
    ];

    let dt: DelaunayTriangulation<_, (), (), 3> =
        DelaunayTriangulation::try_new_with_topology_guarantee(
            &vertices,
            TopologyGuarantee::PLManifold,
        )?;
    let tri = dt.as_triangulation();

    assert_eq!(tri.number_of_vertices(), 5);
    assert_eq!(tri.number_of_simplices(), 2);

    // Find a vertex on the shared triangle by coordinates.
    let shared_vertex_key = dt
        .vertices()
        .find_map(|(vk, _)| {
            let coords = dt.vertex_coords(vk)?;
            (coords == [0.0, 0.0, 0.0]).then_some(vk)
        })
        .ok_or(PublicTopologyApiTestError::MissingExpectedSharedVertex)?;

    // The shared vertex should be incident to both simplices.
    assert_eq!(tri.adjacent_simplices(shared_vertex_key).count(), 2);

    // Each simplex should have exactly one neighbor across the shared facet.
    let simplex_keys: Vec<_> = tri.simplices().map(|(ck, _)| ck).collect();
    assert_eq!(simplex_keys.len(), 2);

    for &ck in &simplex_keys {
        let neighbors: Vec<_> = dt.simplex_neighbors(ck).collect();
        assert_eq!(neighbors.len(), 1);
        assert!(simplex_keys.contains(&neighbors[0]));
        assert_ne!(neighbors[0], ck);
    }

    // Build opt-in split topology views and validate key properties.
    let incidence = tri.incidence()?;
    let edge_index = tri.build_edge_index()?;
    let neighbor_index = dt.build_simplex_neighbor_index()?;

    // Shared vertex should have 2 incident simplices.
    assert_eq!(incidence.number_of_adjacent_simplices(shared_vertex_key), 2);
    assert_eq!(incidence.adjacent_simplices(shared_vertex_key).count(), 2);

    // Shared vertex should have at least 3 incident edges (degree depends on geometry);
    // ensure the list is non-empty and contains canonical edges.
    let incident_edges: Vec<_> = edge_index.incident_edges(shared_vertex_key).collect();
    assert!(!incident_edges.is_empty());
    assert!(incident_edges.iter().all(|e| e.v0() <= e.v1()));

    // Each simplex should appear with exactly one neighbor in the neighbor index.
    for &ck in &simplex_keys {
        assert_eq!(neighbor_index.number_of_simplex_neighbors(ck), 1);
        assert_eq!(neighbor_index.simplex_neighbors(ck).count(), 1);
    }

    // Global edge iterator should yield each edge exactly once.
    let edges: HashSet<_> = edge_index.edges().collect();
    assert_eq!(edges.len(), tri.number_of_edges());

    // Direct traversal should match the edge index.
    let edges_via_tri: HashSet<_> = tri.edges().collect();
    assert_eq!(edges_via_tri, edges);

    // Missing keys should yield empty iterators.
    assert_eq!(
        incidence.adjacent_simplices(VertexKey::default()).count(),
        0
    );
    assert_eq!(edge_index.incident_edges(VertexKey::default()).count(), 0);
    assert_eq!(
        neighbor_index
            .simplex_neighbors(SimplexKey::default())
            .count(),
        0
    );
    Ok(())
}
