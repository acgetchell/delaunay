//! Integration tests for the public topology traversal and adjacency APIs.
//!
//! These tests cover:
//! - Global edge enumeration via [`DelaunayTriangulation::edges`]
//! - Vertex incident edges via [`DelaunayTriangulation::incident_edges`]
//! - Simplex neighborhood traversal via [`DelaunayTriangulation::simplex_neighbors`]
//! - Building and validating the opt-in [`AdjacencyIndex`]

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
    AdjacencyIndex(#[from] AdjacencyIndexBuildError),
    #[error(transparent)]
    Query(#[from] QueryError),
    #[error("single tetrahedron triangulation has no vertices")]
    EmptySingleTetrahedronVertices,
    #[error("single tetrahedron triangulation has no simplices")]
    EmptySingleTetrahedronSimplices,
    #[error("simplex key from triangulation has no vertices")]
    MissingSimplexVertices,
    #[error("double tetrahedron did not contain the expected shared vertex")]
    MissingExpectedSharedVertex,
}

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

    let index = tri.build_adjacency_index()?;
    let edges_with_index: HashSet<_> = dt.edges_with_index(&index)?.collect();
    assert_eq!(edges_with_index, edges);

    // Pick an arbitrary vertex; in a tetrahedron its degree is 3.
    let v0 = dt
        .vertices()
        .next()
        .map(|(vertex_key, _)| vertex_key)
        .ok_or(PublicTopologyApiTestError::EmptySingleTetrahedronVertices)?;
    assert_eq!(dt.incident_edges(v0).count(), 3);
    assert_eq!(dt.incident_edges_with_index(&index, v0)?.count(), 3);

    let incident: HashSet<_> = dt.incident_edges(v0).collect();
    assert_eq!(incident.len(), 3);

    // A single tetrahedron has no simplex neighbors.
    let simplex_key = dt
        .simplices()
        .next()
        .map(|(simplex_key, _)| simplex_key)
        .ok_or(PublicTopologyApiTestError::EmptySingleTetrahedronSimplices)?;
    assert_eq!(dt.simplex_neighbors(simplex_key).count(), 0);
    assert_eq!(
        dt.simplex_neighbors_with_index(&index, simplex_key)?
            .count(),
        0
    );

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
fn adjacency_index_on_double_tetrahedron() -> Result<(), PublicTopologyApiTestError> {
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

    // Build opt-in adjacency index and validate key properties.
    let index = tri.build_adjacency_index()?;

    // Triangulation-level with_index helpers should match the index and the baseline APIs.
    assert_eq!(
        tri.adjacent_simplices_with_index(&index, shared_vertex_key)?
            .count(),
        2
    );
    assert_eq!(
        tri.number_of_adjacent_simplices_with_index(&index, shared_vertex_key)?,
        2
    );

    for &ck in &simplex_keys {
        assert_eq!(dt.simplex_neighbors_with_index(&index, ck)?.count(), 1);
    }

    // Shared vertex should have 2 incident simplices.
    assert_eq!(index.number_of_adjacent_simplices(shared_vertex_key), 2);
    assert_eq!(index.adjacent_simplices(shared_vertex_key).count(), 2);

    // Shared vertex should have at least 3 incident edges (degree depends on geometry);
    // ensure the list is non-empty and contains canonical edges.
    let incident_edges: Vec<_> = index.incident_edges(shared_vertex_key).collect();
    assert!(!incident_edges.is_empty());
    assert!(incident_edges.iter().all(|e| e.v0() <= e.v1()));

    // Triangulation-level with_index helper should match index-based incident edges.
    let incident_edges_with_index: HashSet<_> = dt
        .incident_edges_with_index(&index, shared_vertex_key)?
        .collect();
    assert_eq!(incident_edges_with_index.len(), incident_edges.len());

    // Each simplex should appear with exactly one neighbor in the index.
    for &ck in &simplex_keys {
        assert_eq!(index.number_of_simplex_neighbors(ck), 1);
        assert_eq!(index.simplex_neighbors(ck).count(), 1);
    }

    // Global edge iterator should yield each edge exactly once.
    let edges: HashSet<_> = index.edges().collect();
    assert_eq!(edges.len(), tri.number_of_edges());

    // Triangulation-level edges_with_index should match the index edges().
    let edges_via_tri: HashSet<_> = tri.edges_with_index(&index)?.collect();
    assert_eq!(edges_via_tri, edges);

    // Missing keys should yield empty iterators.
    assert_eq!(index.adjacent_simplices(VertexKey::default()).count(), 0);
    assert_eq!(index.incident_edges(VertexKey::default()).count(), 0);
    assert_eq!(index.simplex_neighbors(SimplexKey::default()).count(), 0);

    assert_eq!(
        tri.adjacent_simplices_with_index(&index, VertexKey::default())?
            .count(),
        0
    );
    assert_eq!(
        dt.incident_edges_with_index(&index, VertexKey::default())?
            .count(),
        0
    );
    assert_eq!(
        dt.simplex_neighbors_with_index(&index, SimplexKey::default())?
            .count(),
        0
    );
    Ok(())
}
