//! Integration tests for the public topology traversal and adjacency APIs.
//!
//! These tests cover:
//! - Global edge enumeration via [`DelaunayTriangulation::edges`]
//! - Vertex incident edges via [`DelaunayTriangulation::incident_edges`]
//! - Cell neighborhood traversal via [`DelaunayTriangulation::cell_neighbors`]
//! - Building and validating the opt-in [`AdjacencyIndex`]

use delaunay::prelude::query::*;

#[test]
fn edges_and_incident_edges_on_single_tetrahedron() {
    // Single tetrahedron: 4 vertices, 1 cell, 6 unique edges.
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    let tri = dt.triangulation();

    assert_eq!(dt.number_of_vertices(), 4);
    assert_eq!(dt.number_of_cells(), 1);

    let edge_count = dt.edges().count();
    assert_eq!(edge_count, 6);

    let edges: std::collections::HashSet<_> = dt.edges().collect();
    assert_eq!(edges.len(), 6);

    let index = tri.build_adjacency_index().unwrap();
    let edges_with_index: std::collections::HashSet<_> = dt.edges_with_index(&index).collect();
    assert_eq!(edges_with_index, edges);

    // Pick an arbitrary vertex; in a tetrahedron its degree is 3.
    let v0 = dt.vertices().next().unwrap().0;
    assert_eq!(dt.incident_edges(v0).count(), 3);
    assert_eq!(dt.incident_edges_with_index(&index, v0).count(), 3);

    let incident: std::collections::HashSet<_> = dt.incident_edges(v0).collect();
    assert_eq!(incident.len(), 3);

    // A single tetrahedron has no cell neighbors.
    let cell_key = dt.cells().next().unwrap().0;
    assert_eq!(dt.cell_neighbors(cell_key).count(), 0);
    assert_eq!(dt.cell_neighbors_with_index(&index, cell_key).count(), 0);

    // Geometry accessors are zero-allocation and should succeed for keys from this triangulation.
    // They are also forwarded on `DelaunayTriangulation`.
    assert_eq!(dt.vertex_coords(v0), tri.vertex_coords(v0));
    assert!(dt.vertex_coords(v0).is_some());

    assert_eq!(dt.cell_vertices(cell_key), tri.cell_vertices(cell_key));
    assert_eq!(dt.cell_vertices(cell_key).unwrap().len(), 4);
}

#[test]
fn adjacency_index_on_double_tetrahedron() {
    // Two tetrahedra sharing a triangular facet.
    let vertices: Vec<_> = vec![
        // Shared triangle
        vertex!([0.0, 0.0, 0.0]),
        vertex!([2.0, 0.0, 0.0]),
        vertex!([1.0, 2.0, 0.0]),
        // Two apices
        vertex!([1.0, 0.7, 1.5]),
        vertex!([1.0, 0.7, -1.5]),
    ];

    let dt: DelaunayTriangulation<_, (), (), 3> = DelaunayTriangulation::new(&vertices).unwrap();
    let tri = dt.triangulation();

    assert_eq!(tri.number_of_vertices(), 5);
    assert_eq!(tri.number_of_cells(), 2);

    // Find a vertex on the shared triangle by coordinates.
    let shared_vertex_key = dt
        .vertices()
        .find_map(|(vk, _)| {
            let coords = dt.vertex_coords(vk)?;
            (coords == [0.0, 0.0, 0.0]).then_some(vk)
        })
        .unwrap();

    // The shared vertex should be incident to both cells.
    assert_eq!(tri.adjacent_cells(shared_vertex_key).count(), 2);

    // Each cell should have exactly one neighbor across the shared facet.
    let cell_keys: Vec<_> = tri.cells().map(|(ck, _)| ck).collect();
    assert_eq!(cell_keys.len(), 2);

    for &ck in &cell_keys {
        let neighbors: Vec<_> = dt.cell_neighbors(ck).collect();
        assert_eq!(neighbors.len(), 1);
        assert!(cell_keys.contains(&neighbors[0]));
        assert_ne!(neighbors[0], ck);
    }

    // Build opt-in adjacency index and validate key properties.
    let index = tri.build_adjacency_index().unwrap();

    // Triangulation-level with_index helpers should match the index and the baseline APIs.
    assert_eq!(
        tri.adjacent_cells_with_index(&index, shared_vertex_key)
            .count(),
        2
    );
    assert_eq!(
        tri.number_of_adjacent_cells_with_index(&index, shared_vertex_key),
        2
    );

    for &ck in &cell_keys {
        assert_eq!(dt.cell_neighbors_with_index(&index, ck).count(), 1);
    }

    // Shared vertex should have 2 incident cells.
    assert_eq!(index.number_of_adjacent_cells(shared_vertex_key), 2);
    assert_eq!(index.adjacent_cells(shared_vertex_key).count(), 2);

    // Shared vertex should have at least 3 incident edges (degree depends on geometry);
    // ensure the list is non-empty and contains canonical edges.
    let incident_edges: Vec<_> = index.incident_edges(shared_vertex_key).collect();
    assert!(!incident_edges.is_empty());
    assert!(incident_edges.iter().all(|e| e.v0() <= e.v1()));

    // Triangulation-level with_index helper should match index-based incident edges.
    let incident_edges_with_index: std::collections::HashSet<_> = dt
        .incident_edges_with_index(&index, shared_vertex_key)
        .collect();
    assert_eq!(incident_edges_with_index.len(), incident_edges.len());

    // Each cell should appear with exactly one neighbor in the index.
    for &ck in &cell_keys {
        assert_eq!(index.number_of_cell_neighbors(ck), 1);
        assert_eq!(index.cell_neighbors(ck).count(), 1);
    }

    // Global edge iterator should yield each edge exactly once.
    let edges: std::collections::HashSet<_> = index.edges().collect();
    assert_eq!(edges.len(), tri.number_of_edges());

    // Triangulation-level edges_with_index should match the index edges().
    let edges_via_tri: std::collections::HashSet<_> = tri.edges_with_index(&index).collect();
    assert_eq!(edges_via_tri, edges);

    // Missing keys should yield empty iterators.
    assert_eq!(index.adjacent_cells(VertexKey::default()).count(), 0);
    assert_eq!(index.incident_edges(VertexKey::default()).count(), 0);
    assert_eq!(index.cell_neighbors(CellKey::default()).count(), 0);

    assert_eq!(
        tri.adjacent_cells_with_index(&index, VertexKey::default())
            .count(),
        0
    );
    assert_eq!(
        dt.incident_edges_with_index(&index, VertexKey::default())
            .count(),
        0
    );
    assert_eq!(
        dt.cell_neighbors_with_index(&index, CellKey::default())
            .count(),
        0
    );
}
