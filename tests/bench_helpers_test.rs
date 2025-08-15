//! Integration tests for benchmark helper functions

#[path = "../benches/helpers.rs"]
mod helpers;

use delaunay::prelude::*;
use helpers::clear_all_neighbors;

#[test]
fn test_clear_all_neighbors() {
    // Create a triangulation with more vertices to ensure multiple cells with neighbors
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
        vertex!([0.5, 0.5, 0.5]), // Additional vertex to create multiple tetrahedra
    ];

    let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    // Verify triangulation was created with neighbors
    tds.assign_neighbors()
        .expect("assign_neighbors() failed; Tds is invalid");

    // Should have some neighbors initially
    let has_neighbors = tds
        .cells()
        .values()
        .any(|cell| cell.neighbors.as_ref().is_some());
    assert!(has_neighbors, "Initial triangulation should have neighbors");

    // Clear all neighbors using our helper function
    clear_all_neighbors(&mut tds);

    // Verify all neighbors were cleared
    let final_neighbor_count: usize = tds
        .cells()
        .values()
        .map(|cell| cell.neighbors.as_ref().map_or(0, std::vec::Vec::len))
        .sum();

    assert_eq!(
        final_neighbor_count, 0,
        "All neighbors should be cleared after calling clear_all_neighbors"
    );
}

#[test]
fn test_clear_all_neighbors_empty_triangulation() {
    // Test with a triangulation that has no cells
    let vertices: Vec<Vertex<f64, (), 3>> = vec![];

    // This should create an empty triangulation
    if let Ok(mut tds) = Tds::<f64, (), (), 3>::new(&vertices) {
        // Should not panic on empty triangulation
        clear_all_neighbors(&mut tds);
        assert_eq!(tds.number_of_cells(), 0);
    } else {
        // It's expected that creating a triangulation with no vertices might fail
        // This is acceptable behavior
    }
}
