#![expect(deprecated)]
//! Edge case integration tests for Triangulation Data Structure (TDS)
//!
//! This module tests TDS operations that are not covered by unit tests or basic integration tests:
//! - **Removal operations**: Testing `remove_cell_by_key`, `remove_cells_by_keys`, `remove_duplicate_cells`
//! - **Topology consistency after removal**: Ensuring neighbor/incident cell relationships remain valid
//! - **Stress tests**: Large triangulations (marked `#[ignore]` for CI performance)
//!
//! These tests complement:
//! - `tds_basic_integration.rs` - Basic TDS construction and neighbor assignment
//! - Unit tests in source - Validation, neighbor symmetry, error handling

use delaunay::core::triangulation_data_structure::{CellKey, Tds};
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use delaunay::vertex;

// ============================================================================
// Cell Removal Tests
// ============================================================================

#[test]
fn test_remove_single_cell() {
    // Create a 2-cell TDS
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.5, 1.0, 0.0]),
        vertex!([0.5, 0.5, 1.0]),
        vertex!([0.5, 0.5, -1.0]),
    ];
    let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    assert_eq!(tds.number_of_cells(), 2);

    // Remove one cell
    let cell_key = tds.cell_keys().next().unwrap();
    let removed = tds.remove_cell_by_key(cell_key);

    assert!(removed.is_some(), "Should successfully remove the cell");
    assert_eq!(tds.number_of_cells(), 1, "Should have 1 cell remaining");
}

#[test]
fn test_remove_nonexistent_cell() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    // Try to remove a cell that doesn't exist (using default key)
    let nonexistent_key = CellKey::default();
    let removed = tds.remove_cell_by_key(nonexistent_key);

    assert!(removed.is_none(), "Should return None for nonexistent cell");
    assert_eq!(tds.number_of_cells(), 1, "Cell count should be unchanged");
}

#[test]
fn test_remove_multiple_cells() {
    // Create a larger TDS
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.5, 1.0, 0.0]),
        vertex!([0.5, 0.5, 1.0]),
        vertex!([0.5, 0.5, -1.0]),
        vertex!([1.5, 0.5, 0.0]),
    ];
    let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    let initial_count = tds.number_of_cells();
    assert!(initial_count >= 2, "Should have at least 2 cells");

    // Collect first 2 cell keys
    let cells_to_remove: Vec<_> = tds.cell_keys().take(2).collect();
    assert_eq!(cells_to_remove.len(), 2);

    // Remove them in batch
    let removed_count = tds.remove_cells_by_keys(&cells_to_remove);

    assert_eq!(removed_count, 2, "Should remove 2 cells");
    assert_eq!(
        tds.number_of_cells(),
        initial_count - 2,
        "Cell count should decrease by 2"
    );
}

#[test]
fn test_remove_cells_with_some_nonexistent() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.5, 1.0, 0.0]),
        vertex!([0.5, 0.5, 1.0]),
        vertex!([0.5, 0.5, -1.0]),
    ];
    let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    let existing_key = tds.cell_keys().next().unwrap();
    let nonexistent_key = CellKey::default();

    let cells_to_remove = vec![existing_key, nonexistent_key];
    let removed_count = tds.remove_cells_by_keys(&cells_to_remove);

    assert_eq!(
        removed_count, 1,
        "Should remove only the existing cell, skip nonexistent"
    );
}

// ============================================================================
// Duplicate Cell Removal Tests
// ============================================================================

#[test]
fn test_remove_duplicate_cells_no_duplicates() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    let removed = tds.remove_duplicate_cells().unwrap();

    assert_eq!(removed, 0, "Should find no duplicates in clean TDS");
    assert_eq!(tds.number_of_cells(), 1);
}

// ============================================================================
// Neighbor Clearing Tests
// ============================================================================

#[test]
fn test_clear_all_neighbors() {
    // Use 5 vertices to create 2 cells that will have neighbors
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.5, 1.0, 0.0]),
        vertex!([0.5, 0.5, 1.0]),
        vertex!([0.5, 0.5, -1.0]),
    ];
    let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    // Initially should have neighbors assigned
    let has_neighbors_before = tds.cells().any(|(_, cell)| cell.neighbors().is_some());
    assert!(
        has_neighbors_before,
        "TDS should have neighbors assigned after construction"
    );

    // Clear all neighbors
    tds.clear_all_neighbors();

    // All cells should now have no neighbors
    for (_key, cell) in tds.cells() {
        assert!(
            cell.neighbors().is_none(),
            "All cells should have no neighbors after clearing"
        );
    }
}

#[test]
fn test_reassign_neighbors_after_clearing() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.5, 1.0, 0.0]),
        vertex!([0.5, 0.5, 1.0]),
        vertex!([0.5, 0.5, -1.0]),
    ];
    let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    // Clear neighbors
    tds.clear_all_neighbors();

    // Reassign them
    tds.assign_neighbors().unwrap();

    // Should have neighbors again
    let has_neighbors_after = tds.cells().any(|(_, cell)| cell.neighbors().is_some());
    assert!(
        has_neighbors_after,
        "TDS should have neighbors after reassignment"
    );

    // Validation should pass
    assert!(
        tds.is_valid().is_ok(),
        "TDS should be valid after neighbor reassignment"
    );
}

// ============================================================================
// Public Method Coverage Tests
// ============================================================================

#[test]
fn test_set_neighbors_by_key() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.5, 1.0, 0.0]),
        vertex!([0.5, 0.5, 1.0]),
        vertex!([0.5, 0.5, -1.0]),
    ];
    let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    let cell_key = tds.cell_keys().next().unwrap();
    let new_neighbors = vec![None, None, None, None];

    let result = tds.set_neighbors_by_key(cell_key, &new_neighbors);
    assert!(result.is_ok(), "Should set neighbors successfully");

    // Verify neighbors were cleared
    let cell = tds.get_cell(cell_key).unwrap();
    assert!(
        cell.neighbors().is_none(),
        "Neighbors should be cleared when all None"
    );
}

#[test]
fn test_find_cells_containing_vertex_by_key() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.5, 1.0, 0.0]),
        vertex!([0.5, 0.5, 1.0]),
        vertex!([0.5, 0.5, -1.0]),
    ];
    let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    // Get first vertex key
    let vertex_key = tds.vertex_keys().next().unwrap();

    // Find cells containing this vertex
    let containing_cells = tds.find_cells_containing_vertex_by_key(vertex_key);

    assert!(
        !containing_cells.is_empty(),
        "Vertex should be in at least one cell"
    );

    // Verify each returned cell actually contains the vertex
    for cell_key in containing_cells {
        let cell = tds.get_cell(cell_key).unwrap();
        assert!(
            cell.vertices().contains(&vertex_key),
            "Returned cell should contain the vertex"
        );
    }
}

#[test]
fn test_build_facet_to_cells_map() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.0, 1.0, 0.0]),
        vertex!([0.0, 0.0, 1.0]),
    ];
    let tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    let facet_map = tds.build_facet_to_cells_map();
    assert!(facet_map.is_ok(), "Should build facet map successfully");

    let facet_map = facet_map.unwrap();
    assert!(!facet_map.is_empty(), "Facet map should not be empty");

    // Each facet in a single tetrahedron should be boundary (appear once)
    for (_facet_key, facet_handles) in facet_map {
        assert_eq!(facet_handles.len(), 1, "Boundary facets should appear once");
    }
}

#[test]
fn test_assign_incident_cells() {
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0]),
        vertex!([1.0, 0.0, 0.0]),
        vertex!([0.5, 1.0, 0.0]),
        vertex!([0.5, 0.5, 1.0]),
        vertex!([0.5, 0.5, -1.0]),
    ];
    let mut tds: Tds<f64, (), (), 3> = Tds::new(&vertices).unwrap();

    // Clear incident cells by modifying vertices through their keys
    let vertex_keys: Vec<_> = tds.vertex_keys().collect();
    for vkey in vertex_keys {
        if let Some(vertex) = tds.get_vertex_by_key_mut(vkey) {
            vertex.incident_cell = None;
        }
    }

    // Reassign them
    let result = tds.assign_incident_cells();
    assert!(result.is_ok(), "Should assign incident cells successfully");

    // Verify all vertices have incident cells
    for (_key, vertex) in tds.vertices() {
        assert!(
            vertex.incident_cell.is_some(),
            "All vertices should have incident cells after assignment"
        );
    }
}

#[test]
fn test_empty_tds() {
    let tds: Tds<f64, (), (), 3> = Tds::empty();
    assert_eq!(tds.number_of_vertices(), 0);
    assert_eq!(tds.number_of_cells(), 0);
    assert_eq!(tds.dim(), -1);
}

#[test]
fn test_add_vertex_to_empty_tds() {
    let mut tds: Tds<f64, (), (), 3> = Tds::empty();
    let result = tds.add(vertex!([1.0, 2.0, 3.0]));
    assert!(result.is_ok());
    assert_eq!(tds.number_of_vertices(), 1);
    assert_eq!(tds.dim(), 0);
}

#[test]
fn test_add_duplicate_vertex() {
    let mut tds: Tds<f64, (), (), 3> = Tds::empty();
    tds.add(vertex!([1.0, 2.0, 3.0])).unwrap();
    let result = tds.add(vertex!([1.0, 2.0, 3.0]));
    assert!(result.is_err(), "Should error on duplicate coordinates");
}

// ============================================================================
// Stress Tests (marked #[ignore] to keep CI fast)
// ============================================================================

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "Stress test - enable with --features slow-tests"
)]
fn test_stress_1000_vertices_2d() {
    use rand::Rng;
    let mut rng = rand::rng();

    // Generate points first, then convert to vertices
    let points: Vec<Point<f64, 2>> = (0..1000)
        .map(|_| {
            let x: f64 = rng.random();
            let y: f64 = rng.random();
            Point::new([x, y])
        })
        .collect();
    let vertices = Vertex::<f64, (), 2>::from_points(&points);

    let tds = Tds::<f64, (), (), 2>::new(&vertices);
    assert!(tds.is_ok(), "Should handle 1000 vertices in 2D");

    if let Ok(tds) = tds {
        assert!(tds.is_valid().is_ok(), "1000-vertex 2D TDS should be valid");
        println!(
            "✓ Stress test 2D: 1000 vertices → {} cells",
            tds.number_of_cells()
        );
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "Stress test - enable with --features slow-tests"
)]
fn test_stress_1000_vertices_3d() {
    use rand::Rng;
    let mut rng = rand::rng();

    // Generate points first, then convert to vertices
    let points: Vec<Point<f64, 3>> = (0..1000)
        .map(|_| {
            let x: f64 = rng.random();
            let y: f64 = rng.random();
            let z: f64 = rng.random();
            Point::new([x, y, z])
        })
        .collect();
    let vertices = Vertex::<f64, (), 3>::from_points(&points);

    let tds = Tds::<f64, (), (), 3>::new(&vertices);
    assert!(tds.is_ok(), "Should handle 1000 vertices in 3D");

    if let Ok(tds) = tds {
        assert!(tds.is_valid().is_ok(), "1000-vertex 3D TDS should be valid");
        println!(
            "✓ Stress test 3D: 1000 vertices → {} cells",
            tds.number_of_cells()
        );
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "Stress test - enable with --features slow-tests"
)]
fn test_stress_5000_vertices_3d() {
    use rand::Rng;
    let mut rng = rand::rng();

    // Generate points first, then convert to vertices
    let points: Vec<Point<f64, 3>> = (0..5000)
        .map(|_| {
            let x: f64 = rng.random();
            let y: f64 = rng.random();
            let z: f64 = rng.random();
            Point::new([x, y, z])
        })
        .collect();
    let vertices = Vertex::<f64, (), 3>::from_points(&points);

    let tds = Tds::<f64, (), (), 3>::new(&vertices);
    assert!(tds.is_ok(), "Should handle 5000 vertices in 3D");

    if let Ok(tds) = tds {
        assert!(tds.is_valid().is_ok(), "5000-vertex 3D TDS should be valid");
        println!(
            "✓ Stress test 3D: 5000 vertices → {} cells",
            tds.number_of_cells()
        );
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "Stress test - enable with --features slow-tests"
)]
fn test_stress_removal_operations() {
    use rand::Rng;
    let mut rng = rand::rng();

    // Generate points first, then convert to vertices
    let points: Vec<Point<f64, 3>> = (0..500)
        .map(|_| {
            let x: f64 = rng.random();
            let y: f64 = rng.random();
            let z: f64 = rng.random();
            Point::new([x, y, z])
        })
        .collect();
    let vertices = Vertex::<f64, (), 3>::from_points(&points);

    let mut tds = Tds::<f64, (), (), 3>::new(&vertices).unwrap();

    let initial_cells = tds.number_of_cells();
    println!("Initial cells: {initial_cells}");

    // Remove 10% of cells
    let cells_to_remove: Vec<_> = tds.cell_keys().take(initial_cells / 10).collect();

    let removed = tds.remove_cells_by_keys(&cells_to_remove);
    println!("Removed {removed} cells");

    assert_eq!(
        tds.number_of_cells(),
        initial_cells - removed,
        "Cell count should decrease by removal count"
    );

    println!(
        "✓ Stress test removal: {} → {} cells",
        initial_cells,
        tds.number_of_cells()
    );
}
