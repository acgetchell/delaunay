//! Basic integration tests for TDS creation, neighbor assignment, and validation.
//!
//! This module contains fundamental integration tests that verify correct behavior of
//! triangulation data structure (TDS) operations including:
//! - Basic TDS creation with various vertex configurations
//! - Neighbor assignment and connectivity
//! - Boundary facet computation
//! - Basic validation operations
//!
//! These tests focus on simple, well-understood geometries to establish baseline
//! functionality. For more complex scenarios, see other integration tests in this directory.

#![expect(deprecated)] // Tests use deprecated Tds::new() until migration to DelaunayTriangulation

use delaunay::core::traits::boundary_analysis::BoundaryAnalysis;
use delaunay::core::triangulation_data_structure::{Tds, TriangulationConstructionState};
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use uuid::Uuid;

// =============================================================================
// TDS CREATION TESTS
// =============================================================================

#[test]
fn test_tds_creates_one_cell() {
    // This should create 1 tetrahedron (minimal 3D simplex)
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
    ];
    let vertices = Vertex::from_points(&points);
    let tds: Tds<f64, usize, usize, 3> = Tds::new(&vertices).unwrap();

    println!(
        "Created TDS with {} vertices and {} cells",
        tds.number_of_vertices(),
        tds.number_of_cells()
    );

    assert_eq!(tds.number_of_vertices(), 4, "Should have 4 vertices");
    assert_eq!(tds.number_of_cells(), 1, "Should have 1 cell");
    assert!(
        delaunay::core::util::is_delaunay(&tds).is_ok(),
        "Single-cell triangulation should be globally Delaunay"
    );
}

#[test]
fn test_tds_creates_two_cells() {
    // This should create 2 adjacent tetrahedra sharing a triangular face
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.5, 1.0, 0.0]),
        Point::new([0.5, 0.5, 1.0]),
        Point::new([0.5, 0.5, -1.0]), // Point on opposite side creates second cell
    ];
    let vertices = Vertex::from_points(&points);
    let tds: Tds<f64, usize, usize, 3> = Tds::new(&vertices).unwrap();

    println!(
        "Created TDS with {} vertices and {} cells",
        tds.number_of_vertices(),
        tds.number_of_cells()
    );

    assert_eq!(tds.number_of_vertices(), 5, "Should have 5 vertices");
    assert_eq!(tds.number_of_cells(), 2, "Should have 2 cells");
    assert!(
        delaunay::core::util::is_delaunay(&tds).is_ok(),
        "Two-cell triangulation should be globally Delaunay"
    );
}

// =============================================================================
// NEIGHBOR ASSIGNMENT TESTS
// =============================================================================

#[test]
fn test_initial_simplex_has_neighbors() {
    // Create initial simplex
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.5, 1.0, 0.0]),
        Point::new([0.5, 0.5, 1.0]),
    ];
    let vertices = Vertex::from_points(&points);
    let tds: Tds<f64, usize, usize, 3> = Tds::new(&vertices).unwrap();

    println!("TDS has {} cells", tds.number_of_cells());

    // Check if the cell has neighbors assigned
    for (cell_key, cell) in tds.cells() {
        println!(
            "Cell {cell_key:?} has neighbors: {}",
            cell.neighbors().is_some()
        );
        if let Some(neighbors) = cell.neighbors() {
            println!("  Neighbors count: {}", neighbors.len());
            let non_none_count = neighbors.iter().filter(|n| n.is_some()).count();
            println!("  Non-None neighbors: {non_none_count}");
        }
    }

    // Check if we can compute boundary facets
    match tds.boundary_facets() {
        Ok(boundary_facets) => {
            let count = boundary_facets.count();
            println!("\nBoundary facets: {count}");
            println!("Expected: 4 (for a single tetrahedron, all 4 faces are boundary)");
            assert_eq!(count, 4, "Single tetrahedron should have 4 boundary facets");
            assert!(
                delaunay::core::util::is_delaunay(&tds).is_ok(),
                "Single tetrahedron triangulation should be globally Delaunay"
            );
        }
        Err(e) => {
            eprintln!("Failed to get boundary facets: {e}");
            panic!("Cannot compute boundary facets: {e}");
        }
    }
}

#[test]
fn test_two_cells_share_facet() {
    // Create two cells that share a triangular facet
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.5, 1.0, 0.0]),
        Point::new([0.5, 0.5, 1.0]),
        Point::new([0.5, 0.5, -1.0]),
    ];
    let vertices = Vertex::from_points(&points);
    let tds: Tds<f64, usize, usize, 3> = Tds::new(&vertices).unwrap();

    assert_eq!(tds.number_of_cells(), 2, "Should have 2 cells");

    // Each cell should have neighbors assigned
    let mut cells_with_neighbors = 0;
    let mut total_neighbor_connections = 0;

    for (_cell_key, cell) in tds.cells() {
        if let Some(neighbors) = cell.neighbors() {
            cells_with_neighbors += 1;
            total_neighbor_connections += neighbors.iter().filter(|n| n.is_some()).count();
        }
    }

    println!("Cells with neighbors: {cells_with_neighbors}/2");
    println!("Total neighbor connections: {total_neighbor_connections}");

    assert_eq!(
        cells_with_neighbors, 2,
        "Both cells should have neighbor data"
    );
    assert!(
        total_neighbor_connections >= 2,
        "Should have at least 2 neighbor connections (cells are neighbors to each other)"
    );

    // Boundary facets: 2 cells sharing a facet means 6 boundary facets
    // (each cell has 4 faces, minus 2 shared faces = 8 - 2 = 6)
    match tds.boundary_facets() {
        Ok(boundary_facets) => {
            let count = boundary_facets.count();
            println!("\nBoundary facets: {count}");
            println!("Expected: 6 (two tetrahedra sharing one face have 6 external faces)");
            assert_eq!(
                count, 6,
                "Two cells sharing a facet should have 6 boundary facets"
            );
            assert!(
                delaunay::core::util::is_delaunay(&tds).is_ok(),
                "Two-cell triangulation should be globally Delaunay"
            );
        }
        Err(e) => {
            eprintln!("Failed to get boundary facets: {e}");
            panic!("Cannot compute boundary facets: {e}");
        }
    }
}

// =============================================================================
// DUPLICATE HANDLING TESTS
// =============================================================================

#[test]
fn test_duplicate_uuid_with_duplicate_coordinates_is_caught() {
    // Test that duplicate UUIDs are caught even when coordinates are duplicated
    // This verifies the fix where UUID checking happens before coordinate deduplication
    let uuid = Uuid::new_v4();
    let point = Point::new([1.0, 2.0, 3.0]);

    let vertices: Vec<Vertex<f64, usize, 3>> = vec![
        Vertex::new_with_uuid(point, uuid, Some(0)),
        Vertex::new_with_uuid(point, uuid, Some(1)),
    ];

    let result = Tds::<f64, usize, usize, 3>::new(&vertices);
    assert!(
        result.is_err(),
        "Should error on duplicate UUID even with duplicate coordinates"
    );

    if let Err(e) = result {
        let error_msg = format!("{e}");
        assert!(
            error_msg.contains("Duplicate UUID"),
            "Error should mention duplicate UUID, got: {error_msg}"
        );
    }
}

#[test]
fn test_duplicate_uuid_with_different_coordinates_is_caught() {
    // Test that duplicate UUIDs are caught when coordinates differ
    let uuid = Uuid::new_v4();

    let vertices: Vec<Vertex<f64, usize, 3>> = vec![
        Vertex::new_with_uuid(Point::new([1.0, 2.0, 3.0]), uuid, Some(0)),
        Vertex::new_with_uuid(Point::new([4.0, 5.0, 6.0]), uuid, Some(1)),
    ];

    let result = Tds::<f64, usize, usize, 3>::new(&vertices);
    assert!(
        result.is_err(),
        "Should error on duplicate UUID with different coordinates"
    );

    if let Err(e) = result {
        let error_msg = format!("{e}");
        assert!(
            error_msg.contains("Duplicate UUID"),
            "Error should mention duplicate UUID, got: {error_msg}"
        );
    }
}

#[test]
fn test_construction_state_reflects_unique_vertices() {
    // Test that construction_state is calculated based on unique vertices after deduplication
    // Use 2D to need only 3 vertices instead of 4
    let point1 = Point::new([0.0, 0.0]);
    let point2 = Point::new([1.0, 0.0]);

    // Create vertices with duplicates: 5 vertices but only 2 unique points
    let vertices: Vec<Vertex<f64, usize, 2>> = vec![
        Vertex::new_with_uuid(point1, Uuid::new_v4(), Some(0)),
        Vertex::new_with_uuid(point1, Uuid::new_v4(), Some(1)),
        Vertex::new_with_uuid(point2, Uuid::new_v4(), Some(2)),
        Vertex::new_with_uuid(point2, Uuid::new_v4(), Some(3)),
        Vertex::new_with_uuid(point1, Uuid::new_v4(), Some(4)),
    ];

    // Should error due to insufficient vertices (2 < D+1 = 3 for 2D)
    let result = Tds::<f64, usize, usize, 2>::new(&vertices);
    assert!(
        result.is_err(),
        "Should error when unique vertices (2) < D+1 (3)"
    );

    // Verify error message mentions insufficient vertices
    if let Err(e) = result {
        let error_msg = format!("{e}");
        assert!(
            error_msg.contains("Insufficient") || error_msg.contains("insufficient"),
            "Error should mention insufficient vertices, got: {error_msg}"
        );
    }
}

#[test]
fn test_construction_state_constructed_after_dedup() {
    // Test that construction_state is Constructed when unique vertices >= D+1
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
    ];

    // Add duplicates of the same 4 points
    let mut vertices: Vec<Vertex<f64, usize, 3>> = Vec::new();
    for point in &points {
        vertices.push(Vertex::new_with_uuid(*point, Uuid::new_v4(), Some(0)));
    }
    // Add duplicate coordinates (should be skipped)
    for point in &points {
        vertices.push(Vertex::new_with_uuid(*point, Uuid::new_v4(), Some(1)));
    }

    let tds = Tds::<f64, usize, usize, 3>::new(&vertices).unwrap();

    // Should have 4 unique vertices
    assert_eq!(
        tds.number_of_vertices(),
        4,
        "Should have 4 unique vertices after deduplication"
    );

    // construction_state should be Constructed (4 >= D+1 = 4)
    assert!(
        matches!(
            tds.construction_state,
            TriangulationConstructionState::Constructed
        ),
        "construction_state should be Constructed for 4 unique vertices in 3D, got: {:?}",
        tds.construction_state
    );
}
