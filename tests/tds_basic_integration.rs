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

use delaunay::core::traits::boundary_analysis::BoundaryAnalysis;
use delaunay::core::triangulation_data_structure::Tds;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;

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
    let vertices = Vertex::from_points(points);
    let tds: Tds<f64, usize, usize, 3> = Tds::new(&vertices).unwrap();

    println!(
        "Created TDS with {} vertices and {} cells",
        tds.number_of_vertices(),
        tds.number_of_cells()
    );

    assert_eq!(tds.number_of_vertices(), 4, "Should have 4 vertices");
    assert_eq!(tds.number_of_cells(), 1, "Should have 1 cell");
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
    let vertices = Vertex::from_points(points);
    let tds: Tds<f64, usize, usize, 3> = Tds::new(&vertices).unwrap();

    println!(
        "Created TDS with {} vertices and {} cells",
        tds.number_of_vertices(),
        tds.number_of_cells()
    );

    assert_eq!(tds.number_of_vertices(), 5, "Should have 5 vertices");
    assert_eq!(tds.number_of_cells(), 2, "Should have 2 cells");
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
    let vertices = Vertex::from_points(points);
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
    let vertices = Vertex::from_points(points);
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
        }
        Err(e) => {
            eprintln!("Failed to get boundary facets: {e}");
            panic!("Cannot compute boundary facets: {e}");
        }
    }
}
