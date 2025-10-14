//! Integration tests for neighbor assignment in triangulation cells.

use delaunay::core::traits::boundary_analysis::BoundaryAnalysis;
use delaunay::core::triangulation_data_structure::Tds;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;

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
