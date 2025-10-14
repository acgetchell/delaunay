//! Integration tests for TDS creation with various vertex configurations.

use delaunay::core::triangulation_data_structure::Tds;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;

#[test]
fn test_tds_creates_two_cells() {
    // This should create 2 adjacent tetrahedra
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.5, 1.0, 0.0]),
        Point::new([0.5, 0.5, 1.0]),
        Point::new([0.5, 0.5, -1.0]),
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

#[test]
fn test_tds_creates_one_cell() {
    // This should create 1 tetrahedron
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
