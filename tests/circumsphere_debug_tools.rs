//! # Circumsphere Debug Tools
//!
//! This test module provides interactive debugging and testing tools for circumsphere
//! calculations. It demonstrates and compares three methods for testing whether a point
//! lies inside the circumsphere of a simplex in 2D, 3D, and 4D.
//!
//! ## Usage
//!
//! Run specific test functions with:
//! ```bash
//! cargo test --test circumsphere_debug_tools test_2d_circumsphere_debug -- --nocapture
//! cargo test --test circumsphere_debug_tools test_3d_circumsphere_debug -- --nocapture
//! cargo test --test circumsphere_debug_tools test_all_debug -- --ignored --nocapture
//! ```

use delaunay::geometry::util::squared_norm;
use delaunay::prelude::*;
use nalgebra as na;
use serde::{Deserialize, Serialize};

// Macro for standard test output formatting
macro_rules! test_output {
    ($name:expr, $vertices:expr, $test_points:expr) => {
        test_circumsphere_generic($name, $vertices, $test_points)
    };
}

// Macro for creating test point arrays with descriptions
macro_rules! test_points {
    ($($coords:expr => $desc:expr),* $(,)?) => {
        vec![$( ($coords, $desc) ),*]
    };
}

// Macro for running multiple tests in sequence
macro_rules! run_tests {
    ($($test_fn:ident),* $(,)?) => {
        $( $test_fn(); )*
    };
}

// Macro for printing test results with consistent formatting
macro_rules! print_result {
    ($method:expr, $result:expr) => {
        println!(
            "  {:<18} {}",
            format!("{}:", $method),
            match $result {
                &Ok(InSphere::INSIDE) => "INSIDE",
                &Ok(InSphere::BOUNDARY) => "BOUNDARY",
                &Ok(InSphere::OUTSIDE) => "OUTSIDE",
                &Err(_) => "ERROR",
            }
        );
    };
}

// Test functions organized by category
#[test]
fn test_2d_circumsphere_debug() {
    test_2d_circumsphere();
}

#[test]
fn test_3d_circumsphere_debug() {
    test_3d_circumsphere();
}

#[test]
fn test_4d_circumsphere_debug() {
    test_4d_circumsphere();
}

#[test]
fn test_all_orientations_debug() {
    test_all_orientations();
}

#[test]
fn test_all_basic_debug() {
    run_all_basic_tests();
}

#[test]
fn test_3d_simplex_analysis_debug() {
    test_3d_simplex_analysis();
}

#[test]
fn test_3d_matrix_analysis_debug() {
    test_3d_matrix_analysis();
}

#[test]
fn test_3d_properties_debug() {
    debug_3d_circumsphere_properties();
}

#[test]
fn test_4d_properties_debug() {
    debug_4d_circumsphere_properties();
}

#[test]
fn test_compare_methods_debug() {
    compare_circumsphere_methods();
}

#[test]
fn test_orientation_impact_debug() {
    demonstrate_orientation_impact_on_circumsphere();
}

#[test]
fn test_containment_debug() {
    test_circumsphere_containment();
}

#[test]
fn test_4d_methods_debug() {
    test_4d_circumsphere_methods();
}

#[test]
#[ignore = "heavyweight aggregate test with lots of output - run manually for debugging"]
fn test_all_debug() {
    run_all_debug_tests();
}

#[test]
fn test_single_2d_point_debug() {
    test_single_2d_point();
}

#[test]
fn test_single_3d_point_debug() {
    test_single_3d_point();
}

#[test]
fn test_single_4d_point_debug() {
    test_single_4d_point();
}

#[test]
fn test_all_single_points_debug() {
    test_all_single_points();
}

#[test]
#[ignore = "comprehensive test suite with extensive output - run manually for full analysis"]
fn test_everything_debug() {
    run_everything();
}

/// Test 2D circumsphere methods with a triangle
fn test_2d_circumsphere() {
    let vertices = vec![
        vertex!([0.0, 0.0], 0),
        vertex!([1.0, 0.0], 1),
        vertex!([0.0, 1.0], 2),
    ];

    let test_points = test_points!(
        [0.5, 0.5] => "circumcenter_region",
        [0.1, 0.1] => "clearly_inside",
        [0.9, 0.9] => "possibly_outside",
        [2.0, 2.0] => "far_outside",
        [0.0, 0.0] => "vertex_origin",
        [0.5, 0.0] => "edge_midpoint",
        [0.25, 0.25] => "inside_triangle",
        [std::f64::consts::FRAC_1_SQRT_2, 0.0] => "boundary_distance", // approximately on circumcircle
        [1e-15, 1e-15] => "numerical_precision", // test floating point precision issues
    );

    test_output!("2D", &vertices, test_points);
}

/// Test a single 2D point against all circumsphere methods
fn test_2d_point(
    vertices: &[Vertex<f64, i32, 2>],
    coords: [f64; 2],
    description: &str,
    center: &[f64; 2],
    radius: f64,
) {
    test_point_generic(vertices, coords, description, center, radius);
}

/// Generic function to test circumsphere methods for any dimension
fn test_circumsphere_generic<const D: usize>(
    dimension_name: &str,
    vertices: &[Vertex<f64, i32, D>],
    test_points: Vec<([f64; D], &str)>,
) where
    [f64; D]: Copy + Sized + Serialize + for<'de> Deserialize<'de>,
{
    println!("Testing {dimension_name} circumsphere methods");
    println!("=============================================");

    // Print vertices
    println!("{dimension_name} vertices:");
    for (i, vertex) in vertices.iter().enumerate() {
        let coords: [f64; D] = vertex.into();
        println!("  v{i}: {coords:?}");
    }
    println!();

    // Calculate circumcenter and circumradius
    let vertex_points: Vec<Point<f64, D>> = vertices.iter().map(Point::from).collect();
    match (circumcenter(&vertex_points), circumradius(&vertex_points)) {
        (Ok(center), Ok(radius)) => {
            println!("Circumcenter: {:?}", center.to_array());
            println!("Circumradius: {radius:.6}");
            println!();

            for (coords, description) in test_points {
                test_point_generic(vertices, coords, description, &center.to_array(), radius);
            }
        }
        (Err(e), _) => println!("Error calculating circumcenter: {e}"),
        (_, Err(e)) => println!("Error calculating circumradius: {e}"),
    }

    println!("{dimension_name} circumsphere testing completed.\n");
}

/// Generic function to test a single point against circumsphere methods
fn test_point_generic<const D: usize>(
    vertices: &[Vertex<f64, i32, D>],
    coords: [f64; D],
    description: &str,
    center: &[f64; D],
    radius: f64,
) where
    [f64; D]: Copy + Sized + Serialize + for<'de> Deserialize<'de>,
{
    let test_vertex: Vertex<f64, i32, D> = vertex!(coords, 99);

    let vertex_points: Vec<Point<f64, D>> = vertices.iter().map(Point::from).collect();
    let result_insphere = insphere(&vertex_points, Point::from(&test_vertex));
    let result_distance = insphere_distance(&vertex_points, Point::from(&test_vertex));
    let result_lifted = insphere_lifted(&vertex_points, Point::from(&test_vertex));

    // Calculate actual distance to center using nalgebra
    let distance_to_center = na::distance(
        &na::Point::<f64, D>::from(*center),
        &na::Point::<f64, D>::from(coords),
    );

    println!("Point {description} {coords:?}:");
    println!("  Distance to center: {distance_to_center:.6}");
    println!("  Expected inside: {}", distance_to_center <= radius);
    print_result!("determinant-based", &result_insphere);
    print_result!("distance-based", &result_distance);
    print_result!("matrix-based", &result_lifted);

    // Check agreement between methods
    let methods_agree =
        if let (Ok(r1), Ok(r2), Ok(r3)) = (&result_insphere, &result_distance, &result_lifted) {
            r1 == r2 && r2 == r3
        } else {
            false
        };

    if methods_agree {
        println!("  ✓ All methods agree");
    } else {
        println!("  ⚠ Methods disagree");
    }
    println!();
}

/// Test 3D circumsphere methods with a tetrahedron
fn test_3d_circumsphere() {
    // Create a unit tetrahedron: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0], 0),
        vertex!([1.0, 0.0, 0.0], 1),
        vertex!([0.0, 1.0, 0.0], 2),
        vertex!([0.0, 0.0, 1.0], 3),
    ];

    let test_points = test_points!(
        [0.5, 0.5, 0.5] => "circumcenter_region",
        [0.1, 0.1, 0.1] => "clearly_inside",
        [0.9, 0.9, 0.9] => "possibly_outside",
        [2.0, 2.0, 2.0] => "far_outside",
        [0.0, 0.0, 0.0] => "vertex_origin",
        [0.25, 0.25, 0.0] => "face_center",
        [0.2, 0.2, 0.2] => "inside_tetrahedron",
    );

    test_output!("3D (tetrahedron)", &vertices, test_points);
}

/// Test a single 3D point against all circumsphere methods
fn test_3d_point(
    vertices: &[Vertex<f64, i32, 3>],
    coords: [f64; 3],
    description: &str,
    center: &[f64; 3],
    radius: f64,
) {
    test_point_generic(vertices, coords, description, center, radius);
}

/// Test 4D circumsphere methods with a 4-simplex
fn test_4d_circumsphere() {
    // Create a unit 4-simplex: vertices at origin and unit vectors along each axis
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0], 0),
        vertex!([1.0, 0.0, 0.0, 0.0], 1),
        vertex!([0.0, 1.0, 0.0, 0.0], 2),
        vertex!([0.0, 0.0, 1.0, 0.0], 3),
        vertex!([0.0, 0.0, 0.0, 1.0], 4),
    ];

    let test_points = test_points!(
        [0.5, 0.5, 0.5, 0.5] => "circumcenter",
        [0.1, 0.1, 0.1, 0.1] => "clearly_inside",
        [0.9, 0.9, 0.9, 0.9] => "possibly_outside",
        [2.0, 2.0, 2.0, 2.0] => "far_outside",
        [0.0, 0.0, 0.0, 0.0] => "vertex_origin",
        [0.25, 0.0, 0.0, 0.0] => "axis_point",
        [0.2, 0.2, 0.2, 0.2] => "inside_simplex",
    );

    test_output!("4D (4-simplex)", &vertices, test_points);
}

/// Test a single 4D point against all circumsphere methods
fn test_4d_point(
    vertices: &[Vertex<f64, i32, 4>],
    coords: [f64; 4],
    description: &str,
    center: &[f64; 4],
    radius: f64,
) {
    test_point_generic(vertices, coords, description, center, radius);
}

/// Run all orientation tests for 2D, 3D, and 4D
fn test_all_orientations() {
    println!("=============================================");
    println!("Testing 2D, 3D, and 4D simplex orientations");
    println!("=============================================");
    test_simplex_orientation();
    println!("Orientation tests completed\n");
}

/// Test and compare both 4D circumsphere containment methods
fn test_4d_circumsphere_methods() {
    println!("=============================================");
    println!("Testing 4D circumsphere containment methods:");
    println!("  circumsphere_contains_vertex vs circumsphere_contains_vertex_matrix");
    println!("=============================================");

    // Create a unit 4-simplex: vertices at origin and unit vectors along each axis
    let vertices: Vec<Vertex<f64, i32, 4>> = vec![
        vertex!([0.0, 0.0, 0.0, 0.0], 1),
        vertex!([1.0, 0.0, 0.0, 0.0], 1),
        vertex!([0.0, 1.0, 0.0, 0.0], 1),
        vertex!([0.0, 0.0, 1.0, 0.0], 1),
        vertex!([0.0, 0.0, 0.0, 1.0], 2),
    ];

    // Calculate circumcenter and circumradius for reference
    let vertex_points: Vec<Point<f64, 4>> = vertices.iter().map(Point::from).collect();
    match circumcenter(&vertex_points) {
        Ok(center) => {
            println!("Circumcenter: {:?}", center.to_array());
            match circumradius(&vertex_points) {
                Ok(radius) => {
                    println!("Circumradius: {radius}");
                    println!();

                    // Test various points
                    let test_points = vec![
                        ([0.5, 0.5, 0.5, 0.5], "circumcenter"),
                        ([0.1, 0.1, 0.1, 0.1], "clearly_inside"),
                        ([0.9, 0.9, 0.9, 0.9], "possibly_outside"),
                        ([10.0, 10.0, 10.0, 10.0], "far_outside"),
                        ([0.0, 0.0, 0.0, 0.0], "origin"),
                        ([0.25, 0.0, 0.0, 0.0], "axis_point"),
                        ([0.5, 0.5, 0.0, 0.0], "equidistant"),
                    ];

                    for (coords, description) in test_points {
                        test_point_generic(
                            vertices.as_slice(),
                            coords,
                            description,
                            &center.to_array(),
                            radius,
                        );
                    }
                }
                Err(e) => println!("Error calculating circumradius: {e}"),
            }
        }
        Err(e) => println!("Error calculating circumcenter: {e}"),
    }

    println!("4D method comparison completed.\n");
}

#[expect(
    clippy::too_many_lines,
    reason = "Intentional verbose debug harness; kept as a single flow for manual inspection"
)]
fn test_circumsphere_containment() {
    println!("Testing circumsphere containment:");
    println!("  determinant-based (insphere) vs distance-based (insphere_distance)");
    println!("=============================================");

    // Define the 4D simplex vertices that form a unit 5-cell
    // This creates a 4D simplex with vertices at the origin and unit vectors
    // along each coordinate axis. The circumcenter is at [0.5, 0.5, 0.5, 0.5] and
    // the circumradius is √4/2 = 1.0.
    let vertices: [Vertex<f64, i32, 4>; 5] = [
        vertex!([0.0, 0.0, 0.0, 0.0], 0), // Origin
        vertex!([1.0, 0.0, 0.0, 0.0], 1), // Unit vector along x-axis
        vertex!([0.0, 1.0, 0.0, 0.0], 2), // Unit vector along y-axis
        vertex!([0.0, 0.0, 1.0, 0.0], 3), // Unit vector along z-axis
        vertex!([0.0, 0.0, 0.0, 1.0], 4), // Unit vector along w-axis
    ];

    println!("4D simplex vertices:");
    for (i, vertex) in vertices.iter().enumerate() {
        let coords: [f64; 4] = vertex.into();
        println!(
            "  v{}: [{}, {}, {}, {}]",
            i, coords[0], coords[1], coords[2], coords[3]
        );
    }
    println!();

    // Test points that should be inside the circumsphere
    // These are points with small coordinates that should be well within
    // the circumsphere radius of √4/2 = 1.0
    let test_points_inside: [Vertex<f64, i32, 4>; 5] = [
        vertex!([0.25, 0.25, 0.25, 0.25], 10),
        vertex!([0.1, 0.1, 0.1, 0.1], 11),
        vertex!([0.2, 0.2, 0.2, 0.2], 12),
        vertex!([0.3, 0.2, 0.1, 0.0], 13),
        vertex!([0.0, 0.0, 0.0, 0.0], 14), // Origin should be inside
    ];

    // Calculate circumcenter and circumradius for testing
    let vertex_points: Vec<Point<f64, 4>> = vertices.iter().map(Point::from).collect();
    let (Ok(center), Ok(radius)) = (circumcenter(&vertex_points), circumradius(&vertex_points))
    else {
        println!("Error calculating circumcenter or circumradius");
        return;
    };

    println!("Testing points that should be INSIDE the circumsphere:");
    for point in test_points_inside {
        let coords: [f64; 4] = point.into();
        test_point_generic(&vertices, coords, "inside", &center.to_array(), radius);
    }
    println!();

    // Test points that should be outside (or on the boundary) of the circumsphere
    // These include points with large coordinates and points along the axes
    // that extend beyond the simplex vertices
    let test_points_outside: [Vertex<f64, i32, 4>; 6] = [
        vertex!([2.0, 2.0, 2.0, 2.0], 20),
        vertex!([1.0, 1.0, 1.0, 1.0], 21), // boundary
        vertex!([0.8, 0.8, 0.8, 0.8], 22),
        vertex!([1.5, 0.0, 0.0, 0.0], 23),
        vertex!([0.0, 1.5, 0.0, 0.0], 24),
        vertex!([0.0, 0.0, 1.5, 0.0], 25),
    ];

    println!("Testing points that should be OUTSIDE (or on boundary) the circumsphere:");
    for (i, point) in test_points_outside.iter().enumerate() {
        let vertex_points: Vec<Point<f64, 4>> = vertices.iter().map(Point::from).collect();
        let result_determinant = insphere(&vertex_points, Point::from(point));
        let result_distance = insphere_distance(&vertex_points, Point::from(point));
        let coords: [f64; 4] = point.into();
        println!(
            "  Point {}: [{}, {}, {}, {}] -> Det: {}, Dist: {}",
            i,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            match result_determinant {
                Ok(InSphere::INSIDE) => "INSIDE",
                Ok(InSphere::BOUNDARY) => "BOUNDARY",
                Ok(InSphere::OUTSIDE) => "OUTSIDE",
                Err(_) => "ERROR",
            },
            match result_distance {
                Ok(InSphere::INSIDE) => "INSIDE",
                Ok(InSphere::BOUNDARY) => "BOUNDARY",
                Ok(InSphere::OUTSIDE) => "OUTSIDE",
                Err(_) => "ERROR",
            }
        );
    }
    println!();

    // Test edge cases - points on the simplex vertices themselves
    // These should be on the boundary of the circumsphere (distance = radius)
    println!("Testing the simplex vertices themselves:");
    for (i, vertex) in vertices.iter().enumerate() {
        let vertex_points: Vec<Point<f64, 4>> = vertices.iter().map(Point::from).collect();
        let result = insphere(&vertex_points, Point::from(vertex));
        let coords: [f64; 4] = vertex.into();
        println!(
            "  Vertex {}: [{}, {}, {}, {}] -> {}",
            i,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            match result {
                Ok(InSphere::INSIDE) => "INSIDE",
                Ok(InSphere::BOUNDARY) => "BOUNDARY",
                Ok(InSphere::OUTSIDE) => "OUTSIDE",
                Err(_) => "ERROR",
            }
        );
    }
    println!();

    // Additional boundary testing with points on edges and faces of the tetrahedron
    // These points lie on the boundary of the 4D simplex and test
    // numerical stability near the boundary
    let boundary_points: [Vertex<f64, i32, 4>; 5] = [
        vertex!([0.5, 0.5, 0.0, 0.0], 30),
        vertex!([0.5, 0.0, 0.5, 0.0], 31),
        vertex!([0.0, 0.5, 0.5, 0.0], 32),
        vertex!([0.33, 0.33, 0.33, 0.01], 33),
        vertex!([0.25, 0.25, 0.25, 0.25], 34),
    ];

    println!("Testing boundary/edge points:");
    for (i, point) in boundary_points.iter().enumerate() {
        let vertex_points: Vec<Point<f64, 4>> = vertices.iter().map(Point::from).collect();
        let result = insphere(&vertex_points, Point::from(point));
        let coords: [f64; 4] = point.into();
        println!(
            "  Point {}: [{}, {}, {}, {}] -> {}",
            i,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            match result {
                Ok(InSphere::INSIDE) => "INSIDE",
                Ok(InSphere::BOUNDARY) => "BOUNDARY",
                Ok(InSphere::OUTSIDE) => "OUTSIDE",
                Err(_) => "ERROR",
            }
        );
    }
}

fn test_simplex_orientation() {
    println!("\n=============================================");
    println!("Testing simplex orientation:");
    println!("=============================================");

    // Create vertices for testing
    let vertices = create_unit_4d_simplex();

    // Test the original 4D simplex's orientation
    let vertex_points: Vec<Point<f64, 4>> = vertices.iter().map(Point::from).collect();
    let orientation_original = simplex_orientation(&vertex_points);
    println!(
        "Original 4D simplex orientation: {}",
        match orientation_original {
            Ok(Orientation::POSITIVE) => "POSITIVE",
            Ok(Orientation::NEGATIVE) => "NEGATIVE",
            Ok(Orientation::DEGENERATE) => "DEGENERATE",
            Err(_) => "ERROR",
        }
    );

    // Create a negatively oriented 4D simplex by swapping two vertices
    let vertices_negative = create_negative_4d_simplex();

    let vertex_points_negative: Vec<Point<f64, 4>> =
        vertices_negative.iter().map(Point::from).collect();
    let orientation_negative = simplex_orientation(&vertex_points_negative);
    println!(
        "Negatively oriented 4D simplex: {}",
        match orientation_negative {
            Ok(Orientation::POSITIVE) => "POSITIVE",
            Ok(Orientation::NEGATIVE) => "NEGATIVE",
            Ok(Orientation::DEGENERATE) => "DEGENERATE",
            Err(_) => "ERROR",
        }
    );

    // Test 3D orientation (tetrahedron) for comparison
    let tetrahedron_vertices: [Vertex<f64, i32, 3>; 4] = [
        vertex!([0.0, 0.0, 0.0], 0), // Origin
        vertex!([1.0, 0.0, 0.0], 1), // Unit vector along x-axis
        vertex!([0.0, 1.0, 0.0], 2), // Unit vector along y-axis
        vertex!([0.0, 0.0, 1.0], 3), // Unit vector along z-axis
    ];

    let tetrahedron_points: Vec<Point<f64, 3>> =
        tetrahedron_vertices.iter().map(Point::from).collect();
    let orientation_3d = simplex_orientation(&tetrahedron_points);
    println!(
        "3D tetrahedron orientation: {}",
        match orientation_3d {
            Ok(Orientation::POSITIVE) => "POSITIVE",
            Ok(Orientation::NEGATIVE) => "NEGATIVE",
            Ok(Orientation::DEGENERATE) => "DEGENERATE",
            Err(_) => "ERROR",
        }
    );

    // Test 2D orientation (triangle)
    let triangle_vertices: [Vertex<f64, i32, 2>; 3] = [
        vertex!([0.0, 0.0], 0),
        vertex!([1.0, 0.0], 1),
        vertex!([0.0, 1.0], 2),
    ];

    let triangle_points: Vec<Point<f64, 2>> = triangle_vertices.iter().map(Point::from).collect();
    let orientation_2d = simplex_orientation(&triangle_points);
    println!(
        "2D triangle orientation: {}",
        match orientation_2d {
            Ok(Orientation::POSITIVE) => "POSITIVE",
            Ok(Orientation::NEGATIVE) => "NEGATIVE",
            Ok(Orientation::DEGENERATE) => "DEGENERATE",
            Err(_) => "ERROR",
        }
    );

    // Test 2D orientation with reversed vertex order
    let triangle_vertices_reversed: [Vertex<f64, i32, 2>; 3] = [
        vertex!([0.0, 0.0], 0),
        vertex!([0.0, 1.0], 2), // Swapped order
        vertex!([1.0, 0.0], 1), // Swapped order
    ];

    let triangle_points_reversed: Vec<Point<f64, 2>> =
        triangle_vertices_reversed.iter().map(Point::from).collect();
    let orientation_2d_reversed = simplex_orientation(&triangle_points_reversed);
    println!(
        "2D triangle (reversed order): {}",
        match orientation_2d_reversed {
            Ok(Orientation::POSITIVE) => "POSITIVE",
            Ok(Orientation::NEGATIVE) => "NEGATIVE",
            Ok(Orientation::DEGENERATE) => "DEGENERATE",
            Err(_) => "ERROR",
        }
    );

    // Test degenerate case (collinear points in 2D)
    let collinear_vertices: [Vertex<f64, i32, 2>; 3] = [
        vertex!([0.0, 0.0], 0),
        vertex!([1.0, 0.0], 1),
        vertex!([2.0, 0.0], 2), // Collinear point
    ];

    let collinear_points: Vec<Point<f64, 2>> = collinear_vertices.iter().map(Point::from).collect();
    let orientation_collinear = simplex_orientation(&collinear_points);
    println!(
        "Collinear 2D points: {}",
        match orientation_collinear {
            Ok(Orientation::POSITIVE) => "POSITIVE",
            Ok(Orientation::NEGATIVE) => "NEGATIVE",
            Ok(Orientation::DEGENERATE) => "DEGENERATE",
            Err(_) => "ERROR",
        }
    );
}

fn create_unit_4d_simplex() -> [Vertex<f64, i32, 4>; 5] {
    [
        vertex!([0.0, 0.0, 0.0, 0.0], 0),
        vertex!([1.0, 0.0, 0.0, 0.0], 1),
        vertex!([0.0, 1.0, 0.0, 0.0], 2),
        vertex!([0.0, 0.0, 1.0, 0.0], 3),
        vertex!([0.0, 0.0, 0.0, 1.0], 4),
    ]
}

fn create_negative_4d_simplex() -> [Vertex<f64, i32, 4>; 5] {
    [
        vertex!([0.0, 0.0, 0.0, 0.0], 0),
        vertex!([0.0, 1.0, 0.0, 0.0], 2),
        vertex!([1.0, 0.0, 0.0, 0.0], 1),
        vertex!([0.0, 0.0, 1.0, 0.0], 3),
        vertex!([0.0, 0.0, 0.0, 1.0], 4),
    ]
}

fn demonstrate_orientation_impact_on_circumsphere() {
    println!("\n--- Impact of orientation on circumsphere testing ---");

    // Create vertices for testing
    let vertices = create_unit_4d_simplex();
    let vertices_negative = create_negative_4d_simplex();

    let test_point: Vertex<f64, i32, 4> = vertex!([0.25, 0.25, 0.25, 0.25], 100);

    let vertex_points: Vec<Point<f64, 4>> = vertices.iter().map(Point::from).collect();
    let vertex_points_negative: Vec<Point<f64, 4>> =
        vertices_negative.iter().map(Point::from).collect();
    let inside_positive = insphere(&vertex_points, Point::from(&test_point));
    let inside_negative = insphere(&vertex_points_negative, Point::from(&test_point));

    println!(
        "Point [0.25, 0.25, 0.25, 0.25] in positive 4D simplex: {}",
        match inside_positive {
            Ok(InSphere::INSIDE) => "INSIDE",
            Ok(InSphere::BOUNDARY) => "BOUNDARY",
            Ok(InSphere::OUTSIDE) => "OUTSIDE",
            Err(_) => "ERROR",
        }
    );
    println!(
        "Point [0.25, 0.25, 0.25, 0.25] in negative 4D simplex: {}",
        match inside_negative {
            Ok(InSphere::INSIDE) => "INSIDE",
            Ok(InSphere::BOUNDARY) => "BOUNDARY",
            Ok(InSphere::OUTSIDE) => "OUTSIDE",
            Err(_) => "ERROR",
        }
    );

    println!("\nNote: The insphere function automatically handles");
    println!("      orientation by calling simplex_orientation internally.");
    println!("      Both results should be the same regardless of vertex ordering!");

    println!("\nTest completed!");
}

fn test_3d_simplex_analysis() {
    println!("\n=============================================");
    println!("3D Simplex Analysis for Test Debugging");
    println!("=============================================");

    let simplex_vertices_3d = create_3d_test_simplex();

    let simplex_points_3d: Vec<Point<f64, 3>> =
        simplex_vertices_3d.iter().map(Point::from).collect();
    match circumcenter(&simplex_points_3d) {
        Ok(circumcenter_3d) => match circumradius(&simplex_points_3d) {
            Ok(circumradius_3d) => {
                print_3d_simplex_info(&circumcenter_3d, circumradius_3d);
                test_point_against_3d_simplex(
                    &simplex_vertices_3d,
                    &circumcenter_3d,
                    circumradius_3d,
                );
            }
            Err(e) => println!("Error calculating circumradius: {e}"),
        },
        Err(e) => println!("Error calculating circumcenter: {e}"),
    }
}

fn create_3d_test_simplex() -> Vec<Vertex<f64, i32, 3>> {
    let vertex1_3d: Vertex<f64, i32, 3> = vertex!([0.0, 0.0, 0.0], 1);
    let vertex2_3d = vertex!([1.0, 0.0, 0.0], 1);
    let vertex3_3d = vertex!([0.0, 1.0, 0.0], 1);
    let vertex4_3d = vertex!([0.0, 0.0, 1.0], 2);
    vec![vertex1_3d, vertex2_3d, vertex3_3d, vertex4_3d]
}

fn print_3d_simplex_info(circumcenter_3d: &Point<f64, 3>, circumradius_3d: f64) {
    println!("3D Simplex vertices:");
    println!("  v1: (0, 0, 0)");
    println!("  v2: (1, 0, 0)");
    println!("  v3: (0, 1, 0)");
    println!("  v4: (0, 0, 1)");
    println!();
    println!("Circumcenter: {:?}", circumcenter_3d.to_array());
    println!("Circumradius: {circumradius_3d:.6}");
    println!();
}

fn test_point_against_3d_simplex(
    simplex_vertices_3d: &[Vertex<f64, i32, 3>],
    circumcenter_3d: &Point<f64, 3>,
    circumradius_3d: f64,
) {
    // Test the point [0.9, 0.9, 0.9]
    let test_vertex_3d = vertex!([0.9, 0.9, 0.9], 3);

    // Calculate distance from circumcenter to test point
    let distance_to_test_3d = na::distance(
        &na::Point::<f64, 3>::from(circumcenter_3d.to_array()),
        &na::Point::<f64, 3>::from([0.9, 0.9, 0.9]),
    );

    println!("Test point [0.9, 0.9, 0.9]:");
    println!("  Distance from circumcenter: {distance_to_test_3d:.6}");
    println!("  Circumradius: {circumradius_3d:.6}");
    println!(
        "  Inside circumsphere: {}",
        (distance_to_test_3d - circumradius_3d).abs() < f64::EPSILON
            || distance_to_test_3d < circumradius_3d
    );
    println!();

    // Test both methods
    test_circumsphere_methods(simplex_vertices_3d, test_vertex_3d);
    test_boundary_vertex_case(simplex_vertices_3d);
}

fn test_circumsphere_methods(
    simplex_vertices: &[Vertex<f64, i32, 3>],
    test_vertex: Vertex<f64, i32, 3>,
) {
    let simplex_points: Vec<Point<f64, 3>> = simplex_vertices.iter().map(Point::from).collect();
    match insphere(&simplex_points, Point::from(&test_vertex)) {
        Ok(standard_method_3d) => match insphere_lifted(&simplex_points, Point::from(&test_vertex))
        {
            Ok(matrix_method_3d) => {
                println!("Determinant-based method result: {standard_method_3d:?}");
                println!("Matrix-based method result: {matrix_method_3d:?}");
            }
            Err(e) => println!("Matrix method error: {e}"),
        },
        Err(e) => println!("Standard method error: {e}"),
    }
}

fn test_boundary_vertex_case(simplex_vertices: &[Vertex<f64, i32, 3>]) {
    println!();
    println!("Testing boundary vertex (vertex1):");
    let vertex1 = simplex_vertices[0];
    let simplex_points: Vec<Point<f64, 3>> = simplex_vertices.iter().map(Point::from).collect();
    match insphere(&simplex_points, Point::from(&vertex1)) {
        Ok(standard_vertex) => match insphere_lifted(&simplex_points, Point::from(&vertex1)) {
            Ok(matrix_vertex) => {
                println!("Determinant-based method for vertex1: {standard_vertex:?}");
                println!("Matrix-based method for vertex1: {matrix_vertex:?}");
            }
            Err(e) => println!("Matrix method error for vertex1: {e}"),
        },
        Err(e) => {
            println!("Standard method error for vertex1: {e}");
        }
    }
}

// Type alias to simplify complex return type
type Setup3DResult = (Vec<Vertex<f64, i32, 3>>, [f64; 3], Vertex<f64, i32, 3>);

/// Set up the 3D test matrix data
fn setup_3d_matrix_test() -> Setup3DResult {
    println!("\n=============================================");
    println!("3D Matrix Method Analysis - Step by Step");
    println!("=============================================");

    // Create the 3D simplex: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    let vertex1: Vertex<f64, i32, 3> = vertex!([0.0, 0.0, 0.0], 1);
    let vertex2 = vertex!([1.0, 0.0, 0.0], 1);
    let vertex3 = vertex!([0.0, 1.0, 0.0], 1);
    let vertex4 = vertex!([0.0, 0.0, 1.0], 2);
    let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];

    println!("3D Simplex vertices:");
    println!("  v0: (0, 0, 0)");
    println!("  v1: (1, 0, 0)");
    println!("  v2: (0, 1, 0)");
    println!("  v3: (0, 0, 1)");
    println!();

    // Test point
    let test_point = [0.9, 0.9, 0.9];
    let test_vertex = vertex!(test_point, 3);

    // Get reference vertex (first vertex)
    let ref_coords = [0.0, 0.0, 0.0];
    println!("Reference vertex (v0): {ref_coords:?}");
    println!();

    (simplex_vertices, test_point, test_vertex)
}

/// Build and analyze the matrix for the 3D test
fn build_and_analyze_matrix(simplex_vertices: &[Vertex<f64, i32, 3>]) -> (f64, bool) {
    // Manually build the matrix as in the matrix method
    let mut matrix = delaunay::geometry::matrix::Matrix::zeros(4, 4); // D+1 x D+1 for D=3

    println!("Building matrix rows:");

    // Row 0: v1 - v0 = (1,0,0) - (0,0,0) = (1,0,0), ||v1-v0||² = 1
    let v1_rel = [1.0, 0.0, 0.0];
    let v1_norm2 = 1.0;
    matrix[(0, 0)] = v1_rel[0];
    matrix[(0, 1)] = v1_rel[1];
    matrix[(0, 2)] = v1_rel[2];
    matrix[(0, 3)] = v1_norm2;
    println!(
        "  Row 0 (v1-v0): [{}, {}, {}, {}]",
        v1_rel[0], v1_rel[1], v1_rel[2], v1_norm2
    );

    // Row 1: v2 - v0 = (0,1,0) - (0,0,0) = (0,1,0), ||v2-v0||² = 1
    let v2_rel = [0.0, 1.0, 0.0];
    let v2_norm2 = 1.0;
    matrix[(1, 0)] = v2_rel[0];
    matrix[(1, 1)] = v2_rel[1];
    matrix[(1, 2)] = v2_rel[2];
    matrix[(1, 3)] = v2_norm2;
    println!(
        "  Row 1 (v2-v0): [{}, {}, {}, {}]",
        v2_rel[0], v2_rel[1], v2_rel[2], v2_norm2
    );

    // Row 2: v3 - v0 = (0,0,1) - (0,0,0) = (0,0,1), ||v3-v0||² = 1
    let v3_rel = [0.0, 0.0, 1.0];
    let v3_norm2 = 1.0;
    matrix[(2, 0)] = v3_rel[0];
    matrix[(2, 1)] = v3_rel[1];
    matrix[(2, 2)] = v3_rel[2];
    matrix[(2, 3)] = v3_norm2;
    println!(
        "  Row 2 (v3-v0): [{}, {}, {}, {}]",
        v3_rel[0], v3_rel[1], v3_rel[2], v3_norm2
    );

    // Row 3: test_point - v0 = (0.9,0.9,0.9) - (0,0,0) = (0.9,0.9,0.9), ||test-v0||² = 0.9² + 0.9² + 0.9² = 2.43
    let test_rel = [0.9, 0.9, 0.9];
    let test_norm2 = squared_norm(test_rel);
    matrix[(3, 0)] = test_rel[0];
    matrix[(3, 1)] = test_rel[1];
    matrix[(3, 2)] = test_rel[2];
    matrix[(3, 3)] = test_norm2;
    println!(
        "  Row 3 (test-v0): [{}, {}, {}, {}]",
        test_rel[0], test_rel[1], test_rel[2], test_norm2
    );

    println!();
    println!("Matrix:");
    for i in 0..4 {
        println!(
            "  [{:5.1}, {:5.1}, {:5.1}, {:5.1}]",
            matrix[(i, 0)],
            matrix[(i, 1)],
            matrix[(i, 2)],
            matrix[(i, 3)]
        );
    }

    let det = matrix.determinant();
    println!();
    println!("Determinant: {det:.6}");

    // Check simplex orientation and return determinant and matrix result
    let simplex_points: Vec<Point<f64, 3>> = simplex_vertices.iter().map(Point::from).collect();
    match simplex_orientation(&simplex_points) {
        Ok(is_positive_orientation) => {
            let is_positive = matches!(is_positive_orientation, Orientation::POSITIVE);
            println!(
                "Simplex orientation: {} (positive: {})",
                if is_positive { "POSITIVE" } else { "NEGATIVE" },
                is_positive
            );

            // Apply the sign interpretation from the matrix method
            let eps = 1e-12;
            let matrix_result = if det.abs() <= eps {
                // boundary; treat as inside for comparison, or print explicitly
                true
            } else if is_positive {
                det < 0.0 // For positive orientation, negative det means inside
            } else {
                det > 0.0 // For negative orientation, positive det means inside
            };

            println!(
                "Matrix method interpretation: det {} 0.0 means {}",
                if det < 0.0 { "<" } else { ">" },
                if matrix_result { "INSIDE" } else { "OUTSIDE" }
            );
            println!(
                "Matrix method result: {}",
                if matrix_result { "INSIDE" } else { "OUTSIDE" }
            );

            (det, matrix_result)
        }
        Err(e) => {
            println!("Error determining simplex orientation: {e}");
            (det, false)
        }
    }
}

/// Compare methods with geometry for the 3D test
fn compare_methods_with_geometry(
    simplex_vertices: &[Vertex<f64, i32, 3>],
    test_point: [f64; 3],
    test_vertex: Vertex<f64, i32, 3>,
) -> Option<(f64, f64, bool, InSphere, InSphere)> {
    let simplex_points: Vec<Point<f64, 3>> = simplex_vertices.iter().map(Point::from).collect();
    match circumcenter(&simplex_points) {
        Ok(circumcenter) => {
            match circumradius(&simplex_points) {
                Ok(circumradius) => {
                    let distance_to_test = na::distance(
                        &na::Point::<f64, 3>::from(circumcenter.to_array()),
                        &na::Point::<f64, 3>::from(test_point),
                    );

                    println!();
                    println!("Geometric verification:");
                    println!("  Circumcenter: {:?}", circumcenter.to_array());
                    println!("  Circumradius: {circumradius:.6}");
                    println!("  Distance to test point: {distance_to_test:.6}");
                    println!(
                        "  Geometric truth (distance < radius): {}",
                        (distance_to_test - circumradius).abs() < f64::EPSILON
                            || distance_to_test < circumradius
                    );

                    // Compare with both methods
                    match insphere(&simplex_points, Point::from(&test_vertex)) {
                        Ok(standard_result) => {
                            match insphere_lifted(&simplex_points, Point::from(&test_vertex)) {
                                Ok(matrix_method_result) => Some((
                                    distance_to_test,
                                    circumradius,
                                    (distance_to_test - circumradius).abs() < f64::EPSILON
                                        || distance_to_test < circumradius,
                                    standard_result,
                                    matrix_method_result,
                                )),
                                Err(e) => {
                                    println!("Matrix method error: {e}");
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            println!("Standard method error: {e}");
                            None
                        }
                    }
                }
                Err(e) => {
                    println!("Error calculating circumradius: {e}");
                    None
                }
            }
        }
        Err(e) => {
            println!("Error calculating circumcenter: {e}");
            None
        }
    }
}

/// Print the comparison results for the 3D test
fn print_method_comparison_results(
    geometric_truth: bool,
    standard_result: InSphere,
    matrix_method_result: InSphere,
) {
    println!();
    println!("Method comparison:");
    println!("  Standard method: {standard_result:?}");
    println!("  Matrix method: {matrix_method_result:?}");
    println!("  Geometric truth: {geometric_truth}");

    println!();
    let standard_inside = matches!(standard_result, InSphere::INSIDE | InSphere::BOUNDARY);
    if standard_inside == geometric_truth {
        println!("✓ Standard method matches geometric truth");
    } else {
        println!("✗ Standard method disagrees with geometric truth");
    }

    let matrix_inside = matches!(matrix_method_result, InSphere::INSIDE | InSphere::BOUNDARY);
    let matrix_agrees = matrix_inside == geometric_truth;
    let methods_agree = standard_inside == matrix_inside;

    if matrix_agrees {
        println!("✓ Matrix method matches geometric truth");
    } else {
        println!("✗ Matrix method disagrees with geometric truth");
        println!("  NOTE: This disagreement is expected for this simplex geometry");
        println!("        due to the matrix method's inverted sign convention.");
    }

    println!();
    if methods_agree {
        println!("✓ Both methods agree with each other");
    } else {
        println!("⚠ Methods disagree (expected for this matrix formulation)");
        println!("  The matrix method uses coordinates relative to the first vertex,");
        println!("  which produces an inverted sign convention compared to the standard method.");
        println!("  Both methods are mathematically correct but use different interpretations.");
    }
}

/// Main function for 3D matrix analysis - orchestrates all the smaller functions
fn test_3d_matrix_analysis() {
    let (simplex_vertices, test_point, test_vertex) = setup_3d_matrix_test();
    build_and_analyze_matrix(&simplex_vertices);

    if let Some((
        _distance_to_test,
        _circumradius,
        geometric_truth,
        standard_result,
        matrix_method_result,
    )) = compare_methods_with_geometry(&simplex_vertices, test_point, test_vertex)
    {
        print_method_comparison_results(geometric_truth, standard_result, matrix_method_result);
    }
}

/// Debug 3D circumsphere properties analysis
fn debug_3d_circumsphere_properties() {
    println!("=== 3D Unit Tetrahedron Analysis ===");

    // Unit tetrahedron: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    let vertex1: Vertex<f64, i32, 3> = vertex!([0.0, 0.0, 0.0], 1);
    let vertex2 = vertex!([1.0, 0.0, 0.0], 1);
    let vertex3 = vertex!([0.0, 1.0, 0.0], 1);
    let vertex4 = vertex!([0.0, 0.0, 1.0], 2);
    let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];

    let simplex_points: Vec<Point<f64, 3>> = simplex_vertices.iter().map(Point::from).collect();
    let center = circumcenter(&simplex_points).unwrap();
    let radius = circumradius(&simplex_points).unwrap();

    println!("Circumcenter: {:?}", center.to_array());
    println!("Circumradius: {radius}");

    // Test the point (0.9, 0.9, 0.9)
    let distance_to_center = na::distance(
        &na::Point::<f64, 3>::from(center.to_array()),
        &na::Point::<f64, 3>::from([0.9, 0.9, 0.9]),
    );
    println!("Point (0.9, 0.9, 0.9) distance to circumcenter: {distance_to_center}");
    println!(
        "Is point inside circumsphere (distance < radius)? {}",
        distance_to_center < radius
    );

    let test_vertex: Vertex<f64, i32, 3> = vertex!([0.9, 0.9, 0.9], 4);

    let standard_result = insphere_distance(&simplex_points, Point::from(&test_vertex)).unwrap();
    let matrix_result = insphere_lifted(&simplex_points, Point::from(&test_vertex)).unwrap();

    println!("Standard method result: {standard_result:?}");
    println!("Matrix method result: {matrix_result:?}");
}

/// Debug 4D circumsphere properties analysis
fn debug_4d_circumsphere_properties() {
    println!("\n=== 4D Symmetric Simplex Analysis ===");

    // Regular 4D simplex with vertices forming a specific pattern
    let vertex1: Vertex<f64, Option<()>, 4> = vertex!([1.0, 1.0, 1.0, 1.0]);
    let vertex2 = vertex!([1.0, -1.0, -1.0, -1.0]);
    let vertex3 = vertex!([-1.0, 1.0, -1.0, -1.0]);
    let vertex4 = vertex!([-1.0, -1.0, 1.0, -1.0]);
    let vertex5 = vertex!([-1.0, -1.0, -1.0, 1.0]);
    let simplex_vertices_4d = vec![vertex1, vertex2, vertex3, vertex4, vertex5];

    let simplex_points_4d: Vec<Point<f64, 4>> =
        simplex_vertices_4d.iter().map(Point::from).collect();
    let center_4d = circumcenter(&simplex_points_4d).unwrap();
    let radius_4d = circumradius(&simplex_points_4d).unwrap();

    println!("4D Circumcenter: {:?}", center_4d.to_array());
    println!("4D Circumradius: {radius_4d}");

    // Test the origin (0, 0, 0, 0)
    let distance_to_center_4d = na::distance(
        &na::Point::<f64, 4>::from(center_4d.to_array()),
        &na::Point::<f64, 4>::from([0.0, 0.0, 0.0, 0.0]),
    );
    println!("Origin distance to circumcenter: {distance_to_center_4d}");
    println!(
        "Is origin inside circumsphere (distance < radius)? {}",
        distance_to_center_4d < radius_4d
    );

    let origin_vertex: Vertex<f64, Option<()>, 4> = vertex!([0.0, 0.0, 0.0, 0.0]);

    let standard_result_4d =
        insphere_distance(&simplex_points_4d, Point::from(&origin_vertex)).unwrap();
    let matrix_result_4d =
        insphere_lifted(&simplex_points_4d, Point::from(&origin_vertex)).unwrap();

    println!("Standard method result for origin: {standard_result_4d:?}");
    println!("Matrix method result for origin: {matrix_result_4d:?}");
}

/// Compare results between standard and matrix methods
fn compare_circumsphere_methods() {
    println!("\n=== Comparing Circumsphere Methods ===");

    // Compare results between standard and matrix methods
    let vertex1: Vertex<f64, Option<()>, 2> = vertex!([0.0, 0.0]);
    let vertex2 = vertex!([1.0, 0.0]);
    let vertex3 = vertex!([0.0, 1.0]);
    let simplex_vertices = [vertex1, vertex2, vertex3];

    // Test various points
    let test_points = [
        Point::new([0.1, 0.1]),   // Should be inside
        Point::new([0.5, 0.5]),   // Circumcenter region
        Point::new([10.0, 10.0]), // Far outside
        Point::new([0.25, 0.25]), // Inside
        Point::new([2.0, 2.0]),   // Outside
    ];

    for (i, point) in test_points.iter().enumerate() {
        let test_vertex: Vertex<f64, Option<()>, 2> = vertex!(point.to_array());
        let simplex_points: Vec<Point<f64, 2>> = simplex_vertices.iter().map(Point::from).collect();

        let standard_result =
            insphere_distance(&simplex_points, Point::from(&test_vertex)).unwrap();
        let matrix_result = insphere_lifted(&simplex_points, Point::from(&test_vertex)).unwrap();

        println!(
            "Point {i}: {:?} -> Standard: {:?}, Matrix: {:?}",
            point.to_array(),
            standard_result,
            matrix_result
        );
    }
}

/// Run all basic dimensional tests and orientation tests
fn run_all_basic_tests() {
    println!("Running all basic tests...");
    println!();

    run_tests!(
        test_2d_circumsphere,
        test_3d_circumsphere,
        test_4d_circumsphere,
        test_all_orientations
    );

    println!("All basic tests completed!");
}

/// Run all debug tests
fn run_all_debug_tests() {
    println!("Running all debug tests...");
    println!();

    run_tests!(
        test_3d_simplex_analysis,
        test_3d_matrix_analysis,
        debug_3d_circumsphere_properties,
        debug_4d_circumsphere_properties,
        compare_circumsphere_methods,
        demonstrate_orientation_impact_on_circumsphere,
        test_circumsphere_containment,
        test_4d_circumsphere_methods
    );

    println!("All debug tests completed!");
}

/// Run all tests and all debug functions
fn run_everything() {
    println!("Running everything...");
    println!();

    // Run all basic tests
    run_all_basic_tests();

    println!();
    println!("{}", "=".repeat(60));
    println!("Now running debug tests...");
    println!("{}", "=".repeat(60));
    println!();

    // Run all debug tests
    run_all_debug_tests();

    println!();
    println!("Everything completed successfully!");
}

/// Test a single specific 2D point against triangle circumsphere
fn test_single_2d_point() {
    println!("Testing single 2D point against triangle circumsphere");
    println!("=====================================================");

    // Create a unit triangle: (0,0), (1,0), (0,1)
    let vertices = vec![
        vertex!([0.0, 0.0], 0),
        vertex!([1.0, 0.0], 1),
        vertex!([0.0, 1.0], 2),
    ];

    // Calculate circumcenter and circumradius
    let vertex_points: Vec<Point<f64, 2>> = vertices.iter().map(Point::from).collect();
    match (circumcenter(&vertex_points), circumradius(&vertex_points)) {
        (Ok(center), Ok(radius)) => {
            println!("Triangle vertices:");
            for (i, vertex) in vertices.iter().enumerate() {
                let coords: [f64; 2] = vertex.into();
                println!("  v{i}: {coords:?}");
            }
            println!();
            println!("Circumcenter: {:?}", center.to_array());
            println!("Circumradius: {radius:.6}");
            println!();

            // Test a specific interesting point: (0.3, 0.3) - should be inside
            test_2d_point(
                &vertices,
                [0.3, 0.3],
                "test_point",
                &center.to_array(),
                radius,
            );
        }
        (Err(e), _) => println!("Error calculating circumcenter: {e}"),
        (_, Err(e)) => println!("Error calculating circumradius: {e}"),
    }

    println!("Single 2D point test completed.\n");
}

/// Test a single specific 3D point against tetrahedron circumsphere
fn test_single_3d_point() {
    println!("Testing single 3D point against tetrahedron circumsphere");
    println!("========================================================");

    // Create a unit tetrahedron: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0], 0),
        vertex!([1.0, 0.0, 0.0], 1),
        vertex!([0.0, 1.0, 0.0], 2),
        vertex!([0.0, 0.0, 1.0], 3),
    ];

    // Calculate circumcenter and circumradius
    let vertex_points: Vec<Point<f64, 3>> = vertices.iter().map(Point::from).collect();
    match (circumcenter(&vertex_points), circumradius(&vertex_points)) {
        (Ok(center), Ok(radius)) => {
            println!("Tetrahedron vertices:");
            for (i, vertex) in vertices.iter().enumerate() {
                let coords: [f64; 3] = vertex.into();
                println!("  v{i}: {coords:?}");
            }
            println!();
            println!("Circumcenter: {:?}", center.to_array());
            println!("Circumradius: {radius:.6}");
            println!();

            // Test a specific interesting point: (0.4, 0.4, 0.4) - should be inside
            test_3d_point(
                &vertices,
                [0.4, 0.4, 0.4],
                "test_point",
                &center.to_array(),
                radius,
            );
        }
        (Err(e), _) => println!("Error calculating circumcenter: {e}"),
        (_, Err(e)) => println!("Error calculating circumradius: {e}"),
    }

    println!("Single 3D point test completed.\n");
}

/// Test a single specific 4D point against 4D simplex circumsphere
fn test_single_4d_point() {
    println!("Testing single 4D point against 4D simplex circumsphere");
    println!("=======================================================");

    // Create a unit 4-simplex: vertices at origin and unit vectors along each axis
    let vertices = vec![
        vertex!([0.0, 0.0, 0.0, 0.0], 0),
        vertex!([1.0, 0.0, 0.0, 0.0], 1),
        vertex!([0.0, 1.0, 0.0, 0.0], 2),
        vertex!([0.0, 0.0, 1.0, 0.0], 3),
        vertex!([0.0, 0.0, 0.0, 1.0], 4),
    ];

    // Calculate circumcenter and circumradius
    let vertex_points: Vec<Point<f64, 4>> = vertices.iter().map(Point::from).collect();
    match (circumcenter(&vertex_points), circumradius(&vertex_points)) {
        (Ok(center), Ok(radius)) => {
            println!("4D simplex vertices:");
            for (i, vertex) in vertices.iter().enumerate() {
                let coords: [f64; 4] = vertex.into();
                println!("  v{i}: {coords:?}");
            }
            println!();
            println!("Circumcenter: {:?}", center.to_array());
            println!("Circumradius: {radius:.6}");
            println!();

            // Test a specific interesting point: (0.3, 0.3, 0.3, 0.3) - should be inside
            test_4d_point(
                &vertices,
                [0.3, 0.3, 0.3, 0.3],
                "test_point",
                &center.to_array(),
                radius,
            );
        }
        (Err(e), _) => println!("Error calculating circumcenter: {e}"),
        (_, Err(e)) => println!("Error calculating circumradius: {e}"),
    }

    println!("Single 4D point test completed.\n");
}

/// Test single specific points in all dimensions
fn test_all_single_points() {
    println!("Testing single specific points in all dimensions");
    println!("================================================");
    println!();

    run_tests!(
        test_single_2d_point,
        test_single_3d_point,
        test_single_4d_point
    );

    println!("All single point tests completed!");
}
