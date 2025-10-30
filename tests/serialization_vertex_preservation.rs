//! Integration test for vertex preservation during serialization/deserialization.
//!
//! This test investigates whether vertices are properly preserved through the full lifecycle:
//! 1. Input vertices (possibly with duplicates)
//! 2. Triangulation construction (may merge duplicates)
//! 3. Serialization to JSON
//! 4. Deserialization from JSON
//!
//! The test helps determine if vertex loss is expected behavior (duplicate removal)
//! or a bug in serialization/deserialization.

use approx::relative_eq;
use delaunay::core::Tds;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use std::collections::HashSet;

/// Check if two points are approximately equal (coordinate-wise)
/// Uses relative epsilon comparison suitable for JSON serialization roundtrips
fn points_approx_equal<const D: usize>(p1: &Point<f64, D>, p2: &Point<f64, D>) -> bool {
    p1.coords()
        .iter()
        .zip(p2.coords().iter())
        .all(|(a, b)| relative_eq!(a, b, epsilon = 1e-14, max_relative = 1e-14))
}

/// Test vertex preservation with duplicate coordinates
#[test]
fn test_vertex_preservation_with_duplicates_3d() {
    // Create vertices with duplicate coordinates but different data
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
        // Duplicate coordinate
        Point::new([0.0, 0.0, 0.0]),
    ];
    let vertices = Vertex::<f64, Option<()>, 3>::from_points(points);

    let input_coords: HashSet<_> = vertices.iter().map(|v| *v.point()).collect();
    println!("Input vertices: {}", vertices.len());
    println!("Unique input coordinates: {}", input_coords.len());

    // Construct triangulation
    let tds =
        Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).expect("Tds construction failed");

    let tds_vertex_count = tds.vertices().count();
    let tds_coords: HashSet<_> = tds.vertices().map(|(_, v)| *v.point()).collect();
    println!("Vertices after Tds construction: {tds_vertex_count}");
    println!(
        "Unique coordinates after Tds construction: {}",
        tds_coords.len()
    );

    // Serialize
    let json = serde_json::to_string(&tds).expect("Serialization failed");
    println!("JSON size: {} bytes", json.len());

    // Deserialize
    let deserialized: Tds<f64, Option<()>, Option<()>, 3> =
        serde_json::from_str(&json).expect("Deserialization failed");

    let deser_vertex_count = deserialized.vertices().count();
    let deser_coords: HashSet<_> = deserialized.vertices().map(|(_, v)| *v.point()).collect();
    println!("Vertices after deserialization: {deser_vertex_count}");
    println!(
        "Unique coordinates after deserialization: {}",
        deser_coords.len()
    );

    // Analysis
    let vertices_len = vertices.len();
    let input_coords_len = input_coords.len();
    println!("\n=== Analysis ===");
    println!("Input vertices: {vertices_len}");
    println!("Unique input coords: {input_coords_len}");
    println!("After Tds construction: {tds_vertex_count}");
    println!("After serialization roundtrip: {deser_vertex_count}");

    // Check if vertex loss happens during construction or serialization
    if tds_vertex_count < vertices_len {
        println!("\n⚠️  Vertices lost during Tds construction (likely duplicate merging)");
        println!("   Input: {vertices_len} -> After construction: {tds_vertex_count}");
    }

    if deser_vertex_count < tds_vertex_count {
        println!("\n🚨 CRITICAL: Vertices lost during serialization/deserialization!");
        println!("   Before: {tds_vertex_count} -> After: {deser_vertex_count}");
        panic!("Serialization lost vertices: {tds_vertex_count} -> {deser_vertex_count}");
    }

    // Verify coordinate preservation (with tolerance for JSON precision)
    assert_eq!(
        tds_coords.len(),
        deser_coords.len(),
        "Coordinate count mismatch"
    );
    for tds_coord in &tds_coords {
        assert!(
            deser_coords
                .iter()
                .any(|dc| points_approx_equal(tds_coord, dc)),
            "Coordinate {tds_coord:?} not found in deserialized set (within tolerance)"
        );
    }

    println!("\n✅ Serialization preserved all vertices from constructed Tds");
}

/// Test vertex preservation without duplicates (baseline)
#[test]
fn test_vertex_preservation_without_duplicates_3d() {
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
        Point::new([0.5, 0.5, 0.5]),
    ];
    let vertices = Vertex::<f64, Option<()>, 3>::from_points(points);

    println!("Input vertices (no duplicates): {}", vertices.len());

    let tds =
        Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).expect("Tds construction failed");
    let tds_vertex_count = tds.vertices().count();
    println!("Vertices after Tds construction: {tds_vertex_count}");

    let json = serde_json::to_string(&tds).expect("Serialization failed");
    let deserialized: Tds<f64, Option<()>, Option<()>, 3> =
        serde_json::from_str(&json).expect("Deserialization failed");

    let deser_vertex_count = deserialized.vertices().count();
    println!("Vertices after deserialization: {deser_vertex_count}");

    assert_eq!(
        vertices.len(),
        tds_vertex_count,
        "Tds construction should preserve all unique vertices"
    );

    assert_eq!(
        tds_vertex_count, deser_vertex_count,
        "Serialization should preserve all vertices"
    );

    println!("✅ All vertices preserved through construction and serialization");
}

/// Test with many duplicates to stress-test behavior
#[test]
fn test_vertex_preservation_many_duplicates_3d() {
    let base_point = Point::new([0.5, 0.5, 0.5]);
    let mut points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
    ];

    // Add 10 vertices with the same coordinate
    for _ in 0..10 {
        points.push(base_point);
    }

    let vertices = Vertex::<f64, Option<()>, 3>::from_points(points);

    println!("Input vertices (with many duplicates): {}", vertices.len());
    let unique_coords: HashSet<_> = vertices.iter().map(|v| *v.point()).collect();
    let unique_coords_len = unique_coords.len();
    println!("Unique coordinates: {unique_coords_len}");

    let tds =
        Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).expect("Tds construction failed");
    let tds_vertex_count = tds.vertices().count();
    println!("Vertices after Tds construction: {tds_vertex_count}");

    let json = serde_json::to_string(&tds).expect("Serialization failed");
    let deserialized: Tds<f64, Option<()>, Option<()>, 3> =
        serde_json::from_str(&json).expect("Deserialization failed");

    let deser_vertex_count = deserialized.vertices().count();
    println!("Vertices after deserialization: {deser_vertex_count}");
    assert_eq!(
        tds_vertex_count, deser_vertex_count,
        "Serialization must preserve vertex count from constructed Tds"
    );

    println!(
        "✅ Serialization preserved vertices (duplicates merged during construction as expected)"
    );
}

/// Test to verify exact vertex coordinate preservation (not just count)
#[test]
fn test_vertex_coordinate_preservation_3d() {
    let points = vec![
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
    ];
    let vertices = Vertex::<f64, Option<()>, 3>::from_points(points);

    let tds =
        Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices).expect("Tds construction failed");

    // Collect original vertex coordinates
    let original_coords: HashSet<_> = tds.vertices().map(|(_, v)| *v.point()).collect();

    let json = serde_json::to_string(&tds).expect("Serialization failed");
    let deserialized: Tds<f64, Option<()>, Option<()>, 3> =
        serde_json::from_str(&json).expect("Deserialization failed");

    // Collect deserialized vertex coordinates
    let deserialized_coords: HashSet<_> =
        deserialized.vertices().map(|(_, v)| *v.point()).collect();

    // Verify coordinate preservation with tolerance for JSON precision
    assert_eq!(
        original_coords.len(),
        deserialized_coords.len(),
        "Coordinate count mismatch"
    );
    for orig_coord in &original_coords {
        assert!(
            deserialized_coords
                .iter()
                .any(|dc| points_approx_equal(orig_coord, dc)),
            "Coordinate {orig_coord:?} not preserved through serialization (within tolerance)"
        );
    }

    println!("✅ Vertex coordinates exactly preserved");
}
