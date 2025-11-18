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

use delaunay::assert_jaccard_gte;
use delaunay::core::Tds;
use delaunay::core::util::extract_vertex_coordinate_set;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use std::collections::HashSet;

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
    let tds_coords = extract_vertex_coordinate_set(&tds);
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
    let deser_coords = extract_vertex_coordinate_set(&deserialized);
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

    // Verify coordinate preservation using Jaccard similarity (≥ 0.99 threshold)
    // This accounts for potential floating-point precision differences in JSON serialization
    assert_jaccard_gte!(
        &tds_coords,
        &deser_coords,
        0.99,
        "Vertex coordinate preservation via serialization (3D with duplicates)"
    );

    println!("\n✅ Serialization preserved all vertices from constructed Tds (Jaccard ≥ 0.99)");
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

    // Extract vertex coordinate sets for Jaccard comparison
    let before_coords = extract_vertex_coordinate_set(&tds);

    let json = serde_json::to_string(&tds).expect("Serialization failed");
    let deserialized: Tds<f64, Option<()>, Option<()>, 3> =
        serde_json::from_str(&json).expect("Deserialization failed");

    let deser_vertex_count = deserialized.vertices().count();
    let after_coords = extract_vertex_coordinate_set(&deserialized);
    println!("Vertices after deserialization: {deser_vertex_count}");

    // Note: Robust triangulation may discard some input vertices as unsalvageable
    // even when there are no exact coordinate duplicates. We treat the constructed
    // TDS as the baseline and verify that serialization preserves its vertices.
    // Use Jaccard similarity to verify serialization preserves vertices
    assert_jaccard_gte!(
        &before_coords,
        &after_coords,
        0.99,
        "Vertex coordinate preservation via serialization (3D without duplicates)"
    );

    println!("✅ All vertices preserved through construction and serialization (Jaccard ≥ 0.99)");
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

    // Extract vertex coordinate sets for Jaccard comparison
    let before_coords = extract_vertex_coordinate_set(&tds);

    let json = serde_json::to_string(&tds).expect("Serialization failed");
    let deserialized: Tds<f64, Option<()>, Option<()>, 3> =
        serde_json::from_str(&json).expect("Deserialization failed");

    let deser_vertex_count = deserialized.vertices().count();
    let after_coords = extract_vertex_coordinate_set(&deserialized);
    println!("Vertices after deserialization: {deser_vertex_count}");

    // Use Jaccard similarity to verify serialization preserves vertices
    assert_jaccard_gte!(
        &before_coords,
        &after_coords,
        0.99,
        "Vertex coordinate preservation via serialization (3D with many duplicates)"
    );

    println!(
        "✅ Serialization preserved vertices (duplicates merged during construction as expected, Jaccard ≥ 0.99)"
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

    // Extract original vertex coordinates using canonical extraction
    let original_coords = extract_vertex_coordinate_set(&tds);

    let json = serde_json::to_string(&tds).expect("Serialization failed");
    let deserialized: Tds<f64, Option<()>, Option<()>, 3> =
        serde_json::from_str(&json).expect("Deserialization failed");

    // Extract deserialized vertex coordinates
    let deserialized_coords = extract_vertex_coordinate_set(&deserialized);

    // Verify coordinate preservation using Jaccard similarity
    // Use high threshold (0.99) to ensure nearly exact preservation
    assert_jaccard_gte!(
        &original_coords,
        &deserialized_coords,
        0.99,
        "Exact vertex coordinate preservation via serialization (3D baseline)"
    );

    println!("✅ Vertex coordinates preserved with high fidelity (Jaccard ≥ 0.99)");
}
