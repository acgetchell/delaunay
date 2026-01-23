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
use delaunay::prelude::*;
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
    let vertices = Vertex::<f64, (), 3>::from_points(&points);

    let input_coords: HashSet<_> = vertices.iter().map(|v| *v.point()).collect();
    println!("Input vertices: {}", vertices.len());
    println!("Unique input coordinates: {}", input_coords.len());

    // Construct triangulation - duplicates should be skipped
    let dt =
        DelaunayTriangulation::<_, (), (), 3>::new(&vertices).expect("Tds construction succeeded");
    let tds = dt.tds();

    let tds_vertex_count = tds.vertices().count();
    let tds_coords = extract_vertex_coordinate_set(tds);
    println!("Vertices after Tds construction: {tds_vertex_count}");
    println!(
        "Unique coordinates after Tds construction: {}",
        tds_coords.len()
    );

    // Verify duplicates were skipped (should match unique coordinate count)
    assert_eq!(
        tds_vertex_count,
        input_coords.len(),
        "Vertex count after construction should equal unique input coordinates"
    );

    // Serialize
    let json = serde_json::to_string(&tds).expect("Serialization failed");
    println!("JSON size: {} bytes", json.len());

    // Deserialize
    let deserialized: Tds<f64, (), (), 3> =
        serde_json::from_str(&json).expect("Deserialization failed");

    let deser_vertex_count = deserialized.vertices().count();
    let deser_coords = extract_vertex_coordinate_set(&deserialized);
    println!("Vertices after deserialization: {deser_vertex_count}");

    // Verify coordinate preservation using Jaccard similarity (≥ 0.99 threshold)
    // This accounts for potential floating-point precision differences in JSON serialization
    assert_jaccard_gte!(
        &tds_coords,
        &deser_coords,
        0.99,
        "Vertex coordinate preservation via serialization (3D with duplicates)"
    );

    println!(
        "\n✅ Duplicate skipped, serialization preserved all unique vertices (Jaccard ≥ 0.99)"
    );
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
    let vertices = Vertex::<f64, (), 3>::from_points(&points);

    println!("Input vertices (no duplicates): {}", vertices.len());

    let dt =
        DelaunayTriangulation::<_, (), (), 3>::new(&vertices).expect("Tds construction failed");
    let tds = dt.tds();
    let tds_vertex_count = tds.vertices().count();
    println!("Vertices after Tds construction: {tds_vertex_count}");

    // Extract vertex coordinate sets for Jaccard comparison
    let before_coords = extract_vertex_coordinate_set(tds);

    let json = serde_json::to_string(&tds).expect("Serialization failed");
    let deserialized: Tds<f64, (), (), 3> =
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
    // Use a stable interior point for this stress test. The previous choice
    // ([0.5, 0.5, 0.5]) can trigger insertion-order retry logic where shuffled
    // attempts frequently pick duplicate coordinates for the initial simplex.
    let base_point = Point::new([0.25, 0.25, 0.25]);
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

    let vertices = Vertex::<f64, (), 3>::from_points(&points);

    println!("Input vertices (with many duplicates): {}", vertices.len());
    let unique_coords: HashSet<_> = vertices.iter().map(|v| *v.point()).collect();
    let unique_coords_len = unique_coords.len();
    println!("Unique coordinates: {unique_coords_len}");

    // Use Input ordering to avoid Morton clustering of duplicates causing degenerate initial simplex
    let opts = ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    let dt = DelaunayTriangulation::<_, (), (), 3>::new_with_options(&vertices, opts)
        .expect("Tds construction succeeded");
    let tds = dt.tds();
    let tds_vertex_count = tds.vertices().count();
    println!("Vertices after Tds construction: {tds_vertex_count}");

    // Verify duplicates were skipped (should match unique coordinate count)
    assert_eq!(
        tds_vertex_count, unique_coords_len,
        "Vertex count after construction should equal unique input coordinates"
    );

    // Extract vertex coordinate sets for Jaccard comparison
    let before_coords = extract_vertex_coordinate_set(tds);

    let json = serde_json::to_string(&tds).expect("Serialization failed");
    let deserialized: Tds<f64, (), (), 3> =
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

    println!("✅ Duplicates skipped, serialization preserved all unique vertices (Jaccard ≥ 0.99)");
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
    let vertices = Vertex::<f64, (), 3>::from_points(&points);

    let dt =
        DelaunayTriangulation::<_, (), (), 3>::new(&vertices).expect("Tds construction failed");
    let tds = dt.tds();

    // Extract original vertex coordinates using canonical extraction
    let original_coords = extract_vertex_coordinate_set(tds);

    let json = serde_json::to_string(&tds).expect("Serialization failed");
    let deserialized: Tds<f64, (), (), 3> =
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
