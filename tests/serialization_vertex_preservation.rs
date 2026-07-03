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
use delaunay::prelude::construction::{
    ConstructionOptions, DelaunayTriangulation, InsertionOrderStrategy, TopologyGuarantee,
};
use delaunay::prelude::geometry::*;
use delaunay::prelude::query::extract_vertex_coordinate_set;
use delaunay::prelude::tds::Tds;
use delaunay::try_vertices_from_points;
use std::collections::HashSet;

#[cfg(feature = "diagnostics")]
macro_rules! diag_debug {
    ($($arg:tt)*) => {
        tracing::debug!($($arg)*);
    };
}

#[cfg(not(feature = "diagnostics"))]
macro_rules! diag_debug {
    ($($arg:tt)*) => {};
}

/// Test vertex preservation with duplicate coordinates
#[test]
fn test_vertex_preservation_with_duplicates_3d() {
    // Create vertices with duplicate coordinates but different data
    let points = vec![
        Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        // Duplicate coordinate
        Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
    ];
    let vertices = try_vertices_from_points(&points).expect("finite point coordinates");

    let input_coords: HashSet<_> = vertices.iter().map(|v| *v.point()).collect();
    diag_debug!(
        input_vertices = vertices.len(),
        unique_input_coordinates = input_coords.len(),
        "vertex preservation input"
    );

    // Construct triangulation - duplicates should be skipped
    let dt = DelaunayTriangulation::builder(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .build()
        .expect("Tds construction succeeded");
    let tds = dt.tds();

    let tds_vertex_count = tds.vertices().count();
    let tds_coords = extract_vertex_coordinate_set(tds);
    diag_debug!(
        tds_vertex_count,
        unique_tds_coordinates = tds_coords.len(),
        "vertex preservation after construction"
    );

    // Verify duplicates were skipped (should match unique coordinate count)
    assert_eq!(
        tds_vertex_count,
        input_coords.len(),
        "Vertex count after construction should equal unique input coordinates"
    );

    // Serialize
    let json = serde_json::to_string(&tds).expect("Serialization failed");
    diag_debug!(json_bytes = json.len(), "serialized TDS size");

    // Deserialize
    let deserialized: Tds<(), (), 3> = serde_json::from_str(&json).expect("Deserialization failed");

    let deser_vertex_count = deserialized.vertices().count();
    let deser_coords = extract_vertex_coordinate_set(&deserialized);
    diag_debug!(
        deser_vertex_count,
        unique_deserialized_coordinates = deser_coords.len(),
        "vertex preservation after deserialization"
    );
    assert_eq!(deser_vertex_count, tds_vertex_count);

    // Verify coordinate preservation using Jaccard similarity (≥ 0.99 threshold)
    // This accounts for potential floating-point precision differences in JSON serialization
    assert_jaccard_gte!(
        &tds_coords,
        &deser_coords,
        0.99,
        "Vertex coordinate preservation via serialization (3D with duplicates)"
    );
}

/// Test vertex preservation without duplicates (baseline)
#[test]
fn test_vertex_preservation_without_duplicates_3d() {
    let points = vec![
        Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        Point::try_new([0.5, 0.5, 0.5]).expect("finite point coordinates"),
    ];
    let vertices = try_vertices_from_points(&points).expect("finite point coordinates");

    let dt = DelaunayTriangulation::builder(&vertices)
        .topology_guarantee(TopologyGuarantee::PLManifold)
        .build()
        .expect("Tds construction failed");
    let tds = dt.tds();
    let tds_vertex_count = tds.vertices().count();
    diag_debug!(
        input_vertices = vertices.len(),
        tds_vertex_count,
        "vertex preservation baseline after construction"
    );

    // Extract vertex coordinate sets for Jaccard comparison
    let before_coords = extract_vertex_coordinate_set(tds);

    let json = serde_json::to_string(&tds).expect("Serialization failed");
    let deserialized: Tds<(), (), 3> = serde_json::from_str(&json).expect("Deserialization failed");

    let deser_vertex_count = deserialized.vertices().count();
    let after_coords = extract_vertex_coordinate_set(&deserialized);
    diag_debug!(
        deser_vertex_count,
        unique_deserialized_coordinates = after_coords.len(),
        "vertex preservation baseline after deserialization"
    );
    assert_eq!(deser_vertex_count, tds_vertex_count);

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
}

/// Test with many duplicates to stress-test behavior
#[test]
fn test_vertex_preservation_many_duplicates_3d() {
    // Use a stable interior point for this stress test. The previous choice
    // ([0.5, 0.5, 0.5]) can trigger insertion-order retry logic where shuffled
    // attempts frequently pick duplicate coordinates for the initial simplex.
    let base_point = Point::try_new([0.25, 0.25, 0.25]).expect("finite point coordinates");
    let mut points = vec![
        Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
    ];

    // Add 10 vertices with the same coordinate
    for _ in 0..10 {
        points.push(base_point);
    }

    let vertices = try_vertices_from_points(&points).expect("finite point coordinates");

    let unique_coords: HashSet<_> = vertices.iter().map(|v| *v.point()).collect();
    let unique_coords_len = unique_coords.len();
    diag_debug!(
        input_vertices = vertices.len(),
        unique_input_coordinates = unique_coords_len,
        "many-duplicate vertex preservation input"
    );

    // Use Input ordering to avoid Hilbert dedup collapsing duplicates before the initial simplex
    let opts = ConstructionOptions::default().with_insertion_order(InsertionOrderStrategy::Input);
    let dt = DelaunayTriangulation::builder(&vertices)
        .construction_options(opts)
        .build()
        .expect("Tds construction succeeded");
    let tds = dt.tds();
    let tds_vertex_count = tds.vertices().count();
    diag_debug!(
        tds_vertex_count,
        "many-duplicate vertex preservation after construction"
    );

    // Verify duplicates were skipped (should match unique coordinate count)
    assert_eq!(
        tds_vertex_count, unique_coords_len,
        "Vertex count after construction should equal unique input coordinates"
    );

    // Extract vertex coordinate sets for Jaccard comparison
    let before_coords = extract_vertex_coordinate_set(tds);

    let json = serde_json::to_string(&tds).expect("Serialization failed");
    let deserialized: Tds<(), (), 3> = serde_json::from_str(&json).expect("Deserialization failed");

    let deser_vertex_count = deserialized.vertices().count();
    let after_coords = extract_vertex_coordinate_set(&deserialized);
    diag_debug!(
        deser_vertex_count,
        unique_deserialized_coordinates = after_coords.len(),
        "many-duplicate vertex preservation after deserialization"
    );
    assert_eq!(deser_vertex_count, tds_vertex_count);

    // Use Jaccard similarity to verify serialization preserves vertices
    assert_jaccard_gte!(
        &before_coords,
        &after_coords,
        0.99,
        "Vertex coordinate preservation via serialization (3D with many duplicates)"
    );
}
