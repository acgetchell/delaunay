//! Property-based tests for serialization/deserialization roundtrips.
//!
//! This module uses proptest to verify that serialization and deserialization
//! preserve all important properties of triangulation structures, including:
//! - Triangulation structure preservation (vertices, cells, neighbors)
//! - Cell and vertex data preservation
//! - Triangulation validity after roundtrip
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

use approx::relative_eq;
use delaunay::core::triangulation_data_structure::Tds;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use proptest::prelude::*;

/// Check if two points are approximately equal (coordinate-wise)
/// Uses relative epsilon comparison suitable for JSON serialization roundtrips
fn points_approx_equal<const D: usize>(p1: &Point<f64, D>, p2: &Point<f64, D>) -> bool {
    p1.coords()
        .iter()
        .zip(p2.coords().iter())
        .all(|(a, b)| relative_eq!(a, b, epsilon = 1e-14, max_relative = 1e-14))
}

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

// =============================================================================
// DIMENSIONAL TEST GENERATION MACROS
// =============================================================================

/// Macro to generate serialization property tests for a given dimension
macro_rules! test_serialization_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {
            proptest! {
                /// Property: Triangulation structure preserved after JSON roundtrip
                #[test]
                fn [<prop_triangulation_json_roundtrip_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        // Serialize to JSON
                        let json = serde_json::to_string(&tds).expect("Serialization failed");

                        // Deserialize from JSON
                        let deserialized: Tds<f64, Option<()>, Option<()>, $dim> =
                            serde_json::from_str(&json).expect("Deserialization failed");

                        // Verify structure preservation
                        prop_assert_eq!(
                            deserialized.number_of_vertices(),
                            tds.number_of_vertices(),
                            "{}D vertex count should be preserved",
                            $dim
                        );
                        prop_assert_eq!(
                            deserialized.number_of_cells(),
                            tds.number_of_cells(),
                            "{}D cell count should be preserved",
                            $dim
                        );
                        prop_assert_eq!(
                            deserialized.dim(),
                            tds.dim(),
                            "{}D dimension should be preserved",
                            $dim
                        );
                    }
                }

                /// Property: Deserialized triangulation remains valid
                #[test]
                fn [<prop_deserialized_triangulation_valid_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        if tds.is_valid().is_ok() {
                            // Serialize and deserialize
                            let json = serde_json::to_string(&tds).expect("Serialization failed");
                            let deserialized: Tds<f64, Option<()>, Option<()>, $dim> =
                                serde_json::from_str(&json).expect("Deserialization failed");

                            // Deserialized triangulation should also be valid
                            prop_assert!(
                                deserialized.is_valid().is_ok(),
                                "{}D deserialized triangulation should be valid: {:?}",
                                $dim,
                                deserialized.is_valid().err()
                            );
                        }
                    }
                }

                /// Property: Vertex coordinates preserved after roundtrip
                #[test]
                fn [<prop_vertex_coordinates_preserved_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        // Filter: Skip minimal/degenerate configurations
                        // Need more than minimal simplex (D+1) to have meaningful serialization test
                        prop_assume!(tds.number_of_vertices() > $dim + 1);
                        // Also skip invalid TDS (can happen with nearly-degenerate geometries)
                        prop_assume!(tds.is_valid().is_ok());

                        // Collect original vertex points
                        let original_points: Vec<_> = tds.vertices()
                            .map(|(_, v)| *v.point())
                            .collect();

                        // Serialize and deserialize
                        let json = serde_json::to_string(&tds).expect("Serialization failed");
                        let deserialized: Tds<f64, Option<()>, Option<()>, $dim> =
                            serde_json::from_str(&json).expect("Deserialization failed");

                        // Collect deserialized vertex points
                        let deserialized_points: Vec<_> = deserialized.vertices()
                            .map(|(_, v)| *v.point())
                            .collect();

                        // Compare counts
                        prop_assert_eq!(
                            deserialized_points.len(),
                            original_points.len(),
                            "{}D vertex count mismatch",
                            $dim
                        );

                        // Check that all original points exist in deserialized
                        // (order might differ, so we check set equality)
                        // Use approximate comparison due to JSON floating-point precision limits
                        for orig_point in &original_points {
                            let found = deserialized_points.iter()
                                .any(|deser_point| points_approx_equal(orig_point, deser_point));
                            prop_assert!(
                                found,
                                "{}D vertex point {:?} not found after roundtrip (within tolerance)",
                                $dim,
                                orig_point
                            );
                        }
                    }
                }

                /// Property: Neighbor relationships preserved after roundtrip
                #[test]
                fn [<prop_neighbor_relationships_preserved_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        // Ensure neighbor relationships are fully assigned before comparison.
                        tds.assign_neighbors()
                            .expect("assign_neighbors should succeed for constructed Tds");

                        // Count original neighbor relationships after explicit assignment
                        let mut original_neighbor_count = 0;
                        for (_key, cell) in tds.cells() {
                            if let Some(neighbors) = cell.neighbors() {
                                original_neighbor_count += neighbors.iter().flatten().count();
                            }
                        }

                        // Serialize and deserialize
                        let json = serde_json::to_string(&tds).expect("Serialization failed");
                        let deserialized: Tds<f64, Option<()>, Option<()>, $dim> =
                            serde_json::from_str(&json).expect("Deserialization failed");

                        // Count deserialized neighbor relationships
                        let mut deserialized_neighbor_count = 0;
                        for (_key, cell) in deserialized.cells() {
                            if let Some(neighbors) = cell.neighbors() {
                                deserialized_neighbor_count += neighbors.iter().flatten().count();
                            }
                        }

                        // Neighbor counts should match
                        prop_assert_eq!(
                            deserialized_neighbor_count,
                            original_neighbor_count,
                            "{}D neighbor relationship count should be preserved",
                            $dim
                        );
                    }
                }
            }
        }
    };
}

// Generate tests for dimensions 2-5
// Parameters: dimension, min_vertices, max_vertices
test_serialization_properties!(2, 4, 10);
test_serialization_properties!(3, 5, 12);
test_serialization_properties!(4, 6, 14);
test_serialization_properties!(5, 7, 16);

// Debug regression test for neighbor preservation in 2D
// Uses the minimal failing case previously captured by proptest to
// inspect neighbor counts before and after JSON roundtrip.
#[test]
#[ignore = "debug-only: investigate neighbor preservation regression"]
fn debug_neighbor_preservation_2d_regression() {
    use delaunay::core::triangulation_data_structure::Tds;
    use delaunay::core::vertex::Vertex;
    use delaunay::geometry::point::Point;
    use delaunay::geometry::traits::coordinate::Coordinate;

    // Regression case derived from an earlier proptest failure (inlined for stability)
    let points: Vec<Point<f64, 2>> = vec![
        Point::new([0.0, 28.167_639_636_534_61]),
        Point::new([-81.778_725_768_602_44, -77.483_132_207_214_21]),
        Point::new([-51.191_093_688_796_81, 72.419_431_583_746_07]),
        Point::new([93.267_946_397_941_36, -79.624_891_695_467_43]),
        Point::new([38.034_505_600_190_656, 12.366_848_224_818_94]),
        Point::new([-84.042_855_780_040_57, 38.291_427_688_983]),
        Point::new([0.0, 0.0]),
    ];

    let vertices: Vec<Vertex<f64, Option<()>, 2>> = Vertex::from_points(&points);
    let mut tds = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices)
        .expect("regression case should construct a valid Tds");
    tds.assign_neighbors()
        .expect("assign_neighbors should succeed for constructed Tds");

    println!(
        "Original: dim={} cells={} vertices={}",
        tds.dim(),
        tds.number_of_cells(),
        tds.number_of_vertices()
    );

    let mut original_neighbor_count = 0;
    for (key, cell) in tds.cells() {
        if let Some(neighbors) = cell.neighbors() {
            let count = neighbors.iter().flatten().count();
            println!("  cell {key:?} has {count} neighbors");
            original_neighbor_count += count;
        } else {
            println!("  cell {key:?} has no neighbors");
        }
    }
    println!("Original neighbor count: {original_neighbor_count}");

    let json = serde_json::to_string(&tds).expect("Serialization failed");
    let deserialized: Tds<f64, Option<()>, Option<()>, 2> =
        serde_json::from_str(&json).expect("Deserialization failed");

    println!(
        "Deserialized: dim={} cells={} vertices={}",
        deserialized.dim(),
        deserialized.number_of_cells(),
        deserialized.number_of_vertices()
    );

    let mut deserialized_neighbor_count = 0;
    for (key, cell) in deserialized.cells() {
        if let Some(neighbors) = cell.neighbors() {
            let count = neighbors.iter().flatten().count();
            println!("  cell {key:?} has {count} neighbors");
            deserialized_neighbor_count += count;
        } else {
            println!("  cell {key:?} has no neighbors");
        }
    }
    println!(
        "Deserialized neighbor count: {deserialized_neighbor_count} (original {original_neighbor_count})"
    );
}
