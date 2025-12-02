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
use delaunay::prelude::*;
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
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices) {
                        // Serialize to JSON
                        let json = serde_json::to_string(&dt).expect("Serialization failed");

                        // Deserialize from JSON
                        let deserialized: DelaunayTriangulation<_, (), (), $dim> =
                            serde_json::from_str(&json).expect("Deserialization failed");

                        // Verify structure preservation
                        prop_assert_eq!(
                            deserialized.number_of_vertices(),
                            dt.number_of_vertices(),
                            "{}D vertex count should be preserved",
                            $dim
                        );
                        prop_assert_eq!(
                            deserialized.number_of_cells(),
                            dt.number_of_cells(),
                            "{}D cell count should be preserved",
                            $dim
                        );
                        prop_assert_eq!(
                            deserialized.dim(),
                            dt.dim(),
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
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices) {
                        if dt.is_valid().is_ok() {
                            // Serialize and deserialize
                            let json = serde_json::to_string(&dt).expect("Serialization failed");
                            let deserialized: DelaunayTriangulation<_, (), (), $dim> =
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
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices) {
                        // Filter: Skip minimal/degenerate configurations
                        // Need more than minimal simplex (D+1) to have meaningful serialization test
                        prop_assume!(dt.number_of_vertices() > $dim + 1);
                        // Also skip invalid TDS (can happen with nearly-degenerate geometries)
                        prop_assume!(dt.is_valid().is_ok());

                        // Collect original vertex points
                        let original_points: Vec<_> = dt.vertices()
                            .map(|(_, v)| *v.point())
                            .collect();

                        // Serialize and deserialize
                        let json = serde_json::to_string(&dt).expect("Serialization failed");
                        let deserialized: DelaunayTriangulation<_, (), (), $dim> =
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
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices) {
                        // Count original neighbor relationships
                        let mut original_neighbor_count = 0;
                        for (_key, cell) in dt.cells() {
                            if let Some(neighbors) = cell.neighbors() {
                                original_neighbor_count += neighbors.iter().flatten().count();
                            }
                        }

                        // Serialize and deserialize
                        let json = serde_json::to_string(&dt).expect("Serialization failed");
                        let deserialized: DelaunayTriangulation<_, (), (), $dim> =
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
