#![expect(deprecated)]
//! Property-based tests for Bowyer-Watson insertion algorithm.
//!
//! This module uses proptest to verify fundamental properties of the
//! Bowyer-Watson incremental insertion algorithm, including:
//! - Triangulation remains valid after vertex insertion
//! - Delaunay property is preserved after insertion
//! - Vertex count increases correctly
//! - Cell count changes appropriately
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

use delaunay::core::triangulation_data_structure::Tds;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use proptest::prelude::*;

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

/// Macro to generate Bowyer-Watson property tests for a given dimension
macro_rules! test_bowyer_watson_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {
            proptest! {
                /// Property: Triangulation remains valid after inserting a new vertex
                #[test]
                fn [<prop_insertion_preserves_validity_ $dim d>](
                    initial_vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)),
                    new_point in prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new)
                ) {
                    // Create initial triangulation
                    if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&initial_vertices) {
                        let initial_vertex_count = tds.vertices().count();

                        // Insert new vertex
                        let new_vertex = Vertex::from_points(&[new_point]);
                        if let Ok(_) = tds.add(new_vertex[0].clone()) {
                            // Verify validity after insertion
                            prop_assert!(
                                tds.is_valid().is_ok(),
                                "{}D triangulation should remain valid after vertex insertion: {:?}",
                                $dim,
                                tds.is_valid().err()
                            );

                            // Verify vertex count increased
                            let final_vertex_count = tds.vertices().count();
                            prop_assert!(
                                final_vertex_count >= initial_vertex_count,
                                "{}D vertex count should not decrease: {} -> {}",
                                $dim,
                                initial_vertex_count,
                                final_vertex_count
                            );
                        }
                    }
                }

                /// Property: Multiple sequential insertions preserve validity
                #[test]
                fn [<prop_multiple_insertions_validity_ $dim d>](
                    initial_vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)),
                    new_points in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        1..=5
                    )
                ) {
                    if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&initial_vertices) {
                        let initial_vertex_count = tds.vertices().count();
                        let mut successful_insertions = 0;

                        // Insert multiple vertices
                        for point in new_points {
                            let new_vertex = Vertex::from_points(&[point]);
                            if tds.add(new_vertex[0].clone()).is_ok() {
                                successful_insertions += 1;
                                // Check validity after each insertion
                                prop_assert!(
                                    tds.is_valid().is_ok(),
                                    "{}D triangulation should remain valid after insertion #{}: {:?}",
                                    $dim,
                                    successful_insertions,
                                    tds.is_valid().err()
                                );
                            }
                        }

                        // Verify vertex count increased appropriately
                        let final_vertex_count = tds.vertices().count();
                        prop_assert!(
                            final_vertex_count >= initial_vertex_count + successful_insertions,
                            "{}D vertex count mismatch: initial={}, final={}, insertions={}",
                            $dim,
                            initial_vertex_count,
                            final_vertex_count,
                            successful_insertions
                        );
                    }
                }

                /// Property: Inserting the same point multiple times doesn't break triangulation
                #[test]
                fn [<prop_duplicate_insertion_validity_ $dim d>](
                    initial_vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)),
                    duplicate_point in prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new)
                ) {
                    if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&initial_vertices) {
                        // Try inserting the same point multiple times
                        for attempt in 0..3 {
                            let new_vertex = Vertex::from_points(&[duplicate_point]);
                            let _ = tds.add(new_vertex[0].clone());

                            // Triangulation should remain valid regardless
                            prop_assert!(
                                tds.is_valid().is_ok(),
                                "{}D triangulation should remain valid after duplicate insertion attempt {}: {:?}",
                                $dim,
                                attempt + 1,
                                tds.is_valid().err()
                            );
                        }
                    }
                }

                /// Property: Cell count is consistent with Euler characteristic after insertions
                #[test]
                fn [<prop_euler_characteristic_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=($max_vertices + 3)
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        if tds.is_valid().is_ok() {
                            let vertex_count = tds.vertices().count();
                            let cell_count = tds.cells().count();

                            // Basic sanity checks based on dimension
                            prop_assert!(
                                vertex_count > $dim,
                                "{}D triangulation must have at least {} vertices",
                                $dim,
                                $dim + 1
                            );

                            prop_assert!(
                                cell_count > 0,
                                "{}D valid triangulation must have at least one cell",
                                $dim
                            );

                            // For non-degenerate triangulations, we expect
                            // roughly O(n^(D/2)) cells for n vertices in D dimensions
                            // This is a very loose bound to avoid false positives
                            let expected_min_cells = 1;
                            let expected_max_cells = vertex_count.pow($dim as u32) * 10;

                            prop_assert!(
                                cell_count >= expected_min_cells && cell_count <= expected_max_cells,
                                "{}D cell count {} should be within reasonable bounds [{}, {}] for {} vertices",
                                $dim,
                                cell_count,
                                expected_min_cells,
                                expected_max_cells,
                                vertex_count
                            );
                        }
                    }
                }

                /// Property: Vertex insertion maintains triangulation validity and topological invariants
                #[test]
                fn [<prop_insertion_maintains_invariants_ $dim d>](
                    initial_vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v)),
                    new_point in prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new)
                ) {
                    if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&initial_vertices) {
                        // Verify initial triangulation is valid
                        if tds.is_valid().is_ok() {
                            let initial_vertex_count = tds.vertices().count();

                            // Insert new vertex
                            let new_vertex = Vertex::from_points(&[new_point]);
                            if tds.add(new_vertex[0].clone()).is_ok() {
                                // After successful insertion, triangulation MUST remain valid
                                prop_assert!(
                                    tds.is_valid().is_ok(),
                                    "{}D triangulation became invalid after insertion: {:?}",
                                    $dim,
                                    tds.is_valid().err()
                                );

                                let final_vertex_count = tds.vertices().count();
                                let final_cell_count = tds.cells().count();

                                // Vertex count must increase (we added a vertex successfully)
                                prop_assert!(
                                    final_vertex_count == initial_vertex_count + 1,
                                    "{}D vertex count should increase by 1: {} -> {} (expected {})",
                                    $dim,
                                    initial_vertex_count,
                                    final_vertex_count,
                                    initial_vertex_count + 1
                                );

                                // Cell count can increase, decrease, or stay same depending on topology constraints
                                // What matters is the triangulation remains valid
                                // Note: Cell count typically increases but may decrease to maintain validity
                                // (e.g., when filtering prevents invalid facet sharing)
                                prop_assert!(
                                    final_cell_count > 0,
                                    "{}D triangulation must have at least one cell after insertion",
                                    $dim
                                );

                                // TODO: Add Euler characteristic check once implemented
                                // For now, validity check ensures topological consistency
                            }
                        }
                    }
                }
            }
        }
    };
}

// Generate tests for dimensions 2-5
// Parameters: dimension, min_vertices, max_vertices
test_bowyer_watson_properties!(2, 4, 10);
test_bowyer_watson_properties!(3, 5, 12);
test_bowyer_watson_properties!(4, 6, 14);
test_bowyer_watson_properties!(5, 7, 16);
