//! Property-based tests for Cell operations.
//!
//! This module uses proptest to verify fundamental properties of Cell
//! data structures in d-dimensional triangulations, including:
//! - Cell vertex uniqueness (no duplicate vertices)
//! - Cell dimension consistency (D+1 vertices for D-dimensional cells)
//! - Cell neighbor relationship validity
//! - Cell UUID uniqueness and consistency
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

use delaunay::prelude::query::*;
use delaunay::prelude::topology::validation::*;
use proptest::prelude::*;
use std::collections::HashSet;

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

/// Macro to generate cell property tests for a given dimension
macro_rules! test_cell_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal, $expected_vertices:literal, $max_neighbors:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {
                /// Property: All cells should have unique vertices (no duplicates)
                $(#[$attr])*
                #[test]
                fn [<prop_cell_vertices_are_unique_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        for (_cell_key, cell) in dt.cells() {
                            let vertex_keys = cell.vertices();
                            let unique_vertices: HashSet<_> = vertex_keys.iter().collect();
                            prop_assert_eq!(unique_vertices.len(), vertex_keys.len());
                        }
                    }
                }

                /// Property: Each cell should have exactly D+1 vertices
                $(#[$attr])*
                #[test]
                fn [<prop_cell_has_correct_vertex_count_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        for (_cell_key, cell) in dt.cells() {
                            prop_assert_eq!(cell.vertices().len(), $expected_vertices);
                        }
                    }
                }

                /// Property: Each cell should have at most D+1 neighbors
                $(#[$attr])*
                #[test]
                fn [<prop_cell_neighbor_count_bounded_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        for (_cell_key, cell) in dt.cells() {
                            if let Some(neighbors) = cell.neighbors() {
                                prop_assert!(neighbors.len() <= $max_neighbors);
                            }
                        }
                    }
                }

                /// Property: All cells in a triangulation should have unique UUIDs
                $(#[$attr])*
                #[test]
                fn [<prop_cell_uuids_are_unique_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let mut seen_uuids = HashSet::new();
                        for (_cell_key, cell) in dt.cells() {
                            prop_assert!(seen_uuids.insert(cell.uuid()));
                        }
                    }
                }

                /// Property: Cells retrieved from a valid triangulation should pass validation
                $(#[$attr])*
                #[test]
                fn [<prop_cells_in_valid_tds_are_valid_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<FastKernel<f64>, (), (), $dim>::new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        if dt.tds().validate().is_ok() {
                            for (_cell_key, cell) in dt.cells() {
                                prop_assert_eq!(cell.vertices().len(), $expected_vertices);
                                prop_assert!(cell.uuid().as_u128() != 0);
                            }
                        }
                    }
                }
            }
        }
    };
}

// Generate tests for dimensions 2-5
// Parameters: dimension, min_vertices, max_vertices, expected_vertices (D+1), max_neighbors (D+1)
test_cell_properties!(2, 4, 10, 3, 3);
test_cell_properties!(3, 5, 12, 4, 4);
test_cell_properties!(4, 6, 14, 5, 5, #[ignore = "Slow (>60s) in test-integration"]);
test_cell_properties!(5, 7, 16, 6, 6, #[ignore = "Slow (>60s) in test-integration"]);
