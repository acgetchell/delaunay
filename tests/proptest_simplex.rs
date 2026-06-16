//! Property-based tests for Simplex operations.
//!
//! This module uses proptest to verify fundamental properties of Simplex
//! data structures in d-dimensional triangulations, including:
//! - Simplex vertex uniqueness (no duplicate vertices)
//! - Simplex dimension consistency (D+1 vertices for D-dimensional simplices)
//! - Simplex neighbor relationship validity
//! - Simplex UUID uniqueness and consistency
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

use delaunay::prelude::query::*;
use delaunay::prelude::topology::validation::*;
use delaunay::try_vertices_from_points;
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

/// Macro to generate simplex property tests for a given dimension
macro_rules! test_simplex_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal, $expected_vertices:literal, $max_neighbors:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {
                /// Property: All simplices should have unique vertices (no duplicates)
                $(#[$attr])*
                #[test]
                fn [<prop_simplex_vertices_are_unique_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        for (_simplex_key, simplex) in dt.simplices() {
                            let vertex_keys = simplex.vertices();
                            let unique_vertices: HashSet<_> = vertex_keys.iter().collect();
                            prop_assert_eq!(unique_vertices.len(), vertex_keys.len());
                        }
                    }
                }

                /// Property: Each simplex should have exactly D+1 vertices
                $(#[$attr])*
                #[test]
                fn [<prop_simplex_has_correct_vertex_count_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        for (_simplex_key, simplex) in dt.simplices() {
                            prop_assert_eq!(simplex.vertices().len(), $expected_vertices);
                        }
                    }
                }

                /// Property: Each simplex should have at most D+1 neighbors
                $(#[$attr])*
                #[test]
                fn [<prop_simplex_neighbor_count_bounded_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        for (_simplex_key, simplex) in dt.simplices() {
                            if let Some(neighbors) = simplex.neighbors() {
                                prop_assert!(neighbors.len() <= $max_neighbors);
                            }
                        }
                    }
                }

                /// Property: All simplices in a triangulation should have unique UUIDs
                $(#[$attr])*
                #[test]
                fn [<prop_simplex_uuids_are_unique_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let mut seen_uuids = HashSet::new();
                        for (_simplex_key, simplex) in dt.simplices() {
                            prop_assert!(seen_uuids.insert(simplex.uuid()));
                        }
                    }
                }

                /// Property: Simplices retrieved from a valid triangulation should pass validation
                $(#[$attr])*
                #[test]
                fn [<prop_simplices_in_valid_tds_are_valid_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<AdaptiveKernel<f64>, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        if dt.tds().validate().is_ok() {
                            for (_simplex_key, simplex) in dt.simplices() {
                                prop_assert_eq!(simplex.vertices().len(), $expected_vertices);
                                prop_assert!(simplex.uuid().as_u128() != 0);
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
test_simplex_properties!(2, 4, 10, 3, 3);
test_simplex_properties!(3, 5, 12, 4, 4);
test_simplex_properties!(4, 6, 14, 5, 5, #[cfg(feature = "slow-tests")]);
test_simplex_properties!(5, 7, 16, 6, 6, #[cfg(feature = "slow-tests")]);
