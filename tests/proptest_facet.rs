//! Property-based tests for Facet operations.
//!
//! This module uses proptest to verify fundamental properties of Facet
//! operations in d-dimensional triangulations, including:
//! - Facet vertex count correctness (D vertices for D-dimensional simplex)
//! - Facet-simplex relationship validity
//! - Facet incidence multiplicity (one-sided or two-sided)
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

use delaunay::prelude::construction::{DelaunayTriangulation, TopologyGuarantee};
use delaunay::prelude::query::*;
use delaunay::try_vertices_from_points;
use proptest::prelude::*;
use std::collections::HashMap;

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

/// Macro to generate facet property tests for a given dimension
macro_rules! test_facet_properties {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal, $expected_facet_vertices:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {
                /// Property: Each facet should have exactly D vertices (one less than simplex)
                $(#[$attr])*
                #[test]
                fn [<prop_facet_has_correct_vertex_count_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::builder(&vertices).topology_guarantee(TopologyGuarantee::PLManifold).build() {
                        for facet in dt.facets() {
                            let facet = facet.expect("facet iterator should resolve valid facets");
                            let vertex_count = facet.vertices().count();
                            prop_assert_eq!(
                                vertex_count,
                                $expected_facet_vertices,
                                "{}D facet should have exactly {} vertices",
                                $dim,
                                $expected_facet_vertices
                            );
                        }
                    }
                }

                /// Property: Facets have one fewer vertex than their containing simplex
                $(#[$attr])*
                #[test]
                fn [<prop_facet_vertex_count_less_than_simplex_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::builder(&vertices).topology_guarantee(TopologyGuarantee::PLManifold).build() {
                        for facet in dt.facets() {
                            let facet = facet.expect("facet iterator should resolve valid facets");
                            let simplex_vertex_count = facet.simplex().vertices().len();
                            let facet_vertex_count = facet.vertices().count();
                            prop_assert_eq!(
                                facet_vertex_count,
                                simplex_vertex_count - 1,
                                "{}D facet should have one fewer vertex than simplex",
                                $dim
                            );
                        }
                    }
                }

                /// Property: Each simplex has valid facets
                $(#[$attr])*
                #[test]
                fn [<prop_simplex_has_valid_facets_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::builder(&vertices).topology_guarantee(TopologyGuarantee::PLManifold).build() {
                        for facet in dt.facets() {
                            prop_assert!(
                                facet.is_ok(),
                                "{}D public facet iterator should only yield valid facets",
                                $dim
                            );
                        }
                    }
                }

                /// Property: Each simplex should have exactly D+1 facets
                $(#[$attr])*
                #[test]
                fn [<prop_simplex_facet_count_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::builder(&vertices).topology_guarantee(TopologyGuarantee::PLManifold).build() {
                        let facet_count = dt
                            .facets()
                            .try_fold(0_usize, |count, facet| facet.map(|_| count + 1))
                            .expect("facet iterator should resolve valid facets");
                        prop_assert_eq!(
                            facet_count,
                            dt.number_of_simplices() * ($dim + 1),
                            "{}D triangulation should have exactly {} facets per simplex",
                            $dim,
                            $dim + 1
                        );
                    }
                }
            }
        }
    };
}

// Generate tests for dimensions 2-5
// Parameters: dimension, min_vertices, max_vertices, expected_facet_vertices (D)
test_facet_properties!(2, 4, 10, 2);
test_facet_properties!(3, 5, 12, 3);
test_facet_properties!(4, 6, 14, 4, #[cfg(feature = "slow-tests")]);
test_facet_properties!(5, 7, 16, 5, #[cfg(feature = "slow-tests")]);

// Additional invariant: facet multiplicity (each facet should belong to 1 or 2 simplices)
macro_rules! test_facet_multiplicity {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal $(, #[$attr:meta])*) => {
        pastey::paste! {
            proptest! {
                $(#[$attr])*
                #[test]
                fn [<prop_facet_multiplicity_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(|coords| Point::try_new(coords).expect("finite point coordinates")),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| try_vertices_from_points(&v).expect("finite point coordinates"))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::builder(&vertices).topology_guarantee(TopologyGuarantee::PLManifold).build() {
                        // Ensure we're checking a valid triangulation to avoid degenerate edge cases
                        prop_assume!(dt.is_valid_structure().is_ok());

                        let mut counts: HashMap<u64, usize> = HashMap::new();

                        for facet in dt.facets() {
                            let facet = facet.expect("facet iterator should resolve valid facets");
                            *counts.entry(facet.key()).or_default() += 1;
                        }

                        for (facet, c) in counts {
                            prop_assert!(c == 1 || c == 2, "facet {:?} appears {} times (must be 1 or 2)", facet, c);
                        }
                    }
                }
            }
        }
    };
}

test_facet_multiplicity!(2, 4, 10);
test_facet_multiplicity!(3, 5, 12);
test_facet_multiplicity!(4, 6, 14, #[cfg(feature = "slow-tests")]);
test_facet_multiplicity!(5, 7, 16, #[cfg(feature = "slow-tests")]);
