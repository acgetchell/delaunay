//! Property-based tests for Facet operations.
//!
//! This module uses proptest to verify fundamental properties of Facet
//! operations in d-dimensional triangulations, including:
//! - Facet vertex count correctness (D vertices for D-dimensional simplex)
//! - Facet-simplex relationship validity
//! - Facet boundary multiplicity (1 for boundary, 2 for interior)
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

use delaunay::prelude::query::*;
use delaunay::prelude::tds::facet_key_from_vertices;
use delaunay::prelude::topology::validation::*;
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
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let tds = dt.tds();
                        for simplex_key in tds.simplex_keys() {
                            // Each simplex has D+1 facets (one opposite each vertex)
                            for facet_index in 0..=($dim as u8) {
                                if let Ok(facet) = FacetView::try_new(&tds, simplex_key, facet_index) {
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
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let tds = dt.tds();
                        for simplex_key in tds.simplex_keys() {
                            prop_assert!(
                                tds.simplex(simplex_key).is_some(),
                                "simplex key from iterator should exist: {simplex_key:?}"
                            );
                            let simplex = tds.simplex(simplex_key).expect("checked above");
                            let simplex_vertex_count = simplex.vertices().len();

                            for facet_index in 0..=($dim as u8) {
                                if let Ok(facet) = FacetView::try_new(&tds, simplex_key, facet_index) {
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
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let tds = dt.tds();
                        // Check that each facet is valid
                        for simplex_key in tds.simplex_keys() {
                            for facet_index in 0..=($dim as u8) {
                                // Each facet should be constructible
                                prop_assert!(
                                    FacetView::try_new(&tds, simplex_key, facet_index).is_ok(),
                                    "{}D facet {} of simplex should be valid",
                                    $dim,
                                    facet_index
                                );
                            }
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
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let tds = dt.tds();
                        for simplex_key in tds.simplex_keys() {
                            let mut facet_count = 0;
                            for facet_index in 0..=($dim as u8) {
                                if FacetView::try_new(&tds, simplex_key, facet_index).is_ok() {
                                    facet_count += 1;
                                }
                            }
                            prop_assert_eq!(
                                facet_count,
                                $dim + 1,
                                "{}D simplex should have exactly {} facets",
                                $dim,
                                $dim + 1
                            );
                        }
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
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::try_new_with_topology_guarantee(
                        &vertices,
                        TopologyGuarantee::PLManifold,
                    ) {
                        let tds = dt.tds();
                        // Ensure we're checking a valid triangulation to avoid degenerate edge cases
                        prop_assume!(tds.is_valid().is_ok());

                        let mut counts: HashMap<u64, usize> = HashMap::new();

                        for (_simplex_key, simplex) in tds.simplices() {
                            let vs = simplex.vertices();
                            for i in 0..vs.len() {
                                let facet: Vec<_> = vs
                                    .iter()
                                    .copied()
                                    .enumerate()
                                    .filter_map(|(j, vk)| (j != i).then_some(vk))
                                    .collect();
                                let key = facet_key_from_vertices(&facet);
                                *counts.entry(key).or_default() += 1;
                            }
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
