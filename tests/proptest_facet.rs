//! Property-based tests for Facet operations.
//!
//! This module uses proptest to verify fundamental properties of Facet
//! operations in d-dimensional triangulations, including:
//! - Facet vertex count correctness (D vertices for D-dimensional simplex)
//! - Facet-cell relationship validity
//! - Facet boundary multiplicity (1 for boundary, 2 for interior)
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

use delaunay::prelude::*;
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
    ($dim:literal, $min_vertices:literal, $max_vertices:literal, $expected_facet_vertices:literal) => {
        pastey::paste! {
            proptest! {
                /// Property: Each facet should have exactly D vertices (one less than cell)
                #[test]
                fn [<prop_facet_has_correct_vertex_count_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices) {
                        let tds = dt.tds();
                        for cell_key in tds.cell_keys() {
                            // Each cell has D+1 facets (one opposite each vertex)
                            for facet_index in 0..=($dim as u8) {
                                if let Ok(facet) = FacetView::new(&tds, cell_key, facet_index) {
                                    if let Ok(facet_vertices) = facet.vertices() {
                                        let vertex_count = facet_vertices.count();
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
                }

                /// Property: Facets have one fewer vertex than their containing cell
                #[test]
                fn [<prop_facet_vertex_count_less_than_cell_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices) {
                        let tds = dt.tds();
                        for cell_key in tds.cell_keys() {
                            if let Some(cell) = tds.get_cell(cell_key) {
                                let cell_vertex_count = cell.vertices().len();

                                for facet_index in 0..=($dim as u8) {
                                    if let Ok(facet) = FacetView::new(&tds, cell_key, facet_index) {
                                        if let Ok(facet_vertices) = facet.vertices() {
                                            let facet_vertex_count = facet_vertices.count();
                                            prop_assert_eq!(
                                                facet_vertex_count,
                                                cell_vertex_count - 1,
                                                "{}D facet should have one fewer vertex than cell",
                                                $dim
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                /// Property: Each cell has valid facets
                #[test]
                fn [<prop_cell_has_valid_facets_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices) {
                        let tds = dt.tds();
                        // Check that each facet is valid
                        for cell_key in tds.cell_keys() {
                            for facet_index in 0..=($dim as u8) {
                                // Each facet should be constructible
                                prop_assert!(
                                    FacetView::new(&tds, cell_key, facet_index).is_ok(),
                                    "{}D facet {} of cell should be valid",
                                    $dim,
                                    facet_index
                                );
                            }
                        }
                    }
                }

                /// Property: Each cell should have exactly D+1 facets
                #[test]
                fn [<prop_cell_facet_count_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices) {
                        let tds = dt.tds();
                        for cell_key in tds.cell_keys() {
                            let mut facet_count = 0;
                            for facet_index in 0..=($dim as u8) {
                                if FacetView::new(&tds, cell_key, facet_index).is_ok() {
                                    facet_count += 1;
                                }
                            }
                            prop_assert_eq!(
                                facet_count,
                                $dim + 1,
                                "{}D cell should have exactly {} facets",
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
test_facet_properties!(4, 6, 14, 4);
test_facet_properties!(5, 7, 16, 5);

// Additional invariant: facet multiplicity (each facet should belong to 1 or 2 cells)
macro_rules! test_facet_multiplicity {
    ($dim:literal, $min_vertices:literal, $max_vertices:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_facet_multiplicity_ $dim d>](
                    vertices in prop::collection::vec(
                        prop::array::[<uniform $dim>](finite_coordinate()).prop_map(Point::new),
                        $min_vertices..=$max_vertices
                    ).prop_map(|v| Vertex::from_points(&v))
                ) {
                    if let Ok(dt) = DelaunayTriangulation::<_, (), (), $dim>::new(&vertices) {
                        let tds = dt.tds();
                        // Ensure we're checking a valid triangulation to avoid degenerate edge cases
                        prop_assume!(tds.is_valid().is_ok());

                        let mut counts: HashMap<u64, usize> = HashMap::new();

                        for (_cell_key, cell) in tds.cells() {
                            let vs = cell.vertices();
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
test_facet_multiplicity!(4, 6, 14);
test_facet_multiplicity!(5, 7, 16);
