//! Property-based tests for triangulation invariants.
//!
//! This module uses proptest to verify structural properties of Delaunay
//! triangulations that must hold universally, including:
//! - Neighbor symmetry (if A neighbors B, then B neighbors A)
//! - Vertex-cell incidence consistency
//! - No duplicate cells in valid triangulations
//! - Triangulation remains valid after vertex insertion

use delaunay::core::triangulation_data_structure::Tds;
use delaunay::core::util::jaccard_index;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use delaunay::vertex;
use proptest::prelude::*;
use std::collections::HashSet;

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates in a reasonable range
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

/// Strategy for generating 2D vertices
fn vertex_2d() -> impl Strategy<Value = Point<f64, 2>> {
    prop::array::uniform2(finite_coordinate()).prop_map(Point::new)
}

/// Strategy for generating 3D vertices
fn vertex_3d() -> impl Strategy<Value = Point<f64, 3>> {
    prop::array::uniform3(finite_coordinate()).prop_map(Point::new)
}

/// Strategy for generating 4D vertices
fn vertex_4d() -> impl Strategy<Value = Point<f64, 4>> {
    prop::array::uniform4(finite_coordinate()).prop_map(Point::new)
}

/// Strategy for generating 5D vertices
fn vertex_5d() -> impl Strategy<Value = Point<f64, 5>> {
    prop::array::uniform5(finite_coordinate()).prop_map(Point::new)
}

/// Strategy for generating a small collection of 2D vertices (4-10 vertices)
fn small_vertex_set_2d() -> impl Strategy<Value = Vec<Vertex<f64, Option<()>, 2>>> {
    prop::collection::vec(vertex_2d(), 4..=10).prop_map(|v| Vertex::from_points(&v))
}

/// Strategy for generating a small collection of 3D vertices (5-12 vertices)
fn small_vertex_set_3d() -> impl Strategy<Value = Vec<Vertex<f64, Option<()>, 3>>> {
    prop::collection::vec(vertex_3d(), 5..=12).prop_map(|v| Vertex::from_points(&v))
}

/// Strategy for generating a small collection of 4D vertices (6-14 vertices)
fn small_vertex_set_4d() -> impl Strategy<Value = Vec<Vertex<f64, Option<()>, 4>>> {
    prop::collection::vec(vertex_4d(), 6..=14).prop_map(|v| Vertex::from_points(&v))
}

/// Strategy for generating a small collection of 5D vertices (7-16 vertices)
fn small_vertex_set_5d() -> impl Strategy<Value = Vec<Vertex<f64, Option<()>, 5>>> {
    prop::collection::vec(vertex_5d(), 7..=16).prop_map(|v| Vertex::from_points(&v))
}

// =============================================================================
// TRIANGULATION VALIDITY TESTS
// =============================================================================

// Macros to generate dimension-specific property tests
macro_rules! gen_triangulation_validity {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_triangulation_from_vertices_is_valid_ $dim d>](vertices in [<small_vertex_set_ $dim d>]()) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        prop_assert!(tds.is_valid().is_ok(),
                            "{}D triangulation should be valid: {:?}",
                            $dim, tds.is_valid().err());
                    }
                }
            }
        }
    };
}

macro_rules! gen_neighbor_symmetry {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_neighbor_symmetry_ $dim d>](vertices in [<small_vertex_set_ $dim d>]()) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        for (cell_key, cell) in tds.cells() {
                            if let Some(neighbors) = cell.neighbors() {
                                for neighbor_key in neighbors.iter().flatten() {
                                    let found_reciprocal = tds
                                        .get_cell(*neighbor_key)
                                        .and_then(|c| c.neighbors())
                                        .is_some_and(|nn| nn.iter().any(|n| n == &Some(cell_key)));

                                    if !found_reciprocal {
                                        // Enhanced diagnostics with Jaccard similarity
                                        let cell_neighbors: HashSet<_> = neighbors
                                            .iter()
                                            .flatten()
                                            .copied()
                                            .collect();
                                        let neighbor_neighbors: HashSet<_> = tds
                                            .get_cell(*neighbor_key)
                                            .and_then(|c| c.neighbors())
                                            .map(|nn| nn.iter().flatten().copied().collect())
                                            .unwrap_or_default();

                                        let similarity = jaccard_index(&cell_neighbors, &neighbor_neighbors)
                                            .expect("Jaccard computation should not overflow for neighbor sets");
                                        let intersection: Vec<_> = cell_neighbors
                                            .intersection(&neighbor_neighbors)
                                            .take(5)
                                            .collect();

                                        prop_assert!(
                                            false,
                                            "{}D neighbor relationship should be symmetric\n\
                                             Cell {:?} has neighbor {:?}, but reciprocal not found\n\
                                             Jaccard similarity between neighbor sets: {:.6}\n\
                                             Cell neighbors: {} total\n\
                                             Neighbor's neighbors: {} total\n\
                                             Common neighbors (first 5): {:?}",
                                            $dim,
                                            cell_key,
                                            neighbor_key,
                                            similarity,
                                            cell_neighbors.len(),
                                            neighbor_neighbors.len(),
                                            intersection
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };
}

macro_rules! gen_neighbor_index_semantics {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_neighbor_index_semantics_ $dim d>](vertices in [<small_vertex_set_ $dim d>]()) {
                    // Use stack-allocated buffer for D facet vertices (D ≤ 7 typical)
                    use delaunay::core::collections::SimplexVertexBuffer;
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        prop_assume!(tds.is_valid().is_ok());
                        for (cell_key, cell) in tds.cells() {
                            if let Some(neighbors) = cell.neighbors() {
                                let a_vertices = cell.vertices();
                                for (i, nb) in neighbors.iter().enumerate() {
                                    if let Some(b_key) = nb {
                                        let b_cell = tds.get_cell(*b_key).unwrap();
                                        let b_vertices = b_cell.vertices();
                                        let mut a_facet: SimplexVertexBuffer<_> = a_vertices.iter().enumerate()
                                            .filter_map(|(idx, &vk)| (idx != i).then_some(vk))
                                            .collect();
                                        a_facet.sort_unstable();
                                        let mut found_j = None;
                                        for j in 0..b_vertices.len() {
                                            let mut b_facet: SimplexVertexBuffer<_> = b_vertices.iter().enumerate()
                                                .filter_map(|(idx, &vk)| (idx != j).then_some(vk))
                                                .collect();
                                            b_facet.sort_unstable();
                                            if b_facet == a_facet { found_j = Some(j); break; }
                                        }
                                        prop_assert!(found_j.is_some(), "Facet mismatch between neighbor cells");
                                        if let Some(j) = found_j && let Some(b_neighs) = b_cell.neighbors() {
                                            prop_assert_eq!(b_neighs.get(j).copied().flatten(), Some(cell_key),
                                                "Reciprocal neighbor at correct index not found");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };
}

macro_rules! gen_cell_vertices_exist_in_tds {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_cell_vertices_exist_in_tds_ $dim d>](vertices in [<small_vertex_set_ $dim d>]()) {
                    use std::collections::HashSet;
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        let all_vertex_keys: HashSet<_> = tds.vertex_keys().collect();
                        for (_cell_key, cell) in tds.cells() {
                            for vertex_key in cell.vertices() {
                                prop_assert!(all_vertex_keys.contains(vertex_key),
                                    "{}D cell vertex should exist in TDS", $dim);
                            }
                        }
                    }
                }
            }
        }
    };
}

macro_rules! gen_no_duplicate_cells {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_no_duplicate_cells_ $dim d>](vertices in [<small_vertex_set_ $dim d>]()) {
                    use std::collections::HashSet;
                    use delaunay::core::collections::CellVertexBuffer;
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        let mut seen = HashSet::new();
                        for (_cell_key, cell) in tds.cells() {
                            // Use stack-allocated buffer for D+1 vertices (D ≤ 7 typical)
                            let mut vs: CellVertexBuffer = cell.vertices().iter().copied().collect();
                            vs.sort();
                            prop_assert!(seen.insert(vs), "Found duplicate {}D cell", $dim);
                        }
                    }
                }
            }
        }
    };
}

macro_rules! gen_incremental_insertion_validity {
    ($dim:literal, $min:literal, $max:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_incremental_insertion_maintains_validity_ $dim d>](
                    initial_points in prop::collection::vec([<vertex_ $dim d>](), $min..=$max),
                    additional_point in [<vertex_ $dim d>](),
                ) {
                    let initial_vertices = Vertex::from_points(&initial_points);
                    if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&initial_vertices) {
                        prop_assert!(tds.is_valid().is_ok(), "Initial {}D triangulation should be valid", $dim);
                        let additional_vertex = vertex!(additional_point);
                        if tds.add(additional_vertex).is_ok() {
                            prop_assert!(tds.is_valid().is_ok(), "{}D triangulation should remain valid after insertion: {:?}", $dim, tds.is_valid().err());
                        }
                    }
                }
            }
        }
    };
}

macro_rules! gen_dimension_consistency {
    ($dim:literal, $min_vertices:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_dimension_consistency_ $dim d>](vertices in [<small_vertex_set_ $dim d>]()) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        if tds.number_of_vertices() >= $min_vertices && tds.number_of_cells() > 0 {
                            prop_assert_eq!(tds.dim(), $dim as i32, "{}D triangulation dimension mismatch", $dim);
                        }
                    }
                }
            }
        }
    };
}

macro_rules! gen_vertex_count_consistency {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_vertex_count_consistency_ $dim d>](vertices in [<small_vertex_set_ $dim d>]()) {
                    if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, $dim>::new(&vertices) {
                        let keys = tds.vertex_keys().count();
                        let n = tds.number_of_vertices();
                        prop_assert_eq!(keys, n, "{}D vertex keys count should match number_of_vertices", $dim);
                    }
                }
            }
        }
    };
}

// =============================================================================
// TRIANGULATION VALIDITY TESTS
// =============================================================================

gen_triangulation_validity!(2);
gen_triangulation_validity!(3);
gen_triangulation_validity!(4);
gen_triangulation_validity!(5);

// =============================================================================
// NEIGHBOR SYMMETRY TESTS
// =============================================================================

gen_neighbor_symmetry!(2);

gen_neighbor_symmetry!(3);

gen_neighbor_symmetry!(4);

gen_neighbor_symmetry!(5);

// =============================================================================
// NEIGHBOR INDEX SEMANTICS TESTS
// =============================================================================

gen_neighbor_index_semantics!(2);

gen_neighbor_index_semantics!(3);

gen_neighbor_index_semantics!(4);

gen_neighbor_index_semantics!(5);

// =============================================================================
// VERTEX-CELL INCIDENCE TESTS
// =============================================================================

gen_cell_vertices_exist_in_tds!(2);

gen_cell_vertices_exist_in_tds!(3);

gen_cell_vertices_exist_in_tds!(4);

gen_cell_vertices_exist_in_tds!(5);

// =============================================================================
// NO DUPLICATE CELLS TESTS
// =============================================================================

gen_no_duplicate_cells!(2);

gen_no_duplicate_cells!(3);

gen_no_duplicate_cells!(4);

gen_no_duplicate_cells!(5);

// =============================================================================
// INCREMENTAL CONSTRUCTION TESTS
// =============================================================================

gen_incremental_insertion_validity!(2, 3, 5);

gen_incremental_insertion_validity!(3, 4, 6);

gen_incremental_insertion_validity!(4, 5, 7);

gen_incremental_insertion_validity!(5, 6, 8);

// =============================================================================
// DIMENSION CONSISTENCY TESTS
// =============================================================================

gen_dimension_consistency!(2, 3);

gen_dimension_consistency!(3, 4);

gen_dimension_consistency!(4, 5);

gen_dimension_consistency!(5, 6);

// =============================================================================
// VERTEX COUNT CONSISTENCY TESTS
// =============================================================================

gen_vertex_count_consistency!(2);

gen_vertex_count_consistency!(3);

gen_vertex_count_consistency!(4);

gen_vertex_count_consistency!(5);
