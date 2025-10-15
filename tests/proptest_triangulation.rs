//! Property-based tests for triangulation invariants.
//!
//! This module uses proptest to verify structural properties of Delaunay
//! triangulations that must hold universally, including:
//! - Neighbor symmetry (if A neighbors B, then B neighbors A)
//! - Vertex-cell incidence consistency
//! - No duplicate cells in valid triangulations
//! - Triangulation remains valid after vertex insertion

use delaunay::core::triangulation_data_structure::Tds;
use delaunay::core::vertex::Vertex;
use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use delaunay::vertex;
use proptest::prelude::*;

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

/// Strategy for generating a small collection of 2D vertices (4-10 vertices)
fn small_vertex_set_2d() -> impl Strategy<Value = Vec<Vertex<f64, Option<()>, 2>>> {
    prop::collection::vec(vertex_2d(), 4..=10).prop_map(Vertex::from_points)
}

/// Strategy for generating a small collection of 3D vertices (5-12 vertices)
fn small_vertex_set_3d() -> impl Strategy<Value = Vec<Vertex<f64, Option<()>, 3>>> {
    prop::collection::vec(vertex_3d(), 5..=12).prop_map(Vertex::from_points)
}

// =============================================================================
// TRIANGULATION VALIDITY TESTS
// =============================================================================

proptest! {
    /// Property: A triangulation constructed from valid vertices should pass
    /// the is_valid() check.
    #[test]
    fn prop_triangulation_from_vertices_is_valid_2d(vertices in small_vertex_set_2d()) {
        // Attempt to create triangulation
        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices) {
            // If construction succeeds, triangulation should be valid
            prop_assert!(
                tds.is_valid().is_ok(),
                "Triangulation constructed from vertices should be valid: {:?}",
                tds.is_valid().err()
            );
        }
        // If construction fails, that's acceptable (e.g., degenerate cases)
    }

    /// Property: 3D triangulation validity
    #[test]
    fn prop_triangulation_from_vertices_is_valid_3d(vertices in small_vertex_set_3d()) {
        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
            prop_assert!(
                tds.is_valid().is_ok(),
                "3D triangulation should be valid: {:?}",
                tds.is_valid().err()
            );
        }
    }
}

// =============================================================================
// NEIGHBOR SYMMETRY TESTS
// =============================================================================

proptest! {
    /// Property: Neighbor relationships are symmetric - if cell A has cell B
    /// as a neighbor at index i, then cell B should have cell A as a neighbor.
    #[test]
    fn prop_neighbor_symmetry_2d(vertices in small_vertex_set_2d()) {
        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices) {
            // Check all cells for neighbor symmetry
            for (cell_key, cell) in tds.cells() {
                if let Some(neighbors) = cell.neighbors() {
                    for (neighbor_index, neighbor_opt) in neighbors.iter().enumerate() {
                        if let Some(neighbor_key) = neighbor_opt {
                            let neighbor_cell = tds.cells().get(*neighbor_key).unwrap();
                            // The neighbor should have this cell as one of its neighbors
                            let mut found_reciprocal = false;
                            if let Some(neighbor_neighbors) = neighbor_cell.neighbors() {
                                for n in neighbor_neighbors {
                                    if n == &Some(cell_key) {
                                        found_reciprocal = true;
                                        break;
                                    }
                                }
                            }
                            prop_assert!(
                                found_reciprocal,
                                "Neighbor relationship should be symmetric: cell {:?} has neighbor {:?} at index {}, but reciprocal not found",
                                cell_key,
                                neighbor_key,
                                neighbor_index
                            );
                        }
                    }
                }
            }
        }
    }

    /// Property: 3D neighbor symmetry
    #[test]
    fn prop_neighbor_symmetry_3d(vertices in small_vertex_set_3d()) {
        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
            for (cell_key, cell) in tds.cells() {
                if let Some(neighbors) = cell.neighbors() {
                    for neighbor_key in neighbors.iter().flatten() {
                        let neighbor_cell = tds.cells().get(*neighbor_key).unwrap();
                        let mut found_reciprocal = false;
                        if let Some(neighbor_neighbors) = neighbor_cell.neighbors() {
                            for n in neighbor_neighbors {
                                if n == &Some(cell_key) {
                                    found_reciprocal = true;
                                    break;
                                }
                            }
                        }
                        prop_assert!(
                            found_reciprocal,
                            "3D neighbor relationship should be symmetric"
                        );
                    }
                }
            }
        }
    }
}

// =============================================================================
// VERTEX-CELL INCIDENCE TESTS
// =============================================================================

proptest! {
    /// Property: Every cell in the triangulation should reference valid vertices
    /// that exist in the TDS.
    #[test]
    fn prop_cell_vertices_exist_in_tds_2d(vertices in small_vertex_set_2d()) {
        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices) {
            let all_vertex_keys: Vec<_> = tds.vertices().keys().collect();

            for (_cell_key, cell) in tds.cells() {
                // Get all vertex keys from the cell
                let cell_vertex_keys = cell.vertices();

                // Each vertex key should exist in the TDS
                for vertex_key in cell_vertex_keys {
                    prop_assert!(
                        all_vertex_keys.contains(vertex_key),
                        "Cell vertex {:?} should exist in TDS vertex set",
                        vertex_key
                    );
                }
            }
        }
    }

    /// Property: 3D cell-vertex incidence
    #[test]
    fn prop_cell_vertices_exist_in_tds_3d(vertices in small_vertex_set_3d()) {
        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
            let all_vertex_keys: Vec<_> = tds.vertices().keys().collect();

            for (_cell_key, cell) in tds.cells() {
                let cell_vertex_keys = cell.vertices();

                for vertex_key in cell_vertex_keys {
                    prop_assert!(
                        all_vertex_keys.contains(vertex_key),
                        "3D cell vertex should exist in TDS"
                    );
                }
            }
        }
    }
}

// =============================================================================
// NO DUPLICATE CELLS TESTS
// =============================================================================

proptest! {
    /// Property: No two cells should have exactly the same set of vertices
    /// (duplicate cells violate triangulation invariants).
    #[test]
    fn prop_no_duplicate_cells_2d(vertices in small_vertex_set_2d()) {
        use std::collections::HashSet;

        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices) {
            let mut seen_vertex_sets = HashSet::new();

            for (_cell_key, cell) in tds.cells() {
                // Collect vertex keys for this cell and sort them for comparison
                let mut cell_vertices = cell.vertices().to_vec();
                cell_vertices.sort();

                // Check if we've seen this vertex set before
                prop_assert!(
                    seen_vertex_sets.insert(cell_vertices.clone()),
                    "Found duplicate cell with vertices {:?}",
                    cell_vertices
                );
            }
        }
    }

    /// Property: 3D no duplicate cells
    #[test]
    fn prop_no_duplicate_cells_3d(vertices in small_vertex_set_3d()) {
        use std::collections::HashSet;

        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
            let mut seen_vertex_sets = HashSet::new();

            for (_cell_key, cell) in tds.cells() {
                let mut cell_vertices = cell.vertices().to_vec();
                cell_vertices.sort();

                prop_assert!(
                    seen_vertex_sets.insert(cell_vertices),
                    "Found duplicate 3D cell"
                );
            }
        }
    }
}

// =============================================================================
// INCREMENTAL CONSTRUCTION TESTS
// =============================================================================

proptest! {
    /// Property: Adding vertices incrementally should maintain triangulation validity.
    #[test]
    fn prop_incremental_insertion_maintains_validity_2d(
        initial_points in prop::collection::vec(vertex_2d(), 3..=5),
        additional_point in vertex_2d(),
    ) {
        // Create initial triangulation
        let initial_vertices = Vertex::from_points(initial_points);
        if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, 2>::new(&initial_vertices) {
            // Should be valid initially
            prop_assert!(tds.is_valid().is_ok(), "Initial triangulation should be valid");

            // Add one more vertex
            let additional_vertex = vertex!(additional_point);
            if tds.add(additional_vertex).is_ok() {
                // Should still be valid after insertion
                prop_assert!(
                    tds.is_valid().is_ok(),
                    "Triangulation should remain valid after vertex insertion: {:?}",
                    tds.is_valid().err()
                );
            }
        }
    }

    /// Property: 3D incremental construction validity
    #[test]
    fn prop_incremental_insertion_maintains_validity_3d(
        initial_points in prop::collection::vec(vertex_3d(), 4..=6),
        additional_point in vertex_3d(),
    ) {
        let initial_vertices = Vertex::from_points(initial_points);
        if let Ok(mut tds) = Tds::<f64, Option<()>, Option<()>, 3>::new(&initial_vertices) {
            prop_assert!(tds.is_valid().is_ok(), "Initial 3D triangulation should be valid");

            let additional_vertex = vertex!(additional_point);
            if tds.add(additional_vertex).is_ok() {
                prop_assert!(
                    tds.is_valid().is_ok(),
                    "3D triangulation should remain valid after insertion: {:?}",
                    tds.is_valid().err()
                );
            }
        }
    }
}

// =============================================================================
// DIMENSION CONSISTENCY TESTS
// =============================================================================

proptest! {
    /// Property: The dimension of the triangulation should match the vertex dimension
    /// once enough vertices are added.
    #[test]
    fn prop_dimension_consistency_2d(vertices in small_vertex_set_2d()) {
        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices) {
            // If we have at least 3 non-degenerate vertices, dimension should be 2
            if tds.number_of_vertices() >= 3 && tds.number_of_cells() > 0 {
                prop_assert_eq!(
                    tds.dim(),
                    2,
                    "2D triangulation with {} vertices and {} cells should have dimension 2",
                    tds.number_of_vertices(),
                    tds.number_of_cells()
                );
            }
        }
    }

    /// Property: 3D dimension consistency
    #[test]
    fn prop_dimension_consistency_3d(vertices in small_vertex_set_3d()) {
        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
            // If we have at least 4 non-degenerate vertices, dimension should be 3
            if tds.number_of_vertices() >= 4 && tds.number_of_cells() > 0 {
                prop_assert_eq!(
                    tds.dim(),
                    3,
                    "3D triangulation with {} vertices should have dimension 3",
                    tds.number_of_vertices()
                );
            }
        }
    }
}

// =============================================================================
// VERTEX COUNT CONSISTENCY TESTS
// =============================================================================

proptest! {
    /// Property: The number of vertices in the TDS should match the number
    /// of unique vertex keys.
    #[test]
    fn prop_vertex_count_consistency_2d(vertices in small_vertex_set_2d()) {
        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 2>::new(&vertices) {
            let vertex_keys_count = tds.vertices().keys().count();
            let number_of_vertices = tds.number_of_vertices();

            prop_assert_eq!(
                vertex_keys_count,
                number_of_vertices,
                "Vertex keys count ({}) should match number_of_vertices ({})",
                vertex_keys_count,
                number_of_vertices
            );
        }
    }

    /// Property: 3D vertex count consistency
    #[test]
    fn prop_vertex_count_consistency_3d(vertices in small_vertex_set_3d()) {
        if let Ok(tds) = Tds::<f64, Option<()>, Option<()>, 3>::new(&vertices) {
            let vertex_keys_count = tds.vertices().keys().count();
            let number_of_vertices = tds.number_of_vertices();

            prop_assert_eq!(
                vertex_keys_count,
                number_of_vertices,
                "3D vertex keys count should match number_of_vertices"
            );
        }
    }
}
