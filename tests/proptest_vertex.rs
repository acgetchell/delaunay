//! Property-based tests for Vertex data structure.
//!
//! This module uses proptest to verify fundamental properties of Vertex operations,
//! including:
//! - Vertex equality is reflexive, symmetric, and transitive
//! - Vertex hashing consistency with equality (Eq/Hash contract)
//! - UUID uniqueness across generated vertices
//! - Point coordinate validation (rejects NaN/Infinity)
//! - Vertex ordering consistency (lexicographic by coordinates)
//!
//! Tests are generated for dimensions 2D-5D using macros to reduce duplication.

#![allow(unused_imports)] // Imports used in macro expansion

use delaunay::prelude::*;
use proptest::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates
fn finite_coordinate() -> impl Strategy<Value = f64> {
    (-100.0..100.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

/// Strategy for generating non-finite f64 coordinates (NaN, Infinity)
fn non_finite_coordinate() -> impl Strategy<Value = f64> {
    prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY),]
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Compute hash of a vertex for testing hash consistency
fn compute_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

// =============================================================================
// DIMENSIONAL TEST GENERATION MACROS
// =============================================================================

/// Macro to generate vertex property tests for a given dimension
macro_rules! test_vertex_properties {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                /// Property: Vertex equality is reflexive (v == v)
                #[test]
                fn [<prop_vertex_equality_reflexive_ $dim d>](coords in prop::array::[<uniform $dim>](finite_coordinate())) {
                    let vertex: Vertex<f64, (), $dim> = vertex!(coords);
                    prop_assert_eq!(vertex, vertex, "{}D: Vertex should equal itself", $dim);
                }

                /// Property: Vertex equality is symmetric (v1 == v2 implies v2 == v1)
                #[test]
                fn [<prop_vertex_equality_symmetric_ $dim d>](coords in prop::array::[<uniform $dim>](finite_coordinate())) {
                    let v1: Vertex<f64, (), $dim> = vertex!(coords);
                    let v2: Vertex<f64, (), $dim> = vertex!(coords);
                    prop_assert_eq!(v1, v2, "{}D: Vertices with same coords should be equal", $dim);
                    prop_assert_eq!(v2, v1, "{}D: Equality should be symmetric", $dim);
                }

                /// Property: Vertex equality is transitive (v1 == v2 and v2 == v3 implies v1 == v3)
                #[test]
                fn [<prop_vertex_equality_transitive_ $dim d>](coords in prop::array::[<uniform $dim>](finite_coordinate())) {
                    let v1: Vertex<f64, (), $dim> = vertex!(coords);
                    let v2: Vertex<f64, (), $dim> = vertex!(coords);
                    let v3: Vertex<f64, (), $dim> = vertex!(coords);

                    prop_assert_eq!(v1, v2, "{}D: v1 should equal v2", $dim);
                    prop_assert_eq!(v2, v3, "{}D: v2 should equal v3", $dim);
                    prop_assert_eq!(v1, v3, "{}D: v1 should equal v3 (transitivity)", $dim);
                }

                /// Property: Equal vertices have equal hashes (Eq/Hash contract)
                #[test]
                fn [<prop_vertex_hash_consistency_ $dim d>](coords in prop::array::[<uniform $dim>](finite_coordinate())) {
                    let v1: Vertex<f64, (), $dim> = vertex!(coords);
                    let v2: Vertex<f64, (), $dim> = vertex!(coords);

                    prop_assert_eq!(v1, v2, "{}D: Vertices should be equal", $dim);

                    let hash1 = compute_hash(&v1);
                    let hash2 = compute_hash(&v2);
                    prop_assert_eq!(
                        hash1, hash2,
                        "{}D: Equal vertices must have equal hashes", $dim
                    );
                }

                /// Property: Vertices with different coordinates have different equality
                #[test]
                fn [<prop_vertex_inequality_ $dim d>](coords1 in prop::array::[<uniform $dim>](finite_coordinate()), coords2 in prop::array::[<uniform $dim>](finite_coordinate())) {
                    let v1: Vertex<f64, (), $dim> = vertex!(coords1);
                    let v2: Vertex<f64, (), $dim> = vertex!(coords2);

                    // If coordinates differ (component-wise), vertices should not be equal
                    if coords1.iter().zip(coords2.iter()).any(|(a, b)| (a - b).abs() > 1e-12) {
                        prop_assert_ne!(v1, v2, "{}D: Vertices with different coords should not be equal", $dim);
                    }
                }

                /// Property: UUIDs are unique across generated vertices
                #[test]
                fn [<prop_vertex_uuid_uniqueness_ $dim d>](coords_list in prop::collection::vec(prop::array::[<uniform $dim>](finite_coordinate()), 1..=20_usize)) {
                    let vertices: Vec<Vertex<f64, (), $dim>> = coords_list
                        .into_iter()
                        .map(|coords| vertex!(coords))
                        .collect();

                    let mut seen_uuids = HashSet::new();
                    for vertex in &vertices {
                        prop_assert!(
                            seen_uuids.insert(vertex.uuid()),
                            "{}D: Each vertex should have a unique UUID",
                            $dim
                        );
                    }

                    prop_assert_eq!(
                        seen_uuids.len(),
                        vertices.len(),
                        "{}D: All UUIDs should be unique",
                        $dim
                    );
                }

                /// Property: Vertices can be stored in HashMap with consistent lookup
                #[test]
                fn [<prop_vertex_hashmap_usage_ $dim d>](coords_list in prop::collection::vec(prop::array::[<uniform $dim>](finite_coordinate()), 1..=10_usize)) {
                    let mut map: HashMap<Vertex<f64, (), $dim>, usize> = HashMap::new();

                    // Insert vertices with their indices
                    for (i, coords) in coords_list.iter().enumerate() {
                        let vertex: Vertex<f64, (), $dim> = vertex!(*coords);
                        map.insert(vertex, i);
                    }

                    // Lookup using new vertices with same coordinates
                    for (expected_index, coords) in coords_list.iter().enumerate() {
                        let lookup_vertex: Vertex<f64, (), $dim> = vertex!(*coords);
                        if let Some(&actual_index) = map.get(&lookup_vertex) {
                            prop_assert_eq!(
                                actual_index,
                                expected_index,
                                "{}D: HashMap lookup should find correct vertex",
                                $dim
                            );
                        } else {
                            return Err(TestCaseError::fail(format!(
                                "{}D: HashMap lookup failed for vertex at index {}",
                                $dim, expected_index
                            )));
                        }
                    }
                }

                /// Property: Vertex dimension matches const parameter
                #[test]
                fn [<prop_vertex_dimension_ $dim d>](coords in prop::array::[<uniform $dim>](finite_coordinate())) {
                    let vertex: Vertex<f64, (), $dim> = vertex!(coords);
                    prop_assert_eq!(vertex.dim(), $dim, "{}D: Vertex dimension should match D", $dim);
                }

                /// Property: Valid vertices pass validation
                #[test]
                fn [<prop_vertex_validation_ $dim d>](coords in prop::array::[<uniform $dim>](finite_coordinate())) {
                    let vertex: Vertex<f64, (), $dim> = vertex!(coords);
                    prop_assert!(
                        vertex.is_valid().is_ok(),
                        "{}D: Vertex with finite coordinates should be valid",
                        $dim
                    );
                }

                /// Property: Vertex ordering is consistent with lexicographic coordinate order
                #[test]
                fn [<prop_vertex_ordering_ $dim d>](coords1 in prop::array::[<uniform $dim>](finite_coordinate()), coords2 in prop::array::[<uniform $dim>](finite_coordinate())) {
                    let v1: Vertex<f64, (), $dim> = vertex!(coords1);
                    let v2: Vertex<f64, (), $dim> = vertex!(coords2);

                    // Compare vertices
                    let vertex_cmp = v1.partial_cmp(&v2);

                    // Compare coordinate arrays lexicographically
                    let coords_cmp = coords1.partial_cmp(&coords2);

                    prop_assert_eq!(
                        vertex_cmp, coords_cmp,
                        "{}D: Vertex ordering should match lexicographic coordinate order",
                        $dim
                    );
                }

                /// Property: Vertex data is preserved
                #[test]
                fn [<prop_vertex_data_preservation_ $dim d>](coords in prop::array::[<uniform $dim>](finite_coordinate()), data in prop::num::i32::ANY) {
                    let vertex: Vertex<f64, i32, $dim> = vertex!(coords, data);
                    prop_assert_eq!(
                        vertex.data,
                        Some(data),
                        "{}D: Vertex data should be preserved",
                        $dim
                    );
                }

                /// Property: Vertex point() returns correct coordinates
                #[test]
                fn [<prop_vertex_point_access_ $dim d>](coords in prop::array::[<uniform $dim>](finite_coordinate())) {
                    let vertex: Vertex<f64, (), $dim> = vertex!(coords);
                    let point = vertex.point();
                    let point_coords: [f64; $dim] = point.into();

                    for (i, (&coord, &point_coord)) in coords.iter().zip(point_coords.iter()).enumerate() {
                        prop_assert!(
                            (coord - point_coord).abs() < 1e-10,
                            "{}D: Coordinate {} should match: {} vs {}",
                            $dim, i, coord, point_coord
                        );
                    }
                }

                /// Property: Vertices can be converted to coordinate arrays
                #[test]
                fn [<prop_vertex_to_array_conversion_ $dim d>](coords in prop::array::[<uniform $dim>](finite_coordinate())) {
                    let vertex: Vertex<f64, (), $dim> = vertex!(coords);
                    let array: [f64; $dim] = vertex.into();

                    for (i, (&original, &converted)) in coords.iter().zip(array.iter()).enumerate() {
                        prop_assert!(
                            (original - converted).abs() < 1e-10,
                            "{}D: Array conversion should preserve coordinate {}: {} vs {}",
                            $dim, i, original, converted
                        );
                    }
                }
            }
        }
    };
}

// Generate tests for dimensions 2-5
test_vertex_properties!(2);
test_vertex_properties!(3);
test_vertex_properties!(4);
test_vertex_properties!(5);

// =============================================================================
// CROSS-DIMENSIONAL TESTS
// =============================================================================

proptest! {
    /// Property: Vertices with NaN coordinates should be detected as invalid
    #[test]
    fn prop_vertex_rejects_nan_2d(valid_coord in finite_coordinate(), nan_coord in non_finite_coordinate(), nan_index in 0..2_usize) {
        let mut coords = [valid_coord; 2];
        coords[nan_index] = nan_coord;

        // Point validation should catch non-finite values during construction
        let point_result: Result<Point<f64, 2>, _> = Point::try_from(coords);
        prop_assert!(
            point_result.is_err(),
            "Point construction should reject non-finite coordinates"
        );
    }


}

/// Property: Empty vertex (with nil UUID) should fail validation
#[test]
fn prop_empty_vertex_invalid_3d() {
    let vertex: Vertex<f64, (), 3> = Vertex::empty();
    let validation_result = vertex.is_valid();
    assert!(
        validation_result.is_err(),
        "Empty vertex with nil UUID should fail validation"
    );
}
