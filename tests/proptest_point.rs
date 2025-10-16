//! Property-based tests for Point operations.
//!
//! This module uses proptest to verify fundamental properties of Point
//! data structures, including:
//! - Hash consistency (equal points have equal hashes)
//! - Equality reflexivity, symmetry, and transitivity
//! - Serialization/deserialization roundtrips
//! - Coordinate extraction consistency
//! - NaN handling determinism

use delaunay::geometry::point::Point;
use delaunay::geometry::traits::coordinate::Coordinate;
use proptest::prelude::*;
use rustc_hash::FxHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates
fn finite_f64() -> impl Strategy<Value = f64> {
    (-1000.0..1000.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

/// Strategy for generating 2D points with finite coordinates
fn point_2d() -> impl Strategy<Value = Point<f64, 2>> {
    prop::array::uniform2(finite_f64()).prop_map(Point::new)
}

/// Strategy for generating 3D points with finite coordinates
fn point_3d() -> impl Strategy<Value = Point<f64, 3>> {
    prop::array::uniform3(finite_f64()).prop_map(Point::new)
}

/// Strategy for generating 4D points with finite coordinates
fn point_4d() -> impl Strategy<Value = Point<f64, 4>> {
    prop::array::uniform4(finite_f64()).prop_map(Point::new)
}

/// Strategy for generating 5D points with finite coordinates
fn point_5d() -> impl Strategy<Value = Point<f64, 5>> {
    prop::array::uniform5(finite_f64()).prop_map(Point::new)
}

/// Helper function to compute hash of a point
fn hash_point<T: Hash>(point: &T) -> u64 {
    let mut hasher = FxHasher::default();
    point.hash(&mut hasher);
    hasher.finish()
}

// =============================================================================
// EQUALITY AND HASH CONSISTENCY TESTS
// =============================================================================

proptest! {
    /// Property: Equal points must have equal hashes (2D)
    /// This is a fundamental requirement for using Points as HashMap keys.
    #[test]
    fn prop_equal_points_equal_hashes_2d(coords in prop::array::uniform2(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);

        prop_assert_eq!(point1, point2, "Points with same coordinates should be equal");
        prop_assert_eq!(
            hash_point(&point1),
            hash_point(&point2),
            "Equal points must have equal hashes"
        );
    }

    /// Property: Equal points must have equal hashes (3D)
    #[test]
    fn prop_equal_points_equal_hashes_3d(coords in prop::array::uniform3(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);

        prop_assert_eq!(point1, point2);
        prop_assert_eq!(hash_point(&point1), hash_point(&point2));
    }

    /// Property: Equal points must have equal hashes (4D)
    #[test]
    fn prop_equal_points_equal_hashes_4d(coords in prop::array::uniform4(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);

        prop_assert_eq!(point1, point2);
        prop_assert_eq!(hash_point(&point1), hash_point(&point2));
    }

    /// Property: Equal points must have equal hashes (5D)
    #[test]
    fn prop_equal_points_equal_hashes_5d(coords in prop::array::uniform5(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);

        prop_assert_eq!(point1, point2);
        prop_assert_eq!(hash_point(&point1), hash_point(&point2));
    }

    /// Property: Reflexivity - a point equals itself (2D)
    #[test]
    fn prop_equality_reflexive_2d(point in point_2d()) {
        prop_assert_eq!(point, point, "Point should equal itself");
    }

    /// Property: Reflexivity - a point equals itself (3D)
    #[test]
    fn prop_equality_reflexive_3d(point in point_3d()) {
        prop_assert_eq!(point, point, "Point should equal itself");
    }

    /// Property: Reflexivity - a point equals itself (4D)
    #[test]
    fn prop_equality_reflexive_4d(point in point_4d()) {
        prop_assert_eq!(point, point, "Point should equal itself");
    }

    /// Property: Reflexivity - a point equals itself (5D)
    #[test]
    fn prop_equality_reflexive_5d(point in point_5d()) {
        prop_assert_eq!(point, point, "Point should equal itself");
    }

    /// Property: Symmetry - if a == b, then b == a (2D)
    #[test]
    fn prop_equality_symmetric_2d(coords in prop::array::uniform2(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);

        prop_assert_eq!(point1, point2);
        prop_assert_eq!(point2, point1, "Equality should be symmetric");
    }

    /// Property: Symmetry - if a == b, then b == a (3D)
    #[test]
    fn prop_equality_symmetric_3d(coords in prop::array::uniform3(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);

        prop_assert_eq!(point1, point2);
        prop_assert_eq!(point2, point1);
    }

    /// Property: Symmetry - if a == b, then b == a (4D)
    #[test]
    fn prop_equality_symmetric_4d(coords in prop::array::uniform4(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);

        prop_assert_eq!(point1, point2);
        prop_assert_eq!(point2, point1);
    }

    /// Property: Symmetry - if a == b, then b == a (5D)
    #[test]
    fn prop_equality_symmetric_5d(coords in prop::array::uniform5(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);

        prop_assert_eq!(point1, point2);
        prop_assert_eq!(point2, point1);
    }

    /// Property: Transitivity - if a == b and b == c, then a == c (2D)
    #[test]
    fn prop_equality_transitive_2d(coords in prop::array::uniform2(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);
        let point3 = Point::new(coords);

        prop_assert_eq!(point1, point2);
        prop_assert_eq!(point2, point3);
        prop_assert_eq!(point1, point3, "Equality should be transitive");
    }

    /// Property: Transitivity - if a == b and b == c, then a == c (3D)
    #[test]
    fn prop_equality_transitive_3d(coords in prop::array::uniform3(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);
        let point3 = Point::new(coords);

        prop_assert_eq!(point1, point2);
        prop_assert_eq!(point2, point3);
        prop_assert_eq!(point1, point3, "Equality should be transitive");
    }

    /// Property: Transitivity - if a == b and b == c, then a == c (4D)
    #[test]
    fn prop_equality_transitive_4d(coords in prop::array::uniform4(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);
        let point3 = Point::new(coords);

        prop_assert_eq!(point1, point2);
        prop_assert_eq!(point2, point3);
        prop_assert_eq!(point1, point3);
    }

    /// Property: Transitivity - if a == b and b == c, then a == c (5D)
    #[test]
    fn prop_equality_transitive_5d(coords in prop::array::uniform5(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);
        let point3 = Point::new(coords);

        prop_assert_eq!(point1, point2);
        prop_assert_eq!(point2, point3);
        prop_assert_eq!(point1, point3);
    }

    /// Property: Points can be used as HashMap keys
    #[test]
    fn prop_hashmap_key_usage_2d(
        point1 in point_2d(),
        point2 in point_2d(),
        value1: u32,
        value2: u32,
    ) {
        let mut map = HashMap::new();
        map.insert(point1, value1);
        map.insert(point2, value2);

        // Check HashMap behavior based on key equality
        if point1 == point2 {
            // Equal keys: second insert overwrites value
            prop_assert_eq!(map.len(), 1, "Equal points should map to same entry");
            prop_assert_eq!(map.get(&point1), Some(&value2), "Second insert should overwrite");
            prop_assert_eq!(map.get(&point2), Some(&value2), "Equal keys map to same value");
        } else {
            // Distinct keys: both entries present
            prop_assert_eq!(map.len(), 2, "Different points should create separate entries");
            prop_assert_eq!(map.get(&point1), Some(&value1));
            prop_assert_eq!(map.get(&point2), Some(&value2));
        }
    }
}

// =============================================================================
// COORDINATE EXTRACTION TESTS
// =============================================================================

proptest! {
    /// Property: Coordinate extraction roundtrip - coords() matches construction (2D)
    #[test]
    fn prop_coordinate_roundtrip_2d(coords in prop::array::uniform2(finite_f64())) {
        let point = Point::new(coords);
        for (i, &coord) in coords.iter().enumerate() {
            prop_assert!(approx::relative_eq!(point.coords()[i], coord, epsilon = 1e-10));
        }
    }

    /// Property: Coordinate extraction roundtrip - coords() matches construction (3D)
    #[test]
    fn prop_coordinate_roundtrip_3d(coords in prop::array::uniform3(finite_f64())) {
        let point = Point::new(coords);
        for (i, &coord) in coords.iter().enumerate() {
            prop_assert!(approx::relative_eq!(point.coords()[i], coord, epsilon = 1e-10));
        }
    }

    /// Property: Coordinate extraction roundtrip - coords() matches construction (4D)
    #[test]
    fn prop_coordinate_roundtrip_4d(coords in prop::array::uniform4(finite_f64())) {
        let point = Point::new(coords);
        for (i, &coord) in coords.iter().enumerate() {
            prop_assert!(approx::relative_eq!(point.coords()[i], coord, epsilon = 1e-10));
        }
    }

    /// Property: Coordinate extraction roundtrip - coords() matches construction (5D)
    #[test]
    fn prop_coordinate_roundtrip_5d(coords in prop::array::uniform5(finite_f64())) {
        let point = Point::new(coords);
        for (i, &coord) in coords.iter().enumerate() {
            prop_assert!(approx::relative_eq!(point.coords()[i], coord, epsilon = 1e-10));
        }
    }

    /// Property: Into<[T; D]> conversion matches coords() (2D)
    #[test]
    fn prop_into_conversion_matches_coords_2d(coords in prop::array::uniform2(finite_f64())) {
        let point = Point::new(coords);
        let extracted: [f64; 2] = point.into();
        for i in 0..2 {
            prop_assert!(approx::relative_eq!(extracted[i], point.coords()[i], epsilon = 1e-10));
            prop_assert!(approx::relative_eq!(extracted[i], coords[i], epsilon = 1e-10));
        }
    }

    /// Property: Into<[T; D]> conversion matches coords() (3D)
    #[test]
    fn prop_into_conversion_matches_coords_3d(coords in prop::array::uniform3(finite_f64())) {
        let point = Point::new(coords);
        let extracted: [f64; 3] = point.into();
        for i in 0..3 {
            prop_assert!(approx::relative_eq!(extracted[i], point.coords()[i], epsilon = 1e-10));
            prop_assert!(approx::relative_eq!(extracted[i], coords[i], epsilon = 1e-10));
        }
    }

    /// Property: Into<[T; D]> conversion matches coords() (4D)
    #[test]
    fn prop_into_conversion_matches_coords_4d(coords in prop::array::uniform4(finite_f64())) {
        let point = Point::new(coords);
        let extracted: [f64; 4] = point.into();
        for i in 0..4 {
            prop_assert!(approx::relative_eq!(extracted[i], point.coords()[i], epsilon = 1e-10));
            prop_assert!(approx::relative_eq!(extracted[i], coords[i], epsilon = 1e-10));
        }
    }

    /// Property: Into<[T; D]> conversion matches coords() (5D)
    #[test]
    fn prop_into_conversion_matches_coords_5d(coords in prop::array::uniform5(finite_f64())) {
        let point = Point::new(coords);
        let extracted: [f64; 5] = point.into();
        for i in 0..5 {
            prop_assert!(approx::relative_eq!(extracted[i], point.coords()[i], epsilon = 1e-10));
            prop_assert!(approx::relative_eq!(extracted[i], coords[i], epsilon = 1e-10));
        }
    }

    /// Property: get() method consistency with array indexing (2D)
    #[test]
    fn prop_get_method_consistency_2d(coords in prop::array::uniform2(finite_f64())) {
        let point = Point::new(coords);

        for (i, &coord) in coords.iter().enumerate() {
            prop_assert_eq!(
                point.get(i),
                Some(coord),
                "get({}) should return coords[{}]", i, i
            );
        }

        prop_assert_eq!(point.get(2), None, "get(2) should return None for 2D point");
        prop_assert_eq!(point.get(100), None, "get(100) should return None");
    }

    /// Property: get() method consistency with array indexing (3D)
    #[test]
    fn prop_get_method_consistency_3d(coords in prop::array::uniform3(finite_f64())) {
        let point = Point::new(coords);

        for (i, &coord) in coords.iter().enumerate() {
            prop_assert_eq!(point.get(i), Some(coord));
        }

        prop_assert_eq!(point.get(3), None);
        prop_assert_eq!(point.get(100), None);
    }

    /// Property: get() method consistency with array indexing (4D)
    #[test]
    fn prop_get_method_consistency_4d(coords in prop::array::uniform4(finite_f64())) {
        let point = Point::new(coords);

        for (i, &coord) in coords.iter().enumerate() {
            prop_assert_eq!(
                point.get(i),
                Some(coord),
                "get({}) should return coords[{}]", i, i
            );
        }

        prop_assert_eq!(point.get(4), None, "get(4) should return None for 4D point");
        prop_assert_eq!(point.get(100), None, "get(100) should return None");
    }

    /// Property: get() method consistency with array indexing (5D)
    #[test]
    fn prop_get_method_consistency_5d(coords in prop::array::uniform5(finite_f64())) {
        let point = Point::new(coords);

        for (i, &coord) in coords.iter().enumerate() {
            prop_assert_eq!(point.get(i), Some(coord));
        }

        prop_assert_eq!(point.get(5), None);
        prop_assert_eq!(point.get(100), None);
    }
}

// =============================================================================
// SERIALIZATION TESTS
// =============================================================================
// Note: JSON serialization can lose floating-point precision, so exact
// equality roundtrip tests are not universal properties. For now, we rely
// on unit tests that control precision explicitly.

// =============================================================================
// NAN HANDLING TESTS
// =============================================================================

proptest! {
    /// Property: NaN coordinates are handled consistently
    /// Points with NaN should equal themselves (custom equality semantics)
    #[test]
    fn prop_nan_handling_consistency_2d(finite_coord in finite_f64()) {
        let point_with_nan = Point::new([f64::NAN, finite_coord]);

        // Custom equality: NaN == NaN for Points
        prop_assert_eq!(point_with_nan, point_with_nan, "Point with NaN should equal itself");

        // Hash should be consistent
        let hash1 = hash_point(&point_with_nan);
        let hash2 = hash_point(&point_with_nan);
        prop_assert_eq!(hash1, hash2, "Hash of point with NaN should be consistent");

        // Can be used as HashMap key
        let mut map = HashMap::new();
        map.insert(point_with_nan, 42);
        prop_assert_eq!(map.get(&point_with_nan), Some(&42), "Point with NaN should work as HashMap key");
    }

    /// Property: Multiple NaN points should be equal (all NaN bit patterns treated equal)
    #[test]
    fn prop_multiple_nan_points_equal_2d(finite_coord in finite_f64()) {
        let point1 = Point::new([f64::NAN, finite_coord]);
        let point2 = Point::new([f64::NAN, finite_coord]);

        // Even though created separately, NaN points should be equal
        prop_assert_eq!(point1, point2, "Different NaN points with same finite coords should be equal");
        prop_assert_eq!(hash_point(&point1), hash_point(&point2), "Should have equal hashes");
    }
}

// =============================================================================
// VALIDATION TESTS
// =============================================================================

proptest! {
    /// Property: Finite coordinates should validate successfully (2D)
    #[test]
    fn prop_finite_coordinates_validate_2d(coords in prop::array::uniform2(finite_f64())) {
        let point = Point::new(coords);
        prop_assert!(point.validate().is_ok(), "Finite coordinates should validate");
    }

    /// Property: Finite coordinates should validate successfully (3D)
    #[test]
    fn prop_finite_coordinates_validate_3d(coords in prop::array::uniform3(finite_f64())) {
        let point = Point::new(coords);
        prop_assert!(point.validate().is_ok());
    }

    /// Property: Finite coordinates should validate successfully (4D)
    #[test]
    fn prop_finite_coordinates_validate_4d(coords in prop::array::uniform4(finite_f64())) {
        let point = Point::new(coords);
        prop_assert!(point.validate().is_ok());
    }

    /// Property: Finite coordinates should validate successfully (5D)
    #[test]
    fn prop_finite_coordinates_validate_5d(coords in prop::array::uniform5(finite_f64())) {
        let point = Point::new(coords);
        prop_assert!(point.validate().is_ok());
    }

    /// Property: Points with infinite coordinates should fail validation (2D)
    #[test]
    fn prop_infinite_coordinates_fail_validation_2d(finite_coord in finite_f64()) {
        let point = Point::new([f64::INFINITY, finite_coord]);
        prop_assert!(point.validate().is_err(), "Infinite coordinate should fail validation");

        let point_neg_inf = Point::new([f64::NEG_INFINITY, finite_coord]);
        prop_assert!(point_neg_inf.validate().is_err(), "Negative infinity should fail validation");
    }

    /// Property: Points with infinite coordinates should fail validation (3D)
    #[test]
    fn prop_infinite_coordinates_fail_validation_3d(
        finite_coord1 in finite_f64(),
        finite_coord2 in finite_f64(),
    ) {
        let point = Point::new([f64::INFINITY, finite_coord1, finite_coord2]);
        prop_assert!(point.validate().is_err(), "Infinite coordinate should fail validation");

        let point_neg_inf = Point::new([f64::NEG_INFINITY, finite_coord1, finite_coord2]);
        prop_assert!(point_neg_inf.validate().is_err(), "Negative infinity should fail validation");
    }

    /// Property: Points with infinite coordinates should fail validation (4D)
    #[test]
    fn prop_infinite_coordinates_fail_validation_4d(
        finite_coord1 in finite_f64(),
        finite_coord2 in finite_f64(),
        finite_coord3 in finite_f64(),
    ) {
        let point = Point::new([f64::INFINITY, finite_coord1, finite_coord2, finite_coord3]);
        prop_assert!(point.validate().is_err());

        let point_neg_inf = Point::new([f64::NEG_INFINITY, finite_coord1, finite_coord2, finite_coord3]);
        prop_assert!(point_neg_inf.validate().is_err());
    }

    /// Property: Points with infinite coordinates should fail validation (5D)
    #[test]
    fn prop_infinite_coordinates_fail_validation_5d(
        finite_coord1 in finite_f64(),
        finite_coord2 in finite_f64(),
        finite_coord3 in finite_f64(),
        finite_coord4 in finite_f64(),
    ) {
        let point = Point::new([f64::INFINITY, finite_coord1, finite_coord2, finite_coord3, finite_coord4]);
        prop_assert!(point.validate().is_err());

        let point_neg_inf = Point::new([f64::NEG_INFINITY, finite_coord1, finite_coord2, finite_coord3, finite_coord4]);
        prop_assert!(point_neg_inf.validate().is_err());
    }

    /// Property: Points with NaN coordinates should fail validation (2D)
    #[test]
    fn prop_nan_coordinates_fail_validation_2d(finite_coord in finite_f64()) {
        let point = Point::new([f64::NAN, finite_coord]);
        prop_assert!(point.validate().is_err(), "NaN coordinate should fail validation");
    }

    /// Property: Points with NaN coordinates should fail validation (3D)
    #[test]
    fn prop_nan_coordinates_fail_validation_3d(
        finite_coord1 in finite_f64(),
        finite_coord2 in finite_f64(),
    ) {
        let point = Point::new([f64::NAN, finite_coord1, finite_coord2]);
        prop_assert!(point.validate().is_err());
    }

    /// Property: Points with NaN coordinates should fail validation (4D)
    #[test]
    fn prop_nan_coordinates_fail_validation_4d(
        finite_coord1 in finite_f64(),
        finite_coord2 in finite_f64(),
        finite_coord3 in finite_f64(),
    ) {
        let point = Point::new([f64::NAN, finite_coord1, finite_coord2, finite_coord3]);
        prop_assert!(point.validate().is_err());
    }

    /// Property: Points with NaN coordinates should fail validation (5D)
    #[test]
    fn prop_nan_coordinates_fail_validation_5d(
        finite_coord1 in finite_f64(),
        finite_coord2 in finite_f64(),
        finite_coord3 in finite_f64(),
        finite_coord4 in finite_f64(),
    ) {
        let point = Point::new([f64::NAN, finite_coord1, finite_coord2, finite_coord3, finite_coord4]);
        prop_assert!(point.validate().is_err());
    }
}

// =============================================================================
// ORDERING TESTS
// =============================================================================

proptest! {
    /// Property: Ordering is consistent with equality (2D)
    #[test]
    fn prop_ordering_consistent_with_equality_2d(coords in prop::array::uniform2(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);

        if point1 == point2 {
            prop_assert_eq!(
                point1.partial_cmp(&point2),
                Some(std::cmp::Ordering::Equal),
                "Equal points should compare as Equal"
            );
        }
    }

    /// Property: Ordering is consistent with equality (3D)
    #[test]
    fn prop_ordering_consistent_with_equality_3d(coords in prop::array::uniform3(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);

        if point1 == point2 {
            prop_assert_eq!(
                point1.partial_cmp(&point2),
                Some(std::cmp::Ordering::Equal)
            );
        }
    }

    /// Property: Ordering is consistent with equality (4D)
    #[test]
    fn prop_ordering_consistent_with_equality_4d(coords in prop::array::uniform4(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);

        if point1 == point2 {
            prop_assert_eq!(
                point1.partial_cmp(&point2),
                Some(std::cmp::Ordering::Equal)
            );
        }
    }

    /// Property: Ordering is consistent with equality (5D)
    #[test]
    fn prop_ordering_consistent_with_equality_5d(coords in prop::array::uniform5(finite_f64())) {
        let point1 = Point::new(coords);
        let point2 = Point::new(coords);

        if point1 == point2 {
            prop_assert_eq!(
                point1.partial_cmp(&point2),
                Some(std::cmp::Ordering::Equal)
            );
        }
    }

    /// Property: Ordering is antisymmetric (2D)
    #[test]
    fn prop_ordering_antisymmetric_2d(point1 in point_2d(), point2 in point_2d()) {
        use std::cmp::Ordering;

        match (point1.partial_cmp(&point2), point2.partial_cmp(&point1)) {
            (Some(Ordering::Less), Some(Ordering::Greater))
            | (Some(Ordering::Greater), Some(Ordering::Less))
            | (Some(Ordering::Equal), Some(Ordering::Equal)) => {},
            _ => prop_assert!(false, "Ordering should be antisymmetric"),
        }
    }

    /// Property: Ordering is antisymmetric (3D)
    #[test]
    fn prop_ordering_antisymmetric_3d(point1 in point_3d(), point2 in point_3d()) {
        use std::cmp::Ordering;

        match (point1.partial_cmp(&point2), point2.partial_cmp(&point1)) {
            (Some(Ordering::Less), Some(Ordering::Greater))
            | (Some(Ordering::Greater), Some(Ordering::Less))
            | (Some(Ordering::Equal), Some(Ordering::Equal)) => {},
            _ => prop_assert!(false, "Ordering should be antisymmetric"),
        }
    }

    /// Property: Ordering is antisymmetric (4D)
    #[test]
    fn prop_ordering_antisymmetric_4d(point1 in point_4d(), point2 in point_4d()) {
        use std::cmp::Ordering;

        match (point1.partial_cmp(&point2), point2.partial_cmp(&point1)) {
            (Some(Ordering::Less), Some(Ordering::Greater))
            | (Some(Ordering::Greater), Some(Ordering::Less))
            | (Some(Ordering::Equal), Some(Ordering::Equal)) => {},
            _ => prop_assert!(false, "Ordering should be antisymmetric"),
        }
    }

    /// Property: Ordering is antisymmetric (5D)
    #[test]
    fn prop_ordering_antisymmetric_5d(point1 in point_5d(), point2 in point_5d()) {
        use std::cmp::Ordering;

        match (point1.partial_cmp(&point2), point2.partial_cmp(&point1)) {
            (Some(Ordering::Less), Some(Ordering::Greater))
            | (Some(Ordering::Greater), Some(Ordering::Less))
            | (Some(Ordering::Equal), Some(Ordering::Equal)) => {},
            _ => prop_assert!(false, "Ordering should be antisymmetric"),
        }
    }

    /// Property: Ordering is transitive (2D)
    #[test]
    fn prop_ordering_transitive_2d(
        point1 in point_2d(),
        point2 in point_2d(),
        point3 in point_2d(),
    ) {
        use std::cmp::Ordering;

        if point1.partial_cmp(&point2) == Some(Ordering::Less)
            && point2.partial_cmp(&point3) == Some(Ordering::Less)
        {
            prop_assert_eq!(
                point1.partial_cmp(&point3),
                Some(Ordering::Less),
                "Ordering should be transitive"
            );
        }
    }

    /// Property: Ordering is transitive (3D)
    #[test]
    fn prop_ordering_transitive_3d(
        point1 in point_3d(),
        point2 in point_3d(),
        point3 in point_3d(),
    ) {
        use std::cmp::Ordering;

        if point1.partial_cmp(&point2) == Some(Ordering::Less)
            && point2.partial_cmp(&point3) == Some(Ordering::Less)
        {
            prop_assert_eq!(
                point1.partial_cmp(&point3),
                Some(Ordering::Less)
            );
        }
    }

    /// Property: Ordering is transitive (4D)
    #[test]
    fn prop_ordering_transitive_4d(
        point1 in point_4d(),
        point2 in point_4d(),
        point3 in point_4d(),
    ) {
        use std::cmp::Ordering;

        if point1.partial_cmp(&point2) == Some(Ordering::Less)
            && point2.partial_cmp(&point3) == Some(Ordering::Less)
        {
            prop_assert_eq!(
                point1.partial_cmp(&point3),
                Some(Ordering::Less)
            );
        }
    }

    /// Property: Ordering is transitive (5D)
    #[test]
    fn prop_ordering_transitive_5d(
        point1 in point_5d(),
        point2 in point_5d(),
        point3 in point_5d(),
    ) {
        use std::cmp::Ordering;

        if point1.partial_cmp(&point2) == Some(Ordering::Less)
            && point2.partial_cmp(&point3) == Some(Ordering::Less)
        {
            prop_assert_eq!(
                point1.partial_cmp(&point3),
                Some(Ordering::Less)
            );
        }
    }
}
