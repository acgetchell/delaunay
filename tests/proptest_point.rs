//! Property-based tests for Point operations.
//!
//! This module uses proptest to verify fundamental properties of Point
//! data structures, including:
//! - Hash consistency (equal points have equal hashes)
//! - Equality reflexivity, symmetry, and transitivity
//! - Serialization/deserialization roundtrips
//! - Coordinate extraction consistency
//! - NaN handling determinism

use delaunay::prelude::*;
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
macro_rules! gen_equal_points_equal_hashes {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_equal_points_equal_hashes_ $dim d>](coords in prop::array::[<uniform $dim>](finite_f64())) {
                    let p1 = Point::new(coords);
                    let p2 = Point::new(coords);
                    prop_assert_eq!(p1, p2);
                    prop_assert_eq!(hash_point(&p1), hash_point(&p2));
                }
            }
        }
    };
}

macro_rules! gen_equality_reflexive {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_equality_reflexive_ $dim d>](point in [<point_ $dim d>]()) {
                    prop_assert_eq!(point, point);
                }
            }
        }
    };
}

macro_rules! gen_equality_symmetric {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_equality_symmetric_ $dim d>](coords in prop::array::[<uniform $dim>](finite_f64())) {
                    let p1 = Point::new(coords);
                    let p2 = Point::new(coords);
                    prop_assert_eq!(p1, p2);
                    prop_assert_eq!(p2, p1);
                }
            }
        }
    };
}

macro_rules! gen_equality_transitive {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_equality_transitive_ $dim d>](coords in prop::array::[<uniform $dim>](finite_f64())) {
                    let p1 = Point::new(coords);
                    let p2 = Point::new(coords);
                    let p3 = Point::new(coords);
                    prop_assert_eq!(p1, p2);
                    prop_assert_eq!(p2, p3);
                    prop_assert_eq!(p1, p3);
                }
            }
        }
    };
}

macro_rules! gen_hashmap_key_usage {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_hashmap_key_usage_ $dim d>](
                    point1 in [<point_ $dim d>](),
                    point2 in [<point_ $dim d>](),
                    value1: u32,
                    value2: u32,
                ) {
                    let mut map = HashMap::new();
                    map.insert(point1, value1);
                    map.insert(point2, value2);
                    if point1 == point2 {
                        prop_assert_eq!(map.len(), 1);
                        prop_assert_eq!(map.get(&point1), Some(&value2));
                        prop_assert_eq!(map.get(&point2), Some(&value2));
                    } else {
                        prop_assert_eq!(map.len(), 2);
                        prop_assert_eq!(map.get(&point1), Some(&value1));
                        prop_assert_eq!(map.get(&point2), Some(&value2));
                    }
                }
            }
        }
    };
}

macro_rules! gen_coordinate_roundtrip {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_coordinate_roundtrip_ $dim d>](coords in prop::array::[<uniform $dim>](finite_f64())) {
                    let point = Point::new(coords);
                    for (i, &coord) in coords.iter().enumerate() {
                        prop_assert!(approx::relative_eq!(point.coords()[i], coord, epsilon = 1e-10));
                    }
                }
            }
        }
    };
}

macro_rules! gen_into_conversion_matches_coords {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_into_conversion_matches_coords_ $dim d>](coords in prop::array::[<uniform $dim>](finite_f64())) {
                    let point = Point::new(coords);
                    let extracted: [f64; $dim] = point.into();
                    for i in 0..$dim {
                        prop_assert!(approx::relative_eq!(extracted[i], point.coords()[i], epsilon = 1e-10));
                        prop_assert!(approx::relative_eq!(extracted[i], coords[i], epsilon = 1e-10));
                    }
                }
            }
        }
    };
}

macro_rules! gen_get_method_consistency {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_get_method_consistency_ $dim d>](coords in prop::array::[<uniform $dim>](finite_f64())) {
                    let point = Point::new(coords);
                    for (i, &coord) in coords.iter().enumerate() {
                        prop_assert_eq!(point.get(i), Some(coord));
                    }
                    prop_assert_eq!(point.get($dim), None);
                    prop_assert_eq!(point.get(100), None);
                }
            }
        }
    };
}

macro_rules! gen_finite_coordinates_validate {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_finite_coordinates_validate_ $dim d>](coords in prop::array::[<uniform $dim>](finite_f64())) {
                    let point = Point::new(coords);
                    prop_assert!(point.validate().is_ok());
                }
            }
        }
    };
}

macro_rules! gen_infinite_coordinates_fail_validation {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_infinite_coordinates_fail_validation_ $dim d>](coords in prop::array::[<uniform $dim>](finite_f64())) {
                    let mut arr = coords;
                    arr[0] = f64::INFINITY;
                    let point = Point::new(arr);
                    prop_assert!(point.validate().is_err());
                    let mut arr2 = coords;
                    arr2[0] = f64::NEG_INFINITY;
                    let point_neg = Point::new(arr2);
                    prop_assert!(point_neg.validate().is_err());
                }
            }
        }
    };
}

macro_rules! gen_nan_coordinates_fail_validation {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_nan_coordinates_fail_validation_ $dim d>](coords in prop::array::[<uniform $dim>](finite_f64())) {
                    let mut arr = coords;
                    arr[0] = f64::NAN;
                    let point = Point::new(arr);
                    prop_assert!(point.validate().is_err());
                }
            }
        }
    };
}

macro_rules! gen_ordering_equal_consistency {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_ordering_consistent_with_equality_ $dim d>](coords in prop::array::[<uniform $dim>](finite_f64())) {
                    let p1 = Point::new(coords);
                    let p2 = Point::new(coords);
                    if p1 == p2 {
                        prop_assert_eq!(p1.partial_cmp(&p2), Some(std::cmp::Ordering::Equal));
                    }
                }
            }
        }
    };
}

macro_rules! gen_ordering_antisymmetric {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_ordering_antisymmetric_ $dim d>](p1 in [<point_ $dim d>](), p2 in [<point_ $dim d>]()) {
                    use std::cmp::Ordering;
                    match (p1.partial_cmp(&p2), p2.partial_cmp(&p1)) {
                        (Some(Ordering::Less), Some(Ordering::Greater))
                        | (Some(Ordering::Greater), Some(Ordering::Less))
                        | (Some(Ordering::Equal), Some(Ordering::Equal)) => {}
                        _ => prop_assert!(false, "Ordering should be antisymmetric"),
                    }
                }
            }
        }
    };
}

macro_rules! gen_ordering_transitive {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_ordering_transitive_ $dim d>](p1 in [<point_ $dim d>](), p2 in [<point_ $dim d>](), p3 in [<point_ $dim d>]()) {
                    use std::cmp::Ordering;
                    if p1.partial_cmp(&p2) == Some(Ordering::Less) && p2.partial_cmp(&p3) == Some(Ordering::Less) {
                        prop_assert_eq!(p1.partial_cmp(&p3), Some(Ordering::Less));
                    }
                }
            }
        }
    };
}

macro_rules! gen_nan_handling_consistency {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_nan_handling_consistency_ $dim d>](finite_coord in finite_f64()) {
                    let mut arr = [finite_coord; $dim];
                    arr[0] = f64::NAN;
                    let p = Point::new(arr);
                    prop_assert_eq!(p, p);
                    let h1 = hash_point(&p); let h2 = hash_point(&p);
                    prop_assert_eq!(h1, h2);
                }
                #[test]
                fn [<prop_multiple_nan_points_equal_ $dim d>](finite_coord in finite_f64()) {
                    let mut arr = [finite_coord; $dim];
                    arr[0] = f64::NAN;
                    let p1 = Point::new(arr);
                    let p2 = Point::new(arr);
                    prop_assert_eq!(p1, p2);
                    prop_assert_eq!(hash_point(&p1), hash_point(&p2));
                }
            }
        }
    };
}

// =============================================================================
// EQUALITY AND HASH CONSISTENCY TESTS
// =============================================================================

gen_equal_points_equal_hashes!(2);
gen_equal_points_equal_hashes!(3);
gen_equal_points_equal_hashes!(4);
gen_equal_points_equal_hashes!(5);

gen_equality_reflexive!(2);
gen_equality_reflexive!(3);
gen_equality_reflexive!(4);
gen_equality_reflexive!(5);

gen_equality_symmetric!(2);
gen_equality_symmetric!(3);
gen_equality_symmetric!(4);
gen_equality_symmetric!(5);

gen_equality_transitive!(2);
gen_equality_transitive!(3);
gen_equality_transitive!(4);
gen_equality_transitive!(5);

gen_hashmap_key_usage!(2);

// =============================================================================
// COORDINATE EXTRACTION TESTS
// =============================================================================

gen_coordinate_roundtrip!(2);
gen_coordinate_roundtrip!(3);
gen_coordinate_roundtrip!(4);
gen_coordinate_roundtrip!(5);

gen_into_conversion_matches_coords!(2);
gen_into_conversion_matches_coords!(3);
gen_into_conversion_matches_coords!(4);
gen_into_conversion_matches_coords!(5);

gen_get_method_consistency!(2);
gen_get_method_consistency!(3);
gen_get_method_consistency!(4);
gen_get_method_consistency!(5);

// =============================================================================
// NAN HANDLING TESTS
// =============================================================================

gen_nan_handling_consistency!(2);
gen_nan_handling_consistency!(3);
gen_nan_handling_consistency!(4);
gen_nan_handling_consistency!(5);

// =============================================================================
// VALIDATION TESTS
// =============================================================================

gen_finite_coordinates_validate!(2);
gen_finite_coordinates_validate!(3);
gen_finite_coordinates_validate!(4);
gen_finite_coordinates_validate!(5);

gen_infinite_coordinates_fail_validation!(2);
gen_infinite_coordinates_fail_validation!(3);
gen_infinite_coordinates_fail_validation!(4);
gen_infinite_coordinates_fail_validation!(5);

gen_nan_coordinates_fail_validation!(2);
gen_nan_coordinates_fail_validation!(3);
gen_nan_coordinates_fail_validation!(4);
gen_nan_coordinates_fail_validation!(5);

// =============================================================================
// ORDERING TESTS
// =============================================================================

gen_ordering_equal_consistency!(2);
gen_ordering_equal_consistency!(3);
gen_ordering_equal_consistency!(4);
gen_ordering_equal_consistency!(5);

gen_ordering_antisymmetric!(2);
gen_ordering_antisymmetric!(3);
gen_ordering_antisymmetric!(4);
gen_ordering_antisymmetric!(5);

gen_ordering_transitive!(2);
gen_ordering_transitive!(3);
gen_ordering_transitive!(4);
gen_ordering_transitive!(5);
