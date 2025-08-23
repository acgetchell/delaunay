//! Data and operations on d-dimensional points.
//!
//! # Special Floating-Point Equality Semantics
//!
//! This module implements custom equality semantics for floating-point coordinates
//! that differ from the IEEE 754 standard. Specifically, `NaN` values are treated
//! as equal to themselves to satisfy the requirements of the `Eq` trait and enable
//! Points to be used as keys in hash-based collections.
//!
//! This means that for Points containing floating-point coordinates:
//! - `Point::new([f64::NAN]) == Point::new([f64::NAN])` returns `true`
//! - Points with NaN values can be used as `HashMap` keys
//! - All NaN bit patterns are considered equal
//!
//! If you need standard IEEE 754 equality semantics, compare the coordinates
//! directly instead of using Point equality.

#![allow(clippy::similar_names)]

// =============================================================================
// IMPORTS
// =============================================================================

use crate::geometry::traits::coordinate::{
    Coordinate, CoordinateConversionError, CoordinateScalar, CoordinateValidationError,
};
use num_traits::cast;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::hash::{Hash, Hasher};

// =============================================================================
// POINT STRUCT DEFINITION
// =============================================================================

#[derive(Clone, Copy, Debug, PartialOrd)]
/// The [Point] struct represents a point in a D-dimensional space, where the
/// coordinates are of generic type `T`.
///
/// # Generic Type Support
///
/// The Point struct supports any floating-point type `T` that implements the
/// `CoordinateScalar` trait, including `f32`, `f64`, and other floating-point
/// types. This generalization allows for flexibility in precision requirements
/// and memory usage across different applications.
///
/// # Properties
///
/// * `coords`: `coords` is a private property of the [Point]. It is an array of
///   type `T` with a length of `D`. The type `T` is a generic type parameter
///   constrained to implement `CoordinateScalar`, ensuring it has all necessary
///   traits for coordinate operations. The length `D` is a constant unsigned
///   integer known at compile time.
///
/// Points are intended to be immutable once created, so the `coords` field is
/// private to prevent modification after instantiation.
pub struct Point<T, const D: usize>
where
    T: CoordinateScalar,
{
    /// The coordinates of the point.
    coords: [T; D],
}

/// Implementation of the `Coordinate` trait for Point
/// This allows Point to be used as a coordinate type directly
impl<T, const D: usize> Coordinate<T, D> for Point<T, D>
where
    T: CoordinateScalar,
{
    /// Create a new Point from an array of coordinates
    #[inline]
    fn new(coords: [T; D]) -> Self {
        Self { coords }
    }

    /// Extract the coordinates as an array
    #[inline]
    fn to_array(&self) -> [T; D] {
        self.coords
    }

    /// Get the coordinate at the specified index
    #[inline]
    fn get(&self, index: usize) -> Option<T> {
        self.coords.get(index).copied()
    }

    /// Validate that all coordinates are finite (no NaN or infinite values)
    fn validate(&self) -> Result<(), CoordinateValidationError> {
        // Verify all coordinates are finite
        for (index, &coord) in self.coords.iter().enumerate() {
            if !coord.is_finite_generic() {
                return Err(CoordinateValidationError::InvalidCoordinate {
                    coordinate_index: index,
                    coordinate_value: format!("{coord:?}"),
                    dimension: D,
                });
            }
        }
        Ok(())
    }

    /// Hash the coordinate values
    fn hash_coordinate<H: Hasher>(&self, state: &mut H) {
        for &coord in &self.coords {
            coord.hash_coord(state);
        }
    }

    /// Check if two coordinates are equal using `OrderedEq`
    fn ordered_equals(&self, other: &Self) -> bool {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .all(|(a, b)| a.ordered_eq(b))
    }
}

// =============================================================================
// STANDARD TRAIT IMPLEMENTATIONS
// =============================================================================

// Implement Hash using the Coordinate trait
impl<T, const D: usize> Hash for Point<T, D>
where
    T: CoordinateScalar,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_coordinate(state);
    }
}

// Implement PartialEq using the Coordinate trait
impl<T, const D: usize> PartialEq for Point<T, D>
where
    T: CoordinateScalar,
{
    fn eq(&self, other: &Self) -> bool {
        self.ordered_equals(other)
    }
}

// Implement Eq using the Coordinate trait
impl<T, const D: usize> Eq for Point<T, D> where T: CoordinateScalar {}

// Manual implementations for traits that can't be derived due to [T; D] limitations

// Implement Default manually
impl<T, const D: usize> Default for Point<T, D>
where
    T: CoordinateScalar + Default,
{
    fn default() -> Self {
        Self {
            coords: [T::default(); D],
        }
    }
}

// Implement Serialize manually
impl<T, const D: usize> Serialize for Point<T, D>
where
    T: CoordinateScalar + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeTuple;
        let mut tuple = serializer.serialize_tuple(D)?;
        for coord in &self.coords {
            tuple.serialize_element(coord)?;
        }
        tuple.end()
    }
}

// Implement Deserialize manually
impl<'de, T, const D: usize> Deserialize<'de> for Point<T, D>
where
    T: CoordinateScalar + DeserializeOwned,
{
    fn deserialize<DE>(deserializer: DE) -> Result<Self, DE::Error>
    where
        DE: serde::Deserializer<'de>,
    {
        use serde::de::{Error, SeqAccess, Visitor};
        use std::fmt;
        use std::marker::PhantomData;

        struct ArrayVisitor<T, const D: usize>(PhantomData<T>);

        impl<'de, T, const D: usize> Visitor<'de> for ArrayVisitor<T, D>
        where
            T: CoordinateScalar + DeserializeOwned,
        {
            type Value = Point<T, D>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_fmt(format_args!("an array of {D} coordinates"))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                // Collect coordinates into a Vec first, then convert to array
                let mut coords = Vec::with_capacity(D);
                for i in 0..D {
                    let coord = seq
                        .next_element()?
                        .ok_or_else(|| Error::invalid_length(i, &self))?;
                    coords.push(coord);
                }

                // Convert Vec to array
                let coords_len = coords.len();
                let coords_array: [T; D] = coords
                    .try_into()
                    .map_err(|_| Error::invalid_length(coords_len, &self))?;

                Ok(Point::new(coords_array))
            }
        }

        deserializer.deserialize_tuple(D, ArrayVisitor(PhantomData))
    }
}

// =============================================================================
// TYPE CONVERSION IMPLEMENTATIONS
// =============================================================================

/// Fallible conversions for Point from arrays with potentially different scalar types.
///
/// This replaces the previous infallible From<[T; D]> which silently defaulted on
/// cast failures. Now, conversions will return an error if any coordinate cannot be
/// cast into the target type, or if a non-finite value is encountered post-cast.
impl<T, U, const D: usize> TryFrom<[T; D]> for Point<U, D>
where
    T: cast::NumCast + std::fmt::Debug,
    U: CoordinateScalar + cast::NumCast,
{
    type Error = CoordinateConversionError;

    #[inline]
    fn try_from(coords: [T; D]) -> Result<Self, Self::Error> {
        let mut out: [U; D] = [U::zero(); D];
        for (i, c) in coords.into_iter().enumerate() {
            // Store debug representation before moving c
            let c_debug = format!("{c:?}");
            // Attempt numeric cast
            let v: U =
                cast::cast(c).ok_or_else(|| CoordinateConversionError::ConversionFailed {
                    coordinate_index: i,
                    coordinate_value: c_debug,
                    from_type: std::any::type_name::<T>(),
                    to_type: std::any::type_name::<U>(),
                })?;
            // Validate finiteness after cast
            if !v.is_finite_generic() {
                return Err(CoordinateConversionError::NonFiniteValue {
                    coordinate_index: i,
                    coordinate_value: format!("{v:?}"),
                });
            }
            out[i] = v;
        }
        Ok(Self::new(out))
    }
}

/// Enable conversions from Point to coordinate arrays - using Coordinate trait
impl<T, const D: usize> From<Point<T, D>> for [T; D]
where
    T: CoordinateScalar,
{
    /// # Example
    ///
    /// ```rust
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    /// let point = Point::new([1.0, 2.0]);
    /// let coords: [f64; 2] = point.into();
    /// assert_eq!(coords, [1.0, 2.0]);
    /// ```
    #[inline]
    fn from(point: Point<T, D>) -> [T; D] {
        point.to_array()
    }
}

impl<T, const D: usize> From<&Point<T, D>> for [T; D]
where
    T: CoordinateScalar,
{
    /// # Example
    ///
    /// ```rust
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    /// let point = Point::new([3.0, 4.0]);
    /// let coords: [f64; 2] = (&point).into();
    /// assert_eq!(coords, [3.0, 4.0]);
    /// ```
    #[inline]
    fn from(point: &Point<T, D>) -> [T; D] {
        point.to_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::collections::hash_map::DefaultHasher;
    use std::collections::{HashMap, HashSet};
    use std::hash::{Hash, Hasher};

    // Helper function to get hash value for any hashable type
    fn get_hash<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    // Helper function to test basic point properties
    fn test_basic_point_properties<T, const D: usize>(
        point: &Point<T, D>,
        expected_coords: [T; D],
        expected_dim: usize,
    ) where
        T: CoordinateScalar,
    {
        assert_eq!(point.to_array(), expected_coords);
        assert_eq!(point.dim(), expected_dim);
    }

    // Helper function to test point equality and hash consistency
    fn test_point_equality_and_hash<T, const D: usize>(
        point1: Point<T, D>,
        point2: Point<T, D>,
        should_be_equal: bool,
    ) where
        T: CoordinateScalar,
        Point<T, D>: Hash,
    {
        if should_be_equal {
            assert_eq!(point1, point2);
            assert_eq!(get_hash(&point1), get_hash(&point2));
        } else {
            assert_ne!(point1, point2);
            // Note: Different points may still hash to same value (hash collisions)
        }
    }

    // =============================================================================
    // BASIC POINT CREATION TESTS
    // =============================================================================

    #[test]
    fn point_default() {
        let point: Point<f64, 4> = Point::default();

        let coords = point.to_array();
        assert_relative_eq!(
            coords.as_slice(),
            [0.0, 0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-9
        );

        // Human readable output for cargo test -- --nocapture
        println!("Default: {point:?}");
    }

    #[test]
    fn point_new() {
        let point = Point::new([1.0, 2.0, 3.0, 4.0]);

        let coords = point.to_array();
        assert_relative_eq!(
            coords.as_slice(),
            [1.0, 2.0, 3.0, 4.0].as_slice(),
            epsilon = 1e-9
        );

        // Human readable output for cargo test -- --nocapture
        println!("Point: {point:?}");
    }

    #[test]
    fn point_copy() {
        let point = Point::new([1.0, 2.0, 3.0, 4.0]);
        let point_copy = point;

        assert_eq!(point, point_copy);
        let coords1 = point.to_array();
        let coords2 = point_copy.to_array();
        assert_relative_eq!(coords1.as_slice(), coords2.as_slice(), epsilon = 1e-9);
        assert_eq!(point.dim(), point_copy.dim());
    }

    #[test]
    fn point_dim() {
        let point = Point::new([1.0, 2.0, 3.0, 4.0]);

        assert_eq!(point.dim(), 4);

        // Human readable output for cargo test -- --nocapture
        println!("Point: {:?} is {}-D", point, point.dim());
    }

    #[test]
    fn point_origin() {
        let point: Point<f64, 4> = Point::origin();

        let coords = point.to_array();
        assert_relative_eq!(
            coords.as_slice(),
            [0.0, 0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-9
        );

        // Human readable output for cargo test -- --nocapture
        println!("Origin: {:?} is {}-D", point, point.dim());
    }

    #[test]
    fn point_serialization() {
        use serde_test::{Token, assert_tokens};

        let point: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);
        assert_tokens(
            &point,
            &[
                Token::Tuple { len: 3 },
                Token::F64(1.0),
                Token::F64(2.0),
                Token::F64(3.0),
                Token::TupleEnd,
            ],
        );
    }

    #[test]
    fn point_to_and_from_json() {
        let point: Point<f64, 4> = Point::default();
        let serialized = serde_json::to_string(&point).unwrap();

        assert_eq!(serialized, "[0.0,0.0,0.0,0.0]");

        let deserialized: Point<f64, 4> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized, point);

        // Human readable output for cargo test -- --nocapture
        println!("Serialized: {serialized:?}");
    }

    #[test]
    fn point_from_array_f32_to_f64() {
        let coords = [1.5f32, 2.5f32, 3.5f32, 4.5f32];
        let point: Point<f64, 4> = Point::new(coords.map(std::convert::Into::into));

        let result_coords = point.to_array();
        assert_relative_eq!(
            result_coords.as_slice(),
            [1.5, 2.5, 3.5, 4.5].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(point.dim(), 4);
    }

    #[test]
    fn point_from_array_same_type() {
        // Test conversion when source and target types are the same
        let coords_f32 = [1.0f32, 2.0f32, 3.0f32];
        let point_f32: Point<f32, 3> = Point::new(coords_f32);
        let result_f32 = point_f32.to_array();
        assert_relative_eq!(
            result_f32.as_slice(),
            [1.0f32, 2.0f32, 3.0f32].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn point_from_array_float_to_float() {
        // Test conversion from f32 to f32 (same type)
        let coords_f32 = [1.5f32, 2.5f32];
        let point_f32: Point<f32, 2> = Point::new(coords_f32);
        let result_f32 = point_f32.to_array();
        assert_relative_eq!(
            result_f32.as_slice(),
            [1.5f32, 2.5f32].as_slice(),
            epsilon = 1e-9
        );

        // Test conversion from f32 to f64 (safe upcast)
        let coords_f32 = [1.5f32, 2.5f32];
        let point_f64: Point<f64, 2> = Point::new(coords_f32.map(std::convert::Into::into));
        let result_f64 = point_f64.to_array();
        assert_relative_eq!(
            result_f64.as_slice(),
            [1.5f64, 2.5f64].as_slice(),
            epsilon = 1e-9
        );
    }

    // =============================================================================
    // HASH AND EQUALITY TESTS
    // =============================================================================

    #[test]
    fn point_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let point1 = Point::new([1.0, 2.0, 3.0]);
        let point2 = Point::new([1.0, 2.0, 3.0]);
        let point3 = Point::new([1.0, 2.0, 4.0]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        let mut hasher3 = DefaultHasher::new();

        point1.hash(&mut hasher1);
        point2.hash(&mut hasher2);
        point3.hash(&mut hasher3);

        // Same points should have same hash
        assert_eq!(hasher1.finish(), hasher2.finish());
        // Different points should have different hash (with high probability)
        assert_ne!(hasher1.finish(), hasher3.finish());
    }

    #[test]
    fn point_hash_in_hashmap() {
        use std::collections::HashMap;

        let mut map: HashMap<Point<f64, 2>, i32> = HashMap::new();

        let point1 = Point::new([1.0, 2.0]);
        let point2 = Point::new([3.0, 4.0]);
        let point3 = Point::new([1.0, 2.0]); // Same as point1

        map.insert(point1, 10);
        map.insert(point2, 20);

        assert_eq!(map.get(&point3), Some(&10)); // Should find point1's value
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn point_partial_eq() {
        let point1 = Point::new([1.0, 2.0, 3.0]);
        let point2 = Point::new([1.0, 2.0, 3.0]);
        let point3 = Point::new([1.0, 2.0, 4.0]);

        assert_eq!(point1, point2);
        assert_ne!(point1, point3);
        assert_ne!(point2, point3);
    }

    #[test]
    fn point_partial_ord() {
        let point1 = Point::new([1.0, 2.0, 3.0]);
        let point2 = Point::new([1.0, 2.0, 4.0]);
        let point3 = Point::new([1.0, 3.0, 0.0]);
        let point4 = Point::new([2.0, 0.0, 0.0]);

        // Lexicographic ordering
        assert!(point1 < point2); // 3.0 < 4.0 in last coordinate
        assert!(point1 < point3); // 2.0 < 3.0 in second coordinate
        assert!(point1 < point4); // 1.0 < 2.0 in first coordinate
        assert!(point2 > point1);
    }

    #[test]
    fn point_1d() {
        let point: Point<f64, 1> = Point::new([42.0]);
        test_basic_point_properties(&point, [42.0], 1);

        let origin: Point<f64, 1> = Point::origin();
        test_basic_point_properties(&origin, [0.0], 1);
    }

    #[test]
    fn point_2d() {
        let point: Point<f64, 2> = Point::new([1.0, 2.0]);
        test_basic_point_properties(&point, [1.0, 2.0], 2);

        let origin: Point<f64, 2> = Point::origin();
        test_basic_point_properties(&origin, [0.0, 0.0], 2);
    }

    #[test]
    fn point_3d() {
        let point: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);
        test_basic_point_properties(&point, [1.0, 2.0, 3.0], 3);

        let origin: Point<f64, 3> = Point::origin();
        test_basic_point_properties(&origin, [0.0, 0.0, 0.0], 3);
    }

    #[test]
    fn point_5d() {
        let point: Point<f64, 5> = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        test_basic_point_properties(&point, [1.0, 2.0, 3.0, 4.0, 5.0], 5);

        let origin: Point<f64, 5> = Point::origin();
        test_basic_point_properties(&origin, [0.0, 0.0, 0.0, 0.0, 0.0], 5);
    }

    #[test]
    fn point_with_f32() {
        let point: Point<f32, 2> = Point::new([1.5, 2.5]);

        let coords = point.to_array();
        assert_relative_eq!(coords.as_slice(), [1.5, 2.5].as_slice(), epsilon = 1e-9);
        assert_eq!(point.dim(), 2);

        let origin: Point<f32, 2> = Point::origin();
        let origin_coords = origin.to_array();
        assert_relative_eq!(
            origin_coords.as_slice(),
            [0.0, 0.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn point_debug_format() {
        let point = Point::new([1.0, 2.0, 3.0]);
        let debug_str = format!("{point:?}");

        assert!(debug_str.contains("Point"));
        assert!(debug_str.contains("coords"));
        assert!(debug_str.contains("1.0"));
        assert!(debug_str.contains("2.0"));
        assert!(debug_str.contains("3.0"));
    }

    #[test]
    fn point_eq_trait() {
        let point1 = Point::new([1.0, 2.0, 3.0]);
        let point2 = Point::new([1.0, 2.0, 3.0]);
        let point3 = Point::new([1.0, 2.0, 4.0]);

        // Test Eq trait (transitivity, reflexivity, symmetry)
        assert_eq!(point1, point1); // reflexive
        assert_eq!(point1, point2); // symmetric
        assert_eq!(point2, point1); // symmetric
        assert_ne!(point1, point3);
        assert_ne!(point3, point1);
    }

    #[test]
    fn point_comprehensive_serialization() {
        // Test with different types and dimensions
        let point_3d = Point::new([1.0, 2.0, 3.0]);
        let serialized_3d = serde_json::to_string(&point_3d).unwrap();
        let deserialized_3d: Point<f64, 3> = serde_json::from_str(&serialized_3d).unwrap();
        assert_eq!(point_3d, deserialized_3d);

        let point_2d = Point::new([10.5, -5.3]);
        let serialized_2d = serde_json::to_string(&point_2d).unwrap();
        let deserialized_2d: Point<f64, 2> = serde_json::from_str(&serialized_2d).unwrap();
        assert_eq!(point_2d, deserialized_2d);

        let point_1d = Point::new([42.0]);
        let serialized_1d = serde_json::to_string(&point_1d).unwrap();
        let deserialized_1d: Point<f64, 1> = serde_json::from_str(&serialized_1d).unwrap();
        assert_eq!(point_1d, deserialized_1d);
    }

    #[test]
    fn point_negative_coordinates() {
        let point = Point::new([-1.0, -2.0, -3.0]);

        assert_relative_eq!(
            point.to_array().as_slice(),
            [-1.0, -2.0, -3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(point.dim(), 3);

        // Test with mixed positive/negative
        let mixed_point = Point::new([1.0, -2.0, 3.0, -4.0]);
        assert_relative_eq!(
            mixed_point.to_array().as_slice(),
            [1.0, -2.0, 3.0, -4.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn point_zero_coordinates() {
        let zero_point = Point::new([0.0, 0.0, 0.0]);
        let origin: Point<f64, 3> = Point::origin();

        assert_eq!(zero_point, origin);
        assert_relative_eq!(
            zero_point.to_array().as_slice(),
            [0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn point_large_coordinates() {
        let large_point = Point::new([1e6, 2e6, 3e6]);

        let coords = large_point.to_array();
        assert_relative_eq!(
            coords.as_slice(),
            [1_000_000.0, 2_000_000.0, 3_000_000.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(large_point.dim(), 3);
    }

    #[test]
    fn point_small_coordinates() {
        let small_point = Point::new([1e-6, 2e-6, 3e-6]);

        let coords = small_point.to_array();
        assert_relative_eq!(
            coords.as_slice(),
            [0.000_001, 0.000_002, 0.000_003].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(small_point.dim(), 3);
    }

    #[test]
    fn point_ordering_edge_cases() {
        use std::cmp::Ordering;

        let point1 = Point::new([1.0, 2.0]);
        let point2 = Point::new([1.0, 2.0]);

        // Test that equal points are not less than each other
        assert_ne!(point1.partial_cmp(&point2), Some(Ordering::Less));
        assert_ne!(point2.partial_cmp(&point1), Some(Ordering::Less));
        assert!(point1 <= point2);
        assert!(point2 <= point1);
        assert!(point1 >= point2);
        assert!(point2 >= point1);
    }

    #[test]
    fn point_hash_f32() {
        use std::collections::HashMap;

        let mut map: HashMap<Point<f32, 2>, i32> = HashMap::new();

        let point1 = Point::new([1.5f32, 2.5f32]);
        let point2 = Point::new([3.5f32, 4.5f32]);
        let point3 = Point::new([1.5f32, 2.5f32]); // Same as point1

        map.insert(point1, 10);
        map.insert(point2, 20);

        assert_eq!(map.get(&point3), Some(&10)); // Should find point1's value
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn point_eq_different_types() {
        // Test Eq for f64
        let point_f64_1 = Point::new([1.0, 2.0]);
        let point_f64_2 = Point::new([1.0, 2.0]);
        let point_f64_3 = Point::new([1.0, 2.1]);

        assert_eq!(point_f64_1, point_f64_2);
        assert_ne!(point_f64_1, point_f64_3);

        // Test Eq for f32
        let point_f32_1 = Point::new([1.5f32, 2.5f32]);
        let point_f32_2 = Point::new([1.5f32, 2.5f32]);
        let point_f32_3 = Point::new([1.5f32, 2.6f32]);

        assert_eq!(point_f32_1, point_f32_2);
        assert_ne!(point_f32_1, point_f32_3);
    }

    #[test]
    fn point_hash_consistency_floating_point() {
        // Test that OrderedFloat provides consistent hashing for NaN-free floats
        let point1 = Point::new([1.0, 2.0, 3.5]);
        let point2 = Point::new([1.0, 2.0, 3.5]);
        test_point_equality_and_hash(point1, point2, true);

        // Test with f32
        let point_f32_1 = Point::new([1.5f32, 2.5f32]);
        let point_f32_2 = Point::new([1.5f32, 2.5f32]);
        test_point_equality_and_hash(point_f32_1, point_f32_2, true);
    }

    #[test]
    fn point_implicit_conversion_to_coordinates() {
        let point: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);

        // Test implicit conversion from owned point
        let coords_owned: [f64; 3] = point.into();
        assert_relative_eq!(coords_owned.as_slice(), [1.0, 2.0, 3.0].as_slice());

        // Create a new point for reference test
        let point_ref: Point<f64, 3> = Point::new([4.0, 5.0, 6.0]);

        // Test implicit conversion from point reference
        let coords_ref: [f64; 3] = (&point_ref).into();
        assert_relative_eq!(coords_ref.as_slice(), [4.0, 5.0, 6.0].as_slice());

        // Verify the original point is still available after reference conversion
        assert_relative_eq!(point_ref.to_array().as_slice(), [4.0, 5.0, 6.0].as_slice());
    }

    // =============================================================================
    // VALIDATION TESTS
    // =============================================================================

    #[test]
    fn point_is_valid_f64() {
        // Test valid f64 points
        let valid_point = Point::new([1.0, 2.0, 3.0]);
        assert!(valid_point.validate().is_ok());

        let valid_negative = Point::new([-1.0, -2.0, -3.0]);
        assert!(valid_negative.validate().is_ok());

        let valid_zero = Point::new([0.0, 0.0, 0.0]);
        assert!(valid_zero.validate().is_ok());

        let valid_mixed = Point::new([1.0, -2.5, 0.0, 42.7]);
        assert!(valid_mixed.validate().is_ok());

        // Test invalid f64 points with NaN
        let invalid_nan_single = Point::new([1.0, f64::NAN, 3.0]);
        assert!(invalid_nan_single.validate().is_err());

        let invalid_nan_all = Point::new([f64::NAN, f64::NAN, f64::NAN]);
        assert!(invalid_nan_all.validate().is_err());

        let invalid_nan_first = Point::new([f64::NAN, 2.0, 3.0]);
        assert!(invalid_nan_first.validate().is_err());

        let invalid_nan_last = Point::new([1.0, 2.0, f64::NAN]);
        assert!(invalid_nan_last.validate().is_err());

        // Test invalid f64 points with infinity
        let invalid_pos_inf = Point::new([1.0, f64::INFINITY, 3.0]);
        assert!(invalid_pos_inf.validate().is_err());

        let invalid_neg_inf = Point::new([1.0, f64::NEG_INFINITY, 3.0]);
        assert!(invalid_neg_inf.validate().is_err());

        let invalid_both_inf = Point::new([f64::INFINITY, f64::NEG_INFINITY]);
        assert!(invalid_both_inf.validate().is_err());

        // Test mixed invalid cases
        let invalid_nan_and_inf = Point::new([f64::NAN, f64::INFINITY, 1.0]);
        assert!(invalid_nan_and_inf.validate().is_err());
    }

    #[test]
    fn point_is_valid_f32() {
        // Test valid f32 points
        let valid_point = Point::new([1.0f32, 2.0f32, 3.0f32]);
        assert!(valid_point.validate().is_ok());

        let valid_negative = Point::new([-1.5f32, -2.5f32]);
        assert!(valid_negative.validate().is_ok());

        let valid_zero = Point::new([0.0f32]);
        assert!(valid_zero.validate().is_ok());

        // Test invalid f32 points with NaN
        let invalid_nan = Point::new([1.0f32, f32::NAN]);
        assert!(invalid_nan.validate().is_err());

        let invalid_all_nan = Point::new([f32::NAN, f32::NAN, f32::NAN, f32::NAN]);
        assert!(invalid_all_nan.validate().is_err());

        // Test invalid f32 points with infinity
        let invalid_pos_inf = Point::new([f32::INFINITY, 2.0f32]);
        assert!(invalid_pos_inf.validate().is_err());

        let invalid_neg_inf = Point::new([1.0f32, f32::NEG_INFINITY]);
        assert!(invalid_neg_inf.validate().is_err());

        // Test edge cases with very small and large values (but finite)
        let valid_small = Point::new([f32::MIN_POSITIVE, -f32::MIN_POSITIVE]);
        assert!(valid_small.validate().is_ok());

        let valid_large = Point::new([f32::MAX, -f32::MAX]);
        assert!(valid_large.validate().is_ok());
    }

    #[test]
    fn point_is_valid_different_dimensions() {
        // Test 1D points
        let valid_1d_f64 = Point::new([42.0]);
        assert!(valid_1d_f64.validate().is_ok());

        let invalid_1d_nan = Point::new([f64::NAN]);
        assert!(invalid_1d_nan.validate().is_err());

        // Test 2D points
        let valid_2d = Point::new([1.0, 2.0]);
        assert!(valid_2d.validate().is_ok());

        let invalid_2d = Point::new([1.0, f64::INFINITY]);
        assert!(invalid_2d.validate().is_err());

        // Test higher dimensional points
        let valid_5d = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(valid_5d.validate().is_ok());

        let invalid_5d = Point::new([1.0, 2.0, f64::NAN, 4.0, 5.0]);
        assert!(invalid_5d.validate().is_err());

        // Test 10D point
        let valid_10d = Point::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert!(valid_10d.validate().is_ok());

        let invalid_10d = Point::new([
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            f64::NEG_INFINITY,
            7.0,
            8.0,
            9.0,
            10.0,
        ]);
        assert!(invalid_10d.validate().is_err());
    }

    #[test]
    fn point_is_valid_edge_cases() {
        // Test with very small finite values
        let tiny_valid = Point::new([f64::MIN_POSITIVE, -f64::MIN_POSITIVE, 0.0]);
        assert!(tiny_valid.validate().is_ok());

        // Test with very large finite values
        let large_valid = Point::new([f64::MAX, -f64::MAX]);
        assert!(large_valid.validate().is_ok());

        // Test subnormal numbers (should be valid)
        let subnormal = f64::MIN_POSITIVE / 2.0;
        let subnormal_point = Point::new([subnormal, -subnormal]);
        assert!(subnormal_point.validate().is_ok());

        // Test zero and negative zero
        let zero_point = Point::new([0.0, -0.0]);
        assert!(zero_point.validate().is_ok());

        // Mixed valid and invalid in same point should be invalid
        let mixed_invalid = Point::new([1.0, 2.0, 3.0, f64::NAN, 5.0]);
        assert!(mixed_invalid.validate().is_err());

        // All coordinates must be valid for point to be valid
        let one_invalid = Point::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, f64::INFINITY]);
        assert!(one_invalid.validate().is_err());
    }

    #[test]
    fn point_nan_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Test that OrderedFloat provides consistent hashing for NaN values
        // Note: Equality comparison for NaN still follows IEEE standard (NaN != NaN)
        // but hashing uses OrderedFloat which treats all NaN values as equivalent

        let point_nan1 = Point::new([f64::NAN, 2.0]);
        let point_nan2 = Point::new([f64::NAN, 2.0]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        point_nan1.hash(&mut hasher1);
        point_nan2.hash(&mut hasher2);

        // NaN points with same structure should hash to same value
        assert_eq!(hasher1.finish(), hasher2.finish());

        // Test with f32 NaN
        let point_f32_nan1 = Point::new([f32::NAN, 1.0f32]);
        let point_f32_nan2 = Point::new([f32::NAN, 1.0f32]);

        let mut hasher_f32_1 = DefaultHasher::new();
        let mut hasher_f32_2 = DefaultHasher::new();

        point_f32_nan1.hash(&mut hasher_f32_1);
        point_f32_nan2.hash(&mut hasher_f32_2);

        assert_eq!(hasher_f32_1.finish(), hasher_f32_2.finish());
    }

    #[test]
    fn point_infinity_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Test that OrderedFloat provides consistent hashing for infinity values
        let point_pos_inf1 = Point::new([f64::INFINITY, 2.0]);
        let point_pos_inf2 = Point::new([f64::INFINITY, 2.0]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        point_pos_inf1.hash(&mut hasher1);
        point_pos_inf2.hash(&mut hasher2);

        // Same infinity points should hash to same value
        assert_eq!(hasher1.finish(), hasher2.finish());

        // Test negative infinity
        let point_neg_inf1 = Point::new([f64::NEG_INFINITY, 2.0]);
        let point_neg_inf2 = Point::new([f64::NEG_INFINITY, 2.0]);

        let mut hasher_neg1 = DefaultHasher::new();
        let mut hasher_neg2 = DefaultHasher::new();

        point_neg_inf1.hash(&mut hasher_neg1);
        point_neg_inf2.hash(&mut hasher_neg2);

        assert_eq!(hasher_neg1.finish(), hasher_neg2.finish());

        // Positive and negative infinity should hash differently
        assert_ne!(hasher1.finish(), hasher_neg1.finish());
    }

    #[test]
    fn point_nan_infinity_hash_consistency() {
        use std::collections::HashMap;

        // Test that points with NaN can be used as HashMap keys
        let mut map: HashMap<Point<f64, 2>, i32> = HashMap::new();

        let point_nan1 = Point::new([f64::NAN, 2.0]);
        let point_nan2 = Point::new([f64::NAN, 2.0]); // Should be equal to point_nan1
        let point_inf = Point::new([f64::INFINITY, 2.0]);

        map.insert(point_nan1, 100);
        map.insert(point_inf, 200);

        // Should be able to retrieve using equivalent NaN point
        assert_eq!(map.get(&point_nan2), Some(&100));
        assert_eq!(map.len(), 2);

        // Test with f32
        let mut map_f32: HashMap<Point<f32, 1>, i32> = HashMap::new();

        let point_f32_nan1 = Point::new([f32::NAN]);
        let point_f32_nan2 = Point::new([f32::NAN]);

        map_f32.insert(point_f32_nan1, 300);
        assert_eq!(map_f32.get(&point_f32_nan2), Some(&300));
    }

    #[test]
    fn point_nan_equality_comparison() {
        // Test that NaN == NaN using our OrderedEq implementation
        // This is different from IEEE 754 standard where NaN != NaN

        // f64 NaN comparisons
        let point_nan1 = Point::new([f64::NAN, 2.0, 3.0]);
        let point_nan2 = Point::new([f64::NAN, 2.0, 3.0]);
        let point_nan3 = Point::new([f64::NAN, f64::NAN, f64::NAN]);
        let point_nan4 = Point::new([f64::NAN, f64::NAN, f64::NAN]);

        // Points with NaN should be equal when all coordinates match
        assert_eq!(point_nan1, point_nan2);
        assert_eq!(point_nan3, point_nan4);

        // Points with different NaN positions should not be equal
        let point_nan_diff1 = Point::new([f64::NAN, 2.0, 3.0]);
        let point_nan_diff2 = Point::new([1.0, f64::NAN, 3.0]);
        assert_ne!(point_nan_diff1, point_nan_diff2);

        // f32 NaN comparisons
        let point_f32_nan1 = Point::new([f32::NAN, 1.5f32]);
        let point_f32_nan2 = Point::new([f32::NAN, 1.5f32]);
        assert_eq!(point_f32_nan1, point_f32_nan2);

        // Mixed NaN and normal values
        let point_mixed1 = Point::new([1.0, f64::NAN, 3.0, 4.0]);
        let point_mixed2 = Point::new([1.0, f64::NAN, 3.0, 4.0]);
        let point_mixed3 = Point::new([1.0, f64::NAN, 3.0, 5.0]); // Different last coordinate

        assert_eq!(point_mixed1, point_mixed2);
        assert_ne!(point_mixed1, point_mixed3);
    }

    #[test]
    fn point_nan_vs_normal_comparison() {
        // Test that NaN points are not equal to points with normal values

        let point_normal = Point::new([1.0, 2.0, 3.0]);
        let point_nan = Point::new([f64::NAN, 2.0, 3.0]);
        let point_nan_all = Point::new([f64::NAN, f64::NAN, f64::NAN]);

        // NaN points should not equal normal points
        assert_ne!(point_normal, point_nan);
        assert_ne!(point_normal, point_nan_all);
        assert_ne!(point_nan, point_normal);
        assert_ne!(point_nan_all, point_normal);

        // Test with f32
        let point_f32_normal = Point::new([1.0f32, 2.0f32]);
        let point_f32_nan = Point::new([f32::NAN, 2.0f32]);

        assert_ne!(point_f32_normal, point_f32_nan);
        assert_ne!(point_f32_nan, point_f32_normal);
    }

    #[test]
    fn point_infinity_comparison() {
        // Test comparison behavior with infinity values

        // Positive infinity comparisons
        let point_pos_inf1 = Point::new([f64::INFINITY, 2.0]);
        let point_pos_inf2 = Point::new([f64::INFINITY, 2.0]);
        assert_eq!(point_pos_inf1, point_pos_inf2);

        // Negative infinity comparisons
        let point_neg_inf1 = Point::new([f64::NEG_INFINITY, 2.0]);
        let point_neg_inf2 = Point::new([f64::NEG_INFINITY, 2.0]);
        assert_eq!(point_neg_inf1, point_neg_inf2);

        // Positive vs negative infinity should not be equal
        assert_ne!(point_pos_inf1, point_neg_inf1);

        // Infinity vs normal values should not be equal
        let point_normal = Point::new([1.0, 2.0]);
        assert_ne!(point_pos_inf1, point_normal);
        assert_ne!(point_neg_inf1, point_normal);

        // Test with f32
        let point_f32_pos_inf1 = Point::new([f32::INFINITY]);
        let point_f32_pos_inf2 = Point::new([f32::INFINITY]);
        let point_f32_neg_inf = Point::new([f32::NEG_INFINITY]);

        assert_eq!(point_f32_pos_inf1, point_f32_pos_inf2);
        assert_ne!(point_f32_pos_inf1, point_f32_neg_inf);
    }

    #[test]
    fn point_nan_infinity_mixed_comparison() {
        // Test comparisons with mixed NaN and infinity values

        let point_nan_inf1 = Point::new([f64::NAN, f64::INFINITY, 1.0]);
        let point_nan_inf2 = Point::new([f64::NAN, f64::INFINITY, 1.0]);
        let point_nan_inf3 = Point::new([f64::NAN, f64::NEG_INFINITY, 1.0]);

        // Same NaN/infinity pattern should be equal
        assert_eq!(point_nan_inf1, point_nan_inf2);

        // Different infinity signs should not be equal
        assert_ne!(point_nan_inf1, point_nan_inf3);

        // Test various combinations
        let point_all_special = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
        let point_all_special_copy =
            Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
        let point_all_special_diff =
            Point::new([f64::NAN, f64::NEG_INFINITY, f64::INFINITY, f64::NAN]);

        assert_eq!(point_all_special, point_all_special_copy);
        assert_ne!(point_all_special, point_all_special_diff);
    }

    #[test]
    fn point_nan_reflexivity() {
        // Test that NaN points are equal to themselves (reflexivity)

        let point_nan = Point::new([f64::NAN, f64::NAN, f64::NAN]);
        assert_eq!(point_nan, point_nan);

        let point_mixed = Point::new([1.0, f64::NAN, 3.0, f64::INFINITY]);
        assert_eq!(point_mixed, point_mixed);

        // Test with f32
        let point_f32_nan = Point::new([f32::NAN, f32::NAN]);
        assert_eq!(point_f32_nan, point_f32_nan);
    }

    #[test]
    fn point_nan_symmetry() {
        // Test that NaN equality is symmetric (if a == b, then b == a)

        let point_a = Point::new([f64::NAN, 2.0, f64::INFINITY]);
        let point_b = Point::new([f64::NAN, 2.0, f64::INFINITY]);

        assert_eq!(point_a, point_b);
        assert_eq!(point_b, point_a); // Should be symmetric

        // Test with f32
        let point_f32_a = Point::new([f32::NAN, 1.0f32, f32::NEG_INFINITY]);
        let point_f32_b = Point::new([f32::NAN, 1.0f32, f32::NEG_INFINITY]);

        assert_eq!(point_f32_a, point_f32_b);
        assert_eq!(point_f32_b, point_f32_a);
    }

    #[test]
    fn point_nan_transitivity() {
        // Test that NaN equality is transitive (if a == b and b == c, then a == c)

        let point_a = Point::new([f64::NAN, 2.0, 3.0]);
        let point_b = Point::new([f64::NAN, 2.0, 3.0]);
        let point_c = Point::new([f64::NAN, 2.0, 3.0]);

        assert_eq!(point_a, point_b);
        assert_eq!(point_b, point_c);
        assert_eq!(point_a, point_c); // Should be transitive

        // Test with complex special values
        let point_complex_a = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
        let point_complex_b = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
        let point_complex_c = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);

        assert_eq!(point_complex_a, point_complex_b);
        assert_eq!(point_complex_b, point_complex_c);
        assert_eq!(point_complex_a, point_complex_c);
    }

    #[test]
    fn point_nan_different_bit_patterns() {
        // Test that different NaN bit patterns are considered equal
        // Note: Rust's f64::NAN is a specific bit pattern, but there are many possible NaN values

        // Create different NaN values
        let nan1 = f64::NAN;
        #[allow(clippy::zero_divided_by_zero)]
        let nan2 = 0.0f64 / 0.0f64; // Another way to create NaN
        let nan3 = f64::INFINITY - f64::INFINITY; // Yet another way

        // Verify they are all NaN
        assert!(nan1.is_nan());
        assert!(nan2.is_nan());
        assert!(nan3.is_nan());

        // Points with different NaN bit patterns should be equal
        let point1 = Point::new([nan1, 1.0]);
        let point2 = Point::new([nan2, 1.0]);
        let point3 = Point::new([nan3, 1.0]);

        assert_eq!(point1, point2);
        assert_eq!(point2, point3);
        assert_eq!(point1, point3);

        // Test with f32 as well
        let f32_nan1 = f32::NAN;
        #[allow(clippy::zero_divided_by_zero)]
        let f32_nan2 = 0.0f32 / 0.0f32;

        let point_f32_1 = Point::new([f32_nan1]);
        let point_f32_2 = Point::new([f32_nan2]);

        assert_eq!(point_f32_1, point_f32_2);
    }

    #[test]
    fn point_nan_in_different_dimensions() {
        // Test NaN behavior across different dimensionalities

        // 1D
        let point_1d_a = Point::new([f64::NAN]);
        let point_1d_b = Point::new([f64::NAN]);
        assert_eq!(point_1d_a, point_1d_b);

        // 2D
        let point_2d_a = Point::new([f64::NAN, f64::NAN]);
        let point_2d_b = Point::new([f64::NAN, f64::NAN]);
        assert_eq!(point_2d_a, point_2d_b);

        // 3D
        let point_3d_a = Point::new([f64::NAN, 1.0, f64::NAN]);
        let point_3d_b = Point::new([f64::NAN, 1.0, f64::NAN]);
        assert_eq!(point_3d_a, point_3d_b);

        // 5D
        let point_5d_a = Point::new([f64::NAN, 1.0, f64::NAN, f64::INFINITY, f64::NAN]);
        let point_5d_b = Point::new([f64::NAN, 1.0, f64::NAN, f64::INFINITY, f64::NAN]);
        assert_eq!(point_5d_a, point_5d_b);

        // 10D with mixed special values
        let point_10d_a = Point::new([
            f64::NAN,
            1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            f64::NAN,
            42.0,
            f64::NAN,
        ]);
        let point_10d_b = Point::new([
            f64::NAN,
            1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            f64::NAN,
            42.0,
            f64::NAN,
        ]);
        assert_eq!(point_10d_a, point_10d_b);
    }

    #[test]
    fn point_nan_zero_comparison() {
        // Test comparison between NaN, positive zero, and negative zero

        let point_nan = Point::new([f64::NAN, f64::NAN]);
        let point_pos_zero = Point::new([0.0, 0.0]);
        let point_neg_zero = Point::new([-0.0, -0.0]);
        let point_mixed_zero = Point::new([0.0, -0.0]);

        // NaN should not equal any zero
        assert_ne!(point_nan, point_pos_zero);
        assert_ne!(point_nan, point_neg_zero);
        assert_ne!(point_nan, point_mixed_zero);

        // Different zeros should be equal (0.0 == -0.0 in IEEE 754)
        assert_eq!(point_pos_zero, point_neg_zero);
        assert_eq!(point_pos_zero, point_mixed_zero);
        assert_eq!(point_neg_zero, point_mixed_zero);

        // Test with f32
        let point_f32_nan = Point::new([f32::NAN]);
        let point_f32_zero = Point::new([0.0f32]);
        let point_f32_neg_zero = Point::new([-0.0f32]);

        assert_ne!(point_f32_nan, point_f32_zero);
        assert_ne!(point_f32_nan, point_f32_neg_zero);
        assert_eq!(point_f32_zero, point_f32_neg_zero);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn point_extreme_dimensions() {
        // Test with high dimensional points (limited by serde trait implementations)

        // Test 20D point
        let coords_20d = [1.0; 20];
        let point_20d = Point::new(coords_20d);
        assert_eq!(point_20d.dim(), 20);
        assert_relative_eq!(point_20d.to_array().as_slice(), coords_20d.as_slice());
        assert!(point_20d.validate().is_ok());

        // Test 25D point
        let coords_25d = [2.5; 25];
        let point_25d = Point::new(coords_25d);
        assert_eq!(point_25d.dim(), 25);
        assert_relative_eq!(point_25d.to_array().as_slice(), coords_25d.as_slice());
        assert!(point_25d.validate().is_ok());

        // Test 32D point with mixed values (max supported by std traits)
        let mut coords_32d = [0.0; 32];
        for (i, coord) in coords_32d.iter_mut().enumerate() {
            *coord = i as f64;
        }
        let point_32d = Point::new(coords_32d);
        assert_eq!(point_32d.dim(), 32);
        assert_relative_eq!(point_32d.to_array().as_slice(), coords_32d.as_slice());
        assert!(point_32d.validate().is_ok());

        // Test high dimensional point with NaN
        let mut coords_with_nan = [1.0; 25];
        coords_with_nan[12] = f64::NAN;
        let point_with_nan = Point::new(coords_with_nan);
        assert!(point_with_nan.validate().is_err());

        // Test equality for high dimensional points
        let point_20d_copy = Point::new([1.0; 20]);
        assert_eq!(point_20d, point_20d_copy);

        // Test with 30D points
        let coords_30d_a = [std::f64::consts::PI; 30];
        let coords_30d_b = [std::f64::consts::PI; 30];
        let point_30d_a = Point::new(coords_30d_a);
        let point_30d_b = Point::new(coords_30d_b);
        assert_eq!(point_30d_a, point_30d_b);
        assert!(point_30d_a.validate().is_ok());
    }

    #[test]
    fn point_boundary_numeric_values() {
        // Test with extreme numeric values

        // Test with very large f64 values
        let large_point = Point::new([f64::MAX, f64::MAX / 2.0, 1e308]);
        assert!(large_point.validate().is_ok());
        assert_relative_eq!(large_point.to_array()[0], f64::MAX);

        // Test with very small f64 values
        let small_point = Point::new([f64::MIN, f64::MIN_POSITIVE, 1e-308]);
        assert!(small_point.validate().is_ok());

        // Test with subnormal numbers
        let subnormal = f64::MIN_POSITIVE / 2.0;
        let subnormal_point = Point::new([subnormal, -subnormal, 0.0]);
        assert!(subnormal_point.validate().is_ok());

        // Test f32 extremes
        let extreme_f32_point = Point::new([f32::MAX, f32::MIN, f32::MIN_POSITIVE]);
        assert!(extreme_f32_point.validate().is_ok());
    }

    #[test]
    fn point_clone_and_copy_semantics() {
        // Test that Point correctly implements Clone and Copy

        let original = Point::new([1.0, 2.0, 3.0]);

        // Test explicit cloning
        #[allow(clippy::clone_on_copy)]
        let cloned = original.clone();
        assert_relative_eq!(original.to_array().as_slice(), cloned.to_array().as_slice());

        // Test copy semantics (should work implicitly)
        let copied = original; // This should copy, not move
        assert_eq!(original, copied);

        // Original should still be accessible after copy
        assert_eq!(original.dim(), 3);
        assert_eq!(copied.dim(), 3);

        // Test with f32
        let f32_point = Point::new([1.5f32, 2.5f32, 3.5f32, 4.5f32]);
        let f32_copied = f32_point;
        assert_eq!(f32_point, f32_copied);
    }

    #[test]
    fn point_partial_ord_comprehensive() {
        use std::cmp::Ordering;

        // Test lexicographic ordering in detail
        let point_a = Point::new([1.0, 2.0, 3.0]);
        let point_b = Point::new([1.0, 2.0, 4.0]); // Greater in last coordinate
        let point_c = Point::new([1.0, 3.0, 0.0]); // Greater in second coordinate
        let point_d = Point::new([2.0, 0.0, 0.0]); // Greater in first coordinate

        // Test all comparison operators
        assert!(point_a < point_b);
        assert!(point_b > point_a);
        assert!(point_a <= point_b);
        assert!(point_b >= point_a);

        assert!(point_a < point_c);
        assert!(point_a < point_d);
        assert!(point_c < point_d);

        // Test partial_cmp directly
        assert_eq!(point_a.partial_cmp(&point_b), Some(Ordering::Less));
        assert_eq!(point_b.partial_cmp(&point_a), Some(Ordering::Greater));
        assert_eq!(point_a.partial_cmp(&point_a), Some(Ordering::Equal));

        // Test with negative numbers
        let neg_point_a = Point::new([-1.0, -2.0]);
        let neg_point_b = Point::new([-1.0, -1.0]);
        assert!(neg_point_a < neg_point_b); // -2.0 < -1.0

        // Test with mixed positive/negative
        let mixed_a = Point::new([-1.0, 2.0]);
        let mixed_b = Point::new([1.0, -2.0]);
        assert!(mixed_a < mixed_b); // -1.0 < 1.0

        // Test with zeros
        let zero_a = Point::new([0.0, 0.0]);
        let zero_b = Point::new([0.0, 0.0]);
        assert_eq!(zero_a.partial_cmp(&zero_b), Some(Ordering::Equal));

        // Test with special float values (where defined)
        let inf_point = Point::new([f64::INFINITY]);
        let normal_point = Point::new([1.0]);
        // Note: PartialOrd with NaN/Infinity may have special behavior
        assert!(normal_point < inf_point);
    }

    #[test]
    fn point_memory_layout_and_size() {
        use std::mem;

        // Test that Point has the expected memory layout
        // Point should be the same size as its coordinate array

        assert_eq!(mem::size_of::<Point<f64, 3>>(), mem::size_of::<[f64; 3]>());
        assert_eq!(mem::size_of::<Point<f32, 4>>(), mem::size_of::<[f32; 4]>());

        // Test alignment
        assert_eq!(
            mem::align_of::<Point<f64, 3>>(),
            mem::align_of::<[f64; 3]>()
        );

        // Test with different dimensions
        assert_eq!(mem::size_of::<Point<f64, 1>>(), 8); // 1 * 8 bytes
        assert_eq!(mem::size_of::<Point<f64, 2>>(), 16); // 2 * 8 bytes
        assert_eq!(mem::size_of::<Point<f64, 10>>(), 80); // 10 * 8 bytes

        assert_eq!(mem::size_of::<Point<f32, 1>>(), 4); // 1 * 4 bytes
        assert_eq!(mem::size_of::<Point<f32, 2>>(), 8); // 2 * 4 bytes
    }

    #[test]
    fn point_zero_dimensional() {
        // Test 0-dimensional points (edge case)
        let point_0d: Point<f64, 0> = Point::new([]);
        assert_eq!(point_0d.dim(), 0);
        assert_relative_eq!(point_0d.to_array().as_slice(), ([] as [f64; 0]).as_slice());
        assert!(point_0d.validate().is_ok());

        // Test equality for 0D points
        let point_0d_2: Point<f64, 0> = Point::new([]);
        assert_eq!(point_0d, point_0d_2);

        // Test hashing for 0D points
        let hash_0d = get_hash(&point_0d);
        let hash_0d_2 = get_hash(&point_0d_2);
        assert_eq!(hash_0d, hash_0d_2);

        // Test origin for 0D
        let origin_0d: Point<f64, 0> = Point::origin();
        assert_eq!(origin_0d, point_0d);
    }

    #[test]
    fn point_serialization_edge_cases() {
        // Test serialization with special floating point values

        // Test with NaN
        let point_with_nan = Point::new([f64::NAN, 1.0, 2.0]);
        let serialized_nan = serde_json::to_string(&point_with_nan).unwrap();
        // NaN serializes as null in JSON
        assert!(serialized_nan.contains("null"));

        // Test with infinity
        let point_with_inf = Point::new([f64::INFINITY, 1.0]);
        let serialized_inf = serde_json::to_string(&point_with_inf).unwrap();
        // Infinity serializes as null in JSON
        assert!(serialized_inf.contains("null"));

        // Test with negative infinity
        let point_with_neg_inf = Point::new([f64::NEG_INFINITY, 1.0]);
        let serialized_neg_inf = serde_json::to_string(&point_with_neg_inf).unwrap();
        assert!(serialized_neg_inf.contains("null"));

        // Test with very large numbers
        let point_large = Point::new([1e100, -1e100, 0.0]);
        let serialized_large = serde_json::to_string(&point_large).unwrap();
        let deserialized_large: Point<f64, 3> = serde_json::from_str(&serialized_large).unwrap();
        assert_eq!(point_large, deserialized_large);

        // Test with very small numbers
        let point_small = Point::new([1e-100, -1e-100, 0.0]);
        let serialized_small = serde_json::to_string(&point_small).unwrap();
        let deserialized_small: Point<f64, 3> = serde_json::from_str(&serialized_small).unwrap();
        assert_eq!(point_small, deserialized_small);
    }

    #[test]
    fn point_conversion_edge_cases() {
        // Test edge cases in type conversions

        // Test conversion with potential precision loss (should still work)
        let precise_coords = [1.000_000_000_000_001_f64, 2.000_000_000_000_002_f64];
        let point_precise: Point<f64, 2> = Point::new(precise_coords);
        assert_relative_eq!(
            point_precise.to_array().as_slice(),
            precise_coords.as_slice()
        );

        // Test conversion from array reference
        let coords_ref = &[1.0f32, 2.0f32, 3.0f32];
        let point_from_ref: Point<f64, 3> = Point::new(coords_ref.map(std::convert::Into::into));
        assert_relative_eq!(
            point_from_ref.to_array().as_slice(),
            [1.0f64, 2.0f64, 3.0f64].as_slice()
        );

        // Test conversion to array with different methods
        let point = Point::new([1.0, 2.0, 3.0]);

        // Using Into trait
        let coords_into: [f64; 3] = point.into();
        assert_relative_eq!(coords_into.as_slice(), [1.0, 2.0, 3.0].as_slice());

        // Using From trait with reference
        let point_ref = Point::new([4.0, 5.0]);
        let coords_from_ref: [f64; 2] = (&point_ref).into();
        assert_relative_eq!(coords_from_ref.as_slice(), [4.0, 5.0].as_slice());

        // Verify original point is still usable after reference conversion
        assert_relative_eq!(point_ref.to_array().as_slice(), [4.0, 5.0].as_slice());
    }

    #[test]
    fn point_cast_conversions() {
        // Test the cast()-based TryFrom<[T; D]> implementation

        // Test f32 to f64 conversion (safe upcast) using TryFrom
        let coords_f32: [f32; 3] = [1.5, 2.5, 3.5];
        let point_f64: Point<f64, 3> = Point::try_from(coords_f32).unwrap();

        // Verify the conversion worked correctly
        assert_relative_eq!(
            point_f64.to_array().as_slice(),
            [1.5f64, 2.5f64, 3.5f64].as_slice(),
            epsilon = 1e-9
        );

        // Test same type conversion (no actual cast needed)
        let coords_f64: [f64; 2] = [10.0, 20.0];
        let point_f64_same: Point<f64, 2> = Point::try_from(coords_f64).unwrap();
        assert_relative_eq!(
            point_f64_same.to_array().as_slice(),
            [10.0, 20.0].as_slice()
        );

        // Test with integer type conversions
        let coords_i32: [i32; 4] = [1, 2, 3, 4];
        let point_f64_from_int: Point<f64, 4> = Point::try_from(coords_i32).unwrap();
        assert_relative_eq!(
            point_f64_from_int.to_array().as_slice(),
            [1.0, 2.0, 3.0, 4.0].as_slice(),
            epsilon = 1e-9
        );

        // Test with large values that are within range
        let coords_large_i32: [i32; 2] = [i32::MAX, i32::MIN];
        let point_f64_from_large: Point<f64, 2> = Point::try_from(coords_large_i32).unwrap();
        assert_relative_eq!(
            point_f64_from_large.to_array().as_slice(),
            [f64::from(i32::MAX), f64::from(i32::MIN)].as_slice(),
            epsilon = 1e-9
        );

        // Test with mixed typical values
        let coords_mixed: [f32; 3] = [0.0, 1.5, -3.5];
        let point_mixed: Point<f64, 3> = Point::try_from(coords_mixed).unwrap();
        assert_relative_eq!(
            point_mixed.to_array().as_slice(),
            [0.0, 1.5, -3.5].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn point_hash_special_values() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Test for NaN
        let point_nan1 = Point::new([f64::NAN, 2.0]);
        let point_nan2 = Point::new([f64::NAN, 2.0]);

        let mut hasher_nan1 = DefaultHasher::new();
        let mut hasher_nan2 = DefaultHasher::new();

        point_nan1.hash(&mut hasher_nan1);
        point_nan2.hash(&mut hasher_nan2);

        assert_eq!(hasher_nan1.finish(), hasher_nan2.finish());

        // Test for positive infinity
        let point_inf1 = Point::new([f64::INFINITY, 2.0]);
        let point_inf2 = Point::new([f64::INFINITY, 2.0]);

        let mut hasher_inf1 = DefaultHasher::new();
        let mut hasher_inf2 = DefaultHasher::new();

        point_inf1.hash(&mut hasher_inf1);
        point_inf2.hash(&mut hasher_inf2);

        assert_eq!(hasher_inf1.finish(), hasher_inf2.finish());

        // Test for negative infinity
        let point_neg_inf1 = Point::new([f64::NEG_INFINITY, 2.0]);
        let point_neg_inf2 = Point::new([f64::NEG_INFINITY, 2.0]);

        let mut hasher_neg_inf1 = DefaultHasher::new();
        let mut hasher_neg_inf2 = DefaultHasher::new();

        point_neg_inf1.hash(&mut hasher_neg_inf1);
        point_neg_inf2.hash(&mut hasher_neg_inf2);

        assert_eq!(hasher_neg_inf1.finish(), hasher_neg_inf2.finish());

        // Test for +0.0 and -0.0
        let point_pos_zero = Point::new([0.0, 2.0]);
        let point_neg_zero = Point::new([-0.0, 2.0]);

        let mut hasher_pos_zero = DefaultHasher::new();
        let mut hasher_neg_zero = DefaultHasher::new();

        point_pos_zero.hash(&mut hasher_pos_zero);
        point_neg_zero.hash(&mut hasher_neg_zero);

        assert_eq!(hasher_pos_zero.finish(), hasher_neg_zero.finish());
    }

    #[test]
    fn point_hashmap_special_values() {
        use std::collections::HashMap;

        let mut map: HashMap<Point<f64, 2>, &str> = HashMap::new();

        let point_nan = Point::new([f64::NAN, 2.0]);
        let point_inf = Point::new([f64::INFINITY, 2.0]);
        let point_neg_inf = Point::new([f64::NEG_INFINITY, 2.0]);
        let point_zero = Point::new([0.0, 2.0]);

        map.insert(point_nan, "NaN Point");
        map.insert(point_inf, "Infinity Point");
        map.insert(point_neg_inf, "Negative Infinity Point");
        map.insert(point_zero, "Zero Point");

        assert_eq!(map[&Point::new([f64::NAN, 2.0])], "NaN Point");
        assert_eq!(map[&Point::new([f64::INFINITY, 2.0])], "Infinity Point");
        assert_eq!(
            map[&Point::new([f64::NEG_INFINITY, 2.0])],
            "Negative Infinity Point"
        );
        assert_eq!(map[&Point::new([-0.0, 2.0])], "Zero Point");
    }

    #[test]
    fn point_hashset_special_values() {
        use std::collections::HashSet;

        let mut set: HashSet<Point<f64, 2>> = HashSet::new();

        set.insert(Point::new([f64::NAN, 2.0]));
        set.insert(Point::new([f64::INFINITY, 2.0]));
        set.insert(Point::new([f64::NEG_INFINITY, 2.0]));
        set.insert(Point::new([0.0, 2.0]));
        set.insert(Point::new([-0.0, 2.0]));

        assert_eq!(set.len(), 4); // 0.0 and -0.0 should be considered equal here

        assert!(set.contains(&Point::new([f64::NAN, 2.0])));
        assert!(set.contains(&Point::new([f64::INFINITY, 2.0])));
        assert!(set.contains(&Point::new([f64::NEG_INFINITY, 2.0])));
        assert!(set.contains(&Point::new([-0.0, 2.0])));
    }

    #[test]
    fn point_hash_distribution_basic() {
        use std::collections::HashSet;

        // Test that different points generally produce different hashes
        // (This is a probabilistic test, not a guarantee)

        let mut hashes = HashSet::new();

        // Generate a variety of points and collect their hashes
        for i in 0..100 {
            let point = Point::new([f64::from(i), f64::from(i * 2)]);
            let hash = get_hash(&point);
            hashes.insert(hash);
        }

        // We should have close to 100 unique hashes (allowing for some collisions)
        assert!(
            hashes.len() > 90,
            "Hash distribution seems poor: {} unique hashes out of 100",
            hashes.len()
        );

        // Test with negative values
        for i in -50..50 {
            let point = Point::new([f64::from(i), f64::from(i * 3), f64::from(i * 5)]);
            let hash = get_hash(&point);
            hashes.insert(hash);
        }

        // Should have even more unique hashes now
        assert!(
            hashes.len() > 140,
            "Hash distribution with negatives: {} unique hashes",
            hashes.len()
        );
    }

    #[test]
    fn point_validation_error_details() {
        // Test CoordinateValidationError with specific error details

        // Test invalid coordinate at specific index
        let invalid_point = Point::new([1.0, f64::NAN, 3.0]);
        let result = invalid_point.validate();
        assert!(result.is_err());

        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 1);
            assert_eq!(dimension, 3);
            assert!(coordinate_value.contains("NaN"));
        } else {
            panic!("Expected InvalidCoordinate error");
        }

        // Test with infinity at different positions
        let inf_point = Point::new([f64::INFINITY, 2.0, 3.0, 4.0]);
        let result = inf_point.validate();
        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 0);
            assert_eq!(dimension, 4);
            assert!(coordinate_value.contains("inf"));
        } else {
            panic!("Expected InvalidCoordinate error");
        }

        // Test with negative infinity at last position
        let neg_inf_point = Point::new([1.0, 2.0, f64::NEG_INFINITY]);
        let result = neg_inf_point.validate();
        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 2);
            assert_eq!(dimension, 3);
            assert!(coordinate_value.contains("inf"));
        }

        // Test f32 validation errors
        let invalid_f32_point = Point::new([1.0f32, f32::NAN, 3.0f32]);
        let result = invalid_f32_point.validate();
        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 1);
            assert_eq!(dimension, 3);
            assert!(coordinate_value.contains("NaN"));
        }
    }

    #[test]
    fn point_validation_error_display() {
        // Test error message formatting
        let invalid_point = Point::new([1.0, f64::NAN, 3.0]);
        let result = invalid_point.validate();

        if let Err(error) = result {
            let error_msg = format!("{error}");
            assert!(error_msg.contains("Invalid coordinate at index 1"));
            assert!(error_msg.contains("in dimension 3"));
            assert!(error_msg.contains("NaN"));
        } else {
            panic!("Expected validation error");
        }

        // Test with infinity
        let inf_point = Point::new([f64::INFINITY]);
        let result = inf_point.validate();

        if let Err(error) = result {
            let error_msg = format!("{error}");
            assert!(error_msg.contains("Invalid coordinate at index 0"));
            assert!(error_msg.contains("in dimension 1"));
            assert!(error_msg.contains("inf"));
        }
    }

    #[test]
    fn point_validation_error_clone_and_eq() {
        // Test that CoordinateValidationError can be cloned and compared
        let invalid_point = Point::new([f64::NAN, 2.0]);
        let result1 = invalid_point.validate();
        let result2 = invalid_point.validate();

        assert!(result1.is_err());
        assert!(result2.is_err());

        let error1 = result1.unwrap_err();
        let error2 = result2.unwrap_err();

        // Test Clone
        let error1_clone = error1.clone();
        assert_eq!(error1, error1_clone);

        // Test PartialEq
        assert_eq!(error1, error2);

        // Test Debug
        let debug_output = format!("{error1:?}");
        assert!(debug_output.contains("InvalidCoordinate"));
        assert!(debug_output.contains("coordinate_index"));
        assert!(debug_output.contains("dimension"));
    }

    #[test]
    fn point_validation_all_coordinate_types() {
        // Test validation with different coordinate types

        // Floating point types can be invalid
        assert!(Point::new([1.0f32, 2.0f32]).validate().is_ok());
        assert!(Point::new([1.0f64, 2.0f64]).validate().is_ok());
        assert!(Point::new([f32::NAN, 2.0f32]).validate().is_err());
        assert!(Point::new([f64::NAN, 2.0f64]).validate().is_err());
    }

    #[test]
    fn point_validation_first_invalid_coordinate() {
        // Test that validation returns the FIRST invalid coordinate found
        let multi_invalid = Point::new([1.0, f64::NAN, f64::INFINITY, f64::NAN]);
        let result = multi_invalid.validate();

        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index, ..
        }) = result
        {
            // Should return the first invalid coordinate (index 1, not 2 or 3)
            assert_eq!(coordinate_index, 1);
        } else {
            panic!("Expected InvalidCoordinate error");
        }

        // Test with invalid at index 0
        let first_invalid = Point::new([f64::INFINITY, f64::NAN, 3.0]);
        let result = first_invalid.validate();

        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index, ..
        }) = result
        {
            assert_eq!(coordinate_index, 0);
        }
    }

    #[test]
    fn point_hashmap_with_special_values() {
        use std::collections::HashMap;

        let mut point_map: HashMap<Point<f64, 3>, &str> = HashMap::new();

        // Insert points with various special values
        let point_normal = Point::new([1.0, 2.0, 3.0]);
        let point_nan = Point::new([f64::NAN, 2.0, 3.0]);
        let point_inf = Point::new([f64::INFINITY, 2.0, 3.0]);
        let point_neg_inf = Point::new([f64::NEG_INFINITY, 2.0, 3.0]);

        point_map.insert(point_normal, "normal point");
        point_map.insert(point_nan, "point with NaN");
        point_map.insert(point_inf, "point with +∞");
        point_map.insert(point_neg_inf, "point with -∞");

        assert_eq!(point_map.len(), 4);

        // Test retrieval with equivalent points
        let point_normal_copy = Point::new([1.0, 2.0, 3.0]);
        let point_nan_copy = Point::new([f64::NAN, 2.0, 3.0]);
        let point_inf_copy = Point::new([f64::INFINITY, 2.0, 3.0]);
        let point_neg_inf_copy = Point::new([f64::NEG_INFINITY, 2.0, 3.0]);

        assert!(point_map.contains_key(&point_normal_copy));
        assert!(point_map.contains_key(&point_nan_copy));
        assert!(point_map.contains_key(&point_inf_copy));
        assert!(point_map.contains_key(&point_neg_inf_copy));

        // Test retrieval of values
        assert_eq!(point_map.get(&point_normal_copy), Some(&"normal point"));
        assert_eq!(point_map.get(&point_nan_copy), Some(&"point with NaN"));
        assert_eq!(point_map.get(&point_inf_copy), Some(&"point with +∞"));
        assert_eq!(point_map.get(&point_neg_inf_copy), Some(&"point with -∞"));

        // Demonstrate that NaN points can be used as keys reliably
        let mut nan_counter = HashMap::new();
        for _ in 0..5 {
            let nan_point = Point::new([f64::NAN, 1.0]);
            *nan_counter.entry(nan_point).or_insert(0) += 1;
        }
        assert_eq!(*nan_counter.values().next().unwrap(), 5);
    }

    #[test]
    fn point_hashset_with_special_values() {
        use std::collections::HashSet;

        let mut point_set: HashSet<Point<f64, 2>> = HashSet::new();

        // Add various points including duplicates with special values
        let points = vec![
            Point::new([1.0, 2.0]),
            Point::new([1.0, 2.0]), // Duplicate normal point
            Point::new([f64::NAN, 2.0]),
            Point::new([f64::NAN, 2.0]), // Duplicate NaN point
            Point::new([f64::INFINITY, 2.0]),
            Point::new([f64::INFINITY, 2.0]), // Duplicate infinity point
            Point::new([0.0, -0.0]),          // Zero and negative zero (equal)
            Point::new([-0.0, 0.0]),          // Different zero combination
        ];

        for point in points {
            point_set.insert(point);
        }

        // Should have 4 unique points: normal, NaN, ∞, and two different zero combinations
        // Note: [0.0, -0.0] and [-0.0, 0.0] are different points because only corresponding
        // coordinates are compared for equality (0.0 == -0.0 but the positions differ)
        assert_eq!(point_set.len(), 4);

        // Test membership
        let test_nan = Point::new([f64::NAN, 2.0]);
        let test_inf = Point::new([f64::INFINITY, 2.0]);
        let test_normal = Point::new([1.0, 2.0]);

        assert!(point_set.contains(&test_nan));
        assert!(point_set.contains(&test_inf));
        assert!(point_set.contains(&test_normal));
    }

    #[test]
    fn point_mathematical_properties_comprehensive() {
        // Test mathematical properties with various special values
        let point_a = Point::new([f64::NAN, 2.0, f64::INFINITY]);
        let point_b = Point::new([f64::NAN, 2.0, f64::INFINITY]);
        let point_c = Point::new([f64::NAN, 2.0, f64::INFINITY]);

        // Reflexivity: a == a
        assert_eq!(point_a, point_a);

        // Symmetry: if a == b, then b == a
        let symmetry_ab = point_a == point_b;
        let symmetry_ba = point_b == point_a;
        assert_eq!(symmetry_ab, symmetry_ba);
        assert!(symmetry_ab && symmetry_ba);

        // Transitivity: if a == b and b == c, then a == c
        let trans_ab = point_a == point_b;
        let trans_bc = point_b == point_c;
        let trans_ac = point_a == point_c;
        assert!(trans_ab && trans_bc && trans_ac);

        // Test with mixed special values
        let point_mixed1 = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
        let point_mixed2 = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
        let point_mixed3 = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);

        // All should be equal
        assert_eq!(point_mixed1, point_mixed2);
        assert_eq!(point_mixed2, point_mixed3);
        assert_eq!(point_mixed1, point_mixed3);

        // Test reflexivity with mixed values
        assert_eq!(point_mixed1, point_mixed1);
    }

    #[test]
    fn point_numeric_types_f32() {
        // Test f32 points
        let point_f32_1 = Point::new([1.5f32, 2.5f32]);
        let point_f32_2 = Point::new([1.5f32, 2.5f32]);
        let point_f32_nan = Point::new([f32::NAN, 2.5f32]);
        let point_f32_nan2 = Point::new([f32::NAN, 2.5f32]);

        assert_eq!(point_f32_1, point_f32_2);
        assert_eq!(point_f32_nan, point_f32_nan2);

        // Test f32 infinity
        let point_f32_inf1 = Point::new([f32::INFINITY, 1.0f32]);
        let point_f32_inf2 = Point::new([f32::INFINITY, 1.0f32]);
        let point_f32_neg_inf = Point::new([f32::NEG_INFINITY, 1.0f32]);

        assert_eq!(point_f32_inf1, point_f32_inf2);
        assert_ne!(point_f32_inf1, point_f32_neg_inf);

        // Test f32 in HashMap
        let mut f32_map: HashMap<Point<f32, 2>, &str> = HashMap::new();
        f32_map.insert(point_f32_1, "f32 point");
        f32_map.insert(point_f32_nan, "f32 NaN point");

        let lookup_f32 = Point::new([1.5f32, 2.5f32]);
        let lookup_f32_nan = Point::new([f32::NAN, 2.5f32]);

        assert!(f32_map.contains_key(&lookup_f32));
        assert!(f32_map.contains_key(&lookup_f32_nan));
        assert_eq!(f32_map.get(&lookup_f32), Some(&"f32 point"));
        assert_eq!(f32_map.get(&lookup_f32_nan), Some(&"f32 NaN point"));
    }

    #[test]
    fn point_integer_like_values() {
        // Test integer-like values using f64
        let point_int_1 = Point::new([10.0, 20.0, 30.0]);
        let point_int_2 = Point::new([10.0, 20.0, 30.0]);
        let point_int_3 = Point::new([10.0, 20.0, 31.0]);

        assert_eq!(point_int_1, point_int_2);
        assert_ne!(point_int_1, point_int_3);

        // Test in HashMap
        let mut int_map: HashMap<Point<f64, 2>, String> = HashMap::new();
        int_map.insert(Point::new([1.0, 2.0]), "integer-like point".to_string());

        let lookup_key = Point::new([1.0, 2.0]);
        assert!(int_map.contains_key(&lookup_key));
        assert_eq!(int_map.get(&lookup_key).unwrap(), "integer-like point");
    }

    #[test]
    fn point_floating_point_precision() {
        // Test that we can distinguish between very close floating point values
        let point_epsilon1 = Point::new([1.0 + f64::EPSILON, 2.0]);
        let point_epsilon2 = Point::new([1.0, 2.0]);
        assert_ne!(point_epsilon1, point_epsilon2);

        // Test with values that should be exactly equal
        let point_exact1 = Point::new([0.1 + 0.2, 1.0]);
        let point_exact2 = Point::new([0.3, 1.0]);
        // Note: Due to floating point representation, 0.1 + 0.2 != 0.3
        // This test demonstrates the exact equality behavior
        assert_ne!(point_exact1, point_exact2);

        // Test that points with slightly different values are not approximately equal
        // (demonstrating that we use exact equality, not approximate)
        let point_a = Point::new([1.0, 2.0]);
        let point_b = Point::new([1.0 + f64::EPSILON, 2.0]);
        assert_ne!(point_a, point_b);

        // But points with exactly the same values are equal
        let point_same1 = Point::new([1.0, 2.0]);
        let point_same2 = Point::new([1.0, 2.0]);
        assert_eq!(point_same1, point_same2);
    }

    #[test]
    fn point_zero_and_negative_zero() {
        // Test zero and negative zero behavior
        let point_pos_zero = Point::new([0.0, 0.0]);
        let point_neg_zero = Point::new([-0.0, -0.0]);
        let point_mixed_zero = Point::new([0.0, -0.0]);
        let point_mixed_zero2 = Point::new([-0.0, 0.0]);

        // All should be equal (0.0 == -0.0 in IEEE 754)
        assert_eq!(point_pos_zero, point_neg_zero);
        assert_eq!(point_pos_zero, point_mixed_zero);
        assert_eq!(point_pos_zero, point_mixed_zero2);
        assert_eq!(point_neg_zero, point_mixed_zero);
        assert_eq!(point_neg_zero, point_mixed_zero2);
        assert_eq!(point_mixed_zero, point_mixed_zero2);

        // Test hashing consistency
        let hash_pos = get_hash(&point_pos_zero);
        let hash_neg = get_hash(&point_neg_zero);
        let hash_mixed1 = get_hash(&point_mixed_zero);
        let hash_mixed2 = get_hash(&point_mixed_zero2);

        assert_eq!(hash_pos, hash_neg);
        assert_eq!(hash_pos, hash_mixed1);
        assert_eq!(hash_pos, hash_mixed2);
    }

    #[test]
    fn point_nan_different_creation_methods() {
        // Test that different ways of creating NaN are treated as equal
        let nan1 = f64::NAN;
        let nan2 = f64::NAN;
        let nan3 = f64::NAN;

        let point_nan_variant1 = Point::new([nan1, 1.0]);
        let point_nan_variant2 = Point::new([nan2, 1.0]);
        let point_nan_variant3 = Point::new([nan3, 1.0]);

        assert_eq!(point_nan_variant1, point_nan_variant2);
        assert_eq!(point_nan_variant2, point_nan_variant3);
        assert_eq!(point_nan_variant1, point_nan_variant3);

        // Test hash consistency
        let hash1 = get_hash(&point_nan_variant1);
        let hash2 = get_hash(&point_nan_variant2);
        let hash3 = get_hash(&point_nan_variant3);

        assert_eq!(hash1, hash2);
        assert_eq!(hash2, hash3);
    }

    #[test]
    fn point_mixed_special_values_comprehensive() {
        // Test various combinations of special values
        let point_all_special = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0]);
        let point_all_special_copy =
            Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0]);

        assert_eq!(point_all_special, point_all_special_copy);

        // Test different combinations
        let point_combo1 = Point::new([f64::NAN, 1.0, f64::INFINITY]);
        let point_combo2 = Point::new([f64::NAN, 1.0, f64::INFINITY]);
        let point_combo3 = Point::new([f64::NAN, 1.0, f64::NEG_INFINITY]); // Different

        assert_eq!(point_combo1, point_combo2);
        assert_ne!(point_combo1, point_combo3);

        // Test in collections
        let mut special_set: HashSet<Point<f64, 3>> = HashSet::new();
        special_set.insert(point_combo1);
        special_set.insert(point_combo2); // Should not increase size
        special_set.insert(point_combo3); // Should increase size

        assert_eq!(special_set.len(), 2);
    }

    // =============================================================================
    // CONVERSION ERROR TESTS
    // =============================================================================

    #[test]
    fn point_try_from_conversion_errors() {
        // Test non-finite value errors (NaN after cast)
        let coords_with_nan = [f64::NAN, 1.0, 2.0];
        let result: Result<Point<f32, 3>, _> = Point::try_from(coords_with_nan);
        assert!(result.is_err());
        match result.unwrap_err() {
            CoordinateConversionError::NonFiniteValue {
                coordinate_index, ..
            } => {
                assert_eq!(coordinate_index, 0);
            }
            CoordinateConversionError::ConversionFailed { .. } => {
                panic!("Expected NonFiniteValue error")
            }
        }

        // Test non-finite value errors (infinity after cast)
        let coords_with_inf = [1.0, f64::INFINITY, 2.0];
        let result: Result<Point<f32, 3>, _> = Point::try_from(coords_with_inf);
        assert!(result.is_err());
        match result.unwrap_err() {
            CoordinateConversionError::NonFiniteValue {
                coordinate_index, ..
            } => {
                assert_eq!(coordinate_index, 1);
            }
            CoordinateConversionError::ConversionFailed { .. } => {
                panic!("Expected NonFiniteValue error")
            }
        }

        // Test conversion failure (overflow cases if we had them)
        // Note: With num_traits::cast, most reasonable numeric conversions succeed,
        // so ConversionFailed errors are rare in practice for standard numeric types.
        // But the infrastructure is there for edge cases or custom numeric types.
    }

    #[test]
    fn point_try_from_success_cases() {
        // Test successful conversions that should work fine

        // f32 to f64 (upcast)
        let coords_f32 = [1.5f32, 2.5f32, 3.5f32];
        let result: Result<Point<f64, 3>, _> = Point::try_from(coords_f32);
        assert!(result.is_ok());
        let point = result.unwrap();
        assert_relative_eq!(
            point.to_array().as_slice(),
            [1.5f64, 2.5f64, 3.5f64].as_slice(),
            epsilon = 1e-9
        );

        // i32 to f64
        let coords_i32 = [1i32, -2i32, 3i32];
        let result: Result<Point<f64, 3>, _> = Point::try_from(coords_i32);
        assert!(result.is_ok());
        let point = result.unwrap();
        assert_relative_eq!(
            point.to_array().as_slice(),
            [1.0f64, -2.0f64, 3.0f64].as_slice(),
            epsilon = 1e-9
        );

        // Same type (f64 to f64)
        let coords_f64 = [1.0f64, 2.0f64];
        let result: Result<Point<f64, 2>, _> = Point::try_from(coords_f64);
        assert!(result.is_ok());
        let point = result.unwrap();
        assert_relative_eq!(
            point.to_array().as_slice(),
            [1.0f64, 2.0f64].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn point_try_from_error_details() {
        // Test error message formatting for NonFiniteValue
        let coords_with_nan = [f64::NAN, 1.0];
        let result: Result<Point<f32, 2>, _> = Point::try_from(coords_with_nan);
        assert!(result.is_err());

        let error = result.unwrap_err();
        let error_msg = format!("{error}");
        assert!(error_msg.contains("Non-finite value"));
        assert!(error_msg.contains("coordinate index 0"));
        assert!(error_msg.contains("NaN"));

        // Test error cloning and equality
        let coords_with_inf = [f64::INFINITY, 2.0];
        let result2: Result<Point<f32, 2>, _> = Point::try_from(coords_with_inf);
        let error2 = result2.unwrap_err();
        let error2_clone = error2.clone();
        assert_eq!(error2, error2_clone);
    }

    #[test]
    fn point_try_from_different_error_positions() {
        // Test error at different coordinate positions
        let test_cases = [
            ([f64::NAN, 1.0, 2.0, 3.0], 0),          // First coordinate
            ([1.0, f64::NAN, 2.0, 3.0], 1),          // Second coordinate
            ([1.0, 2.0, f64::INFINITY, 3.0], 2),     // Third coordinate
            ([1.0, 2.0, 3.0, f64::NEG_INFINITY], 3), // Fourth coordinate
        ];

        for &(coords, expected_index) in &test_cases {
            let result: Result<Point<f32, 4>, _> = Point::try_from(coords);
            assert!(result.is_err());
            match result.unwrap_err() {
                CoordinateConversionError::NonFiniteValue {
                    coordinate_index, ..
                } => {
                    assert_eq!(coordinate_index, expected_index);
                }
                CoordinateConversionError::ConversionFailed { .. } => {
                    panic!("Expected NonFiniteValue error at position {expected_index}")
                }
            }
        }
    }

    #[test]
    fn point_try_from_first_error_reported() {
        // When multiple coordinates have errors, the first one should be reported
        let coords_multi_error = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        let result: Result<Point<f32, 3>, _> = Point::try_from(coords_multi_error);
        assert!(result.is_err());

        match result.unwrap_err() {
            CoordinateConversionError::NonFiniteValue {
                coordinate_index, ..
            } => {
                // Should report the first error (index 0, not 1 or 2)
                assert_eq!(coordinate_index, 0);
            }
            CoordinateConversionError::ConversionFailed { .. } => {
                panic!("Expected NonFiniteValue error")
            }
        }
    }

    #[test]
    fn point_trait_completeness() {
        // Helper functions for compile-time trait checks
        fn assert_send<T: Send>(_: T) {}
        fn assert_sync<T: Sync>(_: T) {}

        // Test that Point implements all expected traits

        let point = Point::new([1.0, 2.0, 3.0]);

        // Test Debug trait
        let debug_output = format!("{point:?}");
        assert!(!debug_output.is_empty());
        assert!(debug_output.contains("Point"));

        // Test Default trait
        let default_point: Point<f64, 3> = Point::default();
        assert_relative_eq!(
            default_point.to_array().as_slice(),
            [0.0, 0.0, 0.0].as_slice()
        );

        // Test PartialOrd trait (ordering)
        let point_smaller = Point::new([1.0, 2.0, 2.9]);
        assert!(point_smaller < point);

        // Test that Send and Sync are implemented (compile-time check)
        assert_send(point);
        assert_sync(point);

        // Test Clone and Copy
        #[allow(clippy::clone_on_copy)]
        let cloned = point.clone();
        let copied = point;

        // Verify copy worked by using the copied value
        assert_eq!(copied.dim(), cloned.dim());

        // Test that point can be used in collections requiring Hash + Eq
        let mut set = std::collections::HashSet::new();
        set.insert(point);
        assert!(set.contains(&point));
    }
}
