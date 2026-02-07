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
#![forbid(unsafe_code)]

use crate::geometry::traits::coordinate::{
    Coordinate, CoordinateConversionError, CoordinateScalar, CoordinateValidationError,
};
use num_traits::cast;
use serde::de::{Error, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use std::any;
use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

// =============================================================================
// POINT STRUCT DEFINITION
// =============================================================================

#[derive(Clone, Copy, Debug)]
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
///
/// # Examples
///
/// ```rust
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
///
/// let p = Point::new([1.0, 2.0]);
/// assert_eq!(p.coords(), &[1.0, 2.0]);
/// ```
pub struct Point<T, const D: usize>
where
    T: CoordinateScalar,
{
    /// The coordinates of the point.
    coords: [T; D],
}

// =============================================================================
// PUBLIC API
// =============================================================================

impl<T, const D: usize> Point<T, D>
where
    T: CoordinateScalar,
{
    /// Returns a reference to the point's coordinates as an array.
    ///
    /// This method provides read-only access to the internal coordinate array
    /// without copying. For owned coordinates, use the `Into<[T; D]>` trait
    /// implementation via `.into()`.
    ///
    /// Note: In highly generic code (e.g., `K::Scalar`, const `D`), prefer
    /// `coords()` or `to_array()` over `into()` to avoid type inference ambiguity.
    ///
    /// # Example
    ///
    /// ```rust
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    ///
    /// let point = Point::new([1.0, 2.0, 3.0]);
    /// let coords = point.coords();
    /// assert_eq!(coords, &[1.0, 2.0, 3.0]);
    ///
    /// // For owned coordinates, use Into
    /// let owned_coords: [f64; 3] = point.into();
    /// assert_eq!(owned_coords, [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    #[must_use]
    pub const fn coords(&self) -> &[T; D] {
        &self.coords
    }
}

// =============================================================================
// TRAIT IMPLEMENTATIONS
// =============================================================================

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
            coord.hash_scalar(state);
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

// Implement PartialOrd using OrderedCmp for consistent ordering with special floating-point values
impl<T, const D: usize> PartialOrd for Point<T, D>
where
    T: CoordinateScalar,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Perform lexicographic comparison using ordered comparison for each coordinate
        for (a, b) in self.coords.iter().zip(other.coords.iter()) {
            match a.ordered_partial_cmp(b) {
                Some(Ordering::Equal) => {}
                other_ordering => return other_ordering,
            }
        }
        Some(Ordering::Equal)
    }
}

// Manual implementations for traits that can't be derived due to [T; D] limitations

// Implement Default manually
impl<T, const D: usize> Default for Point<T, D>
where
    T: CoordinateScalar,
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
    T: CoordinateScalar,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeTuple;
        let mut tuple = serializer.serialize_tuple(D)?;
        for coord in &self.coords {
            if coord.is_finite_generic() {
                tuple.serialize_element(coord)?;
            } else if coord.is_nan() {
                // Serialize NaN as null for JSON compatibility
                tuple.serialize_element(&Option::<T>::None)?;
            } else if coord.is_infinite() {
                if coord.is_sign_positive() {
                    // Serialize positive infinity as the string "Infinity"
                    tuple.serialize_element("Infinity")?;
                } else {
                    // Serialize negative infinity as the string "-Infinity"
                    tuple.serialize_element("-Infinity")?;
                }
            } else {
                // Fallback for any other non-finite values
                tuple.serialize_element(&Option::<T>::None)?;
            }
        }
        tuple.end()
    }
}

/// Format-agnostic representation for coordinate values during deserialization.
/// This enum allows the deserializer to work with any format (JSON, CBOR, bincode, etc.)
/// without being tied to specific format types.
#[derive(Deserialize)]
#[serde(untagged)]
enum CoordRepr<T> {
    /// Regular numeric value
    Num(T),
    /// String representation (case-insensitive special values: "Infinity"/"Inf", "-Infinity"/"-Inf", "NaN")
    Str(String),
    /// Null value (will be converted to NaN)
    Null,
}

// Implement Deserialize manually with null -> NaN mapping
impl<'de, T, const D: usize> Deserialize<'de> for Point<T, D>
where
    T: CoordinateScalar,
{
    fn deserialize<DE>(deserializer: DE) -> Result<Self, DE::Error>
    where
        DE: serde::Deserializer<'de>,
    {
        struct ArrayVisitor<T, const D: usize>(PhantomData<T>);

        impl<'de, T, const D: usize> Visitor<'de> for ArrayVisitor<T, D>
        where
            T: CoordinateScalar,
        {
            type Value = Point<T, D>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_fmt(format_args!(
                    "an array of {D} coordinates (numbers, null, \"Infinity\", \"-Infinity\", \"NaN\", or their case-insensitive variants)"
                ))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                // Collect coordinates into a Vec first, then convert to array
                let mut coords = Vec::with_capacity(D);
                for i in 0..D {
                    // Deserialize each element using the format-agnostic enum
                    let element: CoordRepr<T> = seq
                        .next_element()?
                        .ok_or_else(|| Error::invalid_length(i, &self))?;

                    let coord = match element {
                        CoordRepr::Num(value) => {
                            // Handle regular numeric values - already the correct type
                            value
                        }
                        CoordRepr::Str(s) => {
                            // Handle special string representations (case-insensitive)
                            let sl = s.trim().to_ascii_lowercase();
                            match sl.as_str() {
                                "infinity" | "inf" => T::infinity(),
                                "-infinity" | "-inf" => T::neg_infinity(),
                                "nan" => T::nan(),
                                _ => {
                                    return Err(Error::custom(format!(
                                        "Unknown special value: {s}"
                                    )));
                                }
                            }
                        }
                        CoordRepr::Null => {
                            // Handle null values as NaN
                            T::nan()
                        }
                    };

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
    T: cast::NumCast + fmt::Debug,
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
                    from_type: any::type_name::<T>(),
                    to_type: any::type_name::<U>(),
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
    use std::cmp::Ordering;
    use std::collections::hash_map::DefaultHasher;
    use std::collections::{HashMap, HashSet};
    use std::hash::{Hash, Hasher};
    use std::mem;

    // Helper function to get hash value for any hashable type
    fn get_hash<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
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
    // MACROS FOR DIMENSIONAL TESTING
    // =============================================================================

    /// Macro to test basic point operations across multiple dimensions (2D-5D).
    ///
    /// This macro generates tests for common point operations across different
    /// dimensionalities, reducing code duplication while maintaining explicit
    /// test coverage.
    macro_rules! test_point_across_dimensions {
        // Test point creation and basic properties
        (creation: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D
                let point_2d = Point::new([1.0, 2.0]);
                assert_relative_eq!(point_2d.to_array().as_slice(), [1.0, 2.0].as_slice());
                assert_eq!(point_2d.dim(), 2);

                // 3D
                let point_3d = Point::new([1.0, 2.0, 3.0]);
                assert_relative_eq!(point_3d.to_array().as_slice(), [1.0, 2.0, 3.0].as_slice());
                assert_eq!(point_3d.dim(), 3);

                // 4D
                let point_4d = Point::new([1.0, 2.0, 3.0, 4.0]);
                assert_relative_eq!(
                    point_4d.to_array().as_slice(),
                    [1.0, 2.0, 3.0, 4.0].as_slice()
                );
                assert_eq!(point_4d.dim(), 4);

                // 5D
                let point_5d = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
                assert_relative_eq!(
                    point_5d.to_array().as_slice(),
                    [1.0, 2.0, 3.0, 4.0, 5.0].as_slice()
                );
                assert_eq!(point_5d.dim(), 5);
            }
        };

        // Test point equality across dimensions
        (equality: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D
                let p2d_a = Point::new([1.0, 2.0]);
                let p2d_b = Point::new([1.0, 2.0]);
                let p2d_c = Point::new([1.0, 3.0]);
                assert_eq!(p2d_a, p2d_b);
                assert_ne!(p2d_a, p2d_c);

                // 3D
                let p3d_a = Point::new([1.0, 2.0, 3.0]);
                let p3d_b = Point::new([1.0, 2.0, 3.0]);
                let p3d_c = Point::new([1.0, 2.0, 4.0]);
                assert_eq!(p3d_a, p3d_b);
                assert_ne!(p3d_a, p3d_c);

                // 4D
                let p4d_a = Point::new([1.0, 2.0, 3.0, 4.0]);
                let p4d_b = Point::new([1.0, 2.0, 3.0, 4.0]);
                let p4d_c = Point::new([1.0, 2.0, 3.0, 5.0]);
                assert_eq!(p4d_a, p4d_b);
                assert_ne!(p4d_a, p4d_c);

                // 5D
                let p5d_a = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
                let p5d_b = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
                let p5d_c = Point::new([1.0, 2.0, 3.0, 4.0, 6.0]);
                assert_eq!(p5d_a, p5d_b);
                assert_ne!(p5d_a, p5d_c);
            }
        };

        // Test point hashing across dimensions
        (hashing: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D
                let p2d_a = Point::new([1.0, 2.0]);
                let p2d_b = Point::new([1.0, 2.0]);
                assert_eq!(get_hash(&p2d_a), get_hash(&p2d_b));

                // 3D
                let p3d_a = Point::new([1.0, 2.0, 3.0]);
                let p3d_b = Point::new([1.0, 2.0, 3.0]);
                assert_eq!(get_hash(&p3d_a), get_hash(&p3d_b));

                // 4D
                let p4d_a = Point::new([1.0, 2.0, 3.0, 4.0]);
                let p4d_b = Point::new([1.0, 2.0, 3.0, 4.0]);
                assert_eq!(get_hash(&p4d_a), get_hash(&p4d_b));

                // 5D
                let p5d_a = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
                let p5d_b = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
                assert_eq!(get_hash(&p5d_a), get_hash(&p5d_b));
            }
        };

        // Test point ordering across dimensions
        (ordering: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D - lexicographic ordering
                let p2d_a = Point::new([1.0, 2.0]);
                let p2d_b = Point::new([1.0, 3.0]);
                assert!(p2d_a < p2d_b);
                assert!(p2d_b > p2d_a);

                // 3D
                let p3d_a = Point::new([1.0, 2.0, 3.0]);
                let p3d_b = Point::new([1.0, 2.0, 4.0]);
                assert!(p3d_a < p3d_b);
                assert!(p3d_b > p3d_a);

                // 4D
                let p4d_a = Point::new([1.0, 2.0, 3.0, 4.0]);
                let p4d_b = Point::new([1.0, 2.0, 3.0, 5.0]);
                assert!(p4d_a < p4d_b);
                assert!(p4d_b > p4d_a);

                // 5D
                let p5d_a = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
                let p5d_b = Point::new([1.0, 2.0, 3.0, 4.0, 6.0]);
                assert!(p5d_a < p5d_b);
                assert!(p5d_b > p5d_a);
            }
        };

        // Test point validation across dimensions
        (validation: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D - valid and invalid
                let valid_2d = Point::new([1.0, 2.0]);
                assert!(valid_2d.validate().is_ok());
                let invalid_2d = Point::new([f64::NAN, 2.0]);
                assert!(invalid_2d.validate().is_err());

                // 3D
                let valid_3d = Point::new([1.0, 2.0, 3.0]);
                assert!(valid_3d.validate().is_ok());
                let invalid_3d = Point::new([1.0, f64::INFINITY, 3.0]);
                assert!(invalid_3d.validate().is_err());

                // 4D
                let valid_4d = Point::new([1.0, 2.0, 3.0, 4.0]);
                assert!(valid_4d.validate().is_ok());
                let invalid_4d = Point::new([1.0, 2.0, f64::NEG_INFINITY, 4.0]);
                assert!(invalid_4d.validate().is_err());

                // 5D
                let valid_5d = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
                assert!(valid_5d.validate().is_ok());
                let invalid_5d = Point::new([1.0, 2.0, 3.0, f64::NAN, 5.0]);
                assert!(invalid_5d.validate().is_err());
            }
        };

        // Test point serialization across dimensions
        (serialization: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D
                let p2d = Point::new([1.0, 2.0]);
                let json2d = serde_json::to_string(&p2d).unwrap();
                let de2d: Point<f64, 2> = serde_json::from_str(&json2d).unwrap();
                assert_eq!(p2d, de2d);

                // 3D
                let p3d = Point::new([1.0, 2.0, 3.0]);
                let json3d = serde_json::to_string(&p3d).unwrap();
                let de3d: Point<f64, 3> = serde_json::from_str(&json3d).unwrap();
                assert_eq!(p3d, de3d);

                // 4D
                let p4d = Point::new([1.0, 2.0, 3.0, 4.0]);
                let json4d = serde_json::to_string(&p4d).unwrap();
                let de4d: Point<f64, 4> = serde_json::from_str(&json4d).unwrap();
                assert_eq!(p4d, de4d);

                // 5D
                let p5d = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
                let json5d = serde_json::to_string(&p5d).unwrap();
                let de5d: Point<f64, 5> = serde_json::from_str(&json5d).unwrap();
                assert_eq!(p5d, de5d);
            }
        };

        // Test point origin across dimensions
        (origin: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D
                let origin_2d: Point<f64, 2> = Point::origin();
                assert_relative_eq!(origin_2d.to_array().as_slice(), [0.0, 0.0].as_slice());

                // 3D
                let origin_3d: Point<f64, 3> = Point::origin();
                assert_relative_eq!(origin_3d.to_array().as_slice(), [0.0, 0.0, 0.0].as_slice());

                // 4D
                let origin_4d: Point<f64, 4> = Point::origin();
                assert_relative_eq!(
                    origin_4d.to_array().as_slice(),
                    [0.0, 0.0, 0.0, 0.0].as_slice()
                );

                // 5D
                let origin_5d: Point<f64, 5> = Point::origin();
                assert_relative_eq!(
                    origin_5d.to_array().as_slice(),
                    [0.0, 0.0, 0.0, 0.0, 0.0].as_slice()
                );
            }
        };

        // Test HashMap usage across dimensions
        (hashmap: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D
                let mut map2d: HashMap<Point<f64, 2>, i32> = HashMap::new();
                let p2d = Point::new([1.0, 2.0]);
                map2d.insert(p2d, 42);
                assert_eq!(map2d.get(&Point::new([1.0, 2.0])), Some(&42));

                // 3D
                let mut map3d: HashMap<Point<f64, 3>, i32> = HashMap::new();
                let p3d = Point::new([1.0, 2.0, 3.0]);
                map3d.insert(p3d, 42);
                assert_eq!(map3d.get(&Point::new([1.0, 2.0, 3.0])), Some(&42));

                // 4D
                let mut map4d: HashMap<Point<f64, 4>, i32> = HashMap::new();
                let p4d = Point::new([1.0, 2.0, 3.0, 4.0]);
                map4d.insert(p4d, 42);
                assert_eq!(map4d.get(&Point::new([1.0, 2.0, 3.0, 4.0])), Some(&42));

                // 5D
                let mut map5d: HashMap<Point<f64, 5>, i32> = HashMap::new();
                let p5d = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
                map5d.insert(p5d, 42);
                assert_eq!(map5d.get(&Point::new([1.0, 2.0, 3.0, 4.0, 5.0])), Some(&42));
            }
        };

        // Test Copy semantics across dimensions
        (copy: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D
                let p2d_original = Point::new([1.0, 2.0]);
                let p2d_copy = p2d_original;
                assert_eq!(p2d_original, p2d_copy);
                assert_relative_eq!(
                    p2d_original.to_array().as_slice(),
                    p2d_copy.to_array().as_slice()
                );

                // 3D
                let p3d_original = Point::new([1.0, 2.0, 3.0]);
                let p3d_copy = p3d_original;
                assert_eq!(p3d_original, p3d_copy);

                // 4D
                let p4d_original = Point::new([1.0, 2.0, 3.0, 4.0]);
                let p4d_copy = p4d_original;
                assert_eq!(p4d_original, p4d_copy);

                // 5D
                let p5d_original = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
                let p5d_copy = p5d_original;
                assert_eq!(p5d_original, p5d_copy);
            }
        };
    }

    // Generate dimensional tests using the macro
    test_point_across_dimensions!(creation: point_creation_dimensional);
    test_point_across_dimensions!(equality: point_equality_dimensional);
    test_point_across_dimensions!(hashing: point_hashing_dimensional);
    test_point_across_dimensions!(ordering: point_ordering_dimensional);
    test_point_across_dimensions!(validation: point_validation_dimensional);
    test_point_across_dimensions!(serialization: point_serialization_dimensional);
    test_point_across_dimensions!(origin: point_origin_dimensional);
    test_point_across_dimensions!(hashmap: point_hashmap_dimensional);
    test_point_across_dimensions!(copy: point_copy_dimensional);

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

    // point_new, point_copy, point_dim removed - covered by point_creation_dimensional and point_copy_dimensional

    #[test]
    fn point_coords() {
        // Test coords() method provides read-only access
        let point = Point::new([1.0, 2.0, 3.0]);
        let coords_ref = point.coords();
        assert_relative_eq!(
            coords_ref.as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );

        // Test that it returns a reference (not copying)
        assert_eq!(coords_ref.len(), 3);
        assert_relative_eq!(coords_ref[0], 1.0, epsilon = 1e-9);
        assert_relative_eq!(coords_ref[1], 2.0, epsilon = 1e-9);
        assert_relative_eq!(coords_ref[2], 3.0, epsilon = 1e-9);

        // Test with different dimensions
        let point_2d = Point::new([5.5, -2.5]);
        assert_relative_eq!(
            point_2d.coords().as_slice(),
            [5.5, -2.5].as_slice(),
            epsilon = 1e-9
        );

        let point_4d = Point::new([1.0, 2.0, 3.0, 4.0]);
        assert_relative_eq!(
            point_4d.coords().as_slice(),
            [1.0, 2.0, 3.0, 4.0].as_slice(),
            epsilon = 1e-9
        );

        // Test with f32
        let point_f32 = Point::new([1.0f32, 2.0f32, 3.0f32]);
        assert_relative_eq!(
            point_f32.coords().as_slice(),
            [1.0f32, 2.0f32, 3.0f32].as_slice(),
            epsilon = 1e-6
        );

        // Test with 5D
        let point_5d = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_relative_eq!(
            point_5d.coords().as_slice(),
            [1.0, 2.0, 3.0, 4.0, 5.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(point_5d.coords().len(), 5);

        println!("coords() provides efficient read-only access to coordinates");
    }

    // point_origin removed - covered by point_origin_dimensional

    // point_serialization removed - basic cases covered by point_serialization_dimensional

    #[test]
    fn point_from_array_f32_to_f64() {
        let coords = [1.5f32, 2.5f32, 3.5f32, 4.5f32];
        let point: Point<f64, 4> = Point::new(coords.map(Into::into));

        let result_coords = point.to_array();
        assert_relative_eq!(
            result_coords.as_slice(),
            [1.5, 2.5, 3.5, 4.5].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(point.dim(), 4);
    }

    #[test]
    fn point_type_conversions() {
        // Test same-type conversion (f32 to f32)
        let coords_f32 = [1.5f32, 2.5f32, 3.5f32];
        let point_f32: Point<f32, 3> = Point::new(coords_f32);
        let result_f32 = point_f32.to_array();
        assert_relative_eq!(
            result_f32.as_slice(),
            [1.5f32, 2.5f32, 3.5f32].as_slice(),
            epsilon = 1e-9
        );

        // Test safe upcast conversion (f32 to f64)
        let coords_f32_upcast = [1.5f32, 2.5f32];
        let point_f64: Point<f64, 2> = Point::new(coords_f32_upcast.map(Into::into));
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

    // point_hash, point_hash_in_hashmap, point_partial_eq, point_partial_ord
    // removed - covered by point_hashing_dimensional, point_hashmap_dimensional,
    // point_equality_dimensional, and point_ordering_dimensional

    // point_multidimensional_comprehensive removed - covered by dimensional macro tests
    // (point_creation_dimensional, point_origin_dimensional)

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

        // Test with very large and small numbers (roundtrip)
        let point_large = Point::new([1e100, -1e100, 0.0]);
        let serialized_large = serde_json::to_string(&point_large).unwrap();
        let deserialized_large: Point<f64, 3> = serde_json::from_str(&serialized_large).unwrap();
        assert_eq!(point_large, deserialized_large);

        let point_small = Point::new([1e-100, -1e-100, 0.0]);
        let serialized_small = serde_json::to_string(&point_small).unwrap();
        let deserialized_small: Point<f64, 3> = serde_json::from_str(&serialized_small).unwrap();
        assert_eq!(point_small, deserialized_small);
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
    fn point_special_values_hash_consistency() {
        // Test that OrderedFloat provides consistent hashing for NaN and infinity values

        // Test NaN hash consistency
        let point_nan1 = Point::new([f64::NAN, 2.0]);
        let point_nan2 = Point::new([f64::NAN, 2.0]);
        assert_eq!(get_hash(&point_nan1), get_hash(&point_nan2));

        // Test infinity hash consistency
        let point_pos_inf1 = Point::new([f64::INFINITY, 2.0]);
        let point_pos_inf2 = Point::new([f64::INFINITY, 2.0]);
        assert_eq!(get_hash(&point_pos_inf1), get_hash(&point_pos_inf2));

        let point_neg_inf1 = Point::new([f64::NEG_INFINITY, 2.0]);
        let point_neg_inf2 = Point::new([f64::NEG_INFINITY, 2.0]);
        assert_eq!(get_hash(&point_neg_inf1), get_hash(&point_neg_inf2));

        // Positive and negative infinity should hash differently
        assert_ne!(get_hash(&point_pos_inf1), get_hash(&point_neg_inf1));

        // Test HashMap usage with special values
        let mut map: HashMap<Point<f64, 2>, i32> = HashMap::new();
        let point_nan_lookup = Point::new([f64::NAN, 2.0]);
        let point_inf_lookup = Point::new([f64::INFINITY, 2.0]);

        map.insert(point_nan1, 100);
        map.insert(point_pos_inf1, 200);

        // Should be able to retrieve using equivalent points
        assert_eq!(map.get(&point_nan_lookup), Some(&100));
        assert_eq!(map.get(&point_inf_lookup), Some(&200));
        assert_eq!(map.len(), 2);

        // Test with f32 types
        let point_f32_nan1 = Point::new([f32::NAN, 1.0f32]);
        let point_f32_nan2 = Point::new([f32::NAN, 1.0f32]);
        assert_eq!(get_hash(&point_f32_nan1), get_hash(&point_f32_nan2));
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
    fn point_nan_equality_properties() {
        // Test that NaN equality follows mathematical properties: reflexivity, symmetry, and transitivity

        // Test reflexivity: NaN points are equal to themselves
        let point_nan = Point::new([f64::NAN, f64::NAN, f64::NAN]);
        assert_eq!(point_nan, point_nan);
        let point_mixed = Point::new([1.0, f64::NAN, 3.0, f64::INFINITY]);
        assert_eq!(point_mixed, point_mixed);

        // Test symmetry: if a == b, then b == a
        let point_a = Point::new([f64::NAN, 2.0, f64::INFINITY]);
        let point_b = Point::new([f64::NAN, 2.0, f64::INFINITY]);
        assert_eq!(point_a, point_b);
        assert_eq!(point_b, point_a);

        // Test transitivity: if a == b and b == c, then a == c
        let point_c = Point::new([f64::NAN, 2.0, f64::INFINITY]);
        assert_eq!(point_a, point_b);
        assert_eq!(point_b, point_c);
        assert_eq!(point_a, point_c);

        // Test with f32 types
        let point_f32_a = Point::new([f32::NAN, 1.0f32, f32::NEG_INFINITY]);
        let point_f32_b = Point::new([f32::NAN, 1.0f32, f32::NEG_INFINITY]);
        assert_eq!(point_f32_a, point_f32_b);
        assert_eq!(point_f32_b, point_f32_a);
    }

    #[test]
    fn point_nan_different_bit_patterns() {
        // Test that different NaN bit patterns are considered equal
        // Note: Rust's f64::NAN is a specific bit pattern, but there are many possible NaN values

        // Create different NaN values
        let nan1 = f64::NAN;
        #[expect(clippy::zero_divided_by_zero)]
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
        #[expect(clippy::zero_divided_by_zero)]
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
    #[expect(clippy::cast_precision_loss)]
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
        #[expect(clippy::clone_on_copy)]
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
    fn point_partial_ord_special_values() {
        // Test NaN vs NaN comparison (should be Some(Equal) with OrderedCmp)
        let point_nan1 = Point::new([f64::NAN, 1.0]);
        let point_nan2 = Point::new([f64::NAN, 1.0]);
        let point_normal = Point::new([1.0, 1.0]);

        // NaN should be equal to itself
        assert_eq!(point_nan1.partial_cmp(&point_nan2), Some(Ordering::Equal));

        // Test NaN vs normal comparison
        // In OrderedFloat semantics, NaN is greater than all other values
        assert_eq!(
            point_nan1.partial_cmp(&point_normal),
            Some(Ordering::Greater)
        );
        assert_eq!(point_normal.partial_cmp(&point_nan1), Some(Ordering::Less));

        // Test infinity vs normal comparison
        let point_inf = Point::new([f64::INFINITY, 1.0]);
        let point_neg_inf = Point::new([f64::NEG_INFINITY, 1.0]);
        let point_normal2 = Point::new([2.0, 1.0]);

        // Infinity should be greater than normal values
        assert_eq!(
            point_inf.partial_cmp(&point_normal2),
            Some(Ordering::Greater)
        );
        assert_eq!(point_normal2.partial_cmp(&point_inf), Some(Ordering::Less));

        // Test negative infinity vs normal
        assert_eq!(
            point_neg_inf.partial_cmp(&point_normal2),
            Some(Ordering::Less)
        );
        assert_eq!(
            point_normal2.partial_cmp(&point_neg_inf),
            Some(Ordering::Greater)
        );

        // Positive infinity should be greater than negative infinity
        assert_eq!(
            point_inf.partial_cmp(&point_neg_inf),
            Some(Ordering::Greater)
        );
        assert_eq!(point_neg_inf.partial_cmp(&point_inf), Some(Ordering::Less));

        // NaN should be greater than infinity in OrderedFloat semantics
        assert_eq!(point_nan1.partial_cmp(&point_inf), Some(Ordering::Greater));
        assert_eq!(point_inf.partial_cmp(&point_nan1), Some(Ordering::Less));

        // Test that comparison operators work
        assert!(point_normal < point_inf); // Normal < Infinity
        assert!(point_normal2 < point_inf); // Normal < Infinity
        assert!(point_neg_inf < point_normal); // -Infinity < Normal
        assert!(point_inf > point_normal); // Infinity > Normal
        assert!(point_nan1 > point_normal); // NaN > Normal (in OrderedFloat)
        assert!(point_nan1 > point_inf); // NaN > Infinity (in OrderedFloat)

        // Test mixed coordinates
        let point_mixed1 = Point::new([1.0, f64::NAN]);
        let point_mixed2 = Point::new([1.0, 2.0]);
        assert_eq!(
            point_mixed1.partial_cmp(&point_mixed2),
            Some(Ordering::Greater)
        ); // NaN in second coordinate

        let point_mixed3 = Point::new([f64::NEG_INFINITY, 2.0]);
        let point_mixed4 = Point::new([1.0, 2.0]);
        assert_eq!(
            point_mixed3.partial_cmp(&point_mixed4),
            Some(Ordering::Less)
        ); // -inf in first coordinate

        // Test with f32 as well
        let point_f32_nan = Point::new([f32::NAN, 1.0f32]);
        let point_f32_normal = Point::new([1.0f32, 1.0f32]);
        assert_eq!(
            point_f32_nan.partial_cmp(&point_f32_normal),
            Some(Ordering::Greater)
        );
        assert!(point_f32_nan > point_f32_normal);
    }

    #[test]
    fn point_memory_layout_and_size() {
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
    fn point_serialize_nan_infinity_comprehensive() {
        // Test comprehensive serialization of NaN and infinity values

        // Single NaN coordinate
        let point_nan_single = Point::new([f64::NAN, 1.0, 2.0]);
        let json_nan_single = serde_json::to_string(&point_nan_single).unwrap();
        assert_eq!(json_nan_single, "[null,1.0,2.0]");

        // Multiple NaN coordinates
        let point_nan_multiple = Point::new([f64::NAN, f64::NAN, 1.0]);
        let json_nan_multiple = serde_json::to_string(&point_nan_multiple).unwrap();
        assert_eq!(json_nan_multiple, "[null,null,1.0]");

        // All NaN coordinates
        let point_all_nan = Point::new([f64::NAN, f64::NAN]);
        let json_all_nan = serde_json::to_string(&point_all_nan).unwrap();
        assert_eq!(json_all_nan, "[null,null]");

        // Single positive infinity
        let point_pos_inf = Point::new([f64::INFINITY, 1.0]);
        let json_pos_inf = serde_json::to_string(&point_pos_inf).unwrap();
        assert_eq!(json_pos_inf, "[\"Infinity\",1.0]");

        // Single negative infinity
        let point_neg_inf = Point::new([1.0, f64::NEG_INFINITY]);
        let json_neg_inf = serde_json::to_string(&point_neg_inf).unwrap();
        assert_eq!(json_neg_inf, "[1.0,\"-Infinity\"]");

        // Mixed NaN and infinity
        let point_mixed = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0]);
        let json_mixed = serde_json::to_string(&point_mixed).unwrap();
        assert_eq!(json_mixed, "[null,\"Infinity\",\"-Infinity\",1.0]");

        // All special values
        let point_all_special = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
        let json_all_special = serde_json::to_string(&point_all_special).unwrap();
        assert_eq!(json_all_special, "[null,\"Infinity\",\"-Infinity\"]");
    }

    #[test]
    fn point_serialize_f32_nan_infinity() {
        // Test f32 NaN and infinity serialization

        let point_f32_nan = Point::new([f32::NAN, 1.0f32]);
        let json_f32_nan = serde_json::to_string(&point_f32_nan).unwrap();
        assert_eq!(json_f32_nan, "[null,1.0]");

        let point_f32_inf = Point::new([f32::INFINITY, f32::NEG_INFINITY]);
        let json_f32_inf = serde_json::to_string(&point_f32_inf).unwrap();
        assert_eq!(json_f32_inf, "[\"Infinity\",\"-Infinity\"]");
    }

    #[test]
    fn point_deserialize_null_maps_to_nan() {
        // With custom Deserialize, JSON null deserializes to NaN
        let json = "[null,1.0,2.0]";
        let p: Point<f64, 3> = serde_json::from_str(json).unwrap();
        let coords = p.to_array();
        assert!(coords[0].is_nan());
        assert_relative_eq!(coords[1], 1.0);
        assert_relative_eq!(coords[2], 2.0);
    }

    #[test]
    fn point_deserialize_format_agnostic_comprehensive() {
        // Test the format-agnostic deserialization improvements with CoordRepr enum

        // Test 1: Regular numeric values (NumCast improvement)
        let json_regular = "[1.0, 2.5, 4.25]";
        let point_regular: Point<f64, 3> = serde_json::from_str(json_regular).unwrap();
        assert_relative_eq!(
            point_regular.to_array().as_slice(),
            [1.0, 2.5, 4.25].as_slice()
        );

        // Test 2: Mixed special values using format-agnostic approach
        let json_special = "[1.0, null, \"Infinity\", \"-Infinity\"]";
        let point_special: Point<f64, 4> = serde_json::from_str(json_special).unwrap();
        let coords = point_special.to_array();
        assert_relative_eq!(coords[0], 1.0);
        assert!(coords[1].is_nan());
        assert!(coords[2].is_infinite() && coords[2].is_sign_positive());
        assert!(coords[3].is_infinite() && coords[3].is_sign_negative());

        // Test 3: All null values
        let json_all_null = "[null, null, null]";
        let point_all_null: Point<f64, 3> = serde_json::from_str(json_all_null).unwrap();
        let all_null_coords = point_all_null.to_array();
        assert!(all_null_coords.iter().all(|&x| x.is_nan()));

        // Test 4: All special string values
        let json_all_special = "[\"Infinity\", \"-Infinity\", \"Infinity\"]";
        let point_all_special: Point<f64, 3> = serde_json::from_str(json_all_special).unwrap();
        let special_coords = point_all_special.to_array();
        assert!(special_coords[0].is_infinite() && special_coords[0].is_sign_positive());
        assert!(special_coords[1].is_infinite() && special_coords[1].is_sign_negative());
        assert!(special_coords[2].is_infinite() && special_coords[2].is_sign_positive());

        // Test 5: Serialization roundtrip with format-agnostic deserialization
        let original = Point::new([1.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: Point<f64, 5> = serde_json::from_str(&serialized).unwrap();

        // Compare coordinates (can't use == because of NaN)
        let orig_coords = original.to_array();
        let deser_coords = deserialized.to_array();
        assert_relative_eq!(orig_coords[0], deser_coords[0]);
        assert!(orig_coords[1].is_nan() && deser_coords[1].is_nan());
        assert!(
            orig_coords[2].is_infinite()
                && orig_coords[2].is_sign_positive()
                && deser_coords[2].is_infinite()
                && deser_coords[2].is_sign_positive()
        );
        assert!(
            orig_coords[3].is_infinite()
                && orig_coords[3].is_sign_negative()
                && deser_coords[3].is_infinite()
                && deser_coords[3].is_sign_negative()
        );
        assert_relative_eq!(orig_coords[4], deser_coords[4]);

        // Test 6: Test with different numeric types to verify NumCast improvement
        let json_f32 = "[1.5, 2.5]";
        let point_f32: Point<f32, 2> = serde_json::from_str(json_f32).unwrap();
        assert_relative_eq!(point_f32.to_array().as_slice(), [1.5f32, 2.5f32].as_slice());

        // Test 7: Invalid special string should fail gracefully
        let json_invalid = "[1.0, \"NotASpecialValue\", 2.0]";
        let result: Result<Point<f64, 3>, _> = serde_json::from_str(json_invalid);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Unknown special value"));
    }

    #[test]
    fn point_deserialize_case_insensitive_special_values() {
        // Test case-insensitive deserialization of special values

        // Test case-insensitive "Infinity" variants
        let test_cases = vec![
            (r#"["infinity", 1.0]"#, "lowercase infinity"),
            (r#"["INFINITY", 1.0]"#, "uppercase infinity"),
            (r#"["Infinity", 1.0]"#, "mixed case infinity"),
            (r#"["inf", 1.0]"#, "lowercase inf"),
            (r#"["INF", 1.0]"#, "uppercase inf"),
            (r#"["Inf", 1.0]"#, "mixed case inf"),
        ];

        for (json_str, description) in test_cases {
            let point: Point<f64, 2> = serde_json::from_str(json_str).unwrap_or_else(|e| {
                panic!("Failed to deserialize {description} ({json_str}): {e}")
            });

            assert!(
                point.coords[0].is_infinite() && point.coords[0].is_sign_positive(),
                "First coordinate should be positive infinity for {description}"
            );
            assert_relative_eq!(point.coords[1], 1.0, epsilon = 1e-10);
        }

        // Test negative infinity variants
        let neg_inf_cases = vec![
            (r#"["-infinity", 2.0]"#, "lowercase -infinity"),
            (r#"["-INFINITY", 2.0]"#, "uppercase -infinity"),
            (r#"["-Infinity", 2.0]"#, "mixed case -infinity"),
            (r#"["-inf", 2.0]"#, "lowercase -inf"),
            (r#"["-INF", 2.0]"#, "uppercase -inf"),
            (r#"["-Inf", 2.0]"#, "mixed case -inf"),
        ];

        for (json_str, description) in neg_inf_cases {
            let point: Point<f64, 2> = serde_json::from_str(json_str).unwrap_or_else(|e| {
                panic!("Failed to deserialize {description} ({json_str}): {e}")
            });

            assert!(
                point.coords[0].is_infinite() && point.coords[0].is_sign_negative(),
                "First coordinate should be negative infinity for {description}"
            );
            assert_relative_eq!(point.coords[1], 2.0, epsilon = 1e-10);
        }

        // Test case-insensitive "NaN" variants
        let nan_cases = vec![
            (r#"["nan", 3.0]"#, "lowercase nan"),
            (r#"["NaN", 3.0]"#, "mixed case NaN"),
            (r#"["NAN", 3.0]"#, "uppercase NAN"),
            (r#"["Nan", 3.0]"#, "title case Nan"),
        ];

        for (json_str, description) in nan_cases {
            let point: Point<f64, 2> = serde_json::from_str(json_str).unwrap_or_else(|e| {
                panic!("Failed to deserialize {description} ({json_str}): {e}")
            });

            assert!(
                point.coords[0].is_nan(),
                "First coordinate should be NaN for {description}"
            );
            assert_relative_eq!(point.coords[1], 3.0, epsilon = 1e-10);
        }

        // Test whitespace trimming
        let whitespace_cases = vec![
            (r#"[" infinity ", 1.0]"#, "spaces around infinity"),
            (r#"["\tinf\n", 2.0]"#, "tabs and newlines around inf"),
            (r#"["  NaN  ", 3.0]"#, "spaces around NaN"),
        ];

        for (json_str, description) in whitespace_cases {
            let point: Point<f64, 2> = serde_json::from_str(json_str).unwrap_or_else(|e| {
                panic!("Failed to deserialize {description} ({json_str}): {e}")
            });

            if description.contains("infinity") || description.contains("inf") {
                assert!(
                    point.coords[0].is_infinite() && point.coords[0].is_sign_positive(),
                    "First coordinate should be positive infinity for {description}"
                );
            } else {
                assert!(
                    point.coords[0].is_nan(),
                    "First coordinate should be NaN for {description}"
                );
            }
        }

        // Test combined case insensitive values
        let combined = r#"["INFINITY", "-inf", "Nan", 42.0]"#;
        let point: Point<f64, 4> = serde_json::from_str(combined).unwrap();
        assert!(point.coords[0].is_infinite() && point.coords[0].is_sign_positive());
        assert!(point.coords[1].is_infinite() && point.coords[1].is_sign_negative());
        assert!(point.coords[2].is_nan());
        assert_relative_eq!(point.coords[3], 42.0, epsilon = 1e-10);

        // Test that unknown special values still fail
        let invalid = r#"["unknown_special", 1.0]"#;
        let result: Result<Point<f64, 2>, _> = serde_json::from_str(invalid);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unknown special value")
        );
    }

    #[test]
    fn point_serialize_edge_values() {
        // Test serialization of edge finite values

        // Very large but finite values
        let point_max = Point::new([f64::MAX, f64::MIN]);
        let json_max = serde_json::to_string(&point_max).unwrap();
        assert!(!json_max.contains("null")); // Should not be null

        // Very small but finite values
        let point_min = Point::new([f64::MIN_POSITIVE, -f64::MIN_POSITIVE]);
        let json_min = serde_json::to_string(&point_min).unwrap();
        assert!(!json_min.contains("null")); // Should not be null

        // Zero and negative zero
        let point_zero = Point::new([0.0, -0.0]);
        let json_zero = serde_json::to_string(&point_zero).unwrap();
        assert!(!json_zero.contains("null")); // Should not be null
        assert_eq!(json_zero, "[0.0,-0.0]");
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
        let point_from_ref: Point<f64, 3> = Point::new(coords_ref.map(Into::into));
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

    // =============================================================================
    // TryFrom CONVERSION ERROR TESTS
    // =============================================================================

    #[test]
    fn point_try_from_overflow_f64_to_f32() {
        // Test that overflow during f64 to f32 conversion produces NonFiniteValue error
        let large_coords = [f64::MAX, 1.0];
        let result: Result<Point<f32, 2>, _> = Point::try_from(large_coords);

        assert!(result.is_err(), "f64::MAX should overflow when cast to f32");

        if let Err(CoordinateConversionError::NonFiniteValue {
            coordinate_index,
            coordinate_value,
        }) = result
        {
            assert_eq!(coordinate_index, 0);
            assert!(coordinate_value.contains("inf") || coordinate_value.contains("Inf"));
        } else {
            panic!("Expected NonFiniteValue error, got: {result:?}");
        }
    }

    #[test]
    fn point_try_from_negative_overflow_f64_to_f32() {
        // Test negative overflow
        let large_negative_coords = [f64::MIN, 1.0];
        let result: Result<Point<f32, 2>, _> = Point::try_from(large_negative_coords);

        assert!(result.is_err(), "f64::MIN should overflow when cast to f32");

        if let Err(CoordinateConversionError::NonFiniteValue {
            coordinate_index,
            coordinate_value,
        }) = result
        {
            assert_eq!(coordinate_index, 0);
            assert!(coordinate_value.contains("inf") || coordinate_value.contains("Inf"));
        } else {
            panic!("Expected NonFiniteValue error");
        }
    }

    #[test]
    fn point_try_from_multiple_overflow_coordinates() {
        // Test that the first overflowing coordinate is reported
        let coords = [1.0, f64::MAX, f64::MIN, f64::MAX];
        let result: Result<Point<f32, 4>, _> = Point::try_from(coords);

        assert!(result.is_err());

        if let Err(CoordinateConversionError::NonFiniteValue {
            coordinate_index, ..
        }) = result
        {
            // Should report the first overflow at index 1
            assert_eq!(coordinate_index, 1);
        } else {
            panic!("Expected NonFiniteValue error");
        }
    }

    #[test]
    fn point_try_from_successful_conversions() {
        // Test successful conversions that don't overflow

        // f32 to f64 (safe upcast)
        let coords_f32: [f32; 3] = [1.5, -2.5, 3.5];
        let point_f64: Point<f64, 3> = Point::try_from(coords_f32).unwrap();
        assert_relative_eq!(
            point_f64.to_array().as_slice(),
            [1.5f64, -2.5f64, 3.5f64].as_slice(),
            epsilon = 1e-9
        );

        // i32 to f64
        let coords_i32: [i32; 4] = [1, -2, 3, -4];
        let point_from_int: Point<f64, 4> = Point::try_from(coords_i32).unwrap();
        assert_relative_eq!(
            point_from_int.to_array().as_slice(),
            [1.0, -2.0, 3.0, -4.0].as_slice(),
            epsilon = 1e-9
        );

        // Same type conversion (f64 to f64)
        let coords_same: [f64; 2] = [10.0, 20.0];
        let point_same: Point<f64, 2> = Point::try_from(coords_same).unwrap();
        assert_relative_eq!(point_same.to_array().as_slice(), [10.0, 20.0].as_slice());
    }

    #[test]
    fn point_try_from_edge_case_values() {
        // Test with values close to f32 limits (should succeed)
        let coords_near_f32_max: [f64; 2] = [f64::from(f32::MAX), f64::from(f32::MIN)];
        let result: Result<Point<f32, 2>, _> = Point::try_from(coords_near_f32_max);
        assert!(result.is_ok(), "Values within f32 range should convert");

        // Test with zero and negative zero
        let coords_zero: [f64; 2] = [0.0, -0.0];
        let point_zero: Point<f32, 2> = Point::try_from(coords_zero).unwrap();
        assert_relative_eq!(point_zero.to_array()[0], 0.0f32);
        assert_relative_eq!(point_zero.to_array()[1], -0.0f32);

        // Test with very small values
        let coords_small: [f64; 2] = [1e-10, -1e-10];
        let point_small: Point<f32, 2> = Point::try_from(coords_small).unwrap();
        // These may underflow to zero in f32, but should still be finite
        assert!(point_small.to_array()[0].is_finite());
        assert!(point_small.to_array()[1].is_finite());
    }

    #[test]
    fn point_try_from_integer_to_float_conversions() {
        // Test various integer types to floating point

        // u32 to f64
        let coords_u32: [u32; 3] = [100, 200, 300];
        let point_u32: Point<f64, 3> = Point::try_from(coords_u32).unwrap();
        assert_relative_eq!(
            point_u32.to_array().as_slice(),
            [100.0, 200.0, 300.0].as_slice(),
            epsilon = 1e-9
        );

        // i16 to f32
        let coords_i16: [i16; 2] = [-100, 200];
        let point_i16: Point<f32, 2> = Point::try_from(coords_i16).unwrap();
        assert_relative_eq!(
            point_i16.to_array().as_slice(),
            [-100.0f32, 200.0f32].as_slice(),
            epsilon = 1e-6
        );

        // Large but representable integers
        let coords_large_i32: [i32; 2] = [1_000_000, -1_000_000];
        let point_large: Point<f64, 2> = Point::try_from(coords_large_i32).unwrap();
        assert_relative_eq!(
            point_large.to_array().as_slice(),
            [1_000_000.0, -1_000_000.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn point_try_from_all_coordinates_must_be_finite() {
        // Test that all coordinates are validated during conversion

        // Valid conversion - all finite
        let valid_coords: [f32; 3] = [1.0, 2.0, 3.0];
        let result: Result<Point<f64, 3>, _> = Point::try_from(valid_coords);
        assert!(result.is_ok());

        // Invalid - produces infinity after conversion
        let invalid_coords = [1.0, f64::MAX, 3.0];
        let result: Result<Point<f32, 3>, _> = Point::try_from(invalid_coords);
        assert!(
            result.is_err(),
            "Should fail if any coordinate becomes non-finite"
        );
    }

    // =============================================================================
    // DIM() METHOD EXPLICIT TESTS
    // =============================================================================

    #[test]
    fn point_dim_method_explicit() {
        // Test the dim() method explicitly across various dimensions

        let point_1d: Point<f64, 1> = Point::new([1.0]);
        assert_eq!(point_1d.dim(), 1);

        let point_2d: Point<f64, 2> = Point::new([1.0, 2.0]);
        assert_eq!(point_2d.dim(), 2);

        let point_3d: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);
        assert_eq!(point_3d.dim(), 3);

        let point_5d: Point<f64, 5> = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(point_5d.dim(), 5);

        let point_10d: Point<f64, 10> = Point::new([0.0; 10]);
        assert_eq!(point_10d.dim(), 10);

        let point_32d: Point<f64, 32> = Point::new([0.0; 32]);
        assert_eq!(point_32d.dim(), 32);
    }

    #[test]
    fn point_dim_with_different_types() {
        // Test dim() with different coordinate types

        let point_f32: Point<f32, 3> = Point::new([1.0, 2.0, 3.0]);
        assert_eq!(point_f32.dim(), 3);

        let point_f64: Point<f64, 4> = Point::new([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(point_f64.dim(), 4);
    }

    // =============================================================================
    // TO_ARRAY() METHOD EXPLICIT TESTS
    // =============================================================================

    #[test]
    fn point_to_array_explicit() {
        // Test to_array() method explicitly

        let point = Point::new([1.0, 2.0, 3.0]);
        let arr = point.to_array();
        assert_relative_eq!(arr.as_slice(), [1.0, 2.0, 3.0].as_slice());

        // Verify that to_array() returns a copy, not a reference
        let point2 = Point::new([4.0, 5.0]);
        let arr2 = point2.to_array();
        assert_relative_eq!(arr2.as_slice(), [4.0, 5.0].as_slice());

        // Test with different dimensions
        let point_1d = Point::new([42.0]);
        assert_relative_eq!(point_1d.to_array().as_slice(), [42.0].as_slice());

        let point_5d = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_relative_eq!(
            point_5d.to_array().as_slice(),
            [1.0, 2.0, 3.0, 4.0, 5.0].as_slice()
        );
    }

    #[test]
    fn point_to_array_with_special_values() {
        // Test to_array() with special floating-point values

        let point_nan = Point::new([f64::NAN, 1.0, 2.0]);
        let arr = point_nan.to_array();
        assert!(arr[0].is_nan());
        assert_relative_eq!(arr[1], 1.0);
        assert_relative_eq!(arr[2], 2.0);

        let point_inf = Point::new([f64::INFINITY, f64::NEG_INFINITY]);
        let arr_inf = point_inf.to_array();
        assert!(arr_inf[0].is_infinite() && arr_inf[0].is_sign_positive());
        assert!(arr_inf[1].is_infinite() && arr_inf[1].is_sign_negative());
    }

    // =============================================================================
    // ORDERED_EQUALS() AND HASH_COORDINATE() DIRECT TESTS
    // =============================================================================

    #[test]
    fn point_ordered_equals_direct() {
        // Test ordered_equals() method directly

        let point1 = Point::new([1.0, 2.0, 3.0]);
        let point2 = Point::new([1.0, 2.0, 3.0]);
        let point3 = Point::new([1.0, 2.0, 4.0]);

        assert!(point1.ordered_equals(&point2));
        assert!(!point1.ordered_equals(&point3));

        // Test with NaN (should be equal)
        let point_nan1 = Point::new([f64::NAN, 2.0]);
        let point_nan2 = Point::new([f64::NAN, 2.0]);
        assert!(point_nan1.ordered_equals(&point_nan2));

        // Test with infinity
        let point_inf1 = Point::new([f64::INFINITY, 1.0]);
        let point_inf2 = Point::new([f64::INFINITY, 1.0]);
        assert!(point_inf1.ordered_equals(&point_inf2));
    }

    #[test]
    fn point_hash_coordinate_direct() {
        // Test hash_coordinate() method directly

        let point1 = Point::new([1.0, 2.0, 3.0]);
        let point2 = Point::new([1.0, 2.0, 3.0]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        point1.hash_coordinate(&mut hasher1);
        point2.hash_coordinate(&mut hasher2);

        assert_eq!(hasher1.finish(), hasher2.finish());

        // Test with different points
        let point3 = Point::new([1.0, 2.0, 4.0]);
        let mut hasher3 = DefaultHasher::new();
        point3.hash_coordinate(&mut hasher3);

        assert_ne!(hasher1.finish(), hasher3.finish());
    }

    #[test]
    fn point_hash_coordinate_special_values() {
        // Test hash_coordinate() with special values

        let point_nan1 = Point::new([f64::NAN, 2.0]);
        let point_nan2 = Point::new([f64::NAN, 2.0]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        point_nan1.hash_coordinate(&mut hasher1);
        point_nan2.hash_coordinate(&mut hasher2);

        // NaN values should hash consistently
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    // =============================================================================
    // COMPREHENSIVE 1D POINT TESTS
    // =============================================================================

    #[test]
    fn point_1d_comprehensive() {
        // Test 1D points comprehensively

        // Creation
        let point = Point::new([42.0]);
        assert_eq!(point.dim(), 1);
        assert_relative_eq!(point.to_array().as_slice(), [42.0].as_slice());

        // Equality
        let point2 = Point::new([42.0]);
        assert_eq!(point, point2);

        let point3 = Point::new([43.0]);
        assert_ne!(point, point3);

        // Hashing
        assert_eq!(get_hash(&point), get_hash(&point2));
        assert_ne!(get_hash(&point), get_hash(&point3));

        // Ordering
        assert!(point < point3);
        assert!(point3 > point);

        // Validation
        assert!(point.validate().is_ok());
        let invalid_1d = Point::new([f64::NAN]);
        assert!(invalid_1d.validate().is_err());

        // Origin
        let origin: Point<f64, 1> = Point::origin();
        assert_relative_eq!(origin.to_array().as_slice(), [0.0].as_slice());

        // Serialization
        let json = serde_json::to_string(&point).unwrap();
        assert_eq!(json, "[42.0]");
        let deserialized: Point<f64, 1> = serde_json::from_str(&json).unwrap();
        assert_eq!(point, deserialized);
    }

    #[test]
    fn point_1d_special_values() {
        // Test 1D points with special values

        let point_nan = Point::new([f64::NAN]);
        let point_nan2 = Point::new([f64::NAN]);
        assert_eq!(point_nan, point_nan2);

        let point_inf = Point::new([f64::INFINITY]);
        let point_neg_inf = Point::new([f64::NEG_INFINITY]);
        assert_ne!(point_inf, point_neg_inf);
        assert!(point_neg_inf < point_inf);

        // NaN should be greater than infinity in OrderedFloat semantics
        assert!(point_nan > point_inf);
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
        // Test error message formatting for NonFiniteValue with NaN
        let coords_with_nan = [f64::NAN, 1.0];
        let result: Result<Point<f32, 2>, _> = Point::try_from(coords_with_nan);
        assert!(result.is_err());

        let error = result.unwrap_err();
        let error_msg = format!("{error}");
        assert!(error_msg.contains("Non-finite value"));
        assert!(error_msg.contains("coordinate index 0"));
        assert!(error_msg.contains("NaN"));

        // Test error cloning and equality with infinity
        let coords_with_inf = [f64::INFINITY, 2.0];
        let result2: Result<Point<f32, 2>, _> = Point::try_from(coords_with_inf);
        let error2 = result2.unwrap_err();
        let error2_clone = error2.clone();
        assert_eq!(error2, error2_clone);

        // Test overflow error details (f64::MAX overflows to f32)
        let coords_overflow = [f64::MAX, 1.0];
        let result3: Result<Point<f32, 2>, _> = Point::try_from(coords_overflow);
        match result3 {
            Err(CoordinateConversionError::NonFiniteValue {
                coordinate_index,
                coordinate_value,
            }) => {
                assert_eq!(coordinate_index, 0);
                assert!(!coordinate_value.is_empty());
                assert!(coordinate_value.contains("inf") || coordinate_value.contains("Inf"));
            }
            _ => panic!("Expected NonFiniteValue error for overflow"),
        }
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
    fn point_deserialize_nan_handling() {
        // Test deserialization of null values mapping to NaN

        // Create JSON with null value
        let json_with_null = "[null,1.0,2.0]";

        let result: Result<Point<f64, 3>, _> = serde_json::from_str(json_with_null);

        // Should successfully deserialize with null mapped to NaN
        assert!(result.is_ok());
        let point = result.unwrap();

        // First coordinate should be NaN
        let coords = point.to_array();
        assert!(coords[0].is_nan());
        assert_relative_eq!(coords[1], 1.0);
        assert_relative_eq!(coords[2], 2.0);

        // Test with multiple nulls
        let json_multiple_nulls = "[null,null,3.0]";
        let result_multi: Result<Point<f64, 3>, _> = serde_json::from_str(json_multiple_nulls);
        assert!(result_multi.is_ok());
        let point_multi = result_multi.unwrap();

        let coords_multi = point_multi.to_array();
        assert!(coords_multi[0].is_nan());
        assert!(coords_multi[1].is_nan());
        assert_relative_eq!(coords_multi[2], 3.0);

        // Test with f32
        let json_f32_null = "[null,1.5]";
        let result_f32: Result<Point<f32, 2>, _> = serde_json::from_str(json_f32_null);
        assert!(result_f32.is_ok());
        let point_f32 = result_f32.unwrap();

        let coords_f32 = point_f32.to_array();
        assert!(coords_f32[0].is_nan());
        assert_relative_eq!(coords_f32[1], 1.5);
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
        #[expect(clippy::clone_on_copy)]
        let cloned = point.clone();
        let copied = point;

        // Verify copy worked by using the copied value
        assert_eq!(copied.dim(), cloned.dim());

        // Test that point can be used in collections requiring Hash + Eq
        let mut set = HashSet::new();
        set.insert(point);
        assert!(set.contains(&point));
    }
}
