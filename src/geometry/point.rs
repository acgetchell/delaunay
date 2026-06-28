//! Data and operations on d-dimensional points.
//!
//! Points are validated domain values: every stored coordinate is a finite `f64`.

#![allow(clippy::similar_names)]
#![forbid(unsafe_code)]

use crate::geometry::traits::coordinate::{
    Coordinate, CoordinateConversionError, CoordinateConversionValue, CoordinateValidationError,
    HashCoordinate, InvalidCoordinateValue,
};
use num_traits::cast;
use serde::de::{Error, SeqAccess, Visitor};
use serde::ser::SerializeTuple;
use serde::{Deserialize, Serialize};
use std::any;
use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt;
use std::hash::{Hash, Hasher};

// =============================================================================
// POINT STRUCT DEFINITION
// =============================================================================

#[derive(Clone, Copy, Debug)]
pub(crate) struct ValidatedCoordinates<const D: usize> {
    values: [f64; D],
}

impl<const D: usize> ValidatedCoordinates<D> {
    #[inline]
    pub(crate) fn try_new(mut values: [f64; D]) -> Result<Self, CoordinateValidationError> {
        for (index, coord) in values.iter_mut().enumerate() {
            if !coord.is_finite() {
                return Err(CoordinateValidationError::InvalidCoordinate {
                    coordinate_index: index,
                    coordinate_value: InvalidCoordinateValue::from_debug(coord),
                    dimension: D,
                });
            }
            if *coord == 0.0 {
                *coord = 0.0;
            }
        }
        Ok(Self { values })
    }

    /// Builds validated coordinates from values whose finiteness was already proved.
    #[inline]
    pub(in crate::geometry) fn from_prevalidated_finite_values(mut values: [f64; D]) -> Self {
        for coord in &mut values {
            if *coord == 0.0 {
                *coord = 0.0;
            }
        }
        Self { values }
    }

    #[inline]
    pub(crate) const fn as_array(&self) -> &[f64; D] {
        &self.values
    }

    #[inline]
    pub(crate) const fn into_array(self) -> [f64; D] {
        self.values
    }

    #[inline]
    fn ordered_equals(&self, other: &Self) -> bool {
        self.values
            .iter()
            .zip(other.values.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            let ordering = a.total_cmp(b);
            if ordering != Ordering::Equal {
                return ordering;
            }
        }
        Ordering::Equal
    }

    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &coord in &self.values {
            HashCoordinate::hash_scalar(&coord, state);
        }
    }
}

impl<const D: usize> Default for ValidatedCoordinates<D> {
    fn default() -> Self {
        Self { values: [0.0; D] }
    }
}

#[derive(Clone, Copy, Debug)]
/// The [Point] struct represents a point in a D-dimensional space.
///
/// # Invariants
///
/// `Point<D>` stores a validated coordinate array. Public construction is
/// fallible so non-finite coordinates are rejected before storage, and signed
/// zero is canonicalized so equality, hashing, and ordering agree.
///
/// # Properties
///
/// * `coords`: A private validated coordinate array (length `D` is known at compile time).
///   The field is private to keep points immutable once created.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::Point;
///
/// let p = Point::try_new([1.0, 2.0])?;
/// assert_eq!(p.coords(), &[1.0, 2.0]);
/// # Ok::<(), delaunay::prelude::geometry::CoordinateValidationError>(())
/// ```
pub struct Point<const D: usize> {
    /// The coordinates of the point.
    coords: ValidatedCoordinates<D>,
}

// =============================================================================
// PUBLIC API
// =============================================================================

impl<const D: usize> Point<D> {
    /// Creates a point from raw coordinates after proving every coordinate is finite.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinateValidationError::InvalidCoordinate`] when any coordinate
    /// is `NaN`, positive infinity, or negative infinity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::{CoordinateValidationError, Point};
    ///
    /// let point = Point::try_new([1.0, 2.0])?;
    /// assert_eq!(point.coords(), &[1.0, 2.0]);
    /// # Ok::<(), CoordinateValidationError>(())
    /// ```
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::{CoordinateValidationError, Point};
    ///
    /// std::assert_matches!(
    ///     Point::<2>::try_new([f64::NAN, 0.0]),
    ///     Err(CoordinateValidationError::InvalidCoordinate { coordinate_index: 0, .. })
    /// );
    /// ```
    #[inline]
    pub fn try_new(coords: [f64; D]) -> Result<Self, CoordinateValidationError> {
        Ok(Self::from_validated_coordinates(
            ValidatedCoordinates::try_new(coords)?,
        ))
    }

    /// Creates a point from a proof-bearing coordinate wrapper.
    #[inline]
    #[must_use]
    pub(crate) const fn from_validated_coordinates(coords: ValidatedCoordinates<D>) -> Self {
        Self { coords }
    }

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
    /// use delaunay::prelude::geometry::Point;
    ///
    /// let point = Point::try_new([1.0, 2.0, 3.0])?;
    /// let coords = point.coords();
    /// assert_eq!(coords, &[1.0, 2.0, 3.0]);
    ///
    /// // For owned coordinates, use Into
    /// let owned_coords: [f64; 3] = point.into();
    /// assert_eq!(owned_coords, [1.0, 2.0, 3.0]);
    /// # Ok::<(), delaunay::prelude::geometry::CoordinateValidationError>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn coords(&self) -> &[f64; D] {
        self.validated_coords().as_array()
    }

    /// Returns the invariant-carrying coordinate tuple used by internal code.
    #[inline]
    #[must_use]
    pub(crate) const fn validated_coords(&self) -> &ValidatedCoordinates<D> {
        &self.coords
    }
}

// =============================================================================
// TRAIT IMPLEMENTATIONS
// =============================================================================

impl<const D: usize> Coordinate<D> for Point<D> {
    /// Create a new Point from an array of coordinates.
    #[inline]
    fn try_new(coords: [f64; D]) -> Result<Self, CoordinateValidationError> {
        Self::try_new(coords)
    }

    /// Extract the coordinates as an array
    #[inline]
    fn to_array(&self) -> [f64; D] {
        self.coords.into_array()
    }

    /// Get the coordinate at the specified index
    #[inline]
    fn get(&self, index: usize) -> Option<f64> {
        self.coords.as_array().get(index).copied()
    }

    /// Validates the point's stored coordinate invariant.
    #[inline]
    fn validate(&self) -> Result<(), CoordinateValidationError> {
        Ok(())
    }

    /// Hash the coordinate values
    fn hash_coordinate<H: Hasher>(&self, state: &mut H) {
        self.coords.hash(state);
    }

    /// Check if two coordinates are equal using `OrderedEq`
    fn ordered_equals(&self, other: &Self) -> bool {
        self.coords.ordered_equals(&other.coords)
    }
}

// =============================================================================
// STANDARD TRAIT IMPLEMENTATIONS
// =============================================================================

// Implement Hash using the Coordinate trait
impl<const D: usize> Hash for Point<D> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_coordinate(state);
    }
}

// Implement PartialEq using the Coordinate trait
impl<const D: usize> PartialEq for Point<D> {
    fn eq(&self, other: &Self) -> bool {
        self.ordered_equals(other)
    }
}

// Implement Eq using the Coordinate trait
impl<const D: usize> Eq for Point<D> {}

// Implement PartialOrd using OrderedCmp for consistent ordering with special floating-point values
impl<const D: usize> PartialOrd for Point<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.coords.cmp(&other.coords))
    }
}

// Manual implementations for traits that can't be derived due to [T; D] limitations

// Implement Default manually
impl<const D: usize> Default for Point<D> {
    fn default() -> Self {
        Self {
            coords: ValidatedCoordinates::default(),
        }
    }
}

// Implement Serialize manually
impl<const D: usize> Serialize for Point<D> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut tuple = serializer.serialize_tuple(D)?;
        for coord in self.coords.as_array() {
            tuple.serialize_element(coord)?;
        }
        tuple.end()
    }
}

/// Format-agnostic representation for coordinate values during deserialization.
/// This enum allows the deserializer to work with any format (JSON, CBOR, bincode, etc.)
/// without being tied to specific format types.
#[derive(Deserialize)]
#[serde(untagged)]
enum CoordRepr {
    /// Regular numeric value
    Num(f64),
    /// String representation, rejected because `Point` stores only finite coordinates.
    Str(String),
    /// Null value, rejected because `Point` stores only finite coordinates.
    Null,
}

// Implement Deserialize manually so raw serialized coordinates parse through the validated boundary.
impl<'de, const D: usize> Deserialize<'de> for Point<D> {
    fn deserialize<DE>(deserializer: DE) -> Result<Self, DE::Error>
    where
        DE: serde::Deserializer<'de>,
    {
        struct ArrayVisitor<const D: usize>;

        impl<'de, const D: usize> Visitor<'de> for ArrayVisitor<D> {
            type Value = Point<D>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_fmt(format_args!("an array of {D} finite numeric coordinates"))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                // Collect coordinates into a Vec first, then convert to array
                let mut coords = Vec::with_capacity(D);
                for i in 0..D {
                    // Deserialize each element using the format-agnostic enum
                    let element: CoordRepr = seq
                        .next_element()?
                        .ok_or_else(|| Error::invalid_length(i, &self))?;

                    let coord = match element {
                        CoordRepr::Num(value) => value,
                        CoordRepr::Str(s) => {
                            return Err(Error::custom(format!(
                                "non-finite coordinate string is not valid for Point: {s}"
                            )));
                        }
                        CoordRepr::Null => {
                            return Err(Error::custom("null is not a valid Point coordinate"));
                        }
                    };

                    coords.push(coord);
                }

                // Convert Vec to array
                let coords_len = coords.len();
                let coords_array: [f64; D] = coords
                    .try_into()
                    .map_err(|_| Error::invalid_length(coords_len, &self))?;

                Point::try_new(coords_array).map_err(Error::custom)
            }
        }

        deserializer.deserialize_tuple(D, ArrayVisitor::<D>)
    }
}

// =============================================================================
// TYPE CONVERSION IMPLEMENTATIONS
// =============================================================================

/// Fallible conversions for [Point] from arrays into finite `f64` coordinates.
///
/// This replaces the previous infallible From<[T; D]> which silently defaulted on
/// cast failures. Now, conversions will return an error if any coordinate cannot be
/// cast into the target type, or if a non-finite value is encountered post-cast.
/// Signed zero is canonicalized during conversion: `-0.0` is stored as `0.0`.
impl<T, const D: usize> TryFrom<[T; D]> for Point<D>
where
    T: cast::NumCast + Copy + fmt::Debug + PartialEq,
{
    type Error = CoordinateConversionError;

    #[inline]
    fn try_from(coords: [T; D]) -> Result<Self, Self::Error> {
        let mut out = [0.0; D];
        for (i, c) in coords.into_iter().enumerate() {
            let coordinate_value = CoordinateConversionValue::from_numeric_debug(&c);
            // Attempt numeric cast
            let v: f64 =
                cast::cast(c).ok_or_else(|| CoordinateConversionError::ConversionFailed {
                    coordinate_index: i,
                    coordinate_value: coordinate_value.clone(),
                    from_type: any::type_name::<T>(),
                    to_type: any::type_name::<f64>(),
                })?;
            // Validate finiteness after cast
            if !v.is_finite() {
                return Err(CoordinateConversionError::NonFiniteValue {
                    coordinate_index: i,
                    coordinate_value: InvalidCoordinateValue::from_debug(&v),
                });
            }
            let round_trip: T =
                cast::cast(v).ok_or_else(|| CoordinateConversionError::ConversionFailed {
                    coordinate_index: i,
                    coordinate_value: coordinate_value.clone(),
                    from_type: any::type_name::<f64>(),
                    to_type: any::type_name::<T>(),
                })?;
            if round_trip != c {
                return Err(CoordinateConversionError::ConversionFailed {
                    coordinate_index: i,
                    coordinate_value,
                    from_type: any::type_name::<T>(),
                    to_type: any::type_name::<f64>(),
                });
            }
            out[i] = if v == 0.0 { 0.0 } else { v };
        }
        Ok(Self::from_validated_coordinates(ValidatedCoordinates {
            values: out,
        }))
    }
}

/// Enable conversions from Point to coordinate arrays - using Coordinate trait
impl<const D: usize> From<Point<D>> for [f64; D] {
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::Point;
    /// let point = Point::try_new([1.0, 2.0])?;
    /// let coords: [f64; 2] = point.into();
    /// assert_eq!(coords, [1.0, 2.0]);
    /// # Ok::<(), delaunay::prelude::geometry::CoordinateValidationError>(())
    /// ```
    #[inline]
    fn from(point: Point<D>) -> [f64; D] {
        point.coords.into_array()
    }
}

impl<const D: usize> From<&Point<D>> for [f64; D] {
    /// # Example
    ///
    /// ```rust
    /// use delaunay::prelude::geometry::Point;
    /// let point = Point::try_new([3.0, 4.0])?;
    /// let coords: [f64; 2] = (&point).into();
    /// assert_eq!(coords, [3.0, 4.0]);
    /// # Ok::<(), delaunay::prelude::geometry::CoordinateValidationError>(())
    /// ```
    #[inline]
    fn from(point: &Point<D>) -> [f64; D] {
        point.coords.into_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::assert_matches;
    use std::cmp::Ordering;
    use std::collections::hash_map::DefaultHasher;
    use std::collections::{HashMap, HashSet};
    use std::hash::{Hash, Hasher};
    use std::mem;

    // Helper function to hash any hashable value with the default hasher.
    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    // Helper function to test point equality and hash consistency
    fn test_point_equality_and_hash<const D: usize>(
        point1: Point<D>,
        point2: Point<D>,
        should_be_equal: bool,
    ) where
        Point<D>: Hash,
    {
        if should_be_equal {
            assert_eq!(point1, point2);
            assert_eq!(hash_of(&point1), hash_of(&point2));
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
                let point_2d = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
                assert_relative_eq!(point_2d.to_array().as_slice(), [1.0, 2.0].as_slice());
                assert_eq!(point_2d.dim(), 2);

                // 3D
                let point_3d = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
                assert_relative_eq!(point_3d.to_array().as_slice(), [1.0, 2.0, 3.0].as_slice());
                assert_eq!(point_3d.dim(), 3);

                // 4D
                let point_4d =
                    Point::try_new([1.0, 2.0, 3.0, 4.0]).expect("finite point coordinates");
                assert_relative_eq!(
                    point_4d.to_array().as_slice(),
                    [1.0, 2.0, 3.0, 4.0].as_slice()
                );
                assert_eq!(point_4d.dim(), 4);

                // 5D
                let point_5d =
                    Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
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
                let p2d_a = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
                let p2d_b = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
                let p2d_c = Point::try_new([1.0, 3.0]).expect("finite point coordinates");
                assert_eq!(p2d_a, p2d_b);
                assert_ne!(p2d_a, p2d_c);

                // 3D
                let p3d_a = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
                let p3d_b = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
                let p3d_c = Point::try_new([1.0, 2.0, 4.0]).expect("finite point coordinates");
                assert_eq!(p3d_a, p3d_b);
                assert_ne!(p3d_a, p3d_c);

                // 4D
                let p4d_a = Point::try_new([1.0, 2.0, 3.0, 4.0]).expect("finite point coordinates");
                let p4d_b = Point::try_new([1.0, 2.0, 3.0, 4.0]).expect("finite point coordinates");
                let p4d_c = Point::try_new([1.0, 2.0, 3.0, 5.0]).expect("finite point coordinates");
                assert_eq!(p4d_a, p4d_b);
                assert_ne!(p4d_a, p4d_c);

                // 5D
                let p5d_a =
                    Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
                let p5d_b =
                    Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
                let p5d_c =
                    Point::try_new([1.0, 2.0, 3.0, 4.0, 6.0]).expect("finite point coordinates");
                assert_eq!(p5d_a, p5d_b);
                assert_ne!(p5d_a, p5d_c);
            }
        };

        // Test point hashing across dimensions
        (hashing: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D
                let p2d_a = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
                let p2d_b = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
                assert_eq!(hash_of(&p2d_a), hash_of(&p2d_b));

                // 3D
                let p3d_a = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
                let p3d_b = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
                assert_eq!(hash_of(&p3d_a), hash_of(&p3d_b));

                // 4D
                let p4d_a = Point::try_new([1.0, 2.0, 3.0, 4.0]).expect("finite point coordinates");
                let p4d_b = Point::try_new([1.0, 2.0, 3.0, 4.0]).expect("finite point coordinates");
                assert_eq!(hash_of(&p4d_a), hash_of(&p4d_b));

                // 5D
                let p5d_a =
                    Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
                let p5d_b =
                    Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
                assert_eq!(hash_of(&p5d_a), hash_of(&p5d_b));
            }
        };

        // Test point ordering across dimensions
        (ordering: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D - lexicographic ordering
                let p2d_a = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
                let p2d_b = Point::try_new([1.0, 3.0]).expect("finite point coordinates");
                assert!(p2d_a < p2d_b);
                assert!(p2d_b > p2d_a);

                // 3D
                let p3d_a = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
                let p3d_b = Point::try_new([1.0, 2.0, 4.0]).expect("finite point coordinates");
                assert!(p3d_a < p3d_b);
                assert!(p3d_b > p3d_a);

                // 4D
                let p4d_a = Point::try_new([1.0, 2.0, 3.0, 4.0]).expect("finite point coordinates");
                let p4d_b = Point::try_new([1.0, 2.0, 3.0, 5.0]).expect("finite point coordinates");
                assert!(p4d_a < p4d_b);
                assert!(p4d_b > p4d_a);

                // 5D
                let p5d_a =
                    Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
                let p5d_b =
                    Point::try_new([1.0, 2.0, 3.0, 4.0, 6.0]).expect("finite point coordinates");
                assert!(p5d_a < p5d_b);
                assert!(p5d_b > p5d_a);
            }
        };

        // Test point validation across dimensions
        (validation: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D - valid and invalid
                let valid_2d = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
                assert!(valid_2d.validate().is_ok());
                assert!(Point::<2>::try_new([f64::NAN, 2.0]).is_err());

                // 3D
                let valid_3d = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
                assert!(valid_3d.validate().is_ok());
                assert!(Point::<3>::try_new([1.0, f64::INFINITY, 3.0]).is_err());

                // 4D
                let valid_4d =
                    Point::try_new([1.0, 2.0, 3.0, 4.0]).expect("finite point coordinates");
                assert!(valid_4d.validate().is_ok());
                assert!(Point::<4>::try_new([1.0, 2.0, f64::NEG_INFINITY, 4.0]).is_err());

                // 5D
                let valid_5d =
                    Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
                assert!(valid_5d.validate().is_ok());
                assert!(Point::<5>::try_new([1.0, 2.0, 3.0, f64::NAN, 5.0]).is_err());
            }
        };

        // Test point serialization across dimensions
        (serialization: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D
                let p2d = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
                let json2d = serde_json::to_string(&p2d).unwrap();
                let de2d: Point<2> = serde_json::from_str(&json2d).unwrap();
                assert_eq!(p2d, de2d);

                // 3D
                let p3d = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
                let json3d = serde_json::to_string(&p3d).unwrap();
                let de3d: Point<3> = serde_json::from_str(&json3d).unwrap();
                assert_eq!(p3d, de3d);

                // 4D
                let p4d = Point::try_new([1.0, 2.0, 3.0, 4.0]).expect("finite point coordinates");
                let json4d = serde_json::to_string(&p4d).unwrap();
                let de4d: Point<4> = serde_json::from_str(&json4d).unwrap();
                assert_eq!(p4d, de4d);

                // 5D
                let p5d =
                    Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
                let json5d = serde_json::to_string(&p5d).unwrap();
                let de5d: Point<5> = serde_json::from_str(&json5d).unwrap();
                assert_eq!(p5d, de5d);
            }
        };

        // Test point origin across dimensions
        (origin: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D
                let origin_2d: Point<2> = Point::origin();
                assert_relative_eq!(origin_2d.to_array().as_slice(), [0.0, 0.0].as_slice());

                // 3D
                let origin_3d: Point<3> = Point::origin();
                assert_relative_eq!(origin_3d.to_array().as_slice(), [0.0, 0.0, 0.0].as_slice());

                // 4D
                let origin_4d: Point<4> = Point::origin();
                assert_relative_eq!(
                    origin_4d.to_array().as_slice(),
                    [0.0, 0.0, 0.0, 0.0].as_slice()
                );

                // 5D
                let origin_5d: Point<5> = Point::origin();
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
                let mut map2d: HashMap<Point<2>, i32> = HashMap::new();
                let p2d = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
                map2d.insert(p2d, 42);
                assert_eq!(
                    map2d.get(&Point::try_new([1.0, 2.0]).expect("finite point coordinates")),
                    Some(&42)
                );

                // 3D
                let mut map3d: HashMap<Point<3>, i32> = HashMap::new();
                let p3d = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
                map3d.insert(p3d, 42);
                assert_eq!(
                    map3d.get(&Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates")),
                    Some(&42)
                );

                // 4D
                let mut map4d: HashMap<Point<4>, i32> = HashMap::new();
                let p4d = Point::try_new([1.0, 2.0, 3.0, 4.0]).expect("finite point coordinates");
                map4d.insert(p4d, 42);
                assert_eq!(
                    map4d.get(
                        &Point::try_new([1.0, 2.0, 3.0, 4.0]).expect("finite point coordinates")
                    ),
                    Some(&42)
                );

                // 5D
                let mut map5d: HashMap<Point<5>, i32> = HashMap::new();
                let p5d =
                    Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
                map5d.insert(p5d, 42);
                assert_eq!(
                    map5d.get(
                        &Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0])
                            .expect("finite point coordinates")
                    ),
                    Some(&42)
                );
            }
        };

        // Test Copy semantics across dimensions
        (copy: $test_name:ident) => {
            #[test]
            fn $test_name() {
                // 2D
                let p2d_original = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
                let p2d_copy = p2d_original;
                assert_eq!(p2d_original, p2d_copy);
                assert_relative_eq!(
                    p2d_original.to_array().as_slice(),
                    p2d_copy.to_array().as_slice()
                );

                // 3D
                let p3d_original =
                    Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
                let p3d_copy = p3d_original;
                assert_eq!(p3d_original, p3d_copy);

                // 4D
                let p4d_original =
                    Point::try_new([1.0, 2.0, 3.0, 4.0]).expect("finite point coordinates");
                let p4d_copy = p4d_original;
                assert_eq!(p4d_original, p4d_copy);

                // 5D
                let p5d_original =
                    Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
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
        let point: Point<4> = Point::default();

        let coords = point.to_array();
        assert_relative_eq!(
            coords.as_slice(),
            [0.0, 0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn point_coords() {
        // Test coords() method provides read-only access
        let point = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
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
        let point_2d = Point::try_new([5.5, -2.5]).expect("finite point coordinates");
        assert_relative_eq!(
            point_2d.coords().as_slice(),
            [5.5, -2.5].as_slice(),
            epsilon = 1e-9
        );

        let point_4d = Point::try_new([1.0, 2.0, 3.0, 4.0]).expect("finite point coordinates");
        assert_relative_eq!(
            point_4d.coords().as_slice(),
            [1.0, 2.0, 3.0, 4.0].as_slice(),
            epsilon = 1e-9
        );

        // Test with 5D
        let point_5d = Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
        assert_relative_eq!(
            point_5d.coords().as_slice(),
            [1.0, 2.0, 3.0, 4.0, 5.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(point_5d.coords().len(), 5);
    }

    #[test]
    fn point_from_integer_array_to_f64() {
        let coords = [1, 2, 3, 4];
        let point: Point<4> =
            Point::try_new(coords.map(Into::into)).expect("finite point coordinates");

        let result_coords = point.to_array();
        assert_relative_eq!(
            result_coords.as_slice(),
            [1.0, 2.0, 3.0, 4.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(point.dim(), 4);
    }

    #[test]
    fn point_type_conversions() {
        // Test same-type conversion.
        let coords_f64 = [1.5, 2.5];
        let point_f64: Point<2> = Point::try_new(coords_f64).expect("finite point coordinates");
        let result_f64 = point_f64.to_array();
        assert_relative_eq!(result_f64.as_slice(), [1.5, 2.5].as_slice(), epsilon = 1e-9);
    }

    // =============================================================================
    // HASH AND EQUALITY TESTS
    // =============================================================================

    #[test]
    fn point_debug_format() {
        let point = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        let debug_str = format!("{point:?}");

        assert!(debug_str.contains("Point"));
        assert!(debug_str.contains("coords"));
        assert!(debug_str.contains("1.0"));
        assert!(debug_str.contains("2.0"));
        assert!(debug_str.contains("3.0"));
    }

    #[test]
    fn point_eq_trait() {
        let point1 = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        let point2 = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        let point3 = Point::try_new([1.0, 2.0, 4.0]).expect("finite point coordinates");

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
        let point_3d = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        let serialized_3d = serde_json::to_string(&point_3d).unwrap();
        let deserialized_3d: Point<3> = serde_json::from_str(&serialized_3d).unwrap();
        assert_eq!(point_3d, deserialized_3d);

        let point_2d = Point::try_new([10.5, -5.3]).expect("finite point coordinates");
        let serialized_2d = serde_json::to_string(&point_2d).unwrap();
        let deserialized_2d: Point<2> = serde_json::from_str(&serialized_2d).unwrap();
        assert_eq!(point_2d, deserialized_2d);

        let point_1d = Point::try_new([42.0]).expect("finite point coordinates");
        let serialized_1d = serde_json::to_string(&point_1d).unwrap();
        let deserialized_1d: Point<1> = serde_json::from_str(&serialized_1d).unwrap();
        assert_eq!(point_1d, deserialized_1d);

        // Test with very large and small numbers (roundtrip)
        let point_large = Point::try_new([1e100, -1e100, 0.0]).expect("finite point coordinates");
        let serialized_large = serde_json::to_string(&point_large).unwrap();
        let deserialized_large: Point<3> = serde_json::from_str(&serialized_large).unwrap();
        assert_eq!(point_large, deserialized_large);

        let point_small = Point::try_new([1e-100, -1e-100, 0.0]).expect("finite point coordinates");
        let serialized_small = serde_json::to_string(&point_small).unwrap();
        let deserialized_small: Point<3> = serde_json::from_str(&serialized_small).unwrap();
        assert_eq!(point_small, deserialized_small);
    }

    #[test]
    fn point_negative_coordinates() {
        let point = Point::try_new([-1.0, -2.0, -3.0]).expect("finite point coordinates");

        assert_relative_eq!(
            point.to_array().as_slice(),
            [-1.0, -2.0, -3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(point.dim(), 3);

        // Test with mixed positive/negative
        let mixed_point = Point::try_new([1.0, -2.0, 3.0, -4.0]).expect("finite point coordinates");
        assert_relative_eq!(
            mixed_point.to_array().as_slice(),
            [1.0, -2.0, 3.0, -4.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn point_zero_coordinates() {
        let zero_point = Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates");
        let origin: Point<3> = Point::origin();

        assert_eq!(zero_point, origin);
        assert_relative_eq!(
            zero_point.to_array().as_slice(),
            [0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn point_large_coordinates() {
        let large_point = Point::try_new([1e6, 2e6, 3e6]).expect("finite point coordinates");

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
        let small_point = Point::try_new([1e-6, 2e-6, 3e-6]).expect("finite point coordinates");

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
        let point1 = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
        let point2 = Point::try_new([1.0, 2.0]).expect("finite point coordinates");

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
        let point_f64_1 = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
        let point_f64_2 = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
        let point_f64_3 = Point::try_new([1.0, 2.1]).expect("finite point coordinates");

        assert_eq!(point_f64_1, point_f64_2);
        assert_ne!(point_f64_1, point_f64_3);
    }

    #[test]
    fn point_hash_consistency_floating_point() {
        // Test that OrderedFloat provides consistent hashing for NaN-free floats
        let point1 = Point::try_new([1.0, 2.0, 3.5]).expect("finite point coordinates");
        let point2 = Point::try_new([1.0, 2.0, 3.5]).expect("finite point coordinates");
        test_point_equality_and_hash(point1, point2, true);
    }

    #[test]
    fn point_implicit_conversion_to_coordinates() {
        let point: Point<3> = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");

        // Test implicit conversion from owned point
        let coords_owned: [f64; 3] = point.into();
        assert_relative_eq!(coords_owned.as_slice(), [1.0, 2.0, 3.0].as_slice());

        // Create a new point for reference test
        let point_ref: Point<3> =
            Point::try_new([4.0, 5.0, 6.0]).expect("finite point coordinates");

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
        let valid_point = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        assert!(valid_point.validate().is_ok());

        let valid_negative = Point::try_new([-1.0, -2.0, -3.0]).expect("finite point coordinates");
        assert!(valid_negative.validate().is_ok());

        let valid_zero = Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates");
        assert!(valid_zero.validate().is_ok());

        let valid_mixed = Point::try_new([1.0, -2.5, 0.0, 42.7]).expect("finite point coordinates");
        assert!(valid_mixed.validate().is_ok());

        // Test invalid f64 input with NaN
        assert!(Point::<3>::try_new([1.0, f64::NAN, 3.0]).is_err());
        assert!(Point::<3>::try_new([f64::NAN, f64::NAN, f64::NAN]).is_err());
        assert!(Point::<3>::try_new([f64::NAN, 2.0, 3.0]).is_err());
        assert!(Point::<3>::try_new([1.0, 2.0, f64::NAN]).is_err());

        // Test invalid f64 input with infinity
        assert!(Point::<3>::try_new([1.0, f64::INFINITY, 3.0]).is_err());
        assert!(Point::<3>::try_new([1.0, f64::NEG_INFINITY, 3.0]).is_err());
        assert!(Point::<2>::try_new([f64::INFINITY, f64::NEG_INFINITY]).is_err());

        // Test mixed invalid cases
        assert!(Point::<3>::try_new([f64::NAN, f64::INFINITY, 1.0]).is_err());
    }

    #[test]
    fn point_is_valid_different_dimensions() {
        // Test 1D points
        let valid_1d_f64 = Point::try_new([42.0]).expect("finite point coordinates");
        assert!(valid_1d_f64.validate().is_ok());

        assert!(Point::<1>::try_new([f64::NAN]).is_err());

        // Test 2D points
        let valid_2d = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
        assert!(valid_2d.validate().is_ok());

        assert!(Point::<2>::try_new([1.0, f64::INFINITY]).is_err());

        // Test higher dimensional points
        let valid_5d = Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
        assert!(valid_5d.validate().is_ok());

        assert!(Point::<5>::try_new([1.0, 2.0, f64::NAN, 4.0, 5.0]).is_err());

        // Test 10D point
        let valid_10d = Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
            .expect("finite point coordinates");
        assert!(valid_10d.validate().is_ok());

        let invalid_10d = [
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
        ];
        assert!(Point::<10>::try_new(invalid_10d).is_err());
    }

    #[test]
    fn point_is_valid_edge_cases() {
        // Test with very small finite values
        let tiny_valid = Point::try_new([f64::MIN_POSITIVE, -f64::MIN_POSITIVE, 0.0])
            .expect("finite point coordinates");
        assert!(tiny_valid.validate().is_ok());

        // Test with very large finite values
        let large_valid = Point::try_new([f64::MAX, -f64::MAX]).expect("finite point coordinates");
        assert!(large_valid.validate().is_ok());

        // Test subnormal numbers (should be valid)
        let subnormal = f64::MIN_POSITIVE / 2.0;
        let subnormal_point =
            Point::try_new([subnormal, -subnormal]).expect("finite point coordinates");
        assert!(subnormal_point.validate().is_ok());

        // Test zero and negative zero
        let zero_point = Point::try_new([0.0, -0.0]).expect("finite point coordinates");
        assert!(zero_point.validate().is_ok());

        // Mixed valid and invalid in same point should be invalid
        assert!(Point::<5>::try_new([1.0, 2.0, 3.0, f64::NAN, 5.0]).is_err());

        // All coordinates must be valid for point to be valid
        let one_invalid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, f64::INFINITY];
        assert!(Point::<8>::try_new(one_invalid).is_err());
    }

    #[test]
    fn point_rejects_non_finite_bit_patterns() {
        let nan1 = f64::NAN;
        let nan2 = f64::from_bits(0x7ff8_0000_0000_0001);
        let nan3 = f64::from_bits(0x7ff8_0000_0000_0002);

        // Verify they are all NaN
        assert!(nan1.is_nan());
        assert!(nan2.is_nan());
        assert!(nan3.is_nan());
        assert!(Point::<2>::try_new([nan1, 1.0]).is_err());
        assert!(Point::<2>::try_new([nan2, 1.0]).is_err());
        assert!(Point::<2>::try_new([nan3, 1.0]).is_err());
    }

    #[test]
    fn point_zero_comparison() {
        let point_pos_zero = Point::try_new([0.0, 0.0]).expect("finite point coordinates");
        let point_neg_zero = Point::try_new([-0.0, -0.0]).expect("finite point coordinates");
        let point_mixed_zero = Point::try_new([0.0, -0.0]).expect("finite point coordinates");

        assert_eq!(point_pos_zero, point_neg_zero);
        assert_eq!(point_pos_zero, point_mixed_zero);
        assert_eq!(point_neg_zero, point_mixed_zero);
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        reason = "test inputs intentionally cover large integer-to-float conversions"
    )]
    fn point_extreme_dimensions() {
        // Test with high dimensional points (limited by serde trait implementations)

        // Test 20D point
        let coords_20d = [1.0; 20];
        let point_20d = Point::try_new(coords_20d).expect("finite point coordinates");
        assert_eq!(point_20d.dim(), 20);
        assert_relative_eq!(point_20d.to_array().as_slice(), coords_20d.as_slice());
        assert!(point_20d.validate().is_ok());

        // Test 25D point
        let coords_25d = [2.5; 25];
        let point_25d = Point::try_new(coords_25d).expect("finite point coordinates");
        assert_eq!(point_25d.dim(), 25);
        assert_relative_eq!(point_25d.to_array().as_slice(), coords_25d.as_slice());
        assert!(point_25d.validate().is_ok());

        // Test 32D point with mixed values (max supported by std traits)
        let mut coords_32d = [0.0; 32];
        for (i, coord) in coords_32d.iter_mut().enumerate() {
            *coord = i as f64;
        }
        let point_32d = Point::try_new(coords_32d).expect("finite point coordinates");
        assert_eq!(point_32d.dim(), 32);
        assert_relative_eq!(point_32d.to_array().as_slice(), coords_32d.as_slice());
        assert!(point_32d.validate().is_ok());

        // Test high dimensional point with NaN
        let mut coords_with_nan = [1.0; 25];
        coords_with_nan[12] = f64::NAN;
        assert!(Point::<25>::try_new(coords_with_nan).is_err());

        // Test equality for high dimensional points
        let point_20d_copy = Point::try_new([1.0; 20]).expect("finite point coordinates");
        assert_eq!(point_20d, point_20d_copy);

        // Test with 30D points
        let coords_30d_a = [std::f64::consts::PI; 30];
        let coords_30d_b = [std::f64::consts::PI; 30];
        let point_30d_a = Point::try_new(coords_30d_a).expect("finite point coordinates");
        let point_30d_b = Point::try_new(coords_30d_b).expect("finite point coordinates");
        assert_eq!(point_30d_a, point_30d_b);
        assert!(point_30d_a.validate().is_ok());
    }

    #[test]
    fn point_boundary_numeric_values() {
        // Test with extreme numeric values

        // Test with very large f64 values
        let large_point =
            Point::try_new([f64::MAX, f64::MAX / 2.0, 1e308]).expect("finite point coordinates");
        assert!(large_point.validate().is_ok());
        assert_relative_eq!(large_point.to_array()[0], f64::MAX);

        // Test with very small f64 values
        let small_point = Point::try_new([f64::MIN, f64::MIN_POSITIVE, 1e-308])
            .expect("finite point coordinates");
        assert!(small_point.validate().is_ok());

        // Test with subnormal numbers
        let subnormal = f64::MIN_POSITIVE / 2.0;
        let subnormal_point =
            Point::try_new([subnormal, -subnormal, 0.0]).expect("finite point coordinates");
        assert!(subnormal_point.validate().is_ok());
    }

    #[test]
    fn point_clone_and_copy_semantics() {
        // Test that Point correctly implements Clone and Copy

        let original = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");

        // Test explicit cloning
        #[expect(
            clippy::clone_on_copy,
            reason = "test asserts explicit clone behavior for copy coordinates"
        )]
        let cloned = original.clone();
        assert_relative_eq!(original.to_array().as_slice(), cloned.to_array().as_slice());

        // Test copy semantics (should work implicitly)
        let copied = original; // This should copy, not move
        assert_eq!(original, copied);

        // Original should still be accessible after copy
        assert_eq!(original.dim(), 3);
        assert_eq!(copied.dim(), 3);
    }

    #[test]
    fn point_partial_ord_comprehensive() {
        // Test lexicographic ordering in detail
        let point_a = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        let point_b = Point::try_new([1.0, 2.0, 4.0]).expect("finite point coordinates"); // Greater in last coordinate
        let point_c = Point::try_new([1.0, 3.0, 0.0]).expect("finite point coordinates"); // Greater in second coordinate
        let point_d = Point::try_new([2.0, 0.0, 0.0]).expect("finite point coordinates"); // Greater in first coordinate

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
        let neg_point_a = Point::try_new([-1.0, -2.0]).expect("finite point coordinates");
        let neg_point_b = Point::try_new([-1.0, -1.0]).expect("finite point coordinates");
        assert!(neg_point_a < neg_point_b); // -2.0 < -1.0

        // Test with mixed positive/negative
        let mixed_a = Point::try_new([-1.0, 2.0]).expect("finite point coordinates");
        let mixed_b = Point::try_new([1.0, -2.0]).expect("finite point coordinates");
        assert!(mixed_a < mixed_b); // -1.0 < 1.0

        // Test with zeros
        let zero_a = Point::try_new([0.0, 0.0]).expect("finite point coordinates");
        let zero_b = Point::try_new([0.0, 0.0]).expect("finite point coordinates");
        assert_eq!(zero_a.partial_cmp(&zero_b), Some(Ordering::Equal));

        assert!(Point::<1>::try_new([f64::INFINITY]).is_err());
        assert!(Point::<1>::try_new([f64::NEG_INFINITY]).is_err());
        assert!(Point::<1>::try_new([f64::NAN]).is_err());
    }

    #[test]
    fn point_signed_zero_eq_hash_and_order_are_consistent() {
        let positive_zero = Point::try_new([0.0, -0.0, 1.0]).unwrap();
        let negative_zero = Point::try_new([-0.0, 0.0, 1.0]).unwrap();

        assert_eq!(
            positive_zero.coords().map(f64::to_bits),
            [0.0_f64.to_bits(), 0.0_f64.to_bits(), 1.0_f64.to_bits()]
        );
        assert_eq!(
            negative_zero.coords().map(f64::to_bits),
            [0.0_f64.to_bits(), 0.0_f64.to_bits(), 1.0_f64.to_bits()]
        );
        assert_eq!(positive_zero, negative_zero);
        assert_eq!(hash_of(&positive_zero), hash_of(&negative_zero));
        assert_eq!(
            positive_zero.partial_cmp(&negative_zero),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn point_try_from_rejects_lossy_integer_coordinates() {
        let err = Point::<1>::try_from([9_007_199_254_740_993_u64])
            .expect_err("integer coordinates that cannot round-trip through f64 must fail");

        assert_matches!(
            err,
            CoordinateConversionError::ConversionFailed {
                coordinate_index: 0,
                ..
            }
        );
    }

    #[test]
    fn point_memory_layout_and_size() {
        // Test that Point has the expected memory layout
        // Point should be the same size as its coordinate array

        assert_eq!(mem::size_of::<Point<3>>(), mem::size_of::<[f64; 3]>());

        // Test alignment
        assert_eq!(mem::align_of::<Point<3>>(), mem::align_of::<[f64; 3]>());

        // Test with different dimensions
        assert_eq!(mem::size_of::<Point<1>>(), 8); // 1 * 8 bytes
        assert_eq!(mem::size_of::<Point<2>>(), 16); // 2 * 8 bytes
        assert_eq!(mem::size_of::<Point<10>>(), 80); // 10 * 8 bytes
    }

    #[test]
    fn point_zero_dimensional() {
        // Test 0-dimensional points (edge case)
        let point_0d: Point<0> = Point::try_new([]).expect("finite point coordinates");
        assert_eq!(point_0d.dim(), 0);
        assert_relative_eq!(point_0d.to_array().as_slice(), ([] as [f64; 0]).as_slice());
        assert!(point_0d.validate().is_ok());

        // Test equality for 0D points
        let point_0d_2: Point<0> = Point::try_new([]).expect("finite point coordinates");
        assert_eq!(point_0d, point_0d_2);

        // Test hashing for 0D points
        let hash_0d = hash_of(&point_0d);
        let hash_0d_2 = hash_of(&point_0d_2);
        assert_eq!(hash_0d, hash_0d_2);

        // Test origin for 0D
        let origin_0d: Point<0> = Point::origin();
        assert_eq!(origin_0d, point_0d);
    }

    #[test]
    fn point_rejects_nan_infinity_public_construction() {
        assert!(Point::<3>::try_new([f64::NAN, 1.0, 2.0]).is_err());
        assert!(Point::<3>::try_new([f64::NAN, f64::NAN, 1.0]).is_err());
        assert!(Point::<2>::try_new([f64::NAN, f64::NAN]).is_err());
        assert!(Point::<2>::try_new([f64::INFINITY, 1.0]).is_err());
        assert!(Point::<2>::try_new([1.0, f64::NEG_INFINITY]).is_err());
        assert!(Point::<4>::try_new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0,]).is_err());
    }

    #[test]
    fn point_deserialize_rejects_null() {
        let json = "[null,1.0,2.0]";
        let result: Result<Point<3>, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn point_deserialize_format_agnostic_comprehensive() {
        // Test the format-agnostic deserialization improvements with CoordRepr enum

        // Test 1: Regular numeric values (NumCast improvement)
        let json_regular = "[1.0, 2.5, 4.25]";
        let point_regular: Point<3> = serde_json::from_str(json_regular).unwrap();
        assert_relative_eq!(
            point_regular.to_array().as_slice(),
            [1.0, 2.5, 4.25].as_slice()
        );

        // Test 2: Mixed non-finite values are rejected at the parse boundary
        let json_special = "[1.0, null, \"Infinity\", \"-Infinity\"]";
        assert!(serde_json::from_str::<Point<4>>(json_special).is_err());

        // Test 3: All null values
        let json_all_null = "[null, null, null]";
        assert!(serde_json::from_str::<Point<3>>(json_all_null).is_err());

        // Test 4: All special string values
        let json_all_special = "[\"Infinity\", \"-Infinity\", \"Infinity\"]";
        assert!(serde_json::from_str::<Point<3>>(json_all_special).is_err());

        // Test 5: Serialization roundtrip with finite values
        let original =
            Point::try_new([1.5, -2.25, 3.75, 0.0, -0.0]).expect("finite point coordinates");
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: Point<5> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(original, deserialized);

        // Test 6: Invalid special string should fail gracefully
        let json_invalid = "[1.0, \"NotASpecialValue\", 2.0]";
        let result: Result<Point<3>, _> = serde_json::from_str(json_invalid);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("non-finite coordinate string"));
    }

    #[test]
    fn point_deserialize_rejects_case_insensitive_special_values() {
        let test_cases = vec![
            (r#"["infinity", 1.0]"#, "lowercase infinity"),
            (r#"["INFINITY", 1.0]"#, "uppercase infinity"),
            (r#"["Infinity", 1.0]"#, "mixed case infinity"),
            (r#"["inf", 1.0]"#, "lowercase inf"),
            (r#"["INF", 1.0]"#, "uppercase inf"),
            (r#"["Inf", 1.0]"#, "mixed case inf"),
        ];

        for (json_str, description) in test_cases {
            assert!(
                serde_json::from_str::<Point<2>>(json_str).is_err(),
                "special value should be rejected for {description}"
            );
        }

        let neg_inf_cases = vec![
            (r#"["-infinity", 2.0]"#, "lowercase -infinity"),
            (r#"["-INFINITY", 2.0]"#, "uppercase -infinity"),
            (r#"["-Infinity", 2.0]"#, "mixed case -infinity"),
            (r#"["-inf", 2.0]"#, "lowercase -inf"),
            (r#"["-INF", 2.0]"#, "uppercase -inf"),
            (r#"["-Inf", 2.0]"#, "mixed case -inf"),
        ];

        for (json_str, description) in neg_inf_cases {
            assert!(
                serde_json::from_str::<Point<2>>(json_str).is_err(),
                "special value should be rejected for {description}"
            );
        }

        let nan_cases = vec![
            (r#"["nan", 3.0]"#, "lowercase nan"),
            (r#"["NaN", 3.0]"#, "mixed case NaN"),
            (r#"["NAN", 3.0]"#, "uppercase NAN"),
            (r#"["Nan", 3.0]"#, "title case Nan"),
        ];

        for (json_str, description) in nan_cases {
            assert!(
                serde_json::from_str::<Point<2>>(json_str).is_err(),
                "special value should be rejected for {description}"
            );
        }

        let whitespace_cases = vec![
            (r#"[" infinity ", 1.0]"#, "spaces around infinity"),
            (r#"["\tinf\n", 2.0]"#, "tabs and newlines around inf"),
            (r#"["  NaN  ", 3.0]"#, "spaces around NaN"),
        ];

        for (json_str, description) in whitespace_cases {
            assert!(
                serde_json::from_str::<Point<2>>(json_str).is_err(),
                "special value should be rejected for {description}"
            );
        }

        let combined = r#"["INFINITY", "-inf", "Nan", 42.0]"#;
        assert!(serde_json::from_str::<Point<4>>(combined).is_err());

        // Test that unknown special values still fail
        let invalid = r#"["unknown_special", 1.0]"#;
        let result: Result<Point<2>, _> = serde_json::from_str(invalid);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("non-finite coordinate string")
        );
    }

    #[test]
    fn point_serialize_edge_values() {
        // Test serialization of edge finite values

        // Very large but finite values
        let point_max = Point::try_new([f64::MAX, f64::MIN]).expect("finite point coordinates");
        let json_max = serde_json::to_string(&point_max).unwrap();
        assert!(!json_max.contains("null")); // Should not be null

        // Very small but finite values
        let point_min = Point::try_new([f64::MIN_POSITIVE, -f64::MIN_POSITIVE])
            .expect("finite point coordinates");
        let json_min = serde_json::to_string(&point_min).unwrap();
        assert!(!json_min.contains("null")); // Should not be null

        // Zero and negative zero
        let point_zero = Point::try_new([0.0, -0.0]).expect("finite point coordinates");
        let json_zero = serde_json::to_string(&point_zero).unwrap();
        assert!(!json_zero.contains("null")); // Should not be null
        assert_eq!(json_zero, "[0.0,0.0]");
    }

    #[test]
    fn point_conversion_edge_cases() {
        // Test edge cases in type conversions

        // Test conversion with potential precision loss (should still work)
        let precise_coords = [1.000_000_000_000_001_f64, 2.000_000_000_000_002_f64];
        let point_precise: Point<2> =
            Point::try_new(precise_coords).expect("finite point coordinates");
        assert_relative_eq!(
            point_precise.to_array().as_slice(),
            precise_coords.as_slice()
        );

        // Test conversion from array reference
        let coords_ref = &[1.0, 2.0, 3.0];
        let point_from_ref: Point<3> =
            Point::try_new(*coords_ref).expect("finite point coordinates");
        assert_relative_eq!(
            point_from_ref.to_array().as_slice(),
            [1.0f64, 2.0f64, 3.0f64].as_slice()
        );

        // Test conversion to array with different methods
        let point = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");

        // Using Into trait
        let coords_into: [f64; 3] = point.into();
        assert_relative_eq!(coords_into.as_slice(), [1.0, 2.0, 3.0].as_slice());

        // Using From trait with reference
        let point_ref = Point::try_new([4.0, 5.0]).expect("finite point coordinates");
        let coords_from_ref: [f64; 2] = (&point_ref).into();
        assert_relative_eq!(coords_from_ref.as_slice(), [4.0, 5.0].as_slice());

        // Verify original point is still usable after reference conversion
        assert_relative_eq!(point_ref.to_array().as_slice(), [4.0, 5.0].as_slice());
    }

    #[test]
    fn point_cast_conversions() {
        // Test the cast()-based TryFrom<[T; D]> implementation

        // Test integer to f64 conversion using TryFrom
        let coords_i16: [i16; 3] = [1, 2, 3];
        let point_f64: Point<3> = Point::try_from(coords_i16).unwrap();

        // Verify the conversion worked correctly
        assert_relative_eq!(
            point_f64.to_array().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );

        // Test same type conversion (no actual cast needed)
        let coords_f64: [f64; 2] = [10.0, 20.0];
        let point_f64_same: Point<2> = Point::try_from(coords_f64).unwrap();
        assert_relative_eq!(
            point_f64_same.to_array().as_slice(),
            [10.0, 20.0].as_slice()
        );

        // Test with integer type conversions
        let coords_i32: [i32; 4] = [1, 2, 3, 4];
        let point_f64_from_int: Point<4> = Point::try_from(coords_i32).unwrap();
        assert_relative_eq!(
            point_f64_from_int.to_array().as_slice(),
            [1.0, 2.0, 3.0, 4.0].as_slice(),
            epsilon = 1e-9
        );

        // Test with large values that are within range
        let coords_large_i32: [i32; 2] = [i32::MAX, i32::MIN];
        let point_f64_from_large: Point<2> = Point::try_from(coords_large_i32).unwrap();
        assert_relative_eq!(
            point_f64_from_large.to_array().as_slice(),
            [f64::from(i32::MAX), f64::from(i32::MIN)].as_slice(),
            epsilon = 1e-9
        );

        // Test with mixed typical values
        let coords_mixed: [i16; 3] = [0, 15, -35];
        let point_mixed: Point<3> = Point::try_from(coords_mixed).unwrap();
        assert_relative_eq!(
            point_mixed.to_array().as_slice(),
            [0.0, 15.0, -35.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn point_hash_signed_zero_values() {
        let point_pos_zero = Point::try_new([0.0, 2.0]).expect("finite point coordinates");
        let point_neg_zero = Point::try_new([-0.0, 2.0]).expect("finite point coordinates");

        let mut hasher_pos_zero = DefaultHasher::new();
        let mut hasher_neg_zero = DefaultHasher::new();

        point_pos_zero.hash(&mut hasher_pos_zero);
        point_neg_zero.hash(&mut hasher_neg_zero);

        assert_eq!(hasher_pos_zero.finish(), hasher_neg_zero.finish());
    }

    #[test]
    fn prevalidated_finite_coordinates_canonicalize_signed_zero() {
        let coords = ValidatedCoordinates::from_prevalidated_finite_values([-0.0, 1.0]);
        let point = Point::from_validated_coordinates(coords);

        assert_eq!(point.coords()[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(point.coords()[1].to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn point_hashmap_finite_values() {
        let mut map: HashMap<Point<2>, &str> = HashMap::new();

        let point_zero = Point::try_new([0.0, 2.0]).expect("finite point coordinates");
        let point_regular = Point::try_new([1.0, 2.0]).expect("finite point coordinates");

        map.insert(point_zero, "Zero Point");
        map.insert(point_regular, "Regular Point");

        assert_eq!(
            map[&Point::try_new([-0.0, 2.0]).expect("finite point coordinates")],
            "Zero Point"
        );
        assert_eq!(
            map[&Point::try_new([1.0, 2.0]).expect("finite point coordinates")],
            "Regular Point"
        );
    }

    #[test]
    fn point_hashset_finite_values() {
        let mut set: HashSet<Point<2>> = HashSet::new();

        set.insert(Point::try_new([0.0, 2.0]).expect("finite point coordinates"));
        set.insert(Point::try_new([-0.0, 2.0]).expect("finite point coordinates"));
        set.insert(Point::try_new([1.0, 2.0]).expect("finite point coordinates"));
        set.insert(Point::try_new([1.0, 2.0]).expect("finite point coordinates"));

        assert_eq!(set.len(), 2);

        assert!(set.contains(&Point::try_new([-0.0, 2.0]).expect("finite point coordinates")));
        assert!(set.contains(&Point::try_new([1.0, 2.0]).expect("finite point coordinates")));
    }

    #[test]
    fn point_hash_distribution_basic() {
        // Test that different points generally produce different hashes
        // (This is a probabilistic test, not a guarantee)

        let mut hashes = HashSet::new();

        // Generate a variety of points and collect their hashes
        for i in 0..100 {
            let point =
                Point::try_new([f64::from(i), f64::from(i * 2)]).expect("finite point coordinates");
            let hash = hash_of(&point);
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
            let point = Point::try_new([f64::from(i), f64::from(i * 3), f64::from(i * 5)])
                .expect("finite point coordinates");
            let hash = hash_of(&point);
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
        let result = Point::<3>::try_new([1.0, f64::NAN, 3.0]);
        assert!(result.is_err());

        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 1);
            assert_eq!(dimension, 3);
            assert_eq!(coordinate_value, InvalidCoordinateValue::Nan);
        } else {
            panic!("Expected InvalidCoordinate error");
        }

        // Test with infinity at different positions
        let result = Point::<4>::try_new([f64::INFINITY, 2.0, 3.0, 4.0]);
        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 0);
            assert_eq!(dimension, 4);
            assert_eq!(coordinate_value, InvalidCoordinateValue::PositiveInfinity);
        } else {
            panic!("Expected InvalidCoordinate error");
        }

        // Test with negative infinity at last position
        let result = Point::<3>::try_new([1.0, 2.0, f64::NEG_INFINITY]);
        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 2);
            assert_eq!(dimension, 3);
            assert_eq!(coordinate_value, InvalidCoordinateValue::NegativeInfinity);
        }
    }

    #[test]
    fn point_validation_error_display() {
        // Test error message formatting
        let result = Point::<3>::try_new([1.0, f64::NAN, 3.0]);

        if let Err(error) = result {
            let error_msg = format!("{error}");
            assert!(error_msg.contains("Invalid coordinate at index 1"));
            assert!(error_msg.contains("in dimension 3"));
            assert!(error_msg.contains("NaN"));
        } else {
            panic!("Expected validation error");
        }

        // Test with infinity
        let result = Point::<1>::try_new([f64::INFINITY]);

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
        let result1 = Point::<2>::try_new([f64::NAN, 2.0]);
        let result2 = Point::<2>::try_new([f64::NAN, 2.0]);

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
    fn point_validation_first_invalid_coordinate() {
        // Test that validation returns the FIRST invalid coordinate found
        let result = Point::<4>::try_new([1.0, f64::NAN, f64::INFINITY, f64::NAN]);

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
        let result = Point::<3>::try_new([f64::INFINITY, f64::NAN, 3.0]);

        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index, ..
        }) = result
        {
            assert_eq!(coordinate_index, 0);
        }
    }

    #[test]
    fn point_hashmap_with_finite_values() {
        let mut point_map: HashMap<Point<3>, &str> = HashMap::new();

        let point_normal = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        let point_other = Point::try_new([4.0, 5.0, 6.0]).expect("finite point coordinates");

        point_map.insert(point_normal, "normal point");
        point_map.insert(point_other, "other point");

        assert_eq!(point_map.len(), 2);

        let point_normal_copy = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        let point_other_copy = Point::try_new([4.0, 5.0, 6.0]).expect("finite point coordinates");

        assert!(point_map.contains_key(&point_normal_copy));
        assert!(point_map.contains_key(&point_other_copy));

        assert_eq!(point_map.get(&point_normal_copy), Some(&"normal point"));
        assert_eq!(point_map.get(&point_other_copy), Some(&"other point"));
    }

    #[test]
    fn point_hashset_with_finite_values() {
        let mut point_set: HashSet<Point<2>> = HashSet::new();

        let points = vec![
            Point::try_new([1.0, 2.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 2.0]).expect("finite point coordinates"), // Duplicate normal point
            Point::try_new([0.0, -0.0]).expect("finite point coordinates"), // Zero and negative zero (equal)
            Point::try_new([-0.0, 0.0]).expect("finite point coordinates"), // Different zero combination
        ];

        for point in points {
            point_set.insert(point);
        }

        assert_eq!(point_set.len(), 2);

        let test_normal = Point::try_new([1.0, 2.0]).expect("finite point coordinates");

        assert!(point_set.contains(&test_normal));
        assert!(point_set.contains(&Point::try_new([0.0, 0.0]).expect("finite point coordinates")));
    }

    // =============================================================================
    // TryFrom CONVERSION ERROR TESTS
    // =============================================================================

    #[test]
    fn point_try_from_successful_conversions() {
        // Test successful conversions that don't overflow

        // i32 to f64
        let coords_i32: [i32; 4] = [1, -2, 3, -4];
        let point_from_int: Point<4> = Point::try_from(coords_i32).unwrap();
        assert_relative_eq!(
            point_from_int.to_array().as_slice(),
            [1.0, -2.0, 3.0, -4.0].as_slice(),
            epsilon = 1e-9
        );

        // Same type conversion (f64 to f64)
        let coords_same: [f64; 2] = [10.0, 20.0];
        let point_same: Point<2> = Point::try_from(coords_same).unwrap();
        assert_relative_eq!(point_same.to_array().as_slice(), [10.0, 20.0].as_slice());
    }

    #[test]
    fn point_try_from_edge_case_values() {
        // Test with zero and negative zero
        let coords_zero: [f64; 2] = [0.0, -0.0];
        let point_zero: Point<2> = Point::try_from(coords_zero).unwrap();
        assert_relative_eq!(point_zero.to_array()[0], 0.0);
        assert_relative_eq!(point_zero.to_array()[1], -0.0);

        // Test with very small values
        let coords_small: [f64; 2] = [1e-10, -1e-10];
        let point_small: Point<2> = Point::try_from(coords_small).unwrap();
        assert!(point_small.to_array()[0].is_finite());
        assert!(point_small.to_array()[1].is_finite());
    }

    #[test]
    fn point_try_from_integer_to_float_conversions() {
        // Test various integer types to floating point

        // u32 to f64
        let coords_u32: [u32; 3] = [100, 200, 300];
        let point_u32: Point<3> = Point::try_from(coords_u32).unwrap();
        assert_relative_eq!(
            point_u32.to_array().as_slice(),
            [100.0, 200.0, 300.0].as_slice(),
            epsilon = 1e-9
        );

        // i16 to f64
        let coords_i16: [i16; 2] = [-100, 200];
        let point_i16: Point<2> = Point::try_from(coords_i16).unwrap();
        assert_relative_eq!(
            point_i16.to_array().as_slice(),
            [-100.0, 200.0].as_slice(),
            epsilon = 1e-9
        );

        // Large but representable integers
        let coords_large_i32: [i32; 2] = [1_000_000, -1_000_000];
        let point_large: Point<2> = Point::try_from(coords_large_i32).unwrap();
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
        let valid_coords: [i32; 3] = [1, 2, 3];
        let result: Result<Point<3>, _> = Point::try_from(valid_coords);
        assert!(result.is_ok());

        // Invalid - input contains infinity
        let invalid_coords = [1.0, f64::INFINITY, 3.0];
        let result: Result<Point<3>, _> = Point::try_from(invalid_coords);
        assert!(result.is_err());
    }

    // =============================================================================
    // DIM() METHOD EXPLICIT TESTS
    // =============================================================================

    #[test]
    fn point_dim_method_explicit() {
        // Test the dim() method explicitly across various dimensions

        let point_1d: Point<1> = Point::try_new([1.0]).expect("finite point coordinates");
        assert_eq!(point_1d.dim(), 1);

        let point_2d: Point<2> = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
        assert_eq!(point_2d.dim(), 2);

        let point_3d: Point<3> = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        assert_eq!(point_3d.dim(), 3);

        let point_5d: Point<5> =
            Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
        assert_eq!(point_5d.dim(), 5);

        let point_10d: Point<10> = Point::try_new([0.0; 10]).expect("finite point coordinates");
        assert_eq!(point_10d.dim(), 10);

        let point_32d: Point<32> = Point::try_new([0.0; 32]).expect("finite point coordinates");
        assert_eq!(point_32d.dim(), 32);
    }

    // =============================================================================
    // TO_ARRAY() METHOD EXPLICIT TESTS
    // =============================================================================

    #[test]
    fn point_to_array_explicit() {
        // Test to_array() method explicitly

        let point = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        let arr = point.to_array();
        assert_relative_eq!(arr.as_slice(), [1.0, 2.0, 3.0].as_slice());

        // Verify that to_array() returns a copy, not a reference
        let point2 = Point::try_new([4.0, 5.0]).expect("finite point coordinates");
        let arr2 = point2.to_array();
        assert_relative_eq!(arr2.as_slice(), [4.0, 5.0].as_slice());

        // Test with different dimensions
        let point_1d = Point::try_new([42.0]).expect("finite point coordinates");
        assert_relative_eq!(point_1d.to_array().as_slice(), [42.0].as_slice());

        let point_5d = Point::try_new([1.0, 2.0, 3.0, 4.0, 5.0]).expect("finite point coordinates");
        assert_relative_eq!(
            point_5d.to_array().as_slice(),
            [1.0, 2.0, 3.0, 4.0, 5.0].as_slice()
        );
    }

    // =============================================================================
    // ORDERED_EQUALS() AND HASH_COORDINATE() DIRECT TESTS
    // =============================================================================

    #[test]
    fn point_ordered_equals_direct() {
        // Test ordered_equals() method directly

        let point1 = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        let point2 = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        let point3 = Point::try_new([1.0, 2.0, 4.0]).expect("finite point coordinates");

        assert!(point1.ordered_equals(&point2));
        assert!(!point1.ordered_equals(&point3));
    }

    #[test]
    fn point_hash_coordinate_direct() {
        // Test hash_coordinate() method directly

        let point1 = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");
        let point2 = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        point1.hash_coordinate(&mut hasher1);
        point2.hash_coordinate(&mut hasher2);

        assert_eq!(hasher1.finish(), hasher2.finish());

        // Test with different points
        let point3 = Point::try_new([1.0, 2.0, 4.0]).expect("finite point coordinates");
        let mut hasher3 = DefaultHasher::new();
        point3.hash_coordinate(&mut hasher3);

        assert_ne!(hasher1.finish(), hasher3.finish());
    }

    // =============================================================================
    // COMPREHENSIVE 1D POINT TESTS
    // =============================================================================

    #[test]
    fn point_1d_comprehensive() {
        // Test 1D points comprehensively

        // Creation
        let point = Point::try_new([42.0]).expect("finite point coordinates");
        assert_eq!(point.dim(), 1);
        assert_relative_eq!(point.to_array().as_slice(), [42.0].as_slice());

        // Equality
        let point2 = Point::try_new([42.0]).expect("finite point coordinates");
        assert_eq!(point, point2);

        let point3 = Point::try_new([43.0]).expect("finite point coordinates");
        assert_ne!(point, point3);

        // Hashing
        assert_eq!(hash_of(&point), hash_of(&point2));
        assert_ne!(hash_of(&point), hash_of(&point3));

        // Ordering
        assert!(point < point3);
        assert!(point3 > point);

        // Validation
        assert!(point.validate().is_ok());
        assert!(Point::<1>::try_new([f64::NAN]).is_err());

        // Origin
        let origin: Point<1> = Point::origin();
        assert_relative_eq!(origin.to_array().as_slice(), [0.0].as_slice());

        // Serialization
        let json = serde_json::to_string(&point).unwrap();
        assert_eq!(json, "[42.0]");
        let deserialized: Point<1> = serde_json::from_str(&json).unwrap();
        assert_eq!(point, deserialized);
    }

    #[test]
    fn point_floating_point_precision() {
        // Test that we can distinguish between very close floating point values
        let point_epsilon1 =
            Point::try_new([1.0 + f64::EPSILON, 2.0]).expect("finite point coordinates");
        let point_epsilon2 = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
        assert_ne!(point_epsilon1, point_epsilon2);

        // Test with values that should be exactly equal
        let point_exact1 = Point::try_new([0.1 + 0.2, 1.0]).expect("finite point coordinates");
        let point_exact2 = Point::try_new([0.3, 1.0]).expect("finite point coordinates");
        // Note: Due to floating point representation, 0.1 + 0.2 != 0.3
        // This test demonstrates the exact equality behavior
        assert_ne!(point_exact1, point_exact2);

        // Test that points with slightly different values are not approximately equal
        // (demonstrating that we use exact equality, not approximate)
        let point_a = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
        let point_b = Point::try_new([1.0 + f64::EPSILON, 2.0]).expect("finite point coordinates");
        assert_ne!(point_a, point_b);

        // But points with exactly the same values are equal
        let point_same1 = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
        let point_same2 = Point::try_new([1.0, 2.0]).expect("finite point coordinates");
        assert_eq!(point_same1, point_same2);
    }

    #[test]
    fn point_zero_and_negative_zero() {
        // Test zero and negative zero behavior
        let point_pos_zero = Point::try_new([0.0, 0.0]).expect("finite point coordinates");
        let point_neg_zero = Point::try_new([-0.0, -0.0]).expect("finite point coordinates");
        let point_mixed_zero = Point::try_new([0.0, -0.0]).expect("finite point coordinates");
        let point_mixed_zero2 = Point::try_new([-0.0, 0.0]).expect("finite point coordinates");

        // All should be equal (0.0 == -0.0 in IEEE 754)
        assert_eq!(point_pos_zero, point_neg_zero);
        assert_eq!(point_pos_zero, point_mixed_zero);
        assert_eq!(point_pos_zero, point_mixed_zero2);
        assert_eq!(point_neg_zero, point_mixed_zero);
        assert_eq!(point_neg_zero, point_mixed_zero2);
        assert_eq!(point_mixed_zero, point_mixed_zero2);

        // Test hashing consistency
        let hash_pos = hash_of(&point_pos_zero);
        let hash_neg = hash_of(&point_neg_zero);
        let hash_mixed1 = hash_of(&point_mixed_zero);
        let hash_mixed2 = hash_of(&point_mixed_zero2);

        assert_eq!(hash_pos, hash_neg);
        assert_eq!(hash_pos, hash_mixed1);
        assert_eq!(hash_pos, hash_mixed2);
    }

    // =============================================================================
    // CONVERSION ERROR TESTS
    // =============================================================================

    fn assert_non_finite_coordinate_index(
        error: &CoordinateConversionError,
        expected_index: usize,
    ) {
        let CoordinateConversionError::NonFiniteValue {
            coordinate_index, ..
        } = error
        else {
            panic!("Expected NonFiniteValue error at position {expected_index}");
        };
        assert_eq!(*coordinate_index, expected_index);
    }

    #[test]
    fn point_try_from_conversion_errors() {
        // Test non-finite value errors (NaN)
        let coords_with_nan = [f64::NAN, 1.0, 2.0];
        let result: Result<Point<3>, _> = Point::try_from(coords_with_nan);
        let error = result.unwrap_err();
        assert_non_finite_coordinate_index(&error, 0);

        // Test non-finite value errors (infinity)
        let coords_with_inf = [1.0, f64::INFINITY, 2.0];
        let result: Result<Point<3>, _> = Point::try_from(coords_with_inf);
        let error = result.unwrap_err();
        assert_non_finite_coordinate_index(&error, 1);

        // Test conversion failure (overflow cases if we had them)
        // Note: With num_traits::cast, most reasonable numeric conversions succeed,
        // so ConversionFailed errors are rare in practice for standard numeric types.
        // But the infrastructure is there for edge cases or custom numeric types.
    }

    #[test]
    fn point_try_from_success_cases() {
        // Test successful conversions that should work fine

        // i32 to f64
        let coords_i32 = [1i32, -2i32, 3i32];
        let result: Result<Point<3>, _> = Point::try_from(coords_i32);
        assert!(result.is_ok());
        let point = result.unwrap();
        assert_relative_eq!(
            point.to_array().as_slice(),
            [1.0f64, -2.0f64, 3.0f64].as_slice(),
            epsilon = 1e-9
        );

        // Same type (f64 to f64)
        let coords_f64 = [1.0f64, 2.0f64];
        let result: Result<Point<2>, _> = Point::try_from(coords_f64);
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
        let result: Result<Point<2>, _> = Point::try_from(coords_with_nan);
        assert!(result.is_err());

        let error = result.unwrap_err();
        let error_msg = format!("{error}");
        assert!(error_msg.contains("Non-finite value"));
        assert!(error_msg.contains("coordinate index 0"));
        assert!(error_msg.contains("NaN"));

        // Test error cloning and equality with infinity
        let coords_with_inf = [f64::INFINITY, 2.0];
        let result2: Result<Point<2>, _> = Point::try_from(coords_with_inf);
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
            let result: Result<Point<4>, _> = Point::try_from(coords);
            let error = result.unwrap_err();
            assert_non_finite_coordinate_index(&error, expected_index);
        }
    }

    #[test]
    fn point_try_from_first_error_reported() {
        // When multiple coordinates have errors, the first one should be reported
        let coords_multi_error = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
        let result: Result<Point<3>, _> = Point::try_from(coords_multi_error);
        let error = result.unwrap_err();
        assert_non_finite_coordinate_index(&error, 0);
    }

    #[test]
    fn point_trait_completeness() {
        // Helper functions for compile-time trait checks
        fn assert_send<T: Send>(_: T) {}
        fn assert_sync<T: Sync>(_: T) {}

        // Test that Point implements all expected traits

        let point = Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates");

        // Test Debug trait
        let debug_output = format!("{point:?}");
        assert!(!debug_output.is_empty());
        assert!(debug_output.contains("Point"));

        // Test Default trait
        let default_point: Point<3> = Point::default();
        assert_relative_eq!(
            default_point.to_array().as_slice(),
            [0.0, 0.0, 0.0].as_slice()
        );

        // Test PartialOrd trait (ordering)
        let point_smaller = Point::try_new([1.0, 2.0, 2.9]).expect("finite point coordinates");
        assert!(point_smaller < point);

        // Test that Send and Sync are implemented (compile-time check)
        assert_send(point);
        assert_sync(point);

        // Test Clone and Copy
        #[expect(
            clippy::clone_on_copy,
            reason = "test asserts explicit clone behavior for copy coordinates"
        )]
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
