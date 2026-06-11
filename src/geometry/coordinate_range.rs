//! Validated coordinate-range types.

#![forbid(unsafe_code)]

use crate::geometry::traits::coordinate::FiniteCheck;
pub use crate::geometry::traits::coordinate::InvalidCoordinateValue;
use core::fmt::{self, Debug, Display};

/// Identifies which coordinate-range bound failed validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CoordinateRangeBound {
    /// Lower coordinate bound.
    Minimum,
    /// Upper coordinate bound.
    Maximum,
}

impl fmt::Display for CoordinateRangeBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minimum => f.write_str("minimum"),
            Self::Maximum => f.write_str("maximum"),
        }
    }
}

/// The ordering failure for a coordinate range whose bounds are finite.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CoordinateRangeOrdering {
    /// The bounds are equal.
    Equal,
    /// The minimum bound is greater than the maximum bound.
    Decreasing,
}

impl fmt::Display for CoordinateRangeOrdering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Equal => f.write_str("equal"),
            Self::Decreasing => f.write_str("decreasing"),
        }
    }
}

/// Errors that can occur while constructing validated coordinate ranges.
///
/// Non-finite bounds are reported as [`InvalidCoordinateValue`] categories.
/// Finite but non-increasing bounds preserve their typed `min` and `max`
/// values so callers can inspect the rejected range without parsing a display
/// string.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{CoordinateRangeError, CoordinateRangeOrdering};
///
/// let err = CoordinateRangeError::NonIncreasing {
///     ordering: CoordinateRangeOrdering::Decreasing,
///     min: 1.0,
///     max: 0.0,
/// };
/// std::assert_matches!(err, CoordinateRangeError::NonIncreasing { .. });
/// ```
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum CoordinateRangeError<T = f64> {
    /// A coordinate range bound is non-finite.
    #[error("Invalid coordinate range: {bound} bound is non-finite: {value}")]
    NonFiniteBound {
        /// Which bound failed validation.
        bound: CoordinateRangeBound,
        /// The non-finite bound value.
        value: InvalidCoordinateValue,
    },

    /// Finite coordinate bounds are not strictly increasing.
    #[error(
        "Invalid coordinate range: minimum {min:?} and maximum {max:?} must satisfy min < max ({ordering})"
    )]
    NonIncreasing {
        /// Ordering failure category.
        ordering: CoordinateRangeOrdering,
        /// The minimum value of the range.
        min: T,
        /// The maximum value of the range.
        max: T,
    },
}

/// Finite, strictly increasing coordinate bounds.
///
/// `CoordinateRange<T>` carries the invariant that both bounds are finite and
/// `min < max`, so callers and geometry internals can pass validated bounds
/// inward without rechecking raw tuple bounds at every use site.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::geometry::{CoordinateRange, CoordinateRangeError};
///
/// # fn main() -> Result<(), CoordinateRangeError> {
/// let range = CoordinateRange::try_new(-1.0_f64, 1.0)?;
/// assert_eq!(range.min(), -1.0);
/// assert_eq!(range.max(), 1.0);
/// assert_eq!(range.bounds(), (-1.0, 1.0));
/// # Ok(())
/// # }
/// ```
#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoordinateRange<T> {
    min: T,
    max: T,
}

impl<T> CoordinateRange<T>
where
    T: Debug + FiniteCheck + PartialOrd,
{
    /// Creates a coordinate range from raw bounds.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinateRangeError::NonFiniteBound`] if either bound is
    /// non-finite, or [`CoordinateRangeError::NonIncreasing`] if `min >= max`.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::geometry::{CoordinateRange, CoordinateRangeError};
    ///
    /// # fn main() -> Result<(), CoordinateRangeError> {
    /// let range = CoordinateRange::try_new(0.0_f64, 1.0)?;
    /// assert_eq!(range.bounds(), (0.0, 1.0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_new(min: T, max: T) -> Result<Self, CoordinateRangeError<T>> {
        if !min.is_finite_generic() {
            return Err(CoordinateRangeError::NonFiniteBound {
                bound: CoordinateRangeBound::Minimum,
                value: InvalidCoordinateValue::from_debug(&min),
            });
        }

        if !max.is_finite_generic() {
            return Err(CoordinateRangeError::NonFiniteBound {
                bound: CoordinateRangeBound::Maximum,
                value: InvalidCoordinateValue::from_debug(&max),
            });
        }

        if min >= max {
            let ordering = if min == max {
                CoordinateRangeOrdering::Equal
            } else {
                CoordinateRangeOrdering::Decreasing
            };
            return Err(CoordinateRangeError::NonIncreasing { ordering, min, max });
        }

        Ok(Self { min, max })
    }
}

impl<T> CoordinateRange<T> {
    /// Creates a coordinate range from finite bounds already proven to satisfy `min < max`.
    ///
    /// This constructor is restricted to the geometry module so public callers
    /// must still use [`CoordinateRange::try_new`] or [`TryFrom`].
    pub(in crate::geometry) const fn from_validated_bounds(min: T, max: T) -> Self {
        Self { min, max }
    }

    /// Consumes the range and returns it as a raw `(min, max)` tuple.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::geometry::{CoordinateRange, CoordinateRangeError};
    ///
    /// # fn main() -> Result<(), CoordinateRangeError> {
    /// let range = CoordinateRange::try_new(-1.0_f64, 2.0)?;
    /// assert_eq!(range.bounds(), (-1.0, 2.0));
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn bounds(self) -> (T, T) {
        (self.min, self.max)
    }

    /// Returns the lower coordinate bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::geometry::{CoordinateRange, CoordinateRangeError};
    ///
    /// # fn main() -> Result<(), CoordinateRangeError> {
    /// let range = CoordinateRange::try_new(-1.0_f64, 2.0)?;
    /// assert_eq!(range.min(), -1.0);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn min(self) -> T {
        self.min
    }

    /// Returns the upper coordinate bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::geometry::{CoordinateRange, CoordinateRangeError};
    ///
    /// # fn main() -> Result<(), CoordinateRangeError> {
    /// let range = CoordinateRange::try_new(-1.0_f64, 2.0)?;
    /// assert_eq!(range.max(), 2.0);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn max(self) -> T {
        self.max
    }
}

impl<T> TryFrom<(T, T)> for CoordinateRange<T>
where
    T: Debug + FiniteCheck + PartialOrd,
{
    type Error = CoordinateRangeError<T>;

    fn try_from(bounds: (T, T)) -> Result<Self, Self::Error> {
        Self::try_new(bounds.0, bounds.1)
    }
}

impl<T> From<CoordinateRange<T>> for (T, T) {
    fn from(range: CoordinateRange<T>) -> Self {
        (range.min, range.max)
    }
}

impl<T: Display> fmt::Display for CoordinateRange<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}]", self.min, self.max)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CoordinateRange, CoordinateRangeBound, CoordinateRangeError, CoordinateRangeOrdering,
        InvalidCoordinateValue,
    };
    use crate::geometry::traits::coordinate::FiniteCheck;
    use approx::assert_relative_eq;
    use std::assert_matches;

    #[derive(Debug, PartialEq, PartialOrd)]
    struct NonCopyFiniteScalar(i32);

    impl FiniteCheck for NonCopyFiniteScalar {
        fn is_finite_generic(&self) -> bool {
            true
        }
    }

    #[derive(Debug, PartialEq, PartialOrd)]
    struct CustomNonFiniteScalar(&'static str);

    impl FiniteCheck for CustomNonFiniteScalar {
        fn is_finite_generic(&self) -> bool {
            false
        }
    }

    /// Asserts that finite raw coordinate bounds are rejected with the exact diagnostic payload.
    fn assert_non_increasing_bounds(
        result: &Result<CoordinateRange<f64>, CoordinateRangeError>,
        expected_ordering: CoordinateRangeOrdering,
        expected_min: f64,
        expected_max: f64,
    ) {
        let Err(CoordinateRangeError::NonIncreasing { ordering, min, max }) = result else {
            panic!("expected non-increasing coordinate bounds");
        };
        assert_eq!(*ordering, expected_ordering);
        assert_relative_eq!(*min, expected_min, epsilon = f64::EPSILON);
        assert_relative_eq!(*max, expected_max, epsilon = f64::EPSILON);
    }

    /// Asserts that a non-finite coordinate bound is rejected with the typed payload.
    fn assert_non_finite_bound(
        result: Result<CoordinateRange<f64>, CoordinateRangeError>,
        expected_bound: CoordinateRangeBound,
        expected_value: &InvalidCoordinateValue,
    ) {
        let Err(CoordinateRangeError::NonFiniteBound { bound, value }) = result else {
            panic!("expected non-finite coordinate bound");
        };
        assert_eq!(bound, expected_bound);
        assert_eq!(&value, expected_value);
    }

    #[test]
    fn accepts_f64_finite_increasing_bounds() {
        let range = CoordinateRange::try_new(-2.0_f64, 3.5).unwrap();
        let converted = CoordinateRange::try_from((-2.0_f64, 3.5)).unwrap();
        let raw_bounds = <(f64, f64)>::from(range);

        assert_relative_eq!(range.min(), -2.0, epsilon = f64::EPSILON);
        assert_relative_eq!(range.max(), 3.5, epsilon = f64::EPSILON);
        assert_relative_eq!(range.bounds().0, -2.0, epsilon = f64::EPSILON);
        assert_relative_eq!(range.bounds().1, 3.5, epsilon = f64::EPSILON);
        assert_relative_eq!(converted.min(), range.min(), epsilon = f64::EPSILON);
        assert_relative_eq!(converted.max(), range.max(), epsilon = f64::EPSILON);
        assert_relative_eq!(raw_bounds.0, -2.0, epsilon = f64::EPSILON);
        assert_relative_eq!(raw_bounds.1, 3.5, epsilon = f64::EPSILON);
    }

    #[test]
    fn bounds_does_not_require_copy_scalar() {
        let range = CoordinateRange::try_new(NonCopyFiniteScalar(1), NonCopyFiniteScalar(2))
            .expect("test scalar is finite and increasing");

        let (min, max) = range.bounds();

        assert_eq!(min, NonCopyFiniteScalar(1));
        assert_eq!(max, NonCopyFiniteScalar(2));
    }

    #[test]
    fn display_formats_validated_bounds() {
        let range = CoordinateRange::try_new(-2.0_f64, 3.5).unwrap();
        assert_eq!(range.to_string(), "[-2, 3.5]");
    }

    #[test]
    fn rejects_non_increasing_or_non_finite_bounds_with_exact_payloads() {
        assert_non_increasing_bounds(
            &CoordinateRange::try_new(2.0_f64, 1.0),
            CoordinateRangeOrdering::Decreasing,
            2.0,
            1.0,
        );
        assert_non_increasing_bounds(
            &CoordinateRange::try_new(1.0_f64, 1.0),
            CoordinateRangeOrdering::Equal,
            1.0,
            1.0,
        );
        assert_non_finite_bound(
            CoordinateRange::try_new(f64::NAN, 1.0),
            CoordinateRangeBound::Minimum,
            &InvalidCoordinateValue::Nan,
        );
        assert_non_finite_bound(
            CoordinateRange::try_new(0.0_f64, f64::NAN),
            CoordinateRangeBound::Maximum,
            &InvalidCoordinateValue::Nan,
        );
        assert_non_finite_bound(
            CoordinateRange::try_new(f64::NEG_INFINITY, 1.0),
            CoordinateRangeBound::Minimum,
            &InvalidCoordinateValue::NegativeInfinity,
        );
        assert_non_finite_bound(
            CoordinateRange::try_new(0.0_f64, f64::INFINITY),
            CoordinateRangeBound::Maximum,
            &InvalidCoordinateValue::PositiveInfinity,
        );
    }

    #[test]
    fn custom_non_finite_bound_preserves_debug_payload() {
        let result = CoordinateRange::try_new(
            CustomNonFiniteScalar("custom-lower"),
            CustomNonFiniteScalar("custom-upper"),
        );

        let Err(CoordinateRangeError::NonFiniteBound { bound, value }) = result else {
            panic!("expected custom non-finite coordinate bound");
        };
        assert_eq!(bound, CoordinateRangeBound::Minimum);
        assert_eq!(
            value,
            InvalidCoordinateValue::Other("CustomNonFiniteScalar(\"custom-lower\")".to_string())
        );
    }

    #[test]
    fn try_from_tuple_uses_same_validation_as_constructor() {
        assert_non_increasing_bounds(
            &CoordinateRange::try_from((3.0_f64, 2.0)),
            CoordinateRangeOrdering::Decreasing,
            3.0,
            2.0,
        );

        let range = CoordinateRange::try_from((-3.0_f64, -2.0)).unwrap();
        assert_relative_eq!(range.min(), -3.0, epsilon = f64::EPSILON);
        assert_relative_eq!(range.max(), -2.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn invalid_bounds_display_names_the_invariant() {
        let err = CoordinateRange::try_new(1.0_f64, 1.0).unwrap_err();

        assert_eq!(
            err.to_string(),
            "Invalid coordinate range: minimum 1.0 and maximum 1.0 must satisfy min < max (equal)"
        );
        assert_matches!(err, CoordinateRangeError::NonIncreasing { .. });
    }
}
