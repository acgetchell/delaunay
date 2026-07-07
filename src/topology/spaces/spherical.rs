//! Spherical coordinate and metric backend.
//!
//! This module provides fallible coordinate and metric types for points on
//! `S^D` embedded in `R^(D+1)`.

#![forbid(unsafe_code)]

use thiserror::Error;

/// Typed point on `S^D` embedded in `R^(D+1)`.
///
/// Rust's stable const generics do not allow a public `[f64; D + 1]` field, so
/// this topology backend stores the ambient coordinates in a length-checked
/// vector. Construction proves that the vector length is exactly `D + 1`,
/// every coordinate is finite, and the point has been normalized to the
/// requested radius.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::spaces::SphericalPoint;
///
/// let point = SphericalPoint::<2>::try_new([3.0, 4.0, 0.0])?;
/// assert_eq!(point.ambient_dimension(), 3);
/// assert!((point.coords()[0] - 0.6).abs() < 1e-12);
/// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalPoint<const D: usize> {
    coords: Vec<f64>,
    radius: f64,
}

impl<const D: usize> SphericalPoint<D> {
    /// Creates a unit-radius spherical point from raw ambient coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalPointError`] when the coordinate count is not `D + 1`,
    /// any coordinate is non-finite, or the input vector has zero norm.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalPoint;
    ///
    /// let point = SphericalPoint::<2>::try_new([3.0, 4.0, 0.0])?;
    ///
    /// assert_eq!(point.ambient_dimension(), 3);
    /// assert!((point.coords()[0] - 0.6).abs() < 1e-12);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    pub fn try_new<const N: usize>(coords: [f64; N]) -> Result<Self, SphericalPointError> {
        Self::try_new_with_radius(coords, 1.0)
    }

    /// Creates a spherical point on radius `radius`.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalPointError`] when `radius` is not finite and positive,
    /// the coordinate count is not `D + 1`, any coordinate is non-finite, or the
    /// input vector has zero norm.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalPoint;
    ///
    /// let point = SphericalPoint::<2>::try_new_with_radius([0.0, 3.0, 4.0], 2.0)?;
    ///
    /// assert_eq!(point.radius(), 2.0);
    /// assert!((point.squared_norm() - 4.0).abs() < 1e-12);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    pub fn try_new_with_radius<const N: usize>(
        coords: [f64; N],
        radius: f64,
    ) -> Result<Self, SphericalPointError> {
        Self::try_from_slice_with_radius(&coords, radius)
    }

    /// Creates a unit-radius spherical point from a raw coordinate slice.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalPointError`] when the slice cannot be normalized into
    /// a point on `S^D`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalPoint;
    ///
    /// let raw = vec![0.0, 0.0, 2.0];
    /// let point = SphericalPoint::<2>::try_from_slice(&raw)?;
    ///
    /// assert_eq!(point.coords(), &[0.0, 0.0, 1.0]);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    pub fn try_from_slice(coords: &[f64]) -> Result<Self, SphericalPointError> {
        Self::try_from_slice_with_radius(coords, 1.0)
    }

    /// Creates a spherical point from a raw coordinate slice and radius.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalPointError`] when `radius` is not finite and positive
    /// or when the slice cannot be normalized into a point on `S^D`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalPoint;
    ///
    /// let raw = [1.0, 0.0, 0.0];
    /// let point = SphericalPoint::<2>::try_from_slice_with_radius(&raw, 3.0)?;
    ///
    /// assert_eq!(point.coords(), &[3.0, 0.0, 0.0]);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    pub fn try_from_slice_with_radius(
        coords: &[f64],
        radius: f64,
    ) -> Result<Self, SphericalPointError> {
        validate_radius(radius)?;
        let expected = Self::ambient_dimension_for_intrinsic();
        if coords.len() != expected {
            return Err(SphericalPointError::InvalidAmbientCoordinateCount {
                dimension: D,
                expected,
                actual: coords.len(),
            });
        }

        let mut normalized = coords.to_vec();
        normalize_coordinates(&mut normalized, radius)?;
        Ok(Self {
            coords: normalized,
            radius,
        })
    }

    /// Returns the intrinsic sphere dimension `D`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalPoint;
    ///
    /// let point = SphericalPoint::<3>::try_new([1.0, 0.0, 0.0, 0.0])?;
    ///
    /// assert_eq!(point.intrinsic_dimension(), 3);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    #[must_use]
    pub const fn intrinsic_dimension(&self) -> usize {
        D
    }

    /// Returns the ambient coordinate dimension `D + 1`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalPoint;
    ///
    /// let point = SphericalPoint::<2>::try_new([1.0, 0.0, 0.0])?;
    ///
    /// assert_eq!(point.ambient_dimension(), 3);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    #[must_use]
    pub const fn ambient_dimension(&self) -> usize {
        Self::ambient_dimension_for_intrinsic()
    }

    /// Returns the ambient coordinate dimension for this intrinsic `D`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalPoint;
    ///
    /// assert_eq!(SphericalPoint::<4>::ambient_dimension_for_intrinsic(), 5);
    /// ```
    #[must_use]
    pub const fn ambient_dimension_for_intrinsic() -> usize {
        D + 1
    }

    /// Returns the radius this point was normalized to.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalPoint;
    ///
    /// let point = SphericalPoint::<2>::try_new_with_radius([1.0, 0.0, 0.0], 2.5)?;
    ///
    /// assert_eq!(point.radius(), 2.5);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// Returns the normalized ambient coordinates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalPoint;
    ///
    /// let point = SphericalPoint::<2>::try_new([0.0, 0.0, 8.0])?;
    ///
    /// assert_eq!(point.coords(), &[0.0, 0.0, 1.0]);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    #[must_use]
    pub fn coords(&self) -> &[f64] {
        &self.coords
    }

    /// Returns the squared Euclidean norm of the stored ambient coordinates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalPoint;
    ///
    /// let point = SphericalPoint::<2>::try_new_with_radius([0.0, 1.0, 0.0], 4.0)?;
    ///
    /// assert!((point.squared_norm() - 16.0).abs() < 1e-12);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    #[must_use]
    pub fn squared_norm(&self) -> f64 {
        squared_norm_slice(&self.coords)
    }
}

/// Metric and canonicalization backend for `S^D`.
///
/// `D` is the intrinsic dimension. The backend accepts and emits
/// [`SphericalPoint<D>`] values whose ambient coordinate length is `D + 1`.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::topology::spaces::{SphericalMetric, SphericalPoint};
/// use std::f64::consts::FRAC_PI_2;
///
/// let metric = SphericalMetric::<2>::unit();
/// let x = SphericalPoint::<2>::try_new([1.0, 0.0, 0.0])?;
/// let y = SphericalPoint::<2>::try_new([0.0, 1.0, 0.0])?;
/// assert!((metric.try_distance(&x, &y)? - FRAC_PI_2).abs() < 1e-12);
/// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalMetric<const D: usize> {
    radius: f64,
}

impl<const D: usize> SphericalMetric<D> {
    /// Creates a unit-radius spherical metric.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalMetric;
    ///
    /// let metric = SphericalMetric::<2>::unit();
    ///
    /// assert_eq!(metric.radius(), 1.0);
    /// assert_eq!(metric.ambient_dimension(), 3);
    /// ```
    #[must_use]
    pub const fn unit() -> Self {
        Self { radius: 1.0 }
    }

    /// Creates a spherical metric with explicit radius.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalPointError::InvalidRadius`] when `radius` is not
    /// finite and positive.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalMetric;
    ///
    /// let metric = SphericalMetric::<3>::try_new(2.0)?;
    ///
    /// assert_eq!(metric.radius(), 2.0);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    pub fn try_new(radius: f64) -> Result<Self, SphericalPointError> {
        validate_radius(radius)?;
        Ok(Self { radius })
    }

    /// Returns the sphere radius.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalMetric;
    ///
    /// assert_eq!(SphericalMetric::<2>::try_new(1.5)?.radius(), 1.5);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    #[must_use]
    pub const fn radius(self) -> f64 {
        self.radius
    }

    /// Returns the ambient coordinate dimension `D + 1`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalMetric;
    ///
    /// assert_eq!(SphericalMetric::<3>::unit().ambient_dimension(), 4);
    /// ```
    #[must_use]
    pub const fn ambient_dimension(self) -> usize {
        D + 1
    }

    /// Canonicalizes raw ambient coordinates onto this metric's radius.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalPointError`] when the coordinates cannot represent a
    /// finite nonzero point in `R^(D+1)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalMetric;
    ///
    /// let metric = SphericalMetric::<2>::try_new(2.0)?;
    /// let point = metric.canonicalize([0.0, 0.0, 5.0])?;
    ///
    /// assert_eq!(point.coords(), &[0.0, 0.0, 2.0]);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    pub fn canonicalize<const N: usize>(
        self,
        coords: [f64; N],
    ) -> Result<SphericalPoint<D>, SphericalPointError> {
        SphericalPoint::try_new_with_radius(coords, self.radius)
    }

    /// Canonicalizes a raw ambient coordinate slice onto this metric's radius.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalPointError`] when the slice cannot represent a finite
    /// nonzero point in `R^(D+1)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use delaunay::prelude::topology::spaces::SphericalMetric;
    ///
    /// let metric = SphericalMetric::<2>::unit();
    /// let raw = [3.0, 4.0, 0.0];
    /// let point = metric.canonicalize_slice(&raw)?;
    ///
    /// assert!((point.coords()[1] - 0.8).abs() < 1e-12);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    pub fn canonicalize_slice(
        self,
        coords: &[f64],
    ) -> Result<SphericalPoint<D>, SphericalPointError> {
        SphericalPoint::try_from_slice_with_radius(coords, self.radius)
    }

    /// Returns the geodesic distance between two points on this sphere.
    ///
    /// The returned arc length is in the same units as this metric's radius and
    /// lies in `[0, pi * radius]` up to floating-point roundoff.
    ///
    /// # Errors
    ///
    /// Returns [`SphericalPointError::MismatchedRadius`] when either point was
    /// canonicalized for a different radius than this metric. Returns
    /// [`SphericalPointError::NonFiniteDistance`] when the arc length is not
    /// representable as a finite `f64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// use delaunay::prelude::topology::spaces::{SphericalMetric, SphericalPoint};
    ///
    /// let metric = SphericalMetric::<2>::unit();
    /// let x = SphericalPoint::<2>::try_new([1.0, 0.0, 0.0])?;
    /// let y = SphericalPoint::<2>::try_new([0.0, 1.0, 0.0])?;
    ///
    /// assert!((metric.try_distance(&x, &y)? - FRAC_PI_2).abs() < 1e-12);
    /// # Ok::<(), delaunay::prelude::topology::spaces::SphericalPointError>(())
    /// ```
    pub fn try_distance(
        self,
        a: &SphericalPoint<D>,
        b: &SphericalPoint<D>,
    ) -> Result<f64, SphericalPointError> {
        for point in [a, b] {
            if point.radius().to_bits() != self.radius.to_bits() {
                return Err(SphericalPointError::MismatchedRadius {
                    expected: self.radius,
                    actual: point.radius(),
                });
            }
        }
        let inverse_radius = 1.0 / self.radius;
        let cosine = a
            .coords()
            .iter()
            .zip(b.coords().iter())
            .fold(0.0, |acc, (&left, &right)| {
                (left * inverse_radius).mul_add(right * inverse_radius, acc)
            });
        let distance = self.radius * cosine.clamp(-1.0, 1.0).acos();
        if distance.is_finite() {
            Ok(distance)
        } else {
            Err(SphericalPointError::NonFiniteDistance { distance })
        }
    }
}

/// Errors emitted while constructing or canonicalizing spherical points.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum SphericalPointError {
    /// Raw coordinate arity did not match `D + 1`.
    #[error("spherical S^{dimension} points require {expected} ambient coordinates, got {actual}")]
    InvalidAmbientCoordinateCount {
        /// Intrinsic sphere dimension.
        dimension: usize,
        /// Required ambient coordinate count.
        expected: usize,
        /// Actual coordinate count.
        actual: usize,
    },

    /// Radius was not finite and positive.
    #[error("spherical radius must be finite and positive, got {radius:?}")]
    InvalidRadius {
        /// Invalid radius.
        radius: f64,
    },

    /// A coordinate was non-finite.
    #[error("non-finite spherical coordinate {value:?} at ambient axis {axis}")]
    NonFiniteCoordinate {
        /// Ambient coordinate axis.
        axis: usize,
        /// Non-finite coordinate value.
        value: f64,
    },

    /// The raw vector had zero norm and cannot be projected onto a sphere.
    #[error("cannot normalize zero-length vector onto a sphere")]
    ZeroNorm,

    /// Scaling the coordinate vector produced a non-finite norm.
    #[error("cannot normalize vector with non-finite scaled norm {norm:?}")]
    NonFiniteNorm {
        /// Non-finite norm.
        norm: f64,
    },

    /// Already-normalized points had different radii.
    #[error("spherical point radius {actual:?} does not match expected radius {expected:?}")]
    MismatchedRadius {
        /// Expected radius.
        expected: f64,
        /// Actual radius.
        actual: f64,
    },

    /// The requested geodesic distance is not representable as a finite `f64`.
    #[error("spherical geodesic distance is not finite: {distance:?}")]
    NonFiniteDistance {
        /// Non-finite computed distance.
        distance: f64,
    },
}

/// Converts validated spherical coordinates into an ambient fixed-size array.
///
/// Spherical points store their `D + 1` ambient coordinates in a vector because
/// stable const generics cannot expose `[f64; D + 1]` directly. This helper is
/// the checked bridge back to fixed-size arrays for ambient predicates and
/// convex-hull construction.
pub(crate) fn ambient_array_from_slice<const A: usize>(
    coords: &[f64],
) -> Result<[f64; A], SphericalPointError> {
    if coords.len() != A {
        return Err(SphericalPointError::InvalidAmbientCoordinateCount {
            dimension: A.saturating_sub(1),
            expected: A,
            actual: coords.len(),
        });
    }
    let mut out = [0.0; A];
    out.copy_from_slice(coords);
    Ok(out)
}

/// Validates that a radius can define a spherical metric.
///
/// Public point and metric constructors rely on this helper to keep every
/// stored sphere radius finite and strictly positive.
fn validate_radius(radius: f64) -> Result<(), SphericalPointError> {
    if radius.is_finite() && radius > 0.0 {
        return Ok(());
    }
    Err(SphericalPointError::InvalidRadius { radius })
}

/// Normalizes a finite nonzero coordinate slice onto radius `radius`.
///
/// This is the shared canonicalization path for the spherical coordinate and
/// metric backend. It classifies user-visible normalization failures while
/// scaling by the largest coordinate magnitude so finite vectors can be
/// projected without avoidable overflow or underflow.
fn normalize_coordinates(coords: &mut [f64], radius: f64) -> Result<(), SphericalPointError> {
    let mut max_abs = 0.0_f64;
    for (axis, coord) in coords.iter().copied().enumerate() {
        if !coord.is_finite() {
            return Err(SphericalPointError::NonFiniteCoordinate { axis, value: coord });
        }
        max_abs = max_abs.max(coord.abs());
    }
    if max_abs == 0.0 {
        return Err(SphericalPointError::ZeroNorm);
    }

    let sum_scaled_squares = coords.iter().fold(0.0, |acc, &coord| {
        let scaled = coord / max_abs;
        scaled.mul_add(scaled, acc)
    });
    let scale = sum_scaled_squares.sqrt();
    if !scale.is_finite() {
        return Err(SphericalPointError::NonFiniteNorm { norm: scale });
    }

    for coord in coords {
        *coord = ((*coord / max_abs) / scale) * radius;
    }
    Ok(())
}

/// Computes squared Euclidean norm for stored ambient coordinates.
///
/// This backs [`SphericalPoint::squared_norm`], which validation uses to check
/// that normalized points still lie on the configured sphere.
fn squared_norm_slice(coords: &[f64]) -> f64 {
    coords
        .iter()
        .fold(0.0, |acc, &coord| coord.mul_add(coord, acc))
}

/// Projects finite, nonzero coordinate vectors onto the unit sphere.
///
/// Returns `true` when the slice was normalized in place. Returns `false`
/// without modifying `coords` when any coordinate is non-finite or the vector
/// has zero length. The implementation scales by the largest absolute
/// coordinate before summing squares so very large and very small finite vectors
/// can be normalized without overflow or underflow.
///
/// This helper is crate-private glue for [`crate::topology::traits::GlobalTopology::Spherical`]
/// behavior-model paths that still operate on fixed-size coordinate arrays.
/// New spherical Delaunay construction should use [`SphericalPoint`] and
/// [`SphericalMetric`] directly so the intrinsic dimension `D` and ambient
/// dimension `D + 1` stay explicit.
pub(crate) fn normalize_unit_sphere_coordinates(coords: &mut [f64]) -> bool {
    normalize_coordinates(coords, 1.0).is_ok()
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_1_SQRT_2;

    use approx::assert_relative_eq;

    use super::*;

    fn squared_norm(coords: &[f64]) -> f64 {
        coords
            .iter()
            .fold(0.0, |acc, &coord| coord.mul_add(coord, acc))
    }

    fn assert_unit_norm(coords: &[f64]) {
        assert_relative_eq!(squared_norm(coords), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn spherical_point_handles_near_zero_vector() {
        let point = SphericalPoint::<2>::try_new([f64::MIN_POSITIVE, f64::MIN_POSITIVE, 0.0])
            .expect("finite nonzero ambient vector should normalize");
        let coords = point.coords();
        assert_relative_eq!(coords[0], FRAC_1_SQRT_2);
        assert_relative_eq!(coords[1], FRAC_1_SQRT_2);
        assert_relative_eq!(coords[2], 0.0);
        assert_unit_norm(coords);
    }

    #[test]
    fn spherical_point_handles_large_vector_without_overflow() {
        let point = SphericalPoint::<2>::try_new([f64::MAX, -f64::MAX, 0.0])
            .expect("finite nonzero ambient vector should normalize");
        let coords = point.coords();
        assert_relative_eq!(coords[0], FRAC_1_SQRT_2);
        assert_relative_eq!(coords[1], -FRAC_1_SQRT_2);
        assert_relative_eq!(coords[2], 0.0);
        assert_unit_norm(coords);
    }

    #[test]
    fn spherical_distance_uses_scaled_coordinates_for_large_radius() {
        let radius = f64::MAX / 4.0;
        let metric = SphericalMetric::<2>::try_new(radius)
            .expect("large finite radius should define a metric");
        let x = SphericalPoint::<2>::try_new_with_radius([1.0, 0.0, 0.0], radius)
            .expect("axis point should normalize onto large radius");
        let diagonal = SphericalPoint::<2>::try_new_with_radius([1.0, 1.0, 0.0], radius)
            .expect("diagonal point should normalize onto large radius");

        let distance = metric
            .try_distance(&x, &diagonal)
            .expect("large-radius dot product should avoid overflow");

        assert_relative_eq!(distance / radius, std::f64::consts::FRAC_PI_4);
    }

    #[test]
    fn spherical_distance_rejects_unrepresentable_arc_length() {
        let metric = SphericalMetric::<2>::try_new(f64::MAX)
            .expect("maximum finite radius should define a metric");
        let x = SphericalPoint::<2>::try_new_with_radius([1.0, 0.0, 0.0], f64::MAX)
            .expect("axis point should normalize onto maximum radius");
        let opposite = SphericalPoint::<2>::try_new_with_radius([-1.0, 0.0, 0.0], f64::MAX)
            .expect("opposite point should normalize onto maximum radius");

        assert!(matches!(
            metric.try_distance(&x, &opposite),
            Err(SphericalPointError::NonFiniteDistance { distance }) if distance.is_infinite()
        ));
    }

    #[test]
    fn ambient_array_from_slice_checks_fixed_ambient_arity() {
        let coords = ambient_array_from_slice::<3>(&[1.0, 2.0, 3.0])
            .expect("matching ambient arity should copy into a fixed array");
        assert_relative_eq!(coords[0], 1.0);
        assert_relative_eq!(coords[1], 2.0);
        assert_relative_eq!(coords[2], 3.0);

        let err = ambient_array_from_slice::<3>(&[1.0, 2.0])
            .expect_err("wrong ambient arity should remain typed");
        assert_eq!(
            err,
            SphericalPointError::InvalidAmbientCoordinateCount {
                dimension: 2,
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn normalize_unit_sphere_coordinates_preserves_failed_inputs() {
        let mut zero = [0.0, 0.0, 0.0];
        assert!(!normalize_unit_sphere_coordinates(&mut zero));
        assert_relative_eq!(zero[0], 0.0);
        assert_relative_eq!(zero[1], 0.0);
        assert_relative_eq!(zero[2], 0.0);

        let mut coords = [1.0, f64::INFINITY, 0.0];
        assert!(!normalize_unit_sphere_coordinates(&mut coords));
        assert_relative_eq!(coords[0], 1.0);
        assert!(coords[1].is_infinite());
        assert_relative_eq!(coords[2], 0.0);

        let mut coords = [1.0, f64::NAN, -2.0];
        assert!(!normalize_unit_sphere_coordinates(&mut coords));
        assert_relative_eq!(coords[0], 1.0);
        assert!(coords[1].is_nan());
        assert_relative_eq!(coords[2], -2.0);
    }
}
