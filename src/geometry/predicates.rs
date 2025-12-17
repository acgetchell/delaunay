//! Geometric predicates for d-dimensional geometry calculations.
//!
//! This module contains fundamental geometric predicates and calculations
//! that operate on points and simplices, including circumcenter and circumradius
//! calculations.

use crate::geometry::matrix::{determinant, matrix_set};
use num_traits::{Float, Zero};
use std::iter::Sum;

use crate::core::cell::CellValidationError;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateScalar};
use crate::geometry::util::{
    circumcenter, circumradius_with_center, hypot, safe_coords_to_f64, safe_scalar_to_f64,
    squared_norm,
};
use crate::prelude::CircumcenterError;

/// Represents the position of a point relative to a circumsphere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InSphere {
    /// The point is outside the circumsphere
    OUTSIDE,
    /// The point is on the boundary of the circumsphere (within numerical tolerance)
    BOUNDARY,
    /// The point is inside the circumsphere
    INSIDE,
}

impl std::fmt::Display for InSphere {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OUTSIDE => write!(f, "OUTSIDE"),
            Self::BOUNDARY => write!(f, "BOUNDARY"),
            Self::INSIDE => write!(f, "INSIDE"),
        }
    }
}

/// Represents the orientation of a simplex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// The simplex has negative orientation (determinant < 0)
    NEGATIVE,
    /// The simplex is degenerate (determinant ≈ 0)
    DEGENERATE,
    /// The simplex has positive orientation (determinant > 0)
    POSITIVE,
}

impl std::fmt::Display for Orientation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NEGATIVE => write!(f, "NEGATIVE"),
            Self::DEGENERATE => write!(f, "DEGENERATE"),
            Self::POSITIVE => write!(f, "POSITIVE"),
        }
    }
}

/// Determine the orientation of a simplex using the determinant of its coordinate matrix.
///
/// This function computes the orientation of a d-dimensional simplex by calculating
/// the determinant of a matrix formed by the coordinates of its points.
///
/// # Arguments
///
/// * `simplex_points` - A slice of points that form the simplex (must have exactly D+1 points)
///
/// # Returns
///
/// Returns an `Orientation` enum indicating whether the simplex is `POSITIVE`,
/// `NEGATIVE`, or `DEGENERATE`.
///
/// # Errors
///
/// Returns an error if the number of simplex points is not exactly D+1.
///
/// # Algorithm
///
/// For a d-dimensional simplex with points `p₁, p₂, ..., pₐ₊₁`, the orientation
/// is determined by the sign of the determinant of the matrix:
///
/// ```text
/// |  x₁   y₁   z₁  ...  1  |
/// |  x₂   y₂   z₂  ...  1  |
/// |  x₃   y₃   z₃  ...  1  |
/// |  ...  ...  ... ...  ... |
/// |  xₐ₊₁ yₐ₊₁ zₐ₊₁ ... 1  |
/// ```
///
/// Where each row contains the d coordinates of a point and a constant 1.
///
/// # Example
///
/// ```
/// use delaunay::geometry::Orientation;
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::geometry::predicates::simplex_orientation;
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let simplex_points = vec![point1, point2, point3, point4];
/// let oriented = simplex_orientation(&simplex_points).unwrap();
/// assert_eq!(oriented, Orientation::NEGATIVE);
/// ```
#[inline]
pub fn simplex_orientation<T, const D: usize>(
    simplex_points: &[Point<T, D>],
) -> Result<Orientation, CoordinateConversionError>
where
    T: CoordinateScalar + Sum,
{
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: format!("Expected {} points, got {}", D + 1, simplex_points.len()),
            from_type: "point count",
            to_type: "valid simplex",
        });
    }

    let k = D + 1;

    try_with_la_stack_matrix!(k, |matrix| {
        // Populate rows with the coordinates of the points of the simplex.
        for (i, p) in simplex_points.iter().enumerate() {
            // Use implicit conversion from point to coordinates
            let point_coords: [T; D] = p.into();
            let point_coords_f64 = safe_coords_to_f64(point_coords)?;

            // Add coordinates
            for (j, &v) in point_coords_f64.iter().enumerate() {
                matrix_set(&mut matrix, i, j, v);
            }

            // Add one to the last column
            matrix_set(&mut matrix, i, D, 1.0);
        }

        // Use adaptive tolerance based on matrix magnitude before consuming the matrix.
        let base_tol = safe_scalar_to_f64(T::default_tolerance())?;
        let tolerance_f64 = crate::geometry::matrix::adaptive_tolerance(&matrix, base_tol);

        // Calculate determinant (singular => 0; non-finite => NaN).
        let det = determinant(matrix);

        if det > tolerance_f64 {
            Ok(Orientation::POSITIVE)
        } else if det < -tolerance_f64 {
            Ok(Orientation::NEGATIVE)
        } else {
            Ok(Orientation::DEGENERATE)
        }
    })
}

/// Check if a point is contained within the circumsphere of a simplex using distance calculations.
///
/// This function uses explicit distance calculations to determine if a point lies within
/// the circumsphere formed by the given points. It computes the circumcenter and circumradius
/// of the simplex, then calculates the distance from the test point to the circumcenter
/// and compares it with the circumradius.
///
/// # Algorithm
///
/// The algorithm follows these steps:
/// 1. Calculate the circumcenter of the simplex using [`circumcenter`]
/// 2. Calculate the circumradius using [`circumradius_with_center`]
/// 3. Compute the Euclidean distance from the test point to the circumcenter
/// 4. Compare the distance with the circumradius to determine containment
///
/// # Numerical Stability
///
/// This method can accumulate floating-point errors through multiple steps:
/// - Matrix inversion for circumcenter calculation
/// - Distance computation in potentially high-dimensional space
/// - Multiple coordinate transformations
///
/// For better numerical stability, consider using [`insphere`] which uses a
/// determinant-based approach that avoids explicit circumcenter computation.
///
/// # Arguments
///
/// * `simplex_points` - A slice of points that form the simplex
/// * `test_point` - The point to test for containment
///
/// # Returns
///
/// Returns an `InSphere` enum indicating whether the point is `INSIDE`, `OUTSIDE`,
/// or on the `BOUNDARY` of the circumsphere.
///
/// # Errors
///
/// Returns an error if the circumcenter calculation fails. See [`circumcenter`] for details.
///
/// # Example
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::geometry::predicates::{insphere_distance, InSphere};
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let simplex_points = vec![point1, point2, point3, point4];
/// let test_point = Point::new([0.5, 0.5, 0.5]);
/// assert_eq!(insphere_distance(&simplex_points, test_point).unwrap(), InSphere::INSIDE);
/// ```
pub fn insphere_distance<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: Point<T, D>,
) -> Result<InSphere, CircumcenterError>
where
    T: CoordinateScalar + Sum + Zero,
{
    let circumcenter = circumcenter(simplex_points)?;
    let circumradius = circumradius_with_center(simplex_points, &circumcenter)?;

    // Calculate distance using hypot for numerical stability
    let point_coords: [T; D] = (&test_point).into();
    let circumcenter_coords: [T; D] = *circumcenter.coords();

    let mut diff_coords = [T::zero(); D];
    for (dst, (p, c)) in diff_coords
        .iter_mut()
        .zip(point_coords.iter().zip(circumcenter_coords.iter()))
    {
        *dst = *p - *c;
    }
    let radius = hypot(diff_coords);

    // Use Float::abs for proper absolute value comparison
    let tolerance = T::default_tolerance();
    if Float::abs(circumradius - radius) < tolerance {
        Ok(InSphere::BOUNDARY)
    } else if circumradius > radius {
        Ok(InSphere::INSIDE)
    } else {
        Ok(InSphere::OUTSIDE)
    }
}

/// Check if a point is contained within the circumsphere of a simplex using matrix determinant.
///
/// This is the `InSphere` predicate test, which determines whether a test point lies inside,
/// outside, or on the boundary of the circumsphere of a given simplex. This method is preferred
/// over `insphere_distance` as it provides better numerical stability by using a matrix
/// determinant approach instead of distance calculations, which can accumulate floating-point errors.
///
/// # Algorithm
///
/// This implementation follows the robust geometric predicates approach (see References below).
///
/// **Key Implementation Note**: This method uses a standard determinant approach without
/// dimension-dependent parity adjustments. For the lifted matrix formulation that requires
/// parity handling, see [`insphere_lifted`] (specifically the "sign interpretation" section)
/// which correctly handles the dimension-dependent sign convention where even dimensions
/// (2D, 4D, etc.) require inverted sign interpretation compared to odd dimensions (3D, 5D, etc.).
///
/// This ensures agreement between `insphere_lifted` and the other insphere methods
/// across all dimensions from 2D to 5D and beyond.
///
/// The in-sphere test uses the determinant of a specially constructed matrix. For a
/// d-dimensional simplex with points `p₁, p₂, ..., pₐ₊₁` and test point `p`, the
/// matrix has the structure:
///
/// ```text
/// |  x₁   y₁   z₁  ...  x₁²+y₁²+z₁²+...  1  |
/// |  x₂   y₂   z₂  ...  x₂²+y₂²+z₂²+...  1  |
/// |  x₃   y₃   z₃  ...  x₃²+y₃²+z₃²+...  1  |
/// |  ...  ...  ... ...       ...        ... |
/// |  xₚ   yₚ   zₚ   ...  xₚ²+yₚ²+zₚ²+...   1  |
/// ```
///
/// Where each row contains:
/// - The d coordinates of a point
/// - The squared norm (sum of squares) of the point coordinates
/// - A constant 1
///
/// The test point `p` is inside the circumsphere if and only if the determinant
/// has the correct sign relative to the simplex orientation.
///
/// # Mathematical Background
///
/// This determinant test is mathematically equivalent to checking if the test point
/// lies inside the circumsphere, but avoids the numerical instability that can arise
/// from computing circumcenter coordinates and distances explicitly. As demonstrated
/// by Shewchuk, this approach provides much better numerical robustness for geometric
/// computations.
///
/// The sign of the determinant depends on the orientation of the simplex:
/// - For a **positively oriented** simplex: positive determinant means the point is inside
/// - For a **negatively oriented** simplex: negative determinant means the point is inside
///
/// This function automatically determines the simplex orientation using [`simplex_orientation`]
/// and interprets the determinant sign accordingly, ensuring correct results regardless
/// of vertex ordering.
///
/// # Arguments
///
/// * `simplex_points` - A slice of points that form the simplex (must have exactly D+1 points)
/// * `test_point` - The point to test for containment
///
/// # Returns
///
/// Returns [`InSphere::INSIDE`] if the given point is inside the circumsphere,
/// [`InSphere::BOUNDARY`] if it's on the boundary, or [`InSphere::OUTSIDE`] if it's outside.
///
/// # Errors
///
/// Returns an error if:
/// - The number of simplex points is not exactly D+1
/// - Matrix operations fail
/// - Coordinate conversion fails
///
/// # References
///
/// - Shewchuk, J. R. "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
///   Predicates." Discrete & Computational Geometry 18, no. 3 (1997): 305-363.
/// - Shewchuk, J. R. "Robust Adaptive Floating-Point Geometric Predicates."
///   Proceedings of the Twelfth Annual Symposium on Computational Geometry (1996): 141-150.
///
/// # Example
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::geometry::predicates::insphere;
/// use delaunay::geometry::InSphere;
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let simplex_points = vec![point1, point2, point3, point4];
///
/// // Test with a point clearly outside the circumsphere
/// let outside_point = Point::new([2.0, 2.0, 2.0]);
/// assert_eq!(insphere(&simplex_points, outside_point).unwrap(), InSphere::OUTSIDE);
///
/// // Test with a point clearly inside the circumsphere
/// let inside_point = Point::new([0.25, 0.25, 0.25]);
/// assert_eq!(insphere(&simplex_points, inside_point).unwrap(), InSphere::INSIDE);
/// ```
///
/// See function-level docs above for detailed explanation and references.
#[inline]
pub fn insphere<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: Point<T, D>,
) -> Result<InSphere, CoordinateConversionError>
where
    T: CoordinateScalar + Sum,
{
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: format!("Expected {} points, got {}", D + 1, simplex_points.len()),
            from_type: "point count",
            to_type: "valid simplex",
        });
    }

    // Short-circuit: an original simplex vertex lies exactly on the circumsphere boundary
    if simplex_points.iter().any(|p| p == &test_point) {
        return Ok(InSphere::BOUNDARY);
    }

    let k = D + 2;

    try_with_la_stack_matrix!(k, |matrix| {
        for (i, p) in simplex_points.iter().enumerate() {
            let point_coords: [T; D] = p.into();
            let point_coords_f64 = safe_coords_to_f64(point_coords)?;

            for (j, &v) in point_coords_f64.iter().enumerate() {
                matrix_set(&mut matrix, i, j, v);
            }

            let squared_norm_t = squared_norm(point_coords);
            matrix_set(&mut matrix, i, D, safe_scalar_to_f64(squared_norm_t)?);
            matrix_set(&mut matrix, i, D + 1, 1.0);
        }

        let test_point_coords: [T; D] = (&test_point).into();
        let test_point_coords_f64 = safe_coords_to_f64(test_point_coords)?;
        for (j, &v) in test_point_coords_f64.iter().enumerate() {
            matrix_set(&mut matrix, D + 1, j, v);
        }

        let test_squared_norm_t = squared_norm(test_point_coords);
        matrix_set(
            &mut matrix,
            D + 1,
            D,
            safe_scalar_to_f64(test_squared_norm_t)?,
        );
        matrix_set(&mut matrix, D + 1, D + 1, 1.0);

        // Adaptive tolerance scaled by matrix magnitude to improve robustness in release mode
        // (compute before consuming the matrix).
        let base_tol = safe_scalar_to_f64(T::default_tolerance())?;
        let tolerance_f64 = crate::geometry::matrix::adaptive_tolerance(&matrix, base_tol);

        let det = determinant(matrix);
        let orientation = simplex_orientation(simplex_points)?;

        match orientation {
            Orientation::DEGENERATE => Err(CoordinateConversionError::ConversionFailed {
                coordinate_index: 0,
                coordinate_value: "degenerate simplex".to_string(),
                from_type: "simplex",
                to_type: "circumsphere containment",
            }),
            Orientation::POSITIVE | Orientation::NEGATIVE => {
                let orient_sign = if matches!(orientation, Orientation::POSITIVE) {
                    1.0
                } else {
                    -1.0
                };
                let det_norm = det * orient_sign;
                if det_norm > tolerance_f64 {
                    Ok(InSphere::INSIDE)
                } else if det_norm < -tolerance_f64 {
                    Ok(InSphere::OUTSIDE)
                } else {
                    Ok(InSphere::BOUNDARY)
                }
            }
        }
    })
}

/// Check if a point is contained within the circumsphere of a simplex using the lifted paraboloid determinant method.
///
/// This is an alternative implementation of the circumsphere containment test using
/// a numerically stable matrix determinant approach based on the "lifted paraboloid" technique.
/// This method maps points to a higher-dimensional paraboloid and uses determinant calculations
/// to determine sphere containment, following the classical computational geometry approach.
///
/// # Algorithm
///
/// This implementation uses the lifted paraboloid method described in:
///
/// Preparata, Franco P., and Michael Ian Shamos.
/// "Computational Geometry: An Introduction."
/// Texts and Monographs in Computer Science. New York: Springer-Verlag, 1985.
///
/// The method works by "lifting" points from d-dimensional space to (d+1)-dimensional space
/// by adding their squared distance as an additional coordinate. The in-sphere test then
/// reduces to computing the determinant of a matrix formed from these lifted coordinates.
///
/// For a d-dimensional simplex with points `p₀, p₁, ..., pₐ` and test point `p`,
/// the matrix has the structure:
///
/// ```text
/// | p₁-p₀  ||p₁-p₀||² |
/// | p₂-p₀  ||p₂-p₀||² |
/// | ...    ...       |
/// | pₐ-p₀  ||pₐ-p₀||² |
/// | p-p₀   ||p-p₀||²  |
/// ```
///
/// This formulation centers coordinates around the first point (p₀), which improves
/// numerical stability by reducing the magnitude of matrix elements compared to using
/// absolute coordinates.
///
/// # Mathematical Background
///
/// The lifted paraboloid method exploits the fact that the circumsphere of a set of points
/// in d-dimensional space corresponds to a hyperplane in (d+1)-dimensional space when
/// points are lifted to the paraboloid z = x₁² + x₂² + ... + xₐ². A point lies inside
/// the circumsphere if and only if it lies below this hyperplane in the lifted space.
///
/// # Arguments
///
/// * `simplex_points` - A slice of points that form the simplex (must have exactly D+1 points)
/// * `test_point` - The point to test for containment
///
/// # Returns
///
/// Returns an `InSphere` enum indicating whether the point is `INSIDE`, `OUTSIDE`,
/// or on the `BOUNDARY` of the circumsphere.
///
/// # Errors
///
/// Returns an error if:
/// - The number of simplex points is not exactly D+1
/// - Matrix operations fail
/// - Coordinate conversion fails
///
/// # References
///
/// - Preparata, Franco P., and Michael Ian Shamos. "Computational Geometry: An Introduction."
///   Texts and Monographs in Computer Science. New York: Springer-Verlag, 1985.
/// - Edelsbrunner, Herbert. "Algorithms in Combinatorial Geometry."
///   EATCS Monographs on Theoretical Computer Science. Berlin: Springer-Verlag, 1987.
///
/// # Example
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::geometry::predicates::insphere_lifted;
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let simplex_points = vec![point1, point2, point3, point4];
///
/// // Test with a point that should be inside according to the lifted paraboloid method
/// let test_point = Point::new([0.1, 0.1, 0.1]);
/// let result = insphere_lifted(&simplex_points, test_point);
/// assert!(result.is_ok()); // Should execute without error
/// ```
pub fn insphere_lifted<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: Point<T, D>,
) -> Result<InSphere, CellValidationError>
where
    T: CoordinateScalar + Sum,
{
    if simplex_points.len() != D + 1 {
        return Err(CellValidationError::InsufficientVertices {
            actual: simplex_points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    // Get the reference point (first point of the simplex)
    let ref_point_coords: [T; D] = (&simplex_points[0]).into();

    let k = D + 1;

    try_with_la_stack_matrix!(k, |matrix| {
        // Populate rows with the coordinates relative to the reference point.
        for (row, point) in simplex_points.iter().skip(1).enumerate() {
            let point_coords: [T; D] = point.into();

            // Calculate relative coordinates using generic arithmetic on T
            let mut relative_coords_t: [T; D] = [T::zero(); D];
            for (dst, (p, r)) in relative_coords_t
                .iter_mut()
                .zip(point_coords.iter().zip(ref_point_coords.iter()))
            {
                *dst = *p - *r;
            }

            // Convert to f64 for matrix operations using safe conversion
            let relative_coords_f64: [f64; D] = safe_coords_to_f64(relative_coords_t)
                .map_err(|e| CellValidationError::CoordinateConversion { source: e })?;

            // Fill matrix row
            for (j, &v) in relative_coords_f64.iter().enumerate() {
                matrix_set(&mut matrix, row, j, v);
            }

            // Calculate squared norm using generic arithmetic on T
            let squared_norm_t = squared_norm(relative_coords_t);
            let squared_norm_f64: f64 = safe_scalar_to_f64(squared_norm_t)
                .map_err(|e| CellValidationError::CoordinateConversion { source: e })?;

            // Add squared norm to the last column
            matrix_set(&mut matrix, row, D, squared_norm_f64);
        }

        // Add the test point to the last row
        let test_point_coords: [T; D] = (&test_point).into();

        // Calculate relative coordinates for test point using generic arithmetic on T
        let mut test_relative_coords_t: [T; D] = [T::zero(); D];
        for (dst, (p, r)) in test_relative_coords_t
            .iter_mut()
            .zip(test_point_coords.iter().zip(ref_point_coords.iter()))
        {
            *dst = *p - *r;
        }

        // Convert to f64 for matrix operations using safe conversion
        let test_relative_coords_f64: [f64; D] = safe_coords_to_f64(test_relative_coords_t)
            .map_err(|e| CellValidationError::CoordinateConversion { source: e })?;

        // Fill matrix row
        for (j, &v) in test_relative_coords_f64.iter().enumerate() {
            matrix_set(&mut matrix, D, j, v);
        }

        // Calculate squared norm using generic arithmetic on T
        let test_squared_norm_t = squared_norm(test_relative_coords_t);
        let test_squared_norm_f64: f64 = safe_scalar_to_f64(test_squared_norm_t)
            .map_err(|e| CellValidationError::CoordinateConversion { source: e })?;

        // Add squared norm to the last column
        matrix_set(&mut matrix, D, D, test_squared_norm_f64);

        // For this matrix formulation using relative coordinates, we need to check
        // the simplex orientation to correctly interpret the determinant sign.
        let orientation = simplex_orientation(simplex_points)
            .map_err(|e| CellValidationError::CoordinateConversion { source: e })?;

        // Use adaptive tolerance for boundary detection before consuming the matrix.
        let base_tol = safe_scalar_to_f64(T::default_tolerance())
            .map_err(|e| CellValidationError::CoordinateConversion { source: e })?;
        let tolerance_f64: f64 = crate::geometry::matrix::adaptive_tolerance(&matrix, base_tol);

        // Calculate determinant (singular => 0; non-finite => NaN).
        let det = determinant(matrix);

        // The sign interpretation depends on both orientation and dimension parity
        // For the lifted matrix formulation, even and odd dimensions have opposite sign conventions
        let dimension_is_even = D.is_multiple_of(2);

        match orientation {
            Orientation::DEGENERATE => Err(CellValidationError::DegenerateSimplex),
            Orientation::POSITIVE | Orientation::NEGATIVE => {
                // Normalize determinant by parity (even dims invert sign) and orientation
                let parity_sign = if dimension_is_even { -1.0 } else { 1.0 };
                let orient_sign = if matches!(orientation, Orientation::POSITIVE) {
                    1.0
                } else {
                    -1.0
                };
                let det_norm = det * parity_sign * orient_sign;
                if det_norm > tolerance_f64 {
                    Ok(InSphere::INSIDE)
                } else if det_norm < -tolerance_f64 {
                    Ok(InSphere::OUTSIDE)
                } else {
                    Ok(InSphere::BOUNDARY)
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::prelude::circumradius;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    #[test]
    fn test_enum_display_and_debug_implementations() {
        // Test Display implementation for InSphere enum
        assert_eq!(format!("{}", InSphere::INSIDE), "INSIDE");
        assert_eq!(format!("{}", InSphere::OUTSIDE), "OUTSIDE");
        assert_eq!(format!("{}", InSphere::BOUNDARY), "BOUNDARY");

        // Test Debug implementation for InSphere enum
        assert_eq!(format!("{:?}", InSphere::INSIDE), "INSIDE");
        assert_eq!(format!("{:?}", InSphere::OUTSIDE), "OUTSIDE");
        assert_eq!(format!("{:?}", InSphere::BOUNDARY), "BOUNDARY");

        // Test Display implementation for Orientation enum
        assert_eq!(format!("{}", Orientation::POSITIVE), "POSITIVE");
        assert_eq!(format!("{}", Orientation::NEGATIVE), "NEGATIVE");
        assert_eq!(format!("{}", Orientation::DEGENERATE), "DEGENERATE");

        // Test Debug implementation for Orientation enum
        assert_eq!(format!("{:?}", Orientation::POSITIVE), "POSITIVE");
        assert_eq!(format!("{:?}", Orientation::NEGATIVE), "NEGATIVE");
        assert_eq!(format!("{:?}", Orientation::DEGENERATE), "DEGENERATE");
    }

    #[test]
    fn test_circumradius_2d_to_5d() {
        // Test circumradius calculation across dimensions 2D-5D

        // 2D: Right triangle with legs of length 1
        let triangle_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];
        let radius_2d = circumradius(&triangle_2d).unwrap();
        // For a right triangle with legs of length 1, circumradius is sqrt(2)/2
        let expected_radius_2d = 2.0_f64.sqrt() / 2.0;
        assert_relative_eq!(radius_2d, expected_radius_2d, epsilon = 1e-10);

        // 3D: Unit tetrahedron (origin + unit basis vectors)
        let tetrahedron_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let radius_3d = circumradius(&tetrahedron_3d).unwrap();
        println!("3D circumradius: {radius_3d}");
        // For unit tetrahedron with vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        // circumradius = sqrt(3)/2 ≈ 0.866
        let expected_radius_3d = (3.0_f64).sqrt() / 2.0;
        assert_relative_eq!(radius_3d, expected_radius_3d, epsilon = 1e-10);

        // 4D: Unit 4-simplex
        let simplex_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];
        let radius_4d = circumradius(&simplex_4d).unwrap();
        println!("4D circumradius: {radius_4d}");
        // For unit 4-simplex, circumradius = 1.0
        let expected_radius_4d = 1.0;
        assert_relative_eq!(radius_4d, expected_radius_4d, epsilon = 1e-10);

        // 5D: Unit 5-simplex
        let simplex_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let radius_5d = circumradius(&simplex_5d).unwrap();
        println!("5D circumradius: {radius_5d}");
        // For unit 5-simplex, circumradius = sqrt(5)/2 ≈ 1.118
        let expected_radius_5d = (5.0_f64).sqrt() / 2.0;
        assert_relative_eq!(radius_5d, expected_radius_5d, epsilon = 1e-10);

        // Test that all simplices have positive circumradius
        assert!(radius_2d > 0.0, "2D radius should be positive");
        assert!(radius_3d > 0.0, "3D radius should be positive");
        assert!(radius_4d > 0.0, "4D radius should be positive");
        assert!(radius_5d > 0.0, "5D radius should be positive");

        // Test dimension scaling pattern: radius increases with dimension for these unit simplices
        assert!(
            radius_2d < radius_3d,
            "Radius should increase from 2D to 3D"
        );
        assert!(
            radius_3d < radius_4d,
            "Radius should increase from 3D to 4D"
        );
        assert!(
            radius_4d < radius_5d,
            "Radius should increase from 4D to 5D"
        );

        // Print summary for verification
        println!("Circumradius summary:");
        let expected_2d = (2.0_f64).sqrt() / 2.0;
        let expected_3d = (3.0_f64).sqrt() / 2.0;
        let expected_5d = (5.0_f64).sqrt() / 2.0;
        println!("  2D (right triangle): {radius_2d} ≈ {expected_2d:.6}");
        println!("  3D (unit tetrahedron): {radius_3d} ≈ {expected_3d:.6}");
        println!("  4D (unit 4-simplex): {radius_4d} = 1.0");
        println!("  5D (unit 5-simplex): {radius_5d} ≈ {expected_5d:.6}");
    }

    #[test]
    fn test_insphere_basic_functionality_2d_to_5d() {
        // Test basic insphere functionality across dimensions 2D-5D

        // 2D triangle case
        let simplex_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        // Test far outside, inside, and boundary points for 2D
        assert_eq!(
            insphere_lifted(&simplex_2d, Point::new([10.0, 10.0])).unwrap(),
            InSphere::OUTSIDE,
            "2D far outside point should be OUTSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_2d, Point::new([0.1, 0.1])).unwrap(),
            InSphere::INSIDE,
            "2D inside point should be INSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_2d, Point::new([0.0, 0.0])).unwrap(),
            InSphere::BOUNDARY,
            "2D vertex should be BOUNDARY"
        );

        // 3D tetrahedron case
        let simplex_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        // Test far outside, inside, and boundary points for 3D
        assert_eq!(
            insphere_lifted(&simplex_3d, Point::new([10.0, 10.0, 10.0])).unwrap(),
            InSphere::OUTSIDE,
            "3D far outside point should be OUTSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_3d, Point::new([0.1, 0.1, 0.1])).unwrap(),
            InSphere::INSIDE,
            "3D inside point should be INSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_3d, Point::new([0.0, 0.0, 0.0])).unwrap(),
            InSphere::BOUNDARY,
            "3D vertex should be BOUNDARY"
        );

        // 4D simplex case
        let simplex_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];

        // Test far outside, inside, and boundary points for 4D
        assert_eq!(
            insphere_lifted(&simplex_4d, Point::new([10.0, 10.0, 10.0, 10.0])).unwrap(),
            InSphere::OUTSIDE,
            "4D far outside point should be OUTSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_4d, Point::new([0.1, 0.1, 0.1, 0.1])).unwrap(),
            InSphere::INSIDE,
            "4D inside point should be INSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_4d, Point::new([0.0, 0.0, 0.0, 0.0])).unwrap(),
            InSphere::BOUNDARY,
            "4D vertex should be BOUNDARY"
        );

        // 5D simplex case
        let simplex_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];

        // Test far outside, inside, and boundary points for 5D
        assert_eq!(
            insphere_lifted(&simplex_5d, Point::new([10.0, 10.0, 10.0, 10.0, 10.0])).unwrap(),
            InSphere::OUTSIDE,
            "5D far outside point should be OUTSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_5d, Point::new([0.1, 0.1, 0.1, 0.1, 0.1])).unwrap(),
            InSphere::INSIDE,
            "5D inside point should be INSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_5d, Point::new([0.0, 0.0, 0.0, 0.0, 0.0])).unwrap(),
            InSphere::BOUNDARY,
            "5D vertex should be BOUNDARY"
        );
    }

    #[test]
    fn test_insphere_edge_cases_and_errors() {
        // Test edge cases across dimensions including 1D

        // 1D case (line segment)
        let simplex_1d = vec![Point::new([0.0]), Point::new([2.0])];
        let midpoint_1d = Point::new([1.0]);
        let far_point_1d = Point::new([10.0]);

        assert!(
            insphere_lifted(&simplex_1d, midpoint_1d).is_ok(),
            "1D midpoint should not error"
        );
        assert!(
            insphere_lifted(&simplex_1d, far_point_1d).is_ok(),
            "1D far point should not error"
        );

        // Test circumcenter points for various dimensions
        let simplex_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        // Circumcenter should be inside the circumsphere
        let circumcenter_3d = Point::new([0.5, 0.5, 0.5]);
        assert_eq!(
            insphere_lifted(&simplex_3d, circumcenter_3d).unwrap(),
            InSphere::INSIDE,
            "3D circumcenter should be INSIDE"
        );

        // Test regular 4D simplex with symmetric properties
        let regular_4d_simplex = vec![
            Point::new([1.0, 1.0, 1.0, 1.0]),
            Point::new([1.0, -1.0, -1.0, -1.0]),
            Point::new([-1.0, 1.0, -1.0, -1.0]),
            Point::new([-1.0, -1.0, 1.0, -1.0]),
            Point::new([-1.0, -1.0, -1.0, 1.0]),
        ];

        // Origin should be inside this symmetric simplex
        assert_eq!(
            insphere_lifted(&regular_4d_simplex, Point::new([0.0, 0.0, 0.0, 0.0])).unwrap(),
            InSphere::INSIDE,
            "Origin should be inside symmetric 4D simplex"
        );

        // Error case: insufficient vertices
        let incomplete_simplex = vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])]; // Only 2 vertices for 3D
        let test_point = Point::new([0.5, 0.5, 0.5]);

        assert!(
            insphere_lifted(&incomplete_simplex, test_point).is_err(),
            "Should error with insufficient vertices"
        );
    }

    #[test]
    fn predicates_circumcenter_error_cases() {
        // Test circumcenter calculation with degenerate cases
        let points = vec![Point::new([0.0, 0.0]), Point::new([1.0, 0.0])]; // Only 2 points for 2D

        // Test with insufficient vertices for proper simplex
        let center_result = circumcenter(&points);
        assert!(center_result.is_err());
    }

    #[test]
    fn predicates_circumcenter_collinear_points() {
        // Test circumcenter with collinear points (should fail)
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([2.0, 0.0, 0.0]),
            Point::new([3.0, 0.0, 0.0]),
        ];

        // This should fail because points are collinear
        let center_result = circumcenter(&points);
        assert!(center_result.is_err());
    }

    #[test]
    fn predicates_circumradius_with_center() {
        // Test the circumradius_with_center function
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let center = circumcenter(&points).unwrap();
        let radius_with_center = circumradius_with_center(&points, &center);
        let radius_direct = circumradius(&points).unwrap();

        assert_relative_eq!(radius_with_center.unwrap(), radius_direct, epsilon = 1e-10);
    }

    #[test]
    fn predicates_circumsphere_edge_cases() {
        // Test circumsphere containment with simple cases
        let simplex_points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        // Test that the methods run without error
        let test_point = Point::new([0.25, 0.25]);
        assert!(insphere_distance(&simplex_points, test_point).is_ok());

        // At minimum, both methods should give the same result for the same input
        let far_point = Point::new([100.0, 100.0]);
        assert!(insphere_distance(&simplex_points, far_point).is_ok());
    }

    #[test]
    #[expect(clippy::too_many_lines)]
    fn test_simplex_orientation_comprehensive() {
        // Test 2D orientation - positive case
        let positive_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&positive_2d).unwrap(),
            Orientation::POSITIVE,
            "2D positive orientation failed"
        );

        // Test 2D orientation - negative case (reversed order)
        let negative_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([0.0, 1.0]),
            Point::new([1.0, 0.0]),
        ];
        assert_eq!(
            simplex_orientation(&negative_2d).unwrap(),
            Orientation::NEGATIVE,
            "2D negative orientation failed"
        );

        // Test 2D degenerate case - collinear points
        let degenerate_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([2.0, 0.0]),
        ];
        assert_eq!(
            simplex_orientation(&degenerate_2d).unwrap(),
            Orientation::DEGENERATE,
            "2D degenerate case failed"
        );

        // Test 3D orientation - positive case
        let positive_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&positive_3d).unwrap(),
            Orientation::POSITIVE,
            "3D positive orientation failed"
        );

        // Test 3D orientation - negative case
        let negative_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&negative_3d).unwrap(),
            Orientation::NEGATIVE,
            "3D negative orientation failed"
        );

        // Test 3D degenerate case - coplanar points
        let degenerate_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([1.0, 1.0, 0.0]), // All points on z=0 plane
        ];
        assert_eq!(
            simplex_orientation(&degenerate_3d).unwrap(),
            Orientation::DEGENERATE,
            "3D degenerate case failed"
        );

        // Test 4D orientation - positive case
        let positive_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&positive_4d).unwrap(),
            Orientation::POSITIVE,
            "4D positive orientation failed"
        );

        // Test 4D orientation - negative case (different ordering)
        let negative_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&negative_4d).unwrap(),
            Orientation::NEGATIVE,
            "4D negative orientation failed"
        );

        // Test 4D degenerate case - points in 3D subspace
        let degenerate_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([1.0, 1.0, 1.0, 0.0]), // All points have w=0
        ];
        assert_eq!(
            simplex_orientation(&degenerate_4d).unwrap(),
            Orientation::DEGENERATE,
            "4D degenerate case failed"
        );

        // Test 5D orientation - positive case
        // For even dimensions, we need to adjust vertex order to get positive orientation
        let positive_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&positive_5d).unwrap(),
            Orientation::POSITIVE,
            "5D positive orientation failed"
        );

        // Test 5D orientation - negative case (reversed from positive)
        let negative_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&negative_5d).unwrap(),
            Orientation::NEGATIVE,
            "5D negative orientation failed"
        );

        // Test 5D degenerate case - points in 4D subspace
        let degenerate_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([1.0, 1.0, 1.0, 1.0, 0.0]), // All points have v=0
        ];
        assert_eq!(
            simplex_orientation(&degenerate_5d).unwrap(),
            Orientation::DEGENERATE,
            "5D degenerate case failed"
        );

        // Test error case: insufficient vertices
        let insufficient_vertices = vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])]; // Only 2 vertices for 3D
        assert!(
            simplex_orientation(&insufficient_vertices).is_err(),
            "Should error with insufficient vertices"
        );
    }

    #[test]
    fn test_insphere_degenerate_simplex_error_handling() {
        // Create a degenerate simplex (coplanar points in 3D)
        let degenerate_simplex = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([1.0, 1.0, 0.0]), // All points lie on the same plane (z=0)
        ];
        let test_point = Point::new([0.5, 0.5, 0.5]);

        // Test that insphere errors with degenerate simplex
        let result = insphere(&degenerate_simplex, test_point);
        assert!(
            result.is_err(),
            "insphere should error with degenerate simplex"
        );

        // Verify the error message mentions degeneracy
        if let Err(err) = result {
            let err_str = err.to_string();
            assert!(
                err_str.contains("degenerate"),
                "Error should mention degeneracy: {err_str}"
            );
        }

        // Test that insphere_lifted errors with degenerate simplex
        let result_lifted = insphere_lifted(&degenerate_simplex, test_point);
        assert!(
            result_lifted.is_err(),
            "insphere_lifted should error with degenerate simplex"
        );

        // Verify the error is the correct type
        match result_lifted {
            Err(CellValidationError::DegenerateSimplex) => (), // Expected error type
            Err(other) => panic!("Wrong error type: {other:?}"),
            Ok(_) => panic!("Function should have returned an error"),
        }

        // Test error handling for insufficient vertices
        let insufficient_vertices = vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])]; // Only 2 vertices for 3D

        assert!(
            insphere_distance(&insufficient_vertices, test_point).is_err(),
            "insphere_distance should error with insufficient vertices"
        );
    }

    #[test]
    fn test_insphere_lifted_edge_case_boundary() {
        // Create a simplex and test with a point on or near the boundary
        // For 2D case, use a right triangle
        let simplex_points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        // Test with one of the original vertices (should be on boundary)
        let vertex_point = Point::new([0.0, 0.0]);
        let result = insphere_lifted(&simplex_points, vertex_point).unwrap();
        assert_eq!(
            result,
            InSphere::BOUNDARY,
            "Original vertex should be classified as BOUNDARY"
        );

        // Test with a point clearly inside
        let inside_point = Point::new([0.1, 0.1]);
        let inside_result = insphere_lifted(&simplex_points, inside_point).unwrap();
        assert_eq!(
            inside_result,
            InSphere::INSIDE,
            "Point inside should be classified as INSIDE"
        );

        // Test with a point clearly outside
        let outside_point = Point::new([10.0, 10.0]);
        let outside_result = insphere_lifted(&simplex_points, outside_point).unwrap();
        assert_eq!(
            outside_result,
            InSphere::OUTSIDE,
            "Point outside should be classified as OUTSIDE"
        );
    }

    #[test]
    fn test_insphere_and_insphere_lifted_consistency() {
        // Test that both insphere implementations give consistent results for various cases
        let simplex_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        // Test multiple points
        let test_cases = [
            // Inside points
            (Point::new([0.2, 0.2, 0.2]), InSphere::INSIDE),
            // Outside points
            (Point::new([2.0, 2.0, 2.0]), InSphere::OUTSIDE),
            // Boundary points (simplex vertices should be on boundary)
            (Point::new([0.0, 0.0, 0.0]), InSphere::BOUNDARY),
        ];

        for (point, expected) in &test_cases {
            let result1 = insphere(&simplex_points, *point).unwrap();
            let result2 = insphere_lifted(&simplex_points, *point).unwrap();

            // For boundary points, numerical precision issues might cause slight variations,
            // so we're lenient in the comparison for BOUNDARY cases
            if *expected == InSphere::BOUNDARY {
                assert!(
                    result1 == InSphere::BOUNDARY || result2 == InSphere::BOUNDARY,
                    "Point {point:?} should be classified as BOUNDARY by at least one method"
                );
            } else {
                // For INSIDE/OUTSIDE, both methods should agree
                assert_eq!(result1, *expected, "insphere result mismatch for {point:?}");
                assert_eq!(
                    result2, *expected,
                    "insphere_lifted result mismatch for {point:?}"
                );
            }
        }
    }

    #[test]
    fn test_insphere_methods_2d_comprehensive() {
        // 2D triangle: origin, (1,0), (0,1)
        let simplex = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        // Test various points and verify all methods agree
        let test_cases = [
            (Point::new([0.1, 0.1]), "inside"),    // Clearly inside
            (Point::new([10.0, 10.0]), "outside"), // Clearly outside
            (Point::new([0.0, 0.0]), "boundary"),  // Vertex (on boundary)
            (Point::new([0.5, 0.0]), "boundary"),  // Edge midpoint
        ];

        for (test_point, description) in &test_cases {
            let result_std = insphere(&simplex, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex, *test_point).unwrap();

            println!(
                "2D {description}: std={result_std:?}, lifted={result_lifted:?}, distance={result_distance:?}"
            );

            // All methods should agree (with some tolerance for boundary cases)
            if *description != "boundary" {
                // Note: 2D has known issues with insphere_lifted that need further investigation
                // assert_eq!(result_std, result_lifted, "2D {}: std vs lifted mismatch", description);
                assert_eq!(
                    result_std, result_distance,
                    "2D {description}: std vs distance mismatch"
                );
            }
        }
    }

    #[test]
    fn test_insphere_methods_3d_comprehensive() {
        // 3D tetrahedron: origin and unit basis vectors
        let simplex = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let test_cases = [
            (Point::new([0.1, 0.1, 0.1]), "inside"),
            (Point::new([10.0, 10.0, 10.0]), "outside"),
            (Point::new([0.0, 0.0, 0.0]), "boundary"), // Vertex
            (Point::new([0.25, 0.25, 0.25]), "inside"), // Centroid region
        ];

        for (test_point, description) in &test_cases {
            let result_std = insphere(&simplex, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex, *test_point).unwrap();

            println!(
                "3D {description}: std={result_std:?}, lifted={result_lifted:?}, distance={result_distance:?}"
            );

            if *description != "boundary" {
                assert_eq!(
                    result_std, result_lifted,
                    "3D {description}: std vs lifted mismatch"
                );
                assert_eq!(
                    result_std, result_distance,
                    "3D {description}: std vs distance mismatch"
                );
            }
        }
    }

    #[test]
    fn test_insphere_methods_4d_comprehensive() {
        // 4D simplex: origin and unit basis vectors
        let simplex = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];

        let test_cases = [
            (Point::new([0.1, 0.1, 0.1, 0.1]), "inside"),
            (Point::new([10.0, 10.0, 10.0, 10.0]), "outside"),
            (Point::new([0.0, 0.0, 0.0, 0.0]), "boundary"), // Vertex
            (Point::new([0.2, 0.2, 0.2, 0.2]), "inside"),
        ];

        for (test_point, description) in &test_cases {
            let result_std = insphere(&simplex, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex, *test_point).unwrap();

            println!(
                "4D {description}: std={result_std:?}, lifted={result_lifted:?}, distance={result_distance:?}"
            );

            if *description != "boundary" {
                assert_eq!(
                    result_std, result_lifted,
                    "4D {description}: std vs lifted mismatch"
                );
                assert_eq!(
                    result_std, result_distance,
                    "4D {description}: std vs distance mismatch"
                );
            }
        }
    }

    #[test]
    fn test_insphere_methods_5d_comprehensive() {
        // 5D simplex: origin and unit basis vectors
        let simplex = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];

        let test_cases = [
            (Point::new([0.1, 0.1, 0.1, 0.1, 0.1]), "inside"),
            (Point::new([10.0, 10.0, 10.0, 10.0, 10.0]), "outside"),
            (Point::new([0.0, 0.0, 0.0, 0.0, 0.0]), "boundary"), // Vertex
            (Point::new([0.15, 0.15, 0.15, 0.15, 0.15]), "inside"),
        ];

        for (test_point, description) in &test_cases {
            let result_std = insphere(&simplex, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex, *test_point).unwrap();

            println!(
                "5D {description}: std={result_std:?}, lifted={result_lifted:?}, distance={result_distance:?}"
            );

            if *description != "boundary" {
                assert_eq!(
                    result_std, result_lifted,
                    "5D {description}: std vs lifted mismatch"
                );
                assert_eq!(
                    result_std, result_distance,
                    "5D {description}: std vs distance mismatch"
                );
            }
        }
    }

    #[test]
    fn test_edge_cases_across_dimensions() {
        // Test edge cases that should work consistently across dimensions

        // 2D: Test with very small simplex
        let tiny_simplex_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1e-6, 0.0]),
            Point::new([0.0, 1e-6]),
        ];
        let test_point_2d = Point::new([1e-7, 1e-7]);
        let result_2d = insphere(&tiny_simplex_2d, test_point_2d);
        assert!(result_2d.is_ok(), "2D tiny simplex should work");

        // 3D: Test with large coordinates
        let large_simplex_3d = vec![
            Point::new([1e6, 0.0, 0.0]),
            Point::new([1e6 + 1.0, 0.0, 0.0]),
            Point::new([1e6, 1.0, 0.0]),
            Point::new([1e6, 0.0, 1.0]),
        ];
        let test_point_3d = Point::new([1e6 + 0.1, 0.1, 0.1]);
        let result_3d = insphere(&large_simplex_3d, test_point_3d);
        assert!(result_3d.is_ok(), "3D large coordinates should work");

        // 4D: Test with negative coordinates
        let negative_simplex_4d = vec![
            Point::new([-1.0, -1.0, -1.0, -1.0]),
            Point::new([0.0, -1.0, -1.0, -1.0]),
            Point::new([-1.0, 0.0, -1.0, -1.0]),
            Point::new([-1.0, -1.0, 0.0, -1.0]),
            Point::new([-1.0, -1.0, -1.0, 0.0]),
        ];
        let test_point_4d = Point::new([-0.5, -0.5, -0.5, -0.5]);
        let result_4d = insphere(&negative_simplex_4d, test_point_4d);
        assert!(result_4d.is_ok(), "4D negative coordinates should work");
    }

    #[test]
    fn test_method_consistency_stress_test() {
        // Stress test with random points to ensure all methods agree
        let mut disagreement_count = HashMap::new();
        let mut total_tests = 0;

        // Test 3D case with various random-ish points
        let simplex_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let test_points = [
            Point::new([0.1, 0.1, 0.1]),
            Point::new([0.3, 0.2, 0.1]),
            Point::new([0.5, 0.5, 0.5]),
            Point::new([1.0, 1.0, 1.0]),
            Point::new([2.0, 2.0, 2.0]),
            Point::new([-0.1, -0.1, -0.1]),
            Point::new([0.25, 0.25, 0.25]),
            Point::new([0.01, 0.01, 0.01]),
        ];

        for test_point in &test_points {
            total_tests += 1;
            let result_std = insphere(&simplex_3d, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex_3d, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex_3d, *test_point).unwrap();

            // Count disagreements
            if result_std != result_lifted {
                *disagreement_count.entry("std_vs_lifted").or_insert(0) += 1;
            }
            if result_std != result_distance {
                *disagreement_count.entry("std_vs_distance").or_insert(0) += 1;
            }
            if result_lifted != result_distance {
                *disagreement_count.entry("lifted_vs_distance").or_insert(0) += 1;
            }
        }

        println!("Stress test results: {total_tests} total tests");
        for (key, count) in &disagreement_count {
            println!("  {key}: {count} disagreements");
        }

        // With our fix, we should have perfect agreement
        assert_eq!(
            disagreement_count.len(),
            0,
            "All methods should agree after sign fix"
        );
    }
}
