//! Geometric utility functions for d-dimensional geometry calculations.
//!
//! This module contains utility functions for computing distances, norms, and
//! circumsphere properties of geometric objects. These functions are used by
//! both predicates and other geometric algorithms.

use na::ComplexField;
use nalgebra as na;
use num_traits::{Float, Zero};
use peroxide::fuga::{MatrixTrait, anyhow, zeros};
use std::iter::Sum;

use crate::geometry::matrix::invert;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{
    Coordinate, CoordinateConversionError, CoordinateScalar,
};
use num_traits::cast;

/// Helper function to compute squared norm using generic arithmetic on T.
///
/// This function computes the sum of squares of coordinates using generic
/// arithmetic operations on type T, avoiding premature conversion to f64.
///
/// # Arguments
///
/// * `coords` - Array of coordinates of type T
///
/// # Returns
///
/// The squared norm (sum of squares) as type T
pub fn squared_norm<T, const D: usize>(coords: [T; D]) -> T
where
    T: CoordinateScalar + num_traits::Zero,
{
    coords.iter().fold(T::zero(), |acc, &x| acc + x * x)
}

/// Compute the d-dimensional hypot (Euclidean norm) of a coordinate array.
///
/// This function provides a numerically stable way to compute the Euclidean distance
/// (L2 norm) of a d-dimensional vector. For 2D, it uses the standard library's
/// `f64::hypot` function which provides optimal numerical stability. For higher
/// dimensions, it implements a generalized hypot calculation.
///
/// # Numerical Stability
///
/// The 2D case uses `f64::hypot(a, b)` which avoids overflow and underflow
/// issues when computing `sqrt(a² + b²)`. For higher dimensions, the function
/// implements a similar approach by finding the maximum absolute value and
/// scaling all coordinates relative to it.
///
/// # Arguments
///
/// * `coords` - Array of coordinates of type T
///
/// # Returns
///
/// The Euclidean norm (hypot) as type T
///
/// # Examples
///
/// ```
/// use delaunay::geometry::util::hypot;
///
/// // 2D case - uses std::f64::hypot internally
/// let distance_2d = hypot([3.0, 4.0]);
/// assert_eq!(distance_2d, 5.0);
///
/// // 3D case - uses generalized algorithm
/// let distance_3d = hypot([1.0, 2.0, 2.0]);
/// assert_eq!(distance_3d, 3.0);
///
/// // Higher dimensions
/// let distance_4d = hypot([1.0, 1.0, 1.0, 1.0]);
/// assert_eq!(distance_4d, 2.0);
/// ```
pub fn hypot<T, const D: usize>(coords: [T; D]) -> T
where
    T: CoordinateScalar + num_traits::Zero + From<f64>,
    f64: From<T>,
{
    match D {
        0 => T::zero(),
        1 => Float::abs(coords[0]),
        2 => {
            // Use standard library hypot for optimal 2D performance and stability
            let a_f64: f64 = coords[0].into();
            let b_f64: f64 = coords[1].into();
            let result_f64 = a_f64.hypot(b_f64);
            <T as From<f64>>::from(result_f64)
        }
        _ => {
            // For higher dimensions, implement generalized hypot
            // Find the maximum absolute value to avoid overflow/underflow
            let max_abs = coords
                .iter()
                .map(|&x| Float::abs(x))
                .fold(T::zero(), |acc, x| if x > acc { x } else { acc });

            if max_abs == T::zero() {
                return T::zero();
            }

            // Scale all coordinates by max_abs and compute sum of squares
            let sum_of_scaled_squares = coords
                .iter()
                .map(|&x| {
                    let scaled = x / max_abs;
                    scaled * scaled
                })
                .fold(T::zero(), |acc, x| acc + x);

            // Result is max_abs * sqrt(sum_of_scaled_squares)
            max_abs * Float::sqrt(sum_of_scaled_squares)
        }
    }
}

/// Calculate the circumcenter of a set of points forming a simplex.
///
/// The circumcenter is the unique point equidistant from all points of
/// the simplex. Returns an error if the points do not form a valid simplex or
/// if the computation fails due to degeneracy or numerical issues.
///
/// Using the approach from:
///
/// Lévy, Bruno, and Yang Liu.
/// "Lp Centroidal Voronoi Tessellation and Its Applications."
/// ACM Transactions on Graphics 29, no. 4 (July 26, 2010): 119:1-119:11.
/// <https://doi.org/10.1145/1778765.1778856>.
///
/// The circumcenter C of a simplex with points `x_0`, `x_1`, ..., `x_n` is the
/// solution to the system:
///
/// C = 1/2 (A^-1*B)
///
/// Where:
///
/// A is a matrix (to be inverted) of the form:
///     (x_1-x0) for all coordinates in x1, x0
///     (x2-x0) for all coordinates in x2, x0
///     ... for all `x_n` in the simplex
///
/// These are the perpendicular bisectors of the edges of the simplex.
///
/// And:
///
/// B is a vector of the form:
///     (x_1^2-x0^2) for all coordinates in x1, x0
///     (x_2^2-x0^2) for all coordinates in x2, x0
///     ... for all `x_n` in the simplex
///
/// The resulting vector gives the coordinates of the circumcenter.
///
/// # Arguments
///
/// * `points` - A slice of points that form the simplex
///
/// # Returns
/// The circumcenter as a Point<T, D> if successful, or an error if the
/// simplex is degenerate or the matrix inversion fails.
///
/// # Errors
///
/// Returns an error if:
/// - The points do not form a valid simplex
/// - The matrix inversion fails due to degeneracy
/// - Array conversion fails
///
/// # Example
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::geometry::util::circumcenter;
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let points = vec![point1, point2, point3, point4];
/// let center = circumcenter(&points).unwrap();
/// assert_eq!(center, Point::new([0.5, 0.5, 0.5]));
/// ```
pub fn circumcenter<T, const D: usize>(points: &[Point<T, D>]) -> Result<Point<T, D>, anyhow::Error>
where
    T: CoordinateScalar + ComplexField<RealField = T> + Sum + Zero,
    f64: From<T>,
    T: From<f64>,
{
    if points.is_empty() {
        return Err(anyhow::Error::msg("Empty point set"));
    }

    let dim = points.len() - 1;
    if dim != D {
        return Err(anyhow::Error::msg("Not a simplex!"));
    }

    // Build matrix A and vector B for the linear system
    let mut matrix = zeros(dim, dim);
    let mut b = zeros(dim, 1);
    let coords_0: [T; D] = (&points[0]).into();
    let coords_0_f64: [f64; D] = coords_0.map(std::convert::Into::into);

    for i in 0..dim {
        let coords_point: [T; D] = (&points[i + 1]).into();
        let coords_point_f64: [f64; D] = coords_point.map(std::convert::Into::into);

        // Fill matrix row
        for j in 0..dim {
            matrix[(i, j)] = coords_point_f64[j] - coords_0_f64[j];
        }

        // Calculate squared distance using squared_norm for consistency
        let mut diff_coords = [T::zero(); D];
        for j in 0..D {
            diff_coords[j] = coords_point[j] - coords_0[j];
        }
        let squared_distance = squared_norm(diff_coords);
        let squared_distance_f64: f64 = squared_distance.into();
        b[(i, 0)] = squared_distance_f64;
    }

    let a_inv = invert(&matrix)?;
    let solution = a_inv * b * 0.5;
    let solution_vec = solution.col(0);
    // Try different array conversion approaches
    // Approach 1: Using try_from (most idiomatic)
    let solution_slice: &[f64] = &solution_vec;
    let solution_array: [f64; D] = solution_slice
        .try_into()
        .map_err(|_| anyhow::Error::msg("Failed to convert solution vector to array"))?;

    // Convert solution from f64 back to T
    let solution_array_t: [T; D] = solution_array.map(|x| <T as From<f64>>::from(x));
    Ok(Point::<T, D>::from(solution_array_t))
}

/// Calculate the circumradius of a set of points forming a simplex.
///
/// The circumradius is the distance from the circumcenter to any point of the simplex.
///
/// # Arguments
///
/// * `points` - A slice of points that form the simplex
///
/// # Returns
/// The circumradius as a value of type T if successful, or an error if the
/// circumcenter calculation fails.
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
/// use delaunay::geometry::util::circumradius;
/// use approx::assert_relative_eq;
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let points = vec![point1, point2, point3, point4];
/// let radius = circumradius(&points).unwrap();
/// let expected_radius = (3.0_f64.sqrt() / 2.0);
/// assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
/// ```
pub fn circumradius<T, const D: usize>(points: &[Point<T, D>]) -> Result<T, anyhow::Error>
where
    T: CoordinateScalar + ComplexField<RealField = T> + Sum + Zero,
    f64: From<T>,
    T: From<f64>,
{
    let circumcenter = circumcenter(points)?;
    circumradius_with_center(points, &circumcenter)
}

/// Calculate the circumradius given a precomputed circumcenter.
///
/// This is a helper function that calculates the circumradius when the circumcenter
/// is already known, avoiding redundant computation.
///
/// # Arguments
///
/// * `points` - A slice of points that form the simplex
/// * `circumcenter` - The precomputed circumcenter
///
/// # Returns
/// The circumradius as a value of type T if successful, or an error if the
/// simplex is degenerate or the distance calculation fails.
///
/// # Errors
///
/// Returns an error if:
/// - The points slice is empty
/// - Coordinate conversion fails
/// - Distance calculation fails
///
/// # Example
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::geometry::util::{circumcenter, circumradius_with_center};
/// use approx::assert_relative_eq;
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let points = vec![point1, point2, point3, point4];
/// let center = circumcenter(&points).unwrap();
/// let radius = circumradius_with_center(&points, &center).unwrap();
/// let expected_radius = (3.0_f64.sqrt() / 2.0);
/// assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
/// ```
pub fn circumradius_with_center<T, const D: usize>(
    points: &[Point<T, D>],
    circumcenter: &Point<T, D>,
) -> Result<T, anyhow::Error>
where
    T: CoordinateScalar + ComplexField<RealField = T> + Sum + Zero,
    f64: From<T>,
    T: From<f64>,
{
    if points.is_empty() {
        return Err(anyhow::Error::msg("Empty point set"));
    }

    let point_coords: [T; D] = (&points[0]).into();
    let circumcenter_coords: [T; D] = circumcenter.to_array();

    // Calculate distance using hypot for numerical stability
    let mut diff_coords = [T::zero(); D];
    for i in 0..D {
        diff_coords[i] = circumcenter_coords[i] - point_coords[i];
    }
    let distance = hypot(diff_coords);
    Ok(distance)
}

/// Helper function to convert a point's coordinates to a `[f64; D]` array.
///
/// This function performs safe conversion from generic coordinate type `T` to `f64`,
/// providing detailed error information if any coordinate conversion fails.
///
/// # Arguments
///
/// * `point` - The point whose coordinates need to be converted
///
/// # Returns
///
/// An array of f64 coordinates if successful, or a conversion error if any
/// coordinate cannot be safely converted to f64.
///
/// # Errors
///
/// Returns [`CoordinateConversionError::ConversionFailed`] if any coordinate
/// cannot be converted to f64, including the coordinate index and value that failed.
///
/// # Example
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::util::convert_point_to_f64_coords;
/// use delaunay::geometry::traits::coordinate::Coordinate;
///
/// let point = Point::new([1.0f32, 2.0f32, 3.0f32]);
/// let coords_f64 = convert_point_to_f64_coords(&point).unwrap();
/// assert_eq!(coords_f64, [1.0f64, 2.0f64, 3.0f64]);
/// ```
pub fn convert_point_to_f64_coords<T: CoordinateScalar + Sum, const D: usize>(
    point: &Point<T, D>,
) -> Result<[f64; D], CoordinateConversionError> {
    let point_coords: [T; D] = point.into();
    let mut point_coords_f64: [f64; D] = [0.0; D];
    for (j, &coord) in point_coords.iter().enumerate() {
        point_coords_f64[j] =
            cast(coord).ok_or_else(|| CoordinateConversionError::ConversionFailed {
                coordinate_index: j,
                coordinate_value: format!("{coord:?}"),
                from_type: std::any::type_name::<T>(),
                to_type: "f64",
            })?;
    }
    Ok(point_coords_f64)
}

/// Helper function to convert a scalar value to `f64`.
///
/// This function performs safe conversion from generic coordinate type `T` to `f64`,
/// providing detailed error information if the conversion fails.
///
/// # Arguments
///
/// * `value` - The scalar value to convert
///
/// # Returns
///
/// The value as f64 if successful, or a conversion error if the value
/// cannot be safely converted to f64.
///
/// # Errors
///
/// Returns [`CoordinateConversionError::ConversionFailed`] if the value
/// cannot be converted to f64, including the value that failed conversion.
///
/// # Example
///
/// ```
/// use delaunay::geometry::util::convert_scalar_to_f64;
///
/// let value_f32 = 42.5f32;
/// let value_f64 = convert_scalar_to_f64(value_f32).unwrap();
/// assert_eq!(value_f64, 42.5f64);
/// ```
pub fn convert_scalar_to_f64<T: CoordinateScalar + Sum>(
    value: T,
) -> Result<f64, CoordinateConversionError> {
    cast(value).ok_or_else(|| CoordinateConversionError::ConversionFailed {
        coordinate_index: 0,
        coordinate_value: format!("{value:?}"),
        from_type: std::any::type_name::<T>(),
        to_type: "f64",
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;
    use approx::assert_relative_eq;

    #[test]
    fn test_hypot_2d() {
        // Test 2D case - should use std::f64::hypot
        let distance = hypot([3.0, 4.0]);
        assert_relative_eq!(distance, 5.0, epsilon = 1e-10);

        // Test with zero
        let distance_zero = hypot([0.0, 0.0]);
        assert_relative_eq!(distance_zero, 0.0, epsilon = 1e-10);

        // Test with negative values
        let distance_neg = hypot([-3.0, 4.0]);
        assert_relative_eq!(distance_neg, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hypot_3d() {
        // Test 3D case - uses generalized algorithm
        let distance = hypot([1.0, 2.0, 2.0]);
        assert_relative_eq!(distance, 3.0, epsilon = 1e-10);

        // Test unit vector in 3D
        let distance_unit = hypot([1.0, 0.0, 0.0]);
        assert_relative_eq!(distance_unit, 1.0, epsilon = 1e-10);

        // Test with all equal components
        let distance_equal = hypot([1.0, 1.0, 1.0]);
        assert_relative_eq!(distance_equal, 3.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_hypot_4d() {
        // Test 4D case
        let distance = hypot([1.0, 1.0, 1.0, 1.0]);
        assert_relative_eq!(distance, 2.0, epsilon = 1e-10);

        // Test with zero vector
        let distance_zero = hypot([0.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(distance_zero, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hypot_edge_cases() {
        // Test 0D case
        let distance_0d = hypot::<f64, 0>([]);
        assert_relative_eq!(distance_0d, 0.0, epsilon = 1e-10);

        // Test 1D case
        let distance_1d_pos = hypot([5.0]);
        assert_relative_eq!(distance_1d_pos, 5.0, epsilon = 1e-10);

        let distance_1d_neg = hypot([-5.0]);
        assert_relative_eq!(distance_1d_neg, 5.0, epsilon = 1e-10);

        // Test large values that might cause overflow with naive sqrt(x² + y²)
        let distance_large = hypot([1e200, 1e200]);
        assert!(distance_large.is_finite());
        assert!(distance_large > 0.0);
    }

    #[test]
    fn predicates_circumcenter() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let center = circumcenter(&points).unwrap();

        assert_eq!(center, Point::new([0.5, 0.5, 0.5]));
    }

    #[test]
    fn predicates_circumcenter_fail() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
        ];
        let center = circumcenter(&points);

        assert!(center.is_err());
    }

    #[test]
    fn predicates_circumradius() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let radius = circumradius(&points).unwrap();
        let expected_radius: f64 = 3.0_f64.sqrt() / 2.0;

        assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
    }

    #[test]
    fn predicates_circumcenter_2d() {
        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([2.0, 0.0]),
            Point::new([1.0, 2.0]),
        ];
        let center = circumcenter(&points).unwrap();

        // For this triangle, circumcenter should be at (1.0, 0.75)
        assert_relative_eq!(center.to_array()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(center.to_array()[1], 0.75, epsilon = 1e-10);
    }

    #[test]
    fn predicates_circumradius_2d() {
        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];
        let radius = circumradius(&points).unwrap();

        // For a right triangle with legs of length 1, circumradius is sqrt(2)/2
        let expected_radius = 2.0_f64.sqrt() / 2.0;
        assert_relative_eq!(radius, expected_radius, epsilon = 1e-10);
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

    // =============================================================================
    // COORDINATE CONVERSION FUNCTION TESTS
    // =============================================================================

    #[test]
    fn test_convert_point_to_f64_coords_basic_success() {
        // Test successful conversion from f32 to f64
        let point_f32 = Point::new([1.0f32, 2.0f32, 3.0f32]);
        let result = convert_point_to_f64_coords(&point_f32);

        assert!(result.is_ok());
        let coords_f64 = result.unwrap();
        assert_relative_eq!(coords_f64[0], 1.0f64, epsilon = 1e-10);
        assert_relative_eq!(coords_f64[1], 2.0f64, epsilon = 1e-10);
        assert_relative_eq!(coords_f64[2], 3.0f64, epsilon = 1e-10);
    }

    #[test]
    fn test_convert_point_to_f64_coords_different_dimensions() {
        // Test 1D point
        let point_1d = Point::new([42.0f32]);
        let result_1d = convert_point_to_f64_coords(&point_1d);
        assert!(result_1d.is_ok());
        let r1d = result_1d.unwrap();
        assert_relative_eq!(r1d[0], 42.0f64, epsilon = 1e-10);

        // Test 2D point
        let point_2d = Point::new([1.5f32, 2.5f32]);
        let result_2d = convert_point_to_f64_coords(&point_2d);
        assert!(result_2d.is_ok());
        let r2d = result_2d.unwrap();
        assert_relative_eq!(r2d[0], 1.5f64, epsilon = 1e-10);
        assert_relative_eq!(r2d[1], 2.5f64, epsilon = 1e-10);

        // Test 4D point
        let point_4d = Point::new([1.0f32, 2.0f32, 3.0f32, 4.0f32]);
        let result_4d = convert_point_to_f64_coords(&point_4d);
        assert!(result_4d.is_ok());
        let r4d = result_4d.unwrap();
        assert_relative_eq!(r4d[0], 1.0f64, epsilon = 1e-10);
        assert_relative_eq!(r4d[1], 2.0f64, epsilon = 1e-10);
        assert_relative_eq!(r4d[2], 3.0f64, epsilon = 1e-10);
        assert_relative_eq!(r4d[3], 4.0f64, epsilon = 1e-10);

        // Test 5D point (high dimension)
        let point_5d = Point::new([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32]);
        let result_5d = convert_point_to_f64_coords(&point_5d);
        assert!(result_5d.is_ok());
        let r5d = result_5d.unwrap();
        assert_relative_eq!(r5d[0], 1.0f64, epsilon = 1e-10);
        assert_relative_eq!(r5d[1], 2.0f64, epsilon = 1e-10);
        assert_relative_eq!(r5d[2], 3.0f64, epsilon = 1e-10);
        assert_relative_eq!(r5d[3], 4.0f64, epsilon = 1e-10);
        assert_relative_eq!(r5d[4], 5.0f64, epsilon = 1e-10);
    }

    #[test]
    fn test_convert_point_to_f64_coords_edge_values() {
        // Test with zero coordinates
        let point_zeros = Point::new([0.0f32, 0.0f32, 0.0f32]);
        let result_zeros = convert_point_to_f64_coords(&point_zeros);
        assert!(result_zeros.is_ok());
        let rzeros = result_zeros.unwrap();
        assert_relative_eq!(rzeros[0], 0.0f64, epsilon = 1e-10);
        assert_relative_eq!(rzeros[1], 0.0f64, epsilon = 1e-10);
        assert_relative_eq!(rzeros[2], 0.0f64, epsilon = 1e-10);

        // Test with negative coordinates
        let point_negative = Point::new([-1.0f32, -2.5f32, -0.001f32]);
        let result_negative = convert_point_to_f64_coords(&point_negative);
        assert!(result_negative.is_ok());
        let coords_negative = result_negative.unwrap();
        assert_relative_eq!(coords_negative[0], -1.0f64, epsilon = 1e-6);
        assert_relative_eq!(coords_negative[1], -2.5f64, epsilon = 1e-6);
        assert_relative_eq!(coords_negative[2], -0.001f64, epsilon = 1e-6);

        // Test with very small values
        let point_small = Point::new([1e-6f32, 1e-10f32, 1e-20f32]);
        let result_small = convert_point_to_f64_coords(&point_small);
        assert!(result_small.is_ok());
        let coords = result_small.unwrap();
        assert_relative_eq!(coords[0], 1e-6f64, epsilon = 1e-6); // f32 has limited precision
        assert_relative_eq!(coords[1], 1e-10f64, epsilon = 1e-10); // More relaxed epsilon for f32
        // Note: 1e-20f32 might lose precision when converting from f32

        // Test with large values
        let point_large = Point::new([1e6f32, 1e8f32, 1e10f32]);
        let result_large = convert_point_to_f64_coords(&point_large);
        assert!(result_large.is_ok());
        let coords_large = result_large.unwrap();
        assert_relative_eq!(coords_large[0], 1e6f64, epsilon = 1e-6);
        assert_relative_eq!(coords_large[1], 1e8f64, epsilon = 1e-6);
        assert_relative_eq!(coords_large[2], 1e10f64, epsilon = 1e-6);
    }

    #[test]
    fn test_convert_point_to_f64_coords_already_f64() {
        // Test conversion from f64 to f64 (should be identity)
        let point_f64 = Point::new([1.123_456_789_012_345_f64, 2.987_654_321_098_765_f64]);
        let result = convert_point_to_f64_coords(&point_f64);
        assert!(result.is_ok());
        let coords = result.unwrap();
        assert_relative_eq!(coords[0], 1.123_456_789_012_345_f64, epsilon = 1e-15);
        assert_relative_eq!(coords[1], 2.987_654_321_098_765_f64, epsilon = 1e-15);
    }

    #[test]
    fn test_convert_point_to_f64_coords_special_values() {
        use std::f32;

        // Test with infinity (should work for f32 to f64)
        let point_inf = Point::new([f32::INFINITY, f32::NEG_INFINITY, 0.0f32]);
        let result_inf = convert_point_to_f64_coords(&point_inf);
        assert!(result_inf.is_ok());
        let coords_inf = result_inf.unwrap();
        assert!(coords_inf[0].is_infinite() && coords_inf[0].is_sign_positive());
        assert!(coords_inf[1].is_infinite() && coords_inf[1].is_sign_negative());
        assert_relative_eq!(coords_inf[2], 0.0f64, epsilon = 1e-10);

        // Test with NaN (should work for f32 to f64)
        let point_nan = Point::new([f32::NAN, 1.0f32]);
        let result_nan = convert_point_to_f64_coords(&point_nan);
        assert!(result_nan.is_ok());
        let coords_nan = result_nan.unwrap();
        assert!(coords_nan[0].is_nan());
        assert_relative_eq!(coords_nan[1], 1.0f64, epsilon = 1e-10);
    }

    #[test]
    fn test_convert_point_to_f64_coords_precision() {
        // Test that f64 precision is maintained when converting from f64
        let high_precision_value = 1.123_456_789_012_345_7_f64;
        let point_precise = Point::new([high_precision_value, 0.0f64]);
        let result = convert_point_to_f64_coords(&point_precise);
        assert!(result.is_ok());
        let coords = result.unwrap();
        assert_relative_eq!(coords[0], high_precision_value, epsilon = 1e-15); // Should maintain full f64 precision
    }

    #[test]
    fn test_convert_scalar_to_f64_basic_success() {
        // Test successful conversion from f32 to f64
        let value_f32 = 42.5f32;
        let result = convert_scalar_to_f64(value_f32);

        assert!(result.is_ok());
        assert_relative_eq!(result.unwrap(), 42.5f64, epsilon = 1e-10);
    }

    #[test]
    fn test_convert_scalar_to_f64_different_types() {
        // Test with f64 (identity conversion)
        let value_f64 = 123.456_789_f64;
        let result_f64 = convert_scalar_to_f64(value_f64);
        assert!(result_f64.is_ok());
        assert_relative_eq!(result_f64.unwrap(), 123.456_789_f64, epsilon = 1e-10);

        // Test with f32
        let value_f32 = -987.654f32;
        let result_f32 = convert_scalar_to_f64(value_f32);
        assert!(result_f32.is_ok());
        assert_relative_eq!(result_f32.unwrap(), -987.654f64, epsilon = 1e-3);
    }

    #[test]
    fn test_convert_scalar_to_f64_edge_values() {
        // Test with zero
        let zero_f32 = 0.0f32;
        let result_zero = convert_scalar_to_f64(zero_f32);
        assert!(result_zero.is_ok());
        assert_relative_eq!(result_zero.unwrap(), 0.0f64, epsilon = 1e-10);

        // Test with negative zero
        let neg_zero_f32 = -0.0f32;
        let result_neg_zero = convert_scalar_to_f64(neg_zero_f32);
        assert!(result_neg_zero.is_ok());
        assert_relative_eq!(result_neg_zero.unwrap(), -0.0f64, epsilon = 1e-10);

        // Test with very small positive value
        let small_positive = 1e-30f32;
        let result_small_pos = convert_scalar_to_f64(small_positive);
        assert!(result_small_pos.is_ok());
        let small_pos_value = result_small_pos.unwrap();
        assert!(small_pos_value > 0.0);
        assert!(small_pos_value.is_finite());

        // Test with very small negative value
        let small_negative = -1e-30f32;
        let result_small_neg = convert_scalar_to_f64(small_negative);
        assert!(result_small_neg.is_ok());
        let small_neg_value = result_small_neg.unwrap();
        assert!(small_neg_value < 0.0);
        assert!(small_neg_value.is_finite());

        // Test with large positive value
        let large_positive = 1e30f32;
        let result_large_pos = convert_scalar_to_f64(large_positive);
        assert!(result_large_pos.is_ok());
        let large_pos_value = result_large_pos.unwrap();
        assert!(large_pos_value > 0.0);
        assert!(large_pos_value.is_finite());

        // Test with large negative value
        let large_negative = -1e30f32;
        let result_large_neg = convert_scalar_to_f64(large_negative);
        assert!(result_large_neg.is_ok());
        let large_neg_value = result_large_neg.unwrap();
        assert!(large_neg_value < 0.0);
        assert!(large_neg_value.is_finite());
    }

    #[test]
    fn test_convert_scalar_to_f64_special_values() {
        use std::f32;

        // Test with positive infinity
        let pos_inf_f32 = f32::INFINITY;
        let result_pos_inf = convert_scalar_to_f64(pos_inf_f32);
        assert!(result_pos_inf.is_ok());
        let pos_inf_result = result_pos_inf.unwrap();
        assert!(pos_inf_result.is_infinite() && pos_inf_result.is_sign_positive());

        // Test with negative infinity
        let neg_inf_f32 = f32::NEG_INFINITY;
        let result_neg_inf = convert_scalar_to_f64(neg_inf_f32);
        assert!(result_neg_inf.is_ok());
        let neg_inf_result = result_neg_inf.unwrap();
        assert!(neg_inf_result.is_infinite() && neg_inf_result.is_sign_negative());

        // Test with NaN
        let nan_f32 = f32::NAN;
        let result_nan = convert_scalar_to_f64(nan_f32);
        assert!(result_nan.is_ok());
        assert!(result_nan.unwrap().is_nan());
    }

    #[test]
    fn test_convert_scalar_to_f64_precision() {
        // Test that f64 precision is maintained when converting from f64
        let high_precision_value = 1.123_456_789_012_345_7_f64;
        let result = convert_scalar_to_f64(high_precision_value);
        assert!(result.is_ok());
        assert_relative_eq!(result.unwrap(), high_precision_value, epsilon = 1e-15); // Should maintain full f64 precision

        // Test f32 precision limits
        let f32_max_precision = 1.234_567_f32; // f32 has about 7 decimal digits of precision
        let result_f32_precision = convert_scalar_to_f64(f32_max_precision);
        assert!(result_f32_precision.is_ok());
        assert_relative_eq!(result_f32_precision.unwrap(), 1.234_567_f64, epsilon = 1e-6);
    }

    #[test]
    fn test_convert_point_to_f64_coords_consistency_with_scalar() {
        // Test that converting a point gives the same result as converting each coordinate individually
        let point = Point::new([1.5f32, -2.75f32, 0.125f32]);
        let point_result = convert_point_to_f64_coords(&point).unwrap();

        let scalar_results = [
            convert_scalar_to_f64(1.5f32).unwrap(),
            convert_scalar_to_f64(-2.75f32).unwrap(),
            convert_scalar_to_f64(0.125f32).unwrap(),
        ];

        // Compare each coordinate individually with appropriate epsilon
        for i in 0..point_result.len() {
            assert_relative_eq!(point_result[i], scalar_results[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_convert_functions_error_information() {
        // Test that errors contain proper information
        // Note: Since the cast function for f32->f64 and f64->f64 should always succeed,
        // it's difficult to create a failure case. We'll test the error structure
        // by examining the error type and ensuring proper error formatting.

        // For successful conversions, we can at least verify the error type exists
        let point = Point::new([1.0f32, 2.0f32]);
        let result = convert_point_to_f64_coords(&point);
        assert!(result.is_ok());

        // If we had a case where conversion could fail, we would test:
        // - coordinate_index is correct for point conversion
        // - coordinate_value is properly formatted
        // - from_type and to_type are correctly set
        // This would be more relevant for conversions that could actually fail,
        // such as converting from a custom numeric type that might not fit in f64
    }

    #[test]
    fn test_convert_functions_comprehensive_dimensions() {
        // Test various dimensions to ensure the generic implementation works correctly

        // 0D point (edge case)
        let point_0d: Point<f32, 0> = Point::new([]);
        let result_0d = convert_point_to_f64_coords(&point_0d);
        assert!(result_0d.is_ok());
        let coords_0d: [f64; 0] = result_0d.unwrap();
        let expected_0d: [f64; 0] = [];
        // For 0D arrays, just verify both are empty
        assert_eq!(coords_0d.len(), 0);
        assert_eq!(expected_0d.len(), 0);

        // Test higher dimensions up to reasonable limits
        let point_6d = Point::new([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]);
        let result_6d = convert_point_to_f64_coords(&point_6d);
        assert!(result_6d.is_ok());
        let r6d = result_6d.unwrap();
        let expected_6d = [1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64, 6.0f64];
        for i in 0..6 {
            assert_relative_eq!(r6d[i], expected_6d[i], epsilon = 1e-10);
        }
    }
}
