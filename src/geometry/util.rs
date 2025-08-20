//! Geometric utility functions for d-dimensional geometry calculations.
//!
//! This module contains utility functions for computing distances, norms, and
//! circumsphere properties of geometric objects. These functions are used by
//! both predicates and other geometric algorithms.

use na::{ComplexField, Const, OPoint};
use nalgebra as na;
use num_traits::{Float, Zero};
use peroxide::fuga::{MatrixTrait, anyhow, zeros};
use std::iter::Sum;

use crate::geometry::matrix::invert;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{Coordinate, CoordinateScalar};

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
    [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
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
    [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
    OPoint<T, Const<D>>: From<[f64; D]>,
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
    [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
    OPoint<T, Const<D>>: From<[f64; D]>,
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
}
