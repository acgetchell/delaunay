//! Circumsphere calculations for simplices.
//!
//! This module provides functions for computing the circumcenter and circumradius
//! of simplices in d-dimensional space.

use std::iter::Sum;

use la_stack::{DEFAULT_PIVOT_TOL, LaError, Vector as LaVector};

use crate::geometry::matrix::matrix_set;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{Coordinate, CoordinateScalar};

use super::conversions::{safe_coords_to_f64, safe_scalar_from_f64, safe_scalar_to_f64};
use super::norms::{hypot, squared_norm};

// Re-export error type
pub use super::CircumcenterError;

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
pub fn circumcenter<T, const D: usize>(
    points: &[Point<T, D>],
) -> Result<Point<T, D>, CircumcenterError>
where
    T: CoordinateScalar + Sum,
{
    #[cfg(debug_assertions)]
    if std::env::var_os("DELAUNAY_DEBUG_UNUSED_IMPORTS").is_some() {
        eprintln!(
            "circumsphere::circumcenter called (points_len={}, D={})",
            points.len(),
            D
        );
    }
    if points.is_empty() {
        return Err(CircumcenterError::EmptyPointSet);
    }

    let dim = points.len() - 1;
    if dim != D {
        return Err(CircumcenterError::InvalidSimplex {
            actual: points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    // Build matrix A and vector b for the linear system A * x = b.
    //
    // Here, A is D×D and b is length D, so we can solve with stack-allocated la-stack types.
    let coords_0 = points[0].coords();

    // Use safe coordinate conversion
    let coords_0_f64: [f64; D] = safe_coords_to_f64(coords_0)?;

    let mut a = crate::geometry::matrix::Matrix::<D>::zero();
    let mut b_arr = [0.0f64; D];

    for i in 0..D {
        let coords_point = points[i + 1].coords();

        // Use safe coordinate conversion
        let coords_point_f64: [f64; D] = safe_coords_to_f64(coords_point)?;

        // Fill matrix row
        for j in 0..D {
            matrix_set(&mut a, i, j, coords_point_f64[j] - coords_0_f64[j]);
        }

        // Calculate squared distance using squared_norm for consistency
        let mut diff_coords = [T::zero(); D];
        for j in 0..D {
            diff_coords[j] = coords_point[j] - coords_0[j];
        }
        let squared_distance = squared_norm(&diff_coords);

        // Use safe coordinate conversion for squared distance
        let squared_distance_f64: f64 = safe_scalar_to_f64(squared_distance)?;
        b_arr[i] = squared_distance_f64;
    }

    // Solve for x, then C = x0 + 1/2 * x.
    //
    // Use la-stack's default absolute pivot tolerance first, but fall back to a zero
    // tolerance (exact singular detection) for extremely scaled-but-invertible systems.
    let lu = match a.lu(DEFAULT_PIVOT_TOL) {
        Ok(lu) => lu,
        Err(LaError::Singular { .. }) => {
            // Debugging hook: the fallback typically indicates an extremely scaled-but-invertible
            // system that tripped `DEFAULT_PIVOT_TOL`. Enable this message in debug builds via:
            //   DELAUNAY_DEBUG_LU_FALLBACK=1
            #[cfg(debug_assertions)]
            if std::env::var_os("DELAUNAY_DEBUG_LU_FALLBACK").is_some() {
                eprintln!(
                    "circumcenter<{D}>: LU factorization fell back to zero pivot tolerance (DEFAULT_PIVOT_TOL={DEFAULT_PIVOT_TOL})",
                );
            }

            a.lu(0.0)
                .map_err(|e| CircumcenterError::MatrixInversionFailed {
                    details: format!("LU factorization failed: {e}"),
                })?
        }
        Err(e) => {
            return Err(CircumcenterError::MatrixInversionFailed {
                details: format!("LU factorization failed: {e}"),
            });
        }
    };

    let x = lu
        .solve_vec(LaVector::<D>::new(b_arr))
        .map_err(|e| CircumcenterError::MatrixInversionFailed {
            details: format!("LU solve failed: {e}"),
        })?
        .into_array();

    // Use safe coordinate conversion for solution and add back the first point
    let mut circumcenter_coords = [T::zero(); D];
    for i in 0..D {
        let relative_coord: T = safe_scalar_from_f64(0.5 * x[i])?;
        circumcenter_coords[i] = coords_0[i] + relative_coord;
    }
    Ok(Point::new(circumcenter_coords))
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
pub fn circumradius<T, const D: usize>(points: &[Point<T, D>]) -> Result<T, CircumcenterError>
where
    T: CoordinateScalar + Sum,
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
) -> Result<T, CircumcenterError>
where
    T: CoordinateScalar + Sum,
{
    if points.is_empty() {
        return Err(CircumcenterError::EmptyPointSet);
    }

    let point_coords = points[0].coords();
    let circumcenter_coords = circumcenter.coords();

    // Calculate distance using hypot for numerical stability
    let mut diff_coords = [T::zero(); D];
    for i in 0..D {
        diff_coords[i] = circumcenter_coords[i] - point_coords[i];
    }
    let distance = hypot(&diff_coords);
    Ok(distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;
    use approx::assert_relative_eq;

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
        assert_relative_eq!(center.coords()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(center.coords()[1], 0.75, epsilon = 1e-10);
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

    #[test]
    fn test_circumcenter_regular_simplex_3d() {
        // Test with a regular tetrahedron - use simpler vertices
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.5, 3.0_f64.sqrt() / 2.0, 0.0]),
            Point::new([0.5, 3.0_f64.sqrt() / 6.0, (2.0 / 3.0_f64).sqrt()]),
        ];
        let center = circumcenter(&points).unwrap();

        // For this tetrahedron, verify circumcenter exists and is finite
        let center_coords = center.coords();
        for coord in center_coords {
            assert!(
                coord.is_finite(),
                "Circumcenter coordinates should be finite"
            );
        }

        // Verify all points are equidistant from circumcenter
        let distances: Vec<f64> = points
            .iter()
            .map(|p| {
                let p_coords = *p.coords();
                let diff = [
                    p_coords[0] - center_coords[0],
                    p_coords[1] - center_coords[1],
                    p_coords[2] - center_coords[2],
                ];
                hypot(&diff)
            })
            .collect();

        // All distances should be equal
        for i in 1..distances.len() {
            assert_relative_eq!(distances[0], distances[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_circumcenter_regular_simplex_4d() {
        // Test 4D simplex - use orthonormal basis plus origin
        let points: Vec<Point<f64, 4>> = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];
        let center = circumcenter(&points).unwrap();

        // For this symmetric configuration, circumcenter should be at equal coordinates
        let center_coords = center.coords();
        for &coord in center_coords {
            assert!(
                coord.is_finite(),
                "Circumcenter coordinates should be finite"
            );
            // Should be around 0.5 for this configuration
            assert_relative_eq!(coord, 0.5, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_circumcenter_right_triangle_2d() {
        // Test with right triangle - circumcenter should be at hypotenuse midpoint
        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([4.0, 0.0]),
            Point::new([0.0, 3.0]),
        ];
        let center = circumcenter(&points).unwrap();

        // For right triangle, circumcenter is at midpoint of hypotenuse
        let center_coords = center.coords();
        assert_relative_eq!(center_coords[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(center_coords[1], 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_circumcenter_scaled_simplex() {
        // Test that scaling preserves circumcenter properties
        let scale = 10.0;
        let points = vec![
            Point::new([0.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([1.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([0.0 * scale, 1.0 * scale, 0.0 * scale]),
            Point::new([0.0 * scale, 0.0 * scale, 1.0 * scale]),
        ];
        let center = circumcenter(&points).unwrap();

        // Scaled simplex should have scaled circumcenter
        let expected_center = Point::new([0.5 * scale, 0.5 * scale, 0.5 * scale]);
        let center_coords = center.coords();
        let expected_coords = expected_center.coords();

        for i in 0..3 {
            assert_relative_eq!(center_coords[i], expected_coords[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_circumcenter_translated_simplex() {
        // Test that translation preserves relative circumcenter position
        let translation = [10.0, 20.0, 30.0];
        let points = vec![
            Point::new([
                0.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ]),
            Point::new([
                1.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ]),
            Point::new([
                0.0 + translation[0],
                1.0 + translation[1],
                0.0 + translation[2],
            ]),
            Point::new([
                0.0 + translation[0],
                0.0 + translation[1],
                1.0 + translation[2],
            ]),
        ];
        let center = circumcenter(&points).unwrap();

        // Get the circumcenter of the untranslated simplex for comparison
        let untranslated_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let untranslated_center = circumcenter(&untranslated_points).unwrap();

        // Translated circumcenter should be untranslated circumcenter + translation
        let center_coords = center.coords();
        let untranslated_coords = untranslated_center.coords();

        for i in 0..3 {
            assert_relative_eq!(
                center_coords[i],
                untranslated_coords[i] + translation[i],
                epsilon = 1e-9
            );
        }

        // Also verify the expected absolute values for this specific tetrahedron
        let expected = [10.5, 20.5, 30.5];
        for i in 0..3 {
            assert_relative_eq!(center_coords[i], expected[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_circumcenter_nearly_degenerate_simplex() {
        // Test with points that are nearly collinear (may succeed or fail gracefully)
        let eps = 1e-3; // Use larger epsilon for more robustness
        let points: Vec<Point<f64, 3>> = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.5, eps, 0.0]), // Slightly off the line
            Point::new([0.5, 0.0, eps]), // Slightly off the plane
        ];

        let result = circumcenter(&points);
        // Should either succeed or fail gracefully (don't require success)
        if let Ok(center) = result {
            // If it succeeds, center should have finite coordinates
            let coords = center.coords();
            assert!(
                coords.iter().all(|&x| x.is_finite()),
                "Circumcenter coordinates should be finite"
            );
        } else {
            // If it fails, that's acceptable for this nearly degenerate case
        }
    }

    #[test]
    fn test_circumcenter_empty_points() {
        let points: Vec<Point<f64, 3>> = vec![];
        let result = circumcenter(&points);

        assert!(result.is_err());
        match result.unwrap_err() {
            CircumcenterError::EmptyPointSet => {}
            other => panic!("Expected EmptyPointSet error, got: {other:?}"),
        }
    }

    #[test]
    fn test_circumcenter_wrong_dimension() {
        // Test with 2 points for 3D (need 4 points for 3D circumcenter)
        let points = vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])];
        let result = circumcenter(&points);

        assert!(result.is_err());
        match result.unwrap_err() {
            CircumcenterError::InvalidSimplex {
                actual,
                expected,
                dimension,
            } => {
                assert_eq!(actual, 2);
                assert_eq!(expected, 4); // D + 1 where D = 3
                assert_eq!(dimension, 3);
            }
            other => panic!("Expected InvalidSimplex error, got: {other:?}"),
        }
    }

    #[test]
    fn test_circumcenter_equilateral_triangle_properties() {
        // Test that circumcenter has expected properties for equilateral triangle
        let side_length = 2.0;
        let height = side_length * 3.0_f64.sqrt() / 2.0;

        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([side_length, 0.0]),
            Point::new([side_length / 2.0, height]),
        ];

        let center = circumcenter(&points).unwrap();
        let center_coords = center.coords();

        // For equilateral triangle, circumcenter should be at centroid
        let expected_x = side_length / 2.0;
        let expected_y = height / 3.0;

        assert_relative_eq!(center_coords[0], expected_x, epsilon = 1e-10);
        assert_relative_eq!(center_coords[1], expected_y, epsilon = 1e-10);

        // Verify all vertices are equidistant from circumcenter
        let _center_point = Point::new([center_coords[0], center_coords[1]]);
        let distances: Vec<f64> = points
            .iter()
            .map(|p| {
                let p_coords = *p.coords();
                let diff = [
                    p_coords[0] - center_coords[0],
                    p_coords[1] - center_coords[1],
                ];
                hypot(&diff)
            })
            .collect();

        // All distances should be equal
        for i in 1..distances.len() {
            assert_relative_eq!(distances[0], distances[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_circumcenter_numerical_stability() {
        // Test with points that could cause numerical instability
        let points: Vec<Point<f64, 2>> = vec![
            Point::new([1.0, 0.0]),
            Point::new([1.000_000_1, 0.0]), // Very close to first point
            Point::new([1.000_000_1, 0.000_000_1]), // Forms very thin triangle
        ];

        let result = circumcenter(&points);
        // Should either succeed or fail gracefully (not panic)
        if let Ok(center) = result {
            // If it succeeds, center should have finite coordinates
            let coords = center.coords();
            assert!(
                coords.iter().all(|&x| x.is_finite()),
                "Circumcenter coordinates should be finite"
            );
        } else {
            // If it fails, that's acceptable for this degenerate case
        }
    }

    #[test]
    fn test_circumcenter_1d_case() {
        // Test 1D case (2 points)
        let points = vec![Point::new([0.0]), Point::new([2.0])];

        let center = circumcenter(&points).unwrap();
        let center_coords = center.coords();

        // 1D circumcenter should be at midpoint
        assert_relative_eq!(center_coords[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_circumcenter_high_dimension() {
        // Test higher dimensional case (5D)
        let points: Vec<Point<f64, 5>> = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];

        let result = circumcenter(&points);
        assert!(result.is_ok(), "5D circumcenter should work");

        let center = result.unwrap();
        let center_coords = center.coords();

        // Verify circumcenter has finite coordinates
        for coord in center_coords {
            assert!(
                coord.is_finite(),
                "Circumcenter coordinates should be finite"
            );
        }

        // For this configuration, all points are equidistant from circumcenter
        // Verify all points are at same distance from circumcenter
        let distances: Vec<f64> = points
            .iter()
            .map(|p| {
                let p_coords = *p.coords();
                let diff = [
                    p_coords[0] - center_coords[0],
                    p_coords[1] - center_coords[1],
                    p_coords[2] - center_coords[2],
                    p_coords[3] - center_coords[3],
                    p_coords[4] - center_coords[4],
                ];
                hypot(&diff)
            })
            .collect();

        // All distances should be equal
        for i in 1..distances.len() {
            assert_relative_eq!(distances[0], distances[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn predicates_circumcenter_precise_values() {
        // Test with precisely known circumcenter values
        // Using a simplex where we can calculate the circumcenter analytically
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([6.0, 0.0, 0.0]),
            Point::new([0.0, 8.0, 0.0]),
            Point::new([0.0, 0.0, 10.0]),
        ];

        let center = circumcenter(&points).unwrap();
        let center_coords = center.coords();

        // For this configuration, circumcenter should be at (3, 4, 5)
        assert_relative_eq!(center_coords[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(center_coords[1], 4.0, epsilon = 1e-10);
        assert_relative_eq!(center_coords[2], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_circumcenter_empty_point_set() {
        let empty_points: Vec<Point<f64, 3>> = vec![];
        let result = circumcenter(&empty_points);

        assert!(matches!(result, Err(CircumcenterError::EmptyPointSet)));
    }

    #[test]
    fn test_circumcenter_invalid_simplex() {
        // Test wrong number of points for dimension
        let points_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            // Missing third point for 2D circumcenter
        ];

        let result = circumcenter(&points_2d);
        assert!(matches!(
            result,
            Err(CircumcenterError::InvalidSimplex { .. })
        ));

        // Test too many points
        let points_extra = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
            Point::new([0.5, 0.5]), // Extra point for 2D
        ];

        let result = circumcenter(&points_extra);
        assert!(matches!(
            result,
            Err(CircumcenterError::InvalidSimplex { .. })
        ));
    }

    #[test]
    fn test_circumcenter_degenerate_matrix() {
        // Test collinear points in 2D (should cause matrix inversion to fail)
        let collinear_points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([2.0, 0.0]), // Collinear with first two
        ];

        let result = circumcenter(&collinear_points);
        assert!(matches!(
            result,
            Err(CircumcenterError::MatrixInversionFailed { .. })
        ));
    }
}
