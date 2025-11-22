//! Matrix operations.

// =============================================================================
// IMPORTS
// =============================================================================

use nalgebra as na;
use thiserror::Error;

/// Internal linear algebra matrix type used by this crate.
///
/// This is currently an alias for `nalgebra::DMatrix<f64>`. It is exposed
/// so callers of functions in this module can use the same type in signatures.
/// The underlying type may change in a future release.
pub type Matrix = na::DMatrix<f64>;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Error type for matrix operations.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum MatrixError {
    /// Matrix is singular.
    #[error("Matrix is singular!")]
    SingularMatrix,
}

// =============================================================================
// MATRIX OPERATIONS
// =============================================================================

/// Compute adaptive tolerance scaled by matrix magnitude (infinity norm).
///
/// This computes: `base_tol` + `rel_factor` * ||A||_∞, where ||A||_∞ is the maximum
/// absolute row sum. If the last column is (approximately) all ones, it is
/// excluded from the magnitude estimate to avoid over-inflating tolerance on
/// small simplices (common in orientation/insphere matrices).
///
/// # Arguments
/// - `matrix`: The matrix whose magnitude to estimate
/// - `base_tol`: The base tolerance to start from (e.g., type-specific default)
/// # Returns
///
/// The adaptive tolerance as f64.
///
/// # Examples
///
/// ```rust
/// use delaunay::geometry::matrix::{Matrix, adaptive_tolerance};
///
/// // 3x3 matrix with a trailing column of 1.0s (e.g., orientation/insphere form)
/// let mut m = Matrix::zeros(3, 3);
/// for i in 0..3 { m[(i, 2)] = 1.0; }
/// let base = 1e-12;
/// let tol = adaptive_tolerance(&m, base);
/// // Constant 1.0 column is excluded from the scaling term: tol == base
/// assert!((tol - base).abs() <= f64::EPSILON);
///
/// // If the last column is not all ones, it contributes to scaling
/// let mut m2 = Matrix::zeros(3, 3);
/// for i in 0..3 { m2[(i, 2)] = 2.0; }
/// let tol2 = adaptive_tolerance(&m2, base);
/// assert!(tol2 > base);
/// ```
#[must_use]
pub fn adaptive_tolerance(matrix: &Matrix, base_tol: f64) -> f64 {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();

    // Check if the last column is (approximately) all ones.
    let last_col_is_all_ones =
        ncols > 0 && (0..nrows).all(|i| (matrix[(i, ncols - 1)] - 1.0).abs() <= f64::EPSILON);

    // Infinity norm (max absolute row sum), optionally excluding constant 1 column
    let mut max_row_sum = 0.0f64;
    for i in 0..nrows {
        let mut row_sum = 0.0f64;
        let col_limit = if last_col_is_all_ones {
            ncols - 1
        } else {
            ncols
        };
        for j in 0..col_limit {
            row_sum += matrix[(i, j)].abs();
        }
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }

    let rel_factor = 1e-12f64;
    rel_factor.mul_add(max_row_sum, base_tol)
}

/// Default tolerance for matrix singularity checks.
///
/// This value is chosen to be appropriately small for typical geometric computations
/// while being large enough to handle floating-point precision issues.
pub const SINGULARITY_TOLERANCE: f64 = 1e-12;

/// Checks if a matrix is singular (non-invertible) using tolerance-based comparison.
///
/// A matrix is considered singular if its absolute determinant is less than or equal
/// to the singularity tolerance, or if the determinant is NaN (which can occur with
/// degenerate matrices). This approach is more robust than exact equality
/// checks for floating-point determinants.
///
/// # Arguments
///
/// * `matrix` - The matrix to check for singularity.
///
/// # Returns
///
/// `true` if the matrix is singular (determinant ≈ 0 or NaN), `false` otherwise.
///
/// # Examples
///
/// ```
/// use delaunay::geometry::matrix::{is_singular, Matrix};
///
/// // Clearly singular matrix (determinant = 0)
/// let singular_matrix = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
/// assert!(is_singular(&singular_matrix));
///
/// // Non-singular matrix
/// let regular_matrix = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
/// assert!(!is_singular(&regular_matrix));
///
/// // Nearly singular matrix (determinant very close to 0)
/// let nearly_singular = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 1.0 + 1e-15, 2.0]);
/// assert!(is_singular(&nearly_singular));
/// ```
#[must_use]
pub fn is_singular(matrix: &Matrix) -> bool {
    let det = matrix.determinant();
    det.is_nan() || det.abs() <= SINGULARITY_TOLERANCE
}

/// Inverts a matrix.
///
/// # Arguments
///
/// * `matrix` - A matrix to invert.
///
/// # Returns
///
/// The inverted matrix.
///
/// # Errors
///
/// Returns an error if the matrix is singular (determinant is close to zero within tolerance or NaN) and cannot be inverted.
///
/// # Example
///
/// ```
/// use delaunay::geometry::matrix::{invert, Matrix};
///
/// let matrix = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
/// let inverted_matrix = invert(&matrix).unwrap();
/// let expected = Matrix::from_row_slice(2, 2, &[-2.0, 1.0, 1.5, -0.5]);
/// assert!(inverted_matrix.relative_eq(&expected, 1e-12, 1e-12));
/// ```
pub fn invert(matrix: &Matrix) -> Result<Matrix, MatrixError> {
    if is_singular(matrix) {
        return Err(MatrixError::SingularMatrix);
    }
    matrix
        .clone()
        .try_inverse()
        .ok_or(MatrixError::SingularMatrix)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::geometry::util::{CircumcenterError, circumcenter};
    use approx::assert_relative_eq;

    macro_rules! gen_adaptive_tol_tests {
        ($d:literal) => {
            pastey::paste! {
                #[test]
                fn [<adaptive_tolerance_ignores_constant_one_last_col_ $d d>]() {
                    let n = $d + 1; // orientation/insphere-like square matrix
                    let mut m = Matrix::zeros(n, n);
                    // Set only the last column to ones
                    for i in 0..n { m[(i, n - 1)] = 1.0; }
                    let base = 1e-12;
                    let tol = adaptive_tolerance(&m, base);
                    assert_relative_eq!(tol, base, epsilon = 1e-18);
                }

                #[test]
                fn [<adaptive_tolerance_includes_non_one_last_col_ $d d>]() {
                    let n = $d + 1;
                    let mut m = Matrix::zeros(n, n);
                    // Set only the last column to a constant not equal to 1.0
                    for i in 0..n { m[(i, n - 1)] = 2.0; }
                    let base = 1e-12;
                    let tol = adaptive_tolerance(&m, base);
                    // With only the last column set to 2.0, max row sum = 2.0
                    let expected = base + 2.0e-12;
                    assert_relative_eq!(tol, expected, epsilon = 1e-24);
                }
            }
        };
    }

    gen_adaptive_tol_tests!(2);
    gen_adaptive_tol_tests!(3);
    gen_adaptive_tol_tests!(4);
    gen_adaptive_tol_tests!(5);

    #[test]
    fn matrix_default() {
        let matrix: Matrix = Matrix::default();
        assert_eq!(matrix.nrows(), 0);
        assert_eq!(matrix.ncols(), 0);
    }

    #[test]
    fn matrix_new() {
        let matrix = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_relative_eq!(matrix[(0, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(matrix[(0, 1)], 2.0, epsilon = 1e-12);
        assert_relative_eq!(matrix[(1, 0)], 3.0, epsilon = 1e-12);
        assert_relative_eq!(matrix[(1, 1)], 4.0, epsilon = 1e-12);
    }

    #[test]
    fn matrix_copy() {
        let matrix = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let matrix_copy = matrix.clone();
        assert_eq!(matrix, matrix_copy);
    }

    #[test]
    fn matrix_dim() {
        let matrix = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix.nrows(), 2);
    }

    #[test]
    fn matrix_identity() {
        let matrix: Matrix = Matrix::identity(3, 3);
        let expected = Matrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(matrix, expected);
    }

    #[test]
    fn matrix_zeros() {
        let matrix = Matrix::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(matrix[(i, j)], 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn matrix_inverse() {
        let matrix = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let inverted_matrix = invert(&matrix).unwrap();
        let expected = Matrix::from_row_slice(2, 2, &[-2.0, 1.0, 1.5, -0.5]);
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(inverted_matrix[(i, j)], expected[(i, j)], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn matrix_inverse_of_singular_matrix() {
        let matrix = Matrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        let inverted_matrix = invert(&matrix);
        assert!(inverted_matrix.is_err());
        assert!(
            inverted_matrix
                .unwrap_err()
                .to_string()
                .contains("Matrix is singular")
        );
    }

    #[test]
    fn test_is_singular_function() {
        // Test clearly singular matrix (determinant = 0)
        let singular_matrix = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        assert!(is_singular(&singular_matrix));

        // Test non-singular matrix
        let regular_matrix = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(!is_singular(&regular_matrix));

        // Test nearly singular matrix (determinant very close to 0)
        let nearly_singular = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 1.0 + 1e-15, 2.0]);
        assert!(is_singular(&nearly_singular));

        // Test identity matrix (clearly non-singular)
        let identity = Matrix::identity(3, 3);
        assert!(!is_singular(&identity));

        // Test zero matrix (clearly singular)
        let zero_matrix = Matrix::zeros(2, 2);
        assert!(is_singular(&zero_matrix));
    }

    #[test]
    fn test_tolerance_boundary_cases() {
        // Test matrix with determinant exactly at tolerance
        let det_at_tolerance = SINGULARITY_TOLERANCE;
        // For a 2x2 matrix [[a,b],[c,d]], det = ad - bc
        // We want ad - bc = SINGULARITY_TOLERANCE
        let a = 1.0;
        let b = 0.0;
        let c = 0.0;
        let d = det_at_tolerance;
        let boundary_matrix = Matrix::from_row_slice(2, 2, &[a, b, c, d]);
        assert!(is_singular(&boundary_matrix)); // Should be considered singular

        // Test matrix with determinant just above tolerance
        let d_above = det_at_tolerance * 2.0;
        let above_tolerance_matrix = Matrix::from_row_slice(2, 2, &[a, b, c, d_above]);
        assert!(!is_singular(&above_tolerance_matrix)); // Should not be singular
    }

    #[test]
    fn matrix_error_integration_with_circumcenter() {
        // Test with collinear points that should cause matrix inversion to fail
        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 1.0]),
            Point::new([2.0, 2.0]), // Collinear points - should cause singular matrix
        ];

        let result = circumcenter(&points);
        assert!(result.is_err());

        // The error should be a CircumcenterError containing a MatrixError
        match result.unwrap_err() {
            CircumcenterError::MatrixError(matrix_err) => {
                assert_eq!(matrix_err, super::MatrixError::SingularMatrix);
                assert!(matrix_err.to_string().contains("Matrix is singular"));
            }
            other_err => {
                panic!("Expected MatrixError, got: {other_err:?}");
            }
        }
    }
}
