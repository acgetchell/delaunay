//! Matrix operations.

// =============================================================================
// IMPORTS
// =============================================================================

use peroxide::prelude::*;
use thiserror::Error;

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

/// Default tolerance for matrix singularity checks.
///
/// This value is chosen to be appropriately small for typical geometric computations
/// while being large enough to handle floating-point precision issues.
const SINGULARITY_TOLERANCE: f64 = 1e-12;

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
/// use peroxide::fuga::*;
/// use peroxide::c;
/// use delaunay::geometry::matrix::is_singular;
///
/// // Clearly singular matrix (determinant = 0)
/// let singular_matrix = matrix(c!(1, 2, 2, 4), 2, 2, Row);
/// assert!(is_singular(&singular_matrix));
///
/// // Non-singular matrix
/// let regular_matrix = matrix(c!(1, 2, 3, 4), 2, 2, Row);
/// assert!(!is_singular(&regular_matrix));
///
/// // Nearly singular matrix (determinant very close to 0)
/// let nearly_singular = matrix(c!(1.0, 2.0, 1.0000000000001, 2.0), 2, 2, Row);
/// assert!(is_singular(&nearly_singular));
/// ```
#[must_use]
pub fn is_singular(matrix: &Matrix) -> bool {
    let det = matrix.det();
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
/// use peroxide::fuga::*;
/// use peroxide::c;
/// use delaunay::geometry::matrix::invert;
///
/// let matrix = matrix(c!(1, 2, 3, 4), 2, 2, Row);
/// let inverted_matrix = invert(&matrix);
///
/// assert_eq!(inverted_matrix.unwrap().data, vec![-2.0, 1.0, 1.5, -0.5]);
/// ```
pub fn invert(matrix: &Matrix) -> Result<Matrix, MatrixError> {
    if is_singular(matrix) {
        return Err(MatrixError::SingularMatrix);
    }
    let inv = matrix.inv();
    Ok(inv)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use peroxide::c;
    use peroxide::fuga::*;

    use super::*;

    #[test]
    fn matrix_default() {
        let matrix = Matrix::default();

        assert_eq!(matrix.data, vec![0.0; 0]);

        // Human readable output for cargo test -- --nocapture
        matrix.print();
    }

    #[test]
    fn matrix_new() {
        let matrix = matrix(c!(1, 2, 3, 4), 2, 2, Row);

        assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0]);

        // Human readable output for cargo test -- --nocapture
        matrix.print();
    }

    #[test]
    fn matrix_copy() {
        let matrix = matrix(c!(1, 2, 3, 4), 2, 2, Row);
        let matrix_copy = matrix.clone();

        assert_eq!(matrix, matrix_copy);
    }

    #[test]
    fn matrix_dim() {
        let matrix = matrix(c!(1, 2, 3, 4), 2, 2, Row);

        assert_eq!(matrix.col, 2);
        assert_eq!(matrix.row, 2);

        // Human readable output for cargo test -- --nocapture
        matrix.print();
    }

    #[test]
    fn matrix_identity() {
        let matrix: Matrix = eye(3);

        assert_eq!(
            matrix.data,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );

        // Human readable output for cargo test -- --nocapture
        matrix.print();
    }

    #[test]
    fn matrix_zeros() {
        let matrix = zeros(3, 3);

        assert_eq!(matrix.data, vec![0.0; 9]);

        // Human readable output for cargo test -- --nocapture
        matrix.print();
    }

    #[test]
    fn matrix_inverse() {
        let matrix = matrix(c!(1, 2, 3, 4), 2, 2, Row);
        let inverted_matrix = invert(&matrix).unwrap();

        assert_eq!(inverted_matrix.data, vec![-2.0, 1.0, 1.5, -0.5]);

        // Human readable output for cargo test -- --nocapture
        println!("Original matrix:");
        matrix.print();
        println!("Inverted matrix:");
        inverted_matrix.print();
    }

    #[test]
    fn matrix_inverse_of_singular_matrix() {
        let matrix = matrix(c!(1, 0, 0, 0), 2, 2, Row);
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
        let singular_matrix = matrix(c!(1, 2, 2, 4), 2, 2, Row);
        assert!(is_singular(&singular_matrix));

        // Test non-singular matrix
        let regular_matrix = matrix(c!(1, 2, 3, 4), 2, 2, Row);
        assert!(!is_singular(&regular_matrix));

        // Test nearly singular matrix (determinant very close to 0)
        // Create a matrix with a determinant close to but not exactly 0
        let nearly_singular = matrix(c!(1.0, 2.0, 1.0 + 1e-15, 2.0), 2, 2, Row);
        assert!(is_singular(&nearly_singular));

        // Test identity matrix (clearly non-singular)
        let identity = eye(3);
        assert!(!is_singular(&identity));

        // Test zero matrix (clearly singular - may have NaN determinant)
        let zero_matrix = zeros(2, 2);
        assert!(is_singular(&zero_matrix));
    }

    #[test]
    fn test_nan_determinant_handling() {
        use peroxide::fuga::LinearAlgebra;

        // Test that matrices with NaN determinants are considered singular
        // This can happen with zero matrices in some implementations
        let zero_matrix = zeros(2, 2);
        let det = LinearAlgebra::det(&zero_matrix);
        if det.is_nan() {
            println!("Zero matrix has NaN determinant - this is expected");
        } else {
            println!("Zero matrix has determinant: {det}");
        }
        // Should be singular regardless of whether det is NaN or 0
        assert!(is_singular(&zero_matrix));
    }

    #[test]
    fn test_tolerance_boundary_cases() {
        // Test matrix with determinant exactly at tolerance
        let det_at_tolerance = SINGULARITY_TOLERANCE;
        // Create a 2x2 matrix with specific determinant
        // For a 2x2 matrix [[a,b],[c,d]], det = ad - bc
        // We want ad - bc = SINGULARITY_TOLERANCE
        let a = 1.0;
        let b = 0.0;
        let c = 0.0;
        let d = det_at_tolerance;
        let boundary_matrix = matrix(c!(a, b, c, d), 2, 2, Row);
        assert!(is_singular(&boundary_matrix)); // Should be considered singular

        // Test matrix with determinant just above tolerance
        let d_above = det_at_tolerance * 2.0;
        let above_tolerance_matrix = matrix(c!(a, b, c, d_above), 2, 2, Row);
        assert!(!is_singular(&above_tolerance_matrix)); // Should not be singular
    }

    #[test]
    fn matrix_error_integration_with_circumcenter() {
        use crate::geometry::point::Point;
        use crate::geometry::traits::coordinate::Coordinate;
        use crate::geometry::util::{CircumcenterError, circumcenter};

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

    // #[test]
    // fn matrix_serialization() {
    //     let matrix = matrix(c!(1,2,3,4), 2, 2, Row);
    //     let serialized = serde_json::to_string(&matrix).unwrap();
    //     let deserialized: Matrix = serde_json::from_str(&serialized).unwrap();

    //     assert_eq!(matrix, deserialized);

    //     // Human readable output for cargo test -- --nocapture
    //     println!("Matrix: {:?}", matrix);
    //     println!("Serialized: {}", serialized);
    //     println!("Deserialized: {:?}", deserialized);
    // }
}
