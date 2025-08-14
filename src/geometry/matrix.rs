//! Matrix operations.

// =============================================================================
// IMPORTS
// =============================================================================

use anyhow::Result;
use peroxide::prelude::*;
use thiserror::Error;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Error type for matrix operations.
#[derive(Debug, Error)]
pub enum MatrixError {
    /// Matrix is singular.
    #[error("Matrix is singular!")]
    SingularMatrix,
}

// =============================================================================
// MATRIX OPERATIONS
// =============================================================================

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
/// Returns an error if the matrix is singular (determinant equals 0) and cannot be inverted.
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
pub fn invert(matrix: &Matrix) -> Result<Matrix, anyhow::Error> {
    if matrix.det() == 0.0 {
        return Err(MatrixError::SingularMatrix.into());
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
