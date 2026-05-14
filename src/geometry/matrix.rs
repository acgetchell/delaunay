//! Matrix operations.
//!
//! This module provides small, stack-allocated linear algebra helpers used by
//! geometric predicates and utilities.

#![forbid(unsafe_code)]

use la_stack::{LaError, Matrix as LaMatrix};
use thiserror::Error;

/// Stack-matrix dispatch limit.
///
/// This is chosen so that common predicate matrices can be built as:
/// - orientation: (D+1)×(D+1)
/// - insphere: (D+2)×(D+2)
///
/// With `MAX_STACK_MATRIX_DIM = 7`, we support up to `D = 5` for insphere.
pub const MAX_STACK_MATRIX_DIM: usize = 7;

/// Internal linear algebra matrix type used by this crate for fixed-size operations.
pub type Matrix<const D: usize> = LaMatrix<D>;

/// Error type for matrix operations.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::MatrixError;
///
/// let err = MatrixError::SingularMatrix;
/// assert!(matches!(err, MatrixError::SingularMatrix));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatrixError {
    /// Matrix is singular.
    #[error("Matrix is singular!")]
    SingularMatrix,
    /// Matrix row or column index is outside the concrete stack matrix.
    #[error("matrix index out of bounds: ({row}, {column}) for {dimension}x{dimension}")]
    OutOfBounds {
        /// Requested row index.
        row: usize,
        /// Requested column index.
        column: usize,
        /// Concrete matrix dimension.
        dimension: usize,
    },
}

/// Error type for stack-matrix dispatch and active-block access.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum StackMatrixDispatchError {
    /// The requested matrix size is not supported by the stack-matrix dispatcher.
    #[error("unsupported stack matrix size: {k} (max {max})")]
    UnsupportedDim {
        /// Requested matrix dimension.
        k: usize,
        /// Maximum supported matrix dimension.
        max: usize,
    },
    /// The requested active block size does not match the concrete matrix type.
    #[error("active matrix block size {k} does not match concrete matrix dimension {dim}")]
    ActiveBlockDimensionMismatch {
        /// Requested active matrix dimension.
        k: usize,
        /// Concrete matrix dimension.
        dim: usize,
    },
    /// A linear algebra error originating from `la-stack`.
    #[error(transparent)]
    La {
        /// Typed source error from the linear algebra backend.
        #[from]
        source: LaError,
    },
    /// A matrix access failed inside a dispatched stack-matrix operation.
    #[error(transparent)]
    Matrix {
        /// Typed source error from matrix operations.
        #[from]
        source: MatrixError,
    },
}

/// Default tolerance for matrix singularity checks.
///
/// This value is chosen to be appropriately small for typical geometric computations
/// while being large enough to handle floating-point precision issues.
pub const SINGULARITY_TOLERANCE: f64 = 1e-12;

/// Internal dispatch shared by fallible production matrix creation and tests.
macro_rules! dispatch_la_stack_matrix {
    ($k:expr, |$m:ident| $body:block, $unsupported:expr) => {{
        match $k {
            0 => {
                #[allow(unused_mut)]
                let mut $m = $crate::geometry::matrix::Matrix::<0>::zero();
                $body
            }
            1 => {
                #[allow(unused_mut)]
                let mut $m = $crate::geometry::matrix::Matrix::<1>::zero();
                $body
            }
            2 => {
                #[allow(unused_mut)]
                let mut $m = $crate::geometry::matrix::Matrix::<2>::zero();
                $body
            }
            3 => {
                #[allow(unused_mut)]
                let mut $m = $crate::geometry::matrix::Matrix::<3>::zero();
                $body
            }
            4 => {
                #[allow(unused_mut)]
                let mut $m = $crate::geometry::matrix::Matrix::<4>::zero();
                $body
            }
            5 => {
                #[allow(unused_mut)]
                let mut $m = $crate::geometry::matrix::Matrix::<5>::zero();
                $body
            }
            6 => {
                #[allow(unused_mut)]
                let mut $m = $crate::geometry::matrix::Matrix::<6>::zero();
                $body
            }
            7 => {
                #[allow(unused_mut)]
                let mut $m = $crate::geometry::matrix::Matrix::<7>::zero();
                $body
            }
            _ => $unsupported,
        }
    }};
}

/// Dispatch a runtime `k` (matrix dimension) to a stack-allocated `la_stack::Matrix<k>`.
///
/// This test-only macro is used for concise matrix unit tests. Production code
/// must use [`try_with_la_stack_matrix!`] so unsupported dimensions are reported
/// as typed errors at API boundaries.
#[cfg(test)]
macro_rules! with_la_stack_matrix {
    ($k:expr, |$m:ident| $body:block) => {{
        dispatch_la_stack_matrix!(
            $k,
            |$m| $body,
            panic!(
                "unsupported stack matrix size: {k} (max {max})",
                k = $k,
                max = $crate::geometry::matrix::MAX_STACK_MATRIX_DIM
            )
        )
    }};
}

/// Dispatch a runtime matrix dimension to a stack matrix, returning an error if unsupported.
///
/// The provided block must evaluate to `Result<_, E>`, where `E` can be constructed from
/// [`StackMatrixDispatchError`].
macro_rules! try_with_la_stack_matrix {
    ($k:expr, |$m:ident| $body:block) => {{
        let k = $k;
        dispatch_la_stack_matrix!(
            k,
            |$m| $body,
            Err(
                $crate::geometry::matrix::StackMatrixDispatchError::UnsupportedDim {
                    k,
                    max: $crate::geometry::matrix::MAX_STACK_MATRIX_DIM,
                }
                .into(),
            )
        )
    }};
}

/// Create a zero matrix with the same const-generic dimension as `_template`.
///
/// This is useful inside `with_la_stack_matrix!` bodies where the concrete `N`
/// is hidden by the macro dispatch: calling `matrix_zero_like(&existing)` lets
/// the compiler infer `N` without a second macro expansion.
#[inline]
pub(crate) fn matrix_zero_like<const D: usize>(_template: &Matrix<D>) -> Matrix<D> {
    Matrix::<D>::zero()
}

#[inline]
pub(crate) fn matrix_get<const D: usize>(
    m: &Matrix<D>,
    row: usize,
    column: usize,
) -> Result<f64, MatrixError> {
    m.get(row, column).ok_or(MatrixError::OutOfBounds {
        row,
        column,
        dimension: D,
    })
}

#[inline]
pub(crate) fn matrix_set<const D: usize>(
    m: &mut Matrix<D>,
    row: usize,
    column: usize,
    value: f64,
) -> Result<(), MatrixError> {
    if m.set(row, column, value) {
        return Ok(());
    }

    Err(MatrixError::OutOfBounds {
        row,
        column,
        dimension: D,
    })
}

/// Compute an LU-based determinant, returning 0.0 for singular matrices.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{determinant, Matrix};
///
/// let m = Matrix::<2>::zero();
/// assert_eq!(determinant(&m), 0.0);
/// ```
#[inline]
#[must_use]
pub fn determinant<const D: usize>(m: &Matrix<D>) -> f64 {
    match m.det(0.0) {
        Ok(det) => det,
        Err(LaError::Singular { .. }) => 0.0,
        Err(_) => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn try_with_la_stack_matrix_returns_err_on_unsupported_dim() {
        let k = MAX_STACK_MATRIX_DIM + 1;
        let res: Result<(), StackMatrixDispatchError> =
            try_with_la_stack_matrix!(k, |_m| { Ok(()) });
        assert!(matches!(
            res,
            Err(StackMatrixDispatchError::UnsupportedDim { .. })
        ));
    }

    #[test]
    fn matrix_zero_like_returns_zero_matrix_of_same_size() {
        let k = 4;
        with_la_stack_matrix!(k, |original| {
            // Populate with non-zero data using an f64 counter (avoids usize→f64 cast).
            let mut val = 1.0_f64;
            for i in 0..k {
                for j in 0..k {
                    matrix_set(&mut original, i, j, val).unwrap();
                    val += 1.0;
                }
            }

            let zero = matrix_zero_like(&original);

            // All entries must be zero.
            for i in 0..k {
                for j in 0..k {
                    assert_relative_eq!(matrix_get(&zero, i, j).unwrap(), 0.0);
                }
            }

            // Original must be unchanged.
            let mut expected = 1.0_f64;
            for i in 0..k {
                for j in 0..k {
                    assert_relative_eq!(matrix_get(&original, i, j).unwrap(), expected);
                    expected += 1.0;
                }
            }
        });
    }

    #[test]
    fn matrix_zero_like_works_across_dispatch_sizes() {
        // Verify it compiles and returns zero for several representative sizes.
        for &k in &[2_usize, 3, 6, MAX_STACK_MATRIX_DIM] {
            with_la_stack_matrix!(k, |m| {
                let zero = matrix_zero_like(&m);
                assert_relative_eq!(matrix_get(&zero, 0, 0).unwrap(), 0.0);
                assert_relative_eq!(matrix_get(&zero, k - 1, k - 1).unwrap(), 0.0);
            });
        }
    }

    #[test]
    fn matrix_get_returns_error_on_out_of_bounds_index() {
        let matrix = Matrix::<2>::zero();
        let err = matrix_get(&matrix, 2, 0).unwrap_err();
        assert_eq!(
            err,
            MatrixError::OutOfBounds {
                row: 2,
                column: 0,
                dimension: 2,
            }
        );
    }

    #[test]
    fn matrix_set_returns_error_on_out_of_bounds_index() {
        let mut matrix = Matrix::<2>::zero();
        let err = matrix_set(&mut matrix, 0, 2, 1.0).unwrap_err();
        assert_eq!(
            err,
            MatrixError::OutOfBounds {
                row: 0,
                column: 2,
                dimension: 2,
            }
        );
    }
}
