//! Stack-allocated matrix operations.
//!
//! This module is Delaunay's boundary around the stack-allocated linear algebra
//! functionality provided by `la-stack`.  Geometry code should depend on the
//! local [`Matrix`] alias, checked access helpers, determinant wrappers, and
//! error conversions here rather than reaching into `la-stack` internals
//! directly.
//!
//! Keeping that shim in one file preserves a narrow API boundary: `la-stack`
//! can evolve its dispatch macros, exact-arithmetic fallbacks, tolerance names,
//! and diagnostic variants while the rest of Delaunay keeps speaking in
//! geometry-level concepts such as predicate matrices, checked active blocks,
//! and public construction errors.

#![forbid(unsafe_code)]

use la_stack::Matrix as LaMatrix;
pub(crate) use la_stack::{DEFAULT_SINGULAR_TOL, LaError, Vector as LaVector};
use thiserror::Error;

/// Stack-matrix dispatch limit.
///
/// This is chosen so that common predicate matrices can be built as:
/// - orientation: (D+1)×(D+1)
/// - insphere: (D+2)×(D+2)
///
/// With `MAX_STACK_MATRIX_DIM = 7`, we support up to `D = 5` for insphere.
pub const MAX_STACK_MATRIX_DIM: usize = la_stack::MAX_STACK_MATRIX_DISPATCH_DIM;

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
/// std::assert_matches!(err, MatrixError::SingularMatrix);
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
#[derive(Clone, Debug, Error, PartialEq)]
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

impl From<LaError> for StackMatrixDispatchError {
    fn from(source: LaError) -> Self {
        match source {
            LaError::UnsupportedDimension { requested, max } => {
                Self::UnsupportedDim { k: requested, max }
            }
            LaError::IndexOutOfBounds { row, col, dim } => Self::Matrix {
                source: MatrixError::OutOfBounds {
                    row,
                    column: col,
                    dimension: dim,
                },
            },
            source => Self::La { source },
        }
    }
}

/// Dispatch a runtime `k` (matrix dimension) to a stack-allocated `la_stack::Matrix<k>`.
///
/// This test-only macro is used for concise matrix unit tests. Production code
/// must use [`try_with_la_stack_matrix!`] so unsupported dimensions are reported
/// as typed errors at API boundaries.
#[cfg(test)]
macro_rules! with_la_stack_matrix {
    ($k:expr, |$m:ident| $body:block) => {{
        la_stack::try_with_stack_matrix!($k, |mut $m| -> Result<_, la_stack::LaError> { Ok($body) })
            .expect("test requested an unsupported stack matrix size")
    }};
}

/// Dispatch a runtime matrix dimension to a stack matrix, returning an error if unsupported.
///
/// Unsupported upstream dispatch dimensions are converted from [`LaError`], so callers
/// may return [`StackMatrixDispatchError`] directly or a public error type that implements
/// `From<LaError>` and `From<StackMatrixDispatchError>`.
macro_rules! try_with_la_stack_matrix {
    ($k:expr, |$m:ident| $body:block) => {{
        la_stack::try_with_stack_matrix!($k, |mut $m| -> _ $body)
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

/// Read one entry from a stack matrix, preserving backend index diagnostics.
///
/// This wrapper keeps predicate and geometry helper code on the checked
/// `la-stack` access path while mapping backend index errors into the crate's
/// existing matrix-error vocabulary.
#[inline]
pub(crate) fn matrix_get<const D: usize>(
    m: &Matrix<D>,
    row: usize,
    column: usize,
) -> Result<f64, StackMatrixDispatchError> {
    m.get_checked(row, column).map_err(Into::into)
}

/// Write one finite entry into a stack matrix, preserving backend diagnostics.
///
/// This wrapper is the boundary where predicate matrix construction rejects
/// non-finite values and out-of-bounds indices before later determinant stages
/// can accidentally classify invalid matrix state as geometric degeneracy.
#[inline]
pub(crate) fn matrix_set<const D: usize>(
    m: &mut Matrix<D>,
    row: usize,
    column: usize,
    value: f64,
) -> Result<(), StackMatrixDispatchError> {
    m.set_checked(row, column, value).map_err(Into::into)
}

/// Return a finite determinant and error-bound pair when the f64 fast filter supports the matrix size.
///
/// `Ok(None)` means the closed-form direct determinant path is unavailable or
/// inconclusive for this matrix size, including cases where the direct
/// determinant overflowed to a non-finite value; callers should continue to
/// exact arithmetic when the active matrix block is finite.
#[inline]
pub(crate) fn matrix_fast_filter<const D: usize>(
    m: &Matrix<D>,
) -> Result<Option<(f64, f64)>, StackMatrixDispatchError> {
    match (m.det_direct(), m.det_errbound()) {
        (Ok(Some(det)), Ok(Some(errbound))) if det.is_finite() && errbound.is_finite() => {
            Ok(Some((det, errbound)))
        }
        (Ok(_), Ok(_))
        | (Err(LaError::NonFinite { row: None, .. }), _)
        | (_, Err(LaError::NonFinite { row: None, .. })) => Ok(None),
        (Err(source), _) | (_, Err(source)) => Err(source.into()),
    }
}

/// Return whether the direct determinant is finite when the direct formula is available.
///
/// Predicate code uses this to decide whether a finite direct determinant has
/// already established that the active matrix entries are safe for exact
/// fallback.
#[inline]
pub(crate) fn matrix_direct_det_is_finite<const D: usize>(m: &Matrix<D>) -> bool {
    matches!(m.det_direct(), Ok(Some(det)) if det.is_finite())
}

/// Compute a determinant, returning `0.0` for singular matrices.
///
/// Backend errors other than singularity are surfaced as `NaN` to preserve the
/// historical infallible return type.
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
    match m.det() {
        Ok(det) => det,
        Err(LaError::Singular { .. }) => 0.0,
        Err(_) => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::assert_matches;

    use approx::assert_relative_eq;

    #[test]
    fn try_with_la_stack_matrix_returns_err_on_unsupported_dim() {
        let k = MAX_STACK_MATRIX_DIM + 1;
        let res: Result<(), StackMatrixDispatchError> =
            try_with_la_stack_matrix!(k, |_m| { Ok(()) });
        assert_matches!(
            res,
            Err(StackMatrixDispatchError::UnsupportedDim {
                k: requested,
                max
            }) if requested == k && max == MAX_STACK_MATRIX_DIM
        );
    }

    #[test]
    fn la_index_error_maps_to_matrix_error_with_context() {
        let err = StackMatrixDispatchError::from(LaError::IndexOutOfBounds {
            row: 3,
            col: 4,
            dim: 2,
        });

        assert_eq!(
            err,
            StackMatrixDispatchError::Matrix {
                source: MatrixError::OutOfBounds {
                    row: 3,
                    column: 4,
                    dimension: 2,
                },
            }
        );
    }

    #[test]
    fn stack_matrix_dispatch_error_clones_la_error_source() {
        let source = LaError::Singular { pivot_col: 3 };
        let error = StackMatrixDispatchError::La { source };

        assert_eq!(error.clone(), error);
        assert_eq!(
            error.to_string(),
            StackMatrixDispatchError::La { source }.to_string()
        );
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
            StackMatrixDispatchError::Matrix {
                source: MatrixError::OutOfBounds {
                    row: 2,
                    column: 0,
                    dimension: 2,
                },
            }
        );
    }

    #[test]
    fn matrix_set_returns_error_on_out_of_bounds_index() {
        let mut matrix = Matrix::<2>::zero();
        let err = matrix_set(&mut matrix, 0, 2, 1.0).unwrap_err();
        assert_eq!(
            err,
            StackMatrixDispatchError::Matrix {
                source: MatrixError::OutOfBounds {
                    row: 0,
                    column: 2,
                    dimension: 2,
                },
            }
        );
    }
}
