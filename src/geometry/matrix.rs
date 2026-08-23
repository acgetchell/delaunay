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

/// Typed errors from the stack-allocated linear algebra backend.
///
/// Delaunay re-exports this backend error type at the matrix boundary so public
/// wrappers such as [`determinant`] can preserve exact failure context without
/// exposing the rest of `la-stack` to downstream callers.
pub use la_stack::LaError;
use la_stack::Matrix as LaMatrix;
pub(crate) use la_stack::{
    BigRational, DEFAULT_SINGULAR_TOL, FromPrimitive, Signed, SingularityReason, Vector as LaVector,
};
use num_traits::Zero;
use thiserror::Error;

/// Stack-matrix dispatch limit.
///
/// This is chosen so that common predicate matrices can be built as:
/// - orientation: (D+1)×(D+1)
/// - relative-coordinate insphere: (D+1)×(D+1)
///
/// With `MAX_STACK_MATRIX_DIM = 7`, the relative predicate supports through
/// `D = 6`. Absolute lifted matrices still require `D+2` rows and therefore
/// have a lower stack limit.
pub const MAX_STACK_MATRIX_DIM: usize = la_stack::MAX_STACK_MATRIX_DISPATCH_DIM;

/// Stack-allocated matrix type used by this crate for fixed-size linear algebra.
///
/// This alias is Delaunay's public matrix boundary around `la-stack`: callers
/// can build small matrices for diagnostics and helper APIs without depending
/// on backend module paths.
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
            LaError::UnsupportedDimension { requested, max, .. } => {
                Self::UnsupportedDim { k: requested, max }
            }
            LaError::IndexOutOfBounds { row, col, dim, .. } => Self::Matrix {
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
    m.try_get(row, column).map_err(Into::into)
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
    m.set(row, column, value).map_err(Into::into)
}

/// Solve a runtime-sized finite `f64` system with exact fraction-free elimination.
///
/// The backend converts every IEEE 754 input to its exact rational value before
/// applying Bareiss elimination, so this is an exact solve rather than a
/// floating-point approximation.
pub(crate) fn solve_exact_runtime_system(
    matrix: &[Vec<f64>],
    rhs: &[f64],
) -> Option<Result<Vec<BigRational>, StackMatrixDispatchError>> {
    let dimension = rhs.len();
    if matrix.len() != dimension || matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }

    Some(try_with_la_stack_matrix!(dimension, |stack_matrix| {
        for (row, values) in matrix.iter().enumerate() {
            for (column, value) in values.iter().copied().enumerate() {
                matrix_set(&mut stack_matrix, row, column, value)?;
            }
        }
        let rhs_vector = LaVector::try_new(std::array::from_fn(|index| rhs[index]))?;
        stack_matrix
            .solve_exact(rhs_vector)
            .map(|solution| solution.into_iter().collect())
            .map_err(Into::into)
    }))
}

/// Converts one finite IEEE-754 value to the exact rational it represents.
#[inline]
pub(crate) fn rational_from_f64(value: f64) -> Option<BigRational> {
    value
        .is_finite()
        .then(|| BigRational::from_f64(value))
        .flatten()
}

/// Solves a square system whose coefficients were formed in exact arithmetic.
///
/// Unlike [`solve_exact_runtime_system`], this entry point does not round an
/// already-derived coefficient back through `f64`. It is the shared cold path
/// for geometry whose matrix entries themselves require exact construction.
#[expect(
    clippy::needless_range_loop,
    reason = "index-based elimination keeps pivot row and column operations explicit"
)]
pub(crate) fn solve_rational_system(
    mut matrix: Vec<Vec<BigRational>>,
    mut rhs: Vec<BigRational>,
) -> Option<Vec<BigRational>> {
    let dimension = rhs.len();
    if matrix.len() != dimension || matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }

    for pivot_col in 0..dimension {
        let pivot_row = (pivot_col..dimension).find(|&row| !matrix[row][pivot_col].is_zero())?;
        if pivot_row != pivot_col {
            matrix.swap(pivot_col, pivot_row);
            rhs.swap(pivot_col, pivot_row);
        }

        let pivot = matrix[pivot_col][pivot_col].clone();
        for row in pivot_col + 1..dimension {
            if matrix[row][pivot_col].is_zero() {
                continue;
            }
            let factor = matrix[row][pivot_col].clone() / pivot.clone();
            matrix[row][pivot_col] = BigRational::from_integer(0.into());
            for column in pivot_col + 1..dimension {
                matrix[row][column] = matrix[row][column].clone()
                    - factor.clone() * matrix[pivot_col][column].clone();
            }
            rhs[row] = rhs[row].clone() - factor * rhs[pivot_col].clone();
        }
    }

    let zero = BigRational::from_integer(0.into());
    let mut solution = vec![zero; dimension];
    for row in (0..dimension).rev() {
        let mut value = rhs[row].clone();
        for column in row + 1..dimension {
            value -= matrix[row][column].clone() * solution[column].clone();
        }
        solution[row] = value / matrix[row][row].clone();
    }
    Some(solution)
}

/// Returns the sign of a square rational determinant.
///
/// The elimination is used only after a floating-point interval filter is
/// inconclusive, so allocation and rational growth remain on the cold path.
#[expect(
    clippy::needless_range_loop,
    reason = "index-based elimination keeps determinant row operations explicit"
)]
pub(crate) fn rational_determinant_sign(mut matrix: Vec<Vec<BigRational>>) -> Option<i32> {
    let dimension = matrix.len();
    if matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }
    if dimension == 0 {
        return Some(1);
    }

    let mut sign = 1;
    for pivot_col in 0..dimension {
        let Some(pivot_row) = (pivot_col..dimension).find(|&row| !matrix[row][pivot_col].is_zero())
        else {
            return Some(0);
        };
        if pivot_row != pivot_col {
            matrix.swap(pivot_col, pivot_row);
            sign = -sign;
        }

        let pivot = matrix[pivot_col][pivot_col].clone();
        if pivot.is_negative() {
            sign = -sign;
        }
        for row in pivot_col + 1..dimension {
            if matrix[row][pivot_col].is_zero() {
                continue;
            }
            let factor = matrix[row][pivot_col].clone() / pivot.clone();
            for column in pivot_col + 1..dimension {
                matrix[row][column] = matrix[row][column].clone()
                    - factor.clone() * matrix[pivot_col][column].clone();
            }
        }
    }
    Some(sign)
}

/// Return a determinant and its certified error bound when the f64 fast filter supports the matrix size.
///
/// `Ok(None)` means the closed-form direct determinant path is unavailable or
/// inconclusive for this matrix size, including arithmetic overflow or
/// underflow-sensitive evaluation. Callers should continue to exact arithmetic;
/// `la-stack` matrices are finite by construction.
#[inline]
pub(crate) fn matrix_fast_filter<const D: usize>(
    m: &Matrix<D>,
) -> Result<Option<(f64, f64)>, StackMatrixDispatchError> {
    match m.det_direct_with_errbound() {
        Ok(Some(estimate)) => Ok(Some((
            estimate.determinant(),
            estimate.absolute_error_bound(),
        ))),
        Ok(None) | Err(LaError::NonFinite { .. }) => Ok(None),
        Err(source) => Err(source.into()),
    }
}

/// Compute a determinant, returning `Ok(0.0)` for singular matrices.
///
/// Other backend failures are returned as typed [`LaError`] values.
///
/// # Errors
///
/// Returns [`LaError`] for non-singular backend failures such as non-finite
/// intermediate determinant computations.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{LaError, Matrix, determinant};
///
/// let m = Matrix::<2>::zero();
/// assert_eq!(determinant(&m)?, 0.0);
/// # Ok::<(), LaError>(())
/// ```
#[inline]
pub fn determinant<const D: usize>(m: &Matrix<D>) -> Result<f64, LaError> {
    match m.det() {
        Ok(det) => Ok(det),
        Err(LaError::Singular { .. }) => Ok(0.0),
        Err(source) => Err(source),
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    /// Dispatch a runtime `k` to a stack-allocated matrix for concise unit tests.
    macro_rules! with_la_stack_matrix {
        ($k:expr, |$m:ident| $body:block) => {{
            la_stack::try_with_stack_matrix!($k, |mut $m| -> Result<_, la_stack::LaError> {
                Ok($body)
            })
            .expect("test requested an unsupported stack matrix size")
        }};
    }

    pub(crate) use with_la_stack_matrix;
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
    fn solve_exact_runtime_system_rejects_malformed_shapes() {
        assert_eq!(
            solve_exact_runtime_system(&[vec![1.0, 0.0]], &[1.0, 0.0]),
            None
        );
        assert_eq!(
            solve_exact_runtime_system(&[vec![1.0], vec![0.0, 1.0]], &[1.0, 0.0]),
            None
        );
    }

    #[test]
    fn la_index_error_maps_to_matrix_error_with_context() {
        let err = StackMatrixDispatchError::from(LaError::index_out_of_bounds(3, 4, 2));

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
        let source = LaError::singular_exact(3);
        let error = StackMatrixDispatchError::La { source };

        assert_eq!(error.clone(), error);
        assert_eq!(
            error.to_string(),
            StackMatrixDispatchError::La { source }.to_string()
        );
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

    #[test]
    fn determinant_returns_finite_value_for_regular_matrix() {
        let matrix = Matrix::<2>::try_from_rows([[4.0, 2.0], [1.0, 3.0]]).unwrap();

        assert_relative_eq!(determinant(&matrix).unwrap(), 10.0);
    }

    #[test]
    fn determinant_returns_zero_for_singular_matrix() {
        let matrix = Matrix::<2>::try_from_rows([[1.0, 2.0], [2.0, 4.0]]).unwrap();

        assert_relative_eq!(determinant(&matrix).unwrap(), 0.0);
    }

    #[test]
    fn determinant_preserves_nonfinite_backend_error() {
        let matrix = Matrix::<2>::try_from_rows([[1.0e200, 0.0], [0.0, 1.0e200]]).unwrap();

        assert_matches!(determinant(&matrix), Err(LaError::NonFinite { .. }));
    }
}
