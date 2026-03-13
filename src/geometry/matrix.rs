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
/// use delaunay::geometry::matrix::MatrixError;
///
/// let err = MatrixError::SingularMatrix;
/// assert!(matches!(err, MatrixError::SingularMatrix));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum MatrixError {
    /// Matrix is singular.
    #[error("Matrix is singular!")]
    SingularMatrix,
}

/// Error type for stack-matrix dispatch.
#[derive(Debug, Error)]
pub(crate) enum StackMatrixDispatchError {
    /// The requested matrix size is not supported by the stack-matrix dispatcher.
    #[error("unsupported stack matrix size: {k} (max {max})")]
    UnsupportedDim {
        /// Requested matrix dimension.
        k: usize,
        /// Maximum supported matrix dimension.
        max: usize,
    },
    /// A linear algebra error originating from `la-stack`.
    #[error(transparent)]
    La(#[from] LaError),
}

/// Default tolerance for matrix singularity checks.
///
/// This value is chosen to be appropriately small for typical geometric computations
/// while being large enough to handle floating-point precision issues.
pub const SINGULARITY_TOLERANCE: f64 = 1e-12;

/// Dispatch a runtime `k` (matrix dimension) to a stack-allocated `la_stack::Matrix<k>`.
///
/// This is used to bridge the gap between const-generic sizes and `D+1`/`D+2` shapes,
/// which are not available as stable const-generic expressions.
///
/// # Panics
///
/// Panics if `k` exceeds [`MAX_STACK_MATRIX_DIM`] (7).
macro_rules! with_la_stack_matrix {
    ($k:expr, |$m:ident| $body:block) => {{
        with_la_stack_matrix!(@dispatch $k, $m, $body,
            0, 1, 2, 3, 4, 5, 6, 7)
    }};
    (@dispatch $k:expr, $m:ident, $body:block, $($n:literal),+) => {{
        match $k {
            $(
                $n => {
                    #[allow(unused_mut)]
                    let mut $m = $crate::geometry::matrix::Matrix::<$n>::zero();
                    $body
                }
            )+
            _ => panic!(
                "unsupported stack matrix size: {k} (max {max})",
                k = $k,
                max = $crate::geometry::matrix::MAX_STACK_MATRIX_DIM
            ),
        }
    }};
}

/// Fallible variant of [`with_la_stack_matrix!`] that returns an error instead of panicking.
///
/// The provided block must evaluate to `Result<_, E>`, where `E` can be constructed from
/// [`StackMatrixDispatchError`].
macro_rules! try_with_la_stack_matrix {
    ($k:expr, |$m:ident| $body:block) => {{
        let k = $k;
        if k > $crate::geometry::matrix::MAX_STACK_MATRIX_DIM {
            Err(
                $crate::geometry::matrix::StackMatrixDispatchError::UnsupportedDim {
                    k,
                    max: $crate::geometry::matrix::MAX_STACK_MATRIX_DIM,
                }
                .into(),
            )
        } else {
            with_la_stack_matrix!(k, |$m| $body)
        }
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
pub(crate) fn matrix_get<const D: usize>(m: &Matrix<D>, r: usize, c: usize) -> f64 {
    m.get(r, c)
        .unwrap_or_else(|| unreachable!("matrix index out of bounds: ({r}, {c}) for {D}x{D}"))
}

#[inline]
pub(crate) fn matrix_set<const D: usize>(m: &mut Matrix<D>, r: usize, c: usize, value: f64) {
    let ok = m.set(r, c, value);
    assert!(ok, "matrix index out of bounds: ({r}, {c}) for {D}x{D}");
}

/// Compute an LU-based determinant, returning 0.0 for singular matrices.
///
/// # Examples
///
/// ```rust
/// use delaunay::geometry::matrix::{determinant, Matrix};
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
                    matrix_set(&mut original, i, j, val);
                    val += 1.0;
                }
            }

            let zero = matrix_zero_like(&original);

            // All entries must be zero.
            for i in 0..k {
                for j in 0..k {
                    assert_relative_eq!(matrix_get(&zero, i, j), 0.0);
                }
            }

            // Original must be unchanged.
            let mut expected = 1.0_f64;
            for i in 0..k {
                for j in 0..k {
                    assert_relative_eq!(matrix_get(&original, i, j), expected);
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
                assert_relative_eq!(matrix_get(&zero, 0, 0), 0.0);
                assert_relative_eq!(matrix_get(&zero, k - 1, k - 1), 0.0);
            });
        }
    }
}
