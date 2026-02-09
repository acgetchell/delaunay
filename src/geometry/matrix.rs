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
/// With `MAX_STACK_MATRIX_DIM = 18`, we support up to `D = 16` for insphere.
pub const MAX_STACK_MATRIX_DIM: usize = 18;

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
/// Panics if `k` exceeds [`MAX_STACK_MATRIX_DIM`] (18).
macro_rules! with_la_stack_matrix {
    ($k:expr, |$m:ident| $body:block) => {{
        match $k {
            0 => {
                let mut $m = $crate::geometry::matrix::Matrix::<0>::zero();
                $body
            }
            1 => {
                let mut $m = $crate::geometry::matrix::Matrix::<1>::zero();
                $body
            }
            2 => {
                let mut $m = $crate::geometry::matrix::Matrix::<2>::zero();
                $body
            }
            3 => {
                let mut $m = $crate::geometry::matrix::Matrix::<3>::zero();
                $body
            }
            4 => {
                let mut $m = $crate::geometry::matrix::Matrix::<4>::zero();
                $body
            }
            5 => {
                let mut $m = $crate::geometry::matrix::Matrix::<5>::zero();
                $body
            }
            6 => {
                let mut $m = $crate::geometry::matrix::Matrix::<6>::zero();
                $body
            }
            7 => {
                let mut $m = $crate::geometry::matrix::Matrix::<7>::zero();
                $body
            }
            8 => {
                let mut $m = $crate::geometry::matrix::Matrix::<8>::zero();
                $body
            }
            9 => {
                let mut $m = $crate::geometry::matrix::Matrix::<9>::zero();
                $body
            }
            10 => {
                let mut $m = $crate::geometry::matrix::Matrix::<10>::zero();
                $body
            }
            11 => {
                let mut $m = $crate::geometry::matrix::Matrix::<11>::zero();
                $body
            }
            12 => {
                let mut $m = $crate::geometry::matrix::Matrix::<12>::zero();
                $body
            }
            13 => {
                let mut $m = $crate::geometry::matrix::Matrix::<13>::zero();
                $body
            }
            14 => {
                let mut $m = $crate::geometry::matrix::Matrix::<14>::zero();
                $body
            }
            15 => {
                let mut $m = $crate::geometry::matrix::Matrix::<15>::zero();
                $body
            }
            16 => {
                let mut $m = $crate::geometry::matrix::Matrix::<16>::zero();
                $body
            }
            17 => {
                let mut $m = $crate::geometry::matrix::Matrix::<17>::zero();
                $body
            }
            18 => {
                let mut $m = $crate::geometry::matrix::Matrix::<18>::zero();
                $body
            }
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
        Err(LaError::NonFinite { .. }) => f64::NAN,
    }
}

/// Compute adaptive tolerance scaled by matrix magnitude (infinity norm).
///
/// This computes: `base_tol` + `rel_factor` * ||A||_∞, where ||A||_∞ is the maximum
/// absolute row sum. If the last column is (approximately) all ones, it is
/// excluded from the magnitude estimate to avoid over-inflating tolerance on
/// small simplices (common in orientation/insphere matrices).
///
/// # Examples
///
/// ```rust
/// use delaunay::geometry::matrix::{adaptive_tolerance, Matrix};
///
/// let mut m = Matrix::<2>::zero();
/// m.set(0, 0, 1.0);
/// m.set(1, 1, 1.0);
/// let tol = adaptive_tolerance(&m, 1e-12);
/// assert!(tol >= 1e-12);
/// ```
#[must_use]
pub fn adaptive_tolerance<const D: usize>(matrix: &Matrix<D>, base_tol: f64) -> f64 {
    let nrows = D;
    let ncols = D;

    // Check if the last column is (approximately) all ones.
    let last_col_is_all_ones = ncols > 0
        && (0..nrows).all(|i| (matrix_get(matrix, i, ncols - 1) - 1.0).abs() <= f64::EPSILON);

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
            row_sum += matrix_get(matrix, i, j).abs();
        }
        if row_sum > max_row_sum {
            max_row_sum = row_sum;
        }
    }

    let rel_factor = 1e-12f64;
    rel_factor.mul_add(max_row_sum, base_tol)
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

    macro_rules! gen_adaptive_tol_tests {
        ($d:literal) => {
            pastey::paste! {
                #[test]
                fn [<adaptive_tolerance_ignores_constant_one_last_col_ $d d>]() {
                    let n = $d + 1; // orientation/insphere-like square matrix
                    let base = 1e-12;

                    let tol = with_la_stack_matrix!(n, |m| {
                        for i in 0..n {
                            matrix_set(&mut m, i, n - 1, 1.0);
                        }
                        adaptive_tolerance(&m, base)
                    });

                    assert_relative_eq!(tol, base, epsilon = 1e-18);
                }

                #[test]
                fn [<adaptive_tolerance_includes_non_one_last_col_ $d d>]() {
                    let n = $d + 1;
                    let base = 1e-12;

                    let tol = with_la_stack_matrix!(n, |m| {
                        for i in 0..n {
                            matrix_set(&mut m, i, n - 1, 2.0);
                        }
                        adaptive_tolerance(&m, base)
                    });

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
}
