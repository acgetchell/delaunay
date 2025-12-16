//! Matrix operations.
//!
//! This module provides small, stack-allocated linear algebra helpers used by
//! geometric predicates and utilities.

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
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum MatrixError {
    /// Matrix is singular.
    #[error("Matrix is singular!")]
    SingularMatrix,
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

#[inline]
pub(crate) fn get_unchecked<const D: usize>(m: &Matrix<D>, r: usize, c: usize) -> f64 {
    m.get(r, c)
        .unwrap_or_else(|| unreachable!("matrix index out of bounds: ({r}, {c}) for {D}x{D}"))
}

#[inline]
pub(crate) fn set_unchecked<const D: usize>(m: &mut Matrix<D>, r: usize, c: usize, value: f64) {
    let ok = m.set(r, c, value);
    debug_assert!(ok, "matrix index out of bounds: ({r}, {c}) for {D}x{D}");
}

/// Compute an LU-based determinant, returning 0.0 for singular matrices.
#[inline]
#[must_use]
pub fn determinant<const D: usize>(m: Matrix<D>) -> f64 {
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
#[must_use]
pub fn adaptive_tolerance<const D: usize>(matrix: &Matrix<D>, base_tol: f64) -> f64 {
    let nrows = D;
    let ncols = D;

    // Check if the last column is (approximately) all ones.
    let last_col_is_all_ones = ncols > 0
        && (0..nrows).all(|i| (get_unchecked(matrix, i, ncols - 1) - 1.0).abs() <= f64::EPSILON);

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
            row_sum += get_unchecked(matrix, i, j).abs();
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

    macro_rules! gen_adaptive_tol_tests {
        ($d:literal) => {
            pastey::paste! {
                #[test]
                fn [<adaptive_tolerance_ignores_constant_one_last_col_ $d d>]() {
                    let n = $d + 1; // orientation/insphere-like square matrix
                    let base = 1e-12;

                    let tol = with_la_stack_matrix!(n, |m| {
                        for i in 0..n {
                            set_unchecked(&mut m, i, n - 1, 1.0);
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
                            set_unchecked(&mut m, i, n - 1, 2.0);
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
