//! Simulation of Simplicity (SoS) for deterministic degeneracy resolution.
//!
//! When geometric predicates (orientation, in-sphere) evaluate to exactly zero —
//! meaning points are exactly co-planar, co-spherical, etc. — the SoS technique
//! provides a deterministic non-zero answer without modifying any coordinates.
//!
//! # Algorithm
//!
//! Each input point is conceptually perturbed by an infinitesimal amount
//! `ε^(i+1)` in coordinate `j`, where `i` is the point's index in the input
//! array.  The perturbation is purely symbolic: no floating-point values are
//! changed.  The sign of the perturbed determinant is determined by evaluating
//! sub-determinants (minors) in a specific order until a non-zero one is found.
//!
//! For an `n×n` matrix whose determinant is exactly zero, the SoS expansion
//! produces a sequence of minors ordered by the powers of ε.  The first
//! non-zero minor in this sequence determines the sign.
//!
//! # Key Properties
//!
//! - **Deterministic**: same input always produces the same sign
//! - **No coordinate modification**: purely a decision rule
//! - **Always non-zero**: returns ±1, never 0
//! - **Dimension-generic**: works for any D
//!
//! # References
//!
//! - Edelsbrunner, H. and Mücke, E. P. "Simulation of Simplicity: A Technique
//!   to Cope with Degenerate Cases in Geometric Algorithms." ACM Transactions
//!   on Graphics, 9(1):66–104, 1990.

#![forbid(unsafe_code)]

use crate::geometry::matrix::{Matrix, matrix_set};
use crate::geometry::point::Point;

// =============================================================================
// PUBLIC API
// =============================================================================

/// Compute the `SoS` orientation sign for a degenerate simplex.
///
/// Given `D+1` points whose orientation determinant is exactly zero, this
/// function returns a deterministic non-zero sign (±1) using Simulation of
/// Simplicity.
///
/// The orientation matrix has the form:
///
/// ```text
/// | x₀  y₀  z₀ … 1 |
/// | x₁  y₁  z₁ … 1 |
/// | …               |
/// | xD  yD  zD … 1 |
/// ```
///
/// # Arguments
///
/// * `points` - Exactly `D+1` points (as f64 coordinate arrays) forming the
///   degenerate simplex.  The index of each point in this slice determines its
///   symbolic perturbation priority.
///
/// # Returns
///
/// `1` or `-1`.  Never returns `0`.
///
/// # Panics
///
/// Panics if `points.len() != D + 1`.
#[must_use]
pub fn sos_orientation_sign<const D: usize>(points: &[Point<f64, D>]) -> i32 {
    assert_eq!(
        points.len(),
        D + 1,
        "SoS orientation requires exactly D+1 = {} points, got {}",
        D + 1,
        points.len()
    );

    let n = D + 1; // matrix dimension

    // Build the orientation matrix (same layout as simplex_orientation).
    // Rows = points, columns = [coords..., 1].
    let coords: Vec<[f64; D]> = points.iter().map(|p| *p.coords()).collect();

    // SoS expansion: we remove rows/columns from the matrix in order of
    // decreasing point index and the last column (the "1" column) first.
    //
    // The perturbation assigns ε^(i+1) to point i in each coordinate.
    // For the orientation matrix, the SoS sign is determined by iterating
    // over subsets of rows to remove, in reverse lexicographic order of
    // point indices.
    //
    // Simplified SoS for orientation (Edelsbrunner & Mücke §4.2):
    // We remove one row at a time from the bottom (highest index) and
    // evaluate the sign of the (n-1)×(n-1) minor formed by the remaining
    // rows and columns 0..D-1 (dropping the constant "1" column).
    // The first non-zero minor determines the sign, with an appropriate
    // sign correction for the row/column removal.

    // Try removing each row i, starting from the last (highest perturbation
    // priority = smallest ε power = dominant term).
    for remove_row in (0..n).rev() {
        // Build the (n-1)×(n-1) minor: all rows except `remove_row`,
        // all columns except the last one (the "1" column).
        let minor_sign = orientation_minor_sign::<D>(&coords, remove_row);

        if minor_sign != 0 {
            // Sign correction: removing row `remove_row` and column `n-1`
            // from an n×n matrix contributes (-1)^(remove_row + (n-1)).
            let cofactor_sign = if (remove_row + (n - 1)).is_multiple_of(2) {
                1
            } else {
                -1
            };
            return cofactor_sign * minor_sign;
        }
    }

    // For a full-rank symbolic perturbation this should not happen, but
    // as a fallback return +1 (consistent with the convention that the
    // perturbation resolves toward positive orientation).
    1
}

/// Compute the `SoS` in-sphere sign for a degenerate configuration.
///
/// Given `D+1` simplex points and a test point whose in-sphere determinant is
/// exactly zero (test point lies exactly on the circumsphere), this function
/// returns a deterministic non-zero sign (±1) using Simulation of Simplicity.
///
/// The lifted in-sphere matrix (relative coordinates centered on `simplex[0]`)
/// has the form:
///
/// ```text
/// | Δx₁  Δy₁  … ‖Δp₁‖² |
/// | Δx₂  Δy₂  … ‖Δp₂‖² |
/// | …                    |
/// | Δxₜ  Δyₜ  … ‖Δpₜ‖² |
/// ```
///
/// where `Δpᵢ = pᵢ - p₀` and `t` is the test point.
///
/// # Arguments
///
/// * `simplex` - Exactly `D+1` points defining the simplex (f64 coordinates).
/// * `test` - The test point to classify.
///
/// # Returns
///
/// `1` (INSIDE) or `-1` (OUTSIDE).  Never returns `0`.
///
/// # Panics
///
/// Panics if `simplex.len() != D + 1`.
#[must_use]
pub fn sos_insphere_sign<const D: usize>(simplex: &[Point<f64, D>], test: &Point<f64, D>) -> i32 {
    assert_eq!(
        simplex.len(),
        D + 1,
        "SoS insphere requires exactly D+1 = {} simplex points, got {}",
        D + 1,
        simplex.len()
    );

    let n = D + 1; // matrix dimension (lifted)

    // Build the lifted insphere matrix using relative coordinates (same as
    // insphere_lifted): rows = simplex[1..D] and test, centered on simplex[0].
    let ref_coords = simplex[0].coords();

    // Collect all D+1 rows: D rows from simplex[1..] + 1 row from test.
    let mut rows: Vec<[f64; D]> = Vec::with_capacity(n);
    let mut lifted_col: Vec<f64> = Vec::with_capacity(n);

    for point in simplex.iter().skip(1) {
        let mut rel = [0.0f64; D];
        for j in 0..D {
            rel[j] = point.coords()[j] - ref_coords[j];
        }
        let sq_norm: f64 = rel.iter().map(|&x| x * x).sum();
        rows.push(rel);
        lifted_col.push(sq_norm);
    }

    // Test point row.
    let mut test_rel = [0.0f64; D];
    for j in 0..D {
        test_rel[j] = test.coords()[j] - ref_coords[j];
    }
    let test_sq_norm: f64 = test_rel.iter().map(|&x| x * x).sum();
    rows.push(test_rel);
    lifted_col.push(test_sq_norm);

    // SoS expansion for the insphere matrix.
    // Same approach: remove rows from the bottom, evaluate minor sign.
    for remove_row in (0..n).rev() {
        let minor_sign = insphere_minor_sign::<D>(&rows, &lifted_col, remove_row);

        if minor_sign != 0 {
            // Sign correction for removing row `remove_row` and last column (D).
            let cofactor_sign = if (remove_row + D).is_multiple_of(2) {
                1
            } else {
                -1
            };

            // The lifted insphere uses orient_sign = -rel_sign (same as
            // insphere_lifted).  For SoS we apply the same convention:
            // the cofactor sign already accounts for the row/column removal.
            return cofactor_sign * minor_sign;
        }
    }

    // Fallback: return INSIDE (+1).
    1
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

/// Compute the sign of the (D×D) minor obtained by removing `remove_row`
/// from the orientation coordinate block (all rows, first D columns).
///
/// Uses `la-stack` exact sign when available.
fn orientation_minor_sign<const D: usize>(coords: &[[f64; D]], remove_row: usize) -> i32 {
    let minor_dim = D; // (n-1) × (n-1) but we drop the "1" column → D × D

    if minor_dim == 0 {
        return 1; // 0×0 determinant = 1
    }

    // Use the la-stack macro dispatch for the minor matrix.
    with_la_stack_matrix!(minor_dim, |matrix| {
        let mut r = 0;
        for (i, coord_row) in coords.iter().enumerate() {
            if i == remove_row {
                continue;
            }
            for (j, &val) in coord_row.iter().enumerate() {
                matrix_set(&mut matrix, r, j, val);
            }
            r += 1;
        }

        exact_det_sign(&matrix, minor_dim)
    })
}

/// Compute the sign of the (D×D) minor obtained by removing `remove_row`
/// from the insphere lifted matrix and dropping the last column (lifted norm).
///
/// The minor uses the first D columns (relative coordinates only).
fn insphere_minor_sign<const D: usize>(
    rows: &[[f64; D]],
    _lifted_col: &[f64],
    remove_row: usize,
) -> i32 {
    let minor_dim = D;

    if minor_dim == 0 {
        return 1;
    }

    with_la_stack_matrix!(minor_dim, |matrix| {
        let mut r = 0;
        for (i, data_row) in rows.iter().enumerate() {
            if i == remove_row {
                continue;
            }
            for (j, &val) in data_row.iter().enumerate() {
                matrix_set(&mut matrix, r, j, val);
            }
            r += 1;
        }

        exact_det_sign(&matrix, minor_dim)
    })
}

/// Compute the exact sign of a matrix determinant using la-stack.
///
/// Uses the two-stage approach:
/// 1. `det_direct()` + `det_errbound()` for D ≤ 4 (provable fast filter)
/// 2. `det_sign_exact()` for exact result
///
/// Returns -1, 0, or +1.
fn exact_det_sign<const N: usize>(matrix: &Matrix<N>, _dim: usize) -> i32 {
    // Stage 1: fast filter with provable error bound (D ≤ 4).
    if let (Some(det), Some(bound)) = (matrix.det_direct(), matrix.det_errbound())
        && det.is_finite()
        && bound.is_finite()
    {
        if det > bound {
            return 1;
        }
        if det < -bound {
            return -1;
        }
        // |det| ≤ bound: inconclusive, fall through to exact.
    }

    // Stage 2: exact sign via Bareiss algorithm in BigRational.
    matrix.det_sign_exact().map_or(0, i32::from)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    /// Build a co-linear 2D point set (3 collinear points on x-axis).
    fn collinear_2d() -> Vec<Point<f64, 2>> {
        vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([2.0, 0.0]),
        ]
    }

    /// Build a co-planar 3D point set (4 coplanar points in z=0 plane).
    fn coplanar_3d() -> Vec<Point<f64, 3>> {
        vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 0.0]),
        ]
    }

    /// Build a 4D degenerate simplex (all points in w=0 hyperplane).
    fn degenerate_4d() -> Vec<Point<f64, 4>> {
        vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 0.5, 0.0]),
        ]
    }

    /// Build a 3D simplex and a test point on its circumsphere boundary.
    /// The standard unit tetrahedron has circumcenter at (0.5, 0.5, 0.5)
    /// with circumradius sqrt(3)/2. A vertex is exactly on the circumsphere.
    fn cospherical_3d() -> (Vec<Point<f64, 3>>, Point<f64, 3>) {
        let simplex = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        // (1,1,0) lies on the circumsphere of this tetrahedron:
        // distance from circumcenter (0.5,0.5,0.5) = sqrt(0.25+0.25+0.25) = sqrt(0.75)
        // circumradius = sqrt(0.75), so it's exactly on the boundary.
        let test = Point::new([1.0, 1.0, 0.0]);
        (simplex, test)
    }

    // =========================================================================
    // SOS ORIENTATION TESTS
    // =========================================================================

    #[test]
    fn test_sos_orientation_collinear_2d_returns_nonzero() {
        let points = collinear_2d();
        let sign = sos_orientation_sign(&points);
        assert!(sign == 1 || sign == -1, "SoS must return ±1, got {sign}");
    }

    #[test]
    fn test_sos_orientation_collinear_2d_is_deterministic() {
        let points = collinear_2d();
        let sign1 = sos_orientation_sign(&points);
        let sign2 = sos_orientation_sign(&points);
        assert_eq!(sign1, sign2, "SoS must be deterministic");
    }

    #[test]
    fn test_sos_orientation_coplanar_3d_returns_nonzero() {
        let points = coplanar_3d();
        let sign = sos_orientation_sign(&points);
        assert!(sign == 1 || sign == -1, "SoS must return ±1, got {sign}");
    }

    #[test]
    fn test_sos_orientation_coplanar_3d_is_deterministic() {
        let points = coplanar_3d();
        let sign1 = sos_orientation_sign(&points);
        let sign2 = sos_orientation_sign(&points);
        assert_eq!(sign1, sign2, "SoS must be deterministic");
    }

    #[test]
    fn test_sos_orientation_degenerate_4d_returns_nonzero() {
        let points = degenerate_4d();
        let sign = sos_orientation_sign(&points);
        assert!(sign == 1 || sign == -1, "SoS must return ±1, got {sign}");
    }

    #[test]
    fn test_sos_orientation_degenerate_4d_is_deterministic() {
        let points = degenerate_4d();
        let sign1 = sos_orientation_sign(&points);
        let sign2 = sos_orientation_sign(&points);
        assert_eq!(sign1, sign2, "SoS must be deterministic");
    }

    #[test]
    fn test_sos_orientation_nondegenerate_returns_correct_sign() {
        // Positive orientation triangle
        let positive = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];
        let sign = sos_orientation_sign(&positive);
        // The actual orientation is positive; SoS should agree since the
        // matrix is non-singular (it won't reach the SoS path in
        // AdaptiveKernel, but the function itself should still give the
        // correct sign).
        assert_eq!(sign, 1, "Non-degenerate positive triangle should return +1");
    }

    #[test]
    fn test_sos_orientation_5d_degenerate() {
        // 5D degenerate: all points have last coord = 0
        let points = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 0.5, 0.5, 0.0]),
        ];
        let sign = sos_orientation_sign(&points);
        assert!(
            sign == 1 || sign == -1,
            "SoS must return ±1 for 5D, got {sign}"
        );
    }

    // =========================================================================
    // SOS INSPHERE TESTS
    // =========================================================================

    #[test]
    fn test_sos_insphere_cospherical_3d_returns_nonzero() {
        let (simplex, test) = cospherical_3d();
        let sign = sos_insphere_sign(&simplex, &test);
        assert!(
            sign == 1 || sign == -1,
            "SoS insphere must return ±1, got {sign}"
        );
    }

    #[test]
    fn test_sos_insphere_cospherical_3d_is_deterministic() {
        let (simplex, test) = cospherical_3d();
        let sign1 = sos_insphere_sign(&simplex, &test);
        let sign2 = sos_insphere_sign(&simplex, &test);
        assert_eq!(sign1, sign2, "SoS insphere must be deterministic");
    }

    #[test]
    fn test_sos_insphere_vertex_on_boundary_2d() {
        // 2D: test point is a vertex of the simplex (exactly on circumsphere)
        let simplex = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];
        // The circumcircle passes through all vertices. Test with a
        // different co-circular point.
        let test = Point::new([1.0, 1.0]);
        let sign = sos_insphere_sign(&simplex, &test);
        assert!(
            sign == 1 || sign == -1,
            "SoS insphere must return ±1, got {sign}"
        );
    }

    #[test]
    fn test_sos_insphere_4d_cospherical() {
        // 4D: 5 points on a unit 3-sphere centered at origin.
        let simplex = vec![
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
            Point::new([-1.0, 0.0, 0.0, 0.0]),
        ];
        // Another point on the same unit 3-sphere.
        let test = Point::new([0.0, -1.0, 0.0, 0.0]);
        let sign = sos_insphere_sign(&simplex, &test);
        assert!(
            sign == 1 || sign == -1,
            "SoS insphere must return ±1 for 4D, got {sign}"
        );
    }

    #[test]
    fn test_sos_insphere_deterministic_across_calls() {
        let simplex = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let test = Point::new([1.0, 1.0, 0.0]);
        let results: Vec<i32> = (0..10)
            .map(|_| sos_insphere_sign(&simplex, &test))
            .collect();
        assert!(
            results.iter().all(|&r| r == results[0]),
            "SoS insphere must be deterministic across calls"
        );
    }

    // =========================================================================
    // EDGE CASE TESTS
    // =========================================================================

    #[test]
    #[should_panic(expected = "SoS orientation requires exactly D+1")]
    fn test_sos_orientation_wrong_point_count_panics() {
        let points = vec![Point::new([0.0, 0.0]), Point::new([1.0, 0.0])];
        let _ = sos_orientation_sign(&points);
    }

    #[test]
    #[should_panic(expected = "SoS insphere requires exactly D+1")]
    fn test_sos_insphere_wrong_simplex_count_panics() {
        let simplex = vec![Point::new([0.0, 0.0]), Point::new([1.0, 0.0])];
        let test = Point::new([0.5, 0.5]);
        let _ = sos_insphere_sign(&simplex, &test);
    }

    #[test]
    fn test_exact_det_sign_identity_2x2() {
        let m = Matrix::<2>::from_rows([[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(exact_det_sign(&m, 2), 1);
    }

    #[test]
    fn test_exact_det_sign_singular_2x2() {
        let m = Matrix::<2>::from_rows([[1.0, 2.0], [2.0, 4.0]]);
        assert_eq!(exact_det_sign(&m, 2), 0);
    }

    #[test]
    fn test_exact_det_sign_negative_2x2() {
        let m = Matrix::<2>::from_rows([[0.0, 1.0], [1.0, 0.0]]);
        assert_eq!(exact_det_sign(&m, 2), -1);
    }
}
