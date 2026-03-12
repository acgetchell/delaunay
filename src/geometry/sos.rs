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
//! sub-determinants (cofactors) in a specific order until a non-zero one is
//! found.
//!
//! For an `n×n` matrix whose determinant is exactly zero, the SoS expansion
//! iterates over (row, column) positions in reverse order.  The cofactor at
//! the first non-zero position determines the sign.
//!
//! # Key Properties
//!
//! - **Deterministic**: same input always produces the same sign
//! - **No coordinate modification**: purely a decision rule
//! - **Always non-zero**: returns ±1, never 0
//! - **Dimension-generic**: works for any D
//! - **Translation-invariant**: orientation minors retain the homogeneous "1"
//!   column, ensuring that shifting all points by a constant vector does not
//!   change the result
//!
//! # References
//!
//! - Edelsbrunner, H. and Mücke, E. P. "Simulation of Simplicity: A Technique
//!   to Cope with Degenerate Cases in Geometric Algorithms." ACM Transactions
//!   on Graphics, 9(1):66–104, 1990.

#![forbid(unsafe_code)]

use crate::geometry::matrix::{Matrix, matrix_set};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::CoordinateConversionError;

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
/// # Translation Invariance
///
/// The `SoS` expansion computes cofactors by removing one row and one
/// *coordinate* column, always retaining the constant "1" column.  The
/// resulting D×D minors are translation-invariant because the "1" column
/// allows row reduction that cancels any uniform translation of the inputs.
///
/// # Arguments
///
/// * `points` - Exactly `D+1` points (as f64 coordinate arrays) forming the
///   degenerate simplex.  The index of each point in this slice determines its
///   symbolic perturbation priority.
///
/// # Returns
///
/// `Ok(1)` or `Ok(-1)`.  Never returns `Ok(0)`.
///
/// # Errors
///
/// Returns [`CoordinateConversionError::ConversionFailed`] if
/// `points.len() != D + 1`.
pub fn sos_orientation_sign<const D: usize>(
    points: &[Point<f64, D>],
) -> Result<i32, CoordinateConversionError> {
    if points.len() != D + 1 {
        return Err(CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: format!(
                "SoS orientation requires exactly D+1 = {} points, got {}",
                D + 1,
                points.len()
            ),
            from_type: "point count",
            to_type: "valid simplex",
        });
    }

    let n = D + 1; // matrix dimension

    // Build coordinate block from the (D+1)×(D+1) homogeneous orientation
    // matrix.  Full matrix columns: [coords (D cols), 1].
    //
    // The constant "1" column is NOT symbolically perturbed, so the SoS
    // expansion only iterates over coordinate columns (j = 0..D-1).
    let coords: Vec<[f64; D]> = points.iter().map(|p| *p.coords()).collect();

    // Edelsbrunner & Mücke SoS expansion: iterate (row, coord_col) in reverse
    // order.  The first non-zero cofactor determines the sign.
    //
    // For each (remove_row, remove_col), the minor is D×D: the remaining D-1
    // coordinate columns plus the constant "1" column.  Because the "1" column
    // is always retained, the minor's determinant is translation-invariant.
    for remove_row in (0..n).rev() {
        for remove_col in (0..D).rev() {
            let minor_sign = orientation_cofactor_det::<D>(&coords, remove_row, remove_col);

            if minor_sign != 0 {
                let cofactor_sign = if (remove_row + remove_col).is_multiple_of(2) {
                    1
                } else {
                    -1
                };
                return Ok(cofactor_sign * minor_sign);
            }
        }
    }

    // Fallback: return +1 (all cofactors zero — should not happen for a
    // genuine SoS perturbation, but provides a safe default).
    Ok(1)
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
/// The `SoS` expansion uses cofactors from this full (D+1)×(D+1) lifted matrix,
/// including the lifted (squared-norm) column, so the expansion correctly
/// accounts for the complete insphere geometry.
///
/// # Arguments
///
/// * `simplex` - Exactly `D+1` points defining the simplex (f64 coordinates).
/// * `test` - The test point to classify.
///
/// # Returns
///
/// `Ok(1)` (INSIDE) or `Ok(-1)` (OUTSIDE).  Never returns `Ok(0)`.
///
/// # Errors
///
/// Returns [`CoordinateConversionError::ConversionFailed`] if
/// `simplex.len() != D + 1`.
pub fn sos_insphere_sign<const D: usize>(
    simplex: &[Point<f64, D>],
    test: &Point<f64, D>,
) -> Result<i32, CoordinateConversionError> {
    if simplex.len() != D + 1 {
        return Err(CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: format!(
                "SoS insphere requires exactly D+1 = {} simplex points, got {}",
                D + 1,
                simplex.len()
            ),
            from_type: "point count",
            to_type: "valid simplex",
        });
    }

    let n = D + 1; // matrix dimension: (D+1)×(D+1)

    // Build the full (D+1)×(D+1) lifted insphere matrix using relative
    // coordinates centered on simplex[0].
    // Columns: [Δcoords (D cols), ‖Δp‖² (1 col)] = D+1 columns total.
    let base_coords = simplex[0].coords();

    let mut rel_coords: Vec<[f64; D]> = Vec::with_capacity(n);
    let mut lifted_col: Vec<f64> = Vec::with_capacity(n);

    for point in simplex.iter().skip(1) {
        let mut rel = [0.0f64; D];
        for j in 0..D {
            rel[j] = point.coords()[j] - base_coords[j];
        }
        let sq_norm: f64 = rel.iter().fold(0.0f64, |acc, &x| x.mul_add(x, acc));
        rel_coords.push(rel);
        lifted_col.push(sq_norm);
    }

    // Test point row.
    let mut test_rel = [0.0f64; D];
    for j in 0..D {
        test_rel[j] = test.coords()[j] - base_coords[j];
    }
    let test_sq_norm: f64 = test_rel.iter().fold(0.0f64, |acc, &x| x.mul_add(x, acc));
    rel_coords.push(test_rel);
    lifted_col.push(test_sq_norm);

    // Edelsbrunner & Mücke SoS expansion on the full (D+1)×(D+1) lifted
    // matrix.  All D+1 columns (D coordinate + 1 lifted) are symbolically
    // perturbed, so we iterate over all column positions.
    for remove_row in (0..n).rev() {
        for remove_col in (0..n).rev() {
            let minor_sign =
                insphere_cofactor_det::<D>(&rel_coords, &lifted_col, remove_row, remove_col);

            if minor_sign != 0 {
                let cofactor_sign = if (remove_row + remove_col).is_multiple_of(2) {
                    1
                } else {
                    -1
                };
                return Ok(cofactor_sign * minor_sign);
            }
        }
    }

    // Fallback: return INSIDE (+1).
    Ok(1)
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

/// Compute the sign of the D×D minor from the (D+1)×(D+1) homogeneous
/// orientation matrix, removing `remove_row` and coordinate column
/// `remove_col`.
///
/// The removed column index refers to the coordinate block (0..D).  The
/// constant "1" column is always retained, ensuring translation invariance.
/// The resulting minor has D-1 coordinate columns + 1 "one" column = D columns
/// and D rows, giving a D×D determinant.
fn orientation_cofactor_det<const D: usize>(
    coords: &[[f64; D]],
    remove_row: usize,
    remove_col: usize,
) -> i32 {
    let minor_dim = D; // (D+1) rows − 1; (D−1) coord cols + 1 "one" col = D
    if minor_dim == 0 {
        return 1; // 0×0 determinant = 1
    }

    with_la_stack_matrix!(minor_dim, |matrix| {
        let mut r = 0;
        for (i, coord_row) in coords.iter().enumerate() {
            if i == remove_row {
                continue;
            }
            let mut c = 0;
            // Coordinate columns, skipping the removed one.
            for (j, &val) in coord_row.iter().enumerate() {
                if j == remove_col {
                    continue;
                }
                matrix_set(&mut matrix, r, c, val);
                c += 1;
            }
            // Constant "1" column (always present in the minor).
            matrix_set(&mut matrix, r, c, 1.0);
            r += 1;
        }

        exact_det_sign(&matrix, minor_dim)
    })
}

/// Compute the sign of the D×D minor from the (D+1)×(D+1) lifted insphere
/// matrix, removing `remove_row` and column `remove_col`.
///
/// The full matrix has D relative-coordinate columns (indices 0..D-1) followed
/// by 1 lifted (‖Δp‖²) column (index D).
fn insphere_cofactor_det<const D: usize>(
    rel_coords: &[[f64; D]],
    lifted_col: &[f64],
    remove_row: usize,
    remove_col: usize,
) -> i32 {
    let minor_dim = D; // (D+1) − 1 = D
    if minor_dim == 0 {
        return 1;
    }

    let num_rows = D + 1;
    let num_cols = D + 1;

    with_la_stack_matrix!(minor_dim, |matrix| {
        let mut r = 0;
        for i in 0..num_rows {
            if i == remove_row {
                continue;
            }
            let mut c = 0;
            // Iterate over all columns of the lifted matrix.  Column indices
            // 0..D are relative-coordinate columns; index D is the lifted
            // (squared-norm) column.  We skip `remove_col` and pack the rest
            // into the minor.
            #[expect(clippy::needless_range_loop, reason = "mixed data sources")]
            for j in 0..num_cols {
                if j == remove_col {
                    continue;
                }
                let val = if j < D {
                    rel_coords[i][j]
                } else {
                    lifted_col[i]
                };
                matrix_set(&mut matrix, r, c, val);
                c += 1;
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
pub(crate) fn exact_det_sign<const N: usize>(matrix: &Matrix<N>, _dim: usize) -> i32 {
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
        let sign = sos_orientation_sign(&points).unwrap();
        assert!(sign == 1 || sign == -1, "SoS must return ±1, got {sign}");
    }

    #[test]
    fn test_sos_orientation_collinear_2d_is_deterministic() {
        let points = collinear_2d();
        let sign1 = sos_orientation_sign(&points).unwrap();
        let sign2 = sos_orientation_sign(&points).unwrap();
        assert_eq!(sign1, sign2, "SoS must be deterministic");
    }

    #[test]
    fn test_sos_orientation_coplanar_3d_returns_nonzero() {
        let points = coplanar_3d();
        let sign = sos_orientation_sign(&points).unwrap();
        assert!(sign == 1 || sign == -1, "SoS must return ±1, got {sign}");
    }

    #[test]
    fn test_sos_orientation_coplanar_3d_is_deterministic() {
        let points = coplanar_3d();
        let sign1 = sos_orientation_sign(&points).unwrap();
        let sign2 = sos_orientation_sign(&points).unwrap();
        assert_eq!(sign1, sign2, "SoS must be deterministic");
    }

    #[test]
    fn test_sos_orientation_degenerate_4d_returns_nonzero() {
        let points = degenerate_4d();
        let sign = sos_orientation_sign(&points).unwrap();
        assert!(sign == 1 || sign == -1, "SoS must return ±1, got {sign}");
    }

    #[test]
    fn test_sos_orientation_degenerate_4d_is_deterministic() {
        let points = degenerate_4d();
        let sign1 = sos_orientation_sign(&points).unwrap();
        let sign2 = sos_orientation_sign(&points).unwrap();
        assert_eq!(sign1, sign2, "SoS must be deterministic");
    }

    #[test]
    fn test_sos_orientation_nondegenerate_returns_correct_sign() {
        // Positive orientation triangle.  For this specific non-degenerate
        // configuration the leading SoS cofactor agrees with the true
        // orientation.  (SoS is only guaranteed correct for degenerate inputs;
        // the caller should never invoke SoS for non-degenerate cases.)
        let positive = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];
        let sign = sos_orientation_sign(&positive).unwrap();
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
        let sign = sos_orientation_sign(&points).unwrap();
        assert!(
            sign == 1 || sign == -1,
            "SoS must return ±1 for 5D, got {sign}"
        );
    }

    #[test]
    fn test_sos_orientation_translation_invariant_2d() {
        let points = collinear_2d();
        let sign1 = sos_orientation_sign(&points).unwrap();

        // Translate all points by a large offset.
        let translated: Vec<Point<f64, 2>> = points
            .iter()
            .map(|p| Point::new([p.coords()[0] + 1000.0, p.coords()[1] + 2000.0]))
            .collect();
        let sign2 = sos_orientation_sign(&translated).unwrap();
        assert_eq!(
            sign1, sign2,
            "SoS orientation must be translation-invariant"
        );
    }

    #[test]
    fn test_sos_orientation_translation_invariant_3d() {
        let points = coplanar_3d();
        let sign1 = sos_orientation_sign(&points).unwrap();

        let translated: Vec<Point<f64, 3>> = points
            .iter()
            .map(|p| {
                Point::new([
                    p.coords()[0] + 1e6,
                    p.coords()[1] - 5e5,
                    p.coords()[2] + 7.77,
                ])
            })
            .collect();
        let sign2 = sos_orientation_sign(&translated).unwrap();
        assert_eq!(
            sign1, sign2,
            "SoS orientation must be translation-invariant"
        );
    }

    // =========================================================================
    // SOS INSPHERE TESTS
    // =========================================================================

    #[test]
    fn test_sos_insphere_cospherical_3d_returns_nonzero() {
        let (simplex, test) = cospherical_3d();
        let sign = sos_insphere_sign(&simplex, &test).unwrap();
        assert!(
            sign == 1 || sign == -1,
            "SoS insphere must return ±1, got {sign}"
        );
    }

    #[test]
    fn test_sos_insphere_cospherical_3d_is_deterministic() {
        let (simplex, test) = cospherical_3d();
        let sign1 = sos_insphere_sign(&simplex, &test).unwrap();
        let sign2 = sos_insphere_sign(&simplex, &test).unwrap();
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
        let sign = sos_insphere_sign(&simplex, &test).unwrap();
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
        let sign = sos_insphere_sign(&simplex, &test).unwrap();
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
            .map(|_| sos_insphere_sign(&simplex, &test).unwrap())
            .collect();
        assert!(
            results.iter().all(|&r| r == results[0]),
            "SoS insphere must be deterministic across calls"
        );
    }

    // =========================================================================
    // ERROR HANDLING TESTS
    // =========================================================================

    #[test]
    fn test_sos_orientation_wrong_point_count_returns_error() {
        let points = vec![Point::new([0.0, 0.0]), Point::new([1.0, 0.0])];
        let result = sos_orientation_sign(&points);
        assert!(result.is_err(), "Should return Err for wrong point count");
    }

    #[test]
    fn test_sos_insphere_wrong_simplex_count_returns_error() {
        let simplex = vec![Point::new([0.0, 0.0]), Point::new([1.0, 0.0])];
        let test = Point::new([0.5, 0.5]);
        let result = sos_insphere_sign(&simplex, &test);
        assert!(result.is_err(), "Should return Err for wrong simplex count");
    }

    // =========================================================================
    // INTERNAL HELPER TESTS
    // =========================================================================

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
