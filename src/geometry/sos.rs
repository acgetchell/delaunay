//! Simulation of Simplicity (SoS) for deterministic degeneracy resolution.
//!
//! When geometric predicates (orientation, in-sphere) evaluate to exactly zero —
//! meaning points are exactly co-planar, co-spherical, etc. — the SoS technique
//! provides a deterministic non-zero answer without modifying any coordinates.
//!
//! # Algorithm
//!
//! Points are symbolically perturbed by infinitesimals εᵢ such that
//! ε₀ << ε₁ << … << εₙ (i.e. higher-indexed points receive larger
//! perturbations).  The perturbation is purely symbolic: no floating-point
//! values are changed.
//!
//! For an `n×n` matrix whose determinant is exactly zero, the SoS expansion
//! evaluates cofactors in reverse lexicographic `(row, column)` order.  The
//! first non-zero cofactor determines the sign of the perturbed determinant.
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

    // Fallback: return +1.
    //
    // The Edelsbrunner & Mücke SoS perturbation guarantees a non-zero
    // first-order cofactor for any set of *distinct* points.  All cofactors
    // being zero implies a higher-corank degeneracy (e.g. all points are
    // identical), which is geometrically meaningless.  Returning a
    // deterministic constant is safe and avoids complicating callers.
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
/// # Raw Determinant Sign
///
/// This function returns the sign of the **perturbed insphere determinant**,
/// *not* a normalized INSIDE/OUTSIDE classification.  The relationship
/// between determinant sign and geometric containment depends on the simplex
/// orientation.  Callers must multiply the result by an appropriate
/// orientation factor (as [`AdaptiveKernel::in_sphere`](crate::geometry::kernel::AdaptiveKernel) does) to obtain the
/// correct INSIDE/OUTSIDE semantics.
///
/// # Arguments
///
/// * `simplex` - Exactly `D+1` points defining the simplex (f64 coordinates).
/// * `test` - The test point to classify.
///
/// # Returns
///
/// `Ok(1)` or `Ok(-1)` (raw determinant sign).  Never returns `Ok(0)`.
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

    // Fallback: return +1.
    //
    // Same reasoning as `sos_orientation_sign`: all cofactors zero implies
    // a higher-corank degeneracy (all points identical) that has no
    // geometric meaning.  A deterministic constant is safe.
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

        exact_det_sign(&matrix)
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

        exact_det_sign(&matrix)
    })
}

/// Compute the exact sign of a matrix determinant
///
/// Uses the two-stage approach:
/// 1. `det_direct()` + `det_errbound()` for D ≤ 4 (provable fast filter)
/// 2. `det_sign_exact()` for exact result
///
/// Returns -1, 0, or +1.
pub(crate) fn exact_det_sign<const N: usize>(matrix: &Matrix<N>) -> i32 {
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
    // GENERIC HELPER FUNCTIONS
    // =========================================================================

    /// Build D+1 co-hyperplanar points (all with last coordinate = 0).
    ///
    /// Construction: origin + (D−1) axis-aligned unit vectors (last coord = 0)
    /// + a barycentric combination with weight 0.5 in each active axis.
    fn degenerate_orient_points<const D: usize>() -> Vec<Point<f64, D>> {
        let mut points = Vec::with_capacity(D + 1);
        points.push(Point::new([0.0; D]));
        for i in 0..D.saturating_sub(1) {
            let mut coords = [0.0; D];
            coords[i] = 1.0;
            points.push(Point::new(coords));
        }
        let mut bary = [0.0; D];
        for c in bary.iter_mut().take(D.saturating_sub(1)) {
            *c = 0.5;
        }
        points.push(Point::new(bary));
        points
    }

    /// Build D+1 simplex points and a co-spherical test point.
    ///
    /// The simplex is the origin plus D axis-aligned unit vectors.
    /// The test point (1,1,…,1) lies on the circumsphere (distance from
    /// center = circumradius for all D ≥ 2).
    fn cospherical_points<const D: usize>() -> (Vec<Point<f64, D>>, Point<f64, D>) {
        let mut simplex = Vec::with_capacity(D + 1);
        simplex.push(Point::new([0.0; D]));
        for i in 0..D {
            let mut coords = [0.0; D];
            coords[i] = 1.0;
            simplex.push(Point::new(coords));
        }
        (simplex, Point::new([1.0; D]))
    }

    /// Translate a point by a deterministic per-axis offset.
    fn translate_point<const D: usize>(p: &Point<f64, D>) -> Point<f64, D> {
        const OFFSETS: [f64; 5] = [1e6, -5e5, 7.77, -3.33e4, 42.0];
        let mut coords = [0.0; D];
        for (i, c) in coords.iter_mut().enumerate() {
            *c = p.coords()[i] + OFFSETS[i % OFFSETS.len()];
        }
        Point::new(coords)
    }

    // =========================================================================
    // MACRO-GENERATED PER-DIMENSION TESTS (2D–5D)
    // =========================================================================

    /// Generate the standard `SoS` tests for a given dimension:
    ///
    /// - orientation: degenerate nonzero, deterministic, translation-invariant
    /// - insphere: cospherical nonzero, deterministic (10 calls),
    ///   translation-invariant
    /// - fallback: orientation all-identical → +1, insphere all-identical → +1
    macro_rules! gen_sos_dim_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_sos_orientation_ $dim d_degenerate_nonzero>]() {
                    let points = degenerate_orient_points::<$dim>();
                    let sign = sos_orientation_sign(&points).unwrap();
                    assert!(sign == 1 || sign == -1, "SoS must return ±1, got {sign}");
                }

                #[test]
                fn [<test_sos_orientation_ $dim d_degenerate_deterministic>]() {
                    let points = degenerate_orient_points::<$dim>();
                    let s1 = sos_orientation_sign(&points).unwrap();
                    let s2 = sos_orientation_sign(&points).unwrap();
                    assert_eq!(s1, s2, "SoS must be deterministic");
                }

                #[test]
                fn [<test_sos_orientation_ $dim d_translation_invariant>]() {
                    let points = degenerate_orient_points::<$dim>();
                    let s1 = sos_orientation_sign(&points).unwrap();
                    let translated: Vec<_> = points.iter().map(translate_point).collect();
                    let s2 = sos_orientation_sign(&translated).unwrap();
                    assert_eq!(s1, s2, "SoS orientation must be translation-invariant");
                }

                #[test]
                fn [<test_sos_insphere_ $dim d_cospherical_nonzero>]() {
                    let (simplex, test) = cospherical_points::<$dim>();
                    let sign = sos_insphere_sign(&simplex, &test).unwrap();
                    assert!(
                        sign == 1 || sign == -1,
                        "SoS insphere must return ±1, got {sign}"
                    );
                }

                #[test]
                fn [<test_sos_insphere_ $dim d_cospherical_deterministic>]() {
                    let (simplex, test) = cospherical_points::<$dim>();
                    let results: Vec<i32> = (0..10)
                        .map(|_| sos_insphere_sign(&simplex, &test).unwrap())
                        .collect();
                    assert!(
                        results.iter().all(|&r| r == results[0]),
                        "SoS insphere must be deterministic across calls"
                    );
                }

                #[test]
                fn [<test_sos_insphere_ $dim d_translation_invariant>]() {
                    let (simplex, test) = cospherical_points::<$dim>();
                    let s1 = sos_insphere_sign(&simplex, &test).unwrap();
                    let translated_simplex: Vec<_> =
                        simplex.iter().map(translate_point).collect();
                    let translated_test = translate_point(&test);
                    let s2 =
                        sos_insphere_sign(&translated_simplex, &translated_test).unwrap();
                    assert_eq!(
                        s1, s2,
                        "SoS insphere must be translation-invariant"
                    );
                }

                #[test]
                fn [<test_sos_orientation_ $dim d_all_identical_fallback>]() {
                    let points = vec![Point::new([0.0; $dim]); $dim + 1];
                    assert_eq!(
                        sos_orientation_sign(&points).unwrap(),
                        1,
                        "All-identical fallback must return +1"
                    );
                }

                #[test]
                fn [<test_sos_insphere_ $dim d_all_identical_fallback>]() {
                    let simplex = vec![Point::new([1.0; $dim]); $dim + 1];
                    let test_pt = Point::new([1.0; $dim]);
                    assert_eq!(
                        sos_insphere_sign(&simplex, &test_pt).unwrap(),
                        1,
                        "All-identical insphere fallback must return +1"
                    );
                }
            }
        };
    }

    gen_sos_dim_tests!(2);
    gen_sos_dim_tests!(3);
    gen_sos_dim_tests!(4);
    gen_sos_dim_tests!(5);

    // =========================================================================
    // SOS ORIENTATION — NON-DEGENERATE SPOT CHECK
    // =========================================================================

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
        assert_eq!(exact_det_sign(&m), 1);
    }

    #[test]
    fn test_exact_det_sign_singular_2x2() {
        let m = Matrix::<2>::from_rows([[1.0, 2.0], [2.0, 4.0]]);
        assert_eq!(exact_det_sign(&m), 0);
    }

    #[test]
    fn test_exact_det_sign_negative_2x2() {
        let m = Matrix::<2>::from_rows([[0.0, 1.0], [1.0, 0.0]]);
        assert_eq!(exact_det_sign(&m), -1);
    }

    // =========================================================================
    // EXACT_DET_SIGN — 3×3, 4×4, 5×5
    // =========================================================================

    #[test]
    fn test_exact_det_sign_identity_3x3() {
        let m = Matrix::<3>::from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(exact_det_sign(&m), 1);
    }

    #[test]
    fn test_exact_det_sign_negative_3x3() {
        // Swapping two rows of the identity negates the determinant.
        let m = Matrix::<3>::from_rows([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(exact_det_sign(&m), -1);
    }

    #[test]
    fn test_exact_det_sign_identity_4x4() {
        let m = Matrix::<4>::from_rows([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        assert_eq!(exact_det_sign(&m), 1);
    }

    #[test]
    fn test_exact_det_sign_identity_5x5_bareiss_only() {
        // D ≥ 5: det_direct() and det_errbound() return None.
        // Only the Bareiss exact path runs.
        let m = Matrix::<5>::from_rows([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]);
        assert_eq!(exact_det_sign(&m), 1);
    }

    // =========================================================================
    // EXACT_DET_SIGN — NEAR-SINGULAR (INCONCLUSIVE FAST FILTER → BAREISS)
    // =========================================================================

    #[test]
    fn test_exact_det_sign_near_singular_uses_bareiss() {
        // Base matrix [[1,2,3],[4,5,6],[7,8,9]] is exactly singular.
        // Adding 2^-50 to entry (0,0) gives det = -3 × 2^-50 ≈ -2.66e-15.
        // This is much smaller than the error bound (~8e-13), so the fast
        // filter is inconclusive and Bareiss resolves the sign exactly.
        let perturbation = f64::from_bits(0x3CD0_0000_0000_0000); // 2^-50
        let m = Matrix::<3>::from_rows([
            [1.0 + perturbation, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        assert_eq!(exact_det_sign(&m), -1);
    }

    // =========================================================================
    // EXACT_DET_SIGN — OVERFLOW / NON-FINITE RECOVERY
    // =========================================================================

    #[test]
    fn test_exact_det_sign_overflow_det_recovered_by_bareiss() {
        // Entries are finite but det_direct overflows to infinity.
        // The is_finite() guard skips Stage 1; Bareiss computes exactly.
        let m = Matrix::<2>::from_rows([[1e200, 0.0], [0.0, 1e200]]);
        assert_eq!(exact_det_sign(&m), 1);
    }

    #[test]
    fn test_exact_det_sign_nan_entry_returns_zero() {
        // Non-finite entry → det_sign_exact returns Err → map_or gives 0.
        let m = Matrix::<2>::from_rows([[f64::NAN, 0.0], [0.0, 1.0]]);
        assert_eq!(exact_det_sign(&m), 0);
    }

    // =========================================================================
    // SOS ORIENTATION — 1D EDGE CASE
    // =========================================================================

    #[test]
    fn test_sos_orientation_1d_identical_points() {
        // Two identical 1D points: orientation determinant is exactly zero.
        // SoS must still resolve to ±1.
        let points = vec![Point::new([5.0]), Point::new([5.0])];
        let sign = sos_orientation_sign(&points).unwrap();
        assert!(
            sign == 1 || sign == -1,
            "SoS must return ±1 for 1D, got {sign}"
        );
    }

    #[test]
    fn test_sos_orientation_1d_distinct_degenerate() {
        // D=1 with distinct points is non-degenerate, but SoS still works.
        let points = vec![Point::new([0.0]), Point::new([1.0])];
        let sign = sos_orientation_sign(&points).unwrap();
        assert!(
            sign == 1 || sign == -1,
            "SoS must return ±1 for 1D, got {sign}"
        );
    }
}
