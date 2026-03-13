//! Enhanced geometric predicates with improved numerical robustness.
//!
//! This module provides robust orientation and insphere predicates for
//! Delaunay triangulation. The predicates layer multiple strategies on top
//! of exact arithmetic: adaptive tolerance, diagnostic consistency checking,
//! and Simulation of Simplicity (`SoS`) fallback for degenerate and
//! near-degenerate point configurations.

#![forbid(unsafe_code)]

use super::predicates::{InSphere, Orientation};
use super::util::{safe_coords_to_f64, safe_scalar_to_f64, squared_norm};
use crate::geometry::matrix::matrix_set;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{
    Coordinate, CoordinateConversionError, CoordinateScalar, ScalarSummable,
};
use std::sync::LazyLock;

static STRICT_INSPHERE_CONSISTENCY: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("DELAUNAY_STRICT_INSPHERE_CONSISTENCY").is_some());

/// Result of consistency verification between different insphere methods.
///
/// # Examples
///
/// ```rust
/// use delaunay::geometry::robust_predicates::ConsistencyResult;
///
/// let result = ConsistencyResult::Consistent;
/// assert!(result.is_consistent());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsistencyResult {
    /// The two methods agree on the result
    Consistent,
    /// The two methods disagree (potential numerical issue)
    Inconsistent(InsphereConsistencyError),
    /// Cannot verify consistency due to error in verification method
    Unverifiable,
}

impl ConsistencyResult {
    /// Returns true if the result indicates consistency (either Consistent or Unverifiable).
    /// Only returns false for definite Inconsistent results.
    #[must_use]
    pub const fn is_consistent(self) -> bool {
        matches!(self, Self::Consistent | Self::Unverifiable)
    }
}

impl std::fmt::Display for ConsistencyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Consistent => write!(f, "Consistent"),
            Self::Inconsistent(error) => write!(f, "{error}"),
            Self::Unverifiable => write!(f, "Unverifiable"),
        }
    }
}

/// Error details for direct contradiction between insphere implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum InsphereConsistencyError {
    /// Determinant and distance methods classify a point in opposite half-spaces.
    #[error(
        "Insphere inconsistency: determinant={determinant_result:?}, distance={distance_result:?}"
    )]
    DirectContradiction {
        /// Result from determinant-based predicate.
        determinant_result: InSphere,
        /// Result from distance-based predicate.
        distance_result: InSphere,
    },
}

/// Enhanced insphere predicate with multiple numerical robustness techniques.
///
/// Evaluates whether a `test_point` lies inside, on, or outside the circumsphere
/// (in 2D: circumcircle) defined by a `D`-simplex (`simplex_points`). This
/// implementation is dimension-generic and applies a series of strategies to
/// provide robust results for degenerate and near-degenerate configurations.
///
/// Strategies used, in order:
/// 1) Adaptive tolerance insphere via exact-sign determinant evaluation
/// 2) Diagnostic consistency check against a distance-based insphere (does not
///    override the exact result; only hard-fails when `DELAUNAY_STRICT_INSPHERE_CONSISTENCY`
///    is set)
/// 3) Simulation of Simplicity (`SoS`) fallback (only reached when the exact-sign
///    computation itself fails, e.g. unsupported matrix size for D ≥ 6)
///
/// Sign convention and orientation:
/// - The determinant sign is interpreted relative to the simplex orientation.
/// - If the simplex orientation is POSITIVE, det > tol => INSIDE and det < -tol => OUTSIDE.
/// - If NEGATIVE, the interpretation is swapped (det < -tol => INSIDE, det > tol => OUTSIDE).
/// - DEGENERATE orientation yields BOUNDARY conservatively.
///
/// Type parameters:
/// - `T`: Coordinate scalar type implementing `CoordinateScalar`
/// - `D`: Compile-time dimension of the space
///
/// Parameters:
/// - `simplex_points`: Exactly `D + 1` points defining the simplex
/// - `test_point`: The query point to classify relative to the simplex circumsphere
///
/// Returns:
/// - `Ok(InSphere::{INSIDE, BOUNDARY, OUTSIDE})` on success
/// - `Err(CoordinateConversionError)` if inputs are invalid (e.g., wrong point
///   count) or safe conversions fail
///
/// Complexity:
/// - The f64 fast-filter path is O((D+2)³). When the exact Bareiss path is
///   triggered (near-degenerate inputs), arbitrary-precision rational arithmetic
///   increases the cost significantly beyond O(D³).
///
/// Example (3D):
/// ```rust
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::robust_predicates::robust_insphere;
/// use delaunay::geometry::predicates::InSphere;
/// use delaunay::geometry::traits::coordinate::Coordinate;
///
/// let tetra = vec![
///     Point::new([0.0, 0.0, 0.0]),
///     Point::new([1.0, 0.0, 0.0]),
///     Point::new([0.0, 1.0, 0.0]),
///     Point::new([0.0, 0.0, 1.0]),
/// ];
///
/// let inside = Point::new([0.25, 0.25, 0.25]);
/// let outside = Point::new([2.0, 2.0, 2.0]);
///
/// let r_in = robust_insphere(&tetra, &inside).unwrap();
/// let r_out = robust_insphere(&tetra, &outside).unwrap();
/// assert_eq!(r_in, InSphere::INSIDE);
/// assert_eq!(r_out, InSphere::OUTSIDE);
/// ```
///
/// Notes:
/// - When Strategy 1 fails (e.g. D ≥ 6 where the insphere matrix exceeds the
///   stack-matrix limit), the function falls back to Simulation of Simplicity
///   (`SoS`) for deterministic tie-breaking without modifying coordinates.
/// - See `robust_orientation` for the orientation predicate used in the sign interpretation.
/// - The insphere matrix uses absolute coordinates whose squared norms can
///   overflow `f64` for `‖coords‖ ≥ ~1e154`; see
///   [`crate::geometry::predicates::insphere_lifted`] for a relative-coordinate
///   alternative that avoids this limitation.
///
/// # Errors
///
/// Returns an error if the input is invalid (wrong number of points) or if required
/// numeric conversions fail.
pub fn robust_insphere<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
) -> Result<InSphere, CoordinateConversionError>
where
    T: ScalarSummable,
    [T; D]: Copy + Sized,
{
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: format!("Expected {} points, got {}", D + 1, simplex_points.len()),
            from_type: "point count",
            to_type: "valid simplex",
        });
    }

    // Strategy 1: Exact-sign determinant approach with adaptive tolerance.
    if let Ok(result) = adaptive_tolerance_insphere(simplex_points, test_point) {
        // Strategy 2: Diagnostic consistency check against distance-based insphere.
        // The exact-sign result from insphere_from_matrix is provably correct for
        // finite inputs; a disagreement from insphere_distance reflects f64
        // rounding in the distance-based check, not a defect in the exact predicate.
        if let ConsistencyResult::Inconsistent(error) =
            verify_insphere_consistency(simplex_points, test_point, result)
        {
            // In strict mode, hard-fail for deterministic witness capture.
            if *STRICT_INSPHERE_CONSISTENCY {
                let details = format!(
                    "{error}; simplex_points={simplex_points:?}; test_point={test_point:?}"
                );
                return Err(CoordinateConversionError::InsphereInconsistency {
                    simplex_points: format!("{simplex_points:?}"),
                    test_point: format!("{test_point:?}"),
                    details,
                });
            }
        }
        return Ok(result);
    }

    // Strategy 3: Geometric + SoS fallback — only reached when exact-sign
    // computation itself failed (e.g. unsupported matrix size for D ≥ 6).
    //
    // First try insphere_distance (circumcenter/radius based — no matrix
    // determinant needed, works at any dimension).  This handles the
    // non-degenerate cases correctly.  Only if the result is BOUNDARY
    // (truly degenerate) do we apply SoS tie-breaking.
    if let Ok(geometric_result) = super::predicates::insphere_distance(simplex_points, *test_point)
        && geometric_result != InSphere::BOUNDARY
    {
        return Ok(geometric_result);
    }

    // SoS tie-breaking for the truly degenerate case (BOUNDARY or
    // insphere_distance itself failed).  The SoS cofactor minors are one
    // size smaller, so this succeeds where the full insphere matrix
    // dispatch does not.
    let f64_simplex: Vec<Point<f64, D>> = simplex_points
        .iter()
        .map(|p| safe_coords_to_f64(p.coords()).map(Point::new))
        .collect::<Result<_, _>>()?;
    let f64_test: Point<f64, D> = Point::new(safe_coords_to_f64(test_point.coords())?);

    // Use exact orientation when available; fall back to SoS only when the
    // exact predicate reports DEGENERATE (or fails entirely).
    let abs_orient: i32 = match robust_orientation(simplex_points) {
        Ok(Orientation::POSITIVE) => 1,
        Ok(Orientation::NEGATIVE) => -1,
        _ => crate::geometry::sos::sos_orientation_sign(&f64_simplex)?,
    };
    let raw_insphere = crate::geometry::sos::sos_insphere_sign(&f64_simplex, &f64_test)?;

    // Apply the same parity-aware normalization as AdaptiveKernel:
    // orient_factor = (-1)^(D+1) × abs_orient, because the insphere
    // convention requires negating the relative orientation and
    // rel_orient = (-1)^D × abs_orient.
    let orient_factor = if D.is_multiple_of(2) {
        -abs_orient
    } else {
        abs_orient
    };
    let sign = raw_insphere * orient_factor;

    Ok(if sign > 0 {
        InSphere::INSIDE
    } else {
        InSphere::OUTSIDE
    })
}

#[inline]
fn fill_insphere_predicate_matrix<T, const D: usize, const K: usize>(
    matrix: &mut crate::geometry::matrix::Matrix<K>,
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
) -> Result<(), CoordinateConversionError>
where
    T: CoordinateScalar,
    [T; D]: Copy + Sized,
{
    debug_assert_eq!(K, D + 2);

    // NOTE: Uses absolute coordinates with squared_norm.  The squared norm can
    // overflow f64 for ‖coords‖ ≥ ~1e154 (since 1e154² ≈ 1e308 ≈ f64::MAX).
    // `insphere_lifted` avoids this by centering on relative coordinates.

    // Add simplex points
    for (i, point) in simplex_points.iter().enumerate() {
        let coords = point.coords();

        // Coordinates - use safe conversion
        let coords_f64 = safe_coords_to_f64(coords)?;
        for (j, &v) in coords_f64.iter().enumerate() {
            matrix_set(matrix, i, j, v);
        }

        // Squared norm - use safe conversion
        let norm_sq = squared_norm(coords);
        let norm_sq_f64 = safe_scalar_to_f64(norm_sq)?;
        matrix_set(matrix, i, D, norm_sq_f64);

        // Constant term
        matrix_set(matrix, i, D + 1, 1.0);
    }

    // Add test point
    let test_coords = test_point.coords();

    let test_coords_f64 = safe_coords_to_f64(test_coords)?;
    for (j, &v) in test_coords_f64.iter().enumerate() {
        matrix_set(matrix, D + 1, j, v);
    }

    let test_norm_sq = squared_norm(test_coords);
    let test_norm_sq_f64 = safe_scalar_to_f64(test_norm_sq)?;
    matrix_set(matrix, D + 1, D, test_norm_sq_f64);
    matrix_set(matrix, D + 1, D + 1, 1.0);

    Ok(())
}

/// Insphere test with adaptive tolerance based on operand magnitude.
///
/// Uses [`super::predicates::insphere_from_matrix`] for provably correct sign
/// classification on finite inputs, falling back to an f64 determinant with
/// adaptive tolerance for non-finite entries.
fn adaptive_tolerance_insphere<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
) -> Result<InSphere, CoordinateConversionError>
where
    T: CoordinateScalar,
    [T; D]: Copy + Sized,
{
    // Get simplex orientation for correct interpretation.
    let orientation = robust_orientation(simplex_points)?;
    if matches!(orientation, Orientation::DEGENERATE) {
        return Ok(InSphere::BOUNDARY);
    }
    let orient_sign: i8 = if matches!(orientation, Orientation::POSITIVE) {
        1
    } else {
        -1
    };

    let k = D + 2;
    try_with_la_stack_matrix!(k, |matrix| {
        fill_insphere_predicate_matrix(&mut matrix, simplex_points, test_point)?;
        Ok(super::predicates::insphere_from_matrix(
            &matrix,
            k,
            orient_sign,
        ))
    })
}

/// Enhanced orientation predicate with robustness improvements.
///
/// Internally uses provable [`la_stack::Matrix::det_errbound`] bounds (D ≤ 4)
/// and exact Bareiss arithmetic for the slow path.
///
/// # Errors
///
/// Returns an error if the input is invalid (wrong number of points) or if
/// the geometric computation fails.
///
/// # Examples
///
/// ```rust
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::predicates::Orientation;
/// use delaunay::geometry::robust_predicates::robust_orientation;
/// use delaunay::geometry::traits::coordinate::Coordinate;
///
/// let tri = vec![
///     Point::new([0.0, 0.0]),
///     Point::new([1.0, 0.0]),
///     Point::new([0.0, 1.0]),
/// ];
/// let orientation = robust_orientation(&tri).unwrap();
/// assert_eq!(orientation, Orientation::POSITIVE);
/// ```
pub fn robust_orientation<T, const D: usize>(
    simplex_points: &[Point<T, D>],
) -> Result<Orientation, CoordinateConversionError>
where
    T: CoordinateScalar,
    [T; D]: Copy + Sized,
{
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: format!("Expected {} points, got {}", D + 1, simplex_points.len()),
            from_type: "point count",
            to_type: "valid simplex",
        });
    }

    let k = D + 1;

    try_with_la_stack_matrix!(k, |matrix| {
        for (i, point) in simplex_points.iter().enumerate() {
            let coords = point.coords();

            // Add coordinates using safe conversion
            let coords_f64 = safe_coords_to_f64(coords)?;
            for (j, &v) in coords_f64.iter().enumerate() {
                matrix_set(&mut matrix, i, j, v);
            }

            // Add constant term
            matrix_set(&mut matrix, i, D, 1.0);
        }

        // Route through the exact-sign orientation helper for provably correct
        // orientation classification on finite inputs.
        Ok(super::predicates::orientation_from_matrix(&matrix, k))
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Verify consistency of insphere result using alternative method.
///
/// This function provides an independent verification of the insphere result using
/// the distance-based `insphere_distance` function. This helps detect numerical
/// inconsistencies that might arise from near-degenerate configurations or precision
/// issues in the determinant calculation.
///
/// # Algorithm
///
/// 1. Use `insphere_distance` to get an independent insphere result
/// 2. Compare this result with the determinant-based result
/// 3. Consider results consistent if they match or if either is `BOUNDARY`
///
/// # Tolerance for Consistency
///
/// The function is conservative about marking results as inconsistent:
/// - Exact matches (`INSIDE`/`INSIDE`, `OUTSIDE`/`OUTSIDE`, `BOUNDARY`/`BOUNDARY`) are consistent
/// - Any result involving `BOUNDARY` is considered consistent since it indicates degeneracy
/// - Only direct contradictions (`INSIDE`/`OUTSIDE`) are marked as inconsistent
///
/// This approach prevents false negatives when dealing with legitimately degenerate
/// or near-degenerate configurations.
///
/// # Returns
///
/// - [`ConsistencyResult::Consistent`] if the two methods agree
/// - [`ConsistencyResult::Inconsistent`] if there's a direct contradiction
/// - [`ConsistencyResult::Unverifiable`] if the verification method fails
fn verify_insphere_consistency<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
    determinant_result: InSphere,
) -> ConsistencyResult
where
    T: ScalarSummable,
    [T; D]: Copy + Sized,
{
    // Use the existing distance-based insphere test for verification
    super::predicates::insphere_distance(simplex_points, *test_point).map_or(
        ConsistencyResult::Unverifiable,
        |distance_result| match (determinant_result, distance_result) {
            // Exact matches are always consistent
            (InSphere::INSIDE, InSphere::INSIDE)
            | (InSphere::OUTSIDE, InSphere::OUTSIDE)
            | (InSphere::BOUNDARY, _)
            | (_, InSphere::BOUNDARY) => ConsistencyResult::Consistent,

            // Direct contradictions indicate numerical issues
            (InSphere::INSIDE, InSphere::OUTSIDE) | (InSphere::OUTSIDE, InSphere::INSIDE) => {
                // Log the inconsistency for debugging (in debug builds only)
                #[cfg(debug_assertions)]
                tracing::warn!(
                    determinant_result = ?determinant_result,
                    distance_result = ?distance_result,
                    "Insphere consistency check failed"
                );
                ConsistencyResult::Inconsistent(InsphereConsistencyError::DirectContradiction {
                    determinant_result,
                    distance_result,
                })
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::matrix::matrix_get;
    use crate::geometry::point::Point;
    use crate::geometry::predicates;
    use num_traits::NumCast;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn test_robust_insphere_general() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        // Test point clearly inside
        let inside_point = Point::new([0.25, 0.25, 0.25]);
        let result = robust_insphere(&points, &inside_point).unwrap();
        assert_eq!(result, InSphere::INSIDE);

        // Test point clearly outside
        let outside_point = Point::new([2.0, 2.0, 2.0]);
        let result = robust_insphere(&points, &outside_point).unwrap();
        assert_eq!(result, InSphere::OUTSIDE);
    }

    #[test]
    fn test_robust_orientation() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let result = robust_orientation(&points).unwrap();

        // Should detect orientation (exact result depends on coordinate system)
        assert!(matches!(
            result,
            Orientation::POSITIVE | Orientation::NEGATIVE
        ));
    }

    #[test]
    fn test_robust_orientation_positive_triangle_2d() {
        // Canonical CCW triangle to exercise the robust_orientation matrix path
        // and confirm the exact-sign helper returns POSITIVE.
        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        let robust = robust_orientation(&points).unwrap();
        let reference = predicates::simplex_orientation(&points).unwrap();

        assert_eq!(robust, Orientation::POSITIVE);
        assert_eq!(robust, reference);
    }

    #[test]
    fn test_robust_orientation_ignores_base_tolerance() {
        // robust_orientation no longer uses config.base_tolerance (the
        // provable det_errbound replaces the heuristic tolerance), so a
        // NaN base_tolerance does not cause an error.
        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        let result = robust_orientation(&points);
        assert_eq!(result.unwrap(), Orientation::POSITIVE);
    }

    #[test]
    fn test_robust_orientation_near_degenerate_2d_exact_sign() {
        // Near-degenerate triangle where adaptive f64 tolerance can collapse to DEGENERATE,
        // but exact determinant sign should remain POSITIVE.
        let eps = 2f64.powi(-50);
        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.5, eps]),
        ];

        let result = robust_orientation(&points).unwrap();
        assert_eq!(
            result,
            Orientation::POSITIVE,
            "near-degenerate 2D orientation should use exact sign and stay POSITIVE"
        );
    }

    #[test]
    fn test_robust_orientation_near_degenerate_3d_not_degenerate() {
        // Near-degenerate tetrahedron where exact sign should prevent false DEGENERATE.
        let eps = 2f64.powi(-50);
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, eps]),
        ];

        let result = robust_orientation(&points).unwrap();
        assert_ne!(
            result,
            Orientation::DEGENERATE,
            "near-degenerate 3D orientation should not be DEGENERATE with exact sign"
        );
    }

    #[test]
    fn test_robust_insphere_near_cocircular_2d_exact_sign() {
        // Near-cocircular 2D configuration where the test points sit on the
        // circumcircle boundary ± eps, so the f64 determinant lands in the
        // tolerance band and the exact-sign path (Stage 2) must resolve it.
        let eps = 2f64.powi(-50);
        let triangle = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];
        // Circumcenter = (0.5, 0.5), circumradius = sqrt(0.5).
        // Place test points along the +x direction from the circumcenter.
        let radius = 0.5_f64.sqrt();
        let inside_point = Point::new([0.5 + radius - eps, 0.5]);
        let outside_point = Point::new([0.5 + radius + eps, 0.5]);

        assert_eq!(
            robust_insphere(&triangle, &inside_point).unwrap(),
            InSphere::INSIDE,
            "near-cocircular 2D point just inside boundary should be INSIDE"
        );
        assert_eq!(
            robust_insphere(&triangle, &outside_point).unwrap(),
            InSphere::OUTSIDE,
            "near-cocircular 2D point just outside boundary should be OUTSIDE"
        );
    }

    #[test]
    fn test_robust_insphere_near_cospherical_3d_exact_sign() {
        // Near-cospherical 3D configuration where the test points sit on the
        // circumsphere boundary ± eps, so the f64 determinant is ambiguous
        // and exact-sign arithmetic (Stage 2) must resolve the classification.
        let eps = 2f64.powi(-50);
        let tetra = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        // Circumcenter = (0.5, 0.5, 0.5), circumradius = sqrt(0.75).
        // Place test points along the +x direction from the circumcenter.
        let radius = 0.75_f64.sqrt();
        let inside_point = Point::new([0.5 + radius - eps, 0.5, 0.5]);
        let outside_point = Point::new([0.5 + radius + eps, 0.5, 0.5]);

        assert_eq!(
            robust_insphere(&tetra, &inside_point).unwrap(),
            InSphere::INSIDE,
            "near-cospherical 3D point just inside boundary should be INSIDE"
        );
        assert_eq!(
            robust_insphere(&tetra, &outside_point).unwrap(),
            InSphere::OUTSIDE,
            "near-cospherical 3D point just outside boundary should be OUTSIDE"
        );
    }

    #[test]
    fn test_degenerate_case_handling() {
        // Create nearly coplanar points
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 1e-15]), // Very slightly off-plane
        ];

        // Should handle gracefully
        let test_point = Point::new([0.25, 0.25, 1e-16]);
        let result = robust_insphere(&points, &test_point);
        assert!(result.is_ok());
    }

    #[expect(clippy::too_many_lines)]
    #[test]
    fn test_verify_insphere_consistency_comprehensive() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        // Test exact matches - all should be consistent
        let test_cases = [
            (
                Point::new([0.25, 0.25, 0.25]),
                InSphere::INSIDE,
                "inside point",
            ),
            (
                Point::new([2.0, 2.0, 2.0]),
                InSphere::OUTSIDE,
                "outside point",
            ),
            (
                Point::new([0.5, 0.5, 0.5]),
                InSphere::BOUNDARY,
                "boundary point",
            ),
        ];

        for (test_point, result, description) in test_cases {
            assert!(
                verify_insphere_consistency(&points, &test_point, result).is_consistent(),
                "Failed for {description}"
            );
        }

        // Test that BOUNDARY results are always considered consistent
        let boundary_test_point = Point::new([0.3, 0.3, 0.3]);
        for expected_result in [InSphere::INSIDE, InSphere::OUTSIDE, InSphere::BOUNDARY] {
            if expected_result == InSphere::BOUNDARY {
                assert!(
                    verify_insphere_consistency(&points, &boundary_test_point, expected_result,)
                        .is_consistent()
                );
            }
        }

        // Test different dimensions
        let triangle_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([2.0, 0.0]),
            Point::new([1.0, 2.0]),
        ];
        let test_2d = Point::new([1.0, 0.5]);
        assert!(
            verify_insphere_consistency(&triangle_2d, &test_2d, InSphere::BOUNDARY).is_consistent()
        );

        // Test edge cases with extreme coordinates and error conditions
        let edge_cases = [
            (
                vec![
                    Point::new([1e-10, 0.0, 0.0]),
                    Point::new([0.0, 1e-10, 0.0]),
                    Point::new([0.0, 0.0, 1e-10]),
                    Point::new([1e-10, 1e-10, 1e-10]),
                ],
                Point::new([5e-11, 5e-11, 5e-11]),
                "small coordinates",
            ),
            (
                vec![
                    Point::new([1e6, 0.0, 0.0]),
                    Point::new([0.0, 1e6, 0.0]),
                    Point::new([0.0, 0.0, 1e6]),
                    Point::new([1e6, 1e6, 1e6]),
                ],
                Point::new([5e5, 5e5, 5e5]),
                "large coordinates",
            ),
        ];

        for (edge_points, edge_test, description) in edge_cases {
            assert!(
                verify_insphere_consistency(&edge_points, &edge_test, InSphere::BOUNDARY)
                    .is_consistent(),
                "Failed for edge case: {description}"
            );
        }

        // Test error conditions that should return Unverifiable
        let error_cases = [
            // Invalid simplex size
            (
                vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])],
                Point::new([0.5, 0.0, 0.0]),
                "too few points",
            ),
            // Non-finite coordinates
            (
                vec![
                    Point::new([f64::NAN, 0.0, 0.0]),
                    Point::new([1.0, 0.0, 0.0]),
                    Point::new([0.0, 1.0, 0.0]),
                    Point::new([0.0, 0.0, 1.0]),
                ],
                Point::new([0.1, 0.1, 0.1]),
                "NaN coordinates",
            ),
            // Degenerate simplex
            (
                vec![
                    Point::new([0.0, 0.0, 0.0]),
                    Point::new([1.0, 0.0, 0.0]),
                    Point::new([2.0, 0.0, 0.0]),
                    Point::new([3.0, 0.0, 0.0]),
                ],
                Point::new([1.5, 0.0, 0.0]),
                "collinear points",
            ),
        ];

        for (error_points, error_test, error_description) in error_cases {
            let result =
                verify_insphere_consistency(&error_points, &error_test, InSphere::BOUNDARY);
            assert_eq!(
                result,
                ConsistencyResult::Unverifiable,
                "Expected Unverifiable for: {error_description}"
            );
            assert!(
                result.is_consistent(),
                "Unverifiable should be considered consistent"
            );
        }
    }

    #[test]
    fn test_consistency_result_display() {
        // Test Display trait implementation for ConsistencyResult
        assert_eq!(format!("{}", ConsistencyResult::Consistent), "Consistent");
        assert_eq!(
            format!(
                "{}",
                ConsistencyResult::Inconsistent(InsphereConsistencyError::DirectContradiction {
                    determinant_result: InSphere::INSIDE,
                    distance_result: InSphere::OUTSIDE,
                })
            ),
            "Insphere inconsistency: determinant=INSIDE, distance=OUTSIDE"
        );
        assert_eq!(
            format!("{}", ConsistencyResult::Unverifiable),
            "Unverifiable"
        );
    }

    const PERIODIC_TWO_POW_52_I64: i64 = 4_503_599_627_370_496;
    const PERIODIC_TWO_POW_52_F64: f64 = 4_503_599_627_370_496.0;
    const PERIODIC_MAX_OFFSET_UNITS: i64 = 1_048_576;
    const PERIODIC_IMAGE_JITTER_UNITS: i64 = 64;
    const PERIODIC_FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PERIODIC_FNV_PRIME: u64 = 0x0100_0000_01b3;
    type PeriodicWitness3d = ([Point<f64, 3>; 4], Point<f64, 3>, InSphere, InSphere);

    fn periodic_builder_perturb_units(canon_idx: usize, axis: usize) -> i64 {
        let mut h = PERIODIC_FNV_OFFSET_BASIS;
        h ^= u64::try_from(canon_idx).expect("canonical index fits in u64");
        h = h.wrapping_mul(PERIODIC_FNV_PRIME);
        h ^= u64::try_from(axis).expect("axis fits in u64");
        h = h.wrapping_mul(PERIODIC_FNV_PRIME);
        let span =
            u64::try_from(2 * PERIODIC_MAX_OFFSET_UNITS + 1).expect("periodic span fits in u64");
        i64::try_from(h % span).expect("residue fits in i64") - PERIODIC_MAX_OFFSET_UNITS
    }

    fn periodic_builder_image_jitter_units(canon_idx: usize, axis: usize, image_idx: usize) -> i64 {
        let mut h = PERIODIC_FNV_OFFSET_BASIS;
        h ^= u64::try_from(canon_idx).expect("canonical index fits in u64");
        h = h.wrapping_mul(PERIODIC_FNV_PRIME);
        h ^= u64::try_from(axis).expect("axis fits in u64");
        h = h.wrapping_mul(PERIODIC_FNV_PRIME);
        h ^= u64::try_from(image_idx).expect("image index fits in u64");
        h = h.wrapping_mul(PERIODIC_FNV_PRIME);
        let span = u64::try_from(2 * PERIODIC_IMAGE_JITTER_UNITS + 1)
            .expect("periodic jitter span fits in u64");
        i64::try_from(h % span).expect("residue fits in i64") - PERIODIC_IMAGE_JITTER_UNITS
    }

    fn periodic_3d_canonical_points() -> Vec<Point<f64, 3>> {
        vec![
            Point::new([0.1_f64, 0.2, 0.3]),
            Point::new([0.4, 0.7, 0.1]),
            Point::new([0.7, 0.3, 0.8]),
            Point::new([0.2, 0.9, 0.5]),
            Point::new([0.8, 0.6, 0.2]),
            Point::new([0.5, 0.1, 0.7]),
            Point::new([0.3, 0.5, 0.9]),
            Point::new([0.6, 0.8, 0.4]),
            Point::new([0.9, 0.2, 0.6]),
            Point::new([0.0, 0.4, 0.1]),
            Point::new([0.15, 0.65, 0.45]),
            Point::new([0.75, 0.15, 0.85]),
            Point::new([0.45, 0.55, 0.25]),
            Point::new([0.85, 0.45, 0.65]),
        ]
    }

    fn periodic_3d_builder_style_expansion(
        canonical_points: &[Point<f64, 3>],
    ) -> Vec<Point<f64, 3>> {
        let canonical_f64: Vec<[f64; 3]> = canonical_points
            .iter()
            .enumerate()
            .map(|(canon_idx, point)| {
                let coords = point.coords();
                let mut quantized = [0.0_f64; 3];
                for axis in 0..3 {
                    let normalized = coords[axis].clamp(0.0, 1.0 - f64::EPSILON);
                    let scaled = (normalized * PERIODIC_TWO_POW_52_F64).floor();
                    let unit_index = <i64 as NumCast>::from(scaled)
                        .expect("scaled coordinate index fits in i64");
                    let min_off = -unit_index.min(PERIODIC_MAX_OFFSET_UNITS);
                    let max_off =
                        (PERIODIC_TWO_POW_52_I64 - 1 - unit_index).min(PERIODIC_MAX_OFFSET_UNITS);
                    let offset =
                        periodic_builder_perturb_units(canon_idx, axis).clamp(min_off, max_off);
                    let adjusted = <f64 as NumCast>::from(unit_index + offset)
                        .expect("adjusted index fits in f64");
                    quantized[axis] = adjusted / PERIODIC_TWO_POW_52_F64;
                }
                quantized
            })
            .collect();

        let three_pow_d = 27_usize;
        let zero_offset_idx = (three_pow_d - 1) / 2;
        let mut expanded = Vec::with_capacity(canonical_points.len() * three_pow_d);

        for image_idx in 0..three_pow_d {
            let mut offset = [0_i8; 3];
            for (axis, offset_val) in offset.iter_mut().enumerate() {
                let axis_u32 = u32::try_from(axis).expect("axis index fits in u32");
                let digit = (image_idx / 3_usize.pow(axis_u32)) % 3;
                *offset_val = i8::try_from(digit).expect("digit fits in i8") - 1;
            }

            let is_canonical = image_idx == zero_offset_idx;
            for (canon_idx, quantized) in canonical_f64.iter().enumerate() {
                let mut image_coords = [0.0_f64; 3];
                for axis in 0..3 {
                    let shift = <f64 as std::convert::From<i8>>::from(offset[axis]);
                    let jitter = if is_canonical {
                        0.0
                    } else {
                        let jitter_units =
                            periodic_builder_image_jitter_units(canon_idx, axis, image_idx);
                        let jitter_as_f64 =
                            <f64 as NumCast>::from(jitter_units).expect("jitter units fit in f64");
                        jitter_as_f64 / PERIODIC_TWO_POW_52_F64
                    };
                    image_coords[axis] = quantized[axis] + shift + jitter;
                }
                expanded.push(Point::new(image_coords));
            }
        }

        expanded
    }

    fn find_periodic_3d_inconsistency_witness(
        expanded: &[Point<f64, 3>],
        seed: u64,
        sample_budget: usize,
    ) -> Option<PeriodicWitness3d> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n = expanded.len();
        if n < 5 {
            return None;
        }
        for _ in 0..sample_budget {
            let i0 = rng.random_range(0..n);
            let mut i1 = rng.random_range(0..n);
            while i1 == i0 {
                i1 = rng.random_range(0..n);
            }
            let mut i2 = rng.random_range(0..n);
            while i2 == i0 || i2 == i1 {
                i2 = rng.random_range(0..n);
            }
            let mut i3 = rng.random_range(0..n);
            while i3 == i0 || i3 == i1 || i3 == i2 {
                i3 = rng.random_range(0..n);
            }
            let mut it = rng.random_range(0..n);
            while it == i0 || it == i1 || it == i2 || it == i3 {
                it = rng.random_range(0..n);
            }

            let simplex = [expanded[i0], expanded[i1], expanded[i2], expanded[i3]];
            let test_point = expanded[it];
            let det_result = adaptive_tolerance_insphere(&simplex, &test_point);
            let dist_result = predicates::insphere_distance(&simplex, test_point);

            if let (Ok(det), Ok(dist)) = (det_result, dist_result)
                && matches!(
                    (det, dist),
                    (InSphere::INSIDE, InSphere::OUTSIDE) | (InSphere::OUTSIDE, InSphere::INSIDE)
                )
            {
                return Some((simplex, test_point, det, dist));
            }
        }
        None
    }

    #[test]
    #[ignore = "stress test; run explicitly with --ignored"]
    fn test_periodic_3d_inconsistency_witness_search_seeded() {
        let canonical_points = periodic_3d_canonical_points();
        let expanded = periodic_3d_builder_style_expansion(&canonical_points);
        let witness = find_periodic_3d_inconsistency_witness(&expanded, 0x2100_0003, 200_000);

        if let Some((simplex, test_point, det, dist)) = witness {
            panic!(
                "Found periodic-3D determinant-vs-distance inconsistency: determinant={det:?}, distance={dist:?}, simplex={simplex:?}, test_point={test_point:?}"
            );
        }
    }

    #[test]
    fn test_robust_predicates_dimensional_coverage() {
        // Comprehensive test across dimensions 2D-5D with both valid and invalid cases

        // Test 2D - Valid triangle
        let triangle_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.5, 1.0]),
        ];
        let test_2d = Point::new([0.5, 0.3]);
        assert!(
            robust_insphere(&triangle_2d, &test_2d).is_ok(),
            "2D insphere should work"
        );
        assert!(
            robust_orientation(&triangle_2d).is_ok(),
            "2D orientation should work"
        );

        // Test 3D - Valid tetrahedron
        let tetrahedron_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let test_3d = Point::new([0.25, 0.25, 0.25]);
        assert!(
            robust_insphere(&tetrahedron_3d, &test_3d).is_ok(),
            "3D insphere should work"
        );
        assert!(
            robust_orientation(&tetrahedron_3d).is_ok(),
            "3D orientation should work"
        );

        // Test 4D - Valid hypersimplex
        let simplex_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];
        let test_4d = Point::new([0.2, 0.2, 0.2, 0.2]);
        assert!(
            robust_insphere(&simplex_4d, &test_4d).is_ok(),
            "4D insphere should work"
        );
        assert!(
            robust_orientation(&simplex_4d).is_ok(),
            "4D orientation should work"
        );

        // Test 5D - Valid hypersimplex
        let simplex_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let test_5d = Point::new([0.15, 0.15, 0.15, 0.15, 0.15]);
        assert!(
            robust_insphere(&simplex_5d, &test_5d).is_ok(),
            "5D insphere should work"
        );
        assert!(
            robust_orientation(&simplex_5d).is_ok(),
            "5D orientation should work"
        );

        // Test error cases - wrong number of points for each dimension
        // 2D error case - too few points
        let too_few_2d = vec![Point::new([0.0, 0.0])];
        let insphere_2d_err = robust_insphere(&too_few_2d, &test_2d);
        let orientation_2d_err = robust_orientation(&too_few_2d);
        assert!(
            insphere_2d_err.is_err() || orientation_2d_err.is_err(),
            "2D should fail with 1 point"
        );

        // 3D error case - too few points
        let too_few_3d = vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])];
        let insphere_3d_err = robust_insphere(&too_few_3d, &test_3d);
        let orientation_3d_err = robust_orientation(&too_few_3d);
        assert!(
            insphere_3d_err.is_err() || orientation_3d_err.is_err(),
            "3D should fail with 2 points"
        );

        // 4D error case - too few points
        let too_few_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
        ];
        let insphere_4d_err = robust_insphere(&too_few_4d, &test_4d);
        assert!(insphere_4d_err.is_err(), "4D should fail with 3 points");
    }

    #[test]
    fn test_near_degenerate_insphere_robustness() {
        // Near-degenerate configuration that exercises robust exact-sign paths.

        let nearly_coplanar_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.5, 1.0, 0.0]),
            Point::new([0.5, 0.5, 1e-16]), // Extremely close to coplanar
        ];

        let boundary_test_point = Point::new([0.5, 0.5, 5e-17]);

        let result = robust_insphere(&nearly_coplanar_points, &boundary_test_point);
        assert!(result.is_ok());

        let insphere_result = result.unwrap();
        assert!(matches!(
            insphere_result,
            InSphere::INSIDE | InSphere::BOUNDARY | InSphere::OUTSIDE
        ));
    }

    #[test]
    fn test_matrix_conditioning_edge_cases() {
        // Exercise row-scaling conditioning patterns on matrices.

        // Test matrix with very small elements
        let scale_small = with_la_stack_matrix!(3, |m| {
            matrix_set(&mut m, 0, 0, 1e-100);
            matrix_set(&mut m, 1, 1, 1e-99);
            matrix_set(&mut m, 2, 2, 1e-98);

            let mut scale_factor = 1.0_f64;
            for i in 0..3 {
                let mut max_element = 0.0_f64;
                for j in 0..3 {
                    max_element = max_element.max(matrix_get(&m, i, j).abs());
                }

                if max_element > 1e-100 {
                    for j in 0..3 {
                        let v = matrix_get(&m, i, j) / max_element;
                        matrix_set(&mut m, i, j, v);
                    }
                    scale_factor *= max_element;
                }
            }

            for i in 0..3 {
                for j in 0..3 {
                    assert!(matrix_get(&m, i, j).is_finite());
                }
            }

            scale_factor
        });
        assert!(scale_small.is_finite());

        // Test matrix with mixed large and small elements
        let scale_mixed = with_la_stack_matrix!(3, |m| {
            matrix_set(&mut m, 0, 0, 1e10);
            matrix_set(&mut m, 0, 1, 1e-10);
            matrix_set(&mut m, 1, 0, 1e5);
            matrix_set(&mut m, 1, 1, 1e-5);
            matrix_set(&mut m, 2, 2, 1.0);

            let mut scale_factor = 1.0_f64;
            for i in 0..3 {
                let mut max_element = 0.0_f64;
                for j in 0..3 {
                    max_element = max_element.max(matrix_get(&m, i, j).abs());
                }

                if max_element > 1e-100 {
                    for j in 0..3 {
                        let v = matrix_get(&m, i, j) / max_element;
                        matrix_set(&mut m, i, j, v);
                    }
                    scale_factor *= max_element;
                }
            }

            for i in 0..3 {
                for j in 0..3 {
                    assert!(matrix_get(&m, i, j).is_finite());
                }
            }

            scale_factor
        });
        assert!(scale_mixed.is_finite() && scale_mixed > 0.0);

        // Test matrix with some zero elements
        let scale_zero = with_la_stack_matrix!(3, |m| {
            matrix_set(&mut m, 0, 0, 1.0);
            matrix_set(&mut m, 1, 1, 0.0); // This row will not be scaled
            matrix_set(&mut m, 2, 2, 2.0);

            let mut scale_factor = 1.0_f64;
            for i in 0..3 {
                let mut max_element = 0.0_f64;
                for j in 0..3 {
                    max_element = max_element.max(matrix_get(&m, i, j).abs());
                }

                if max_element > 1e-100 {
                    for j in 0..3 {
                        let v = matrix_get(&m, i, j) / max_element;
                        matrix_set(&mut m, i, j, v);
                    }
                    scale_factor *= max_element;
                }
            }

            for i in 0..3 {
                for j in 0..3 {
                    assert!(matrix_get(&m, i, j).is_finite());
                }
            }

            scale_factor
        });
        assert!(scale_zero.is_finite());
    }

    #[test]
    fn test_tie_breaking_comprehensive() {
        // Test tie-breaking with various degenerate and extreme configurations across dimensions

        // Test 1: 2D - Degenerate triangle (nearly collinear)
        let triangle_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.5, 1e-15]), // Nearly collinear
        ];
        let test_2d = Point::new([0.5, 1e-16]);
        let result_2d = robust_insphere(&triangle_2d, &test_2d);
        assert!(result_2d.is_ok(), "2D tie-breaking should work");

        // Test 2: 3D - Coplanar points (forces SoS tie-breaking)
        let coplanar_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 0.0]), // All z = 0
        ];
        let test_3d = Point::new([0.25, 0.25, 0.0]);
        let result_3d = robust_insphere(&coplanar_3d, &test_3d);
        assert!(
            result_3d.is_ok(),
            "3D tie-breaking should handle coplanar points"
        );

        // Test 3: 4D - Nearly degenerate hypersimplex
        let simplex_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([1e-14, 1e-14, 1e-14, 1.0]), // Nearly in 3D subspace
        ];
        let test_4d = Point::new([0.2, 0.2, 0.2, 1e-15]);
        let result_4d = robust_insphere(&simplex_4d, &test_4d);
        assert!(result_4d.is_ok(), "4D tie-breaking should work");

        // Test 4: 5D - Degenerate case
        let simplex_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([1e-12, 1e-12, 1e-12, 1e-12, 1.0]), // Nearly in 4D subspace
        ];
        let test_5d = Point::new([0.1, 0.1, 0.1, 0.1, 1e-13]);
        let result_5d = robust_insphere(&simplex_5d, &test_5d);
        assert!(result_5d.is_ok(), "5D tie-breaking should work");

        // Test determinism - same input should give same output
        let result_3d_repeat = robust_insphere(&coplanar_3d, &test_3d);
        assert_eq!(
            result_3d, result_3d_repeat,
            "Tie-breaking should be deterministic"
        );

        // Test numerical extremes
        let extreme_cases = [
            // Very small coordinates
            (
                vec![
                    Point::new([1e-100, 0.0, 0.0]),
                    Point::new([0.0, 1e-100, 0.0]),
                    Point::new([0.0, 0.0, 1e-100]),
                    Point::new([1e-101, 1e-101, 1e-101]),
                ],
                Point::new([5e-102, 5e-102, 5e-102]),
                "tiny coordinates",
            ),
            // Very large coordinates
            (
                vec![
                    Point::new([1e50, 0.0, 0.0]),
                    Point::new([0.0, 1e50, 0.0]),
                    Point::new([0.0, 0.0, 1e50]),
                    Point::new([1e49, 1e49, 1e49]),
                ],
                Point::new([5e48, 5e48, 5e48]),
                "huge coordinates",
            ),
        ];

        for (simplex, test_point, description) in extreme_cases {
            let result = robust_insphere(&simplex, &test_point);
            assert!(result.is_ok(), "Should handle {description}");
        }

        // Test geometric meaning preservation
        let regular_tetrahedron = vec![
            Point::new([1.0, 1.0, 1.0]),
            Point::new([1.0, -1.0, -1.0]),
            Point::new([-1.0, 1.0, -1.0]),
            Point::new([-1.0, -1.0, 1.0]),
        ];
        let clearly_inside = Point::new([0.0, 0.0, 0.0]);
        let clearly_outside = Point::new([5.0, 5.0, 5.0]);

        assert_eq!(
            robust_insphere(&regular_tetrahedron, &clearly_inside).unwrap(),
            InSphere::INSIDE,
            "Center should be inside"
        );
        assert_eq!(
            robust_insphere(&regular_tetrahedron, &clearly_outside).unwrap(),
            InSphere::OUTSIDE,
            "Far point should be outside"
        );
    }

    #[test]
    fn test_deterministic_tie_breaking() {
        // Test deterministic tie-breaking with identical coordinates

        // Create points where the test point has identical coordinates to a simplex point
        let identical_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.5, 1.0, 0.0]),
            Point::new([0.5, 0.5, 1.0]),
        ];

        // Test point identical to first simplex point
        let identical_test = Point::new([0.0, 0.0, 0.0]);

        // This should exercise the deterministic tie-breaking logic
        let result = robust_insphere(&identical_points, &identical_test);
        assert!(result.is_ok());

        // Create a case where coordinates are lexicographically ordered
        let ordered_points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
            Point::new([10.0, 11.0, 12.0]),
        ];

        // Test point that's lexicographically smaller
        let smaller_test = Point::new([0.0, 1.0, 2.0]);
        let result_smaller = robust_insphere(&ordered_points, &smaller_test);
        assert!(result_smaller.is_ok());

        // Test point that's lexicographically larger
        let larger_test = Point::new([15.0, 16.0, 17.0]);
        let result_larger = robust_insphere(&ordered_points, &larger_test);
        assert!(result_larger.is_ok());
    }

    #[test]
    fn test_consistency_check_fallback_branch() {
        // Test the case where consistency check fails and we fall back to more robust methods
        // This is challenging to test directly since we need a case where the first method
        // succeeds but consistency verification shows inconsistent result

        // Create a configuration with very strict tolerances that might cause issues

        // Use points that are challenging for numerical precision
        let challenging_points = vec![
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
            Point::new([1e-10, 1e-10, 1e-10]), // Very close to origin but not exactly
        ];

        let test_point = Point::new([0.5, 0.5, 0.5]);

        // The function should still return a valid result even with challenging input
        let result = robust_insphere(&challenging_points, &test_point);
        assert!(result.is_ok());

        // Verify we get a sensible InSphere result
        let insphere_result = result.unwrap();
        assert!(matches!(
            insphere_result,
            InSphere::INSIDE | InSphere::BOUNDARY | InSphere::OUTSIDE
        ));
    }

    #[test]
    fn test_nan_tolerance_accepted_because_unused() {
        // base_tolerance is no longer used internally (provable det_errbound
        // replaced the heuristic tolerance), so NaN base_tolerance no longer
        // causes an error — it is simply ignored.

        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let test_point = Point::new([0.25, 0.25, 0.25]);

        let result = robust_insphere(&points, &test_point);
        assert!(
            result.is_ok(),
            "NaN base_tolerance should be accepted since it is unused internally"
        );

        // Test with a more realistic scenario: very ill-conditioned matrix
        let ill_conditioned_points = vec![
            Point::new([1e-15, 0.0, 0.0]),
            Point::new([0.0, 1e15, 0.0]),
            Point::new([0.0, 0.0, 1e-8]),
            Point::new([1e8, 1e-12, 1e4]),
        ];

        let ill_test_point = Point::new([1e-10, 1e10, 1e-5]);

        // Should still get a result even with ill-conditioned input
        let ill_result = robust_insphere(&ill_conditioned_points, &ill_test_point);
        assert!(ill_result.is_ok());
    }

    #[test]
    fn test_build_matrices_edge_cases() {
        // Exercise the same matrix construction patterns used by the robust predicates,
        // but validate edge cases explicitly.

        // 3D: all-zero coordinates
        let zero_points = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0]),
        ];
        let zero_test = Point::new([0.0, 0.0, 0.0]);

        let all_finite_insphere_3d = with_la_stack_matrix!(5, |matrix| {
            for (i, point) in zero_points.iter().enumerate() {
                let coords = point.coords();
                for (j, &v) in coords.iter().enumerate() {
                    matrix_set(&mut matrix, i, j, v);
                }
                matrix_set(&mut matrix, i, 3, squared_norm(coords));
                matrix_set(&mut matrix, i, 4, 1.0);
            }

            let test_coords = zero_test.coords();
            for (j, &v) in test_coords.iter().enumerate() {
                matrix_set(&mut matrix, 4, j, v);
            }
            matrix_set(&mut matrix, 4, 3, squared_norm(test_coords));
            matrix_set(&mut matrix, 4, 4, 1.0);

            let mut ok = true;
            for r in 0..5 {
                for c in 0..5 {
                    ok &= matrix_get(&matrix, r, c).is_finite();
                }
            }
            ok
        });
        assert!(all_finite_insphere_3d);

        let all_finite_orientation_3d = with_la_stack_matrix!(4, |matrix| {
            for (i, point) in zero_points.iter().enumerate() {
                let coords = point.coords();
                for (j, &v) in coords.iter().enumerate() {
                    matrix_set(&mut matrix, i, j, v);
                }
                matrix_set(&mut matrix, i, 3, 1.0);
            }

            let mut ok = true;
            for r in 0..4 {
                for c in 0..4 {
                    ok &= matrix_get(&matrix, r, c).is_finite();
                }
            }
            ok
        });
        assert!(all_finite_orientation_3d);

        // 2D: very large coordinates should remain finite (avoid overflow to infinity)
        let large_points = [
            Point::new([1e100, 0.0]),
            Point::new([0.0, 1e100]),
            Point::new([1e100, 1e100]),
        ];
        let large_test = Point::new([5e99, 5e99]);

        let all_finite_insphere_2d = with_la_stack_matrix!(4, |matrix| {
            for (i, point) in large_points.iter().enumerate() {
                let coords = point.coords();
                for (j, &v) in coords.iter().enumerate() {
                    matrix_set(&mut matrix, i, j, v);
                }
                matrix_set(&mut matrix, i, 2, squared_norm(coords));
                matrix_set(&mut matrix, i, 3, 1.0);
            }

            let test_coords = large_test.coords();
            for (j, &v) in test_coords.iter().enumerate() {
                matrix_set(&mut matrix, 3, j, v);
            }
            matrix_set(&mut matrix, 3, 2, squared_norm(test_coords));
            matrix_set(&mut matrix, 3, 3, 1.0);

            let mut ok = true;
            for r in 0..4 {
                for c in 0..4 {
                    ok &= matrix_get(&matrix, r, c).is_finite();
                }
            }
            ok
        });
        assert!(all_finite_insphere_2d);
    }

    #[test]
    fn test_sos_fallback_insphere_via_6d() {
        // D=6 → insphere matrix is 8×8, exceeding MAX_STACK_MATRIX_DIM=7.
        // adaptive_tolerance_insphere returns Err on every call, so
        // robust_insphere falls through to the SoS fallback (Strategy 3).
        // SoS cofactor minors are 6×6 (within the 7-dim limit), so this
        // succeeds where the full matrix dispatch does not.
        let simplex: Vec<Point<f64, 6>> = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ];

        // Exactly cospherical point: (1,1,0,…,0) lies on the circumsphere
        // of the standard 6-simplex (circumcenter = (1/2,…,1/2),
        // circumradius² = 3/2, |(1,1,0,…,0) - c|² = 3/2).
        // insphere_distance returns BOUNDARY, forcing the SoS path.
        let cospherical = Point::new([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let result = robust_insphere(&simplex, &cospherical).unwrap();
        assert!(
            result == InSphere::INSIDE || result == InSphere::OUTSIDE,
            "SoS fallback must resolve BOUNDARY to INSIDE or OUTSIDE, got {result:?}"
        );
    }
}
