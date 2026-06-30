//! Enhanced geometric predicates with improved numerical robustness.
//!
//! This module provides robust orientation and insphere predicates for
//! Delaunay triangulation. The predicates layer exact arithmetic with optional
//! diagnostic consistency checking and Simulation of Simplicity (`SoS`) fallback
//! for degenerate and near-degenerate point configurations.

#![forbid(unsafe_code)]

use super::predicates::{
    InSphere, Orientation, insphere_distance, relative_insphere_classification,
    relative_insphere_determinant_sign, relative_insphere_signs, try_orientation_from_matrix,
};
use crate::core::collections::{MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};
use crate::geometry::matrix::{MAX_STACK_MATRIX_DIM, matrix_set};
use crate::geometry::point::Point;
use crate::geometry::sos::{sos_insphere_sign, sos_orientation_sign};
use crate::geometry::traits::coordinate::CoordinateConversionError;
use core::{cmp::Ordering, hint::cold_path};
#[cfg(test)]
use std::cell::Cell;
use std::sync::LazyLock;

static PROCESS_WIDE_STRICT_INSPHERE_CONSISTENCY: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("DELAUNAY_STRICT_INSPHERE_CONSISTENCY").is_some());

#[cfg(test)]
thread_local! {
    static STRICT_INSPHERE_CONSISTENCY_TEST_OVERRIDE: Cell<Option<bool>> =
        const { Cell::new(None) };
}

#[cfg(test)]
struct StrictInsphereConsistencyOverrideGuard {
    previous: Option<bool>,
}

#[cfg(test)]
impl Drop for StrictInsphereConsistencyOverrideGuard {
    fn drop(&mut self) {
        STRICT_INSPHERE_CONSISTENCY_TEST_OVERRIDE.with(|override_value| {
            override_value.set(self.previous);
        });
    }
}

/// Returns whether strict insphere consistency diagnostics are active.
///
/// Production code reads `DELAUNAY_STRICT_INSPHERE_CONSISTENCY` once per
/// process. Unit tests can override the value for the current test thread so
/// branch coverage does not depend on process-wide environment mutation.
fn strict_insphere_consistency_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = STRICT_INSPHERE_CONSISTENCY_TEST_OVERRIDE.with(Cell::get) {
        return enabled;
    }

    *PROCESS_WIDE_STRICT_INSPHERE_CONSISTENCY
}

/// Overrides strict insphere consistency diagnostics for the current test thread.
#[cfg(test)]
fn set_strict_insphere_consistency_for_current_test(
    enabled: bool,
) -> StrictInsphereConsistencyOverrideGuard {
    let previous = STRICT_INSPHERE_CONSISTENCY_TEST_OVERRIDE.with(|override_value| {
        let previous = override_value.get();
        override_value.set(Some(enabled));
        previous
    });
    StrictInsphereConsistencyOverrideGuard { previous }
}

/// Result of consistency verification between different insphere methods.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::ConsistencyResult;
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
#[non_exhaustive]
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
/// The predicate uses the following strategies, in order:
///
/// 1. Exact-sign determinant evaluation using relative coordinates for the
///    supported exact-insphere dimensions.
/// 2. If the process-wide `DELAUNAY_STRICT_INSPHERE_CONSISTENCY` snapshot is
///    set, a diagnostic consistency check against a distance-based insphere.
///    This does not override the exact result; it only hard-fails for
///    deterministic witness capture.
/// 3. A [`Simulation of Simplicity`](crate::geometry::sos) fallback. This is
///    only reached when the exact-sign computation itself is unsupported, such
///    as D ≥ 6 where the insphere matrix exceeds the stack-matrix limit.
///
/// # Sign Convention
///
/// - The determinant sign is interpreted relative to the simplex orientation.
/// - If the simplex orientation is [`Orientation::POSITIVE`], a positive
///   determinant classifies `test_point` as [`InSphere::INSIDE`].
/// - If the simplex orientation is [`Orientation::NEGATIVE`], the
///   interpretation is swapped.
/// - [`Orientation::DEGENERATE`] yields [`InSphere::BOUNDARY`] conservatively.
///
/// # Arguments
///
/// * `simplex_points` - Exactly `D + 1` points defining the simplex.
/// * `test_point` - The query point to classify relative to the simplex
///   circumsphere.
///
/// # Returns
///
/// Returns [`InSphere::INSIDE`], [`InSphere::BOUNDARY`], or
/// [`InSphere::OUTSIDE`] when classification succeeds.
///
/// # Errors
///
/// Returns [`CoordinateConversionError`] if `simplex_points.len() != D + 1`,
/// if a coordinate cannot be converted for predicate evaluation, if a relative
/// squared norm is non-finite, or if strict insphere-consistency diagnostics are
/// enabled and detect a determinant/distance disagreement.
///
/// # Complexity
///
/// The f64 fast-filter path is O(D³). When the exact Bareiss path is triggered
/// for near-degenerate inputs, arbitrary-precision rational arithmetic increases
/// the cost significantly beyond O(D³).
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{
///     Coordinate, CoordinateConversionError, InSphere, Point, robust_insphere,
/// };
///
/// # fn main() -> Result<(), CoordinateConversionError> {
///
/// let tetra = vec![
///     Point::try_from([0.0, 0.0, 0.0])?,
///     Point::try_from([1.0, 0.0, 0.0])?,
///     Point::try_from([0.0, 1.0, 0.0])?,
///     Point::try_from([0.0, 0.0, 1.0])?,
/// ];
///
/// let inside = Point::try_from([0.25, 0.25, 0.25])?;
/// let outside = Point::try_from([2.0, 2.0, 2.0])?;
///
/// let r_in = robust_insphere(&tetra, &inside)?;
/// let r_out = robust_insphere(&tetra, &outside)?;
/// assert_eq!(r_in, InSphere::INSIDE);
/// assert_eq!(r_out, InSphere::OUTSIDE);
/// # Ok(())
/// # }
/// ```
///
/// # Notes
///
/// Absolute-coordinate squared-norm overflow is avoided by using the
/// relative-coordinate lifted formulation shared with
/// [`crate::geometry::predicates::insphere_lifted`] for D ≤ 5. If a relative
/// squared norm is non-finite, the error is returned instead of falling through
/// to a symbolic classification.
pub fn robust_insphere<const D: usize>(
    simplex_points: &[Point<D>],
    test_point: &Point<D>,
) -> Result<InSphere, CoordinateConversionError> {
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::InvalidSimplexPointCount {
            actual: simplex_points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    // Strategy 1: Exact-sign determinant approach with adaptive tolerance.
    match relative_exact_insphere(simplex_points, test_point) {
        Ok(result) => {
            if strict_insphere_consistency_enabled() {
                // Strategy 2: Diagnostic consistency check against distance-based insphere.
                // The exact-sign result is provably correct for finite inputs; a disagreement
                // from insphere_distance reflects f64 rounding in the distance-based check,
                // not a defect in the exact predicate. Keep this opt-in because RobustKernel
                // sits on hot Delaunay repair paths.
                if let ConsistencyResult::Inconsistent(error) =
                    verify_insphere_consistency(simplex_points, test_point, result)
                {
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
        Err(error) if should_use_sos_fallback(&error) => {}
        Err(error) => return Err(error),
    }

    // Strategy 3: Geometric + SoS fallback — only reached when exact-sign
    // computation itself failed (e.g. unsupported matrix size for D ≥ 6).
    // `cold_path()` nudges the optimizer to keep Strategies 1–2 lean; the
    // vast majority of calls return before reaching this point.
    //
    // First try insphere_distance (circumcenter/radius based — no matrix
    // determinant needed, works at any dimension).  This handles the
    // non-degenerate cases correctly.  Only if the result is BOUNDARY
    // (truly degenerate) do we apply SoS tie-breaking.
    cold_path();
    if let Ok(geometric_result) = insphere_distance(simplex_points, *test_point)
        && geometric_result != InSphere::BOUNDARY
    {
        return Ok(geometric_result);
    }

    // SoS tie-breaking for the truly degenerate case (BOUNDARY or
    // insphere_distance itself failed).  The SoS cofactor minors are one
    // size smaller, so this succeeds where the full insphere matrix
    // dispatch does not.
    let mut f64_simplex: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
        SmallBuffer::with_capacity(simplex_points.len());
    for point in simplex_points {
        f64_simplex.push(*point);
    }
    let f64_test = *test_point;

    // Use exact orientation when available; fall back to SoS only when the
    // exact predicate reports DEGENERATE (or fails entirely).
    let abs_orient: i32 = match robust_orientation(simplex_points) {
        Ok(Orientation::POSITIVE) => 1,
        Ok(Orientation::NEGATIVE) => -1,
        _ => sos_orientation_sign(&f64_simplex)?,
    };
    let raw_insphere = sos_insphere_sign(&f64_simplex, &f64_test)?;

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

/// Robust insphere for a simplex already known to be in positive orientation.
///
/// This skips the orientation determinant that [`robust_insphere`] normally uses
/// to normalize the lifted determinant sign. It is intended for hot triangulation
/// paths that evaluate stored simplices after orientation canonicalization.
pub(crate) fn robust_insphere_positive_oriented<const D: usize>(
    simplex_points: &[Point<D>],
    test_point: &Point<D>,
) -> Result<InSphere, CoordinateConversionError> {
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::InvalidSimplexPointCount {
            actual: simplex_points.len(),
            expected: D + 1,
            dimension: D,
        });
    }
    if D > 5 {
        return robust_insphere(simplex_points, test_point);
    }

    let determinant_sign = relative_insphere_determinant_sign(simplex_points, test_point)?;
    let orient_factor = if D.is_multiple_of(2) { -1 } else { 1 };
    let effective_sign = determinant_sign * orient_factor;
    let result = match effective_sign.cmp(&0) {
        Ordering::Greater => InSphere::INSIDE,
        Ordering::Less => InSphere::OUTSIDE,
        Ordering::Equal => InSphere::BOUNDARY,
    };

    if strict_insphere_consistency_enabled()
        && let ConsistencyResult::Inconsistent(error) =
            verify_insphere_consistency(simplex_points, test_point, result)
    {
        let details =
            format!("{error}; simplex_points={simplex_points:?}; test_point={test_point:?}");
        return Err(CoordinateConversionError::InsphereInconsistency {
            simplex_points: format!("{simplex_points:?}"),
            test_point: format!("{test_point:?}"),
            details,
        });
    }

    Ok(result)
}

/// Whether an exact-sign failure should fall through to the geometric + `SoS` fallback.
#[inline]
const fn should_use_sos_fallback(error: &CoordinateConversionError) -> bool {
    matches!(
        error,
        CoordinateConversionError::UnsupportedMatrixDimension { .. }
    )
}

/// Insphere test using exact relative-coordinate determinants when supported.
///
/// Uses the relative-coordinate lifted formulation shared with
/// [`super::predicates::insphere_lifted`] for provably correct sign
/// classification on finite local geometry.
fn relative_exact_insphere<const D: usize>(
    simplex_points: &[Point<D>],
    test_point: &Point<D>,
) -> Result<InSphere, CoordinateConversionError> {
    if D > 5 {
        return Err(CoordinateConversionError::UnsupportedMatrixDimension {
            requested: D + 2,
            max: MAX_STACK_MATRIX_DIM,
        });
    }

    let signs = relative_insphere_signs(simplex_points, test_point)?;
    if signs.relative_orientation == 0 {
        cold_path();
        Ok(InSphere::BOUNDARY)
    } else {
        Ok(relative_insphere_classification(signs))
    }
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
/// use delaunay::prelude::geometry::Point;
/// use delaunay::prelude::geometry::Orientation;
/// use delaunay::prelude::geometry::robust_orientation;
/// use delaunay::prelude::geometry::{Coordinate, CoordinateConversionError};
///
/// # fn main() -> Result<(), CoordinateConversionError> {
/// let tri = vec![
///     Point::try_from([0.0, 0.0])?,
///     Point::try_from([1.0, 0.0])?,
///     Point::try_from([0.0, 1.0])?,
/// ];
/// let orientation = robust_orientation(&tri)?;
/// assert_eq!(orientation, Orientation::POSITIVE);
/// # Ok(())
/// # }
/// ```
pub fn robust_orientation<const D: usize>(
    simplex_points: &[Point<D>],
) -> Result<Orientation, CoordinateConversionError> {
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::InvalidSimplexPointCount {
            actual: simplex_points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    let k = D + 1;

    try_with_la_stack_matrix!(k, |matrix| {
        for (i, point) in simplex_points.iter().enumerate() {
            for (j, &v) in point.coords().iter().enumerate() {
                matrix_set(&mut matrix, i, j, v)?;
            }

            // Add constant term
            matrix_set(&mut matrix, i, D, 1.0)?;
        }

        // Route through the exact-sign orientation helper for provably correct
        // orientation classification on finite inputs.
        Ok(try_orientation_from_matrix(&matrix, k)?)
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
fn verify_insphere_consistency<const D: usize>(
    simplex_points: &[Point<D>],
    test_point: &Point<D>,
    determinant_result: InSphere,
) -> ConsistencyResult {
    // Use the existing distance-based insphere test for verification
    insphere_distance(simplex_points, *test_point).map_or(
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
    use crate::geometry::matrix::{Matrix, matrix_get};
    use crate::geometry::point::Point;
    use crate::geometry::predicates;
    use crate::geometry::util::squared_norm;
    use num_traits::NumCast;
    use rand::{RngExt, SeedableRng};
    use std::assert_matches;
    use std::thread;

    fn matrix_block_is_finite<const N: usize>(matrix: &Matrix<N>, k: usize) -> bool {
        (0..k).all(|row| (0..k).all(|column| matrix_get(matrix, row, column).unwrap().is_finite()))
    }

    #[test]
    fn test_robust_insphere_general() {
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        // Test point clearly inside
        let inside_point = Point::try_new([0.25, 0.25, 0.25]).expect("finite point coordinates");
        let result = robust_insphere(&points, &inside_point).unwrap();
        assert_eq!(result, InSphere::INSIDE);

        // Test point clearly outside
        let outside_point = Point::try_new([2.0, 2.0, 2.0]).expect("finite point coordinates");
        let result = robust_insphere(&points, &outside_point).unwrap();
        assert_eq!(result, InSphere::OUTSIDE);
    }

    #[test]
    fn test_positive_oriented_insphere_matches_robust_insphere() {
        let simplex_2d = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            robust_orientation(&simplex_2d).unwrap(),
            Orientation::POSITIVE
        );
        for point in [
            Point::try_new([0.2, 0.2]).expect("finite point coordinates"),
            Point::try_new([2.0, 2.0]).expect("finite point coordinates"),
        ] {
            assert_eq!(
                robust_insphere_positive_oriented(&simplex_2d, &point).unwrap(),
                robust_insphere(&simplex_2d, &point).unwrap()
            );
        }

        let simplex_3d = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            robust_orientation(&simplex_3d).unwrap(),
            Orientation::POSITIVE
        );
        for point in [
            Point::try_new([0.2, 0.2, 0.2]).expect("finite point coordinates"),
            Point::try_new([2.0, 2.0, 2.0]).expect("finite point coordinates"),
        ] {
            assert_eq!(
                robust_insphere_positive_oriented(&simplex_3d, &point).unwrap(),
                robust_insphere(&simplex_3d, &point).unwrap()
            );
        }

        let simplex_4d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            robust_orientation(&simplex_4d).unwrap(),
            Orientation::POSITIVE
        );
        for point in [
            Point::try_new([0.2, 0.2, 0.2, 0.2]).expect("finite point coordinates"),
            Point::try_new([2.0, 2.0, 2.0, 2.0]).expect("finite point coordinates"),
        ] {
            assert_eq!(
                robust_insphere_positive_oriented(&simplex_4d, &point).unwrap(),
                robust_insphere(&simplex_4d, &point).unwrap()
            );
        }

        let simplex_5d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            robust_orientation(&simplex_5d).unwrap(),
            Orientation::POSITIVE
        );
        for point in [
            Point::try_new([0.2, 0.2, 0.2, 0.2, 0.2]).expect("finite point coordinates"),
            Point::try_new([2.0, 2.0, 2.0, 2.0, 2.0]).expect("finite point coordinates"),
        ] {
            assert_eq!(
                robust_insphere_positive_oriented(&simplex_5d, &point).unwrap(),
                robust_insphere(&simplex_5d, &point).unwrap()
            );
        }
    }

    #[test]
    fn test_positive_oriented_insphere_boundary_and_invalid_count() {
        let simplex = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];
        assert_eq!(robust_orientation(&simplex).unwrap(), Orientation::POSITIVE);

        let boundary = Point::try_new([1.0, 1.0]).expect("finite point coordinates");
        assert_eq!(
            robust_insphere_positive_oriented(&simplex, &boundary).unwrap(),
            InSphere::BOUNDARY
        );

        let too_few = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        ];
        let err = robust_insphere_positive_oriented(
            &too_few,
            &Point::try_new([0.25, 0.25]).expect("finite point coordinates"),
        )
        .unwrap_err();
        assert_eq!(
            err,
            CoordinateConversionError::InvalidSimplexPointCount {
                actual: 2,
                expected: 3,
                dimension: 2,
            }
        );
    }

    #[test]
    fn test_positive_oriented_insphere_uses_robust_fallback_above_stack_dimension() {
        let simplex: Vec<Point<6>> = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        assert_eq!(robust_orientation(&simplex).unwrap(), Orientation::POSITIVE);
        let test_point =
            Point::try_new([0.15, 0.15, 0.15, 0.15, 0.15, 0.15]).expect("finite point coordinates");

        assert_eq!(
            robust_insphere_positive_oriented(&simplex, &test_point).unwrap(),
            robust_insphere(&simplex, &test_point).unwrap()
        );
    }

    #[test]
    fn test_verify_insphere_consistency_reports_direct_contradiction() {
        let simplex = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let inside = Point::try_new([0.25, 0.25, 0.25]).expect("finite point coordinates");

        assert_eq!(
            verify_insphere_consistency(&simplex, &inside, InSphere::OUTSIDE),
            ConsistencyResult::Inconsistent(InsphereConsistencyError::DirectContradiction {
                determinant_result: InSphere::OUTSIDE,
                distance_result: InSphere::INSIDE,
            })
        );
    }

    #[test]
    fn test_strict_insphere_consistency_override_is_thread_local() {
        let process_wide_setting = *PROCESS_WIDE_STRICT_INSPHERE_CONSISTENCY;
        assert_eq!(strict_insphere_consistency_enabled(), process_wide_setting);

        {
            let _guard = set_strict_insphere_consistency_for_current_test(!process_wide_setting);
            assert_eq!(strict_insphere_consistency_enabled(), !process_wide_setting);

            {
                let _nested_guard =
                    set_strict_insphere_consistency_for_current_test(process_wide_setting);
                assert_eq!(strict_insphere_consistency_enabled(), process_wide_setting);
            }
            assert_eq!(strict_insphere_consistency_enabled(), !process_wide_setting);

            let child_setting = thread::spawn(strict_insphere_consistency_enabled)
                .join()
                .expect("strict insphere consistency check thread should not panic");
            assert_eq!(child_setting, process_wide_setting);
        }

        assert_eq!(strict_insphere_consistency_enabled(), process_wide_setting);
    }

    #[test]
    fn test_strict_insphere_consistency_override_exercises_error_path() {
        let _guard = set_strict_insphere_consistency_for_current_test(true);
        let simplex = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let test_point = Point::try_new([0.25, 0.25, 0.25]).expect("finite point coordinates");

        assert_eq!(
            robust_insphere(&simplex, &test_point).unwrap(),
            InSphere::INSIDE
        );
        assert!(
            matches!(
                robust_insphere_positive_oriented(&simplex, &test_point),
                Err(CoordinateConversionError::InsphereInconsistency { .. })
            ),
            "strict consistency override should exercise the positive-oriented diagnostic error path"
        );
    }

    #[test]
    fn test_robust_orientation() {
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        let result = robust_orientation(&points).unwrap();

        // Should detect orientation (exact result depends on coordinate system)
        assert_matches!(result, Orientation::POSITIVE | Orientation::NEGATIVE);
    }

    #[test]
    fn test_robust_orientation_positive_triangle_2d() {
        // Canonical CCW triangle to exercise the robust_orientation matrix path
        // and confirm the exact-sign helper returns POSITIVE.
        let points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];

        let robust = robust_orientation(&points).unwrap();
        let reference = predicates::simplex_orientation(&points).unwrap();

        assert_eq!(robust, Orientation::POSITIVE);
        assert_eq!(robust, reference);
    }

    #[test]
    fn test_robust_orientation_ccw_triangle_2d() {
        // Standard CCW triangle — robust_orientation uses provable
        // det_errbound and exact Bareiss arithmetic, no configuration.
        let points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
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
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, eps]).expect("finite point coordinates"),
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
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, eps]).expect("finite point coordinates"),
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
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];
        // Circumcenter = (0.5, 0.5), circumradius = sqrt(0.5).
        // Place test points along the +x direction from the circumcenter.
        let radius = 0.5_f64.sqrt();
        let inside_point =
            Point::try_new([0.5 + radius - eps, 0.5]).expect("finite point coordinates");
        let outside_point =
            Point::try_new([0.5 + radius + eps, 0.5]).expect("finite point coordinates");

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
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        // Circumcenter = (0.5, 0.5, 0.5), circumradius = sqrt(0.75).
        // Place test points along the +x direction from the circumcenter.
        let radius = 0.75_f64.sqrt();
        let inside_point =
            Point::try_new([0.5 + radius - eps, 0.5, 0.5]).expect("finite point coordinates");
        let outside_point =
            Point::try_new([0.5 + radius + eps, 0.5, 0.5]).expect("finite point coordinates");

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
    fn test_robust_insphere_large_translated_simplex_uses_relative_formulation() {
        // Absolute squared norms overflow for these coordinates:
        // 3 × (1e154)² > f64::MAX.  The local geometry is still well-scaled,
        // so the relative-coordinate exact path should classify normally.
        let base = 1.0e154;
        let delta = 1.0e140;
        let simplex = vec![
            Point::try_new([base, base, base]).expect("finite point coordinates"),
            Point::try_new([base + delta, base, base]).expect("finite point coordinates"),
            Point::try_new([base, base + delta, base]).expect("finite point coordinates"),
            Point::try_new([base, base, base + delta]).expect("finite point coordinates"),
        ];
        let inside_coord = 0.25_f64.mul_add(delta, base);
        let outside_coord = 2.0_f64.mul_add(delta, base);
        let inside_point = Point::try_new([inside_coord, inside_coord, inside_coord])
            .expect("finite point coordinates");
        let outside_point = Point::try_new([outside_coord, outside_coord, outside_coord])
            .expect("finite point coordinates");

        assert_eq!(
            relative_exact_insphere(&simplex, &inside_point).unwrap(),
            InSphere::INSIDE
        );
        assert_eq!(
            robust_insphere(&simplex, &inside_point).unwrap(),
            InSphere::INSIDE
        );
        assert_eq!(
            relative_exact_insphere(&simplex, &outside_point).unwrap(),
            InSphere::OUTSIDE
        );
        assert_eq!(
            robust_insphere(&simplex, &outside_point).unwrap(),
            InSphere::OUTSIDE
        );
    }

    #[test]
    fn test_robust_insphere_errors_when_relative_squared_norm_overflows() {
        let simplex = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let far_point = Point::try_new([1.0e155, 0.0, 0.0]).expect("finite point coordinates");

        let error = robust_insphere(&simplex, &far_point).unwrap_err();
        assert!(
            matches!(error, CoordinateConversionError::NonFiniteValue { .. }),
            "relative squared-norm overflow should surface as a typed conversion error, got {error:?}"
        );
    }

    #[test]
    fn test_degenerate_case_handling() {
        // Create nearly coplanar points
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 0.5, 1e-15]).expect("finite point coordinates"), // Very slightly off-plane
        ];

        // Should handle gracefully
        let test_point = Point::try_new([0.25, 0.25, 1e-16]).expect("finite point coordinates");
        let result = robust_insphere(&points, &test_point);
        assert!(result.is_ok());
    }

    #[expect(
        clippy::too_many_lines,
        reason = "insphere consistency test keeps related robust predicate cases together"
    )]
    #[test]
    fn test_verify_insphere_consistency_comprehensive() {
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        // Test exact matches - all should be consistent
        let test_cases = [
            (
                Point::try_new([0.25, 0.25, 0.25]).expect("finite point coordinates"),
                InSphere::INSIDE,
                "inside point",
            ),
            (
                Point::try_new([2.0, 2.0, 2.0]).expect("finite point coordinates"),
                InSphere::OUTSIDE,
                "outside point",
            ),
            (
                Point::try_new([0.5, 0.5, 0.5]).expect("finite point coordinates"),
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
        let boundary_test_point =
            Point::try_new([0.3, 0.3, 0.3]).expect("finite point coordinates");
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
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([2.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 2.0]).expect("finite point coordinates"),
        ];
        let test_2d = Point::try_new([1.0, 0.5]).expect("finite point coordinates");
        assert!(
            verify_insphere_consistency(&triangle_2d, &test_2d, InSphere::BOUNDARY).is_consistent()
        );

        // Test edge cases with extreme coordinates and error conditions
        let edge_cases = [
            (
                vec![
                    Point::try_new([1e-10, 0.0, 0.0]).expect("finite point coordinates"),
                    Point::try_new([0.0, 1e-10, 0.0]).expect("finite point coordinates"),
                    Point::try_new([0.0, 0.0, 1e-10]).expect("finite point coordinates"),
                    Point::try_new([1e-10, 1e-10, 1e-10]).expect("finite point coordinates"),
                ],
                Point::try_new([5e-11, 5e-11, 5e-11]).expect("finite point coordinates"),
                "small coordinates",
            ),
            (
                vec![
                    Point::try_new([1e6, 0.0, 0.0]).expect("finite point coordinates"),
                    Point::try_new([0.0, 1e6, 0.0]).expect("finite point coordinates"),
                    Point::try_new([0.0, 0.0, 1e6]).expect("finite point coordinates"),
                    Point::try_new([1e6, 1e6, 1e6]).expect("finite point coordinates"),
                ],
                Point::try_new([5e5, 5e5, 5e5]).expect("finite point coordinates"),
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

        assert!(Point::<3>::try_new([f64::NAN, 0.0, 0.0]).is_err());

        // Test error conditions that should return Unverifiable
        let error_cases = [
            // Invalid simplex size
            (
                vec![
                    Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                    Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
                ],
                Point::try_new([0.5, 0.0, 0.0]).expect("finite point coordinates"),
                "too few points",
            ),
            // Degenerate simplex
            (
                vec![
                    Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                    Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
                    Point::try_new([2.0, 0.0, 0.0]).expect("finite point coordinates"),
                    Point::try_new([3.0, 0.0, 0.0]).expect("finite point coordinates"),
                ],
                Point::try_new([1.5, 0.0, 0.0]).expect("finite point coordinates"),
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
    type PeriodicWitness3d = ([Point<3>; 4], Point<3>, InSphere, InSphere);

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

    fn periodic_3d_canonical_points() -> Vec<Point<3>> {
        vec![
            Point::try_new([0.1_f64, 0.2, 0.3]).expect("finite point coordinates"),
            Point::try_new([0.4, 0.7, 0.1]).expect("finite point coordinates"),
            Point::try_new([0.7, 0.3, 0.8]).expect("finite point coordinates"),
            Point::try_new([0.2, 0.9, 0.5]).expect("finite point coordinates"),
            Point::try_new([0.8, 0.6, 0.2]).expect("finite point coordinates"),
            Point::try_new([0.5, 0.1, 0.7]).expect("finite point coordinates"),
            Point::try_new([0.3, 0.5, 0.9]).expect("finite point coordinates"),
            Point::try_new([0.6, 0.8, 0.4]).expect("finite point coordinates"),
            Point::try_new([0.9, 0.2, 0.6]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.4, 0.1]).expect("finite point coordinates"),
            Point::try_new([0.15, 0.65, 0.45]).expect("finite point coordinates"),
            Point::try_new([0.75, 0.15, 0.85]).expect("finite point coordinates"),
            Point::try_new([0.45, 0.55, 0.25]).expect("finite point coordinates"),
            Point::try_new([0.85, 0.45, 0.65]).expect("finite point coordinates"),
        ]
    }

    fn periodic_3d_builder_style_expansion(canonical_points: &[Point<3>]) -> Vec<Point<3>> {
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
                expanded.push(Point::try_new(image_coords).expect("finite point coordinates"));
            }
        }

        expanded
    }

    fn find_periodic_3d_inconsistency_witness(
        expanded: &[Point<3>],
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
            let det_result = relative_exact_insphere(&simplex, &test_point);
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
    fn test_periodic_3d_inconsistency_witness_search_seeded() {
        let canonical_points = periodic_3d_canonical_points();
        let expanded = periodic_3d_builder_style_expansion(&canonical_points);
        let witness = find_periodic_3d_inconsistency_witness(&expanded, 0x2100_0003, 200_000);

        if let Some((simplex, test_point, det, dist)) = witness {
            panic!(
                "Found periodic-3D determinant-vs-distance inconsistency: \
                 determinant={det:?}, distance={dist:?}, simplex={simplex:?}, \
                 test_point={test_point:?}"
            );
        }
    }

    #[test]
    fn test_robust_predicates_dimensional_coverage() {
        // Comprehensive test across dimensions 2D-5D with both valid and invalid cases

        // Test 2D - Valid triangle
        let triangle_2d = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 1.0]).expect("finite point coordinates"),
        ];
        let test_2d = Point::try_new([0.5, 0.3]).expect("finite point coordinates");
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
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let test_3d = Point::try_new([0.25, 0.25, 0.25]).expect("finite point coordinates");
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
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let test_4d = Point::try_new([0.2, 0.2, 0.2, 0.2]).expect("finite point coordinates");
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
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let test_5d =
            Point::try_new([0.15, 0.15, 0.15, 0.15, 0.15]).expect("finite point coordinates");
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
        let too_few_2d = vec![Point::try_new([0.0, 0.0]).expect("finite point coordinates")];
        let insphere_2d_err = robust_insphere(&too_few_2d, &test_2d);
        let orientation_2d_err = robust_orientation(&too_few_2d);
        assert!(
            insphere_2d_err.is_err() || orientation_2d_err.is_err(),
            "2D should fail with 1 point"
        );

        // 3D error case - too few points
        let too_few_3d = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
        ];
        let insphere_3d_err = robust_insphere(&too_few_3d, &test_3d);
        let orientation_3d_err = robust_orientation(&too_few_3d);
        assert!(
            insphere_3d_err.is_err() || orientation_3d_err.is_err(),
            "3D should fail with 2 points"
        );

        // 4D error case - too few points
        let too_few_4d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
        ];
        let insphere_4d_err = robust_insphere(&too_few_4d, &test_4d);
        assert!(insphere_4d_err.is_err(), "4D should fail with 3 points");
    }

    #[test]
    fn test_near_degenerate_insphere_robustness() {
        // Near-degenerate configuration that exercises robust exact-sign paths.

        let nearly_coplanar_points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 0.5, 1e-16]).expect("finite point coordinates"), // Extremely close to coplanar
        ];

        let boundary_test_point =
            Point::try_new([0.5, 0.5, 5e-17]).expect("finite point coordinates");

        let result = robust_insphere(&nearly_coplanar_points, &boundary_test_point);
        assert!(result.is_ok());

        let insphere_result = result.unwrap();
        assert_matches!(
            insphere_result,
            InSphere::INSIDE | InSphere::BOUNDARY | InSphere::OUTSIDE
        );
    }

    #[test]
    fn test_matrix_conditioning_edge_cases() {
        // Exercise row-scaling conditioning patterns on matrices.

        // Test matrix with very small elements
        let scale_small = with_la_stack_matrix!(3, |m| {
            matrix_set(&mut m, 0, 0, 1e-100).unwrap();
            matrix_set(&mut m, 1, 1, 1e-99).unwrap();
            matrix_set(&mut m, 2, 2, 1e-98).unwrap();

            let mut scale_factor = 1.0_f64;
            for i in 0..3 {
                let mut max_element = 0.0_f64;
                for j in 0..3 {
                    max_element = max_element.max(matrix_get(&m, i, j).unwrap().abs());
                }

                if max_element > 1e-100 {
                    for j in 0..3 {
                        let v = matrix_get(&m, i, j).unwrap() / max_element;
                        matrix_set(&mut m, i, j, v).unwrap();
                    }
                    scale_factor *= max_element;
                }
            }

            for i in 0..3 {
                for j in 0..3 {
                    assert!(matrix_get(&m, i, j).unwrap().is_finite());
                }
            }

            scale_factor
        });
        assert!(scale_small.is_finite());

        // Test matrix with mixed large and small elements
        let scale_mixed = with_la_stack_matrix!(3, |m| {
            matrix_set(&mut m, 0, 0, 1e10).unwrap();
            matrix_set(&mut m, 0, 1, 1e-10).unwrap();
            matrix_set(&mut m, 1, 0, 1e5).unwrap();
            matrix_set(&mut m, 1, 1, 1e-5).unwrap();
            matrix_set(&mut m, 2, 2, 1.0).unwrap();

            let mut scale_factor = 1.0_f64;
            for i in 0..3 {
                let mut max_element = 0.0_f64;
                for j in 0..3 {
                    max_element = max_element.max(matrix_get(&m, i, j).unwrap().abs());
                }

                if max_element > 1e-100 {
                    for j in 0..3 {
                        let v = matrix_get(&m, i, j).unwrap() / max_element;
                        matrix_set(&mut m, i, j, v).unwrap();
                    }
                    scale_factor *= max_element;
                }
            }

            for i in 0..3 {
                for j in 0..3 {
                    assert!(matrix_get(&m, i, j).unwrap().is_finite());
                }
            }

            scale_factor
        });
        assert!(scale_mixed.is_finite() && scale_mixed > 0.0);

        // Test matrix with some zero elements
        let scale_zero = with_la_stack_matrix!(3, |m| {
            matrix_set(&mut m, 0, 0, 1.0).unwrap();
            matrix_set(&mut m, 1, 1, 0.0).unwrap(); // This row will not be scaled
            matrix_set(&mut m, 2, 2, 2.0).unwrap();

            let mut scale_factor = 1.0_f64;
            for i in 0..3 {
                let mut max_element = 0.0_f64;
                for j in 0..3 {
                    max_element = max_element.max(matrix_get(&m, i, j).unwrap().abs());
                }

                if max_element > 1e-100 {
                    for j in 0..3 {
                        let v = matrix_get(&m, i, j).unwrap() / max_element;
                        matrix_set(&mut m, i, j, v).unwrap();
                    }
                    scale_factor *= max_element;
                }
            }

            for i in 0..3 {
                for j in 0..3 {
                    assert!(matrix_get(&m, i, j).unwrap().is_finite());
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
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 1e-15]).expect("finite point coordinates"), // Nearly collinear
        ];
        let test_2d = Point::try_new([0.5, 1e-16]).expect("finite point coordinates");
        let result_2d = robust_insphere(&triangle_2d, &test_2d);
        assert!(result_2d.is_ok(), "2D tie-breaking should work");

        // Test 2: 3D - Coplanar points (forces SoS tie-breaking)
        let coplanar_3d = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 0.5, 0.0]).expect("finite point coordinates"), // All z = 0
        ];
        let test_3d = Point::try_new([0.25, 0.25, 0.0]).expect("finite point coordinates");
        let result_3d = robust_insphere(&coplanar_3d, &test_3d);
        assert!(
            result_3d.is_ok(),
            "3D tie-breaking should handle coplanar points"
        );

        // Test 3: 4D - Nearly degenerate hypersimplex
        let simplex_4d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1e-14, 1e-14, 1e-14, 1.0]).expect("finite point coordinates"), // Nearly in 3D subspace
        ];
        let test_4d = Point::try_new([0.2, 0.2, 0.2, 1e-15]).expect("finite point coordinates");
        let result_4d = robust_insphere(&simplex_4d, &test_4d);
        assert!(result_4d.is_ok(), "4D tie-breaking should work");

        // Test 4: 5D - Degenerate case
        let simplex_5d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1e-12, 1e-12, 1e-12, 1e-12, 1.0]).expect("finite point coordinates"), // Nearly in 4D subspace
        ];
        let test_5d =
            Point::try_new([0.1, 0.1, 0.1, 0.1, 1e-13]).expect("finite point coordinates");
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
                    Point::try_new([1e-100, 0.0, 0.0]).expect("finite point coordinates"),
                    Point::try_new([0.0, 1e-100, 0.0]).expect("finite point coordinates"),
                    Point::try_new([0.0, 0.0, 1e-100]).expect("finite point coordinates"),
                    Point::try_new([1e-101, 1e-101, 1e-101]).expect("finite point coordinates"),
                ],
                Point::try_new([5e-102, 5e-102, 5e-102]).expect("finite point coordinates"),
                "tiny coordinates",
            ),
            // Very large coordinates
            (
                vec![
                    Point::try_new([1e50, 0.0, 0.0]).expect("finite point coordinates"),
                    Point::try_new([0.0, 1e50, 0.0]).expect("finite point coordinates"),
                    Point::try_new([0.0, 0.0, 1e50]).expect("finite point coordinates"),
                    Point::try_new([1e49, 1e49, 1e49]).expect("finite point coordinates"),
                ],
                Point::try_new([5e48, 5e48, 5e48]).expect("finite point coordinates"),
                "huge coordinates",
            ),
        ];

        for (simplex, test_point, description) in extreme_cases {
            let result = robust_insphere(&simplex, &test_point);
            assert!(result.is_ok(), "Should handle {description}");
        }

        // Test geometric meaning preservation
        let regular_tetrahedron = vec![
            Point::try_new([1.0, 1.0, 1.0]).expect("finite point coordinates"),
            Point::try_new([1.0, -1.0, -1.0]).expect("finite point coordinates"),
            Point::try_new([-1.0, 1.0, -1.0]).expect("finite point coordinates"),
            Point::try_new([-1.0, -1.0, 1.0]).expect("finite point coordinates"),
        ];
        let clearly_inside = Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates");
        let clearly_outside = Point::try_new([5.0, 5.0, 5.0]).expect("finite point coordinates");

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
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 0.5, 1.0]).expect("finite point coordinates"),
        ];

        // Test point identical to first simplex point
        let identical_test = Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates");

        // This should exercise the deterministic tie-breaking logic
        let result = robust_insphere(&identical_points, &identical_test);
        assert!(result.is_ok());

        // Create a case where coordinates are lexicographically ordered
        let ordered_points = vec![
            Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates"),
            Point::try_new([4.0, 5.0, 6.0]).expect("finite point coordinates"),
            Point::try_new([7.0, 8.0, 9.0]).expect("finite point coordinates"),
            Point::try_new([10.0, 11.0, 12.0]).expect("finite point coordinates"),
        ];

        // Test point that's lexicographically smaller
        let smaller_test = Point::try_new([0.0, 1.0, 2.0]).expect("finite point coordinates");
        let result_smaller = robust_insphere(&ordered_points, &smaller_test);
        assert!(result_smaller.is_ok());

        // Test point that's lexicographically larger
        let larger_test = Point::try_new([15.0, 16.0, 17.0]).expect("finite point coordinates");
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
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
            Point::try_new([1e-10, 1e-10, 1e-10]).expect("finite point coordinates"), // Very close to origin but not exactly
        ];

        let test_point = Point::try_new([0.5, 0.5, 0.5]).expect("finite point coordinates");

        // The function should still return a valid result even with challenging input
        let result = robust_insphere(&challenging_points, &test_point);
        assert!(result.is_ok());

        // Verify we get a sensible InSphere result
        let insphere_result = result.unwrap();
        assert_matches!(
            insphere_result,
            InSphere::INSIDE | InSphere::BOUNDARY | InSphere::OUTSIDE
        );
    }

    #[test]
    fn test_robust_insphere_standard_tetrahedron_inside() {
        // Standard tetrahedron with a clearly interior point — exercises
        // the exact-sign insphere path with no configuration.

        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        let test_point = Point::try_new([0.25, 0.25, 0.25]).expect("finite point coordinates");

        let result = robust_insphere(&points, &test_point);
        assert!(
            result.is_ok(),
            "robust_insphere should succeed on a standard tetrahedron"
        );

        // Test with a more realistic scenario: very ill-conditioned matrix
        let ill_conditioned_points = vec![
            Point::try_new([1e-15, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1e15, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1e-8]).expect("finite point coordinates"),
            Point::try_new([1e8, 1e-12, 1e4]).expect("finite point coordinates"),
        ];

        let ill_test_point = Point::try_new([1e-10, 1e10, 1e-5]).expect("finite point coordinates");

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
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
        ];
        let zero_test = Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates");

        let all_finite_insphere_3d = with_la_stack_matrix!(5, |matrix| {
            for (i, point) in zero_points.iter().enumerate() {
                let coords = point.coords();
                for (j, &v) in coords.iter().enumerate() {
                    matrix_set(&mut matrix, i, j, v).unwrap();
                }
                matrix_set(&mut matrix, i, 3, squared_norm(coords)).unwrap();
                matrix_set(&mut matrix, i, 4, 1.0).unwrap();
            }

            let test_coords = zero_test.coords();
            for (j, &v) in test_coords.iter().enumerate() {
                matrix_set(&mut matrix, 4, j, v).unwrap();
            }
            matrix_set(&mut matrix, 4, 3, squared_norm(test_coords)).unwrap();
            matrix_set(&mut matrix, 4, 4, 1.0).unwrap();

            matrix_block_is_finite(&matrix, 5)
        });
        assert!(all_finite_insphere_3d);

        let all_finite_orientation_3d = with_la_stack_matrix!(4, |matrix| {
            for (i, point) in zero_points.iter().enumerate() {
                let coords = point.coords();
                for (j, &v) in coords.iter().enumerate() {
                    matrix_set(&mut matrix, i, j, v).unwrap();
                }
                matrix_set(&mut matrix, i, 3, 1.0).unwrap();
            }

            matrix_block_is_finite(&matrix, 4)
        });
        assert!(all_finite_orientation_3d);

        // 2D: very large coordinates should remain finite (avoid overflow to infinity)
        let large_points = [
            Point::try_new([1e100, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1e100]).expect("finite point coordinates"),
            Point::try_new([1e100, 1e100]).expect("finite point coordinates"),
        ];
        let large_test = Point::try_new([5e99, 5e99]).expect("finite point coordinates");

        let all_finite_insphere_2d = with_la_stack_matrix!(4, |matrix| {
            for (i, point) in large_points.iter().enumerate() {
                let coords = point.coords();
                for (j, &v) in coords.iter().enumerate() {
                    matrix_set(&mut matrix, i, j, v).unwrap();
                }
                matrix_set(&mut matrix, i, 2, squared_norm(coords)).unwrap();
                matrix_set(&mut matrix, i, 3, 1.0).unwrap();
            }

            let test_coords = large_test.coords();
            for (j, &v) in test_coords.iter().enumerate() {
                matrix_set(&mut matrix, 3, j, v).unwrap();
            }
            matrix_set(&mut matrix, 3, 2, squared_norm(test_coords)).unwrap();
            matrix_set(&mut matrix, 3, 3, 1.0).unwrap();

            matrix_block_is_finite(&matrix, 4)
        });
        assert!(all_finite_insphere_2d);
    }

    #[test]
    fn test_sos_fallback_insphere_via_6d() {
        // D=6 → insphere matrix is 8×8, exceeding MAX_STACK_MATRIX_DIM=7.
        // relative_exact_insphere returns Err on every call, so
        // robust_insphere falls through to the SoS fallback (Strategy 3).
        // SoS cofactor minors are 6×6 (within the 7-dim limit), so this
        // succeeds where the full matrix dispatch does not.
        let simplex: Vec<Point<6>> = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        // Exactly cospherical point: (1,1,0,…,0) lies on the circumsphere
        // of the standard 6-simplex (circumcenter = (1/2,…,1/2),
        // circumradius² = 3/2, |(1,1,0,…,0) - c|² = 3/2).
        // insphere_distance returns BOUNDARY, forcing the SoS path.
        let cospherical =
            Point::try_new([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates");
        let result = robust_insphere(&simplex, &cospherical).unwrap();
        assert!(
            result == InSphere::INSIDE || result == InSphere::OUTSIDE,
            "SoS fallback must resolve BOUNDARY to INSIDE or OUTSIDE, got {result:?}"
        );
    }
}
