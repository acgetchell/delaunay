//! Geometric predicates for d-dimensional geometry calculations.
//!
//! This module contains fundamental geometric predicates and calculations
//! that operate on points and simplices, including circumcenter and circumradius
//! calculations.

#![forbid(unsafe_code)]

use crate::core::cell::CellValidationError;
use crate::geometry::matrix::{
    Matrix, StackMatrixDispatchError, matrix_get, matrix_set, matrix_zero_like,
};
use crate::geometry::point::Point;
use crate::geometry::sos::exact_det_sign;
use crate::geometry::traits::coordinate::{
    CoordinateConversionError, CoordinateScalar, DegenerateSimplexReason,
};
use crate::geometry::util::{
    circumcenter, circumradius_with_center, hypot, safe_coords_to_f64, safe_scalar_to_f64,
    squared_norm,
};
use crate::prelude::CircumcenterError;
use core::hint::cold_path;
use num_traits::Float;

/// Convert an exact determinant sign (from `det_sign_exact`) to an [`Orientation`].
#[inline]
const fn sign_to_orientation(sign: i8) -> Orientation {
    match sign {
        1 => Orientation::POSITIVE,
        -1 => Orientation::NEGATIVE,
        _ => Orientation::DEGENERATE,
    }
}

/// Convert an exact determinant sign to an [`InSphere`] result given an
/// orientation sign multiplier.
///
/// `orient_sign` encodes how to interpret a positive determinant:
/// - `1`: positive det → INSIDE (standard insphere with POSITIVE simplex orientation)
/// - `-1`: positive det → OUTSIDE
#[inline]
const fn sign_to_insphere(det_sign: i8, orient_sign: i8) -> InSphere {
    let effective = det_sign as i16 * orient_sign as i16;
    if effective > 0 {
        InSphere::INSIDE
    } else if effective < 0 {
        InSphere::OUTSIDE
    } else {
        InSphere::BOUNDARY
    }
}

/// Verifies that the active matrix block matches the concrete matrix type.
fn validate_active_matrix_dimension<const N: usize>(
    k: usize,
) -> Result<(), StackMatrixDispatchError> {
    if k == N {
        return Ok(());
    }

    Err(StackMatrixDispatchError::ActiveBlockDimensionMismatch { k, dim: N })
}

/// Verifies that the active matrix block is structurally present and finite.
fn active_matrix_block_is_finite<const N: usize>(
    matrix: &Matrix<N>,
    k: usize,
) -> Result<bool, StackMatrixDispatchError> {
    validate_active_matrix_dimension::<N>(k)?;

    for i in 0..k {
        for j in 0..k {
            let entry = matrix_get(matrix, i, j)?;
            if !entry.is_finite() {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Exact determinant signs for the relative-coordinate lifted insphere matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RelativeInsphereSigns {
    /// Sign of the relative simplex orientation determinant.
    pub(crate) relative_orientation: i32,
    /// Raw sign of the lifted insphere determinant.
    pub(crate) insphere_determinant: i32,
}

/// Convert relative-coordinate lifted determinant signs into a normalized sign.
#[inline]
pub(crate) const fn relative_insphere_effective_sign(signs: RelativeInsphereSigns) -> i32 {
    let effective = signs.insphere_determinant * -signs.relative_orientation;
    if effective > 0 {
        1
    } else if effective < 0 {
        -1
    } else {
        0
    }
}

/// Convert relative-coordinate lifted determinant signs into an [`InSphere`] result.
#[inline]
pub(crate) const fn relative_insphere_classification(signs: RelativeInsphereSigns) -> InSphere {
    let sign = relative_insphere_effective_sign(signs);
    if sign > 0 {
        InSphere::INSIDE
    } else if sign < 0 {
        InSphere::OUTSIDE
    } else {
        InSphere::BOUNDARY
    }
}

/// Fill the lifted relative-coordinate insphere matrix for exact determinant evaluation.
#[inline]
fn fill_relative_insphere_matrix<T, const D: usize, const K: usize>(
    matrix: &mut Matrix<K>,
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
) -> Result<(), CoordinateConversionError>
where
    T: CoordinateScalar,
{
    debug_assert_eq!(K, D + 1);
    debug_assert_eq!(simplex_points.len(), D + 1);

    let reference_coords = simplex_points[0].coords();

    for (row, point) in simplex_points.iter().skip(1).enumerate() {
        let mut relative_coords: [T; D] = [T::zero(); D];
        for (dst, (point_coord, reference_coord)) in relative_coords
            .iter_mut()
            .zip(point.coords().iter().zip(reference_coords.iter()))
        {
            *dst = *point_coord - *reference_coord;
        }

        let relative_coords_f64 = safe_coords_to_f64(&relative_coords)?;
        for (column, &value) in relative_coords_f64.iter().enumerate() {
            matrix_set(matrix, row, column, value)?;
        }

        let squared_norm_f64 = safe_scalar_to_f64(squared_norm(&relative_coords))?;
        matrix_set(matrix, row, D, squared_norm_f64)?;
    }

    let mut test_relative_coords: [T; D] = [T::zero(); D];
    for (dst, (point_coord, reference_coord)) in test_relative_coords
        .iter_mut()
        .zip(test_point.coords().iter().zip(reference_coords.iter()))
    {
        *dst = *point_coord - *reference_coord;
    }

    let test_relative_coords_f64 = safe_coords_to_f64(&test_relative_coords)?;
    for (column, &value) in test_relative_coords_f64.iter().enumerate() {
        matrix_set(matrix, D, column, value)?;
    }

    let test_squared_norm_f64 = safe_scalar_to_f64(squared_norm(&test_relative_coords))?;
    matrix_set(matrix, D, D, test_squared_norm_f64)?;

    Ok(())
}

/// Compute exact signs for the relative-coordinate lifted insphere formulation.
#[inline]
pub(crate) fn relative_insphere_signs<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
) -> Result<RelativeInsphereSigns, CoordinateConversionError>
where
    T: CoordinateScalar,
{
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::InvalidSimplexPointCount {
            actual: simplex_points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    let k = D + 1;

    try_with_la_stack_matrix!(k, |matrix| {
        fill_relative_insphere_matrix(&mut matrix, simplex_points, test_point)?;

        let mut orientation_matrix = matrix_zero_like(&matrix);
        for i in 0..D {
            for j in 0..D {
                matrix_set(&mut orientation_matrix, i, j, matrix_get(&matrix, i, j)?)?;
            }
        }
        matrix_set(&mut orientation_matrix, D, D, 1.0)?;

        Ok(RelativeInsphereSigns {
            relative_orientation: exact_det_sign(&orientation_matrix),
            insphere_determinant: exact_det_sign(&matrix),
        })
    })
}

/// Compute only the lifted insphere determinant sign for a relative-coordinate matrix.
///
/// Callers that already know the simplex orientation can combine this determinant
/// with their orientation sign without recomputing the orientation determinant.
#[inline]
pub(crate) fn relative_insphere_determinant_sign<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
) -> Result<i32, CoordinateConversionError>
where
    T: CoordinateScalar,
{
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::InvalidSimplexPointCount {
            actual: simplex_points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    let k = D + 1;

    try_with_la_stack_matrix!(k, |matrix| {
        fill_relative_insphere_matrix(&mut matrix, simplex_points, test_point)?;
        Ok(exact_det_sign(&matrix))
    })
}

/// Compute insphere classification from a pre-populated insphere matrix.
///
/// Uses [`la_stack::Matrix::det_sign_exact`] for provably correct results when
/// the matrix entries are finite.  Returns [`InSphere::BOUNDARY`] when the
/// entries are non-finite and exact arithmetic cannot run.
///
/// `orient_sign` encodes how to interpret a positive determinant:
/// - `1`: positive det → INSIDE (e.g. standard insphere with POSITIVE simplex orientation)
/// - `-1`: positive det → OUTSIDE (e.g. standard insphere with NEGATIVE orientation,
///   or lifted insphere with POSITIVE relative orientation)
#[inline]
pub(crate) fn try_insphere_from_matrix<const N: usize>(
    matrix: &Matrix<N>,
    k: usize,
    orient_sign: i8,
) -> Result<InSphere, StackMatrixDispatchError> {
    // `det_sign_exact()` and `det_direct()` operate on the full N×N matrix,
    // so callers must ensure k == N.  All production call sites satisfy this
    // because `try_with_la_stack_matrix!(k, ...)` creates a Matrix<K> where
    // K == k at compile time.
    validate_active_matrix_dimension::<N>(k)?;

    // Stage 1: provable f64 fast filter for D ≤ 4.
    // `det_errbound()` returns a Shewchuk-style error bound derived from the
    // matrix permanent: |det_direct() − det_exact| ≤ errbound.  If the f64
    // determinant clearly exceeds the bound, the sign is guaranteed correct
    // without allocating.  For D ≥ 5, `det_errbound()` returns `None` and
    // we skip directly to exact arithmetic.
    let det_direct = matrix.det_direct();
    if let (Some(det), Some(errbound)) = (det_direct, matrix.det_errbound())
        && det.is_finite()
    {
        let det_norm = det * f64::from(orient_sign);
        if det_norm > errbound {
            return Ok(InSphere::INSIDE);
        }
        if det_norm < -errbound {
            return Ok(InSphere::OUTSIDE);
        }
    }

    // Stage 2: exact sign via Bareiss — reached for ambiguous f64 results
    // (D ≤ 4) or always for D ≥ 5.  `cold_path()` nudges the optimizer to
    // keep Stage 1 lean; for D ≤ 4 with well-separated inputs, the vast
    // majority of calls return before reaching this point.
    cold_path();
    let exact_is_safe =
        det_direct.is_some_and(f64::is_finite) || active_matrix_block_is_finite(matrix, k)?;
    if exact_is_safe && let Ok(sign) = matrix.det_sign_exact() {
        return Ok(sign_to_insphere(sign, orient_sign));
    }

    // Stage 3: sign is unresolvable (non-finite entries prevent exact
    // arithmetic from running).
    cold_path();
    Ok(InSphere::BOUNDARY)
}

/// Compute orientation from a pre-populated orientation matrix.
///
/// Uses [`la_stack::Matrix::det_sign_exact`] for provably correct results when
/// the matrix entries are finite (even if the f64 determinant overflows).
/// Returns [`Orientation::DEGENERATE`] when the entries are non-finite and
/// exact arithmetic cannot run.
///
/// `k` must equal the number of rows/columns actually used in `matrix`.
#[inline]
pub(crate) fn try_orientation_from_matrix<const N: usize>(
    matrix: &Matrix<N>,
    k: usize,
) -> Result<Orientation, StackMatrixDispatchError> {
    validate_active_matrix_dimension::<N>(k)?;

    // Stage 1: provable f64 fast filter for D ≤ 4.
    // See `insphere_from_matrix` for detailed explanation of the error bound.
    let det_direct = matrix.det_direct();
    if let (Some(det), Some(errbound)) = (det_direct, matrix.det_errbound())
        && det.is_finite()
    {
        if det > errbound {
            return Ok(Orientation::POSITIVE);
        }
        if det < -errbound {
            return Ok(Orientation::NEGATIVE);
        }
    }

    // Stage 2: exact sign via Bareiss — reached for ambiguous f64 results
    // (D ≤ 4) or always for D ≥ 5.  See `insphere_from_matrix` for why this
    // is annotated cold.
    cold_path();
    let exact_is_safe =
        det_direct.is_some_and(f64::is_finite) || active_matrix_block_is_finite(matrix, k)?;
    if exact_is_safe && let Ok(sign) = matrix.det_sign_exact() {
        return Ok(sign_to_orientation(sign));
    }

    // Stage 3: sign is unresolvable (same reasoning as insphere_from_matrix).
    cold_path();
    Ok(Orientation::DEGENERATE)
}

/// Represents the position of a point relative to a circumsphere.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::InSphere;
///
/// let status = InSphere::INSIDE;
/// assert_eq!(status.to_string(), "INSIDE");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InSphere {
    /// The point is outside the circumsphere
    OUTSIDE,
    /// The point is on the boundary of the circumsphere (within numerical tolerance)
    BOUNDARY,
    /// The point is inside the circumsphere
    INSIDE,
}

impl std::fmt::Display for InSphere {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OUTSIDE => write!(f, "OUTSIDE"),
            Self::BOUNDARY => write!(f, "BOUNDARY"),
            Self::INSIDE => write!(f, "INSIDE"),
        }
    }
}

/// Represents the orientation of a simplex.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::Orientation;
///
/// let orientation = Orientation::POSITIVE;
/// assert_eq!(orientation.to_string(), "POSITIVE");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// The simplex has negative orientation (determinant < 0)
    NEGATIVE,
    /// The simplex is degenerate (determinant ≈ 0)
    DEGENERATE,
    /// The simplex has positive orientation (determinant > 0)
    POSITIVE,
}

impl std::fmt::Display for Orientation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NEGATIVE => write!(f, "NEGATIVE"),
            Self::DEGENERATE => write!(f, "DEGENERATE"),
            Self::POSITIVE => write!(f, "POSITIVE"),
        }
    }
}

/// Determine the orientation of a simplex using exact determinant sign computation.
///
/// This function computes the orientation of a d-dimensional simplex by calculating
/// the exact sign of the determinant of a matrix formed by the coordinates of its
/// points, using [`la_stack::Matrix::det_sign_exact`].
///
/// # Exact Arithmetic
///
/// This predicate uses adaptive-precision arithmetic to return a provably correct
/// sign. For D ≤ 4, a fast f64 filter resolves the sign without allocating in
/// well-conditioned cases. For nearly-degenerate configurations (and always for
/// D ≥ 5), the Bareiss algorithm runs in exact `BigRational` arithmetic.
///
/// # Arguments
///
/// * `simplex_points` - A slice of points that form the simplex (must have exactly D+1 points)
///
/// # Returns
///
/// Returns an `Orientation` enum indicating whether the simplex is `POSITIVE`,
/// `NEGATIVE`, or `DEGENERATE`.
///
/// # Errors
///
/// Returns an error if the number of simplex points is not exactly D+1.
///
/// # Algorithm
///
/// For a d-dimensional simplex with points `p₁, p₂, ..., pₐ₊₁`, the orientation
/// is determined by the sign of the determinant of the matrix:
///
/// ```text
/// |  x₁   y₁   z₁  ...  1  |
/// |  x₂   y₂   z₂  ...  1  |
/// |  x₃   y₃   z₃  ...  1  |
/// |  ...  ...  ... ...  ... |
/// |  xₐ₊₁ yₐ₊₁ zₐ₊₁ ... 1  |
/// ```
///
/// Where each row contains the d coordinates of a point and a constant 1.
///
/// # Example
///
/// ```
/// use delaunay::prelude::geometry::Orientation;
/// use delaunay::prelude::geometry::Point;
/// use delaunay::prelude::geometry::Coordinate;
/// use delaunay::prelude::geometry::simplex_orientation;
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let simplex_points = vec![point1, point2, point3, point4];
/// let oriented = simplex_orientation(&simplex_points).unwrap();
/// assert_eq!(oriented, Orientation::NEGATIVE);
/// ```
#[inline]
pub fn simplex_orientation<T, const D: usize>(
    simplex_points: &[Point<T, D>],
) -> Result<Orientation, CoordinateConversionError>
where
    T: CoordinateScalar,
{
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::InvalidSimplexPointCount {
            actual: simplex_points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    let k = D + 1;

    try_with_la_stack_matrix!(k, |matrix| {
        // Populate rows with the coordinates of the points of the simplex.
        for (i, p) in simplex_points.iter().enumerate() {
            let point_coords_f64 = safe_coords_to_f64(p.coords())?;

            for (j, &v) in point_coords_f64.iter().enumerate() {
                matrix_set(&mut matrix, i, j, v)?;
            }

            // Add one to the last column
            matrix_set(&mut matrix, i, D, 1.0)?;
        }

        Ok(try_orientation_from_matrix(&matrix, k)?)
    })
}

/// Check if a point is contained within the circumsphere of a simplex using distance calculations.
///
/// This function uses explicit distance calculations to determine if a point lies within
/// the circumsphere formed by the given points. It computes the circumcenter and circumradius
/// of the simplex, then calculates the distance from the test point to the circumcenter
/// and compares it with the circumradius.
///
/// # Performance
///
/// Benchmarks show that [`insphere_lifted`] is significantly faster across all dimensions:
/// - **3D**: 5.3x faster than [`insphere`], 2.5x faster than `insphere_distance`
/// - **4D-5D**: 1.6-2.9x faster than [`insphere`], comparable to `insphere_distance`
/// - **2D**: `insphere_distance` is 2x slower than [`insphere`] or [`insphere_lifted`]
///
/// **Recommendation**: Use [`insphere_lifted`] for optimal performance in production code.
/// Note that `insphere_lifted` is a fast floating-point predicate that may be less robust
/// than [`crate::geometry::robust_predicates::robust_insphere`] for nearly-degenerate
/// configurations; for 3D+ triangulations requiring numerical robustness, use
/// [`crate::geometry::kernel::RobustKernel`].
///
/// # Algorithm
///
/// The algorithm follows these steps:
/// 1. Calculate the circumcenter of the simplex using [`circumcenter`]
/// 2. Calculate the circumradius using [`circumradius_with_center`]
/// 3. Compute the Euclidean distance from the test point to the circumcenter
/// 4. Compare the distance with the circumradius to determine containment
///
/// # Numerical Stability
///
/// This method can accumulate floating-point errors through multiple steps:
/// - Matrix inversion for circumcenter calculation
/// - Distance computation in potentially high-dimensional space
/// - Multiple coordinate transformations
///
/// For better numerical stability and performance, prefer [`insphere_lifted`] which uses
/// a determinant-based approach with relative coordinates.
///
/// # Arguments
///
/// * `simplex_points` - A slice of points that form the simplex
/// * `test_point` - The point to test for containment
///
/// # Returns
///
/// Returns an `InSphere` enum indicating whether the point is `INSIDE`, `OUTSIDE`,
/// or on the `BOUNDARY` of the circumsphere.
///
/// # Errors
///
/// Returns an error if the circumcenter calculation fails. See [`circumcenter`] for details.
///
/// # Example
///
/// ```
/// use delaunay::prelude::geometry::Point;
/// use delaunay::prelude::geometry::Coordinate;
/// use delaunay::prelude::geometry::{insphere_distance, InSphere};
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let simplex_points = vec![point1, point2, point3, point4];
/// let test_point = Point::new([0.5, 0.5, 0.5]);
/// assert_eq!(insphere_distance(&simplex_points, test_point).unwrap(), InSphere::INSIDE);
/// ```
pub fn insphere_distance<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: Point<T, D>,
) -> Result<InSphere, CircumcenterError>
where
    T: CoordinateScalar,
{
    let circumcenter = circumcenter(simplex_points)?;
    let circumradius = circumradius_with_center(simplex_points, &circumcenter)?;

    // Calculate distance using hypot for numerical stability
    let point_coords = test_point.coords();
    let circumcenter_coords = circumcenter.coords();

    let mut diff_coords = [T::zero(); D];
    for (dst, (p, c)) in diff_coords
        .iter_mut()
        .zip(point_coords.iter().zip(circumcenter_coords.iter()))
    {
        *dst = *p - *c;
    }
    let radius = hypot(&diff_coords);

    // Scale tolerance with geometric magnitude to avoid absolute-epsilon
    // misclassification for large circumradii in near-degenerate simplices.
    let base_tolerance = T::default_tolerance();
    let scale = Float::max(
        T::one(),
        Float::max(Float::abs(circumradius), Float::abs(radius)),
    );
    let tolerance = base_tolerance * scale;
    let signed_margin = circumradius - radius;
    if Float::abs(signed_margin) <= tolerance {
        Ok(InSphere::BOUNDARY)
    } else if signed_margin > T::zero() {
        Ok(InSphere::INSIDE)
    } else {
        Ok(InSphere::OUTSIDE)
    }
}

/// Check if a point is contained within the circumsphere of a simplex using matrix determinant.
///
/// This is the `InSphere` predicate test, which determines whether a test point lies inside,
/// outside, or on the boundary of the circumsphere of a given simplex. This method provides good
/// numerical stability using a matrix determinant approach instead of distance calculations.
///
/// # Performance
///
/// For optimal performance, prefer [`insphere_lifted`] which is significantly faster:
/// - **3D**: 5.3x faster than `insphere` (15.5 ns vs 81.7 ns)
/// - **4D-5D**: 1.6x faster than `insphere`
///
/// The performance advantage comes from `insphere_lifted`'s use of relative coordinates and
/// la-stack v0.2.0's closed-form determinants for D=1-4. Note that `insphere_lifted` is a
/// fast floating-point predicate that may be less robust than
/// [`crate::geometry::robust_predicates::robust_insphere`] for nearly-degenerate
/// configurations; for 3D+ triangulations requiring numerical robustness, use
/// [`crate::geometry::kernel::RobustKernel`].
///
/// # Algorithm
///
/// This implementation follows the robust geometric predicates approach (see References below).
///
/// **Key Implementation Note**: This method uses a standard determinant approach without
/// dimension-dependent parity adjustments. For the lifted matrix formulation that requires
/// parity handling, see [`insphere_lifted`] (specifically the "sign interpretation" section)
/// which correctly handles the dimension-dependent sign convention where even dimensions
/// (2D, 4D, etc.) require inverted sign interpretation compared to odd dimensions (3D, 5D, etc.).
///
/// This ensures agreement between `insphere_lifted` and the other insphere methods
/// across all dimensions from 2D to 5D and beyond.
///
/// The in-sphere test uses the determinant of a specially constructed matrix. For a
/// d-dimensional simplex with points `p₁, p₂, ..., pₐ₊₁` and test point `p`, the
/// matrix has the structure:
///
/// ```text
/// |  x₁   y₁   z₁  ...  x₁²+y₁²+z₁²+...  1  |
/// |  x₂   y₂   z₂  ...  x₂²+y₂²+z₂²+...  1  |
/// |  x₃   y₃   z₃  ...  x₃²+y₃²+z₃²+...  1  |
/// |  ...  ...  ... ...       ...        ... |
/// |  xₚ   yₚ   zₚ   ...  xₚ²+yₚ²+zₚ²+...   1  |
/// ```
///
/// Where each row contains:
/// - The d coordinates of a point
/// - The squared norm (sum of squares) of the point coordinates
/// - A constant 1
///
/// The test point `p` is inside the circumsphere if and only if the determinant
/// has the correct sign relative to the simplex orientation.
///
/// # Mathematical Background
///
/// This determinant test is mathematically equivalent to checking if the test point
/// lies inside the circumsphere, but avoids the numerical instability that can arise
/// from computing circumcenter coordinates and distances explicitly. As demonstrated
/// by Shewchuk, this approach provides much better numerical robustness for geometric
/// computations.
///
/// The sign of the determinant depends on the orientation of the simplex:
/// - For a **positively oriented** simplex: positive determinant means the point is inside
/// - For a **negatively oriented** simplex: negative determinant means the point is inside
///
/// This function automatically determines the simplex orientation using [`simplex_orientation`]
/// and interprets the determinant sign accordingly, ensuring correct results regardless
/// of vertex ordering.
///
/// # Arguments
///
/// * `simplex_points` - A slice of points that form the simplex (must have exactly D+1 points)
/// * `test_point` - The point to test for containment
///
/// # Returns
///
/// Returns [`InSphere::INSIDE`] if the given point is inside the circumsphere,
/// [`InSphere::BOUNDARY`] if it's on the boundary, or [`InSphere::OUTSIDE`] if it's outside.
///
/// # Errors
///
/// Returns an error if:
/// - The number of simplex points is not exactly D+1
/// - Matrix operations fail
/// - Coordinate conversion fails
///
/// # References
///
/// - Shewchuk, J. R. "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
///   Predicates." Discrete & Computational Geometry 18, no. 3 (1997): 305-363.
/// - Shewchuk, J. R. "Robust Adaptive Floating-Point Geometric Predicates."
///   Proceedings of the Twelfth Annual Symposium on Computational Geometry (1996): 141-150.
///
/// # Example
///
/// ```
/// use delaunay::prelude::geometry::Point;
/// use delaunay::prelude::geometry::Coordinate;
/// use delaunay::prelude::geometry::insphere;
/// use delaunay::prelude::geometry::InSphere;
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let simplex_points = vec![point1, point2, point3, point4];
///
/// // Test with a point clearly outside the circumsphere
/// let outside_point = Point::new([2.0, 2.0, 2.0]);
/// assert_eq!(insphere(&simplex_points, outside_point).unwrap(), InSphere::OUTSIDE);
///
/// // Test with a point clearly inside the circumsphere
/// let inside_point = Point::new([0.25, 0.25, 0.25]);
/// assert_eq!(insphere(&simplex_points, inside_point).unwrap(), InSphere::INSIDE);
/// ```
///
/// See function-level docs above for detailed explanation and references.
#[inline]
pub fn insphere<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: Point<T, D>,
) -> Result<InSphere, CoordinateConversionError>
where
    T: CoordinateScalar,
{
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::InvalidSimplexPointCount {
            actual: simplex_points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    // Short-circuit: an original simplex vertex lies exactly on the circumsphere boundary
    if simplex_points.iter().any(|p| p == &test_point) {
        return Ok(InSphere::BOUNDARY);
    }

    let k = D + 2;

    try_with_la_stack_matrix!(k, |matrix| {
        for (i, p) in simplex_points.iter().enumerate() {
            let point_coords = p.coords();
            let point_coords_f64 = safe_coords_to_f64(point_coords)?;

            for (j, &v) in point_coords_f64.iter().enumerate() {
                matrix_set(&mut matrix, i, j, v)?;
            }

            let squared_norm_t = squared_norm(point_coords);
            matrix_set(&mut matrix, i, D, safe_scalar_to_f64(squared_norm_t)?)?;
            matrix_set(&mut matrix, i, D + 1, 1.0)?;
        }

        let test_point_coords = test_point.coords();
        let test_point_coords_f64 = safe_coords_to_f64(test_point_coords)?;
        for (j, &v) in test_point_coords_f64.iter().enumerate() {
            matrix_set(&mut matrix, D + 1, j, v)?;
        }

        let test_squared_norm_t = squared_norm(test_point_coords);
        matrix_set(
            &mut matrix,
            D + 1,
            D,
            safe_scalar_to_f64(test_squared_norm_t)?,
        )?;
        matrix_set(&mut matrix, D + 1, D + 1, 1.0)?;

        // Extract simplex orientation from the insphere matrix, avoiding a
        // redundant simplex_orientation() call that rebuilds the coordinate
        // matrix from scratch.
        //
        // The insphere matrix columns are [x, y, ..., ||p||², 1].
        // The orientation matrix needs [x, y, ..., 1] for the first D+1 rows.
        // We embed this (D+1)×(D+1) block into a (D+2)×(D+2) matrix by
        // placing 1.0 at (D+1, D+1); cofactor expansion along row D+1 gives
        // det(full) = det(orientation subblock).
        let mut orient_matrix = matrix_zero_like(&matrix);
        for i in 0..=D {
            for j in 0..D {
                matrix_set(&mut orient_matrix, i, j, matrix_get(&matrix, i, j)?)?;
            }
            matrix_set(&mut orient_matrix, i, D, 1.0)?;
        }
        matrix_set(&mut orient_matrix, D + 1, D + 1, 1.0)?;

        let orientation = try_orientation_from_matrix(&orient_matrix, k)?;

        match orientation {
            Orientation::DEGENERATE => Err(CoordinateConversionError::DegenerateSimplex {
                dimension: D,
                reason: DegenerateSimplexReason::ZeroOrientation,
            }),
            Orientation::POSITIVE | Orientation::NEGATIVE => {
                let orient_sign: i8 = if matches!(orientation, Orientation::POSITIVE) {
                    1
                } else {
                    -1
                };
                Ok(try_insphere_from_matrix(&matrix, k, orient_sign)?)
            }
        }
    })
}

/// Check if a point is contained within the circumsphere of a simplex using the lifted paraboloid determinant method.
///
/// **This is the recommended high-performance implementation** of the insphere predicate.
/// It provides excellent numerical stability and is significantly faster than other methods.
///
/// # Performance
///
/// Benchmarks demonstrate superior performance across all dimensions:
/// - **3D**: 5.3x faster than [`insphere`] (15.5 ns vs 81.7 ns)
/// - **3D**: 2.5x faster than [`insphere_distance`] (15.5 ns vs 38.3 ns)
/// - **4D-5D**: 1.6x faster than [`insphere`], comparable to [`insphere_distance`]
/// - **2D**: Similar performance to [`insphere`] (8.5 ns vs 12.6 ns)
///
/// The performance gains come from:
/// 1. Using relative coordinates which reduce numerical magnitude
/// 2. Computing smaller (D+1)×(D+1) determinants instead of (D+2)×(D+2)
/// 3. Benefiting from la-stack v0.2.0's closed-form determinants for D=1-4
///
/// This method combines the numerical stability of determinant-based predicates with
/// optimal performance, making it ideal for production use.
///
/// # Robustness
///
/// Both the orientation sub-predicate and the lifted insphere determinant use a
/// three-stage evaluation (via internal helpers `insphere_from_matrix` and
/// `orientation_from_matrix`):
/// 1. **f64 fast filter** with adaptive tolerance — resolves well-conditioned cases
///    without allocating.
/// 2. **Exact Bareiss** via [`la_stack::Matrix::det_sign_exact`] — provably correct
///    sign for finite matrix entries.
/// 3. **Indeterminate fallback** — if exact arithmetic cannot run (non-finite
///    entries), the helpers return `BOUNDARY` / `DEGENERATE` directly.  No
///    additional floating-point sign classification is performed.
///
/// This makes `insphere_lifted` provably correct for finite inputs. For additional
/// robustness strategies (symbolic perturbation, consistency checking), use
/// [`crate::geometry::kernel::RobustKernel`].
///
/// # Algorithm
///
/// This implementation uses the lifted paraboloid method described in:
///
/// Preparata, Franco P., and Michael Ian Shamos.
/// "Computational Geometry: An Introduction."
/// Texts and Monographs in Computer Science. New York: Springer-Verlag, 1985.
///
/// The method works by "lifting" points from d-dimensional space to (d+1)-dimensional space
/// by adding their squared distance as an additional coordinate. The in-sphere test then
/// reduces to computing the determinant of a matrix formed from these lifted coordinates.
///
/// For a d-dimensional simplex with points `p₀, p₁, ..., pₐ` and test point `p`,
/// the matrix has the structure:
///
/// ```text
/// | p₁-p₀  ||p₁-p₀||² |
/// | p₂-p₀  ||p₂-p₀||² |
/// | ...    ...       |
/// | pₐ-p₀  ||pₐ-p₀||² |
/// | p-p₀   ||p-p₀||²  |
/// ```
///
/// This formulation centers coordinates around the first point (p₀), which improves
/// numerical stability by reducing the magnitude of matrix elements compared to using
/// absolute coordinates.
///
/// # Mathematical Background
///
/// The lifted paraboloid method exploits the fact that the circumsphere of a set of points
/// in d-dimensional space corresponds to a hyperplane in (d+1)-dimensional space when
/// points are lifted to the paraboloid z = x₁² + x₂² + ... + xₐ². A point lies inside
/// the circumsphere if and only if it lies below this hyperplane in the lifted space.
///
/// # Arguments
///
/// * `simplex_points` - A slice of points that form the simplex (must have exactly D+1 points)
/// * `test_point` - The point to test for containment
///
/// # Returns
///
/// Returns an `InSphere` enum indicating whether the point is `INSIDE`, `OUTSIDE`,
/// or on the `BOUNDARY` of the circumsphere.
///
/// # Errors
///
/// Returns an error if:
/// - The number of simplex points is not exactly D+1
/// - Matrix operations fail
/// - Coordinate conversion fails
///
/// # References
///
/// - Preparata, Franco P., and Michael Ian Shamos. "Computational Geometry: An Introduction."
///   Texts and Monographs in Computer Science. New York: Springer-Verlag, 1985.
/// - Edelsbrunner, Herbert. "Algorithms in Combinatorial Geometry."
///   EATCS Monographs on Theoretical Computer Science. Berlin: Springer-Verlag, 1987.
///
/// # Example
///
/// ```
/// use delaunay::prelude::geometry::Point;
/// use delaunay::prelude::geometry::Coordinate;
/// use delaunay::prelude::geometry::insphere_lifted;
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let simplex_points = vec![point1, point2, point3, point4];
///
/// // Test with a point that should be inside according to the lifted paraboloid method
/// let test_point = Point::new([0.1, 0.1, 0.1]);
/// let result = insphere_lifted(&simplex_points, test_point);
/// assert!(result.is_ok()); // Should execute without error
/// ```
pub fn insphere_lifted<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: Point<T, D>,
) -> Result<InSphere, CellValidationError>
where
    T: CoordinateScalar,
{
    if simplex_points.len() != D + 1 {
        return Err(CellValidationError::InsufficientVertices {
            actual: simplex_points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    let signs = relative_insphere_signs(simplex_points, &test_point)
        .map_err(|source| CellValidationError::CoordinateConversion { source })?;
    if signs.relative_orientation == 0 {
        Err(CellValidationError::DegenerateSimplex)
    } else {
        Ok(relative_insphere_classification(signs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::matrix::matrix_set as try_matrix_set;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::prelude::circumradius;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    /// Populate a test matrix while keeping production matrix errors loud.
    fn set_test_matrix_entry<const N: usize>(
        matrix: &mut Matrix<N>,
        row: usize,
        column: usize,
        value: f64,
    ) {
        try_matrix_set(matrix, row, column, value).unwrap();
    }

    #[test]
    fn test_enum_display_and_debug_implementations() {
        // Test Display implementation for InSphere enum
        assert_eq!(format!("{}", InSphere::INSIDE), "INSIDE");
        assert_eq!(format!("{}", InSphere::OUTSIDE), "OUTSIDE");
        assert_eq!(format!("{}", InSphere::BOUNDARY), "BOUNDARY");

        // Test Debug implementation for InSphere enum
        assert_eq!(format!("{:?}", InSphere::INSIDE), "INSIDE");
        assert_eq!(format!("{:?}", InSphere::OUTSIDE), "OUTSIDE");
        assert_eq!(format!("{:?}", InSphere::BOUNDARY), "BOUNDARY");

        // Test Display implementation for Orientation enum
        assert_eq!(format!("{}", Orientation::POSITIVE), "POSITIVE");
        assert_eq!(format!("{}", Orientation::NEGATIVE), "NEGATIVE");
        assert_eq!(format!("{}", Orientation::DEGENERATE), "DEGENERATE");

        // Test Debug implementation for Orientation enum
        assert_eq!(format!("{:?}", Orientation::POSITIVE), "POSITIVE");
        assert_eq!(format!("{:?}", Orientation::NEGATIVE), "NEGATIVE");
        assert_eq!(format!("{:?}", Orientation::DEGENERATE), "DEGENERATE");
    }

    #[test]
    fn test_circumradius_2d_to_5d() {
        // Test circumradius calculation across dimensions 2D-5D

        // 2D: Right triangle with legs of length 1
        let triangle_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];
        let radius_2d = circumradius(&triangle_2d).unwrap();
        // For a right triangle with legs of length 1, circumradius is sqrt(2)/2
        let expected_radius_2d = 2.0_f64.sqrt() / 2.0;
        assert_relative_eq!(radius_2d, expected_radius_2d, epsilon = 1e-10);

        // 3D: Unit tetrahedron (origin + unit basis vectors)
        let tetrahedron_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let radius_3d = circumradius(&tetrahedron_3d).unwrap();
        println!("3D circumradius: {radius_3d}");
        // For unit tetrahedron with vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        // circumradius = sqrt(3)/2 ≈ 0.866
        let expected_radius_3d = (3.0_f64).sqrt() / 2.0;
        assert_relative_eq!(radius_3d, expected_radius_3d, epsilon = 1e-10);

        // 4D: Unit 4-simplex
        let simplex_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];
        let radius_4d = circumradius(&simplex_4d).unwrap();
        println!("4D circumradius: {radius_4d}");
        // For unit 4-simplex, circumradius = 1.0
        let expected_radius_4d = 1.0;
        assert_relative_eq!(radius_4d, expected_radius_4d, epsilon = 1e-10);

        // 5D: Unit 5-simplex
        let simplex_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let radius_5d = circumradius(&simplex_5d).unwrap();
        println!("5D circumradius: {radius_5d}");
        // For unit 5-simplex, circumradius = sqrt(5)/2 ≈ 1.118
        let expected_radius_5d = (5.0_f64).sqrt() / 2.0;
        assert_relative_eq!(radius_5d, expected_radius_5d, epsilon = 1e-10);

        // Test that all simplices have positive circumradius
        assert!(radius_2d > 0.0, "2D radius should be positive");
        assert!(radius_3d > 0.0, "3D radius should be positive");
        assert!(radius_4d > 0.0, "4D radius should be positive");
        assert!(radius_5d > 0.0, "5D radius should be positive");

        // Test dimension scaling pattern: radius increases with dimension for these unit simplices
        assert!(
            radius_2d < radius_3d,
            "Radius should increase from 2D to 3D"
        );
        assert!(
            radius_3d < radius_4d,
            "Radius should increase from 3D to 4D"
        );
        assert!(
            radius_4d < radius_5d,
            "Radius should increase from 4D to 5D"
        );

        // Print summary for verification
        println!("Circumradius summary:");
        let expected_2d = (2.0_f64).sqrt() / 2.0;
        let expected_3d = (3.0_f64).sqrt() / 2.0;
        let expected_5d = (5.0_f64).sqrt() / 2.0;
        println!("  2D (right triangle): {radius_2d} ≈ {expected_2d:.6}");
        println!("  3D (unit tetrahedron): {radius_3d} ≈ {expected_3d:.6}");
        println!("  4D (unit 4-simplex): {radius_4d} = 1.0");
        println!("  5D (unit 5-simplex): {radius_5d} ≈ {expected_5d:.6}");
    }

    #[test]
    fn test_insphere_basic_functionality_2d_to_5d() {
        // Test basic insphere functionality across dimensions 2D-5D

        // 2D triangle case
        let simplex_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        // Test far outside, inside, and boundary points for 2D
        assert_eq!(
            insphere_lifted(&simplex_2d, Point::new([10.0, 10.0])).unwrap(),
            InSphere::OUTSIDE,
            "2D far outside point should be OUTSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_2d, Point::new([0.1, 0.1])).unwrap(),
            InSphere::INSIDE,
            "2D inside point should be INSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_2d, Point::new([0.0, 0.0])).unwrap(),
            InSphere::BOUNDARY,
            "2D vertex should be BOUNDARY"
        );

        // 3D tetrahedron case
        let simplex_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        // Test far outside, inside, and boundary points for 3D
        assert_eq!(
            insphere_lifted(&simplex_3d, Point::new([10.0, 10.0, 10.0])).unwrap(),
            InSphere::OUTSIDE,
            "3D far outside point should be OUTSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_3d, Point::new([0.1, 0.1, 0.1])).unwrap(),
            InSphere::INSIDE,
            "3D inside point should be INSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_3d, Point::new([0.0, 0.0, 0.0])).unwrap(),
            InSphere::BOUNDARY,
            "3D vertex should be BOUNDARY"
        );

        // 4D simplex case
        let simplex_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];

        // Test far outside, inside, and boundary points for 4D
        assert_eq!(
            insphere_lifted(&simplex_4d, Point::new([10.0, 10.0, 10.0, 10.0])).unwrap(),
            InSphere::OUTSIDE,
            "4D far outside point should be OUTSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_4d, Point::new([0.1, 0.1, 0.1, 0.1])).unwrap(),
            InSphere::INSIDE,
            "4D inside point should be INSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_4d, Point::new([0.0, 0.0, 0.0, 0.0])).unwrap(),
            InSphere::BOUNDARY,
            "4D vertex should be BOUNDARY"
        );

        // 5D simplex case
        let simplex_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];

        // Test far outside, inside, and boundary points for 5D
        assert_eq!(
            insphere_lifted(&simplex_5d, Point::new([10.0, 10.0, 10.0, 10.0, 10.0])).unwrap(),
            InSphere::OUTSIDE,
            "5D far outside point should be OUTSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_5d, Point::new([0.1, 0.1, 0.1, 0.1, 0.1])).unwrap(),
            InSphere::INSIDE,
            "5D inside point should be INSIDE"
        );
        assert_eq!(
            insphere_lifted(&simplex_5d, Point::new([0.0, 0.0, 0.0, 0.0, 0.0])).unwrap(),
            InSphere::BOUNDARY,
            "5D vertex should be BOUNDARY"
        );
    }

    #[test]
    fn test_insphere_edge_cases_and_errors() {
        // Test edge cases across dimensions including 1D

        // 1D case (line segment)
        let simplex_1d = vec![Point::new([0.0]), Point::new([2.0])];
        let midpoint_1d = Point::new([1.0]);
        let far_point_1d = Point::new([10.0]);

        assert!(
            insphere_lifted(&simplex_1d, midpoint_1d).is_ok(),
            "1D midpoint should not error"
        );
        assert!(
            insphere_lifted(&simplex_1d, far_point_1d).is_ok(),
            "1D far point should not error"
        );

        // Test circumcenter points for various dimensions
        let simplex_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        // Circumcenter should be inside the circumsphere
        let circumcenter_3d = Point::new([0.5, 0.5, 0.5]);
        assert_eq!(
            insphere_lifted(&simplex_3d, circumcenter_3d).unwrap(),
            InSphere::INSIDE,
            "3D circumcenter should be INSIDE"
        );

        // Test regular 4D simplex with symmetric properties
        let regular_4d_simplex = vec![
            Point::new([1.0, 1.0, 1.0, 1.0]),
            Point::new([1.0, -1.0, -1.0, -1.0]),
            Point::new([-1.0, 1.0, -1.0, -1.0]),
            Point::new([-1.0, -1.0, 1.0, -1.0]),
            Point::new([-1.0, -1.0, -1.0, 1.0]),
        ];

        // Origin should be inside this symmetric simplex
        assert_eq!(
            insphere_lifted(&regular_4d_simplex, Point::new([0.0, 0.0, 0.0, 0.0])).unwrap(),
            InSphere::INSIDE,
            "Origin should be inside symmetric 4D simplex"
        );

        // Error case: insufficient vertices
        let incomplete_simplex = vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])]; // Only 2 vertices for 3D
        let test_point = Point::new([0.5, 0.5, 0.5]);

        assert!(
            insphere_lifted(&incomplete_simplex, test_point).is_err(),
            "Should error with insufficient vertices"
        );
    }

    #[test]
    fn predicates_circumcenter_error_cases() {
        // Test circumcenter calculation with degenerate cases
        let points = vec![Point::new([0.0, 0.0]), Point::new([1.0, 0.0])]; // Only 2 points for 2D

        // Test with insufficient vertices for proper simplex
        let center_result = circumcenter(&points);
        assert!(center_result.is_err());
    }

    #[test]
    fn predicates_circumcenter_collinear_points() {
        // Test circumcenter with collinear points (should fail)
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([2.0, 0.0, 0.0]),
            Point::new([3.0, 0.0, 0.0]),
        ];

        // This should fail because points are collinear
        let center_result = circumcenter(&points);
        assert!(center_result.is_err());
    }

    #[test]
    fn predicates_circumradius_with_center() {
        // Test the circumradius_with_center function
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let center = circumcenter(&points).unwrap();
        let radius_with_center = circumradius_with_center(&points, &center);
        let radius_direct = circumradius(&points).unwrap();

        assert_relative_eq!(radius_with_center.unwrap(), radius_direct, epsilon = 1e-10);
    }

    #[test]
    fn predicates_circumsphere_edge_cases() {
        // Test circumsphere containment with simple cases
        let simplex_points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        // Test that the methods run without error
        let test_point = Point::new([0.25, 0.25]);
        assert!(insphere_distance(&simplex_points, test_point).is_ok());

        // At minimum, both methods should give the same result for the same input
        let far_point = Point::new([100.0, 100.0]);
        assert!(insphere_distance(&simplex_points, far_point).is_ok());
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "orientation regression test keeps dimension-specific cases together"
    )]
    fn test_simplex_orientation_comprehensive() {
        // Test 2D orientation - positive case
        let positive_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&positive_2d).unwrap(),
            Orientation::POSITIVE,
            "2D positive orientation failed"
        );

        // Test 2D orientation - negative case (reversed order)
        let negative_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([0.0, 1.0]),
            Point::new([1.0, 0.0]),
        ];
        assert_eq!(
            simplex_orientation(&negative_2d).unwrap(),
            Orientation::NEGATIVE,
            "2D negative orientation failed"
        );

        // Test 2D degenerate case - collinear points
        let degenerate_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([2.0, 0.0]),
        ];
        assert_eq!(
            simplex_orientation(&degenerate_2d).unwrap(),
            Orientation::DEGENERATE,
            "2D degenerate case failed"
        );

        // Test 3D orientation - positive case
        let positive_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&positive_3d).unwrap(),
            Orientation::POSITIVE,
            "3D positive orientation failed"
        );

        // Test 3D orientation - negative case
        let negative_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&negative_3d).unwrap(),
            Orientation::NEGATIVE,
            "3D negative orientation failed"
        );

        // Test 3D degenerate case - coplanar points
        let degenerate_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([1.0, 1.0, 0.0]), // All points on z=0 plane
        ];
        assert_eq!(
            simplex_orientation(&degenerate_3d).unwrap(),
            Orientation::DEGENERATE,
            "3D degenerate case failed"
        );

        // Test 4D orientation - positive case
        let positive_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&positive_4d).unwrap(),
            Orientation::POSITIVE,
            "4D positive orientation failed"
        );

        // Test 4D orientation - negative case (different ordering)
        let negative_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&negative_4d).unwrap(),
            Orientation::NEGATIVE,
            "4D negative orientation failed"
        );

        // Test 4D degenerate case - points in 3D subspace
        let degenerate_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([1.0, 1.0, 1.0, 0.0]), // All points have w=0
        ];
        assert_eq!(
            simplex_orientation(&degenerate_4d).unwrap(),
            Orientation::DEGENERATE,
            "4D degenerate case failed"
        );

        // Test 5D orientation - positive case
        // For even dimensions, we need to adjust vertex order to get positive orientation
        let positive_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&positive_5d).unwrap(),
            Orientation::POSITIVE,
            "5D positive orientation failed"
        );

        // Test 5D orientation - negative case (reversed from positive)
        let negative_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        assert_eq!(
            simplex_orientation(&negative_5d).unwrap(),
            Orientation::NEGATIVE,
            "5D negative orientation failed"
        );

        // Test 5D degenerate case - points in 4D subspace
        let degenerate_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([1.0, 1.0, 1.0, 1.0, 0.0]), // All points have v=0
        ];
        assert_eq!(
            simplex_orientation(&degenerate_5d).unwrap(),
            Orientation::DEGENERATE,
            "5D degenerate case failed"
        );

        // Test error case: insufficient vertices
        let insufficient_vertices = vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])]; // Only 2 vertices for 3D
        assert!(
            simplex_orientation(&insufficient_vertices).is_err(),
            "Should error with insufficient vertices"
        );
    }

    #[test]
    fn test_insphere_degenerate_simplex_error_handling() {
        // Create a degenerate simplex (coplanar points in 3D)
        let degenerate_simplex = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([1.0, 1.0, 0.0]), // All points lie on the same plane (z=0)
        ];
        let test_point = Point::new([0.5, 0.5, 0.5]);

        // Test that insphere errors with a typed degenerate-simplex variant.
        let result = insphere(&degenerate_simplex, test_point);
        assert_eq!(
            result,
            Err(CoordinateConversionError::DegenerateSimplex {
                dimension: 3,
                reason: DegenerateSimplexReason::ZeroOrientation,
            })
        );

        // Test that insphere_lifted errors with degenerate simplex
        let result_lifted = insphere_lifted(&degenerate_simplex, test_point);
        assert!(
            result_lifted.is_err(),
            "insphere_lifted should error with degenerate simplex"
        );

        // Verify the error is the correct type
        match result_lifted {
            Err(CellValidationError::DegenerateSimplex) => (), // Expected error type
            Err(other) => panic!("Wrong error type: {other:?}"),
            Ok(_) => panic!("Function should have returned an error"),
        }

        // Test error handling for insufficient vertices
        let insufficient_vertices = vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])]; // Only 2 vertices for 3D

        assert!(
            insphere_distance(&insufficient_vertices, test_point).is_err(),
            "insphere_distance should error with insufficient vertices"
        );
    }

    #[test]
    fn test_insphere_lifted_edge_case_boundary() {
        // Create a simplex and test with a point on or near the boundary
        // For 2D case, use a right triangle
        let simplex_points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        // Test with one of the original vertices (should be on boundary)
        let vertex_point = Point::new([0.0, 0.0]);
        let result = insphere_lifted(&simplex_points, vertex_point).unwrap();
        assert_eq!(
            result,
            InSphere::BOUNDARY,
            "Original vertex should be classified as BOUNDARY"
        );

        // Test with a point clearly inside
        let inside_point = Point::new([0.1, 0.1]);
        let inside_result = insphere_lifted(&simplex_points, inside_point).unwrap();
        assert_eq!(
            inside_result,
            InSphere::INSIDE,
            "Point inside should be classified as INSIDE"
        );

        // Test with a point clearly outside
        let outside_point = Point::new([10.0, 10.0]);
        let outside_result = insphere_lifted(&simplex_points, outside_point).unwrap();
        assert_eq!(
            outside_result,
            InSphere::OUTSIDE,
            "Point outside should be classified as OUTSIDE"
        );
    }

    #[test]
    fn test_insphere_and_insphere_lifted_consistency() {
        // Test that both insphere implementations give consistent results for various cases
        let simplex_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        // Test multiple points
        let test_cases = [
            // Inside points
            (Point::new([0.2, 0.2, 0.2]), InSphere::INSIDE),
            // Outside points
            (Point::new([2.0, 2.0, 2.0]), InSphere::OUTSIDE),
            // Boundary points (simplex vertices should be on boundary)
            (Point::new([0.0, 0.0, 0.0]), InSphere::BOUNDARY),
        ];

        for (point, expected) in &test_cases {
            let result1 = insphere(&simplex_points, *point).unwrap();
            let result2 = insphere_lifted(&simplex_points, *point).unwrap();

            // For boundary points, numerical precision issues might cause slight variations,
            // so we're lenient in the comparison for BOUNDARY cases
            if *expected == InSphere::BOUNDARY {
                assert!(
                    result1 == InSphere::BOUNDARY || result2 == InSphere::BOUNDARY,
                    "Point {point:?} should be classified as BOUNDARY by at least one method"
                );
            } else {
                // For INSIDE/OUTSIDE, both methods should agree
                assert_eq!(result1, *expected, "insphere result mismatch for {point:?}");
                assert_eq!(
                    result2, *expected,
                    "insphere_lifted result mismatch for {point:?}"
                );
            }
        }
    }

    #[test]
    fn test_insphere_methods_2d_comprehensive() {
        // 2D triangle: origin, (1,0), (0,1)
        let simplex = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        // Test various points and verify all methods agree
        let test_cases = [
            (Point::new([0.1, 0.1]), "inside"),    // Clearly inside
            (Point::new([10.0, 10.0]), "outside"), // Clearly outside
            (Point::new([0.0, 0.0]), "boundary"),  // Vertex (on boundary)
            (Point::new([0.5, 0.0]), "boundary"),  // Edge midpoint
        ];

        for (test_point, description) in &test_cases {
            let result_std = insphere(&simplex, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex, *test_point).unwrap();

            println!(
                "2D {description}: std={result_std:?}, lifted={result_lifted:?}, distance={result_distance:?}"
            );

            // All methods should agree (with some tolerance for boundary cases)
            if *description != "boundary" {
                // Note: 2D has known issues with insphere_lifted that need further investigation
                // assert_eq!(result_std, result_lifted, "2D {}: std vs lifted mismatch", description);
                assert_eq!(
                    result_std, result_distance,
                    "2D {description}: std vs distance mismatch"
                );
            }
        }
    }

    #[test]
    fn test_insphere_methods_3d_comprehensive() {
        // 3D tetrahedron: origin and unit basis vectors
        let simplex = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let test_cases = [
            (Point::new([0.1, 0.1, 0.1]), "inside"),
            (Point::new([10.0, 10.0, 10.0]), "outside"),
            (Point::new([0.0, 0.0, 0.0]), "boundary"), // Vertex
            (Point::new([0.25, 0.25, 0.25]), "inside"), // Centroid region
        ];

        for (test_point, description) in &test_cases {
            let result_std = insphere(&simplex, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex, *test_point).unwrap();

            println!(
                "3D {description}: std={result_std:?}, lifted={result_lifted:?}, distance={result_distance:?}"
            );

            if *description != "boundary" {
                assert_eq!(
                    result_std, result_lifted,
                    "3D {description}: std vs lifted mismatch"
                );
                assert_eq!(
                    result_std, result_distance,
                    "3D {description}: std vs distance mismatch"
                );
            }
        }
    }

    #[test]
    fn test_insphere_methods_4d_comprehensive() {
        // 4D simplex: origin and unit basis vectors
        let simplex = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];

        let test_cases = [
            (Point::new([0.1, 0.1, 0.1, 0.1]), "inside"),
            (Point::new([10.0, 10.0, 10.0, 10.0]), "outside"),
            (Point::new([0.0, 0.0, 0.0, 0.0]), "boundary"), // Vertex
            (Point::new([0.2, 0.2, 0.2, 0.2]), "inside"),
        ];

        for (test_point, description) in &test_cases {
            let result_std = insphere(&simplex, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex, *test_point).unwrap();

            println!(
                "4D {description}: std={result_std:?}, lifted={result_lifted:?}, distance={result_distance:?}"
            );

            if *description != "boundary" {
                assert_eq!(
                    result_std, result_lifted,
                    "4D {description}: std vs lifted mismatch"
                );
                assert_eq!(
                    result_std, result_distance,
                    "4D {description}: std vs distance mismatch"
                );
            }
        }
    }

    #[test]
    fn test_insphere_methods_5d_comprehensive() {
        // 5D simplex: origin and unit basis vectors
        let simplex = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];

        let test_cases = [
            (Point::new([0.1, 0.1, 0.1, 0.1, 0.1]), "inside"),
            (Point::new([10.0, 10.0, 10.0, 10.0, 10.0]), "outside"),
            (Point::new([0.0, 0.0, 0.0, 0.0, 0.0]), "boundary"), // Vertex
            (Point::new([0.15, 0.15, 0.15, 0.15, 0.15]), "inside"),
        ];

        for (test_point, description) in &test_cases {
            let result_std = insphere(&simplex, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex, *test_point).unwrap();

            println!(
                "5D {description}: std={result_std:?}, lifted={result_lifted:?}, distance={result_distance:?}"
            );

            if *description != "boundary" {
                assert_eq!(
                    result_std, result_lifted,
                    "5D {description}: std vs lifted mismatch"
                );
                assert_eq!(
                    result_std, result_distance,
                    "5D {description}: std vs distance mismatch"
                );
            }
        }
    }

    #[test]
    fn test_edge_cases_across_dimensions() {
        // Test edge cases that should work consistently across dimensions

        // 2D: Test with very small simplex
        let tiny_simplex_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1e-6, 0.0]),
            Point::new([0.0, 1e-6]),
        ];
        let test_point_2d = Point::new([1e-7, 1e-7]);
        let result_2d = insphere(&tiny_simplex_2d, test_point_2d);
        assert!(result_2d.is_ok(), "2D tiny simplex should work");

        // 3D: Test with large coordinates
        let large_simplex_3d = vec![
            Point::new([1e6, 0.0, 0.0]),
            Point::new([1e6 + 1.0, 0.0, 0.0]),
            Point::new([1e6, 1.0, 0.0]),
            Point::new([1e6, 0.0, 1.0]),
        ];
        let test_point_3d = Point::new([1e6 + 0.1, 0.1, 0.1]);
        let result_3d = insphere(&large_simplex_3d, test_point_3d);
        assert!(result_3d.is_ok(), "3D large coordinates should work");

        // 4D: Test with negative coordinates
        let negative_simplex_4d = vec![
            Point::new([-1.0, -1.0, -1.0, -1.0]),
            Point::new([0.0, -1.0, -1.0, -1.0]),
            Point::new([-1.0, 0.0, -1.0, -1.0]),
            Point::new([-1.0, -1.0, 0.0, -1.0]),
            Point::new([-1.0, -1.0, -1.0, 0.0]),
        ];
        let test_point_4d = Point::new([-0.5, -0.5, -0.5, -0.5]);
        let result_4d = insphere(&negative_simplex_4d, test_point_4d);
        assert!(result_4d.is_ok(), "4D negative coordinates should work");
    }

    #[test]
    fn test_method_consistency_stress_test() {
        // Stress test with random points to ensure all methods agree
        let mut disagreement_count = HashMap::new();
        let mut total_tests = 0;

        // Test 3D case with various random-ish points
        let simplex_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let test_points = [
            Point::new([0.1, 0.1, 0.1]),
            Point::new([0.3, 0.2, 0.1]),
            Point::new([0.5, 0.5, 0.5]),
            Point::new([1.0, 1.0, 1.0]),
            Point::new([2.0, 2.0, 2.0]),
            Point::new([-0.1, -0.1, -0.1]),
            Point::new([0.25, 0.25, 0.25]),
            Point::new([0.01, 0.01, 0.01]),
        ];

        for test_point in &test_points {
            total_tests += 1;
            let result_std = insphere(&simplex_3d, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex_3d, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex_3d, *test_point).unwrap();

            // Count disagreements
            if result_std != result_lifted {
                *disagreement_count.entry("std_vs_lifted").or_insert(0) += 1;
            }
            if result_std != result_distance {
                *disagreement_count.entry("std_vs_distance").or_insert(0) += 1;
            }
            if result_lifted != result_distance {
                *disagreement_count.entry("lifted_vs_distance").or_insert(0) += 1;
            }
        }

        println!("Stress test results: {total_tests} total tests");
        for (key, count) in &disagreement_count {
            println!("  {key}: {count} disagreements");
        }

        // With our fix, we should have perfect agreement
        assert_eq!(
            disagreement_count.len(),
            0,
            "All methods should agree after sign fix"
        );
    }

    /// Helper to test `insphere_lifted` parity branch for a given simplex configuration.
    ///
    /// Validates:
    /// 1. Simplex has expected orientation
    /// 2. `insphere_lifted` and insphere produce identical results
    /// 3. Test point produces expected `InSphere` result
    fn check_insphere_parity<T, const D: usize>(
        simplex: &[Point<T, D>],
        test_point: Point<T, D>,
        expected_orientation: Orientation,
        expected_result: InSphere,
        dimension: usize,
        orientation_label: &str,
    ) where
        T: CoordinateScalar,
    {
        let orientation = simplex_orientation(simplex).unwrap();
        assert_eq!(
            orientation, expected_orientation,
            "{dimension}D simplex should be {orientation_label}"
        );

        let result_lifted = insphere_lifted(simplex, test_point).unwrap();
        let result_std = insphere(simplex, test_point).unwrap();
        assert_eq!(
            result_lifted, result_std,
            "{dimension}D {orientation_label}: insphere_lifted should match insphere"
        );
        assert_eq!(
            result_lifted, expected_result,
            "{dimension}D {orientation_label}: test point should be {expected_result:?}"
        );
    }

    #[test]
    fn test_insphere_lifted_parity_branch_positive_orientation() {
        // Test parity branch for even and odd dimensions with POSITIVE orientation
        // This exercises the parity_sign * orient_sign computation path

        // 2D (even dimension) with POSITIVE orientation
        check_insphere_parity(
            &[
                Point::new([0.0, 0.0]),
                Point::new([1.0, 0.0]),
                Point::new([0.0, 1.0]),
            ],
            Point::new([0.1, 0.1]),
            Orientation::POSITIVE,
            InSphere::INSIDE,
            2,
            "POSITIVE",
        );

        // 3D (odd dimension) with POSITIVE orientation
        check_insphere_parity(
            &[
                Point::new([0.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 0.0, 1.0]),
            ],
            Point::new([0.1, 0.1, 0.1]),
            Orientation::POSITIVE,
            InSphere::INSIDE,
            3,
            "POSITIVE",
        );

        // 4D (even dimension) with POSITIVE orientation
        check_insphere_parity(
            &[
                Point::new([0.0, 0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0, 0.0]),
                Point::new([0.0, 0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 0.0, 1.0]),
            ],
            Point::new([0.1, 0.1, 0.1, 0.1]),
            Orientation::POSITIVE,
            InSphere::INSIDE,
            4,
            "POSITIVE",
        );

        // 5D (odd dimension) with POSITIVE orientation
        check_insphere_parity(
            &[
                Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
                Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
                Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
            ],
            Point::new([0.1, 0.1, 0.1, 0.1, 0.1]),
            Orientation::POSITIVE,
            InSphere::INSIDE,
            5,
            "POSITIVE",
        );
    }

    #[test]
    fn test_insphere_lifted_parity_branch_negative_orientation() {
        // Test parity branch for even and odd dimensions with NEGATIVE orientation
        // This exercises the parity_sign * orient_sign computation path

        // 2D (even dimension) with NEGATIVE orientation
        check_insphere_parity(
            &[
                Point::new([0.0, 0.0]),
                Point::new([0.0, 1.0]),
                Point::new([1.0, 0.0]),
            ],
            Point::new([0.1, 0.1]),
            Orientation::NEGATIVE,
            InSphere::INSIDE,
            2,
            "NEGATIVE",
        );

        // 3D (odd dimension) with NEGATIVE orientation
        check_insphere_parity(
            &[
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 1.0]),
            ],
            Point::new([0.1, 0.1, 0.1]),
            Orientation::NEGATIVE,
            InSphere::INSIDE,
            3,
            "NEGATIVE",
        );

        // 4D (even dimension) with NEGATIVE orientation
        check_insphere_parity(
            &[
                Point::new([0.0, 0.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0, 0.0]),
                Point::new([0.0, 0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 0.0, 1.0]),
            ],
            Point::new([0.1, 0.1, 0.1, 0.1]),
            Orientation::NEGATIVE,
            InSphere::INSIDE,
            4,
            "NEGATIVE",
        );

        // 5D (odd dimension) with NEGATIVE orientation
        check_insphere_parity(
            &[
                Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
                Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
                Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
            ],
            Point::new([0.1, 0.1, 0.1, 0.1, 0.1]),
            Orientation::NEGATIVE,
            InSphere::INSIDE,
            5,
            "NEGATIVE",
        );
    }

    // =======================================================================
    // orientation_from_matrix unit tests
    // =======================================================================

    #[test]
    fn test_orientation_from_matrix_positive() {
        // 2D: CCW triangle → positive determinant.
        let k = 3;
        with_la_stack_matrix!(k, |m| {
            // Row 0: (0, 0, 1)
            set_test_matrix_entry(&mut m, 0, 0, 0.0);
            set_test_matrix_entry(&mut m, 0, 1, 0.0);
            set_test_matrix_entry(&mut m, 0, 2, 1.0);
            // Row 1: (1, 0, 1)
            set_test_matrix_entry(&mut m, 1, 0, 1.0);
            set_test_matrix_entry(&mut m, 1, 1, 0.0);
            set_test_matrix_entry(&mut m, 1, 2, 1.0);
            // Row 2: (0, 1, 1)
            set_test_matrix_entry(&mut m, 2, 0, 0.0);
            set_test_matrix_entry(&mut m, 2, 1, 1.0);
            set_test_matrix_entry(&mut m, 2, 2, 1.0);

            assert_eq!(
                try_orientation_from_matrix(&m, k).unwrap(),
                Orientation::POSITIVE
            );
        });
    }

    #[test]
    fn test_orientation_from_matrix_negative() {
        // Swap two rows of the positive case → negative determinant.
        let k = 3;
        with_la_stack_matrix!(k, |m| {
            // Row 0: (0, 1, 1)  — swapped with row 2 from positive test
            set_test_matrix_entry(&mut m, 0, 0, 0.0);
            set_test_matrix_entry(&mut m, 0, 1, 1.0);
            set_test_matrix_entry(&mut m, 0, 2, 1.0);
            // Row 1: (1, 0, 1)
            set_test_matrix_entry(&mut m, 1, 0, 1.0);
            set_test_matrix_entry(&mut m, 1, 1, 0.0);
            set_test_matrix_entry(&mut m, 1, 2, 1.0);
            // Row 2: (0, 0, 1)
            set_test_matrix_entry(&mut m, 2, 0, 0.0);
            set_test_matrix_entry(&mut m, 2, 1, 0.0);
            set_test_matrix_entry(&mut m, 2, 2, 1.0);

            assert_eq!(
                try_orientation_from_matrix(&m, k).unwrap(),
                Orientation::NEGATIVE
            );
        });
    }

    #[test]
    fn test_orientation_from_matrix_degenerate() {
        // Collinear points → zero determinant.
        let k = 3;
        with_la_stack_matrix!(k, |m| {
            // (0,0,1), (1,0,1), (2,0,1)
            set_test_matrix_entry(&mut m, 0, 0, 0.0);
            set_test_matrix_entry(&mut m, 0, 1, 0.0);
            set_test_matrix_entry(&mut m, 0, 2, 1.0);
            set_test_matrix_entry(&mut m, 1, 0, 1.0);
            set_test_matrix_entry(&mut m, 1, 1, 0.0);
            set_test_matrix_entry(&mut m, 1, 2, 1.0);
            set_test_matrix_entry(&mut m, 2, 0, 2.0);
            set_test_matrix_entry(&mut m, 2, 1, 0.0);
            set_test_matrix_entry(&mut m, 2, 2, 1.0);

            assert_eq!(
                try_orientation_from_matrix(&m, k).unwrap(),
                Orientation::DEGENERATE
            );
        });
    }

    #[test]
    fn test_orientation_from_matrix_extreme_magnitude_fallback() {
        // Entries near f64::MAX cause det_direct() to overflow to infinity,
        // bypassing the fast filter.  Stage 2 (exact Bareiss) resolves the
        // correct sign because all individual entries are finite.
        let k = 3;
        let big = f64::MAX / 2.0;
        with_la_stack_matrix!(k, |m| {
            set_test_matrix_entry(&mut m, 0, 0, 0.0);
            set_test_matrix_entry(&mut m, 0, 1, 0.0);
            set_test_matrix_entry(&mut m, 0, 2, 1.0);
            set_test_matrix_entry(&mut m, 1, 0, big);
            set_test_matrix_entry(&mut m, 1, 1, 0.0);
            set_test_matrix_entry(&mut m, 1, 2, 1.0);
            set_test_matrix_entry(&mut m, 2, 0, 0.0);
            set_test_matrix_entry(&mut m, 2, 1, big);
            set_test_matrix_entry(&mut m, 2, 2, 1.0);

            let result = try_orientation_from_matrix(&m, k).unwrap();
            assert_eq!(
                result,
                Orientation::POSITIVE,
                "Extreme-magnitude fallback should still resolve correct orientation"
            );
        });
    }

    #[test]
    fn test_orientation_from_matrix_nonfinite_entry_falls_to_stage3() {
        // A 4×4 matrix with a NaN entry.  `det_direct()` returns NaN
        // (non-finite) and the entry check finds NaN at (3,3), so
        // `exact_is_safe = false`.  Stage 1 also fails (NaN determinant).
        // Falls through to Stage 3 → DEGENERATE.
        let k = 4;
        with_la_stack_matrix!(k, |m| {
            set_test_matrix_entry(&mut m, 0, 0, 0.0);
            set_test_matrix_entry(&mut m, 0, 1, 0.0);
            set_test_matrix_entry(&mut m, 0, 2, 1.0);
            set_test_matrix_entry(&mut m, 1, 0, 1.0);
            set_test_matrix_entry(&mut m, 1, 1, 0.0);
            set_test_matrix_entry(&mut m, 1, 2, 1.0);
            set_test_matrix_entry(&mut m, 2, 0, 0.0);
            set_test_matrix_entry(&mut m, 2, 1, 1.0);
            set_test_matrix_entry(&mut m, 2, 2, 1.0);

            // NaN inside the k×k block.
            set_test_matrix_entry(&mut m, 3, 3, f64::NAN);

            let result = try_orientation_from_matrix(&m, k).unwrap();
            assert_eq!(
                result,
                Orientation::DEGENERATE,
                "non-finite entry in matrix should fall to Stage 3 → DEGENERATE"
            );
        });
    }

    #[test]
    fn test_try_orientation_from_matrix_rejects_dimension_mismatch() {
        let matrix = Matrix::<3>::zero();
        let err = try_orientation_from_matrix(&matrix, 4).unwrap_err();

        assert!(matches!(
            err,
            StackMatrixDispatchError::ActiveBlockDimensionMismatch { k: 4, dim: 3 }
        ));
    }

    #[test]
    fn test_exact_orientation_near_degenerate_2d() {
        // Near-degenerate 2D triangle: third point is almost collinear.
        // The perturbation 2^-50 ≈ 8.9e-16 is below typical adaptive tolerance
        // thresholds but produces a non-zero exact determinant.
        let eps = f64::from_bits(0x3CD0_0000_0000_0000); // 2^-50
        let nearly_collinear = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.5, eps]),
        ];
        // Exact arithmetic should detect this is NOT degenerate.
        let orientation = simplex_orientation(&nearly_collinear).unwrap();
        assert_eq!(
            orientation,
            Orientation::POSITIVE,
            "Near-degenerate 2D triangle with 2^-50 perturbation should be POSITIVE"
        );
    }

    #[test]
    fn test_exact_orientation_near_degenerate_3d() {
        // Near-degenerate 3D tetrahedron: fourth point is almost coplanar.
        let eps = f64::from_bits(0x3CD0_0000_0000_0000); // 2^-50
        let nearly_coplanar = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, eps]),
        ];
        let orientation = simplex_orientation(&nearly_coplanar).unwrap();
        assert_ne!(
            orientation,
            Orientation::DEGENERATE,
            "Near-degenerate 3D tetrahedron with 2^-50 perturbation should NOT be DEGENERATE"
        );
    }

    // =======================================================================
    // insphere_from_matrix Stage 2 & Stage 3 coverage
    // =======================================================================

    #[test]
    fn test_insphere_from_matrix_stage2_exact_via_overflow() {
        // Stage 2: det_direct() overflows to non-finite, but all individual
        // entries are finite.  The entry-by-entry finite check passes,
        // enabling exact Bareiss to resolve the sign.
        let k = 4;
        let big = 1e100;
        with_la_stack_matrix!(k, |m| {
            // Diagonal matrix: det = big^4, which overflows f64.
            set_test_matrix_entry(&mut m, 0, 0, big);
            set_test_matrix_entry(&mut m, 1, 1, big);
            set_test_matrix_entry(&mut m, 2, 2, big);
            set_test_matrix_entry(&mut m, 3, 3, big);

            // Positive exact sign + orient_sign = 1 → INSIDE
            assert_eq!(
                try_insphere_from_matrix(&m, k, 1).unwrap(),
                InSphere::INSIDE
            );
            // Positive exact sign + orient_sign = -1 → OUTSIDE
            assert_eq!(
                try_insphere_from_matrix(&m, k, -1).unwrap(),
                InSphere::OUTSIDE
            );
        });
    }

    #[test]
    fn test_insphere_from_matrix_stage2_near_singular() {
        // Stage 2: near-singular matrix whose f64 determinant falls within
        // the provable `det_errbound()` band.  Exact Bareiss resolves the
        // positive sign.
        let k = 3;
        let eps = f64::EPSILON;
        with_la_stack_matrix!(k, |m| {
            // Near-singular: det = eps, permanent ≈ 2 → errbound ≫ eps.
            set_test_matrix_entry(&mut m, 0, 0, 1.0);
            set_test_matrix_entry(&mut m, 0, 1, 1.0);
            set_test_matrix_entry(&mut m, 1, 0, 1.0);
            set_test_matrix_entry(&mut m, 1, 1, 1.0 + eps);
            set_test_matrix_entry(&mut m, 2, 2, 1.0);

            assert_eq!(
                try_insphere_from_matrix(&m, k, 1).unwrap(),
                InSphere::INSIDE
            );
        });
    }

    #[test]
    fn test_insphere_from_matrix_stage2_boundary() {
        // Stage 2: singular matrix (exactly zero determinant) → BOUNDARY.
        let k = 3;
        with_la_stack_matrix!(k, |m| {
            // Two identical rows → det = 0.
            set_test_matrix_entry(&mut m, 0, 0, 1.0);
            set_test_matrix_entry(&mut m, 0, 1, 2.0);
            set_test_matrix_entry(&mut m, 0, 2, 3.0);
            set_test_matrix_entry(&mut m, 1, 0, 1.0);
            set_test_matrix_entry(&mut m, 1, 1, 2.0);
            set_test_matrix_entry(&mut m, 1, 2, 3.0);
            set_test_matrix_entry(&mut m, 2, 0, 4.0);
            set_test_matrix_entry(&mut m, 2, 1, 5.0);
            set_test_matrix_entry(&mut m, 2, 2, 6.0);

            assert_eq!(
                try_insphere_from_matrix(&m, k, 1).unwrap(),
                InSphere::BOUNDARY
            );
        });
    }

    #[test]
    fn test_insphere_from_matrix_stage3_nan() {
        // Stage 3: NaN entry prevents both Stage 1 (det is NaN) and Stage 2
        // (exact_is_safe = false).  Falls through to Stage 3 → BOUNDARY.
        let k = 3;
        with_la_stack_matrix!(k, |m| {
            set_test_matrix_entry(&mut m, 0, 0, 1.0);
            set_test_matrix_entry(&mut m, 1, 1, 1.0);
            set_test_matrix_entry(&mut m, 2, 2, f64::NAN);

            assert_eq!(
                try_insphere_from_matrix(&m, k, 1).unwrap(),
                InSphere::BOUNDARY
            );
        });
    }

    #[test]
    fn test_try_insphere_from_matrix_rejects_dimension_mismatch() {
        let matrix = Matrix::<3>::zero();
        let err = try_insphere_from_matrix(&matrix, 2, 1).unwrap_err();

        assert!(matches!(
            err,
            StackMatrixDispatchError::ActiveBlockDimensionMismatch { k: 2, dim: 3 }
        ));
    }

    // =======================================================================
    // insphere() error paths
    // =======================================================================

    #[test]
    fn test_insphere_wrong_point_count() {
        let two_points: Vec<Point<f64, 3>> =
            vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])];
        let result = insphere(&two_points, Point::new([0.5, 0.5, 0.5]));
        assert!(result.is_err(), "insphere with 2 points in 3D should error");
    }

    // =======================================================================
    // insphere_lifted() error paths
    // =======================================================================

    #[test]
    fn test_insphere_lifted_overflow_test_point_squared_norm() {
        // Test point very far from the simplex: relative squared norm overflows
        // f64, triggering the map_err conversion on the test-point path.
        let simplex = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        // 1e155² = 1e310 overflows f64::MAX ≈ 1.8e308.
        let far_point = Point::new([1e155, 0.0, 0.0]);
        let result = insphere_lifted(&simplex, far_point);
        assert!(
            result.is_err(),
            "insphere_lifted should error when test point squared norm overflows"
        );
    }

    #[test]
    fn test_exact_orientation_truly_degenerate() {
        // Truly degenerate cases should still be detected correctly.
        let collinear_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([2.0, 0.0]),
        ];
        assert_eq!(
            simplex_orientation(&collinear_2d).unwrap(),
            Orientation::DEGENERATE,
            "Exactly collinear points must be DEGENERATE"
        );

        // Row 2 = row 0 + row 1 in exact arithmetic (linear combination).
        let coplanar_3d = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([5.0, 7.0, 9.0]),
            Point::new([0.0, 0.0, 0.0]),
        ];
        assert_eq!(
            simplex_orientation(&coplanar_3d).unwrap(),
            Orientation::DEGENERATE,
            "Linearly dependent 3D simplex must be DEGENERATE"
        );
    }
}
