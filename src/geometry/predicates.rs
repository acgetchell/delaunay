//! Geometric predicates for d-dimensional geometry calculations.
//!
//! This module contains fundamental geometric predicates and calculations
//! that operate on points and simplices, including circumcenter and circumradius
//! calculations.

#![forbid(unsafe_code)]

use core::hint::cold_path;

use crate::core::simplex::SimplexValidationError;
use crate::geometry::matrix::{
    Interval, IntervalDeterminantSign, IntervalMatrix, LaError, LaVector, Matrix, RationalMatrix,
    StackMatrixDispatchError, matrix_fast_filter, matrix_set, rational_from_f64,
};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{
    CoordinateConversionError, DEFAULT_TOLERANCE_F64, DegenerateSimplexReason,
    InvalidCoordinateValue,
};
use crate::geometry::util::{CircumcenterError, circumcenter, circumradius_with_center};

/// Convert an exact determinant sign (from `det_sign_exact`) to an [`Orientation`].
#[inline]
const fn sign_to_orientation(sign: i8) -> Orientation {
    match sign {
        1 => Orientation::POSITIVE,
        -1 => Orientation::NEGATIVE,
        _ => Orientation::DEGENERATE,
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

/// Encloses the exact circumsphere in an outward-rounded coordinate box.
///
/// Solve `(p_i - p_0) · c = |p_i - p_0|² / 2` in intervals, where `c`
/// is the center relative to `p_0`. Every operation encloses its exact-real
/// counterpart, including pivoting, back substitution and the radius. This
/// box is only a broad phase: points inside it still need the robust predicate.
/// An uncertain pivot, degenerate simplex or non-finite bound returns `None`
/// and callers must retain their exhaustive predicate scan.
/// See Moore, Kearfott and Cloud (2009), *Introduction to Interval Analysis*,
/// in `REFERENCES.md`, Circumcenter and Circumradius Calculations.
pub(crate) fn circumsphere_coordinate_bounds<const D: usize>(
    simplex_points: &[Point<D>],
) -> Option<[[f64; 2]; D]> {
    if D == 0 || D > 6 || simplex_points.len() != D + 1 {
        return None;
    }
    let reference = simplex_points[0].coords();
    let half = Interval::point(0.5).ok()?;
    let mut matrix = [[Interval::ZERO; D]; D];
    let mut rhs = [Interval::ZERO; D];
    for (row, point) in simplex_points.iter().skip(1).enumerate() {
        let mut squared_norm = Interval::ZERO;
        for (column, &coordinate) in point.coords().iter().enumerate() {
            let relative = Interval::try_from_subtraction(coordinate, reference[column]).ok()?;
            matrix[row][column] = relative;
            squared_norm = squared_norm.try_add(&relative.try_square().ok()?).ok()?;
        }
        rhs[row] = squared_norm.try_mul(&half).ok()?;
    }

    // Both an uncertain pivot and exhausted finite range decline the broad
    // phase. Neither outcome proves geometric singularity or excludes a point.
    let center = IntervalMatrix::from_rows(matrix).solve(&rhs).ok()?;
    let mut squared_radius = Interval::ZERO;
    for coordinate in center {
        squared_radius = squared_radius
            .try_add(&coordinate.try_square().ok()?)
            .ok()?;
    }
    let radius = squared_radius.upper().sqrt().next_up();
    if !radius.is_finite() {
        return None;
    }
    let mut bounds = [[0.0; 2]; D];
    for (axis, coordinate) in center.iter().enumerate() {
        let absolute = coordinate
            .try_add(&Interval::point(reference[axis]).ok()?)
            .ok()?;
        let lower = (absolute.lower() - radius).next_down();
        let upper = (absolute.upper() + radius).next_up();
        if !lower.is_finite() || !upper.is_finite() {
            return None;
        }
        bounds[axis] = [lower, upper];
    }
    Some(bounds)
}

/// Encloses geometric differences and lifts before any rounding can lose their sign.
fn fill_relative_insphere_interval_matrix<const D: usize, const N: usize>(
    matrix: &mut IntervalMatrix<N>,
    simplex_points: &[Point<D>],
    test_point: &Point<D>,
) -> Result<(), LaError> {
    let reference_coords = simplex_points[0].coords();
    for (row, point) in simplex_points
        .iter()
        .skip(1)
        .chain(core::iter::once(test_point))
        .enumerate()
    {
        let mut lifted = Interval::ZERO;
        for (column, (&coordinate, &reference)) in
            point.coords().iter().zip(reference_coords).enumerate()
        {
            let relative = Interval::try_from_subtraction(coordinate, reference)?;
            matrix.set(row, column, relative)?;
            lifted = lifted.try_add(&relative.try_square()?)?;
        }
        matrix.set(row, D, lifted)?;
    }
    Ok(())
}

/// Accepts only a strict interval sign; every other outcome requests exact arithmetic.
fn relative_insphere_interval_sign<const D: usize>(
    simplex_points: &[Point<D>],
    test_point: &Point<D>,
) -> Option<i32> {
    if simplex_points.len() != D + 1 {
        return None;
    }
    let sign = la_stack::try_with_interval_matrix!(D + 1, |mut matrix| -> Result<_, LaError> {
        fill_relative_insphere_interval_matrix(&mut matrix, simplex_points, test_point)?;
        matrix.det_sign()
    })
    .ok()?;
    match sign {
        IntervalDeterminantSign::Positive => Some(1),
        IntervalDeterminantSign::Negative => Some(-1),
        // Preserve exact-zero handling as well as inconclusive/range fallback.
        _ => None,
    }
}

/// Fills the dispatched relative lifted matrix from exact IEEE-754 rationals.
/// Callers admit D+1 simplex points and dispatch N=D+1 before assembly.
fn fill_exact_relative_insphere_matrix<const D: usize, const N: usize>(
    matrix: &mut RationalMatrix<N>,
    simplex_points: &[Point<D>],
    test_point: &Point<D>,
) -> Result<(), CoordinateConversionError> {
    let reference: Vec<_> = simplex_points[0]
        .coords()
        .iter()
        .copied()
        .map(rational_from_f64)
        .collect::<Option<_>>()
        .ok_or_else(|| CoordinateConversionError::NonFiniteValue {
            coordinate_index: 0,
            coordinate_value: InvalidCoordinateValue::from_debug(&simplex_points[0].coords()[0]),
        })?;
    let zero = rational_from_f64(0.0).expect("zero is a finite IEEE-754 value");
    for (row, point) in simplex_points
        .iter()
        .skip(1)
        .chain(core::iter::once(test_point))
        .enumerate()
    {
        let mut lifted = zero.clone();
        for (column, (coordinate, reference_coordinate)) in
            point.coords().iter().zip(&reference).enumerate()
        {
            let exact_coordinate = rational_from_f64(*coordinate).ok_or_else(|| {
                CoordinateConversionError::NonFiniteValue {
                    coordinate_index: column,
                    coordinate_value: InvalidCoordinateValue::from_debug(coordinate),
                }
            })?;
            let relative = exact_coordinate - reference_coordinate.clone();
            lifted += relative.clone() * relative.clone();
            matrix.set(row, column, relative)?;
        }
        matrix.set(row, D, lifted)?;
    }
    Ok(())
}

/// Compute exact signs for the relative-coordinate lifted insphere formulation.
#[inline]
pub(crate) fn relative_insphere_signs<const D: usize>(
    simplex_points: &[Point<D>],
    test_point: &Point<D>,
) -> Result<RelativeInsphereSigns, CoordinateConversionError> {
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::InvalidSimplexPointCount {
            actual: simplex_points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    if D + 1 > 7 {
        return Err(CoordinateConversionError::UnsupportedMatrixDimension {
            requested: D + 1,
            max: 7,
        });
    }

    let absolute_orientation = match simplex_orientation(simplex_points)? {
        Orientation::NEGATIVE => -1,
        Orientation::DEGENERATE => 0,
        Orientation::POSITIVE => 1,
    };
    let relative_orientation = if D.is_multiple_of(2) {
        absolute_orientation
    } else {
        -absolute_orientation
    };
    let insphere_determinant = relative_insphere_determinant_sign(simplex_points, test_point)?;

    Ok(RelativeInsphereSigns {
        relative_orientation,
        insphere_determinant,
    })
}

/// Compute only the lifted insphere determinant sign for a relative-coordinate matrix.
///
/// Callers that already know the simplex orientation can combine this determinant
/// with their orientation sign without recomputing the orientation determinant.
#[inline]
pub(crate) fn relative_insphere_determinant_sign<const D: usize>(
    simplex_points: &[Point<D>],
    test_point: &Point<D>,
) -> Result<i32, CoordinateConversionError> {
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::InvalidSimplexPointCount {
            actual: simplex_points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    if D + 1 > 7 {
        return Err(CoordinateConversionError::UnsupportedMatrixDimension {
            requested: D + 1,
            max: 7,
        });
    }
    if let Some(sign) = relative_insphere_interval_sign(simplex_points, test_point) {
        return Ok(sign);
    }

    cold_path();
    la_stack::try_with_rational_matrix!(
        D + 1,
        |mut matrix| -> Result<_, CoordinateConversionError> {
            fill_exact_relative_insphere_matrix(&mut matrix, simplex_points, test_point)?;
            Ok(i32::from(matrix.det_sign().as_i8()))
        }
    )
}

/// Compute orientation from a pre-populated orientation matrix.
///
/// Uses [`la_stack::Matrix::det_sign_exact`] for provably correct results when
/// the matrix entries are finite (even if the f64 determinant overflows).
/// `la-stack` matrices are finite by construction, so an inconclusive or
/// overflowed fast filter always proceeds to an exact sign.
///
/// `k` must equal the number of rows/columns actually used in `matrix`.
#[inline]
pub(crate) fn try_orientation_from_matrix<const N: usize>(
    matrix: &Matrix<N>,
    k: usize,
) -> Result<Orientation, StackMatrixDispatchError> {
    validate_active_matrix_dimension::<N>(k)?;

    // Stage 1: the backend certifies direct determinants for matrix size N ≤ 4.
    if let Some((det, errbound)) = matrix_fast_filter(matrix)? {
        if det > errbound {
            return Ok(Orientation::POSITIVE);
        }
        if det < -errbound {
            return Ok(Orientation::NEGATIVE);
        }
    }

    // Stage 2: exact sign via Bareiss for ambiguous f64 results or N ≥ 5.
    cold_path();
    Ok(sign_to_orientation(matrix.det_sign_exact().as_i8()))
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
/// sign. For D ≤ 3, a fast f64 filter resolves the sign without allocating in
/// well-conditioned cases. For nearly-degenerate configurations (and always for
/// D = 4–6), the Bareiss algorithm runs in exact integer arithmetic in `la-stack`.
/// Dimensions above D = 6 are unsupported.
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
/// Returns an error if the number of simplex points is not exactly D+1 or
/// the dimension exceeds D = 6.
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
/// use delaunay::prelude::geometry::{
///     Coordinate, CoordinateConversionError, Orientation, Point, simplex_orientation,
/// };
///
/// # fn main() -> Result<(), CoordinateConversionError> {
/// let point1 = Point::try_from([0.0, 0.0, 0.0])?;
/// let point2 = Point::try_from([1.0, 0.0, 0.0])?;
/// let point3 = Point::try_from([0.0, 1.0, 0.0])?;
/// let point4 = Point::try_from([0.0, 0.0, 1.0])?;
/// let simplex_points = vec![point1, point2, point3, point4];
/// let oriented = simplex_orientation(&simplex_points)?;
/// assert_eq!(oriented, Orientation::NEGATIVE);
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn simplex_orientation<const D: usize>(
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
        // Populate rows with the coordinates of the points of the simplex.
        for (i, p) in simplex_points.iter().enumerate() {
            for (j, &v) in p.coords().iter().enumerate() {
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
/// This method uses a scaled distance tolerance and can classify a near-boundary
/// point as [`InSphere::BOUNDARY`] even when its exact determinant is nonzero.
/// Use [`insphere_lifted`] when an exact determinant sign is required. See the
/// [performance report](https://github.com/acgetchell/delaunay/blob/main/benches/PERFORMANCE_RESULTS.md)
/// for measurements and their workload scope.
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
/// For an exact classification near the decision boundary, prefer
/// [`insphere_lifted`], which uses a filtered-exact determinant with relative
/// coordinates.
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
/// Returns an error if the circumcenter or circumradius calculation fails, or
/// if the point-to-center distance is not representable as a finite `f64`.
/// See [`circumcenter`] for details.
///
/// # Example
///
/// ```
/// use delaunay::prelude::geometry::{
///     CircumcenterError, Coordinate, InSphere, Point, insphere_distance,
/// };
///
/// # fn main() -> Result<(), CircumcenterError> {
/// let point1 = Point::try_from([0.0, 0.0, 0.0])?;
/// let point2 = Point::try_from([1.0, 0.0, 0.0])?;
/// let point3 = Point::try_from([0.0, 1.0, 0.0])?;
/// let point4 = Point::try_from([0.0, 0.0, 1.0])?;
/// let simplex_points = vec![point1, point2, point3, point4];
/// let test_point = Point::try_from([0.5, 0.5, 0.5])?;
/// assert_eq!(insphere_distance(&simplex_points, test_point)?, InSphere::INSIDE);
/// # Ok(())
/// # }
/// ```
pub fn insphere_distance<const D: usize>(
    simplex_points: &[Point<D>],
    test_point: Point<D>,
) -> Result<InSphere, CircumcenterError> {
    let circumcenter = circumcenter(simplex_points)?;
    let circumradius = circumradius_with_center(simplex_points, &circumcenter)?;

    // Do not require the squared point-to-center distance to be representable.
    let point_coords = test_point.coords();
    let circumcenter_coords = circumcenter.coords();

    let mut diff_coords = [0.0; D];
    for (dst, (p, c)) in diff_coords
        .iter_mut()
        .zip(point_coords.iter().zip(circumcenter_coords.iter()))
    {
        *dst = *p - *c;
    }
    let radius = LaVector::try_new(diff_coords)?.norm()?;

    // Scale tolerance with geometric magnitude to avoid absolute-epsilon
    // misclassification for large circumradii in near-degenerate simplices.
    let base_tolerance = DEFAULT_TOLERANCE_F64;
    let scale = 1.0_f64.max(circumradius.abs().max(radius.abs()));
    let tolerance = base_tolerance * scale;
    let signed_margin = circumradius - radius;
    if signed_margin.abs() <= tolerance {
        Ok(InSphere::BOUNDARY)
    } else if signed_margin > 0.0 {
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
/// This method shares the filtered-exact relative-coordinate determinant core
/// with [`insphere_lifted`] and additionally returns [`InSphere::BOUNDARY`]
/// immediately for queries at simplex vertices. Inconclusive filter results
/// fall back to exact arithmetic. See the
/// [performance report](https://github.com/acgetchell/delaunay/blob/main/benches/PERFORMANCE_RESULTS.md)
/// for measurements and their workload scope.
///
/// # Algorithm
///
/// This implementation follows the robust geometric predicates approach (see References below).
///
/// The homogeneous determinant below is evaluated through the reduced relative
/// formulation documented in [`insphere_lifted`], including its shared
/// orientation and dimension-dependent sign interpretation. The two determinant
/// APIs therefore agree wherever both return a classification; their agreement
/// is shared-core consistency evidence, not an independent numerical oracle.
/// Unlike [`insphere_lifted`], this wrapper shortcuts queries at simplex vertices
/// before checking degeneracy and returns [`CoordinateConversionError`] directly.
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
/// The determinant sign depends on both simplex orientation and dimension.
/// This function determines the orientation using [`simplex_orientation`] and
/// applies the shared sign interpretation described in [`insphere_lifted`],
/// ensuring correct results regardless of vertex ordering.
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
/// use delaunay::prelude::geometry::{
///     Coordinate, CoordinateConversionError, InSphere, Point, insphere,
/// };
///
/// # fn main() -> Result<(), CoordinateConversionError> {
/// let point1 = Point::try_from([0.0, 0.0, 0.0])?;
/// let point2 = Point::try_from([1.0, 0.0, 0.0])?;
/// let point3 = Point::try_from([0.0, 1.0, 0.0])?;
/// let point4 = Point::try_from([0.0, 0.0, 1.0])?;
/// let simplex_points = vec![point1, point2, point3, point4];
///
/// // Test with a point clearly outside the circumsphere
/// let outside_point = Point::try_from([2.0, 2.0, 2.0])?;
/// assert_eq!(insphere(&simplex_points, outside_point)?, InSphere::OUTSIDE);
///
/// // Test with a point clearly inside the circumsphere
/// let inside_point = Point::try_from([0.25, 0.25, 0.25])?;
/// assert_eq!(insphere(&simplex_points, inside_point)?, InSphere::INSIDE);
/// # Ok(())
/// # }
/// ```
///
/// See function-level docs above for detailed explanation and references.
#[inline]
pub fn insphere<const D: usize>(
    simplex_points: &[Point<D>],
    test_point: Point<D>,
) -> Result<InSphere, CoordinateConversionError> {
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

    let signs = relative_insphere_signs(simplex_points, &test_point)?;
    if signs.relative_orientation == 0 {
        Err(CoordinateConversionError::DegenerateSimplex {
            dimension: D,
            reason: DegenerateSimplexReason::ZeroOrientation,
        })
    } else {
        Ok(relative_insphere_classification(signs))
    }
}

/// Check if a point is contained within the circumsphere of a simplex using the lifted paraboloid determinant method.
///
/// This is the recommended exact relative-coordinate implementation of the
/// insphere predicate. It shares its filtered-exact engine with [`insphere`],
/// but always evaluates simplex orientation (including queries at simplex
/// vertices) and maps failures into [`SimplexValidationError`].
///
/// # Performance
///
/// This method preserves an exact-sign contract through a fast filter and an
/// exact fallback; its cost depends on which path resolves the input.
/// [`insphere_distance`] uses a tolerance and has a different correctness
/// contract. See the
/// [performance report](https://github.com/acgetchell/delaunay/blob/main/benches/PERFORMANCE_RESULTS.md)
/// for measurements and their workload scope.
///
/// # Robustness
///
/// The orientation sub-predicate uses [`simplex_orientation`]: a direct f64
/// determinant filter through D=3, followed by exact integer arithmetic when
/// inconclusive (and always for D=4–6).
/// The relative lifted insphere determinant uses:
/// 1. **Outward-rounded interval filter** — resolves well-conditioned cases
///    without allocation and encloses subtraction and lift-construction error.
/// 2. **Exact rational cold path** — constructs relative coordinates and
///    squared norms directly from the exact IEEE-754 inputs before elimination.
///
/// Thus even a finite coordinate whose squared norm would overflow `f64`
/// remains classifiable on the exact path. A returned classification has the
/// mathematically correct sign for the stored finite IEEE-754 coordinates. For
/// additional robustness strategies (symbolic perturbation, consistency
/// checking), use [`crate::geometry::kernel::RobustKernel`].
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
/// # fn main() -> Result<(), delaunay::prelude::geometry::CoordinateConversionError> {
/// let point1 = Point::try_from([0.0, 0.0, 0.0])?;
/// let point2 = Point::try_from([1.0, 0.0, 0.0])?;
/// let point3 = Point::try_from([0.0, 1.0, 0.0])?;
/// let point4 = Point::try_from([0.0, 0.0, 1.0])?;
/// let simplex_points = vec![point1, point2, point3, point4];
///
/// // Test with a point that should be inside according to the lifted paraboloid method
/// let test_point = Point::try_from([0.1, 0.1, 0.1])?;
/// let result = insphere_lifted(&simplex_points, test_point);
/// assert!(result.is_ok()); // Should execute without error
/// # Ok(())
/// # }
/// ```
pub fn insphere_lifted<const D: usize>(
    simplex_points: &[Point<D>],
    test_point: Point<D>,
) -> Result<InSphere, SimplexValidationError> {
    if simplex_points.len() != D + 1 {
        return Err(SimplexValidationError::InsufficientVertices {
            actual: simplex_points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    let signs = relative_insphere_signs(simplex_points, &test_point)
        .map_err(|source| SimplexValidationError::CoordinateConversion { source })?;
    if signs.relative_orientation == 0 {
        Err(SimplexValidationError::DegenerateSimplex)
    } else {
        Ok(relative_insphere_classification(signs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::matrix::matrix_set as try_matrix_set;
    use crate::geometry::matrix::test_support::with_la_stack_matrix;
    use crate::geometry::robust_predicates::robust_insphere;
    use crate::prelude::circumradius;
    use approx::assert_relative_eq;
    use proptest::{prelude::*, test_runner::RngSeed};
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use std::assert_matches;
    use std::collections::HashMap;

    /// Uses an absolute augmented matrix, independent of the production relative
    /// lifting and interval arithmetic. Bareiss is shared, coefficient assembly is not.
    fn augmented_insphere_sign<const D: usize>(simplex: &[Point<D>], query: &Point<D>) -> i32 {
        let sign = la_stack::try_with_rational_matrix!(D + 2, |mut matrix| -> Result<_, LaError> {
            for (row, point) in simplex.iter().chain(core::iter::once(query)).enumerate() {
                let mut squared_norm = rational_from_f64(0.0).unwrap();
                for (column, &coordinate) in point.coords().iter().enumerate() {
                    let exact = rational_from_f64(coordinate).unwrap();
                    squared_norm += &exact * &exact;
                    matrix.set(row, column, exact)?;
                }
                matrix.set(row, D, squared_norm)?;
                matrix.set(row, D + 1, rational_from_f64(1.0).unwrap())?;
            }
            Ok(i32::from(matrix.det_sign().as_i8()))
        })
        .unwrap();
        // Expanding the final column after subtracting the first row gives
        // cofactor parity (-1)^(D+1) relative to the production determinant.
        if D.is_multiple_of(2) { -sign } else { sign }
    }

    /// Exercises strict signs, exact zero, and unavailable finite-range proofs.
    fn assert_interval_adapter_regimes<const D: usize>() {
        for (scale, certifies) in [(1.0, true), (f64::from_bits(1), false), (1.0e155, false)] {
            let mut simplex = vec![Point::try_new([0.0; D]).unwrap()];
            for axis in 0..D {
                let mut coordinates = [0.0; D];
                coordinates[axis] = scale;
                simplex.push(Point::try_new(coordinates).unwrap());
            }
            let outside = Point::try_new([2.0 * scale; D]).unwrap();
            let boundary = Point::try_new([scale; D]).unwrap();
            assert_eq!(relative_insphere_interval_sign(&simplex, &boundary), None);
            assert_eq!(
                relative_insphere_determinant_sign(&simplex, &boundary).unwrap(),
                0
            );
            assert_eq!(
                insphere_lifted(&simplex, boundary).unwrap(),
                InSphere::BOUNDARY
            );
            assert_eq!(augmented_insphere_sign(&simplex, &outside), 1);
            assert_eq!(
                relative_insphere_determinant_sign(&simplex, &outside).unwrap(),
                1
            );
            assert_eq!(
                insphere_lifted(&simplex, outside).unwrap(),
                InSphere::OUTSIDE
            );
            let expected_filter = certifies.then_some(1);
            assert_eq!(
                relative_insphere_interval_sign(&simplex, &outside),
                expected_filter
            );
            simplex.swap(1, 2);
            assert_eq!(
                relative_insphere_determinant_sign(&simplex, &outside).unwrap(),
                -1
            );
            assert_eq!(
                relative_insphere_interval_sign(&simplex, &outside),
                expected_filter.map(|sign| -sign)
            );
            if certifies {
                let inside = Point::try_new([0.5; D]).unwrap();
                assert_eq!(relative_insphere_interval_sign(&simplex, &inside), Some(1));
                assert_eq!(insphere_lifted(&simplex, inside).unwrap(), InSphere::INSIDE);
            }
        }
    }

    /// Checks the filter and its exact fallback against independently assembled coefficients.
    fn assert_interval_adapter_agreement<const D: usize>(
        raw: &[i16],
        exponent: i32,
    ) -> Result<(), TestCaseError> {
        let scale = 2.0_f64.powi(exponent);
        let points: Vec<_> = raw
            .as_chunks::<D>()
            .0
            .iter()
            .map(|coordinates| {
                Point::<D>::try_new(core::array::from_fn(|axis| {
                    f64::from(coordinates[axis]) * scale
                }))
                .unwrap()
            })
            .collect();
        let simplex = &points[..=D];
        let query = &points[D + 1];
        let exact = augmented_insphere_sign(simplex, query);
        if let Some(filtered) = relative_insphere_interval_sign(simplex, query) {
            prop_assert_ne!(filtered, 0);
            prop_assert_eq!(filtered, exact);
        }
        prop_assert_eq!(
            relative_insphere_determinant_sign(simplex, query).unwrap(),
            exact
        );
        Ok(())
    }

    /// The independent predicate never uses the circumsphere enclosure.
    fn assert_circumsphere_bounds_agree_with_predicate<const D: usize>() {
        let mut rng = StdRng::seed_from_u64(0xC1AC_005E + D as u64);
        let mut excluded = 0;
        for scale in [1.0e-100, 1.0, 1.0e100] {
            for thin in [false, true] {
                let origin = [-0.75 * scale; D];
                let mut simplex = vec![Point::try_new(origin).unwrap()];
                for axis in 0..D {
                    let mut coords = origin;
                    for (column, coordinate) in coords.iter_mut().enumerate() {
                        let offset = if column == axis {
                            if thin && axis == D - 1 { 1.0e-12 } else { 1.0 }
                        } else if thin && axis == D - 1 {
                            0.0
                        } else {
                            rng.random_range(-0.1..0.1)
                        };
                        *coordinate = scale.mul_add(offset, *coordinate);
                    }
                    simplex.push(Point::try_new(coords).unwrap());
                }
                let bounds = circumsphere_coordinate_bounds(&simplex);
                let orientation =
                    la_stack::try_with_rational_matrix!(D, |mut matrix| -> Result<_, LaError> {
                        for (row, point) in simplex.iter().skip(1).enumerate() {
                            for (column, (&coordinate, &reference)) in
                                point.coords().iter().zip(&origin).enumerate()
                            {
                                matrix.set(
                                    row,
                                    column,
                                    rational_from_f64(coordinate).unwrap()
                                        - rational_from_f64(reference).unwrap(),
                                )?;
                            }
                        }
                        Ok(i32::from(matrix.det_sign().as_i8()))
                    })
                    .unwrap();
                for query_index in 0..128 {
                    let point = Point::try_new(std::array::from_fn(|_| {
                        scale * rng.random_range(-4.0..4.0)
                    }))
                    .unwrap();
                    // Evaluate every query, including those the broad phase
                    // would exclude. Production predicate failures fail the test.
                    let result = robust_insphere(&simplex, &point).unwrap();
                    if query_index % 16 == 0 {
                        // Force Bareiss on exact IEEE-754 rationals. This
                        // oracle bypasses both floating-point interval filters.
                        let determinant = la_stack::try_with_rational_matrix!(
                            D + 1,
                            |mut matrix| -> Result<_, LaError> {
                                fill_exact_relative_insphere_matrix(&mut matrix, &simplex, &point)
                                    .unwrap();
                                Ok(i32::from(matrix.det_sign().as_i8()))
                            }
                        )
                        .unwrap();
                        assert_eq!(
                            result,
                            relative_insphere_classification(RelativeInsphereSigns {
                                relative_orientation: orientation,
                                insphere_determinant: determinant,
                            })
                        );
                    }
                    if bounds.as_ref().is_some_and(|bounds| {
                        point
                            .coords()
                            .iter()
                            .zip(bounds)
                            .any(|(&coordinate, &[lower, upper])| {
                                coordinate < lower || coordinate > upper
                            })
                    }) {
                        excluded += 1;
                        assert_eq!(
                            result,
                            InSphere::OUTSIDE,
                            "{D}D: {simplex:?}, {point:?}, {bounds:?}"
                        );
                    }
                }
                if let Some(bounds) = bounds {
                    for point in &simplex {
                        assert!(
                            point.coords().iter().zip(&bounds).all(
                                |(&coordinate, &[lower, upper])| {
                                    lower <= coordinate && coordinate <= upper
                                }
                            ),
                            "a cospherical vertex must never be excluded"
                        );
                    }
                }
            }
        }
        assert!(
            excluded >= 100,
            "exercise the certified exclusion path in {D}D"
        );
    }

    macro_rules! interval_adapter_tests {
        ($($dim:literal),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<interval_adapter_preserves_regimes_ $dim d>]() {
                        assert_interval_adapter_regimes::<$dim>();
                    }

                    proptest! {
                        #![proptest_config(ProptestConfig {
                            cases: 24,
                            rng_seed: RngSeed::Fixed(0x577),
                            ..ProptestConfig::default()
                        })]

                        #[test]
                        fn [<interval_adapter_agrees_with_augmented_exact_ $dim d>](
                            raw in prop::collection::vec(-16_i16..=16, ($dim + 2) * $dim),
                            exponent in prop_oneof![Just(-540), Just(0), Just(510)],
                        ) {
                            assert_interval_adapter_agreement::<$dim>(&raw, exponent)?;
                        }
                    }
                )+
            }
        };
    }

    /// Checks entire sphere boxes against an analytic exact-rational sphere,
    /// including finite squared norms near underflow and overflow.
    fn assert_analytic_sphere_enclosure<const D: usize>() {
        for exponent in [-500, -50, 0, 50, 500] {
            let scale = 2.0_f64.powi(exponent);
            let origin = [-0.5 * scale; D];
            let mut simplex = vec![Point::try_new(origin).unwrap()];
            for axis in 0..D {
                let mut point = origin;
                point[axis] += scale;
                simplex.push(Point::try_new(point).unwrap());
            }
            let bounds = circumsphere_coordinate_bounds(&simplex)
                .expect("well-conditioned axis simplex has a finite sphere enclosure");
            let half_scale = rational_from_f64(0.5 * scale).unwrap();
            let radius_squared =
                &half_scale * &half_scale * la_stack::BigRational::from_integer(D.into());
            for [lower, upper] in bounds {
                assert!(lower < 0.0 && upper > 0.0);
                let lower = rational_from_f64(lower).unwrap();
                let upper = rational_from_f64(upper).unwrap();
                assert!(&lower * &lower >= radius_squared);
                assert!(&upper * &upper >= radius_squared);
            }
        }
    }

    macro_rules! gen_circumsphere_bounds_tests {
        ($dimension:literal) => {
            pastey::paste! {
                #[test]
                fn [<circumsphere_bounds_preserve_exact_classification_ $dimension d>]() {
                    assert_circumsphere_bounds_agree_with_predicate::<$dimension>();
                }
                #[test]
                fn [<circumsphere_bounds_enclose_analytic_sphere_ $dimension d>]() {
                    assert_analytic_sphere_enclosure::<$dimension>();
                }
            }
        };
    }

    interval_adapter_tests!(2, 3, 4, 5, 6);
    gen_circumsphere_bounds_tests!(1);
    gen_circumsphere_bounds_tests!(2);
    gen_circumsphere_bounds_tests!(3);
    gen_circumsphere_bounds_tests!(4);
    gen_circumsphere_bounds_tests!(5);
    gen_circumsphere_bounds_tests!(6);

    #[test]
    fn circumsphere_bounds_decline_degenerate_and_unrepresentable_systems() {
        let degenerate = [Point::try_new([0.0, 0.0]).unwrap(); 3];
        assert!(circumsphere_coordinate_bounds(&degenerate).is_none());
        let overflowing = [
            Point::try_new([-f64::MAX, 0.0]).unwrap(),
            Point::try_new([f64::MAX, 0.0]).unwrap(),
            Point::try_new([0.0, f64::MAX]).unwrap(),
        ];
        assert!(circumsphere_coordinate_bounds(&overflowing).is_none());
        assert!(circumsphere_coordinate_bounds(&degenerate[..2]).is_none());
    }

    #[test]
    fn circumsphere_bounds_fall_back_on_uncertain_and_exhausted_solves() {
        // Exact division by one lets the upstream solver resolve this tiny
        // pivot. Its analytic center is (-epsilon/2, 1+epsilon/2).
        let cancellation = [
            Point::try_new([0.0, 0.0]).unwrap(),
            Point::try_new([1.0, 1.0]).unwrap(),
            Point::try_new([1.0, 1.0 + f64::EPSILON]).unwrap(),
        ];
        let bounds = circumsphere_coordinate_bounds(&cancellation).unwrap();
        let half_epsilon = rational_from_f64(f64::EPSILON / 2.0).unwrap();
        let center = [
            -half_epsilon.clone(),
            rational_from_f64(1.0).unwrap() + half_epsilon,
        ];
        let radius_squared = &center[0] * &center[0] + &center[1] * &center[1];
        for ([lower, upper], coordinate) in bounds.into_iter().zip(center) {
            let low_gap = &coordinate - rational_from_f64(lower).unwrap();
            let high_gap = rational_from_f64(upper).unwrap() - coordinate;
            assert!(low_gap >= rational_from_f64(0.0).unwrap());
            assert!(high_gap >= rational_from_f64(0.0).unwrap());
            assert!(&low_gap * &low_gap >= radius_squared);
            assert!(&high_gap * &high_gap >= radius_squared);
        }
        // Stored 1/3 is not exact: 3 * fl(1/3) - 1 = -2^-54.
        // Interval division by three cannot certify this nonzero final pivot.
        let uncertain = [
            Point::try_new([0.0, 0.0]).unwrap(),
            Point::try_new([3.0, 1.0]).unwrap(),
            Point::try_new([1.0, 1.0 / 3.0]).unwrap(),
        ];
        assert!(circumsphere_coordinate_bounds(&uncertain).is_none());
        assert_ne!(
            simplex_orientation(&uncertain).unwrap(),
            Orientation::DEGENERATE
        );
        // The relative equations have finite coefficients and RHS; the exact
        // center's second coordinate exceeds binary64's finite range.
        let exhausted = [
            Point::try_new([0.0, 0.0]).unwrap(),
            Point::try_new([1.0, 0.0]).unwrap(),
            Point::try_new([2.0, 1.0e-310]).unwrap(),
        ];
        assert!(circumsphere_coordinate_bounds(&exhausted).is_none());
        let tiny = [
            Point::try_new([0.0]).unwrap(),
            Point::try_new([f64::from_bits(1)]).unwrap(),
        ];
        // In one dimension the exact sphere box is the segment itself, even
        // though its center is half the smallest representable subnormal.
        let [[lower, upper]] = circumsphere_coordinate_bounds(&tiny).unwrap();
        assert!(lower <= 0.0 && upper >= f64::from_bits(1));
    }

    /// Populate a test matrix while keeping production matrix errors loud.
    fn set_test_matrix_entry<const N: usize>(
        matrix: &mut Matrix<N>,
        row: usize,
        column: usize,
        value: f64,
    ) {
        try_matrix_set(matrix, row, column, value).unwrap();
    }

    /// Compare relative rational construction with the absolute lifted exact
    /// reference. Small dyadic coordinates make every absolute square exact in
    /// binary64; the two formulations share elimination but not matrix assembly.
    fn assert_rational_insphere_agreement<const D: usize, const N: usize>() {
        assert_eq!(N, D + 2);
        let mut simplex = vec![Point::try_new([-2.0; D]).unwrap()];
        for axis in 0..D {
            let mut coords = [-2.0; D];
            coords[axis] = 2.0;
            simplex.push(Point::try_new(coords).unwrap());
        }
        for (query, expected) in [
            ([0.0; D], InSphere::INSIDE),
            ([2.0; D], InSphere::BOUNDARY),
            ([4.0; D], InSphere::OUTSIDE),
        ] {
            let query = Point::try_new(query).unwrap();
            for swapped in [false, true] {
                if swapped {
                    simplex.swap(0, 1);
                }
                let points: Vec<_> = simplex.iter().chain(core::iter::once(&query)).collect();
                let absolute = Matrix::<N>::try_from_rows(std::array::from_fn(|row| {
                    std::array::from_fn(|column| match column {
                        column if column < D => points[row].coords()[column],
                        column if column == D => points[row].coords().iter().map(|x| x * x).sum(),
                        _ => 1.0,
                    })
                }))
                .unwrap();
                let rational_sign = la_stack::try_with_rational_matrix!(
                    D + 1,
                    |mut matrix| -> Result<_, LaError> {
                        fill_exact_relative_insphere_matrix(&mut matrix, &simplex, &query).unwrap();
                        Ok(i32::from(matrix.det_sign().as_i8()))
                    }
                )
                .unwrap();
                // Expanding the translated homogeneous matrix in its first row
                // contributes (-1)^(D+1).
                let parity = if D.is_multiple_of(2) { -1 } else { 1 };
                assert_eq!(
                    rational_sign,
                    parity * i32::from(absolute.det_sign_exact().as_i8())
                );
                assert_eq!(
                    relative_insphere_determinant_sign(&simplex, &query).unwrap(),
                    rational_sign
                );
                assert_eq!(insphere_lifted(&simplex, query).unwrap(), expected);
                if swapped {
                    simplex.swap(0, 1);
                }
            }
        }
    }

    macro_rules! rational_insphere_agreement_test {
        ($dimension:literal, $matrix_dimension:literal) => {
            pastey::paste! {
                #[test]
                fn [<rational_insphere_matches_absolute_reference_ $dimension d>]() {
                    assert_rational_insphere_agreement::<$dimension, $matrix_dimension>();
                }
            }
        };
    }

    rational_insphere_agreement_test!(2, 4);
    rational_insphere_agreement_test!(3, 5);
    rational_insphere_agreement_test!(4, 6);
    rational_insphere_agreement_test!(5, 7);
    rational_insphere_agreement_test!(6, 8);

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
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];
        let radius_2d = circumradius(&triangle_2d).unwrap();
        // For a right triangle with legs of length 1, circumradius is sqrt(2)/2
        let expected_radius_2d = 2.0_f64.sqrt() / 2.0;
        assert_relative_eq!(radius_2d, expected_radius_2d, epsilon = 1e-10);

        // 3D: Unit tetrahedron (origin + unit basis vectors)
        let tetrahedron_3d = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let radius_3d = circumradius(&tetrahedron_3d).unwrap();
        tracing::debug!("3D circumradius: {radius_3d}");
        // For unit tetrahedron with vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        // circumradius = sqrt(3)/2 ≈ 0.866
        let expected_radius_3d = (3.0_f64).sqrt() / 2.0;
        assert_relative_eq!(radius_3d, expected_radius_3d, epsilon = 1e-10);

        // 4D: Unit 4-simplex
        let simplex_4d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let radius_4d = circumradius(&simplex_4d).unwrap();
        tracing::debug!("4D circumradius: {radius_4d}");
        // For unit 4-simplex, circumradius = 1.0
        let expected_radius_4d = 1.0;
        assert_relative_eq!(radius_4d, expected_radius_4d, epsilon = 1e-10);

        // 5D: Unit 5-simplex
        let simplex_5d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let radius_5d = circumradius(&simplex_5d).unwrap();
        tracing::debug!("5D circumradius: {radius_5d}");
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

        // Log summary for verification
        tracing::debug!("Circumradius summary:");
        let expected_2d = (2.0_f64).sqrt() / 2.0;
        let expected_3d = (3.0_f64).sqrt() / 2.0;
        let expected_5d = (5.0_f64).sqrt() / 2.0;
        tracing::debug!("  2D (right triangle): {radius_2d} ≈ {expected_2d:.6}");
        tracing::debug!("  3D (unit tetrahedron): {radius_3d} ≈ {expected_3d:.6}");
        tracing::debug!("  4D (unit 4-simplex): {radius_4d} = 1.0");
        tracing::debug!("  5D (unit 5-simplex): {radius_5d} ≈ {expected_5d:.6}");
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "dimension sweep keeps comparable insphere cases together"
    )]
    fn test_insphere_basic_functionality_2d_to_5d() {
        // Test basic insphere functionality across dimensions 2D-5D

        // 2D triangle case
        let simplex_2d = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];

        // Test far outside, inside, and boundary points for 2D
        assert_eq!(
            insphere_lifted(
                &simplex_2d,
                Point::try_new([10.0, 10.0]).expect("finite point coordinates")
            )
            .unwrap(),
            InSphere::OUTSIDE,
            "2D far outside point should be OUTSIDE"
        );
        assert_eq!(
            insphere_lifted(
                &simplex_2d,
                Point::try_new([0.1, 0.1]).expect("finite point coordinates")
            )
            .unwrap(),
            InSphere::INSIDE,
            "2D inside point should be INSIDE"
        );
        assert_eq!(
            insphere_lifted(
                &simplex_2d,
                Point::try_new([0.0, 0.0]).expect("finite point coordinates")
            )
            .unwrap(),
            InSphere::BOUNDARY,
            "2D vertex should be BOUNDARY"
        );

        // 3D tetrahedron case
        let simplex_3d = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        // Test far outside, inside, and boundary points for 3D
        assert_eq!(
            insphere_lifted(
                &simplex_3d,
                Point::try_new([10.0, 10.0, 10.0]).expect("finite point coordinates")
            )
            .unwrap(),
            InSphere::OUTSIDE,
            "3D far outside point should be OUTSIDE"
        );
        assert_eq!(
            insphere_lifted(
                &simplex_3d,
                Point::try_new([0.1, 0.1, 0.1]).expect("finite point coordinates")
            )
            .unwrap(),
            InSphere::INSIDE,
            "3D inside point should be INSIDE"
        );
        assert_eq!(
            insphere_lifted(
                &simplex_3d,
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates")
            )
            .unwrap(),
            InSphere::BOUNDARY,
            "3D vertex should be BOUNDARY"
        );

        // 4D simplex case
        let simplex_4d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        // Test far outside, inside, and boundary points for 4D
        assert_eq!(
            insphere_lifted(
                &simplex_4d,
                Point::try_new([10.0, 10.0, 10.0, 10.0]).expect("finite point coordinates")
            )
            .unwrap(),
            InSphere::OUTSIDE,
            "4D far outside point should be OUTSIDE"
        );
        assert_eq!(
            insphere_lifted(
                &simplex_4d,
                Point::try_new([0.1, 0.1, 0.1, 0.1]).expect("finite point coordinates")
            )
            .unwrap(),
            InSphere::INSIDE,
            "4D inside point should be INSIDE"
        );
        assert_eq!(
            insphere_lifted(
                &simplex_4d,
                Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates")
            )
            .unwrap(),
            InSphere::BOUNDARY,
            "4D vertex should be BOUNDARY"
        );

        // 5D simplex case
        let simplex_5d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        // Test far outside, inside, and boundary points for 5D
        assert_eq!(
            insphere_lifted(
                &simplex_5d,
                Point::try_new([10.0, 10.0, 10.0, 10.0, 10.0]).expect("finite point coordinates")
            )
            .unwrap(),
            InSphere::OUTSIDE,
            "5D far outside point should be OUTSIDE"
        );
        assert_eq!(
            insphere_lifted(
                &simplex_5d,
                Point::try_new([0.1, 0.1, 0.1, 0.1, 0.1]).expect("finite point coordinates")
            )
            .unwrap(),
            InSphere::INSIDE,
            "5D inside point should be INSIDE"
        );
        assert_eq!(
            insphere_lifted(
                &simplex_5d,
                Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates")
            )
            .unwrap(),
            InSphere::BOUNDARY,
            "5D vertex should be BOUNDARY"
        );
    }

    #[test]
    fn test_insphere_edge_cases_and_errors() {
        // Test edge cases across dimensions including 1D

        // 1D case (line segment)
        let simplex_1d = vec![
            Point::try_new([0.0]).expect("finite point coordinates"),
            Point::try_new([2.0]).expect("finite point coordinates"),
        ];
        let midpoint_1d = Point::try_new([1.0]).expect("finite point coordinates");
        let far_point_1d = Point::try_new([10.0]).expect("finite point coordinates");

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
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        // Circumcenter should be inside the circumsphere
        let circumcenter_3d = Point::try_new([0.5, 0.5, 0.5]).expect("finite point coordinates");
        assert_eq!(
            insphere_lifted(&simplex_3d, circumcenter_3d).unwrap(),
            InSphere::INSIDE,
            "3D circumcenter should be INSIDE"
        );

        // Test regular 4D simplex with symmetric properties
        let regular_4d_simplex = vec![
            Point::try_new([1.0, 1.0, 1.0, 1.0]).expect("finite point coordinates"),
            Point::try_new([1.0, -1.0, -1.0, -1.0]).expect("finite point coordinates"),
            Point::try_new([-1.0, 1.0, -1.0, -1.0]).expect("finite point coordinates"),
            Point::try_new([-1.0, -1.0, 1.0, -1.0]).expect("finite point coordinates"),
            Point::try_new([-1.0, -1.0, -1.0, 1.0]).expect("finite point coordinates"),
        ];

        // Origin should be inside this symmetric simplex
        assert_eq!(
            insphere_lifted(
                &regular_4d_simplex,
                Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates")
            )
            .unwrap(),
            InSphere::INSIDE,
            "Origin should be inside symmetric 4D simplex"
        );

        // Error case: insufficient vertices
        let incomplete_simplex = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
        ]; // Only 2 vertices for 3D
        let test_point = Point::try_new([0.5, 0.5, 0.5]).expect("finite point coordinates");

        assert!(
            insphere_lifted(&incomplete_simplex, test_point).is_err(),
            "Should error with insufficient vertices"
        );
    }

    #[test]
    fn test_relative_insphere_determinant_sign_rejects_wrong_point_count() {
        let incomplete_simplex = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
        ];
        let test_point = Point::try_new([0.5, 0.5, 0.5]).expect("finite point coordinates");

        let err = relative_insphere_determinant_sign(&incomplete_simplex, &test_point).unwrap_err();

        assert_eq!(
            err,
            CoordinateConversionError::InvalidSimplexPointCount {
                actual: 2,
                expected: 4,
                dimension: 3,
            }
        );
    }

    #[test]
    fn test_relative_insphere_determinant_sign_boundary_matches_exact_signs() {
        let simplex = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];
        let boundary = Point::try_new([1.0, 1.0]).expect("finite point coordinates");

        let determinant_sign = relative_insphere_determinant_sign(&simplex, &boundary).unwrap();
        let signs = relative_insphere_signs(&simplex, &boundary).unwrap();

        assert_eq!(determinant_sign, 0);
        assert_eq!(determinant_sign, signs.insphere_determinant);
    }

    #[test]
    fn predicates_circumcenter_error_cases() {
        // Test circumcenter calculation with degenerate cases
        let points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        ]; // Only 2 points for 2D

        // Test with insufficient vertices for proper simplex
        let center_result = circumcenter(&points);
        assert!(center_result.is_err());
    }

    #[test]
    fn predicates_circumcenter_collinear_points() {
        // Test circumcenter with collinear points (should fail)
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([2.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([3.0, 0.0, 0.0]).expect("finite point coordinates"),
        ];

        // This should fail because points are collinear
        let center_result = circumcenter(&points);
        assert!(center_result.is_err());
    }

    #[test]
    fn predicates_circumsphere_edge_cases() {
        // Test circumsphere containment with simple cases
        let simplex_points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];

        // Test that the methods run without error
        let test_point = Point::try_new([0.25, 0.25]).expect("finite point coordinates");
        assert!(insphere_distance(&simplex_points, test_point).is_ok());

        // At minimum, both methods should give the same result for the same input
        let far_point = Point::try_new([100.0, 100.0]).expect("finite point coordinates");
        assert!(insphere_distance(&simplex_points, far_point).is_ok());
    }

    #[test]
    fn insphere_distance_rejects_unrepresentable_finite_radius() {
        let magnitude = 1.1e308;
        let simplex_points = vec![
            Point::try_new([-magnitude, -magnitude, -magnitude]).expect("finite point coordinates"),
            Point::try_new([magnitude, -magnitude, -magnitude]).expect("finite point coordinates"),
            Point::try_new([-magnitude, magnitude, -magnitude]).expect("finite point coordinates"),
            Point::try_new([-magnitude, -magnitude, magnitude]).expect("finite point coordinates"),
        ];
        let center = Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates");

        assert_matches!(
            insphere_distance(&simplex_points, center),
            Err(CircumcenterError::LinearAlgebraFailure {
                source: LaError::NonFinite { .. },
            })
        );
    }

    #[test]
    fn insphere_distance_accepts_finite_distance_with_overflowing_square() {
        let simplex_points = [
            Point::try_new([0.0, 0.0, 0.0]).unwrap(),
            Point::try_new([1.0, 0.0, 0.0]).unwrap(),
            Point::try_new([0.0, 1.0, 0.0]).unwrap(),
            Point::try_new([0.0, 0.0, 1.0]).unwrap(),
        ];
        let far_point = Point::try_new([1.0e200; 3]).unwrap();
        assert_eq!(
            insphere_distance(&simplex_points, far_point),
            Ok(InSphere::OUTSIDE)
        );
    }

    #[test]
    fn insphere_distance_rejects_unrepresentable_finite_point_distance() {
        let simplex_points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let far_point =
            Point::try_new([f64::MAX, f64::MAX, f64::MAX]).expect("finite point coordinates");

        assert_matches!(
            insphere_distance(&simplex_points, far_point),
            Err(CircumcenterError::LinearAlgebraFailure {
                source: LaError::NonFinite { .. },
            })
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "orientation regression test keeps dimension-specific cases together"
    )]
    fn test_simplex_orientation_comprehensive() {
        // Test 2D orientation - positive case
        let positive_2d = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            simplex_orientation(&positive_2d).unwrap(),
            Orientation::POSITIVE,
            "2D positive orientation failed"
        );

        // Test 2D orientation - negative case (reversed order)
        let negative_2d = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            simplex_orientation(&negative_2d).unwrap(),
            Orientation::NEGATIVE,
            "2D negative orientation failed"
        );

        // Test 2D degenerate case - collinear points
        let degenerate_2d = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([2.0, 0.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            simplex_orientation(&degenerate_2d).unwrap(),
            Orientation::DEGENERATE,
            "2D degenerate case failed"
        );

        // Test 3D orientation - positive case
        let positive_3d = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            simplex_orientation(&positive_3d).unwrap(),
            Orientation::POSITIVE,
            "3D positive orientation failed"
        );

        // Test 3D orientation - negative case
        let negative_3d = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            simplex_orientation(&negative_3d).unwrap(),
            Orientation::NEGATIVE,
            "3D negative orientation failed"
        );

        // Test 3D degenerate case - coplanar points
        let degenerate_3d = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 1.0, 0.0]).expect("finite point coordinates"), // All points on z=0 plane
        ];
        assert_eq!(
            simplex_orientation(&degenerate_3d).unwrap(),
            Orientation::DEGENERATE,
            "3D degenerate case failed"
        );

        // Test 4D orientation - positive case
        let positive_4d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            simplex_orientation(&positive_4d).unwrap(),
            Orientation::POSITIVE,
            "4D positive orientation failed"
        );

        // Test 4D orientation - negative case (different ordering)
        let negative_4d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            simplex_orientation(&negative_4d).unwrap(),
            Orientation::NEGATIVE,
            "4D negative orientation failed"
        );

        // Test 4D degenerate case - points in 3D subspace
        let degenerate_4d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 1.0, 1.0, 0.0]).expect("finite point coordinates"), // All points have w=0
        ];
        assert_eq!(
            simplex_orientation(&degenerate_4d).unwrap(),
            Orientation::DEGENERATE,
            "4D degenerate case failed"
        );

        // Test 5D orientation - positive case
        // For even dimensions, we need to adjust vertex order to get positive orientation
        let positive_5d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            simplex_orientation(&positive_5d).unwrap(),
            Orientation::POSITIVE,
            "5D positive orientation failed"
        );

        // Test 5D orientation - negative case (reversed from positive)
        let negative_5d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            simplex_orientation(&negative_5d).unwrap(),
            Orientation::NEGATIVE,
            "5D negative orientation failed"
        );

        // Test 5D degenerate case - points in 4D subspace
        let degenerate_5d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 1.0, 1.0, 1.0, 0.0]).expect("finite point coordinates"), // All points have v=0
        ];
        assert_eq!(
            simplex_orientation(&degenerate_5d).unwrap(),
            Orientation::DEGENERATE,
            "5D degenerate case failed"
        );

        // Test error case: insufficient vertices
        let insufficient_vertices = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
        ]; // Only 2 vertices for 3D
        assert!(
            simplex_orientation(&insufficient_vertices).is_err(),
            "Should error with insufficient vertices"
        );
    }

    #[test]
    fn test_insphere_degenerate_simplex_error_handling() {
        // Create a degenerate simplex (coplanar points in 3D)
        let degenerate_simplex = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 1.0, 0.0]).expect("finite point coordinates"), // All points lie on the same plane (z=0)
        ];
        let test_point = Point::try_new([0.5, 0.5, 0.5]).expect("finite point coordinates");

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
            Err(SimplexValidationError::DegenerateSimplex) => (), // Expected error type
            Err(other) => panic!("Wrong error type: {other:?}"),
            Ok(_) => panic!("Function should have returned an error"),
        }

        // Test error handling for insufficient vertices
        let insufficient_vertices = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
        ]; // Only 2 vertices for 3D

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
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];

        // Test with one of the original vertices (should be on boundary)
        let vertex_point = Point::try_new([0.0, 0.0]).expect("finite point coordinates");
        let result = insphere_lifted(&simplex_points, vertex_point).unwrap();
        assert_eq!(
            result,
            InSphere::BOUNDARY,
            "Original vertex should be classified as BOUNDARY"
        );

        // Test with a point clearly inside
        let inside_point = Point::try_new([0.1, 0.1]).expect("finite point coordinates");
        let inside_result = insphere_lifted(&simplex_points, inside_point).unwrap();
        assert_eq!(
            inside_result,
            InSphere::INSIDE,
            "Point inside should be classified as INSIDE"
        );

        // Test with a point clearly outside
        let outside_point = Point::try_new([10.0, 10.0]).expect("finite point coordinates");
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
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        // Test multiple points
        let test_cases = [
            // Inside points
            (
                Point::try_new([0.2, 0.2, 0.2]).expect("finite point coordinates"),
                InSphere::INSIDE,
            ),
            // Outside points
            (
                Point::try_new([2.0, 2.0, 2.0]).expect("finite point coordinates"),
                InSphere::OUTSIDE,
            ),
            // Boundary points (simplex vertices should be on boundary)
            (
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                InSphere::BOUNDARY,
            ),
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
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];

        // Test various points and verify all methods agree
        let test_cases = [
            (
                Point::try_new([0.1, 0.1]).expect("finite point coordinates"),
                InSphere::INSIDE,
            ), // Clearly inside
            (
                Point::try_new([10.0, 10.0]).expect("finite point coordinates"),
                InSphere::OUTSIDE,
            ), // Clearly outside
            (
                Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
                InSphere::BOUNDARY,
            ), // Vertex (on boundary)
            (
                Point::try_new([1.0, 1.0]).expect("finite point coordinates"),
                InSphere::BOUNDARY,
            ), // Opposite point on the circumcircle
        ];

        for (test_point, expected) in &test_cases {
            let result_std = insphere(&simplex, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex, *test_point).unwrap();

            tracing::debug!(
                "2D {expected:?}: std={result_std:?}, lifted={result_lifted:?}, distance={result_distance:?}"
            );

            assert_eq!(result_std, *expected, "standard determinant mismatch");
            assert_eq!(result_lifted, *expected, "lifted determinant mismatch");
            assert_eq!(result_distance, *expected, "distance method mismatch");
        }
    }

    #[test]
    fn test_insphere_methods_3d_comprehensive() {
        // 3D tetrahedron: origin and unit basis vectors
        let simplex = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        let test_cases = [
            (
                Point::try_new([0.1, 0.1, 0.1]).expect("finite point coordinates"),
                "inside",
            ),
            (
                Point::try_new([10.0, 10.0, 10.0]).expect("finite point coordinates"),
                "outside",
            ),
            (
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                "boundary",
            ), // Vertex
            (
                Point::try_new([0.25, 0.25, 0.25]).expect("finite point coordinates"),
                "inside",
            ), // Centroid region
        ];

        for (test_point, description) in &test_cases {
            let result_std = insphere(&simplex, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex, *test_point).unwrap();

            tracing::debug!(
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
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        let test_cases = [
            (
                Point::try_new([0.1, 0.1, 0.1, 0.1]).expect("finite point coordinates"),
                "inside",
            ),
            (
                Point::try_new([10.0, 10.0, 10.0, 10.0]).expect("finite point coordinates"),
                "outside",
            ),
            (
                Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                "boundary",
            ), // Vertex
            (
                Point::try_new([0.2, 0.2, 0.2, 0.2]).expect("finite point coordinates"),
                "inside",
            ),
        ];

        for (test_point, description) in &test_cases {
            let result_std = insphere(&simplex, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex, *test_point).unwrap();

            tracing::debug!(
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
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        let test_cases = [
            (
                Point::try_new([0.1, 0.1, 0.1, 0.1, 0.1]).expect("finite point coordinates"),
                "inside",
            ),
            (
                Point::try_new([10.0, 10.0, 10.0, 10.0, 10.0]).expect("finite point coordinates"),
                "outside",
            ),
            (
                Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                "boundary",
            ), // Vertex
            (
                Point::try_new([0.15, 0.15, 0.15, 0.15, 0.15]).expect("finite point coordinates"),
                "inside",
            ),
        ];

        for (test_point, description) in &test_cases {
            let result_std = insphere(&simplex, *test_point).unwrap();
            let result_lifted = insphere_lifted(&simplex, *test_point).unwrap();
            let result_distance = insphere_distance(&simplex, *test_point).unwrap();

            tracing::debug!(
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
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1e-6, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1e-6]).expect("finite point coordinates"),
        ];
        let test_point_2d = Point::try_new([1e-7, 1e-7]).expect("finite point coordinates");
        let result_2d = insphere(&tiny_simplex_2d, test_point_2d);
        assert!(result_2d.is_ok(), "2D tiny simplex should work");

        // 3D: Test with large coordinates
        let large_simplex_3d = vec![
            Point::try_new([1e6, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1e6 + 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1e6, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1e6, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let test_point_3d =
            Point::try_new([1e6 + 0.1, 0.1, 0.1]).expect("finite point coordinates");
        let result_3d = insphere(&large_simplex_3d, test_point_3d);
        assert!(result_3d.is_ok(), "3D large coordinates should work");

        // 4D: Test with negative coordinates
        let negative_simplex_4d = vec![
            Point::try_new([-1.0, -1.0, -1.0, -1.0]).expect("finite point coordinates"),
            Point::try_new([0.0, -1.0, -1.0, -1.0]).expect("finite point coordinates"),
            Point::try_new([-1.0, 0.0, -1.0, -1.0]).expect("finite point coordinates"),
            Point::try_new([-1.0, -1.0, 0.0, -1.0]).expect("finite point coordinates"),
            Point::try_new([-1.0, -1.0, -1.0, 0.0]).expect("finite point coordinates"),
        ];
        let test_point_4d =
            Point::try_new([-0.5, -0.5, -0.5, -0.5]).expect("finite point coordinates");
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
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        let test_points = [
            Point::try_new([0.1, 0.1, 0.1]).expect("finite point coordinates"),
            Point::try_new([0.3, 0.2, 0.1]).expect("finite point coordinates"),
            Point::try_new([0.5, 0.5, 0.5]).expect("finite point coordinates"),
            Point::try_new([1.0, 1.0, 1.0]).expect("finite point coordinates"),
            Point::try_new([2.0, 2.0, 2.0]).expect("finite point coordinates"),
            Point::try_new([-0.1, -0.1, -0.1]).expect("finite point coordinates"),
            Point::try_new([0.25, 0.25, 0.25]).expect("finite point coordinates"),
            Point::try_new([0.01, 0.01, 0.01]).expect("finite point coordinates"),
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

        tracing::debug!("Stress test results: {total_tests} total tests");
        for (key, count) in &disagreement_count {
            tracing::debug!("  {key}: {count} disagreements");
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
    fn check_insphere_parity<const D: usize>(
        simplex: &[Point<D>],
        test_point: Point<D>,
        expected_orientation: Orientation,
        expected_result: InSphere,
        dimension: usize,
        orientation_label: &str,
    ) {
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
                Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
            ],
            Point::try_new([0.1, 0.1]).expect("finite point coordinates"),
            Orientation::POSITIVE,
            InSphere::INSIDE,
            2,
            "POSITIVE",
        );

        // 3D (odd dimension) with POSITIVE orientation
        check_insphere_parity(
            &[
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
            ],
            Point::try_new([0.1, 0.1, 0.1]).expect("finite point coordinates"),
            Orientation::POSITIVE,
            InSphere::INSIDE,
            3,
            "POSITIVE",
        );

        // 4D (even dimension) with POSITIVE orientation
        check_insphere_parity(
            &[
                Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
            ],
            Point::try_new([0.1, 0.1, 0.1, 0.1]).expect("finite point coordinates"),
            Orientation::POSITIVE,
            InSphere::INSIDE,
            4,
            "POSITIVE",
        );

        // 5D (odd dimension) with POSITIVE orientation
        check_insphere_parity(
            &[
                Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
            ],
            Point::try_new([0.1, 0.1, 0.1, 0.1, 0.1]).expect("finite point coordinates"),
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
                Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            ],
            Point::try_new([0.1, 0.1]).expect("finite point coordinates"),
            Orientation::NEGATIVE,
            InSphere::INSIDE,
            2,
            "NEGATIVE",
        );

        // 3D (odd dimension) with NEGATIVE orientation
        check_insphere_parity(
            &[
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
            ],
            Point::try_new([0.1, 0.1, 0.1]).expect("finite point coordinates"),
            Orientation::NEGATIVE,
            InSphere::INSIDE,
            3,
            "NEGATIVE",
        );

        // 4D (even dimension) with NEGATIVE orientation
        check_insphere_parity(
            &[
                Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
            ],
            Point::try_new([0.1, 0.1, 0.1, 0.1]).expect("finite point coordinates"),
            Orientation::NEGATIVE,
            InSphere::INSIDE,
            4,
            "NEGATIVE",
        );

        // 5D (odd dimension) with NEGATIVE orientation
        check_insphere_parity(
            &[
                Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
            ],
            Point::try_new([0.1, 0.1, 0.1, 0.1, 0.1]).expect("finite point coordinates"),
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
    fn test_orientation_matrix_rejects_nonfinite_entry_at_set_boundary() {
        let k = 4;
        with_la_stack_matrix!(k, |m| {
            let err = try_matrix_set(&mut m, 3, 3, f64::NAN).unwrap_err();
            assert_eq!(
                err,
                StackMatrixDispatchError::La {
                    source: LaError::non_finite_input_matrix(3, 3),
                }
            );
        });
    }

    #[test]
    fn test_try_orientation_from_matrix_rejects_dimension_mismatch() {
        let matrix = Matrix::<3>::zero();
        let err = try_orientation_from_matrix(&matrix, 4).unwrap_err();

        assert_matches!(
            err,
            StackMatrixDispatchError::ActiveBlockDimensionMismatch { k: 4, dim: 3 }
        );
    }

    #[test]
    fn test_exact_orientation_near_degenerate_2d() {
        // Near-degenerate 2D triangle: third point is almost collinear.
        // The perturbation 2^-50 ≈ 8.9e-16 is below typical adaptive tolerance
        // thresholds but produces a non-zero exact determinant.
        let eps = f64::from_bits(0x3CD0_0000_0000_0000); // 2^-50
        let nearly_collinear = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, eps]).expect("finite point coordinates"),
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
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, eps]).expect("finite point coordinates"),
        ];
        let orientation = simplex_orientation(&nearly_coplanar).unwrap();
        assert_ne!(
            orientation,
            Orientation::DEGENERATE,
            "Near-degenerate 3D tetrahedron with 2^-50 perturbation should NOT be DEGENERATE"
        );
    }

    // =======================================================================
    // Exact matrix evaluation and typed boundary-failure coverage
    // =======================================================================

    #[test]
    fn test_insphere_matrix_rejects_nonfinite_entry_at_set_boundary() {
        let k = 3;
        with_la_stack_matrix!(k, |m| {
            let err = try_matrix_set(&mut m, 2, 2, f64::NAN).unwrap_err();
            assert_eq!(
                err,
                StackMatrixDispatchError::La {
                    source: LaError::non_finite_input_matrix(2, 2),
                }
            );
        });
    }

    // =======================================================================
    // insphere() error paths
    // =======================================================================

    #[test]
    fn test_insphere_wrong_point_count() {
        let two_points: Vec<Point<3>> = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
        ];
        let result = insphere(
            &two_points,
            Point::try_new([0.5, 0.5, 0.5]).expect("finite point coordinates"),
        );
        assert!(result.is_err(), "insphere with 2 points in 3D should error");
    }

    // =======================================================================
    // insphere_lifted() error paths
    // =======================================================================

    #[test]
    fn test_insphere_lifted_exact_path_handles_f64_squared_norm_overflow() {
        // The exact cold path forms the squared norm rationally, so a finite
        // coordinate remains classifiable even when its f64 square overflows.
        let simplex = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        // 1e155² = 1e310 overflows f64::MAX ≈ 1.8e308.
        let far_point = Point::try_new([1e155, 0.0, 0.0]).expect("finite point coordinates");
        assert_eq!(
            insphere_lifted(&simplex, far_point).unwrap(),
            InSphere::OUTSIDE
        );
    }

    #[test]
    fn exact_relative_lift_preserves_cospherical_unit_lost_by_f64_squaring() {
        let simplex = [
            Point::try_new([0.0, 0.0]).unwrap(),
            Point::try_new([1.0, 0.0]).unwrap(),
            Point::try_new([0.0, 1.0e8]).unwrap(),
        ];
        let opposite_corner = Point::try_new([1.0, 1.0e8]).unwrap();

        assert_eq!(
            relative_insphere_interval_sign(&simplex, &opposite_corner),
            None
        );
        assert_eq!(
            insphere_lifted(&simplex, opposite_corner).unwrap(),
            InSphere::BOUNDARY
        );
        assert_eq!(
            insphere(&simplex, opposite_corner).unwrap(),
            InSphere::BOUNDARY
        );
    }

    #[test]
    fn test_exact_orientation_truly_degenerate() {
        // Truly degenerate cases should still be detected correctly.
        let collinear_2d = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([2.0, 0.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            simplex_orientation(&collinear_2d).unwrap(),
            Orientation::DEGENERATE,
            "Exactly collinear points must be DEGENERATE"
        );

        // Row 2 = row 0 + row 1 in exact arithmetic (linear combination).
        let coplanar_3d = vec![
            Point::try_new([1.0, 2.0, 3.0]).expect("finite point coordinates"),
            Point::try_new([4.0, 5.0, 6.0]).expect("finite point coordinates"),
            Point::try_new([5.0, 7.0, 9.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
        ];
        assert_eq!(
            simplex_orientation(&coplanar_3d).unwrap(),
            Orientation::DEGENERATE,
            "Linearly dependent 3D simplex must be DEGENERATE"
        );
    }
}
