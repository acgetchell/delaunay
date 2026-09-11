//! Circumsphere calculations for simplices.
//!
//! This module provides functions for computing the circumcenter and circumradius
//! of simplices in d-dimensional space.

#![forbid(unsafe_code)]

use core::{array::from_fn, fmt, hint::cold_path};

use super::conversions::ValueConversionError;
use crate::geometry::matrix::{
    DEFAULT_SINGULAR_TOL, ExactF64Conversion, LaError, LaVector, Matrix, MatrixError,
    RationalMatrix, RationalVector, SingularityReason, StackMatrixDispatchError, matrix_set,
    rational_from_f64,
};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{
    CoordinateConversionError, CoordinateConversionValue, CoordinateValidationError,
};

/// Geometric measure involved in a degenerate simplex or facet calculation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DegenerateMeasure {
    /// One-dimensional length.
    Length,
    /// Two-dimensional area.
    Area,
    /// Full-dimensional volume.
    Volume,
    /// Sum of boundary facet measures.
    SurfaceArea,
}

impl fmt::Display for DegenerateMeasure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length => f.write_str("length"),
            Self::Area => f.write_str("area"),
            Self::Volume => f.write_str("volume"),
            Self::SurfaceArea => f.write_str("surface area"),
        }
    }
}

/// Geometric degeneracy category for simplex and facet measure failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DegenerateGeometry {
    /// Points coincide, producing zero length.
    CoincidentPoints,
    /// Points are collinear.
    CollinearPoints,
    /// Points are coplanar.
    CoplanarPoints,
    /// Points are collinear or coplanar; the Gram determinant cannot distinguish which.
    CollinearOrCoplanarPoints,
}

impl fmt::Display for DegenerateGeometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoincidentPoints => f.write_str("coincident points"),
            Self::CollinearPoints => f.write_str("collinear points"),
            Self::CoplanarPoints => f.write_str("coplanar points"),
            Self::CollinearOrCoplanarPoints => f.write_str("collinear or coplanar points"),
        }
    }
}

/// Structured reason a geometric measure is invalid or unrepresentable.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CircumcenterFailureReason {
    /// A simplex has zero measure because its points are degenerate.
    DegenerateSimplex {
        /// Measure that collapsed to zero.
        measure: DegenerateMeasure,
        /// Geometric degeneracy category.
        degeneracy: DegenerateGeometry,
    },
    /// A facet has zero measure because its points are degenerate.
    DegenerateFacet {
        /// Measure that collapsed to zero.
        measure: DegenerateMeasure,
        /// Geometric degeneracy category.
        degeneracy: DegenerateGeometry,
    },
    /// A Gram determinant was NaN or infinite.
    NonFiniteGramDeterminant,
    /// A Gram determinant was negative.
    NegativeGramDeterminant,
    /// A derived simplex measure was non-positive.
    NonPositiveSimplexMeasure {
        /// Measure that was expected to be positive.
        measure: DegenerateMeasure,
        /// Rejected measure value.
        value: CoordinateConversionValue,
    },
    /// A derived simplex or facet measure was NaN or infinite.
    NonFiniteMeasure {
        /// Measure that was expected to be finite.
        measure: DegenerateMeasure,
        /// Rejected measure value.
        value: CoordinateConversionValue,
    },
}

impl fmt::Display for CircumcenterFailureReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DegenerateSimplex {
                measure,
                degeneracy,
            } => write!(f, "degenerate simplex with zero {measure} ({degeneracy})"),
            Self::DegenerateFacet {
                measure,
                degeneracy,
            } => write!(f, "degenerate facet with zero {measure} ({degeneracy})"),
            Self::NonFiniteGramDeterminant => f.write_str("Gram determinant is non-finite"),
            Self::NegativeGramDeterminant => {
                f.write_str("Gram matrix has negative determinant (degenerate simplex)")
            }
            Self::NonPositiveSimplexMeasure { measure, value } => {
                write!(f, "degenerate simplex with {measure} ≈ {value}")
            }
            Self::NonFiniteMeasure { measure, value } => {
                write!(f, "{measure} calculation produced non-finite value {value}")
            }
        }
    }
}

/// Structured reason for array conversion failures.
#[derive(Clone, Copy, Debug, thiserror::Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum ArrayConversionFailureReason {
    /// The input length did not match the fixed-size target array.
    #[error("array length mismatch")]
    LengthMismatch,
}

/// Errors that can occur during circumcenter and simplex-measure calculations.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::CircumcenterError;
///
/// let err = CircumcenterError::EmptyPointSet;
/// std::assert_matches!(err, CircumcenterError::EmptyPointSet);
/// ```
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum CircumcenterError {
    /// Empty point set provided.
    #[error("Empty point set")]
    EmptyPointSet,

    /// Points do not form a valid simplex.
    #[error(
        "Points do not form a valid simplex: expected {expected} points for dimension {dimension}, got {actual}"
    )]
    InvalidSimplex {
        /// Number of points provided.
        actual: usize,
        /// Number of points required by the operation (`D + 1` for a full
        /// simplex, or `D` for a codimension-one facet).
        expected: usize,
        /// Dimension.
        dimension: usize,
    },

    /// The ambient dimension is below the minimum required by the operation.
    #[error("Ambient dimension {dimension} is below the required minimum {minimum}")]
    DimensionTooSmall {
        /// Ambient dimension of the input points.
        dimension: usize,
        /// Minimum ambient dimension supported by the operation.
        minimum: usize,
    },

    /// A simplex or facet measure is degenerate, invalid, or unrepresentable.
    ///
    /// Failures from matrix factorization or vector operations are reported
    /// separately as [`Self::LinearAlgebraFailure`].
    #[error("Invalid geometric measure: {reason}")]
    InvalidMeasure {
        /// Structured reason for rejecting the geometric measure.
        reason: CircumcenterFailureReason,
    },

    /// Runtime-dispatched stack matrix dimension is unsupported.
    #[error("Unsupported stack matrix dimension {requested} (maximum supported is {max})")]
    UnsupportedMatrixDimension {
        /// Requested matrix dimension.
        requested: usize,
        /// Maximum supported matrix dimension.
        max: usize,
    },

    /// Internal matrix dispatch requested an active block whose size does not
    /// match the concrete stack matrix.
    ///
    /// Public geometry APIs surface this as a typed error rather than silently
    /// classifying structurally invalid predicate state as degenerate geometry.
    #[error(
        "Active matrix block size {active} does not match concrete matrix dimension {matrix_dimension}"
    )]
    MatrixDimensionMismatch {
        /// Requested active matrix dimension.
        active: usize,
        /// Concrete matrix dimension.
        matrix_dimension: usize,
    },

    /// Linear algebra backend operation failed.
    #[error("Linear algebra failure: {source}")]
    LinearAlgebraFailure {
        /// Typed source error from the linear algebra backend.
        #[source]
        source: LaError,
    },

    /// Matrix operation failed while building or solving a geometry helper matrix.
    #[error("Matrix error: {source}")]
    MatrixError {
        /// Typed source error from matrix operations.
        #[from]
        source: MatrixError,
    },

    /// Array conversion failed.
    #[error("Array conversion failed: {reason}")]
    ArrayConversionFailed {
        /// Structured reason for the array conversion failure.
        reason: ArrayConversionFailureReason,
    },

    /// Coordinate conversion failed while preparing predicate or measure inputs.
    #[error("Coordinate conversion error: {source}")]
    CoordinateConversion {
        /// Typed source error from coordinate conversion.
        #[from]
        source: CoordinateConversionError,
    },

    /// Coordinate validation failed while constructing the circumcenter point.
    #[error("Coordinate validation error: {source}")]
    CoordinateValidation {
        /// Typed source error from coordinate validation.
        #[from]
        source: CoordinateValidationError,
    },

    /// Scalar value conversion failed while converting dimensions or derived measures.
    #[error("Value conversion error: {source}")]
    ValueConversion {
        /// Typed source error from value conversion.
        #[source]
        source: Box<ValueConversionError>,
    },
}

impl From<ValueConversionError> for CircumcenterError {
    fn from(source: ValueConversionError) -> Self {
        Self::ValueConversion {
            source: Box::new(source),
        }
    }
}

impl From<StackMatrixDispatchError> for CircumcenterError {
    fn from(source: StackMatrixDispatchError) -> Self {
        match source {
            StackMatrixDispatchError::UnsupportedDim { k, max } => {
                Self::UnsupportedMatrixDimension { requested: k, max }
            }
            StackMatrixDispatchError::ActiveBlockDimensionMismatch { k, dim } => {
                Self::MatrixDimensionMismatch {
                    active: k,
                    matrix_dimension: dim,
                }
            }
            StackMatrixDispatchError::La { source } => Self::LinearAlgebraFailure { source },
            StackMatrixDispatchError::Matrix { source } => Self::MatrixError { source },
        }
    }
}

impl From<LaError> for CircumcenterError {
    fn from(source: LaError) -> Self {
        Self::from(StackMatrixDispatchError::from(source))
    }
}

/// Calculate the circumcenter of a set of points forming a simplex.
///
/// The circumcenter is the unique point equidistant from all points of
/// the simplex. Returns an error if the points do not form a valid simplex or
/// if the computation fails due to degeneracy or numerical issues.
///
/// Using the approach from:
///
/// Lévy, Bruno, and Yang Liu.
/// "Lp Centroidal Voronoi Tessellation and Its Applications."
/// ACM Transactions on Graphics 29, no. 4 (July 26, 2010): 119:1-119:11.
/// <https://doi.org/10.1145/1778765.1778856>.
///
/// The circumcenter C of a simplex with points `x_0`, `x_1`, ..., `x_n` is the
/// solution to the system:
///
/// C = x₀ + 1/2 (A^-1*B)
///
/// Where:
///
/// A is a matrix (to be inverted) of the form:
///     (x_1-x0) for all coordinates in x1, x0
///     (x2-x0) for all coordinates in x2, x0
///     ... for all `x_n` in the simplex
///
/// These are the perpendicular bisectors of the edges of the simplex.
///
/// And:
///
/// B is a vector of the form:
///     ||x₁-x₀||²
///     ||x₂-x₀||²
///     ... for all `x_n` in the simplex
///
/// The resulting vector gives the coordinates of the circumcenter.
///
/// # Arguments
///
/// * `points` - A slice of points that form the simplex
///
/// # Returns
/// The circumcenter as a `Point<D>` if successful, or an error if the
/// simplex is degenerate or the matrix inversion fails.
///
/// # Errors
///
/// Returns an error if:
/// - The points do not form a valid simplex
/// - The matrix inversion fails due to degeneracy
/// - The computed center cannot be represented with finite coordinates
///
/// # Example
///
/// ```
/// use delaunay::prelude::geometry::{CircumcenterError, Coordinate, Point, circumcenter};
///
/// # fn main() -> Result<(), CircumcenterError> {
/// let point1 = Point::try_from([0.0, 0.0, 0.0])?;
/// let point2 = Point::try_from([1.0, 0.0, 0.0])?;
/// let point3 = Point::try_from([0.0, 1.0, 0.0])?;
/// let point4 = Point::try_from([0.0, 0.0, 1.0])?;
/// let points = vec![point1, point2, point3, point4];
/// let center = circumcenter(&points)?;
/// assert_eq!(center, Point::try_from([0.5, 0.5, 0.5])?);
/// # Ok(())
/// # }
/// ```
pub fn circumcenter<const D: usize>(points: &[Point<D>]) -> Result<Point<D>, CircumcenterError> {
    // LCOV_EXCL_START
    #[cfg(debug_assertions)]
    if std::env::var_os("DELAUNAY_DEBUG_UNUSED_IMPORTS").is_some() {
        tracing::debug!(
            "circumsphere::circumcenter called (points_len={}, D={})",
            points.len(),
            D
        );
    }
    // LCOV_EXCL_STOP
    if points.is_empty() {
        return Err(CircumcenterError::EmptyPointSet);
    }

    let dim = points.len() - 1;
    if dim != D {
        return Err(CircumcenterError::InvalidSimplex {
            actual: points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    // Build matrix A and vector b for the linear system A * x = b.
    //
    // Here, A is D×D and b is length D, so we can solve with stack-allocated la-stack types.
    let coords_0 = points[0].coords();

    let mut a = Matrix::<D>::zero();
    let mut b_arr = [0.0f64; D];
    let mut fast_system_is_finite = true;

    for i in 0..D {
        let coords_point = points[i + 1].coords();

        // Points already prove finite coordinates. Subtraction creates new
        // values, so parse the displacement once before using it in the system.
        let difference =
            match LaVector::<D>::try_new(from_fn(|axis| coords_point[axis] - coords_0[axis])) {
                Ok(difference) => difference,
                Err(LaError::NonFinite { .. }) => {
                    fast_system_is_finite = false;
                    continue;
                }
                Err(source) => return Err(source.into()),
            };
        for (j, &coordinate) in difference.as_array().iter().enumerate() {
            matrix_set(&mut a, i, j, coordinate)?;
        }

        // Rounded squared distances belong only to the finite fast system.
        // On overflow, rebuild from the original coordinates in exact arithmetic.
        match difference.norm_squared() {
            Ok(squared_distance) => b_arr[i] = squared_distance,
            Err(LaError::NonFinite { .. }) => fast_system_is_finite = false,
            Err(source) => return Err(source.into()),
        }
    }

    // Solve for x, then C = x0 + 1/2 * x.
    //
    // Fast path: LU factorization with la-stack's default pivot tolerance.
    // Exact fallback: when LU rejects the matrix as near-singular, use
    // `RationalMatrix::solve` (fraction-free elimination, then explicit final
    // coordinate rounding) for a robust result. This replaces the old `lu(0.0)`
    // zero-tolerance fallback, which could silently accept truly singular matrices.
    let circumcenter_coords = if fast_system_is_finite {
        let b_vec = LaVector::<D>::try_new(b_arr)?;
        match a.lu(DEFAULT_SINGULAR_TOL) {
            Ok(lu) => {
                let solution = lu
                    .solve(b_vec)
                    .map_err(CircumcenterError::from)?
                    .into_array();
                from_fn(|index| 0.5_f64.mul_add(solution[index], coords_0[index]))
            }
            Err(LaError::Singular {
                reason: SingularityReason::Numerical { .. },
                ..
            }) => {
                // Exact-arithmetic fallback: LU rejected the system as
                // near-singular, so we use la-stack's rational-input solve.
                // This path is cold — well-conditioned simplices return above.
                cold_path();
                // LCOV_EXCL_START
                #[cfg(debug_assertions)]
                if std::env::var_os("DELAUNAY_DEBUG_LU_FALLBACK").is_some() {
                    tracing::debug!(
                        "circumcenter<{D}>: LU near-singular, forming and solving the system rationally"
                    );
                }
                // LCOV_EXCL_STOP
                exact_circumcenter_from_points(points)?
            }
            Err(e) => {
                cold_path();
                return Err(e.into());
            }
        }
    } else {
        // Finite source coordinates can still overflow while subtracting or
        // squaring in binary64. Build the cold-path system from the source
        // coordinates instead of publishing an error from a rounded
        // intermediate.
        cold_path();
        exact_circumcenter_from_points(points)?
    };

    Ok(Point::try_new(circumcenter_coords)?)
}

/// Forms and solves the circumcenter system entirely in exact rational arithmetic.
fn exact_circumcenter_from_points<const D: usize>(
    points: &[Point<D>],
) -> Result<[f64; D], CircumcenterError> {
    // Point construction already proves finiteness, and every finite binary64
    // coordinate has an exact rational representation.
    let reference = points[0].coords().map(|coordinate| {
        rational_from_f64(coordinate).expect("validated Point coordinates are finite")
    });
    let zero = rational_from_f64(0.0).expect("zero is finite");
    let half = rational_from_f64(0.5).expect("one half is finite");
    let mut matrix = RationalMatrix::<D>::zero();
    let mut rhs = from_fn(|_| zero.clone());

    for row in 0..D {
        for (column, origin) in reference.iter().enumerate() {
            let coordinate = rational_from_f64(points[row + 1].coords()[column])
                .expect("validated Point coordinates are finite");
            let relative = coordinate - origin;
            rhs[row] += relative.clone() * relative.clone();
            matrix.set(row, column, relative)?;
        }
    }

    let solution = matrix.solve(&RationalVector::try_new(rhs)?)?.into_array();
    let center = RationalVector::<D>::try_from_fn(|index| {
        reference[index].clone() + half.clone() * solution[index].clone()
    })?;
    Ok(center.to_rounded_f64()?.into_array())
}

/// Calculate the circumradius of a set of points forming a simplex.
///
/// The circumradius is the distance from the circumcenter to any point of the simplex.
///
/// # Arguments
///
/// * `points` - A slice of points that form the simplex
///
/// # Returns
/// The finite circumradius in coordinate units if successful, or an error if
/// the center or radius computation fails.
///
/// # Errors
///
/// Propagates errors from [`circumcenter`] and [`circumradius_with_center`].
/// Even when the center is finite, an unrepresentable center-to-vertex
/// displacement or norm returns [`CircumcenterError::LinearAlgebraFailure`]
/// with the underlying [`LaError`].
///
/// # Example
///
/// ```
/// use delaunay::prelude::geometry::{CircumcenterError, Coordinate, Point, circumradius};
/// use approx::assert_relative_eq;
///
/// # fn main() -> Result<(), CircumcenterError> {
/// let point1 = Point::try_from([0.0, 0.0, 0.0])?;
/// let point2 = Point::try_from([1.0, 0.0, 0.0])?;
/// let point3 = Point::try_from([0.0, 1.0, 0.0])?;
/// let point4 = Point::try_from([0.0, 0.0, 1.0])?;
/// let points = vec![point1, point2, point3, point4];
/// let radius = circumradius(&points)?;
/// let expected_radius = (3.0_f64.sqrt() / 2.0);
/// assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
/// # Ok(())
/// # }
/// ```
pub fn circumradius<const D: usize>(points: &[Point<D>]) -> Result<f64, CircumcenterError> {
    let circumcenter = circumcenter(points)?;
    circumradius_with_center(points, &circumcenter)
}

/// Calculate the circumradius given a precomputed circumcenter.
///
/// This is a helper function that calculates the circumradius when the circumcenter
/// is already known, avoiding redundant computation.
/// It measures the distance to `points[0]` and accepts any nonempty slice. The
/// caller must supply the correct center to interpret this distance as a
/// circumradius; this function does not check simplex arity, nondegeneracy, or
/// equidistance from the remaining points.
///
/// # Arguments
///
/// * `points` - A nonempty slice whose first point supplies the radius measurement
/// * `circumcenter` - The precomputed circumcenter
///
/// # Returns
/// The finite non-negative `f64` distance in coordinate units. The result is
/// zero when the first point equals the supplied center.
///
/// # Errors
///
/// Returns an error if:
/// - [`CircumcenterError::EmptyPointSet`] if the points slice is empty.
/// - [`CircumcenterError::LinearAlgebraFailure`] if a displacement or its norm
///   is not representable, preserving the underlying [`LaError`].
///
/// # Example
///
/// ```
/// use delaunay::prelude::geometry::{
///     CircumcenterError, Coordinate, Point, circumcenter, circumradius_with_center,
/// };
/// use approx::assert_relative_eq;
///
/// # fn main() -> Result<(), CircumcenterError> {
/// let point1 = Point::try_from([0.0, 0.0, 0.0])?;
/// let point2 = Point::try_from([1.0, 0.0, 0.0])?;
/// let point3 = Point::try_from([0.0, 1.0, 0.0])?;
/// let point4 = Point::try_from([0.0, 0.0, 1.0])?;
/// let points = vec![point1, point2, point3, point4];
/// let center = circumcenter(&points)?;
/// let radius = circumradius_with_center(&points, &center)?;
/// let expected_radius = (3.0_f64.sqrt() / 2.0);
/// assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
/// # Ok(())
/// # }
/// ```
pub fn circumradius_with_center<const D: usize>(
    points: &[Point<D>],
    circumcenter: &Point<D>,
) -> Result<f64, CircumcenterError> {
    if points.is_empty() {
        return Err(CircumcenterError::EmptyPointSet);
    }

    let point_coords = points[0].coords();
    let circumcenter_coords = circumcenter.coords();

    // The distance may be finite even when its square is unrepresentable.
    let mut diff_coords = [0.0; D];
    for i in 0..D {
        diff_coords[i] = circumcenter_coords[i] - point_coords[i];
    }
    Ok(LaVector::try_new(diff_coords)?.norm()?)
}

#[cfg(test)]
mod tests {
    use std::assert_matches;

    use approx::assert_relative_eq;
    use la_stack::UnrepresentableReason;

    use super::*;
    use crate::geometry::matrix::BigRational;
    use crate::geometry::util::conversions::{ValueConversionFailureReason, safe_usize_to_scalar};

    /// Known lengths exercise the geometry boundary independently of the backend norm.
    fn assert_stable_radius<const D: usize>() {
        let origin = Point::try_new([0.0; D]).unwrap();
        for scale in [1.0e-200, 1.0, 1.0e200] {
            let mut coords = [0.0; D];
            coords[0] = -3.0 * scale;
            coords[1] = 4.0 * scale;
            let point = Point::try_new(coords).unwrap();
            let radius = circumradius_with_center(&[point], &origin).unwrap();
            assert_relative_eq!(radius / scale, 5.0, max_relative = 1e-14);
        }
        for (large, small) in [
            (1.0e200, 1.0e-200),
            (f64::MIN_POSITIVE / 2.0, 0.0),
            (f64::MAX, 0.0),
        ] {
            let mut coords = [small; D];
            coords[0] = large;
            let point = Point::try_new(coords).unwrap();
            let radius = circumradius_with_center(&[point], &origin).unwrap();
            assert_relative_eq!(radius / large, 1.0, max_relative = 1e-14);
        }
        assert_relative_eq!(circumradius_with_center(&[origin], &origin).unwrap(), 0.0);

        // Orthogonal equal axes have center (scale/2, ..., scale/2), including
        // when relative squares underflow or overflow in the fast system.
        for scale in [1.0e-200, 1.0, 1.0e200] {
            let mut simplex = vec![origin];
            for axis in 0..D {
                let mut coords = [0.0; D];
                coords[axis] = scale;
                simplex.push(Point::try_new(coords).unwrap());
            }
            let center = circumcenter(&simplex).unwrap();
            for coordinate in center.coords() {
                assert_relative_eq!(coordinate / scale, 0.5, max_relative = 1e-14);
            }
            let expected = safe_usize_to_scalar(D).unwrap().sqrt() / 2.0;
            assert_relative_eq!(
                circumradius(&simplex).unwrap() / scale,
                expected,
                max_relative = 1e-14
            );
        }
    }

    /// Exact equal squared distances certify the cold-path solve independently
    /// of the matrix backend, including cancellation of enormous relative offsets.
    fn assert_rational_circumcenter_round_trip<const D: usize>() {
        for (origin, opposite, expected) in [(1.25, -0.75, 0.25), (-1.0e308, 1.0e308, 0.0)] {
            let mut points = vec![Point::try_new([origin; D]).unwrap()];
            for axis in 0..D {
                let mut coords = [origin; D];
                coords[axis] = opposite;
                points.push(Point::try_new(coords).unwrap());
            }
            points.swap(1, D);
            let center = exact_circumcenter_from_points(&points).unwrap();
            for coordinate in center {
                assert_relative_eq!(coordinate, expected);
            }
            let squared_distances: Vec<_> = points
                .iter()
                .map(|point| {
                    point
                        .coords()
                        .iter()
                        .zip(center)
                        .map(|(&coordinate, center)| {
                            let delta = rational_from_f64(coordinate).unwrap()
                                - rational_from_f64(center).unwrap();
                            &delta * &delta
                        })
                        .sum::<BigRational>()
                })
                .collect();
            assert!(
                squared_distances
                    .iter()
                    .all(|distance| *distance == squared_distances[0])
            );
            if origin.abs() > 1.0e300 {
                let public_center = circumcenter(&points).unwrap();
                for coordinate in public_center.coords() {
                    assert_relative_eq!(*coordinate, expected);
                }
            }
        }
    }

    macro_rules! rational_circumcenter_test {
        ($dimension:literal) => {
            pastey::paste! {
                #[test]
                fn [<rational_circumcenter_round_trip_ $dimension d>]() {
                    assert_rational_circumcenter_round_trip::<$dimension>();
                }
                #[test]
                fn [<stable_radius_ $dimension d>]() {
                    assert_stable_radius::<$dimension>();
                }
            }
        };
    }

    rational_circumcenter_test!(2);
    rational_circumcenter_test!(3);
    rational_circumcenter_test!(4);
    rational_circumcenter_test!(5);
    rational_circumcenter_test!(6);

    #[test]
    fn rational_circumcenter_reports_unrepresentable_coordinate() {
        let points = [
            Point::try_new([0.0, 0.0]).unwrap(),
            Point::try_new([1.0, 0.0]).unwrap(),
            Point::try_new([0.5, f64::from_bits(1)]).unwrap(),
        ];
        assert_matches!(
            circumcenter(&points),
            Err(CircumcenterError::LinearAlgebraFailure {
                source: LaError::Unrepresentable {
                    index: Some(1),
                    reason: UnrepresentableReason::NotFinite,
                    ..
                },
            })
        );
    }

    #[test]
    fn circumcenter_error_display_names_variants() {
        let empty_error = CircumcenterError::EmptyPointSet;
        let display = format!("{empty_error}");
        assert!(display.contains("Empty point set"));

        let simplex_error = CircumcenterError::InvalidSimplex {
            actual: 2,
            expected: 3,
            dimension: 2,
        };
        let display = format!("{simplex_error}");
        assert!(display.contains("Points do not form a valid simplex"));
    }

    #[test]
    fn degenerate_measure_display_names_all_variants() {
        assert_eq!(DegenerateMeasure::Length.to_string(), "length");
        assert_eq!(DegenerateMeasure::Area.to_string(), "area");
        assert_eq!(DegenerateMeasure::Volume.to_string(), "volume");
        assert_eq!(DegenerateMeasure::SurfaceArea.to_string(), "surface area");
    }

    #[test]
    fn degenerate_geometry_display_names_all_variants() {
        assert_eq!(
            DegenerateGeometry::CoincidentPoints.to_string(),
            "coincident points"
        );
        assert_eq!(
            DegenerateGeometry::CollinearPoints.to_string(),
            "collinear points"
        );
        assert_eq!(
            DegenerateGeometry::CoplanarPoints.to_string(),
            "coplanar points"
        );
        assert_eq!(
            DegenerateGeometry::CollinearOrCoplanarPoints.to_string(),
            "collinear or coplanar points"
        );
    }

    #[test]
    fn circumcenter_failure_reason_display_preserves_typed_payloads() {
        let degenerate_simplex = CircumcenterFailureReason::DegenerateSimplex {
            measure: DegenerateMeasure::Volume,
            degeneracy: DegenerateGeometry::CoplanarPoints,
        };
        assert_eq!(
            degenerate_simplex.to_string(),
            "degenerate simplex with zero volume (coplanar points)"
        );

        let degenerate_facet = CircumcenterFailureReason::DegenerateFacet {
            measure: DegenerateMeasure::Length,
            degeneracy: DegenerateGeometry::CoincidentPoints,
        };
        assert_eq!(
            degenerate_facet.to_string(),
            "degenerate facet with zero length (coincident points)"
        );

        assert_eq!(
            CircumcenterFailureReason::NonFiniteGramDeterminant.to_string(),
            "Gram determinant is non-finite"
        );
        assert_eq!(
            CircumcenterFailureReason::NegativeGramDeterminant.to_string(),
            "Gram matrix has negative determinant (degenerate simplex)"
        );
        assert_eq!(
            CircumcenterFailureReason::NonPositiveSimplexMeasure {
                measure: DegenerateMeasure::SurfaceArea,
                value: CoordinateConversionValue::from_f64(0.0),
            }
            .to_string(),
            "degenerate simplex with surface area ≈ 0.0"
        );
        assert_eq!(
            CircumcenterFailureReason::NonFiniteMeasure {
                measure: DegenerateMeasure::Volume,
                value: CoordinateConversionValue::from_f64(f64::INFINITY),
            }
            .to_string(),
            "volume calculation produced non-finite value inf"
        );
    }

    #[test]
    fn circumcenter_error_conversions_preserve_typed_payloads() {
        let value_error = ValueConversionError::ConversionFailed {
            value: CoordinateConversionValue::from_usize(4),
            from_type: "usize",
            to_type: "f64",
            reason: ValueConversionFailureReason::TargetTypeRejected,
        };
        assert_matches!(
            CircumcenterError::from(value_error),
            CircumcenterError::ValueConversion { source }
                if matches!(
                    *source,
                    ValueConversionError::ConversionFailed {
                        value: CoordinateConversionValue::UnsignedInteger(4),
                        from_type: "usize",
                        to_type: "f64",
                        reason: ValueConversionFailureReason::TargetTypeRejected,
                    }
                )
        );

        assert_eq!(
            CircumcenterError::from(StackMatrixDispatchError::ActiveBlockDimensionMismatch {
                k: 4,
                dim: 3,
            }),
            CircumcenterError::MatrixDimensionMismatch {
                active: 4,
                matrix_dimension: 3,
            }
        );
    }

    #[test]
    fn predicates_circumcenter() {
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let center = circumcenter(&points).unwrap();

        assert_eq!(
            center,
            Point::try_new([0.5, 0.5, 0.5]).expect("finite point coordinates")
        );
    }

    #[test]
    fn predicates_circumcenter_fail() {
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
        ];
        let center = circumcenter(&points);

        assert!(center.is_err());
    }

    #[test]
    fn predicates_circumradius() {
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let radius = circumradius(&points).unwrap();
        let expected_radius: f64 = 3.0_f64.sqrt() / 2.0;

        assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
    }

    #[test]
    fn circumradius_rejects_unrepresentable_finite_tetrahedron_radius() {
        let magnitude = 1.1e308;
        let points = vec![
            Point::try_new([-magnitude, -magnitude, -magnitude]).expect("finite point coordinates"),
            Point::try_new([magnitude, -magnitude, -magnitude]).expect("finite point coordinates"),
            Point::try_new([-magnitude, magnitude, -magnitude]).expect("finite point coordinates"),
            Point::try_new([-magnitude, -magnitude, magnitude]).expect("finite point coordinates"),
        ];

        assert_matches!(
            circumradius(&points),
            Err(CircumcenterError::LinearAlgebraFailure {
                source: LaError::NonFinite { .. },
            })
        );
    }

    #[test]
    fn predicates_circumcenter_2d() {
        let points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([2.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 2.0]).expect("finite point coordinates"),
        ];
        let center = circumcenter(&points).unwrap();

        // For this triangle, circumcenter should be at (1.0, 0.75)
        assert_relative_eq!(center.coords()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(center.coords()[1], 0.75, epsilon = 1e-10);
    }

    #[test]
    fn test_circumradius_with_center_empty_point_set() {
        // Hits the `points.is_empty()` early-return branch in
        // `circumradius_with_center` (previously only exercised by
        // `circumcenter`).
        let points: Vec<Point<3>> = Vec::new();
        let center = Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates");
        match circumradius_with_center(&points, &center) {
            Err(CircumcenterError::EmptyPointSet) => {}
            other => panic!("expected EmptyPointSet, got {other:?}"),
        }
    }

    #[test]
    fn predicates_circumradius_2d() {
        let points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
        ];
        let radius = circumradius(&points).unwrap();

        // For a right triangle with legs of length 1, circumradius is sqrt(2)/2
        let expected_radius = 2.0_f64.sqrt() / 2.0;
        assert_relative_eq!(radius, expected_radius, epsilon = 1e-10);
    }

    #[test]
    fn predicates_circumradius_with_center() {
        // Test the circumradius_with_center function
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        let center = circumcenter(&points).unwrap();
        let radius_with_center = circumradius_with_center(&points, &center);
        let radius_direct = circumradius(&points).unwrap();

        assert_relative_eq!(radius_with_center.unwrap(), radius_direct, epsilon = 1e-10);
    }

    #[test]
    fn test_circumcenter_regular_simplex_3d() {
        // Test with a regular tetrahedron - use simpler vertices
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 3.0_f64.sqrt() / 2.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 3.0_f64.sqrt() / 6.0, (2.0 / 3.0_f64).sqrt()])
                .expect("finite point coordinates"),
        ];
        let center = circumcenter(&points).unwrap();

        // For this tetrahedron, verify circumcenter exists and is finite
        let center_coords = center.coords();
        for coord in center_coords {
            assert!(
                coord.is_finite(),
                "Circumcenter coordinates should be finite"
            );
        }

        // Verify all points are equidistant from circumcenter
        let mut distances = points.iter().map(|p| {
            let p_coords = *p.coords();
            let diff = [
                p_coords[0] - center_coords[0],
                p_coords[1] - center_coords[1],
                p_coords[2] - center_coords[2],
            ];
            LaVector::try_new(diff).unwrap().norm().unwrap()
        });

        // All distances should be equal
        let first_distance = distances.next().expect("fixture has vertices");
        for distance in distances {
            assert_relative_eq!(first_distance, distance, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_circumcenter_regular_simplex_4d() {
        // Test 4D simplex - use orthonormal basis plus origin
        let points: Vec<Point<4>> = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let center = circumcenter(&points).unwrap();

        // For this symmetric configuration, circumcenter should be at equal coordinates
        let center_coords = center.coords();
        for &coord in center_coords {
            assert!(
                coord.is_finite(),
                "Circumcenter coordinates should be finite"
            );
            // Should be around 0.5 for this configuration
            assert_relative_eq!(coord, 0.5, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_circumcenter_right_triangle_2d() {
        // Test with right triangle - circumcenter should be at hypotenuse midpoint
        let points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([4.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 3.0]).expect("finite point coordinates"),
        ];
        let center = circumcenter(&points).unwrap();

        // For right triangle, circumcenter is at midpoint of hypotenuse
        let center_coords = center.coords();
        assert_relative_eq!(center_coords[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(center_coords[1], 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_circumcenter_scaled_simplex() {
        // Test that scaling preserves circumcenter properties
        let scale = 10.0;
        let points = vec![
            Point::try_new([0.0 * scale, 0.0 * scale, 0.0 * scale])
                .expect("finite point coordinates"),
            Point::try_new([1.0 * scale, 0.0 * scale, 0.0 * scale])
                .expect("finite point coordinates"),
            Point::try_new([0.0 * scale, 1.0 * scale, 0.0 * scale])
                .expect("finite point coordinates"),
            Point::try_new([0.0 * scale, 0.0 * scale, 1.0 * scale])
                .expect("finite point coordinates"),
        ];
        let center = circumcenter(&points).unwrap();

        // Scaled simplex should have scaled circumcenter
        let expected_center = Point::try_new([0.5 * scale, 0.5 * scale, 0.5 * scale])
            .expect("finite point coordinates");
        let center_coords = center.coords();
        let expected_coords = expected_center.coords();

        for i in 0..3 {
            assert_relative_eq!(center_coords[i], expected_coords[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_circumcenter_translated_simplex() {
        // Test that translation preserves relative circumcenter position
        let translation = [10.0, 20.0, 30.0];
        let points = vec![
            Point::try_new([
                0.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                1.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                0.0 + translation[0],
                1.0 + translation[1],
                0.0 + translation[2],
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                0.0 + translation[0],
                0.0 + translation[1],
                1.0 + translation[2],
            ])
            .expect("finite point coordinates"),
        ];
        let center = circumcenter(&points).unwrap();

        // Get the circumcenter of the untranslated simplex for comparison
        let untranslated_points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let untranslated_center = circumcenter(&untranslated_points).unwrap();

        // Translated circumcenter should be untranslated circumcenter + translation
        let center_coords = center.coords();
        let untranslated_coords = untranslated_center.coords();

        for i in 0..3 {
            assert_relative_eq!(
                center_coords[i],
                untranslated_coords[i] + translation[i],
                epsilon = 1e-9
            );
        }

        // Also verify the expected absolute values for this specific tetrahedron
        let expected = [10.5, 20.5, 30.5];
        for i in 0..3 {
            assert_relative_eq!(center_coords[i], expected[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_circumcenter_nearly_degenerate_simplex() {
        // Test with points that are nearly collinear (may succeed or fail gracefully)
        let eps = 1e-3; // Use larger epsilon for more robustness
        let points: Vec<Point<3>> = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, eps, 0.0]).expect("finite point coordinates"), // Slightly off the line
            Point::try_new([0.5, 0.0, eps]).expect("finite point coordinates"), // Slightly off the plane
        ];

        let result = circumcenter(&points);
        // Should either succeed or fail gracefully (don't require success)
        if let Ok(center) = result {
            // If it succeeds, center should have finite coordinates
            let coords = center.coords();
            assert!(
                coords.iter().all(|&x| x.is_finite()),
                "Circumcenter coordinates should be finite"
            );
        } else {
            // If it fails, that's acceptable for this nearly degenerate case
        }
    }

    #[test]
    fn test_circumcenter_empty_points() {
        let points: Vec<Point<3>> = vec![];
        let result = circumcenter(&points);

        assert!(result.is_err());
        match result.unwrap_err() {
            CircumcenterError::EmptyPointSet => {}
            other => panic!("Expected EmptyPointSet error, got: {other:?}"),
        }
    }

    #[test]
    fn test_circumcenter_wrong_dimension() {
        // Test with 2 points for 3D (need 4 points for 3D circumcenter)
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
        ];
        let result = circumcenter(&points);

        assert!(result.is_err());
        match result.unwrap_err() {
            CircumcenterError::InvalidSimplex {
                actual,
                expected,
                dimension,
            } => {
                assert_eq!(actual, 2);
                assert_eq!(expected, 4); // D + 1 where D = 3
                assert_eq!(dimension, 3);
            }
            other => panic!("Expected InvalidSimplex error, got: {other:?}"),
        }
    }

    #[test]
    fn test_circumcenter_equilateral_triangle_properties() {
        // Test that circumcenter has expected properties for equilateral triangle
        let side_length = 2.0;
        let height = side_length * 3.0_f64.sqrt() / 2.0;

        let points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([side_length, 0.0]).expect("finite point coordinates"),
            Point::try_new([side_length / 2.0, height]).expect("finite point coordinates"),
        ];

        let center = circumcenter(&points).unwrap();
        let center_coords = center.coords();

        // For equilateral triangle, circumcenter should be at centroid
        let expected_x = side_length / 2.0;
        let expected_y = height / 3.0;

        assert_relative_eq!(center_coords[0], expected_x, epsilon = 1e-10);
        assert_relative_eq!(center_coords[1], expected_y, epsilon = 1e-10);

        // Verify all vertices are equidistant from circumcenter
        let mut distances = points.iter().map(|p| {
            let p_coords = *p.coords();
            let diff = [
                p_coords[0] - center_coords[0],
                p_coords[1] - center_coords[1],
            ];
            LaVector::try_new(diff).unwrap().norm().unwrap()
        });

        // All distances should be equal
        let first_distance = distances.next().expect("fixture has vertices");
        for distance in distances {
            assert_relative_eq!(first_distance, distance, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_circumcenter_numerical_stability() {
        // Test with points that could cause numerical instability
        let points: Vec<Point<2>> = vec![
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.000_000_1, 0.0]).expect("finite point coordinates"), // Very close to first point
            Point::try_new([1.000_000_1, 0.000_000_1]).expect("finite point coordinates"), // Forms very thin triangle
        ];

        let result = circumcenter(&points);
        // Should either succeed or fail gracefully (not panic)
        if let Ok(center) = result {
            // If it succeeds, center should have finite coordinates
            let coords = center.coords();
            assert!(
                coords.iter().all(|&x| x.is_finite()),
                "Circumcenter coordinates should be finite"
            );
        } else {
            // If it fails, that's acceptable for this degenerate case
        }
    }

    #[test]
    fn test_circumcenter_1d_case() {
        // Test 1D case (2 points)
        let points = vec![
            Point::try_new([0.0]).expect("finite point coordinates"),
            Point::try_new([2.0]).expect("finite point coordinates"),
        ];

        let center = circumcenter(&points).unwrap();
        let center_coords = center.coords();

        // 1D circumcenter should be at midpoint
        assert_relative_eq!(center_coords[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_circumcenter_high_dimension() {
        // Test higher dimensional case (5D)
        let points: Vec<Point<5>> = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];

        let result = circumcenter(&points);
        assert!(result.is_ok(), "5D circumcenter should work");

        let center = result.unwrap();
        let center_coords = center.coords();

        // Verify circumcenter has finite coordinates
        for coord in center_coords {
            assert!(
                coord.is_finite(),
                "Circumcenter coordinates should be finite"
            );
        }

        // For this configuration, all points are equidistant from circumcenter
        // Verify all points are at same distance from circumcenter
        let mut distances = points.iter().map(|p| {
            let p_coords = *p.coords();
            let diff = [
                p_coords[0] - center_coords[0],
                p_coords[1] - center_coords[1],
                p_coords[2] - center_coords[2],
                p_coords[3] - center_coords[3],
                p_coords[4] - center_coords[4],
            ];
            LaVector::try_new(diff).unwrap().norm().unwrap()
        });

        // All distances should be equal
        let first_distance = distances.next().expect("fixture has vertices");
        for distance in distances {
            assert_relative_eq!(first_distance, distance, epsilon = 1e-9);
        }
    }

    #[test]
    fn predicates_circumcenter_precise_values() {
        // Test with precisely known circumcenter values
        // Using a simplex where we can calculate the circumcenter analytically
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([6.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 8.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 10.0]).expect("finite point coordinates"),
        ];

        let center = circumcenter(&points).unwrap();
        let center_coords = center.coords();

        // For this configuration, circumcenter should be at (3, 4, 5)
        assert_relative_eq!(center_coords[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(center_coords[1], 4.0, epsilon = 1e-10);
        assert_relative_eq!(center_coords[2], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_circumcenter_empty_point_set() {
        let empty_points: Vec<Point<3>> = vec![];
        let result = circumcenter(&empty_points);

        assert_matches!(result, Err(CircumcenterError::EmptyPointSet));
    }

    #[test]
    fn test_circumcenter_invalid_simplex() {
        // Test wrong number of points for dimension
        let points_2d = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            // Missing third point for 2D circumcenter
        ];

        let result = circumcenter(&points_2d);
        assert_matches!(result, Err(CircumcenterError::InvalidSimplex { .. }));

        // Test too many points
        let points_extra = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 0.5]).expect("finite point coordinates"), // Extra point for 2D
        ];

        let result = circumcenter(&points_extra);
        assert_matches!(result, Err(CircumcenterError::InvalidSimplex { .. }));
    }

    #[test]
    fn test_circumcenter_degenerate_matrix() {
        // Test collinear points in 2D (should cause matrix inversion to fail)
        let collinear_points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([2.0, 0.0]).expect("finite point coordinates"), // Collinear with first two
        ];

        let result = circumcenter(&collinear_points);
        assert_matches!(
            result,
            Err(CircumcenterError::LinearAlgebraFailure {
                source: LaError::Singular { .. }
            })
        );
    }

    #[test]
    fn test_circumcenter_exact_fallback_near_singular_3d() {
        // Near-degenerate tetrahedron: three vertices nearly coplanar with a
        // tiny perturbation off the plane.  The resulting linear system is
        // ill-conditioned enough to trip DEFAULT_SINGULAR_TOL, exercising the
        // rationally formed fallback path.
        let eps = 1e-14; // Perturbation small enough to make LU reject
        let points: Vec<Point<3>> = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 0.5, eps]).expect("finite point coordinates"), // Barely off the z=0 plane
        ];

        let system = Matrix::try_from_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, eps]])
            .expect("finite system matrix");
        assert_matches!(
            system.lu(DEFAULT_SINGULAR_TOL),
            Err(LaError::Singular {
                reason: SingularityReason::Numerical { .. },
                ..
            })
        );

        let result = circumcenter(&points);
        // The exact solver should succeed where LU alone would fail or
        // produce inaccurate results.
        let center = result.expect("exact fallback should handle near-singular system");
        let center_coords = center.coords();

        // All coordinates must be finite
        assert!(
            center_coords.iter().all(|&x| x.is_finite()),
            "Circumcenter coordinates should be finite"
        );

        // Verify equidistance: all vertices should be the same distance
        // from the circumcenter.
        let mut distances = points.iter().map(|p| {
            let diff = [
                p.coords()[0] - center_coords[0],
                p.coords()[1] - center_coords[1],
                p.coords()[2] - center_coords[2],
            ];
            LaVector::try_new(diff).unwrap().norm().unwrap()
        });

        let first_distance = distances.next().expect("fixture has vertices");
        for distance in distances {
            assert_relative_eq!(first_distance, distance, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_circumcenter_exact_fallback_near_singular_2d() {
        // Near-degenerate triangle: two vertices very close together.
        // The system matrix has a row with tiny entries, likely tripping
        // DEFAULT_SINGULAR_TOL.
        let eps = 1e-15;
        let points: Vec<Point<2>> = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, eps]).expect("finite point coordinates"), // Nearly collinear
        ];

        let result = circumcenter(&points);
        let center = result.expect("exact fallback should handle near-singular 2D system");
        let center_coords = center.coords();

        assert!(
            center_coords.iter().all(|&x| x.is_finite()),
            "Circumcenter coordinates should be finite"
        );

        // x-coordinate should be near 0.5 (midpoint of base edge)
        assert_relative_eq!(center_coords[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_circumcenter_rational_system_handles_finite_difference_overflow() {
        let points = [
            Point::try_new([-1.0e308, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0e308, 0.0]).expect("finite point coordinates"),
            Point::try_new([-1.0e308, 2.0]).expect("finite point coordinates"),
        ];

        let center = circumcenter(&points)
            .expect("finite source coordinates should bypass overflowing f64 differences");
        assert_relative_eq!(center.coords()[0], 0.0);
        assert_relative_eq!(center.coords()[1], 1.0);
    }
}
