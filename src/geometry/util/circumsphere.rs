//! Circumsphere calculations for simplices.
//!
//! This module provides functions for computing the circumcenter and circumradius
//! of simplices in d-dimensional space.

#![forbid(unsafe_code)]

use super::conversions::{ValueConversionError, safe_coords_to_f64};
use super::norms::{hypot, squared_norm};
use crate::geometry::matrix::{
    DEFAULT_SINGULAR_TOL, LaError, LaVector, Matrix, MatrixError, StackMatrixDispatchError,
    matrix_set,
};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateConversionValue};
use core::{fmt, hint::cold_path};

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

/// Structured reason for matrix-inversion or measure-degeneracy failures.
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

/// Errors that can occur during circumcenter calculation.
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
        /// Number of points expected (`D + 1`).
        expected: usize,
        /// Dimension.
        dimension: usize,
    },

    /// Matrix inversion failed (degenerate simplex).
    #[error("Matrix inversion failed: {reason}")]
    MatrixInversionFailed {
        /// Structured reason for the matrix inversion failure.
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
/// C = 1/2 (A^-1*B)
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
///     (x_1^2-x0^2) for all coordinates in x1, x0
///     (x_2^2-x0^2) for all coordinates in x2, x0
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
/// - Array conversion fails
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

    // Use safe coordinate conversion
    let coords_0_f64: [f64; D] = safe_coords_to_f64(coords_0)?;

    let mut a = Matrix::<D>::zero();
    let mut b_arr = [0.0f64; D];

    for i in 0..D {
        let coords_point = points[i + 1].coords();

        // Use safe coordinate conversion
        let coords_point_f64: [f64; D] = safe_coords_to_f64(coords_point)?;

        // Fill matrix row
        for j in 0..D {
            matrix_set(&mut a, i, j, coords_point_f64[j] - coords_0_f64[j])?;
        }

        // Calculate squared distance using squared_norm for consistency
        let mut diff_coords = [0.0; D];
        for j in 0..D {
            diff_coords[j] = coords_point_f64[j] - coords_0_f64[j];
        }
        b_arr[i] = squared_norm(&diff_coords);
    }

    // Solve for x, then C = x0 + 1/2 * x.
    //
    // Fast path: LU factorization with la-stack's default pivot tolerance.
    // Exact fallback: when LU rejects the matrix as near-singular, use
    // `solve_exact_rounded_f64` (BigRational Gaussian elimination, then explicit
    // finite f64 rounding) for a robust result. This replaces the old `lu(0.0)`
    // zero-tolerance fallback, which could silently accept truly singular matrices.
    let b_vec = LaVector::<D>::try_new(b_arr)?;
    let x = match a.lu(DEFAULT_SINGULAR_TOL) {
        Ok(lu) => lu
            .solve(b_vec)
            .map_err(CircumcenterError::from)?
            .into_array(),
        Err(LaError::Singular { .. }) => {
            // Exact-arithmetic fallback: LU rejected the system as
            // near-singular, so we pay for BigRational Gaussian elimination.
            // This path is cold — well-conditioned simplices return above.
            cold_path();
            // LCOV_EXCL_START
            #[cfg(debug_assertions)]
            if std::env::var_os("DELAUNAY_DEBUG_LU_FALLBACK").is_some() {
                tracing::debug!(
                    "circumcenter<{D}>: LU near-singular, using solve_exact_rounded_f64"
                );
            }
            // LCOV_EXCL_STOP

            a.solve_exact_rounded_f64(b_vec)
                .map_err(CircumcenterError::from)?
                .into_array()
        }
        Err(e) => {
            cold_path();
            return Err(e.into());
        }
    };

    // Use safe coordinate conversion for solution and add back the first point
    let mut circumcenter_coords = [0.0; D];
    for i in 0..D {
        circumcenter_coords[i] = 0.5_f64.mul_add(x[i], coords_0_f64[i]);
    }
    for value in circumcenter_coords {
        if !value.is_finite() {
            return Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::NonFiniteMeasure {
                    measure: DegenerateMeasure::Volume,
                    value: CoordinateConversionValue::from_numeric_debug(&value),
                },
            });
        }
    }

    Ok(Point::from_validated_coords(circumcenter_coords))
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
/// The circumradius as a value of type T if successful, or an error if the
/// circumcenter calculation fails.
///
/// # Errors
///
/// Returns an error if the circumcenter calculation fails. See [`circumcenter`] for details.
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
///
/// # Arguments
///
/// * `points` - A slice of points that form the simplex
/// * `circumcenter` - The precomputed circumcenter
///
/// # Returns
/// The circumradius as a value of type T if successful, or an error if the
/// simplex is degenerate or the distance calculation fails.
///
/// # Errors
///
/// Returns an error if:
/// - The points slice is empty
/// - Coordinate conversion fails
/// - Distance calculation fails
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

    // Calculate distance using hypot for numerical stability
    let mut diff_coords = [0.0; D];
    for i in 0..D {
        diff_coords[i] = circumcenter_coords[i] - point_coords[i];
    }
    let distance = hypot(&diff_coords);
    Ok(distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;
    use crate::geometry::util::conversions::ValueConversionFailureReason;
    use approx::assert_relative_eq;
    use std::assert_matches;

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
            Point::from_validated_coords([0.0, 0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0, 0.0]),
            Point::from_validated_coords([0.0, 1.0, 0.0]),
            Point::from_validated_coords([0.0, 0.0, 1.0]),
        ];
        let center = circumcenter(&points).unwrap();

        assert_eq!(center, Point::from_validated_coords([0.5, 0.5, 0.5]));
    }

    #[test]
    fn predicates_circumcenter_fail() {
        let points = vec![
            Point::from_validated_coords([0.0, 0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0, 0.0]),
            Point::from_validated_coords([0.0, 1.0, 0.0]),
        ];
        let center = circumcenter(&points);

        assert!(center.is_err());
    }

    #[test]
    fn predicates_circumradius() {
        let points = vec![
            Point::from_validated_coords([0.0, 0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0, 0.0]),
            Point::from_validated_coords([0.0, 1.0, 0.0]),
            Point::from_validated_coords([0.0, 0.0, 1.0]),
        ];
        let radius = circumradius(&points).unwrap();
        let expected_radius: f64 = 3.0_f64.sqrt() / 2.0;

        assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
    }

    #[test]
    fn predicates_circumcenter_2d() {
        let points = vec![
            Point::from_validated_coords([0.0, 0.0]),
            Point::from_validated_coords([2.0, 0.0]),
            Point::from_validated_coords([1.0, 2.0]),
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
        let center = Point::from_validated_coords([0.0, 0.0, 0.0]);
        match circumradius_with_center(&points, &center) {
            Err(CircumcenterError::EmptyPointSet) => {}
            other => panic!("expected EmptyPointSet, got {other:?}"),
        }
    }

    #[test]
    fn predicates_circumradius_2d() {
        let points = vec![
            Point::from_validated_coords([0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0]),
            Point::from_validated_coords([0.0, 1.0]),
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
            Point::from_validated_coords([0.0, 0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0, 0.0]),
            Point::from_validated_coords([0.0, 1.0, 0.0]),
            Point::from_validated_coords([0.0, 0.0, 1.0]),
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
            Point::from_validated_coords([0.0, 0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0, 0.0]),
            Point::from_validated_coords([0.5, 3.0_f64.sqrt() / 2.0, 0.0]),
            Point::from_validated_coords([0.5, 3.0_f64.sqrt() / 6.0, (2.0 / 3.0_f64).sqrt()]),
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
        let distances: Vec<f64> = points
            .iter()
            .map(|p| {
                let p_coords = *p.coords();
                let diff = [
                    p_coords[0] - center_coords[0],
                    p_coords[1] - center_coords[1],
                    p_coords[2] - center_coords[2],
                ];
                hypot(&diff)
            })
            .collect();

        // All distances should be equal
        for i in 1..distances.len() {
            assert_relative_eq!(distances[0], distances[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_circumcenter_regular_simplex_4d() {
        // Test 4D simplex - use orthonormal basis plus origin
        let points: Vec<Point<4>> = vec![
            Point::from_validated_coords([0.0, 0.0, 0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0, 0.0, 0.0]),
            Point::from_validated_coords([0.0, 1.0, 0.0, 0.0]),
            Point::from_validated_coords([0.0, 0.0, 1.0, 0.0]),
            Point::from_validated_coords([0.0, 0.0, 0.0, 1.0]),
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
            Point::from_validated_coords([0.0, 0.0]),
            Point::from_validated_coords([4.0, 0.0]),
            Point::from_validated_coords([0.0, 3.0]),
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
            Point::from_validated_coords([0.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::from_validated_coords([1.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::from_validated_coords([0.0 * scale, 1.0 * scale, 0.0 * scale]),
            Point::from_validated_coords([0.0 * scale, 0.0 * scale, 1.0 * scale]),
        ];
        let center = circumcenter(&points).unwrap();

        // Scaled simplex should have scaled circumcenter
        let expected_center = Point::from_validated_coords([0.5 * scale, 0.5 * scale, 0.5 * scale]);
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
            Point::from_validated_coords([
                0.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ]),
            Point::from_validated_coords([
                1.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ]),
            Point::from_validated_coords([
                0.0 + translation[0],
                1.0 + translation[1],
                0.0 + translation[2],
            ]),
            Point::from_validated_coords([
                0.0 + translation[0],
                0.0 + translation[1],
                1.0 + translation[2],
            ]),
        ];
        let center = circumcenter(&points).unwrap();

        // Get the circumcenter of the untranslated simplex for comparison
        let untranslated_points = vec![
            Point::from_validated_coords([0.0, 0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0, 0.0]),
            Point::from_validated_coords([0.0, 1.0, 0.0]),
            Point::from_validated_coords([0.0, 0.0, 1.0]),
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
            Point::from_validated_coords([0.0, 0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0, 0.0]),
            Point::from_validated_coords([0.5, eps, 0.0]), // Slightly off the line
            Point::from_validated_coords([0.5, 0.0, eps]), // Slightly off the plane
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
            Point::from_validated_coords([0.0, 0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0, 0.0]),
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
            Point::from_validated_coords([0.0, 0.0]),
            Point::from_validated_coords([side_length, 0.0]),
            Point::from_validated_coords([side_length / 2.0, height]),
        ];

        let center = circumcenter(&points).unwrap();
        let center_coords = center.coords();

        // For equilateral triangle, circumcenter should be at centroid
        let expected_x = side_length / 2.0;
        let expected_y = height / 3.0;

        assert_relative_eq!(center_coords[0], expected_x, epsilon = 1e-10);
        assert_relative_eq!(center_coords[1], expected_y, epsilon = 1e-10);

        // Verify all vertices are equidistant from circumcenter
        let _center_point = Point::from_validated_coords([center_coords[0], center_coords[1]]);
        let distances: Vec<f64> = points
            .iter()
            .map(|p| {
                let p_coords = *p.coords();
                let diff = [
                    p_coords[0] - center_coords[0],
                    p_coords[1] - center_coords[1],
                ];
                hypot(&diff)
            })
            .collect();

        // All distances should be equal
        for i in 1..distances.len() {
            assert_relative_eq!(distances[0], distances[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_circumcenter_numerical_stability() {
        // Test with points that could cause numerical instability
        let points: Vec<Point<2>> = vec![
            Point::from_validated_coords([1.0, 0.0]),
            Point::from_validated_coords([1.000_000_1, 0.0]), // Very close to first point
            Point::from_validated_coords([1.000_000_1, 0.000_000_1]), // Forms very thin triangle
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
            Point::from_validated_coords([0.0]),
            Point::from_validated_coords([2.0]),
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
            Point::from_validated_coords([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::from_validated_coords([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::from_validated_coords([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::from_validated_coords([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::from_validated_coords([0.0, 0.0, 0.0, 0.0, 1.0]),
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
        let distances: Vec<f64> = points
            .iter()
            .map(|p| {
                let p_coords = *p.coords();
                let diff = [
                    p_coords[0] - center_coords[0],
                    p_coords[1] - center_coords[1],
                    p_coords[2] - center_coords[2],
                    p_coords[3] - center_coords[3],
                    p_coords[4] - center_coords[4],
                ];
                hypot(&diff)
            })
            .collect();

        // All distances should be equal
        for i in 1..distances.len() {
            assert_relative_eq!(distances[0], distances[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn predicates_circumcenter_precise_values() {
        // Test with precisely known circumcenter values
        // Using a simplex where we can calculate the circumcenter analytically
        let points = vec![
            Point::from_validated_coords([0.0, 0.0, 0.0]),
            Point::from_validated_coords([6.0, 0.0, 0.0]),
            Point::from_validated_coords([0.0, 8.0, 0.0]),
            Point::from_validated_coords([0.0, 0.0, 10.0]),
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
            Point::from_validated_coords([0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0]),
            // Missing third point for 2D circumcenter
        ];

        let result = circumcenter(&points_2d);
        assert_matches!(result, Err(CircumcenterError::InvalidSimplex { .. }));

        // Test too many points
        let points_extra = vec![
            Point::from_validated_coords([0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0]),
            Point::from_validated_coords([0.0, 1.0]),
            Point::from_validated_coords([0.5, 0.5]), // Extra point for 2D
        ];

        let result = circumcenter(&points_extra);
        assert_matches!(result, Err(CircumcenterError::InvalidSimplex { .. }));
    }

    #[test]
    fn test_circumcenter_degenerate_matrix() {
        // Test collinear points in 2D (should cause matrix inversion to fail)
        let collinear_points = vec![
            Point::from_validated_coords([0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0]),
            Point::from_validated_coords([2.0, 0.0]), // Collinear with first two
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
        // solve_exact_rounded_f64 fallback path.
        let eps = 1e-14; // Perturbation small enough to make LU reject
        let points: Vec<Point<3>> = vec![
            Point::from_validated_coords([0.0, 0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0, 0.0]),
            Point::from_validated_coords([0.0, 1.0, 0.0]),
            Point::from_validated_coords([0.5, 0.5, eps]), // Barely off the z=0 plane
        ];

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
        let distances: Vec<f64> = points
            .iter()
            .map(|p| {
                let diff = [
                    p.coords()[0] - center_coords[0],
                    p.coords()[1] - center_coords[1],
                    p.coords()[2] - center_coords[2],
                ];
                hypot(&diff)
            })
            .collect();

        for i in 1..distances.len() {
            assert_relative_eq!(distances[0], distances[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_circumcenter_exact_fallback_near_singular_2d() {
        // Near-degenerate triangle: two vertices very close together.
        // The system matrix has a row with tiny entries, likely tripping
        // DEFAULT_SINGULAR_TOL.
        let eps = 1e-15;
        let points: Vec<Point<2>> = vec![
            Point::from_validated_coords([0.0, 0.0]),
            Point::from_validated_coords([1.0, 0.0]),
            Point::from_validated_coords([0.5, eps]), // Nearly collinear
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
}
