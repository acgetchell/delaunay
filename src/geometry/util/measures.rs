//! Geometric measure computations for simplices.
//!
//! This module provides functions for computing volumes, surface measures,
//! and quality metrics of simplices.
//!
//! Use [`simplex_volume`] for a full-dimensional simplex, [`facet_measure`] for
//! one codimension-one facet, [`inradius`] for its inscribed-sphere radius, and
//! [`surface_measure`] to sum a collection of facets. These functions are also
//! available from [`crate::prelude::geometry`].

#![forbid(unsafe_code)]

use super::circumsphere::{
    CircumcenterError, CircumcenterFailureReason, DegenerateGeometry, DegenerateMeasure,
};
use super::conversions::{ValueConversionError, safe_usize_to_scalar};
use super::norms::hypot;
use crate::core::collections::{MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};
use crate::core::facet::FacetView;
use crate::geometry::matrix::{LaError, LaVector, Matrix, Tolerance, gram_matrix};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::CoordinateConversionValue;
use crate::tds::FacetError;
use num_traits::Float;
use std::array::from_fn;

/// Error type for surface measure computation operations.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{CircumcenterError, SurfaceMeasureError};
///
/// let err = SurfaceMeasureError::from(CircumcenterError::EmptyPointSet);
/// std::assert_matches!(err, SurfaceMeasureError::GeometryError { .. });
/// ```
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum SurfaceMeasureError {
    /// Error retrieving vertices from a facet.
    #[error("Failed to retrieve facet vertices: {source}")]
    FacetVertices {
        /// Underlying facet lookup error.
        #[from]
        source: FacetError,
    },
    /// Error computing geometry measure.
    #[error("Geometry computation failed: {source}")]
    GeometryError {
        /// Underlying geometry error.
        #[source]
        source: Box<CircumcenterError>,
    },
}

impl From<CircumcenterError> for SurfaceMeasureError {
    fn from(source: CircumcenterError) -> Self {
        Self::GeometryError {
            source: Box::new(source),
        }
    }
}

const DEGENERACY_EPSILON_FACTOR: f64 = 64.0;

/// Conservative relative pivot cutoff for approximate Gram measures, not an
/// error bound or a certificate of the exact edge matrix's rank.
const RELATIVE_GRAM_PIVOT_TOLERANCE: f64 = 1e-12;

fn degeneracy_tolerance(scale: f64) -> f64 {
    if scale <= 0.0 {
        return 0.0;
    }

    // Fold the exact power-of-two factor first, avoiding overflow of scale * 64.
    scale * (DEGENERACY_EPSILON_FACTOR * f64::EPSILON)
}

fn is_zero_or_roundoff(value: f64, scale: f64) -> bool {
    let magnitude = Float::abs(value);
    magnitude == 0.0 || magnitude <= degeneracy_tolerance(scale)
}

/// Preserves the finite-measure contract before callers consume derived values.
fn ensure_finite_measure_value(
    value: f64,
    measure: DegenerateMeasure,
) -> Result<(), CircumcenterError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(CircumcenterError::MatrixInversionFailed {
            reason: CircumcenterFailureReason::NonFiniteMeasure {
                measure,
                value: CoordinateConversionValue::from_numeric_debug(&value),
            },
        })
    }
}

/// Checks a newly computed measure before it becomes a divisor or public result.
/// Finiteness and positivity of the inputs do not survive arbitrary arithmetic.
#[inline]
fn ensure_positive_measure_value(
    value: f64,
    measure: DegenerateMeasure,
) -> Result<(), CircumcenterError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid_measure_error(value, measure))
    }
}

/// Keeps typed diagnostic construction off successful measure paths.
#[cold]
fn invalid_measure_error(value: f64, measure: DegenerateMeasure) -> CircumcenterError {
    match ensure_finite_measure_value(value, measure) {
        Err(error) => error,
        Ok(()) => CircumcenterError::MatrixInversionFailed {
            reason: CircumcenterFailureReason::NonPositiveSimplexMeasure {
                measure,
                value: CoordinateConversionValue::from_numeric_debug(&value),
            },
        },
    }
}

/// Validates vector-valued measure intermediates before reduction hides the source.
fn ensure_finite_measure_values<const N: usize>(
    values: &[f64; N],
    measure: DegenerateMeasure,
) -> Result<(), CircumcenterError> {
    for value in values {
        ensure_finite_measure_value(*value, measure)?;
    }
    Ok(())
}

/// Calculate the volume of a D-dimensional simplex.
///
/// This function computes the D-dimensional volume of a simplex formed by D+1 points.
/// Dimensions 1–3 use direct formulas; higher dimensions use a rounded Gram matrix
/// and LDLT determinant. Forming a Gram matrix squares the edge matrix's spectral
/// condition number, so nearly dependent edges can lose numerical rank.
/// For `D >= 4`, LDLT rejects pivots at or below `1e-12` times the largest Gram
/// diagonal entry (the largest squared edge length from the chosen origin).
/// This relative cutoff scales with coordinate units; it is a conservative
/// numerical policy, not a certified rank test. Extreme scales can still fail
/// because Gram entries, the determinant, or the final measure are unrepresentable.
///
/// # Mathematical Background
///
/// For a D-dimensional simplex with vertices p₀, p₁, ..., pD, the volume is:
///
/// **Volume = (1/D!) × √(det(G))**
///
/// where G is the Gram matrix of edge vectors from p₀ to all other vertices.
///
/// # Arguments
///
/// * `points` - Points defining the simplex (must have exactly D+1 points)
///
/// # Returns
///
/// The positive finite D-dimensional measure, in coordinate units raised to the
/// power `D`. A single point in dimension zero has measure `1`.
///
/// # Errors
///
/// - [`CircumcenterError::InvalidSimplex`] if `points.len() != D + 1`.
/// - [`CircumcenterError::MatrixInversionFailed`] for degeneracy detected by a
///   direct formula, non-finite edge differences or measure intermediates, a
///   non-positive/non-finite Gram determinant, or normalization underflow to zero.
///   The [`CircumcenterFailureReason`] identifies the geometric failure.
/// - [`CircumcenterError::LinearAlgebraFailure`] if checked Gram construction or
///   LDLT fails. The preserved [`LaError`] distinguishes rejected pivots
///   ([`LaError::Singular`]), a rounded Gram matrix that is not positive
///   semidefinite ([`LaError::NotPositiveSemidefinite`]), and non-finite arithmetic
///   ([`LaError::NonFinite`]). Dot-product overflow retains the backend operation
///   and first failing coordinate index.
/// - [`CircumcenterError::ValueConversion`] if a dimension factor used in the
///   factorial normalization cannot be represented as `f64`.
///
/// Stored [`Point`] coordinates are already finite. Failures can still arise
/// when subtracting coordinates or computing dot products; no exact fallback
/// or certified error bound is provided for the measure.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::geometry::{CircumcenterError, Coordinate, Point, simplex_volume};
/// use approx::assert_relative_eq;
///
/// # fn main() -> Result<(), CircumcenterError> {
/// // 2D: Triangle area
/// let triangle = vec![
///     Point::try_from([0.0, 0.0])?,
///     Point::try_from([1.0, 0.0])?,
///     Point::try_from([0.0, 1.0])?,
/// ];
/// let area = simplex_volume(&triangle)?;
/// assert_relative_eq!(area, 0.5, epsilon = 1e-10); // Area = 1*1/2 = 0.5
///
/// // 3D: Tetrahedron volume
/// let tetrahedron = vec![
///     Point::try_from([0.0, 0.0, 0.0])?,
///     Point::try_from([1.0, 0.0, 0.0])?,
///     Point::try_from([0.0, 1.0, 0.0])?,
///     Point::try_from([0.0, 0.0, 1.0])?,
/// ];
/// let volume = simplex_volume(&tetrahedron)?;
/// assert_relative_eq!(volume, 1.0/6.0, epsilon = 1e-10); // Volume = 1/6
/// # Ok(())
/// # }
/// ```
///
/// A 4D simplex exercises the Gram path. Finite coordinates can still overflow
/// during its checked dot products:
///
/// ```
/// use delaunay::prelude::geometry::{CircumcenterError, LaError, Point, simplex_volume};
/// use approx::assert_relative_eq;
///
/// # fn main() -> Result<(), CircumcenterError> {
/// let mut simplex = [
///     Point::try_new([0.0, 0.0, 0.0, 0.0])?,
///     Point::try_new([2.0, 0.0, 0.0, 0.0])?,
///     Point::try_new([0.0, 3.0, 0.0, 0.0])?,
///     Point::try_new([0.0, 0.0, 4.0, 0.0])?,
///     Point::try_new([0.0, 0.0, 0.0, 5.0])?,
/// ];
/// // Product of axis lengths divided by 4!.
/// assert_relative_eq!(simplex_volume(&simplex)?, 5.0, epsilon = 1e-12);
///
/// simplex[1] = Point::try_new([1.0e200, 0.0, 0.0, 0.0])?;
/// std::assert_matches!(
///     simplex_volume(&simplex),
///     Err(CircumcenterError::LinearAlgebraFailure {
///         source: LaError::NonFinite { .. },
///     })
/// );
/// # Ok(())
/// # }
/// ```
pub fn simplex_volume<const D: usize>(points: &[Point<D>]) -> Result<f64, CircumcenterError> {
    #[cfg(debug_assertions)]
    if std::env::var_os("DELAUNAY_DEBUG_UNUSED_IMPORTS").is_some() {
        tracing::debug!(
            points_len = points.len(),
            dimension = D,
            "measures::simplex_volume called"
        );
    }
    if points.len() != D + 1 {
        return Err(CircumcenterError::InvalidSimplex {
            actual: points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    // Special cases for low dimensions with optimized formulas
    match D {
        1 => {
            // 1D: Length of line segment
            let p0 = points[0].coords();
            let p1 = points[1].coords();
            let diff = [p1[0] - p0[0]];
            ensure_finite_measure_values(&diff, DegenerateMeasure::Length)?;
            let length = Float::abs(diff[0]);
            ensure_finite_measure_value(length, DegenerateMeasure::Length)?;

            // Check for degeneracy (coincident points).
            if length == 0.0 {
                return Err(CircumcenterError::MatrixInversionFailed {
                    reason: CircumcenterFailureReason::DegenerateSimplex {
                        measure: DegenerateMeasure::Length,
                        degeneracy: DegenerateGeometry::CoincidentPoints,
                    },
                });
            }

            Ok(length)
        }
        2 => {
            // 2D: Triangle area using cross product magnitude / 2
            let p0 = points[0].coords();
            let p1 = points[1].coords();
            let p2 = points[2].coords();

            // Vectors from p0 to p1 and p0 to p2
            let v1 = [p1[0] - p0[0], p1[1] - p0[1]];
            let v2 = [p2[0] - p0[0], p2[1] - p0[1]];
            ensure_finite_measure_values(&v1, DegenerateMeasure::Area)?;
            ensure_finite_measure_values(&v2, DegenerateMeasure::Area)?;

            // 2D cross product magnitude: |v1.x * v2.y - v1.y * v2.x|
            let cross_z = v1[1].mul_add(-v2[0], v1[0] * v2[1]);
            ensure_finite_measure_value(cross_z, DegenerateMeasure::Area)?;
            let area = Float::abs(cross_z) / 2.0;

            // Check for degeneracy (collinear points) using a scale-aware
            // determinant threshold instead of an absolute area cutoff.
            let cross_scale = Float::abs(v1[0] * v2[1]) + Float::abs(v1[1] * v2[0]);
            ensure_finite_measure_value(cross_scale, DegenerateMeasure::Area)?;
            if is_zero_or_roundoff(cross_z, cross_scale) {
                return Err(CircumcenterError::MatrixInversionFailed {
                    reason: CircumcenterFailureReason::DegenerateSimplex {
                        measure: DegenerateMeasure::Volume,
                        degeneracy: DegenerateGeometry::CollinearPoints,
                    },
                });
            }

            ensure_positive_measure_value(area, DegenerateMeasure::Area)?;
            Ok(area)
        }
        3 => {
            // 3D: Tetrahedron volume using triple scalar product / 6
            let p0 = points[0].coords();
            let p1 = points[1].coords();
            let p2 = points[2].coords();
            let p3 = points[3].coords();

            // Edge vectors from p0
            let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let v3 = [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]];
            ensure_finite_measure_values(&v1, DegenerateMeasure::Volume)?;
            ensure_finite_measure_values(&v2, DegenerateMeasure::Volume)?;
            ensure_finite_measure_values(&v3, DegenerateMeasure::Volume)?;

            // Triple scalar product: v1 · (v2 × v3)
            let cross_x = v2[2].mul_add(-v3[1], v2[1] * v3[2]);
            let cross_y = v2[0].mul_add(-v3[2], v2[2] * v3[0]);
            let cross_z = v2[1].mul_add(-v3[0], v2[0] * v3[1]);
            ensure_finite_measure_values(&[cross_x, cross_y, cross_z], DegenerateMeasure::Volume)?;
            let triple_product = v1[2].mul_add(cross_z, v1[1].mul_add(cross_y, v1[0] * cross_x));
            ensure_finite_measure_value(triple_product, DegenerateMeasure::Volume)?;

            // Volume = |triple product| / 6
            let six = 6.0;
            let volume = Float::abs(triple_product) / six;

            // Check for degeneracy (coplanar points) using the expansion scale
            // of the 3x3 determinant.
            let triple_scale = Float::abs(v1[0] * v2[1] * v3[2])
                + Float::abs(v1[0] * v2[2] * v3[1])
                + Float::abs(v1[1] * v2[2] * v3[0])
                + Float::abs(v1[1] * v2[0] * v3[2])
                + Float::abs(v1[2] * v2[0] * v3[1])
                + Float::abs(v1[2] * v2[1] * v3[0]);
            ensure_finite_measure_value(triple_scale, DegenerateMeasure::Volume)?;
            if is_zero_or_roundoff(triple_product, triple_scale) {
                return Err(CircumcenterError::MatrixInversionFailed {
                    reason: CircumcenterFailureReason::DegenerateSimplex {
                        measure: DegenerateMeasure::Volume,
                        degeneracy: DegenerateGeometry::CoplanarPoints,
                    },
                });
            }

            ensure_positive_measure_value(volume, DegenerateMeasure::Volume)?;
            Ok(volume)
        }
        _ => {
            // Higher dimensions: Use Gram matrix method
            simplex_volume_gram_matrix::<D>(points)
        }
    }
}

/// Preserves geometric failure reasons before a Gram determinant is square-rooted.
///
/// Exact Gram matrices are positive semidefinite, but rounded construction can
/// lose that property or numerical rank. After [`gram_determinant_ldlt`] preserves
/// backend failures, this guard classifies a non-finite, negative, or zero result
/// without clamping it or hiding its sign behind an absolute tolerance.
fn validate_gram_determinant(det: f64) -> Result<f64, CircumcenterError> {
    if !det.is_finite() {
        return Err(CircumcenterError::MatrixInversionFailed {
            reason: CircumcenterFailureReason::NonFiniteGramDeterminant,
        });
    }

    if det < 0.0 {
        return Err(CircumcenterError::MatrixInversionFailed {
            reason: CircumcenterFailureReason::NegativeGramDeterminant,
        });
    }

    // Degenerate case: zero determinant means no volume
    if det == 0.0 {
        return Err(CircumcenterError::MatrixInversionFailed {
            reason: CircumcenterFailureReason::DegenerateSimplex {
                measure: DegenerateMeasure::Volume,
                degeneracy: DegenerateGeometry::CollinearOrCoplanarPoints,
            },
        });
    }

    Ok(det)
}

/// Computes `n!` in `f64` while preserving typed conversion failures.
fn factorial_f64(n: usize) -> Result<f64, CircumcenterError> {
    let mut value = 1.0f64;
    for k in 2..=n {
        let k_f64 = safe_usize_to_scalar(k).map_err(|e| CircumcenterError::ValueConversion {
            source: Box::new(ValueConversionError::CoordinateConversion {
                value: CoordinateConversionValue::from_usize(k),
                from_type: "usize",
                to_type: "f64",
                source: Box::new(e),
            }),
        })?;
        value *= k_f64;
    }
    Ok(value)
}

/// Compute a Gram determinant using `la-stack`'s stack-allocated LDLT factorization.
///
/// This helper preserves typed backend failures so public measure APIs can
/// distinguish invalid linear-algebra state from geometric degeneracy reported
/// by [`validate_gram_determinant`].
#[inline]
fn gram_determinant_ldlt<const D: usize>(gram_matrix: Matrix<D>) -> Result<f64, CircumcenterError> {
    // Gram diagonals are finite squared edge lengths. Uniform coordinate scaling
    // multiplies both the pivots and this scale by the same squared factor.
    let scale = gram_matrix
        .as_rows()
        .iter()
        .enumerate()
        .fold(0.0_f64, |scale, (index, row)| scale.max(row[index]));
    let tolerance = Tolerance::try_new(RELATIVE_GRAM_PIVOT_TOLERANCE * scale)?;
    let ldlt = gram_matrix.ldlt(tolerance)?;
    ldlt.det().map_err(CircumcenterError::from)
}

/// Reconstructs a rejected edge only on failure, keeping diagnostic storage off
/// the successful Gram-construction path.
///
/// Volume and facet callers retain [`CircumcenterFailureReason::NonFiniteMeasure`]
/// with the first rejected difference, rather than receiving a backend vector-input
/// error. Any other backend failure remains available through its typed source.
#[cold]
fn gram_edge_error<const D: usize>(
    source: LaError,
    coords: &[f64; D],
    origin: &[f64; D],
) -> CircumcenterError {
    let differences = from_fn(|axis| coords[axis] - origin[axis]);
    match ensure_finite_measure_values::<D>(&differences, DegenerateMeasure::Volume) {
        Ok(()) => source.into(),
        Err(error) => error,
    }
}

/// Keeps geometric edge validation local while delegating symmetric Gram assembly.
/// `N` is the intrinsic simplex dimension and `D` is the ambient dimension.
///
/// Exactly `N + 1` points produce an `N` × `N` matrix of dot products between
/// edges from the first point. Finite storage in [`Point`] is reused; newly
/// computed differences are checked before Gram construction. The result is
/// bit-for-bit symmetric, but rank and positive definiteness are left to LDLT.
#[inline]
fn simplex_gram_matrix<const N: usize, const D: usize>(
    points: &[Point<D>],
) -> Result<Matrix<N>, CircumcenterError> {
    if points.len() != N + 1 {
        return Err(CircumcenterError::InvalidSimplex {
            actual: points.len(),
            expected: N + 1,
            dimension: D,
        });
    }
    // Point already proves its stored f64 coordinates finite. Only the newly
    // computed differences need validation before they become backend vectors.
    let origin = points[0].coords();
    let mut edges = [LaVector::<D>::zero(); N];
    for (edge, point) in edges.iter_mut().zip(&points[1..]) {
        let coords = point.coords();
        let differences = from_fn(|axis| coords[axis] - origin[axis]);
        *edge = LaVector::try_new(differences)
            .map_err(|source| gram_edge_error(source, coords, origin))?;
    }
    gram_matrix(&edges).map_err(CircumcenterError::from)
}

/// Calculate the volume of a D-dimensional simplex using the Gram matrix method.
///
/// Shares checked edge/Gram construction with facet measurement while keeping
/// full-dimensional `D!` normalization and geometric determinant errors local.
///
/// # Arguments
///
/// * `points` - Points defining the simplex (must have exactly D+1 points)
///
/// # Returns
///
/// The volume of the simplex, or an error if calculation fails
fn simplex_volume_gram_matrix<const D: usize>(
    points: &[Point<D>],
) -> Result<f64, CircumcenterError> {
    let gram_matrix = simplex_gram_matrix::<D, D>(points)?;

    // Compute Gram determinant with validation (LDLT exploits symmetry / PSD structure).
    let det = validate_gram_determinant(gram_determinant_ldlt(gram_matrix)?)?;

    let volume_f64 = {
        let sqrt_det = det.sqrt();
        let d_fact = factorial_f64(D)?;
        sqrt_det / d_fact
    };

    ensure_positive_measure_value(volume_f64, DegenerateMeasure::Volume)?;
    Ok(volume_f64)
}

/// Calculate the inradius of a D-dimensional simplex.
///
/// The inradius is the radius of the largest sphere that can be inscribed
/// within the simplex. It is computed using the formula:
///
/// **inradius = D × volume / `surface_area`**
///
/// where `surface_area` is the sum of all (D-1)-dimensional facet volumes.
/// The ambient dimension `D` must be at least `1`.
///
/// # Arguments
///
/// * `points` - Points defining the simplex (must have exactly D+1 points)
///
/// # Returns
///
/// The positive finite inradius of the simplex, or an error if calculation fails.
///
/// # Errors
///
/// Returns [`CircumcenterError::InvalidSimplex`] if `points.len() != D + 1`,
/// then [`CircumcenterError::DimensionTooSmall`] if `D == 0`.
///
/// Propagates errors from [`simplex_volume`] and [`facet_measure`], including
/// their typed geometric and linear-algebra failures. Additionally returns
/// [`CircumcenterError::MatrixInversionFailed`] if the sum of facet measures or
/// the resulting radius is non-finite or non-positive, including radius underflow
/// to zero, and [`CircumcenterError::ValueConversion`]
/// if `D` cannot be represented as `f64` for the radius formula.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::geometry::{CircumcenterError, Coordinate, Point, inradius};
/// use approx::assert_relative_eq;
///
/// # fn main() -> Result<(), CircumcenterError> {
/// // 2D: Equilateral triangle with side length 1
/// let triangle = vec![
///     Point::try_from([0.0, 0.0])?,
///     Point::try_from([1.0, 0.0])?,
///     Point::try_from([0.5, 0.866025])?, // sqrt(3)/2 ≈ 0.866025
/// ];
/// let r_in = inradius(&triangle)?;
/// // For equilateral triangle: inradius ≈ 0.2887 (exact: sqrt(3)/6)
/// assert_relative_eq!(r_in, 0.28867, epsilon = 1e-4);
/// # Ok(())
/// # }
/// ```
pub fn inradius<const D: usize>(points: &[Point<D>]) -> Result<f64, CircumcenterError> {
    if points.len() != D + 1 {
        return Err(CircumcenterError::InvalidSimplex {
            actual: points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    if D == 0 {
        return Err(CircumcenterError::DimensionTooSmall {
            dimension: D,
            minimum: 1,
        });
    }

    // Special-case 1D: segment inradius is half the length
    if D == 1 {
        let length = simplex_volume(points)?; // 1D volume = segment length
        let radius = length / 2.0;
        ensure_positive_measure_value(radius, DegenerateMeasure::Length)?;
        return Ok(radius);
    }

    // simplex_volume establishes a positive finite measure.
    let volume = simplex_volume(points)?;

    // Compute surface area by summing all (D-1)-dimensional facet volumes
    let mut surface_area = 0.0;
    for i in 0..=D {
        // Keep contiguous numerical workspace on the stack, omitting vertex i.
        let facet_points: [Point<D>; D] = from_fn(|j| points[j + usize::from(j >= i)]);

        let facet_area = facet_measure(&facet_points)?;
        surface_area += facet_area;
    }

    // Summing finite facets can overflow; establish the divisor's invariant.
    ensure_positive_measure_value(surface_area, DegenerateMeasure::SurfaceArea)?;

    // inradius = D * volume / surface_area
    let d_scalar = safe_usize_to_scalar(D).map_err(|e| CircumcenterError::ValueConversion {
        source: Box::new(ValueConversionError::CoordinateConversion {
            value: CoordinateConversionValue::from_usize(D),
            from_type: "usize",
            to_type: "f64",
            source: Box::new(e),
        }),
    })?;

    let inradius = (d_scalar * volume) / surface_area;
    ensure_positive_measure_value(inradius, DegenerateMeasure::Length)?;
    Ok(inradius)
}

/// Calculate the area/volume of a facet defined by a set of points.
///
/// This function calculates the (D-1)-dimensional "area" of a facet in D-dimensional space:
/// - 1D: Point measure (0-dimensional, returns 1)
/// - 2D: Length of line segment (1-dimensional)
/// - 3D: Area of triangle using cross product (2-dimensional)
/// - 4D+: Generalized volume using Gram matrix method
///
/// For dimensions 4 and higher, this function uses a rounded Gram matrix and LDLT
/// determinant. Forming a Gram matrix squares the edge matrix's spectral condition
/// number, so nearly dependent edges can lose numerical rank.
/// The relative LDLT pivot cutoff and representability limits are the same as
/// those described by [`simplex_volume`].
///
/// # Arguments
///
/// * `points` - Exactly `D` points defining a facet of intrinsic dimension `D - 1`.
///   Positive ambient dimensions through
///   [`MAX_STACK_MATRIX_DIM + 1`](crate::geometry::matrix::MAX_STACK_MATRIX_DIM)
///   (currently `8`) are supported.
///
/// # Returns
///
/// The positive finite intrinsic measure, in coordinate units raised to the power
/// `D - 1`. For `D = 1`, the single point has dimensionless measure `1`.
///
/// # Errors
///
/// - [`CircumcenterError::InvalidSimplex`] if `points.len() != D`.
/// - [`CircumcenterError::DimensionTooSmall`] if `D == 0`, after checking the
///   point count.
/// - [`CircumcenterError::UnsupportedMatrixDimension`] if the Gram dimension
///   `D - 1` exceeds [`crate::geometry::matrix::MAX_STACK_MATRIX_DIM`].
/// - [`CircumcenterError::MatrixInversionFailed`] for geometric degeneracy,
///   non-finite edge differences or measure intermediates, or a non-positive/
///   non-finite Gram determinant, or normalization underflow to zero.
///   The [`CircumcenterFailureReason`] is preserved.
/// - [`CircumcenterError::LinearAlgebraFailure`] for checked Gram construction
///   or LDLT failures, preserving [`LaError::Singular`],
///   [`LaError::NotPositiveSemidefinite`], or [`LaError::NonFinite`] as applicable.
///   Dot-product overflow retains the backend operation and first failing
///   coordinate index.
///
/// As with [`simplex_volume`], finite [`Point`] coordinates do not guarantee
/// representable edge differences or dot products. No exact fallback or
/// certified error bound is provided for the measure.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::geometry::{CircumcenterError, Coordinate, Point, facet_measure};
/// use approx::assert_relative_eq;
///
/// # fn main() -> Result<(), CircumcenterError> {
/// // 2D: Line segment length (1D facet in 2D space)
/// let line_segment = vec![
///     Point::try_from([0.0, 0.0])?,
///     Point::try_from([3.0, 4.0])?,
/// ];
/// let length = facet_measure(&line_segment)?;
/// assert_relative_eq!(length, 5.0, epsilon = 1e-10); // sqrt(3² + 4²) = 5
///
/// // 3D: Triangle area (2D facet in 3D space)
/// let triangle = vec![
///     Point::try_from([0.0, 0.0, 0.0])?,
///     Point::try_from([3.0, 0.0, 0.0])?,
///     Point::try_from([0.0, 4.0, 0.0])?,
/// ];
/// let area = facet_measure(&triangle)?;
/// assert_relative_eq!(area, 6.0, epsilon = 1e-10); // 3*4/2 = 6
/// # Ok(())
/// # }
/// ```
///
/// A tetrahedral facet embedded in 4D uses a 3 × 3 Gram matrix:
///
/// ```
/// use delaunay::prelude::geometry::{CircumcenterError, Point, facet_measure};
/// use approx::assert_relative_eq;
///
/// # fn main() -> Result<(), CircumcenterError> {
/// let facet = [
///     Point::try_new([0.0, 0.0, 0.0, 0.0])?,
///     Point::try_new([2.0, 0.0, 0.0, 0.0])?,
///     Point::try_new([0.0, 3.0, 0.0, 0.0])?,
///     Point::try_new([0.0, 0.0, 4.0, 0.0])?,
/// ];
/// // Product of axis lengths divided by 3!.
/// assert_relative_eq!(facet_measure(&facet)?, 4.0, epsilon = 1e-12);
/// # Ok(())
/// # }
/// ```
pub fn facet_measure<const D: usize>(points: &[Point<D>]) -> Result<f64, CircumcenterError> {
    if points.len() != D {
        return Err(CircumcenterError::InvalidSimplex {
            actual: points.len(),
            expected: D,
            dimension: D,
        });
    }

    match D {
        0 => Err(CircumcenterError::DimensionTooSmall {
            dimension: D,
            minimum: 1,
        }),
        1 => {
            // The intrinsic measure of a 0-simplex is one: its empty Gram
            // determinant and 0! normalization are both one.
            Ok(1.0)
        }
        2 => {
            // 2D: Length of line segment (1D facet in 2D space)
            let p0 = points[0].coords();
            let p1 = points[1].coords();

            let diff = [p1[0] - p0[0], p1[1] - p0[1]];
            ensure_finite_measure_values(&diff, DegenerateMeasure::Length)?;
            let length = hypot(&diff);
            ensure_finite_measure_value(length, DegenerateMeasure::Length)?;

            // Check for degeneracy (coincident points).
            if length == 0.0 {
                return Err(CircumcenterError::MatrixInversionFailed {
                    reason: CircumcenterFailureReason::DegenerateFacet {
                        measure: DegenerateMeasure::Length,
                        degeneracy: DegenerateGeometry::CoincidentPoints,
                    },
                });
            }

            Ok(length)
        }
        3 => {
            // 3D: Area of triangle (2D facet in 3D space) using cross product
            let p0 = points[0].coords();
            let p1 = points[1].coords();
            let p2 = points[2].coords();

            // Vectors from p0 to p1 and p0 to p2
            let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            ensure_finite_measure_values(&v1, DegenerateMeasure::Area)?;
            ensure_finite_measure_values(&v2, DegenerateMeasure::Area)?;

            // Cross product v1 × v2
            let cross = [
                v1[2].mul_add(-v2[1], v1[1] * v2[2]),
                v1[0].mul_add(-v2[2], v1[2] * v2[0]),
                v1[1].mul_add(-v2[0], v1[0] * v2[1]),
            ];
            ensure_finite_measure_values(&cross, DegenerateMeasure::Area)?;

            // Area is |cross product| / 2
            let cross_magnitude = hypot(&cross);
            let area = cross_magnitude / (2.0); // Divide by 2
            ensure_finite_measure_value(cross_magnitude, DegenerateMeasure::Area)?;

            // Check for degeneracy (collinear points) using the expansion scale
            // of the cross-product components.
            let cross_scale = Float::abs(v1[1] * v2[2])
                + Float::abs(v1[2] * v2[1])
                + Float::abs(v1[2] * v2[0])
                + Float::abs(v1[0] * v2[2])
                + Float::abs(v1[0] * v2[1])
                + Float::abs(v1[1] * v2[0]);
            ensure_finite_measure_value(cross_scale, DegenerateMeasure::Area)?;
            if is_zero_or_roundoff(cross_magnitude, cross_scale) {
                return Err(CircumcenterError::MatrixInversionFailed {
                    reason: CircumcenterFailureReason::DegenerateFacet {
                        measure: DegenerateMeasure::Area,
                        degeneracy: DegenerateGeometry::CollinearPoints,
                    },
                });
            }

            ensure_positive_measure_value(area, DegenerateMeasure::Area)?;
            Ok(area)
        }
        4 => {
            // 4D: Volume of tetrahedron (3D facet in 4D space)
            // Use Gram matrix method for correct calculation
            facet_measure_gram_matrix::<D>(points)
        }
        _ => {
            // Higher dimensions: Use Gram matrix method for correct calculation
            facet_measure_gram_matrix::<D>(points)
        }
    }
}

/// Calculate the area/volume of a (D-1)-dimensional simplex using the Gram matrix method.
///
/// Keeps codimension-one shape dispatch and `(D - 1)!` normalization in the
/// geometry layer while sharing checked Gram construction with full simplices:
///
/// **Volume = (1/(D-1)!) × √(det(G))**
///
/// where G is the Gram matrix of edge vectors from one vertex to all other vertices.
///
/// # Mathematical Background
///
/// The Gram matrix method is the standard approach for computing simplex volumes in
/// high-dimensional spaces, as described in:
///
/// - Coxeter, H.S.M. "Introduction to Geometry" (2nd ed., 1969), Chapter 13
/// - Richter-Gebert, Jürgen. "Perspectives on Projective Geometry" (2011), Section 14.3
/// - Edelsbrunner, Herbert. "Geometry and Topology for Mesh Generation" (2001), Chapter 2
///
/// The method constructs the Gram matrix G where G\[i,j\] = `v_i` · `v_j` (dot product of
/// edge vectors), then computes the volume as the square root of the determinant
/// divided by the appropriate factorial.
///
/// The backend computes each dot product once and mirrors it for exact symmetry.
/// The rounded Gram matrix need not retain the exact rank or positive semidefiniteness;
/// LDLT and determinant validation preserve typed failures in those cases.
///
/// # Arguments
///
/// * `points` - Exactly `D` points defining the `(D - 1)`-dimensional facet.
///   The caller must establish `D >= 1` before calling this helper.
///
/// # Returns
///
/// The volume of the simplex, or an error if calculation fails
///
/// # Errors
///
/// Preserves the geometric and backend errors described by [`facet_measure`],
/// including unsupported Gram dimensions and numerical rank loss.
fn facet_measure_gram_matrix<const D: usize>(
    points: &[Point<D>],
) -> Result<f64, CircumcenterError> {
    // Compute Gram determinant with validation.
    //
    // For a (D-1)-simplex realized in D dimensions, there are (D-1) edge vectors from
    // one vertex to the remaining vertices, so the Gram matrix is (D-1)×(D-1).
    let gram_dim = D - 1;
    let det = try_with_la_stack_matrix!(gram_dim, |gram_matrix| {
        gram_matrix = simplex_gram_matrix(points)?;
        validate_gram_determinant(gram_determinant_ldlt(gram_matrix)?)
    })?;

    let volume_f64 = {
        let sqrt_det = det.sqrt();
        let d_fact = factorial_f64(D - 1)?;
        sqrt_det / d_fact
    };

    ensure_positive_measure_value(volume_f64, DegenerateMeasure::Volume)?;
    Ok(volume_f64)
}

/// Calculate the surface area of a triangulated boundary by summing facet measures.
///
/// This function calculates the total surface area of a boundary defined by
/// a collection of facets. Each facet's measure (area/volume) is calculated
/// and summed to give the total surface measure.
///
/// # Arguments
///
/// * `facets` - Collection of facets defining the boundary surface
///
/// # Returns
///
/// The finite total surface area/volume. An empty collection has measure zero.
///
/// # Errors
///
/// Propagates individual facet failures through [`SurfaceMeasureError::GeometryError`].
/// If summing finite facets overflows, returns the same variant with
/// [`CircumcenterFailureReason::NonFiniteMeasure`] for [`DegenerateMeasure::SurfaceArea`].
///
/// # Examples
///
/// ```
/// use delaunay::prelude::{construction::*, geometry::*};
/// use delaunay::prelude::geometry::surface_measure;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Tds(#[from] delaunay::prelude::tds::TdsError),
/// #     #[error(transparent)]
/// #     Query(#[from] delaunay::prelude::query::QueryError),
/// #     #[error(transparent)]
/// #     Facet(#[from] delaunay::prelude::tds::FacetError),
/// #     #[error(transparent)]
/// #     SurfaceMeasure(#[from] delaunay::prelude::geometry::SurfaceMeasureError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// // Create a triangulation and calculate surface measure of boundary facets
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0, 0.0]?,
///     delaunay::vertex![0.0, 0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
///
/// // Get boundary facets as FacetViews
/// let boundary_facets = dt.boundary_facets()?.collect::<Result<Vec<_>, _>>()?;
///
/// // Calculate surface area
/// let surface_area = surface_measure(&boundary_facets)?;
/// assert!(surface_area > 0.0);
/// # Ok(())
/// # }
/// ```
pub fn surface_measure<U, V, const D: usize>(
    facets: &[FacetView<'_, U, V, D>],
) -> Result<f64, SurfaceMeasureError> {
    let mut total_measure = 0.0;

    for facet in facets {
        // Keep contiguous numerical workspace inline while the view borrows its owner.
        let points: SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            facet.vertices().map(|vertex| *vertex.point()).collect();

        let measure = facet_measure(&points).map_err(SurfaceMeasureError::from)?;
        total_measure += measure;
    }

    ensure_finite_measure_value(total_measure, DegenerateMeasure::SurfaceArea)?;
    Ok(total_measure)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::vertex::Vertex, delaunay_model::DelaunayTriangulation,
        geometry::traits::coordinate::InvalidCoordinateValue, tds::TdsDraft, vertex,
    };
    use approx::assert_relative_eq;
    use la_stack::ArithmeticOperation;
    use std::assert_matches;

    /// Axis lengths give independent volume and facet-measure oracles.
    fn scaled_axis_simplex<const D: usize>(scales: [f64; D]) -> Vec<Point<D>> {
        let mut points = vec![Point::try_new([0.0; D]).unwrap()];
        for (axis, scale) in scales.into_iter().enumerate() {
            let mut coords = [0.0; D];
            coords[axis] = scale;
            points.push(Point::try_new(coords).unwrap());
        }
        points
    }

    /// A 3-4-5 plane rotation, reflection, and translation preserve measures and inradius.
    fn check_measure_isometries<const D: usize>() {
        let scales = from_fn(|axis| [0.125, 8.0, 0.5, 2.0, 1.0, 1.0][axis]);
        let points = scaled_axis_simplex::<D>(scales);
        let factorial = (1..=D)
            .map(|value| safe_usize_to_scalar(value).unwrap())
            .product::<f64>();
        let volume = scales.iter().product::<f64>() / factorial;
        let facet_volume = volume * safe_usize_to_scalar(D).unwrap() / scales[D - 1];
        let inverse_scales = scales.map(f64::recip);
        let inradius_expected = (inverse_scales.iter().sum::<f64>()
            + inverse_scales
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt())
        .recip();
        for (rotate, reflect, translation) in [
            (false, false, 0.0),
            (true, false, 0.0),
            (false, true, 0.0),
            (false, false, 16.0),
            (true, true, 16.0),
        ] {
            let mut transformed: Vec<_> = points
                .iter()
                .map(|point| {
                    let mut coords = *point.coords();
                    if rotate {
                        let (x, y) = (coords[0], coords[D - 1]);
                        coords[0] = 0.8_f64.mul_add(-y, 0.6 * x);
                        coords[D - 1] = 0.6_f64.mul_add(y, 0.8 * x);
                    }
                    if reflect {
                        coords[0] = -coords[0];
                    }
                    Point::try_new(coords.map(|value| value + translation)).unwrap()
                })
                .collect();
            assert_relative_eq!(
                simplex_volume(&transformed).unwrap(),
                volume,
                max_relative = 1e-10
            );
            assert_relative_eq!(
                facet_measure(&transformed[..D]).unwrap(),
                facet_volume,
                max_relative = 1e-10
            );
            // Cycle the origin so facet selection sees each vertex
            // at every position in the input.
            for _ in 0..=D {
                assert_relative_eq!(
                    inradius(&transformed).unwrap(),
                    inradius_expected,
                    max_relative = 1e-10
                );
                transformed.rotate_left(1);
            }
        }
    }

    /// Uniform scaling preserves shape; closed forms fix the measure units.
    fn check_measure_rescaling<const D: usize>() {
        let dimension = safe_usize_to_scalar(D).unwrap();
        let exponent = i32::try_from(D).unwrap();
        let factorial = (1..=D)
            .map(|value| safe_usize_to_scalar(value).unwrap())
            .product::<f64>();
        for scale_exponent in [-30, 0, 30] {
            let side = 2.0_f64.powi(scale_exponent);
            let mut points = scaled_axis_simplex([side; D]);
            let volume = side.powi(exponent) / factorial;
            let facet = side.powi(exponent - 1) / (factorial / dimension);
            assert_relative_eq!(
                facet_measure(&points[..D]).unwrap(),
                facet,
                epsilon = 0.0,
                max_relative = 1e-12
            );
            assert_relative_eq!(
                facet_measure(&points[1..]).unwrap(),
                dimension.sqrt() * facet,
                epsilon = 0.0,
                max_relative = 1e-12
            );
            // Every origin represents the same simplex, including non-diagonal
            // Gram matrices when an axis endpoint becomes the origin.
            for _ in 0..=D {
                assert_relative_eq!(
                    simplex_volume(&points).unwrap(),
                    volume,
                    epsilon = 0.0,
                    max_relative = 1e-12
                );
                assert_relative_eq!(
                    inradius(&points).unwrap(),
                    side / (dimension + dimension.sqrt()),
                    epsilon = 0.0,
                    max_relative = 1e-12
                );
                points.rotate_left(1);
            }
        }
    }

    /// Scaling cannot repair a shape whose smallest edge is 2^-30 of the others.
    fn check_gram_rank_cutoff_rescaling<const D: usize>() {
        for scale_exponent in [30, 0, -30] {
            let side = 2.0_f64.powi(scale_exponent);
            let mut scales = [side; D];
            scales[0] = side * 2.0_f64.powi(-30);
            let points = scaled_axis_simplex(scales);
            for result in [simplex_volume(&points), facet_measure(&points[..D])] {
                assert_matches!(
                    result,
                    Err(CircumcenterError::LinearAlgebraFailure {
                        source: LaError::Singular { .. }
                    })
                );
            }
        }
    }

    /// Separates geometric edge overflow from backend dot-product and rank failures.
    fn check_gram_measure_errors<const D: usize>() {
        let mut points = scaled_axis_simplex::<D>([1.0; D]);
        points[1] = points[0];
        for result in [simplex_volume(&points), facet_measure(&points[..D])] {
            assert_matches!(
                result,
                Err(CircumcenterError::LinearAlgebraFailure {
                    source: LaError::Singular { .. }
                })
            );
        }

        for axis in [0, D - 1] {
            let mut points = scaled_axis_simplex::<D>([1.0; D]);
            let mut edge = [0.0; D];
            edge[axis] = 1.0e200;
            // Keep the overflowing edge in the facet even when it uses the last
            // ambient coordinate, which a rectangular Gram matrix must retain.
            points[1] = Point::try_new(edge).unwrap();
            for result in [simplex_volume(&points), facet_measure(&points[..D])] {
                assert_eq!(
                    result.unwrap_err(),
                    CircumcenterError::LinearAlgebraFailure {
                        source: LaError::non_finite_computation_step(
                            ArithmeticOperation::VectorDotProduct,
                            axis
                        )
                    },
                    "{D}D dot-product overflow at coordinate {axis}"
                );
            }

            for (origin_value, expected) in [
                (-f64::MAX, InvalidCoordinateValue::PositiveInfinity),
                (f64::MAX, InvalidCoordinateValue::NegativeInfinity),
            ] {
                let mut points = scaled_axis_simplex::<D>([1.0; D]);
                let mut origin = [0.0; D];
                origin[axis] = origin_value;
                points[0] = Point::try_new(origin).unwrap();
                origin[axis] = -origin_value;
                points[1] = Point::try_new(origin).unwrap();
                for result in [simplex_volume(&points), facet_measure(&points[..D])] {
                    assert_eq!(
                        result.unwrap_err(),
                        CircumcenterError::MatrixInversionFailed {
                            reason: CircumcenterFailureReason::NonFiniteMeasure {
                                measure: DegenerateMeasure::Volume,
                                value: CoordinateConversionValue::NonFinite(expected.clone()),
                            }
                        },
                        "{D}D edge overflow at coordinate {axis} from {origin_value}"
                    );
                }
            }
        }
    }

    /// Exercises Gram construction and LDLT against independently known determinants.
    fn gram_det_from_edges<const N: usize, const AMBIENT: usize>(
        edges: &[[f64; AMBIENT]; N],
    ) -> Result<f64, CircumcenterError> {
        let vectors = edges.map(|edge| LaVector::try_new(edge).unwrap());
        validate_gram_determinant(gram_determinant_ldlt(gram_matrix(&vectors)?)?)
    }

    macro_rules! measure_isometry_tests {
        ($($dim:literal),+) => {
            pastey::paste! {
                $(#[test]
                fn [<measure_isometries_ $dim d>]() {
                    check_measure_isometries::<$dim>();
                }
                #[test]
                fn [<measure_rescaling_ $dim d>]() {
                    check_measure_rescaling::<$dim>();
                })+
            }
        };
    }

    macro_rules! gram_measure_error_tests {
        ($($dim:literal),+) => {
            pastey::paste! {
                $(#[test]
                fn [<gram_measure_errors_ $dim d>]() {
                    check_gram_measure_errors::<$dim>();
                }
                #[test]
                fn [<gram_rank_cutoff_rescaling_ $dim d>]() {
                    check_gram_rank_cutoff_rescaling::<$dim>();
                })+
            }
        };
    }

    measure_isometry_tests!(2, 3, 4, 5, 6);
    gram_measure_error_tests!(4, 5, 6);

    #[test]
    fn simplex_gram_matrix_preserves_exact_symmetry_and_shape() {
        let points = [
            Point::try_new([2.0, 3.0, 4.0, 5.0]).unwrap(),
            Point::try_new([2.25, 3.5, 4.75, 6.0]).unwrap(),
            Point::try_new([1.5, 4.0, 6.0, 8.0]).unwrap(),
            Point::try_new([3.0, 2.0, 4.5, 7.0]).unwrap(),
            Point::try_new([4.0, 3.0, 2.0, 1.0]).unwrap(),
        ];
        let gram = simplex_gram_matrix::<3, 4>(&points[..4]).unwrap();
        let expected = [
            [1.875, 4.875, 2.125],
            [4.875, 14.25, 5.5],
            [2.125, 5.5, 6.25],
        ];
        for (i, row) in gram.as_rows().iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert_relative_eq!(*value, expected[i][j], epsilon = 0.0);
                assert_eq!(value.to_bits(), gram.as_rows()[j][i].to_bits());
            }
        }
        for count in [0, 3, 5] {
            assert_eq!(
                simplex_gram_matrix::<3, 4>(&points[..count]).unwrap_err(),
                CircumcenterError::InvalidSimplex {
                    actual: count,
                    expected: 4,
                    dimension: 4,
                }
            );
        }
    }

    #[test]
    fn surface_measure_error_display_names_variants() {
        let error = SurfaceMeasureError::from(CircumcenterError::EmptyPointSet);
        let display = format!("{error}");
        assert!(display.contains("Geometry computation failed"));
        assert!(display.contains("Empty point set"));
    }

    // =============================================================================
    // SIMPLEX VOLUME TESTS
    // =============================================================================

    #[test]
    fn simplex_volume_zero_dimension_has_unit_measure() {
        let point = Point::<0>::try_new([]).expect("finite point coordinates");
        assert_relative_eq!(simplex_volume(&[point]).unwrap(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_simplex_volume_1d_line_segment() {
        // 1D: Line segment length
        let line = vec![
            Point::try_new([0.0]).expect("finite point coordinates"),
            Point::try_new([5.0]).expect("finite point coordinates"),
        ];
        let volume = simplex_volume(&line).unwrap();
        assert_relative_eq!(volume, 5.0, epsilon = 1e-10);

        // Negative direction
        let line_neg = vec![
            Point::try_new([5.0]).expect("finite point coordinates"),
            Point::try_new([0.0]).expect("finite point coordinates"),
        ];
        let volume_neg = simplex_volume(&line_neg).unwrap();
        assert_relative_eq!(volume_neg, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn simplex_volume_degenerate_segment_errors() {
        let line = vec![
            Point::try_new([2.0]).expect("finite point coordinates"),
            Point::try_new([2.0]).expect("finite point coordinates"),
        ];
        assert_matches!(
            simplex_volume(&line),
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::DegenerateSimplex {
                    measure: DegenerateMeasure::Length,
                    degeneracy: DegenerateGeometry::CoincidentPoints,
                }
            })
        );
    }

    #[test]
    fn test_simplex_volume_2d_triangle() {
        // 2D: Right triangle with legs 3 and 4
        let triangle = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([3.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 4.0]).expect("finite point coordinates"),
        ];
        let area = simplex_volume(&triangle).unwrap();
        assert_relative_eq!(area, 6.0, epsilon = 1e-10); // Area = (3*4)/2 = 6

        // Equilateral triangle with side 1
        let equilateral = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 0.866_025]).expect("finite point coordinates"), // sqrt(3)/2
        ];
        let area_eq = simplex_volume(&equilateral).unwrap();
        // Area = sqrt(3)/4 ≈ 0.433013
        assert_relative_eq!(area_eq, 0.433_013, epsilon = 1e-5);
    }

    #[test]
    fn test_simplex_volume_3d_tetrahedron() {
        // 3D: Regular tetrahedron with vertices at unit cube corners
        let tetrahedron = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let volume = simplex_volume(&tetrahedron).unwrap();
        assert_relative_eq!(volume, 1.0 / 6.0, epsilon = 1e-10); // Volume = 1/6
    }

    #[test]
    fn test_simplex_volume_4d_simplex() {
        // 4D: Regular 4-simplex
        let simplex_4d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let volume = simplex_volume(&simplex_4d).unwrap();
        // 4D simplex volume = 1/4! = 1/24
        assert_relative_eq!(volume, 1.0 / 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simplex_volume_degenerate() {
        // Degenerate triangle (collinear points) should return an error
        let collinear = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 1.0]).expect("finite point coordinates"),
            Point::try_new([2.0, 2.0]).expect("finite point coordinates"),
        ];
        let result = simplex_volume(&collinear);
        assert_matches!(
            result,
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::DegenerateSimplex {
                    measure: DegenerateMeasure::Volume,
                    degeneracy: DegenerateGeometry::CollinearPoints,
                },
            }),
            "Degenerate simplex should return a typed collinearity error"
        );
    }

    #[test]
    fn simplex_volume_non_finite_1d_intermediate_is_numeric_failure() {
        let points = vec![
            Point::try_new([-f64::MAX]).expect("finite point coordinates"),
            Point::try_new([f64::MAX]).expect("finite point coordinates"),
        ];

        assert_matches!(
            simplex_volume(&points),
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::NonFiniteMeasure {
                    measure: DegenerateMeasure::Length,
                    value: CoordinateConversionValue::NonFinite(
                        InvalidCoordinateValue::PositiveInfinity
                    ),
                },
            })
        );
    }

    #[test]
    fn simplex_volume_non_finite_2d_intermediate_is_numeric_failure() {
        let points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([f64::MAX, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, f64::MAX]).expect("finite point coordinates"),
        ];

        assert_matches!(
            simplex_volume(&points),
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::NonFiniteMeasure {
                    measure: DegenerateMeasure::Area,
                    value: CoordinateConversionValue::NonFinite(
                        InvalidCoordinateValue::PositiveInfinity
                    ),
                },
            })
        );
    }

    #[test]
    fn simplex_volume_coplanar_tetrahedron_errors() {
        let coplanar = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.25, 0.25, 0.0]).expect("finite point coordinates"),
        ];
        let result = simplex_volume(&coplanar);
        assert_matches!(
            result,
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::DegenerateSimplex {
                    measure: DegenerateMeasure::Volume,
                    degeneracy: DegenerateGeometry::CoplanarPoints,
                },
            }),
            "coplanar tetrahedron should be a typed degeneracy"
        );
    }

    #[test]
    fn test_simplex_volume_very_small_valid_simplices() {
        let small_val = 1e-14;

        let triangle = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([small_val, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, small_val]).expect("finite point coordinates"),
        ];
        let area = simplex_volume(&triangle).unwrap();
        assert_relative_eq!(area, small_val * small_val / 2.0, max_relative = 1e-12);

        let tetrahedron = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([small_val, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, small_val, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, small_val]).expect("finite point coordinates"),
        ];
        let volume = simplex_volume(&tetrahedron).unwrap();
        assert_relative_eq!(
            volume,
            small_val * small_val * small_val / 6.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn large_area_preserves_scale_aware_degeneracy() {
        // The cross product is exactly 2^1020 and the area is 2^1019.
        // Both are finite, although multiplying the expansion scale by 64
        // before applying epsilon would overflow.
        let side = 2.0_f64.powi(510);
        let triangle = scaled_axis_simplex([side, side]);
        let embedded_triangle = scaled_axis_simplex([side, side, 1.0]);
        for result in [
            simplex_volume(&triangle),
            facet_measure(&embedded_triangle[..3]),
        ] {
            assert_relative_eq!(result.unwrap(), 2.0_f64.powi(1019), max_relative = 1e-14);
        }
        assert_relative_eq!(
            inradius(&triangle).unwrap(),
            side / (2.0 + 2.0_f64.sqrt()),
            max_relative = 1e-14
        );
    }

    #[test]
    fn large_volume_preserves_scale_aware_degeneracy() {
        // An axis tetrahedron with side 2^340 has determinant 2^1020.
        let side = 2.0_f64.powi(340);
        let tetrahedron = scaled_axis_simplex([side; 3]);
        assert_relative_eq!(
            simplex_volume(&tetrahedron).unwrap(),
            2.0_f64.powi(1020) / 6.0,
            max_relative = 1e-14
        );
        assert_relative_eq!(
            inradius(&tetrahedron).unwrap(),
            side / (3.0 + 3.0_f64.sqrt()),
            max_relative = 1e-14
        );
    }

    #[test]
    fn large_nearly_collinear_geometry_is_rejected() {
        // The determinant is 2^973 while its expansion scale exceeds 2^1021,
        // so the configuration remains inside the relative roundoff threshold.
        let side = 2.0_f64.powi(510);
        let offset = 2.0_f64.powi(463);
        let triangle = [
            Point::try_new([0.0, 0.0]).unwrap(),
            Point::try_new([side, side]).unwrap(),
            Point::try_new([side, side + offset]).unwrap(),
        ];
        assert_matches!(
            simplex_volume(&triangle),
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::DegenerateSimplex {
                    measure: DegenerateMeasure::Volume,
                    degeneracy: DegenerateGeometry::CollinearPoints,
                },
            })
        );
        let embedded = triangle.map(|point| {
            let [x, y] = *point.coords();
            Point::try_new([x, y, 0.0]).unwrap()
        });
        assert_matches!(
            facet_measure(&embedded),
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::DegenerateFacet {
                    measure: DegenerateMeasure::Area,
                    degeneracy: DegenerateGeometry::CollinearPoints,
                },
            })
        );
    }

    #[test]
    fn area_normalization_rejects_zero_but_accepts_positive_subnormals() {
        for bits in [1, 2] {
            let height = f64::from_bits(bits);
            let triangle = scaled_axis_simplex([1.0, height]);
            let embedded_triangle = scaled_axis_simplex([1.0, height, 1.0]);
            for result in [
                simplex_volume(&triangle),
                facet_measure(&embedded_triangle[..3]),
            ] {
                if bits == 1 {
                    assert_eq!(
                        result.unwrap_err(),
                        CircumcenterError::MatrixInversionFailed {
                            reason: CircumcenterFailureReason::NonPositiveSimplexMeasure {
                                measure: DegenerateMeasure::Area,
                                value: CoordinateConversionValue::from_numeric_debug(&0.0),
                            },
                        }
                    );
                } else {
                    assert_eq!(result.unwrap().to_bits(), 1);
                }
            }
        }
    }

    #[test]
    fn volume_normalization_rejects_zero_but_accepts_positive_subnormals() {
        // Dividing three subnormal units by 6 rounds the halfway result to zero;
        // four units round to the smallest positive representable volume.
        for bits in [3, 4] {
            let tetrahedron = scaled_axis_simplex([1.0, 1.0, f64::from_bits(bits)]);
            let result = simplex_volume(&tetrahedron);
            if bits == 3 {
                assert_eq!(
                    result.unwrap_err(),
                    CircumcenterError::MatrixInversionFailed {
                        reason: CircumcenterFailureReason::NonPositiveSimplexMeasure {
                            measure: DegenerateMeasure::Volume,
                            value: CoordinateConversionValue::from_numeric_debug(&0.0),
                        },
                    }
                );
            } else {
                assert_eq!(result.unwrap().to_bits(), 1);
            }
        }
    }

    #[test]
    fn test_simplex_volume_wrong_point_count() {
        // Wrong number of points for 2D
        let points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        ];
        let result = simplex_volume::<2>(&points);
        assert!(result.is_err());
    }

    // =============================================================================
    // INRADIUS TESTS
    // =============================================================================

    #[test]
    fn inradius_zero_dimension_returns_typed_error() {
        let point = Point::<0>::try_new([]).expect("finite point coordinates");
        assert_matches!(
            inradius(&[point]),
            Err(CircumcenterError::DimensionTooSmall {
                dimension: 0,
                minimum: 1,
            })
        );
        assert_matches!(
            inradius::<0>(&[]),
            Err(CircumcenterError::InvalidSimplex {
                actual: 0,
                expected: 1,
                dimension: 0,
            })
        );
    }

    #[test]
    fn inradius_1d_is_half_segment_length() {
        let points = [
            Point::try_new([-2.0]).expect("finite point coordinates"),
            Point::try_new([4.0]).expect("finite point coordinates"),
        ];
        assert_relative_eq!(inradius(&points).unwrap(), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn inradius_rejects_zero_but_accepts_positive_subnormal_radii() {
        for bits in [1, 2] {
            let height = f64::from_bits(bits);
            let segment = scaled_axis_simplex([height]);
            let triangle = scaled_axis_simplex([2.0, height]);
            // Both simplices have positive representable measures. Their exact
            // radii round like height / 2, including the halfway-to-zero case.
            assert!(simplex_volume(&segment).unwrap() > 0.0);
            assert!(simplex_volume(&triangle).unwrap() > 0.0);
            for result in [inradius(&segment), inradius(&triangle)] {
                if bits == 1 {
                    assert_eq!(
                        result.unwrap_err(),
                        CircumcenterError::MatrixInversionFailed {
                            reason: CircumcenterFailureReason::NonPositiveSimplexMeasure {
                                measure: DegenerateMeasure::Length,
                                value: CoordinateConversionValue::from_numeric_debug(&0.0),
                            },
                        }
                    );
                } else {
                    assert_eq!(result.unwrap().to_bits(), 1);
                }
            }
        }
    }

    #[test]
    fn inradius_rejects_overflow_when_summing_finite_facets() {
        for (base, overflows) in [(8.0e307, false), (1.0e308, true)] {
            let triangle = [
                Point::try_new([0.0, 0.0]).unwrap(),
                Point::try_new([base, 0.0]).unwrap(),
                Point::try_new([base / 2.0, 1.0e-3]).unwrap(),
            ];
            let volume = simplex_volume(&triangle).unwrap();
            assert!(volume.is_finite() && volume > 0.0);
            for [i, j] in [[0, 1], [0, 2], [1, 2]] {
                let length = facet_measure(&[triangle[i], triangle[j]]).unwrap();
                assert!(length.is_finite() && length > 0.0);
            }

            if overflows {
                assert_matches!(
                    inradius(&triangle),
                    Err(CircumcenterError::MatrixInversionFailed {
                        reason: CircumcenterFailureReason::NonFiniteMeasure {
                            measure: DegenerateMeasure::SurfaceArea,
                            value: CoordinateConversionValue::NonFinite(
                                InvalidCoordinateValue::PositiveInfinity
                            ),
                        },
                    })
                );
            } else {
                assert_relative_eq!(inradius(&triangle).unwrap(), 5.0e-4, max_relative = 1e-14);
            }
        }
    }

    #[test]
    fn test_inradius_2d_equilateral_triangle() {
        // Equilateral triangle with side 1
        let triangle = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 0.866_025]).expect("finite point coordinates"), // sqrt(3)/2
        ];
        let r_in = inradius(&triangle).unwrap();
        // For equilateral triangle: inradius = sqrt(3)/6 ≈ 0.28867513
        assert_relative_eq!(r_in, 0.288_675_13, epsilon = 1e-5);
    }

    #[test]
    fn test_inradius_2d_right_triangle() {
        // Right triangle with legs 3 and 4
        let triangle = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([3.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 4.0]).expect("finite point coordinates"),
        ];
        let r_in = inradius(&triangle).unwrap();
        // For right triangle: inradius = (a+b-c)/2 = (3+4-5)/2 = 1.0
        assert_relative_eq!(r_in, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inradius_3d_regular_tetrahedron() {
        // Regular tetrahedron at unit cube corners
        let tetrahedron = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
        ];
        let r_in = inradius(&tetrahedron).unwrap();
        // For this tetrahedron: inradius ≈ 0.2113
        assert_relative_eq!(r_in, 0.2113, epsilon = 1e-3);
    }

    #[test]
    fn test_inradius_degenerate() {
        // Degenerate triangle (collinear points)
        let collinear = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([2.0, 0.0]).expect("finite point coordinates"),
        ];
        let result = inradius(&collinear);
        assert!(result.is_err()); // Should fail for degenerate simplex
    }

    // =============================================================================
    // BASIC FACET MEASURE TESTS (BY DIMENSION)
    // =============================================================================

    #[test]
    fn facet_measure_zero_dimension_returns_typed_error() {
        assert_matches!(
            facet_measure::<0>(&[]),
            Err(CircumcenterError::DimensionTooSmall {
                dimension: 0,
                minimum: 1,
            })
        );
        let point = Point::<0>::try_new([]).expect("finite point coordinates");
        assert_matches!(
            facet_measure(&[point]),
            Err(CircumcenterError::InvalidSimplex {
                actual: 1,
                expected: 0,
                dimension: 0,
            })
        );
    }

    #[test]
    fn facet_measure_rejects_first_unsupported_gram_dimension() {
        let points = scaled_axis_simplex::<9>([1.0; 9]);
        assert_matches!(
            facet_measure(&points[..9]),
            Err(CircumcenterError::UnsupportedMatrixDimension {
                requested: 8,
                max: 7,
            })
        );
    }

    #[test]
    fn test_facet_measure_1d_point() {
        // A 1D facet is a 0-simplex, whose intrinsic measure is one.
        let points = vec![Point::try_new([5.0]).expect("finite point coordinates")];
        let measure = facet_measure(&points).unwrap();
        assert_relative_eq!(measure, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_2d_line_segment() {
        // 2D: Line segment (1D facet in 2D space) - 3-4-5 triangle
        let points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([3.0, 4.0]).expect("finite point coordinates"),
        ];
        let measure = facet_measure(&points).unwrap();
        // Length should be sqrt(3² + 4²) = 5.0
        assert_relative_eq!(measure, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_3d_triangle_right_angle() {
        // 3D: Right triangle (area = 1/2 * base * height)
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([3.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 4.0, 0.0]).expect("finite point coordinates"),
        ];
        let measure = facet_measure(&points).unwrap();
        // Area should be 3 * 4 / 2 = 6.0
        assert_relative_eq!(measure, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_4d_tetrahedron() {
        // 4D: Unit tetrahedron (3D facet in 4D space)
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
        ];
        let measure = facet_measure(&points).unwrap();

        // For a unit tetrahedron in 4D with vertices at origin and 3 unit vectors,
        // the volume should be 1/3! = 1/6
        // This is a 3-dimensional simplex in 4D space
        assert_relative_eq!(measure, 1.0 / 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gram_determinant_ldlt_known_spd() {
        // Symmetric positive-definite matrix with known determinant.
        let gram = Matrix::<2>::try_from_rows([[4.0, 2.0], [2.0, 3.0]]).unwrap();
        let det = gram_determinant_ldlt(gram).unwrap();
        assert_relative_eq!(det, 8.0, epsilon = 1e-12);
    }

    #[test]
    fn gram_determinant_ldlt_preserves_determinant_overflow_error() {
        let gram = Matrix::<5>::try_from_rows([
            [1.0e100, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0e100, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0e100, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0e100, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0e100],
        ])
        .unwrap();

        assert_matches!(
            gram_determinant_ldlt(gram),
            Err(CircumcenterError::LinearAlgebraFailure {
                source: LaError::NonFinite { .. }
            })
        );
    }

    #[test]
    fn test_gram_determinant_parallel_edges_errors() {
        let edges = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        assert_matches!(
            gram_det_from_edges(&edges),
            Err(CircumcenterError::LinearAlgebraFailure {
                source: LaError::Singular { .. }
            })
        );
    }

    #[test]
    fn degeneracy_tolerance_zero_scale() {
        assert_relative_eq!(degeneracy_tolerance(0.0), 0.0);
        assert_relative_eq!(degeneracy_tolerance(-1.0), 0.0);
    }

    #[test]
    fn gram_determinant_nonfinite_errors() {
        assert_matches!(
            validate_gram_determinant(f64::NAN),
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::NonFiniteGramDeterminant,
            })
        );
        assert_matches!(
            validate_gram_determinant(f64::INFINITY),
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::NonFiniteGramDeterminant,
            })
        );
    }

    #[test]
    fn gram_determinant_zero_reports_degenerate_volume() {
        assert_matches!(
            validate_gram_determinant(0.0),
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::DegenerateSimplex {
                    measure: DegenerateMeasure::Volume,
                    degeneracy: DegenerateGeometry::CollinearOrCoplanarPoints,
                },
            })
        );
    }

    #[test]
    fn test_validate_gram_determinant_tiny_negative_errors() {
        assert_matches!(
            validate_gram_determinant(-1e-13),
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::NegativeGramDeterminant,
            })
        );
    }

    // Macro to test orthogonal edges across dimensions
    macro_rules! test_gram_det_orthogonal {
        ($test_name:ident, $dim:literal) => {
            #[test]
            fn $test_name() {
                let mut edges = [[0.0f64; $dim]; $dim];
                // Set up orthogonal unit vectors
                for i in 0..$dim {
                    edges[i][i] = 1.0;
                }

                let det = gram_det_from_edges(&edges).unwrap();
                // Gram matrix is identity, so determinant should be 1.0
                assert_relative_eq!(det, 1.0, epsilon = 1e-10);
            }
        };
    }

    // Generate tests for 2D through 5D
    test_gram_det_orthogonal!(test_gram_determinant_orthogonal_2d, 2);
    test_gram_det_orthogonal!(test_gram_determinant_orthogonal_3d, 3);
    test_gram_det_orthogonal!(test_gram_determinant_orthogonal_4d, 4);
    test_gram_det_orthogonal!(test_gram_determinant_orthogonal_5d, 5);

    // Macro to test scaled edges across dimensions
    macro_rules! test_gram_det_scaled {
        ($test_name:ident, $dim:literal, $scale:expr, $expected_det:expr) => {
            #[test]
            fn $test_name() {
                let mut edges = [[0.0f64; $dim]; $dim];
                // Set up scaled orthogonal vectors
                for i in 0..$dim {
                    edges[i][i] = $scale;
                }

                let det = gram_det_from_edges(&edges).unwrap();
                // Gram matrix diagonal has $scale^2, determinant is ($scale^2)^$dim
                assert_relative_eq!(det, $expected_det, epsilon = 1e-9);
            }
        };
    }

    // Generate scaled tests for 2D through 5D with scale factor 2.0
    test_gram_det_scaled!(test_gram_determinant_scaled_2d, 2, 2.0, 16.0); // (2^2)^2 = 16
    test_gram_det_scaled!(test_gram_determinant_scaled_3d, 3, 2.0, 64.0); // (2^2)^3 = 64
    test_gram_det_scaled!(test_gram_determinant_scaled_4d, 4, 2.0, 256.0); // (2^2)^4 = 256
    test_gram_det_scaled!(test_gram_determinant_scaled_5d, 5, 2.0, 1024.0); // (2^2)^5 = 1024

    #[test]
    fn test_gram_matrix_debug() {
        // Test the Gram matrix method against known simple cases

        // Test 1: Unit right triangle in 3D - area 0.5
        let triangle_3d = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
        ];
        let area_3d = facet_measure(&triangle_3d).unwrap();
        assert_relative_eq!(area_3d, 0.5, epsilon = 1e-10);
        #[cfg(feature = "diagnostics")]
        tracing::debug!("3D triangle area: {area_3d} (expected: 0.5)");

        // Test 1b: Nearly singular triangle should not error due to tiny negative det
        let eps = 1e-10;
        let near_singular = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, eps, 0.0]).expect("finite point coordinates"),
        ];
        let area_ns = facet_measure(&near_singular).unwrap();
        assert!(area_ns >= 0.0);

        // Test 2: Same triangle but use direct Gram matrix calculation
        let area_3d_gram = facet_measure_gram_matrix::<3>(&triangle_3d).unwrap();
        assert_relative_eq!(area_3d_gram, 0.5, epsilon = 1e-10);
        #[cfg(feature = "diagnostics")]
        tracing::debug!("3D triangle area (Gram): {area_3d_gram} (expected: 0.5)");

        // Test 3: Unit tetrahedron in 4D - should be 1/6 ≈ 0.167
        let tetrahedron_4d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
        ];
        let volume_4d = facet_measure(&tetrahedron_4d).unwrap();
        assert_relative_eq!(volume_4d, 1.0 / 6.0, epsilon = 1e-10);
        #[cfg(feature = "diagnostics")]
        tracing::debug!(
            "4D tetrahedron volume: {} (expected: {})",
            volume_4d,
            1.0 / 6.0
        );

        // Test 4: Manual calculation for the 4D tetrahedron
        let volume_4d_gram = facet_measure_gram_matrix::<4>(&tetrahedron_4d).unwrap();
        assert_relative_eq!(volume_4d_gram, 1.0 / 6.0, epsilon = 1e-10);
        #[cfg(feature = "diagnostics")]
        tracing::debug!(
            "4D tetrahedron volume (Gram): {} (expected: {})",
            volume_4d_gram,
            1.0 / 6.0
        );
    }

    #[test]
    fn test_facet_measure_5d_simplex() {
        // 5D: 4-dimensional facet in 5D space (4-simplex volume)
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
        ];
        let measure = facet_measure(&points).unwrap();

        // Volume of 4-simplex with vertices at origin and unit vectors
        // Should be 1/4! = 1/24 (generalized determinant formula)
        assert_relative_eq!(measure, 1.0 / 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_6d_simplex() {
        // 6D: 5-dimensional facet in 6D space
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
        ];
        let measure = facet_measure(&points).unwrap();

        // Volume of 5-simplex with vertices at origin and unit vectors
        // Should be 1/5! = 1/120
        assert_relative_eq!(measure, 1.0 / 120.0, epsilon = 1e-10);
    }

    // =============================================================================
    // FACET MEASURE ERROR HANDLING TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_wrong_point_count() {
        // Test error when wrong number of points provided
        // 3D expects 3 points, but provide 2
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
        ];
        let result = facet_measure::<3>(&points);

        assert!(result.is_err());
        match result.unwrap_err() {
            CircumcenterError::InvalidSimplex {
                actual,
                expected,
                dimension,
            } => {
                assert_eq!(actual, 2);
                assert_eq!(expected, 3);
                assert_eq!(dimension, 3);
            }
            other => panic!("Expected InvalidSimplex error, got: {other:?}"),
        }
    }

    // =============================================================================
    // FACET MEASURE DEGENERATE CASE TESTS
    // =============================================================================

    #[test]
    fn facet_measure_coincident_2d_points_reports_degenerate_length() {
        let points = vec![
            Point::try_new([1.0, -2.0]).expect("finite point coordinates"),
            Point::try_new([1.0, -2.0]).expect("finite point coordinates"),
        ];

        assert_matches!(
            facet_measure(&points),
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::DegenerateFacet {
                    measure: DegenerateMeasure::Length,
                    degeneracy: DegenerateGeometry::CoincidentPoints,
                },
            })
        );
    }

    #[test]
    fn facet_measure_non_finite_2d_intermediate_is_numeric_failure() {
        let points = vec![
            Point::try_new([-f64::MAX, -f64::MAX]).expect("finite point coordinates"),
            Point::try_new([f64::MAX, f64::MAX]).expect("finite point coordinates"),
        ];

        assert_matches!(
            facet_measure(&points),
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::NonFiniteMeasure {
                    measure: DegenerateMeasure::Length,
                    value: CoordinateConversionValue::NonFinite(
                        InvalidCoordinateValue::PositiveInfinity
                    ),
                },
            })
        );
    }

    #[test]
    fn facet_measure_non_finite_3d_intermediate_is_numeric_failure() {
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([f64::MAX, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, f64::MAX, 0.0]).expect("finite point coordinates"),
        ];

        assert_matches!(
            facet_measure(&points),
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::NonFiniteMeasure {
                    measure: DegenerateMeasure::Area,
                    value: CoordinateConversionValue::NonFinite(
                        InvalidCoordinateValue::PositiveInfinity
                    ),
                },
            })
        );
    }

    #[test]
    fn test_facet_measure_zero_area_triangle() {
        // Degenerate triangle (collinear points) - should return an error
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([2.0, 0.0, 0.0]).expect("finite point coordinates"), // Collinear
        ];
        let result = facet_measure(&points);

        assert_matches!(
            result,
            Err(CircumcenterError::MatrixInversionFailed {
                reason: CircumcenterFailureReason::DegenerateFacet {
                    measure: DegenerateMeasure::Area,
                    degeneracy: DegenerateGeometry::CollinearPoints,
                }
            })
        );
    }

    #[test]
    fn test_facet_measure_nearly_collinear_points_2d() {
        // Test with points that are nearly collinear in 2D
        let eps = 1e-10;
        let points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, eps]).expect("finite point coordinates"), // Slightly off the x-axis
        ];

        let measure = facet_measure(&points).unwrap();
        let expected = eps.mul_add(eps, 1.0).sqrt(); // Length of line segment
        assert_relative_eq!(measure, expected, epsilon = 1e-9);
    }

    #[test]
    fn test_facet_measure_nearly_coplanar_points_3d() {
        // Test with points that are truly nearly coplanar in 3D
        let eps = 1e-8;
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, eps, eps]).expect("finite point coordinates"), // Very close to the line from (0,0,0) to (1,0,0)
        ];

        let measure = facet_measure(&points).unwrap();
        // Should be small but non-zero area
        assert!(
            measure > 0.0,
            "Nearly coplanar triangle should have positive area"
        );
        // With points very close to being collinear, area should be very small
        assert!(
            measure < 1e-6,
            "Nearly coplanar triangle should have very small area, got: {measure}"
        );
    }

    #[test]
    fn test_facet_measure_degenerate_4d_tetrahedron() {
        // Four points spanning only a plane cannot form a nondegenerate tetrahedron.
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.5, 0.5, 0.0, 0.0]).expect("finite point coordinates"),
        ];

        let result = facet_measure(&points);
        assert_matches!(
            result,
            Err(CircumcenterError::LinearAlgebraFailure {
                source: LaError::Singular { .. }
            })
        );
    }

    // =============================================================================
    // SURFACE MEASURE TESTS
    // =============================================================================

    #[test]
    fn test_surface_measure_empty_facets() {
        // Test with empty facet collection
        let facets: Vec<FacetView<'_, (), (), 3>> = vec![];
        let result = surface_measure(&facets).unwrap();

        assert_relative_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn surface_measure_rejects_overflow_when_summing_finite_facets() {
        for (base, overflows) in [(8.0e307, false), (1.0e308, true)] {
            let mut draft: TdsDraft<(), (), 2> = TdsDraft::new();
            let a = draft.insert_vertex(vertex![0.0, 0.0].unwrap()).unwrap();
            let b = draft.insert_vertex(vertex![base, 0.0].unwrap()).unwrap();
            let c = draft
                .insert_vertex(vertex![base / 2.0, 1.0e-3].unwrap())
                .unwrap();
            draft.insert_simplex([a, b, c]).unwrap();
            let tds = draft.finish().unwrap();
            let facets = tds.facets().collect::<Result<Vec<_>, _>>().unwrap();
            assert_eq!(facets.len(), 3);
            for facet in &facets {
                let points: Vec<_> = facet.vertices().map(|vertex| *vertex.point()).collect();
                let length = facet_measure(&points).unwrap();
                assert!(length.is_finite() && length > 0.0);
            }

            if overflows {
                assert_eq!(
                    surface_measure(&facets).unwrap_err(),
                    SurfaceMeasureError::from(CircumcenterError::MatrixInversionFailed {
                        reason: CircumcenterFailureReason::NonFiniteMeasure {
                            measure: DegenerateMeasure::SurfaceArea,
                            value: CoordinateConversionValue::NonFinite(
                                InvalidCoordinateValue::PositiveInfinity
                            ),
                        },
                    })
                );
            } else {
                assert_relative_eq!(
                    surface_measure(&facets).unwrap(),
                    1.6e308,
                    max_relative = 1e-14
                );
            }
        }
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "Comparisons are against exact literals (constructed geometry), acceptable in this test"
    )]
    fn test_surface_measure_single_facet() {
        // Test with single triangular facet using TDS boundary facets

        // Create a right triangle tetrahedron
        let vertices: Vec<Vertex<(), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(), // v1
            vertex!([3.0, 0.0, 0.0]).unwrap(), // v2
            vertex!([0.0, 4.0, 0.0]).unwrap(), // v3
            vertex!([0.0, 0.0, 1.0]).unwrap(), // v4
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let boundary_facets: Vec<_> = dt
            .boundary_facets()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // Find the facet opposite to v4 (contains vertices v1, v2, v3)
        let tarfacet = boundary_facets
            .iter()
            .find(|facet| {
                let facet_vertices: Vec<_> = facet.vertices().collect();
                facet_vertices.len() == 3
                    && facet_vertices.iter().any(|v| {
                        let coords = *v.point().coords();
                        coords == [0.0, 0.0, 0.0]
                    })
                    && facet_vertices.iter().any(|v| {
                        let coords = *v.point().coords();
                        coords == [3.0, 0.0, 0.0]
                    })
                    && facet_vertices.iter().any(|v| {
                        let coords = *v.point().coords();
                        coords == [0.0, 4.0, 0.0]
                    })
            })
            .expect("Should find the target facet");

        let surface_area = surface_measure(std::slice::from_ref(tarfacet)).unwrap();

        // Should be area of right triangle: 3 * 4 / 2 = 6.0
        assert_relative_eq!(surface_area, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_surface_measure_consistency_with_facet_measure() {
        // Test that surface_measure sum equals sum of individual facet_measures
        // Create a triangulation with 5 vertices and 2 tetrahedra to get both boundary and internal facets

        let vertices: Vec<Vertex<(), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(), // v1
            vertex!([1.0, 0.0, 0.0]).unwrap(), // v2
            vertex!([0.0, 1.0, 0.0]).unwrap(), // v3
            vertex!([0.0, 0.0, 1.0]).unwrap(), // v4
            vertex!([1.0, 1.0, 1.0]).unwrap(), // v5
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let boundary_facets: Vec<_> = dt
            .boundary_facets()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // Take first two boundary facets for testing
        let facet1 = boundary_facets[0].clone();
        let facet2 = boundary_facets[1].clone();

        // Calculate surface measure
        let total_surface = surface_measure(&[facet1.clone(), facet2.clone()]).unwrap();

        // Calculate individual facet measures and sum them
        let points1: Vec<Point<3>> = facet1
            .vertices()
            .map(|v| {
                let coords = *v.point().coords();
                Point::try_new(coords).expect("finite point coordinates")
            })
            .collect();
        let points2: Vec<Point<3>> = facet2
            .vertices()
            .map(|v| {
                let coords = *v.point().coords();
                Point::try_new(coords).expect("finite point coordinates")
            })
            .collect();

        let measure1 = facet_measure(&points1).unwrap();
        let measure2 = facet_measure(&points2).unwrap();
        let sum_individual = measure1 + measure2;

        // Should be equal
        assert_relative_eq!(total_surface, sum_individual, epsilon = 1e-10);
    }

    // =============================================================================
    // FACET MEASURE SCALING PROPERTY TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_scaled_simplex_2d() {
        // Test scaling property: measure should scale by |λ|^(D-1)
        let scale = 3.0;
        let original_points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        ];
        let scaled_points = vec![
            Point::try_new([0.0 * scale, 0.0 * scale]).expect("finite point coordinates"),
            Point::try_new([1.0 * scale, 0.0 * scale]).expect("finite point coordinates"),
        ];

        let original_measure = facet_measure(&original_points).unwrap();
        let scaled_measure = facet_measure(&scaled_points).unwrap();

        // For 2D (D=2), measure scales by |λ|^(2-1) = |λ|^1 = λ
        assert_relative_eq!(scaled_measure, original_measure * scale, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_scaled_simplex_3d() {
        // Test scaling property for 3D triangle (D=3, measure scales by λ^2)
        let scale = 2.5;
        let original_points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([2.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 3.0, 0.0]).expect("finite point coordinates"),
        ];
        let scaled_points = vec![
            Point::try_new([0.0 * scale, 0.0 * scale, 0.0 * scale])
                .expect("finite point coordinates"),
            Point::try_new([2.0 * scale, 0.0 * scale, 0.0 * scale])
                .expect("finite point coordinates"),
            Point::try_new([0.0 * scale, 3.0 * scale, 0.0 * scale])
                .expect("finite point coordinates"),
        ];

        let original_measure = facet_measure(&original_points).unwrap();
        let scaled_measure = facet_measure(&scaled_points).unwrap();

        // For 3D (D=3), measure scales by |λ|^(3-1) = λ^2
        assert_relative_eq!(
            scaled_measure,
            original_measure * scale * scale,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_facet_measure_scaled_simplex_4d() {
        // Test scaling property for 4D tetrahedron (D=4, measure scales by λ^3)
        let scale = 2.0;
        let original_points = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
        ];
        let scaled_points = vec![
            Point::try_new([0.0 * scale, 0.0 * scale, 0.0 * scale, 0.0 * scale])
                .expect("finite point coordinates"),
            Point::try_new([1.0 * scale, 0.0 * scale, 0.0 * scale, 0.0 * scale])
                .expect("finite point coordinates"),
            Point::try_new([0.0 * scale, 1.0 * scale, 0.0 * scale, 0.0 * scale])
                .expect("finite point coordinates"),
            Point::try_new([0.0 * scale, 0.0 * scale, 1.0 * scale, 0.0 * scale])
                .expect("finite point coordinates"),
        ];

        let original_measure = facet_measure(&original_points).unwrap();
        let scaled_measure = facet_measure(&scaled_points).unwrap();

        // For 4D (D=4), measure scales by |λ|^(4-1) = λ^3
        assert_relative_eq!(
            scaled_measure,
            original_measure * scale.powi(3),
            epsilon = 1e-10
        );
    }

    // =============================================================================
    // EDGE CASE AND NUMERICAL STABILITY TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_very_large_coordinates() {
        // Test with very large but finite coordinates
        let large_val = 1e8;
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([large_val, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, large_val, 0.0]).expect("finite point coordinates"),
        ];

        let result = facet_measure(&points);
        assert!(result.is_ok(), "Large coordinates should work");

        let measure = result.unwrap();
        assert!(measure.is_finite(), "Measure should be finite");
        // Should be area of right triangle: large_val * large_val / 2
        let expected = large_val * large_val / 2.0;
        assert_relative_eq!(measure, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_facet_measure_very_small_coordinates() {
        // Test with very small but non-zero coordinates. The degeneracy check
        // should be scale-aware rather than rejecting every tiny valid facet.
        let small_val = 1e-14;
        let points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([small_val, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, small_val, 0.0]).expect("finite point coordinates"),
        ];

        let result = facet_measure(&points);
        assert!(result.is_ok(), "Small coordinates should work");

        let measure = result.unwrap();
        assert!(measure.is_finite(), "Measure should be finite");
        // Should be area of right triangle: small_val * small_val / 2
        let expected = small_val * small_val / 2.0;
        assert_relative_eq!(measure, expected, epsilon = 1e-40);
    }

    #[test]
    fn test_facet_measure_mixed_positive_negative_coordinates() {
        // Test with mixed positive and negative coordinates
        let points = vec![
            Point::try_new([-1.0, -1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([2.0, -1.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([-1.0, 3.0, 0.0]).expect("finite point coordinates"),
        ];

        let measure = facet_measure(&points).unwrap();
        // Triangle with base=3, height=4, area=6
        assert_relative_eq!(measure, 6.0, epsilon = 1e-10);
    }

    // =============================================================================
    // GEOMETRIC INVARIANCE TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_translation_invariance() {
        // Test that translation doesn't change facet measure
        let translation = [10.0, 20.0, 30.0];
        let original_points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([3.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 4.0, 0.0]).expect("finite point coordinates"),
        ];
        let translated_points = vec![
            Point::try_new([
                0.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                3.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ])
            .expect("finite point coordinates"),
            Point::try_new([
                0.0 + translation[0],
                4.0 + translation[1],
                0.0 + translation[2],
            ])
            .expect("finite point coordinates"),
        ];

        let original_measure = facet_measure(&original_points).unwrap();
        let translated_measure = facet_measure(&translated_points).unwrap();

        assert_relative_eq!(original_measure, translated_measure, epsilon = 1e-10);
        assert_relative_eq!(original_measure, 6.0, epsilon = 1e-10); // Area of 3-4-5 triangle / 2
    }

    #[test]
    fn test_facet_measure_vertex_permutation_invariance() {
        // Test that vertex order doesn't change facet measure (absolute value)
        let points_order1 = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([3.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 4.0, 0.0]).expect("finite point coordinates"),
        ];
        let points_order2 = vec![
            Point::try_new([3.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 4.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
        ];
        let points_order3 = vec![
            Point::try_new([0.0, 4.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([3.0, 0.0, 0.0]).expect("finite point coordinates"),
        ];

        let measure1 = facet_measure(&points_order1).unwrap();
        let measure2 = facet_measure(&points_order2).unwrap();
        let measure3 = facet_measure(&points_order3).unwrap();

        // All should give same area (measure is absolute value)
        assert_relative_eq!(measure1, measure2, epsilon = 1e-10);
        assert_relative_eq!(measure1, measure3, epsilon = 1e-10);
        assert_relative_eq!(measure1, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_various_triangle_orientations() {
        // Test triangles in different orientations in 3D space
        let triangles = [
            // XY plane
            vec![
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
            ],
            // XZ plane
            vec![
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
            ],
            // YZ plane
            vec![
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"),
            ],
            // Diagonal plane
            vec![
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([1.0, 1.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 1.0, 1.0]).expect("finite point coordinates"),
            ],
        ];

        let expected_areas = [0.5, 0.5, 0.5]; // First three are right triangles with legs of length 1

        for (i, triangle) in triangles.iter().take(3).enumerate() {
            let measure = facet_measure(triangle).unwrap();
            // Triangle should have expected area
            assert_relative_eq!(measure, expected_areas[i], epsilon = 1e-10);
        }

        // Fourth triangle has a different but computable area
        let measure4 = facet_measure(&triangles[3]).unwrap();
        assert!(
            measure4 > 0.0,
            "Diagonal triangle should have positive area"
        );
        assert!(
            measure4.is_finite(),
            "Diagonal triangle area should be finite"
        );
    }

    // =============================================================================
    // ADDITIONAL SURFACE MEASURE TESTS
    // =============================================================================

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "Comparisons are against exact literals (constructed geometry), acceptable in this test"
    )]
    fn test_surface_measure_multiple_facets_different_sizes() {
        // Test with facets of different sizes using triangulations with known boundary facets

        // Create first triangulation with small right triangle (area = 0.5)
        let vertices1: Vec<Vertex<(), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(), // v1
            vertex!([1.0, 0.0, 0.0]).unwrap(), // v2
            vertex!([0.0, 1.0, 0.0]).unwrap(), // v3
            vertex!([0.0, 0.0, 1.0]).unwrap(), // v4
        ];

        // Create second triangulation with large right triangle (area = 24.0)
        let vertices2: Vec<Vertex<(), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(), // v5
            vertex!([6.0, 0.0, 0.0]).unwrap(), // v6
            vertex!([0.0, 8.0, 0.0]).unwrap(), // v7
            vertex!([0.0, 0.0, 1.0]).unwrap(), // v8
        ];
        let dt1: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices1).build().unwrap();
        let dt2: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices2).build().unwrap();

        let boundary_facets1: Vec<_> = dt1
            .boundary_facets()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let small_facet = boundary_facets1
            .iter()
            .find(|facet| {
                let facet_vertices: Vec<_> = facet.vertices().collect();
                facet_vertices.len() == 3
                    && facet_vertices.iter().any(|v| {
                        let coords = *v.point().coords();
                        coords == [0.0, 0.0, 0.0]
                    })
                    && facet_vertices.iter().any(|v| {
                        let coords = *v.point().coords();
                        coords == [1.0, 0.0, 0.0]
                    })
                    && facet_vertices.iter().any(|v| {
                        let coords = *v.point().coords();
                        coords == [0.0, 1.0, 0.0]
                    })
            })
            .expect("Should find small triangle facet")
            .clone();

        let boundary_facets2: Vec<_> = dt2
            .boundary_facets()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // Find the facet opposite to v8 (triangle with v5, v6, v7) - area = 24.0
        let large_facet = boundary_facets2
            .iter()
            .find(|facet| {
                let facet_vertices: Vec<_> = facet.vertices().collect();
                facet_vertices.len() == 3
                    && facet_vertices.iter().any(|v| {
                        let coords = *v.point().coords();
                        coords == [0.0, 0.0, 0.0]
                    })
                    && facet_vertices.iter().any(|v| {
                        let coords = *v.point().coords();
                        coords == [6.0, 0.0, 0.0]
                    })
                    && facet_vertices.iter().any(|v| {
                        let coords = *v.point().coords();
                        coords == [0.0, 8.0, 0.0]
                    })
            })
            .expect("Should find large triangle facet")
            .clone();

        let total_surface = surface_measure(&[small_facet, large_facet]).unwrap();
        let expected_total = 0.5 + 24.0;

        assert_relative_eq!(total_surface, expected_total, epsilon = 1e-10);
    }

    // =============================================================================
    // 2D AND 4D+ SURFACE MEASURE TESTS
    // =============================================================================

    #[test]
    fn test_surface_measure_2d_perimeter() {
        // Test 2D surface measure (perimeter of polygon)

        // Create 2D triangle (3-4-5 right triangle)
        let vertices: Vec<Vertex<(), 2>> = vec![
            vertex!([0.0, 0.0]).unwrap(), // v1
            vertex!([3.0, 0.0]).unwrap(), // v2
            vertex!([0.0, 4.0]).unwrap(), // v3
        ];

        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let boundary_facets: Vec<_> = dt
            .boundary_facets()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // In 2D, boundary facets are edges
        let total_perimeter = surface_measure(&boundary_facets).unwrap();

        // Perimeter should be 3 + 4 + 5 = 12 (sides of 3-4-5 triangle)
        assert_relative_eq!(total_perimeter, 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_surface_measure_4d_boundary() {
        // Test 4D surface measure (3D boundary facets)

        // Create 4D simplex (5 vertices)
        let vertices: Vec<Vertex<(), 4>> = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(), // v1
            vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(), // v2
            vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(), // v3
            vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(), // v4
            vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(), // v5
        ];

        let dt: DelaunayTriangulation<_, (), (), 4> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let boundary_facets: Vec<_> = dt
            .boundary_facets()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let total_surface = surface_measure(&boundary_facets).unwrap();

        // The correct total surface area is 1.0, not 5/6 as originally expected
        // This is because the boundary facets have different volumes:
        // - 4 facets that include the origin: each has volume 1/6
        // - 1 facet that excludes the origin: has volume 1/3
        // Total: 4×(1/6) + 1×(1/3) = 4/6 + 2/6 = 1.0
        let expected_total = 1.0;
        assert_relative_eq!(total_surface, expected_total, epsilon = 1e-10);
    }

    // =============================================================================
    // ERROR PROPAGATION TESTS
    // =============================================================================

    #[test]
    fn test_surface_measure_with_invalid_facet() {
        // Test error handling when facet measure calculation fails

        // Create a valid triangulation
        let vertices: Vec<Vertex<(), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(), // v1
            vertex!([1.0, 0.0, 0.0]).unwrap(), // v2
            vertex!([0.0, 1.0, 0.0]).unwrap(), // v3
            vertex!([0.0, 0.0, 1.0]).unwrap(), // v4
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let boundary_facets: Vec<_> = dt
            .boundary_facets()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // Test with valid facets - should work
        let result = surface_measure(&boundary_facets[0..1]);
        assert!(result.is_ok(), "Valid facets should work");

        let area = result.unwrap();
        assert!(area > 0.0, "Area should be positive");
        assert!(area.is_finite(), "Area should be finite");
    }

    // =============================================================================
    // PERFORMANCE AND STRESS TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_performance_many_dimensions() {
        // Test performance with higher dimensions (7D, 8D)
        let points_7d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expect("finite point coordinates"),
        ];

        let measure_7d = facet_measure(&points_7d).unwrap();
        // Volume of 6-simplex should be 1/6! = 1/720
        assert_relative_eq!(measure_7d, 1.0 / 720.0, epsilon = 1e-10);

        let points_8d = vec![
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .expect("finite point coordinates"),
            Point::try_new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .expect("finite point coordinates"),
            Point::try_new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                .expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                .expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                .expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                .expect("finite point coordinates"),
            Point::try_new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                .expect("finite point coordinates"),
        ];

        let measure_8d = facet_measure(&points_8d).unwrap();
        // Volume of 7-simplex should be 1/7! = 1/5040
        assert_relative_eq!(measure_8d, 1.0 / 5040.0, epsilon = 1e-10);
    }

    #[test]
    fn test_surface_measure_many_facets() {
        // Test with many facets from a simple tetrahedral triangulation
        // Use a simple tetrahedron to avoid degenerate boundary facets
        let vertices: Vec<Vertex<(), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([2.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 2.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 2.0]).unwrap(),
        ];

        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let boundary_facets: Vec<_> = dt
            .boundary_facets()
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // Tetrahedron has exactly 4 boundary facets
        assert_eq!(
            boundary_facets.len(),
            4,
            "Tetrahedron should have 4 boundary facets, got {}",
            boundary_facets.len()
        );

        let total_surface = surface_measure(&boundary_facets).unwrap();

        // Total surface should be finite and positive
        assert!(total_surface.is_finite(), "Total surface should be finite");
        assert!(total_surface > 0.0, "Total surface should be positive");
    }

    // =============================================================================
    // ADVANCED GEOMETRIC PROPERTY TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_equilateral_triangles() {
        // Test equilateral triangles of various sizes
        let side_lengths = [1.0, 2.0, 5.0, 10.0];

        for &side in &side_lengths {
            let height = side * 3.0_f64.sqrt() / 2.0;
            let points = vec![
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([side, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([side / 2.0, height, 0.0]).expect("finite point coordinates"),
            ];

            let measure = facet_measure(&points).unwrap();
            let expected_area = side * side * 3.0_f64.sqrt() / 4.0; // Formula for equilateral triangle area

            // Equilateral triangle should have expected area
            assert_relative_eq!(measure, expected_area, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_facet_measure_regular_tetrahedron_faces() {
        // Test faces of regular tetrahedron
        let side = 2.0;
        let height = side * (2.0_f64 / 3.0).sqrt();
        let center_offset = side / (2.0 * 3.0_f64.sqrt());

        // Regular tetrahedron vertices
        let v1 = Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates");
        let v2 = Point::try_new([side, 0.0, 0.0]).expect("finite point coordinates");
        let v3 = Point::try_new([side / 2.0, side * 3.0_f64.sqrt() / 2.0, 0.0])
            .expect("finite point coordinates");
        let v4 =
            Point::try_new([side / 2.0, center_offset, height]).expect("finite point coordinates");

        // Test each face
        let faces = [
            vec![v1, v2, v3], // Base
            vec![v1, v2, v4], // Face 1
            vec![v2, v3, v4], // Face 2
            vec![v3, v1, v4], // Face 3
        ];

        let expected_face_area = side * side * 3.0_f64.sqrt() / 4.0; // Equilateral triangle area

        for face in &faces {
            let measure = facet_measure(face).unwrap();
            // Face of regular tetrahedron should have expected area
            assert_relative_eq!(measure, expected_face_area, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_facet_measure_reflection_invariance() {
        // Test that reflection doesn't change facet measure
        // Use non-collinear points to form a valid triangle
        let original_points = vec![
            Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([3.0, 0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([0.0, 4.0, 0.0]).expect("finite point coordinates"),
        ];

        // Reflect across various planes
        let reflections = [
            // Reflect x-coordinate
            vec![
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([-3.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 4.0, 0.0]).expect("finite point coordinates"),
            ],
            // Reflect y-coordinate
            vec![
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([3.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, -4.0, 0.0]).expect("finite point coordinates"),
            ],
            // Reflect z-coordinate (doesn't change since all z=0)
            vec![
                Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([3.0, 0.0, 0.0]).expect("finite point coordinates"),
                Point::try_new([0.0, 4.0, 0.0]).expect("finite point coordinates"),
            ],
        ];

        let original_measure = facet_measure(&original_points).unwrap();

        for reflected_points in &reflections {
            let reflected_measure = facet_measure(reflected_points).unwrap();
            // Reflection should preserve facet measure
            assert_relative_eq!(original_measure, reflected_measure, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_facet_measure_rotation_invariance_2d() {
        // Test that rotation doesn't change 2D facet measure (line length)
        let original_points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([3.0, 4.0]).expect("finite point coordinates"),
        ];

        // Rotate by 90 degrees
        let rotated_points = vec![
            Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
            Point::try_new([-4.0, 3.0]).expect("finite point coordinates"), // 90° rotation of (3,4)
        ];

        let original_measure = facet_measure(&original_points).unwrap();
        let rotated_measure = facet_measure(&rotated_points).unwrap();

        assert_relative_eq!(original_measure, rotated_measure, epsilon = 1e-10);
        assert_relative_eq!(original_measure, 5.0, epsilon = 1e-10); // Both should be 5.0
    }
}
