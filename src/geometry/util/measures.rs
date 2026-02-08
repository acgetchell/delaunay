//! Geometric measure computations for simplices.
//!
//! This module provides functions for computing volumes, surface measures,
//! and quality metrics of simplices.

#![forbid(unsafe_code)]

use super::conversions::{safe_coords_to_f64, safe_scalar_from_f64, safe_usize_to_scalar};
use super::norms::hypot;
use crate::core::facet::FacetView;
use crate::core::traits::data_type::DataType;
use crate::geometry::matrix::{Matrix, matrix_get, matrix_set};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{Coordinate, ScalarAccumulative, ScalarSummable};
use la_stack::{DEFAULT_SINGULAR_TOL, LaError};
use num_traits::Float;
use std::ops::AddAssign;

// Re-export error types
pub use super::{CircumcenterError, SurfaceMeasureError, ValueConversionError};

/// Calculate the volume of a D-dimensional simplex.
///
/// This function computes the D-dimensional volume of a simplex formed by D+1 points.
/// The volume is calculated using the Gram matrix determinant method, which is
/// numerically stable and generalizes correctly to arbitrary dimensions.
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
/// The D-dimensional volume of the simplex, or an error if calculation fails
///
/// # Errors
///
/// Returns an error if:
/// - Wrong number of points provided (expected D+1)
/// - Points are degenerate (volume would be zero)
/// - Coordinate conversion fails
///
/// # Examples
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::geometry::util::simplex_volume;
/// use approx::assert_relative_eq;
///
/// // 2D: Triangle area
/// let triangle = vec![
///     Point::new([0.0, 0.0]),
///     Point::new([1.0, 0.0]),
///     Point::new([0.0, 1.0]),
/// ];
/// let area = simplex_volume(&triangle).unwrap();
/// assert_relative_eq!(area, 0.5, epsilon = 1e-10); // Area = 1*1/2 = 0.5
///
/// // 3D: Tetrahedron volume
/// let tetrahedron = vec![
///     Point::new([0.0, 0.0, 0.0]),
///     Point::new([1.0, 0.0, 0.0]),
///     Point::new([0.0, 1.0, 0.0]),
///     Point::new([0.0, 0.0, 1.0]),
/// ];
/// let volume = simplex_volume(&tetrahedron).unwrap();
/// assert_relative_eq!(volume, 1.0/6.0, epsilon = 1e-10); // Volume = 1/6
/// ```
pub fn simplex_volume<T, const D: usize>(points: &[Point<T, D>]) -> Result<T, CircumcenterError>
where
    T: ScalarSummable,
{
    #[cfg(debug_assertions)]
    if std::env::var_os("DELAUNAY_DEBUG_UNUSED_IMPORTS").is_some() {
        eprintln!(
            "measures::simplex_volume called (points_len={}, D={})",
            points.len(),
            D
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
            let length = Float::abs(diff[0]);

            // Check for degeneracy (coincident points)
            let epsilon = T::from(1e-12).ok_or_else(|| {
                CircumcenterError::ValueConversion(ValueConversionError::ConversionFailed {
                    value: "1e-12".to_string(),
                    from_type: "f64",
                    to_type: std::any::type_name::<T>(),
                    details: "Failed to convert epsilon threshold".to_string(),
                })
            })?;
            if length < epsilon {
                return Err(CircumcenterError::MatrixInversionFailed {
                    details: "Degenerate simplex with zero volume (coincident points)".to_string(),
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

            // 2D cross product magnitude: |v1.x * v2.y - v1.y * v2.x|
            let cross_z = v1[0] * v2[1] - v1[1] * v2[0];
            let area = Float::abs(cross_z) / T::from(2).unwrap_or_else(|| T::one() + T::one());

            // Check for degeneracy (collinear points)
            let epsilon = T::from(1e-12).ok_or_else(|| {
                CircumcenterError::ValueConversion(ValueConversionError::ConversionFailed {
                    value: "1e-12".to_string(),
                    from_type: "f64",
                    to_type: std::any::type_name::<T>(),
                    details: "Failed to convert epsilon threshold".to_string(),
                })
            })?;
            if area < epsilon {
                return Err(CircumcenterError::MatrixInversionFailed {
                    details: "Degenerate simplex with zero volume (collinear points)".to_string(),
                });
            }

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

            // Triple scalar product: v1 · (v2 × v3)
            let cross_x = v2[1] * v3[2] - v2[2] * v3[1];
            let cross_y = v2[2] * v3[0] - v2[0] * v3[2];
            let cross_z = v2[0] * v3[1] - v2[1] * v3[0];
            let triple_product = v1[0] * cross_x + v1[1] * cross_y + v1[2] * cross_z;

            // Volume = |triple product| / 6
            let six = T::from(6)
                .unwrap_or_else(|| T::one() + T::one() + T::one() + T::one() + T::one() + T::one());
            let volume = Float::abs(triple_product) / six;

            // Check for degeneracy (coplanar points)
            let epsilon = T::from(1e-12).ok_or_else(|| {
                CircumcenterError::ValueConversion(ValueConversionError::ConversionFailed {
                    value: "1e-12".to_string(),
                    from_type: "f64",
                    to_type: std::any::type_name::<T>(),
                    details: "Failed to convert epsilon threshold".to_string(),
                })
            })?;
            if volume < epsilon {
                return Err(CircumcenterError::MatrixInversionFailed {
                    details: "Degenerate simplex with zero volume (coplanar points)".to_string(),
                });
            }

            Ok(volume)
        }
        _ => {
            // Higher dimensions: Use Gram matrix method
            simplex_volume_gram_matrix::<T, D>(points)
        }
    }
}

/// Clamp and validate a Gram determinant.
///
/// For valid inputs, Gram determinants should be non-negative.
///
/// In this crate we compute Gram determinants via a symmetry-exploiting LDLT factorization
/// (see [`gram_determinant_ldlt`]), so **negative** determinants should not occur for PSD Gram
/// matrices. We still keep a small negative clamp as a defensive check, since other callers may
/// pass in raw determinants.
///
/// This function treats any non-positive determinant as a degenerate simplex:
/// - non-finite determinants error
/// - sufficiently negative determinants error
/// - determinants in `(-1e-12, 0)` are clamped to `0.0`, and **zero always errors**
///
/// In other words, clamping does not “allow near-zero volumes”; it just avoids propagating
/// tiny negative values caused by floating-point noise.
fn clamp_gram_determinant(mut det: f64) -> Result<f64, CircumcenterError> {
    if !det.is_finite() {
        return Err(CircumcenterError::MatrixInversionFailed {
            details: "Gram determinant is non-finite".to_string(),
        });
    }

    // Clamp small negative values to zero (numerical tolerance)
    if det < 0.0 {
        if det > -1e-12 {
            det = 0.0;
        } else {
            return Err(CircumcenterError::MatrixInversionFailed {
                details: "Gram matrix has negative determinant (degenerate simplex)".to_string(),
            });
        }
    }

    // Degenerate case: zero determinant means no volume
    if det == 0.0 {
        return Err(CircumcenterError::MatrixInversionFailed {
            details: "Degenerate simplex with zero volume (collinear or coplanar points)"
                .to_string(),
        });
    }

    Ok(det)
}

/// Compute a Gram determinant using la-stack's stack-allocated LDLT factorization.
///
/// This mirrors the existing `crate::geometry::matrix::determinant` behavior:
/// - singular/degenerate => 0.0
/// - non-finite => NaN
#[inline]
fn gram_determinant_ldlt<const D: usize>(gram_matrix: Matrix<D>) -> f64 {
    match gram_matrix.ldlt(DEFAULT_SINGULAR_TOL) {
        Ok(ldlt) => ldlt.det(),
        Err(LaError::Singular { .. }) => 0.0,
        Err(LaError::NonFinite { .. }) => f64::NAN,
    }
}

/// Calculate the volume of a D-dimensional simplex using the Gram matrix method.
///
/// This is a helper function that implements the general Gram matrix approach
/// for computing simplex volumes in arbitrary dimensions.
///
/// # Arguments
///
/// * `points` - Points defining the simplex (must have exactly D+1 points)
///
/// # Returns
///
/// The volume of the simplex, or an error if calculation fails
fn simplex_volume_gram_matrix<T, const D: usize>(
    points: &[Point<T, D>],
) -> Result<T, CircumcenterError>
where
    T: ScalarSummable,
{
    // Convert points to f64 and create edge vectors from first point to all others
    let p0_coords = points[0].coords();
    let p0_f64 = safe_coords_to_f64(p0_coords)?;

    let mut edge_matrix = crate::geometry::matrix::Matrix::<D>::zero();
    for (row, point) in points.iter().skip(1).enumerate() {
        let point_f64 = safe_coords_to_f64(point.coords())?;

        for (j, (&p, &p0)) in point_f64.iter().zip(p0_f64.iter()).enumerate() {
            matrix_set(&mut edge_matrix, row, j, p - p0);
        }
    }

    // Compute Gram matrix G where G[i,j] = edge_i · edge_j
    let mut gram_matrix = crate::geometry::matrix::Matrix::<D>::zero();
    for i in 0..D {
        for j in 0..D {
            let mut dot_product = 0.0;
            for k in 0..D {
                dot_product += matrix_get(&edge_matrix, i, k) * matrix_get(&edge_matrix, j, k);
            }
            matrix_set(&mut gram_matrix, i, j, dot_product);
        }
    }

    // Compute Gram determinant with clamping (LDLT exploits symmetry / PSD structure).
    let det = clamp_gram_determinant(gram_determinant_ldlt(gram_matrix))?;

    let volume_f64 = {
        let sqrt_det = det.sqrt();
        // Compute D! in f64 to avoid usize overflow/precision issues
        let mut d_fact = 1.0f64;
        for k in 2..=D {
            let k_f64 = safe_usize_to_scalar::<f64>(k).map_err(|e| {
                CircumcenterError::ValueConversion(ValueConversionError::ConversionFailed {
                    value: k.to_string(),
                    from_type: "usize",
                    to_type: "f64",
                    details: e.to_string(),
                })
            })?;
            d_fact *= k_f64;
        }
        sqrt_det / d_fact
    };

    safe_scalar_from_f64(volume_f64).map_err(CircumcenterError::CoordinateConversion)
}

/// Calculate the inradius of a D-dimensional simplex.
///
/// The inradius is the radius of the largest sphere that can be inscribed
/// within the simplex. It is computed using the formula:
///
/// **inradius = D × volume / `surface_area`**
///
/// where `surface_area` is the sum of all (D-1)-dimensional facet volumes.
///
/// # Arguments
///
/// * `points` - Points defining the simplex (must have exactly D+1 points)
///
/// # Returns
///
/// The inradius of the simplex, or an error if calculation fails
///
/// # Errors
///
/// Returns an error if:
/// - Wrong number of points provided (expected D+1)
/// - Simplex is degenerate (zero volume or surface area)
/// - Coordinate conversion fails
///
/// # Examples
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::geometry::util::inradius;
/// use approx::assert_relative_eq;
///
/// // 2D: Equilateral triangle with side length 1
/// let triangle = vec![
///     Point::new([0.0, 0.0]),
///     Point::new([1.0, 0.0]),
///     Point::new([0.5, 0.866025]), // sqrt(3)/2 ≈ 0.866025
/// ];
/// let r_in = inradius(&triangle).unwrap();
/// // For equilateral triangle: inradius ≈ 0.2887 (exact: sqrt(3)/6)
/// assert_relative_eq!(r_in, 0.28867, epsilon = 1e-4);
/// ```
pub fn inradius<T, const D: usize>(points: &[Point<T, D>]) -> Result<T, CircumcenterError>
where
    T: ScalarSummable + AddAssign<T>,
{
    if points.len() != D + 1 {
        return Err(CircumcenterError::InvalidSimplex {
            actual: points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    // Special-case 1D: segment inradius is half the length
    if D == 1 {
        let length = simplex_volume(points)?; // 1D volume = segment length
        return Ok(length / T::from(2).unwrap_or_else(|| T::one() + T::one()));
    }

    // Compute volume
    let volume = simplex_volume(points)?;

    // Check for degenerate simplex (using same epsilon as simplex_volume for consistency)
    let epsilon = T::from(1e-12).ok_or_else(|| {
        CircumcenterError::ValueConversion(ValueConversionError::ConversionFailed {
            value: "1e-12".to_string(),
            from_type: "f64",
            to_type: std::any::type_name::<T>(),
            details: "Failed to convert epsilon threshold".to_string(),
        })
    })?;
    if volume < epsilon {
        return Err(CircumcenterError::MatrixInversionFailed {
            details: format!("Degenerate simplex with volume ≈ {volume:?}"),
        });
    }

    // Compute surface area by summing all (D-1)-dimensional facet volumes
    let mut surface_area = T::zero();
    for i in 0..=D {
        // Create facet by omitting vertex i
        let facet_points: Vec<Point<T, D>> = points
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, p)| *p)
            .collect();

        if facet_points.len() != D {
            continue;
        }

        let facet_area = facet_measure(&facet_points)?;
        surface_area += facet_area;
    }

    // Check for degenerate surface area
    if surface_area < epsilon {
        return Err(CircumcenterError::MatrixInversionFailed {
            details: format!("Degenerate simplex with surface_area ≈ {surface_area:?}"),
        });
    }

    // inradius = D * volume / surface_area
    let d_scalar = T::from(D).ok_or_else(|| {
        CircumcenterError::ValueConversion(ValueConversionError::ConversionFailed {
            value: D.to_string(),
            from_type: "usize",
            to_type: std::any::type_name::<T>(),
            details: "Failed to convert dimension to coordinate type".to_string(),
        })
    })?;

    let inradius = (d_scalar * volume) / surface_area;
    Ok(inradius)
}

/// Calculate the area/volume of a facet defined by a set of points.
///
/// This function calculates the (D-1)-dimensional "area" of a facet in D-dimensional space:
/// - 1D: Point measure (0-dimensional, returns 0)
/// - 2D: Length of line segment (1-dimensional)
/// - 3D: Area of triangle using cross product (2-dimensional)
/// - 4D+: Generalized volume using Gram matrix method
///
/// For dimensions 4 and higher, this function uses the Gram matrix method for
/// mathematically accurate volume computation.
///
/// # Arguments
///
/// * `points` - Points defining the facet (should have exactly D points for (D-1)-dimensional facet)
///
/// # Returns
///
/// The area/volume of the facet, or an error if calculation fails
///
/// # Errors
///
/// Returns an error if:
/// - Wrong number of points provided
/// - Points are degenerate (collinear/coplanar)
/// - Coordinate conversion fails
///
/// # Examples
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::geometry::util::facet_measure;
/// use approx::assert_relative_eq;
///
/// // 2D: Line segment length (1D facet in 2D space)
/// let line_segment = vec![
///     Point::new([0.0, 0.0]),
///     Point::new([3.0, 4.0]),
/// ];
/// let length = facet_measure(&line_segment).unwrap();
/// assert_relative_eq!(length, 5.0, epsilon = 1e-10); // sqrt(3² + 4²) = 5
///
/// // 3D: Triangle area (2D facet in 3D space)
/// let triangle = vec![
///     Point::new([0.0, 0.0, 0.0]),
///     Point::new([3.0, 0.0, 0.0]),
///     Point::new([0.0, 4.0, 0.0]),
/// ];
/// let area = facet_measure(&triangle).unwrap();
/// assert_relative_eq!(area, 6.0, epsilon = 1e-10); // 3*4/2 = 6
/// ```
pub fn facet_measure<T, const D: usize>(points: &[Point<T, D>]) -> Result<T, CircumcenterError>
where
    T: ScalarSummable,
{
    if points.len() != D {
        return Err(CircumcenterError::InvalidSimplex {
            actual: points.len(),
            expected: D,
            dimension: D,
        });
    }

    match D {
        1 => {
            // 1D: Point measure (0-dimensional facet)
            if points.len() != 1 {
                return Err(CircumcenterError::InvalidSimplex {
                    actual: points.len(),
                    expected: 1,
                    dimension: 1,
                });
            }
            // A 0-dimensional point has measure 0
            Ok(T::zero())
        }
        2 => {
            // 2D: Length of line segment (1D facet in 2D space)
            let p0 = points[0].coords();
            let p1 = points[1].coords();

            let diff = [p1[0] - p0[0], p1[1] - p0[1]];
            let length = hypot(&diff);

            // Check for degeneracy (coincident points)
            let epsilon = T::from(1e-12).unwrap_or_else(T::zero);
            if length < epsilon {
                return Err(CircumcenterError::MatrixInversionFailed {
                    details: "Degenerate facet with zero length (coincident points)".to_string(),
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

            // Cross product v1 × v2
            let cross = [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ];

            // Area is |cross product| / 2
            let cross_magnitude = hypot(&cross);
            let area = cross_magnitude / (T::one() + T::one()); // Divide by 2

            // Check for degeneracy (collinear points)
            let epsilon = T::from(1e-12).unwrap_or_else(T::zero);
            if area < epsilon {
                return Err(CircumcenterError::MatrixInversionFailed {
                    details: "Degenerate facet with zero area (collinear points)".to_string(),
                });
            }

            Ok(area)
        }
        4 => {
            // 4D: Volume of tetrahedron (3D facet in 4D space)
            // Use Gram matrix method for correct calculation
            facet_measure_gram_matrix::<T, D>(points)
        }
        _ => {
            // Higher dimensions: Use Gram matrix method for correct calculation
            facet_measure_gram_matrix::<T, D>(points)
        }
    }
}

/// Calculate the area/volume of a (D-1)-dimensional simplex using the Gram matrix method.
///
/// This function implements the mathematically rigorous approach for computing the volume
/// of a (D-1)-dimensional simplex embedded in D-dimensional space using the Gram matrix
/// determinant formula:
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
/// This approach is numerically stable and generalizes correctly to arbitrary dimensions,
/// unlike methods based on recursive determinant expansion which become computationally
/// intractable in high dimensions.
///
/// # Arguments
///
/// * `points` - Points defining the simplex (should have exactly D points for (D-1)-dimensional facet)
///
/// # Returns
///
/// The volume of the simplex, or an error if calculation fails
///
/// # Errors
///
/// Returns an error if:
/// - Matrix operations fail (singular Gram matrix indicates degenerate simplex)
/// - Coordinate conversion fails
/// - Gram matrix determinant is negative (should never happen for valid input)
fn facet_measure_gram_matrix<T, const D: usize>(
    points: &[Point<T, D>],
) -> Result<T, CircumcenterError>
where
    T: ScalarSummable,
{
    // Convert points to f64.
    let mut coords_f64 = [[0.0f64; D]; D];
    for (dst, p) in coords_f64.iter_mut().zip(points.iter()) {
        *dst = safe_coords_to_f64(p.coords())?;
    }

    // Compute Gram determinant with clamping.
    //
    // For a (D-1)-simplex embedded in D dimensions, there are (D-1) edge vectors from
    // one vertex to the remaining vertices, so the Gram matrix is (D-1)×(D-1).
    let gram_dim = D - 1;
    let det = try_with_la_stack_matrix!(gram_dim, |gram_matrix| {
        for i in 0..gram_dim {
            for j in 0..gram_dim {
                let mut dot_product = 0.0;
                for ((&ai, &aj), &a0) in coords_f64[i + 1]
                    .iter()
                    .zip(coords_f64[j + 1].iter())
                    .zip(coords_f64[0].iter())
                {
                    let di = ai - a0;
                    let dj = aj - a0;
                    dot_product += di * dj;
                }
                matrix_set(&mut gram_matrix, i, j, dot_product);
            }
        }

        clamp_gram_determinant(gram_determinant_ldlt(gram_matrix))
    })?;

    let volume_f64 = {
        let sqrt_det = det.sqrt();
        // Compute (D-1)! in f64 using safe conversion
        let mut d_fact = 1.0f64;
        for k in 2..D {
            let k_f64 = safe_usize_to_scalar::<f64>(k).map_err(|e| {
                CircumcenterError::ValueConversion(ValueConversionError::ConversionFailed {
                    value: k.to_string(),
                    from_type: "usize",
                    to_type: "f64",
                    details: e.to_string(),
                })
            })?;
            d_fact *= k_f64;
        }
        sqrt_det / d_fact
    };

    safe_scalar_from_f64(volume_f64).map_err(CircumcenterError::CoordinateConversion)
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
/// Total surface area/volume, or error if any facet calculation fails
///
/// # Errors
///
/// Returns an error if any individual facet measure calculation fails
///
/// # Examples
///
/// ```
/// use delaunay::prelude::query::*;
/// use delaunay::geometry::util::surface_measure;
///
/// // Create a triangulation and calculate surface measure of boundary facets
/// let vertices = vec![
///     vertex!([0.0, 0.0, 0.0]),
///     vertex!([1.0, 0.0, 0.0]),
///     vertex!([0.0, 1.0, 0.0]),
///     vertex!([0.0, 0.0, 1.0]),
/// ];
/// let dt = DelaunayTriangulation::new(&vertices).unwrap();
/// let tds = dt.tds();
///
/// // Get boundary facets as FacetViews
/// let boundary_facets = tds.boundary_facets().unwrap().collect::<Vec<_>>();
///
/// // Calculate surface area
/// let surface_area = surface_measure(&boundary_facets).unwrap();
/// assert!(surface_area > 0.0);
/// ```
pub fn surface_measure<T, U, V, const D: usize>(
    facets: &[FacetView<'_, T, U, V, D>],
) -> Result<T, SurfaceMeasureError>
where
    T: ScalarAccumulative,
    U: DataType,
    V: DataType,
{
    let mut total_measure = T::zero();

    for facet in facets {
        let facet_vertices = facet.vertices();

        // Convert vertices to Points for measure calculation
        let points: Vec<Point<T, D>> = facet_vertices
            .map_err(SurfaceMeasureError::FacetError)?
            .map(|v| {
                let coords = *v.point().coords();
                Point::new(coords)
            })
            .collect();

        let measure = facet_measure(&points).map_err(SurfaceMeasureError::GeometryError)?;
        total_measure += measure;
    }

    Ok(total_measure)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::boundary_analysis::BoundaryAnalysis;
    use crate::core::vertex::Vertex;
    use crate::geometry::point::Point;
    use crate::vertex;
    use approx::assert_relative_eq;

    // =============================================================================
    // SIMPLEX VOLUME TESTS
    // =============================================================================

    #[test]
    fn test_simplex_volume_1d_line_segment() {
        // 1D: Line segment length
        let line = vec![Point::new([0.0]), Point::new([5.0])];
        let volume = simplex_volume(&line).unwrap();
        assert_relative_eq!(volume, 5.0, epsilon = 1e-10);

        // Negative direction
        let line_neg = vec![Point::new([5.0]), Point::new([0.0])];
        let volume_neg = simplex_volume(&line_neg).unwrap();
        assert_relative_eq!(volume_neg, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simplex_volume_2d_triangle() {
        // 2D: Right triangle with legs 3 and 4
        let triangle = vec![
            Point::new([0.0, 0.0]),
            Point::new([3.0, 0.0]),
            Point::new([0.0, 4.0]),
        ];
        let area = simplex_volume(&triangle).unwrap();
        assert_relative_eq!(area, 6.0, epsilon = 1e-10); // Area = (3*4)/2 = 6

        // Equilateral triangle with side 1
        let equilateral = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.5, 0.866_025]), // sqrt(3)/2
        ];
        let area_eq = simplex_volume(&equilateral).unwrap();
        // Area = sqrt(3)/4 ≈ 0.433013
        assert_relative_eq!(area_eq, 0.433_013, epsilon = 1e-5);
    }

    #[test]
    fn test_simplex_volume_3d_tetrahedron() {
        // 3D: Regular tetrahedron with vertices at unit cube corners
        let tetrahedron = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let volume = simplex_volume(&tetrahedron).unwrap();
        assert_relative_eq!(volume, 1.0 / 6.0, epsilon = 1e-10); // Volume = 1/6
    }

    #[test]
    fn test_simplex_volume_4d_simplex() {
        // 4D: Regular 4-simplex
        let simplex_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];
        let volume = simplex_volume(&simplex_4d).unwrap();
        // 4D simplex volume = 1/4! = 1/24
        assert_relative_eq!(volume, 1.0 / 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simplex_volume_degenerate() {
        // Degenerate triangle (collinear points) should return an error
        let collinear = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 1.0]),
            Point::new([2.0, 2.0]),
        ];
        let result = simplex_volume(&collinear);
        assert!(result.is_err(), "Degenerate simplex should return an error");
    }

    #[test]
    fn test_simplex_volume_wrong_point_count() {
        // Wrong number of points for 2D
        let points = vec![Point::new([0.0, 0.0]), Point::new([1.0, 0.0])];
        let result = simplex_volume::<f64, 2>(&points);
        assert!(result.is_err());
    }

    // =============================================================================
    // INRADIUS TESTS
    // =============================================================================

    #[test]
    fn test_inradius_2d_equilateral_triangle() {
        // Equilateral triangle with side 1
        let triangle = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.5, 0.866_025]), // sqrt(3)/2
        ];
        let r_in = inradius(&triangle).unwrap();
        // For equilateral triangle: inradius = sqrt(3)/6 ≈ 0.28867513
        assert_relative_eq!(r_in, 0.288_675_13, epsilon = 1e-5);
    }

    #[test]
    fn test_inradius_2d_right_triangle() {
        // Right triangle with legs 3 and 4
        let triangle = vec![
            Point::new([0.0, 0.0]),
            Point::new([3.0, 0.0]),
            Point::new([0.0, 4.0]),
        ];
        let r_in = inradius(&triangle).unwrap();
        // For right triangle: inradius = (a+b-c)/2 = (3+4-5)/2 = 1.0
        assert_relative_eq!(r_in, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inradius_3d_regular_tetrahedron() {
        // Regular tetrahedron at unit cube corners
        let tetrahedron = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let r_in = inradius(&tetrahedron).unwrap();
        // For this tetrahedron: inradius ≈ 0.2113
        assert_relative_eq!(r_in, 0.2113, epsilon = 1e-3);
    }

    #[test]
    fn test_inradius_degenerate() {
        // Degenerate triangle (collinear points)
        let collinear = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([2.0, 0.0]),
        ];
        let result = inradius(&collinear);
        assert!(result.is_err()); // Should fail for degenerate simplex
    }

    // =============================================================================
    // BASIC FACET MEASURE TESTS (BY DIMENSION)
    // =============================================================================

    #[test]
    fn test_facet_measure_1d_point() {
        // 1D facet is a single point (0-dimensional) - measure should be 0
        let points = vec![Point::new([5.0])];
        let measure = facet_measure(&points).unwrap();
        assert_relative_eq!(measure, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_2d_line_segment() {
        // 2D: Line segment (1D facet in 2D space) - 3-4-5 triangle
        let points = vec![Point::new([0.0, 0.0]), Point::new([3.0, 4.0])];
        let measure = facet_measure(&points).unwrap();
        // Length should be sqrt(3² + 4²) = 5.0
        assert_relative_eq!(measure, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_3d_triangle_right_angle() {
        // 3D: Right triangle (area = 1/2 * base * height)
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([3.0, 0.0, 0.0]),
            Point::new([0.0, 4.0, 0.0]),
        ];
        let measure = facet_measure(&points).unwrap();
        // Area should be 3 * 4 / 2 = 6.0
        assert_relative_eq!(measure, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_4d_tetrahedron() {
        // 4D: Unit tetrahedron (3D facet in 4D space)
        let points = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
        ];
        let measure = facet_measure(&points).unwrap();

        // For a unit tetrahedron in 4D with vertices at origin and 3 unit vectors,
        // the volume should be 1/3! = 1/6
        // This is a 3-dimensional simplex in 4D space
        assert_relative_eq!(measure, 1.0 / 6.0, epsilon = 1e-10);
    }

    fn gram_det_from_edges<const AMBIENT: usize>(
        edges: &[[f64; AMBIENT]],
    ) -> Result<f64, CircumcenterError> {
        let k = edges.len();

        try_with_la_stack_matrix!(k, |gram_matrix| {
            for i in 0..k {
                for j in 0..k {
                    let mut dot_product = 0.0;
                    for (&a, &b) in edges[i].iter().zip(edges[j].iter()) {
                        dot_product += a * b;
                    }
                    matrix_set(&mut gram_matrix, i, j, dot_product);
                }
            }

            clamp_gram_determinant(gram_determinant_ldlt(gram_matrix))
        })
    }

    #[test]
    fn test_gram_determinant_ldlt_known_spd() {
        // Symmetric positive-definite matrix with known determinant.
        let gram = Matrix::<2>::from_rows([[4.0, 2.0], [2.0, 3.0]]);
        let det = gram_determinant_ldlt(gram);
        assert_relative_eq!(det, 8.0, epsilon = 1e-12);
    }

    #[test]
    fn test_gram_determinant_parallel_edges_errors() {
        let edges = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        assert!(gram_det_from_edges(&edges).is_err());
    }

    #[test]
    fn test_clamp_gram_determinant_tiny_negative_errors() {
        assert!(clamp_gram_determinant(-1e-13).is_err());
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

                let det = gram_det_from_edges::<$dim>(&edges).unwrap();
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

                let det = gram_det_from_edges::<$dim>(&edges).unwrap();
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
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
        ];
        let area_3d = facet_measure(&triangle_3d).unwrap();
        if std::env::var_os("TEST_DEBUG").is_some() {
            println!("3D triangle area: {area_3d} (expected: 0.5)");
        }

        // Test 1b: Nearly singular triangle should not error due to tiny negative det
        let eps = 1e-10;
        let near_singular = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([1.0, eps, 0.0]),
        ];
        let area_ns = facet_measure(&near_singular).unwrap();
        assert!(area_ns >= 0.0);

        // Test 2: Same triangle but use direct Gram matrix calculation
        let area_3d_gram = facet_measure_gram_matrix::<f64, 3>(&triangle_3d).unwrap();
        if std::env::var_os("TEST_DEBUG").is_some() {
            println!("3D triangle area (Gram): {area_3d_gram} (expected: 0.5)");
        }

        // Test 3: Unit tetrahedron in 4D - should be 1/6 ≈ 0.167
        let tetrahedron_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
        ];
        let volume_4d = facet_measure(&tetrahedron_4d).unwrap();
        if std::env::var_os("TEST_DEBUG").is_some() {
            println!(
                "4D tetrahedron volume: {} (expected: {})",
                volume_4d,
                1.0 / 6.0
            );
        }

        // Test 4: Manual calculation for the 4D tetrahedron
        let volume_4d_gram = facet_measure_gram_matrix::<f64, 4>(&tetrahedron_4d).unwrap();
        if std::env::var_os("TEST_DEBUG").is_some() {
            println!(
                "4D tetrahedron volume (Gram): {} (expected: {})",
                volume_4d_gram,
                1.0 / 6.0
            );
        }
    }

    #[test]
    fn test_facet_measure_5d_simplex() {
        // 5D: 4-dimensional facet in 5D space (4-simplex volume)
        let points = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
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
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
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
        let points = vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])];
        let result = facet_measure::<f64, 3>(&points);

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
    fn test_facet_measure_zero_area_triangle() {
        // Degenerate triangle (collinear points) - should return an error
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([2.0, 0.0, 0.0]), // Collinear
        ];
        let result = facet_measure(&points);

        // Should fail with degenerate error
        assert!(result.is_err(), "Collinear points should return an error");
    }

    #[test]
    fn test_facet_measure_nearly_collinear_points_2d() {
        // Test with points that are nearly collinear in 2D
        let eps = 1e-10;
        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, eps]), // Slightly off the x-axis
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
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([1.0, eps, eps]), // Very close to the line from (0,0,0) to (1,0,0)
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
        // Test with points that are coplanar in 4D (all points in 3D subspace)
        let points = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.5, 0.5, 0.0, 0.0]), // In the same 3D subspace
        ];

        let result = facet_measure(&points);
        // Should fail with degenerate error since all points lie in 3D subspace
        assert!(
            result.is_err(),
            "Degenerate 4D tetrahedron should return an error"
        );
    }

    // =============================================================================
    // SURFACE MEASURE TESTS
    // =============================================================================

    #[test]
    fn test_surface_measure_empty_facets() {
        // Test with empty facet collection
        let facets: Vec<FacetView<'_, f64, (), (), 3>> = vec![];
        let result = surface_measure(&facets).unwrap();

        assert_relative_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "Comparisons are against exact literals (constructed geometry), acceptable in this test"
    )]
    fn test_surface_measure_single_facet() {
        // Test with single triangular facet using TDS boundary facets

        // Create a right triangle tetrahedron
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]), // v1
            vertex!([3.0, 0.0, 0.0]), // v2
            vertex!([0.0, 4.0, 0.0]), // v3
            vertex!([0.0, 0.0, 1.0]), // v4
        ];

        let dt: crate::core::delaunay_triangulation::DelaunayTriangulation<_, (), (), 3> =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let boundary_facets: Vec<_> = dt.tds().boundary_facets().unwrap().collect();

        // Find the facet opposite to v4 (contains vertices v1, v2, v3)
        let target_facet = boundary_facets
            .iter()
            .find(|facet| {
                let facet_vertices: Vec<_> = facet.vertices().unwrap().collect();
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

        let surface_area = surface_measure(&[*target_facet]).unwrap();

        // Should be area of right triangle: 3 * 4 / 2 = 6.0
        assert_relative_eq!(surface_area, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_surface_measure_consistency_with_facet_measure() {
        // Test that surface_measure sum equals sum of individual facet_measures
        // Create a triangulation with 5 vertices and 2 tetrahedra to get both boundary and internal facets

        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]), // v1
            vertex!([1.0, 0.0, 0.0]), // v2
            vertex!([0.0, 1.0, 0.0]), // v3
            vertex!([0.0, 0.0, 1.0]), // v4
            vertex!([1.0, 1.0, 1.0]), // v5
        ];

        let dt: crate::core::delaunay_triangulation::DelaunayTriangulation<_, (), (), 3> =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let boundary_facets: Vec<_> = dt.tds().boundary_facets().unwrap().collect();

        // Take first two boundary facets for testing
        let facet1 = boundary_facets[0];
        let facet2 = boundary_facets[1];

        // Calculate surface measure
        let total_surface = surface_measure(&[facet1, facet2]).unwrap();

        // Calculate individual facet measures and sum them
        let points1: Vec<Point<f64, 3>> = facet1
            .vertices()
            .unwrap()
            .map(|v| {
                let coords = *v.point().coords();
                Point::new(coords)
            })
            .collect();
        let points2: Vec<Point<f64, 3>> = facet2
            .vertices()
            .unwrap()
            .map(|v| {
                let coords = *v.point().coords();
                Point::new(coords)
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
        let original_points = vec![Point::new([0.0, 0.0]), Point::new([1.0, 0.0])];
        let scaled_points = vec![
            Point::new([0.0 * scale, 0.0 * scale]),
            Point::new([1.0 * scale, 0.0 * scale]),
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
            Point::new([0.0, 0.0, 0.0]),
            Point::new([2.0, 0.0, 0.0]),
            Point::new([0.0, 3.0, 0.0]),
        ];
        let scaled_points = vec![
            Point::new([0.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([2.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([0.0 * scale, 3.0 * scale, 0.0 * scale]),
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
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
        ];
        let scaled_points = vec![
            Point::new([0.0 * scale, 0.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([1.0 * scale, 0.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([0.0 * scale, 1.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([0.0 * scale, 0.0 * scale, 1.0 * scale, 0.0 * scale]),
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
            Point::new([0.0, 0.0, 0.0]),
            Point::new([large_val, 0.0, 0.0]),
            Point::new([0.0, large_val, 0.0]),
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
        // Test with very small but non-zero coordinates
        // Use 1e-5 so that area (1e-10/2 = 5e-11) is above epsilon threshold (1e-12)
        let small_val = 1e-5;
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([small_val, 0.0, 0.0]),
            Point::new([0.0, small_val, 0.0]),
        ];

        let result = facet_measure(&points);
        assert!(result.is_ok(), "Small coordinates should work");

        let measure = result.unwrap();
        assert!(measure.is_finite(), "Measure should be finite");
        // Should be area of right triangle: small_val * small_val / 2
        let expected = small_val * small_val / 2.0;
        assert_relative_eq!(measure, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_mixed_positive_negative_coordinates() {
        // Test with mixed positive and negative coordinates
        let points = vec![
            Point::new([-1.0, -1.0, 0.0]),
            Point::new([2.0, -1.0, 0.0]),
            Point::new([-1.0, 3.0, 0.0]),
        ];

        let measure = facet_measure(&points).unwrap();
        // Triangle with base=3, height=4, area=6
        assert_relative_eq!(measure, 6.0, epsilon = 1e-10);
    }

    // =============================================================================
    // COORDINATE TYPE TESTS (f32 vs f64)
    // =============================================================================

    #[test]
    fn test_facet_measure_f32_vs_f64_consistency() {
        // Test that f32 and f64 give similar results (within tolerance)
        let points_f64 = vec![
            Point::new([0.0_f64, 0.0_f64, 0.0_f64]),
            Point::new([3.0_f64, 0.0_f64, 0.0_f64]),
            Point::new([0.0_f64, 4.0_f64, 0.0_f64]),
        ];
        let points_f32 = vec![
            Point::new([0.0_f32, 0.0_f32, 0.0_f32]),
            Point::new([3.0_f32, 0.0_f32, 0.0_f32]),
            Point::new([0.0_f32, 4.0_f32, 0.0_f32]),
        ];

        let measure_f64 = facet_measure(&points_f64).unwrap();
        let measure_f32 = facet_measure(&points_f32).unwrap();

        // Convert f32 result to f64 for comparison
        let measure_f32_as_f64 = f64::from(measure_f32);

        // Should be approximately equal (within f32 precision)
        assert_relative_eq!(measure_f64, measure_f32_as_f64, epsilon = 1e-6);
        assert_relative_eq!(measure_f64, 6.0, epsilon = 1e-10);
        assert_relative_eq!(measure_f32_as_f64, 6.0, epsilon = 1e-6);
    }

    // =============================================================================
    // GEOMETRIC INVARIANCE TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_translation_invariance() {
        // Test that translation doesn't change facet measure
        let translation = [10.0, 20.0, 30.0];
        let original_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([3.0, 0.0, 0.0]),
            Point::new([0.0, 4.0, 0.0]),
        ];
        let translated_points = vec![
            Point::new([
                0.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ]),
            Point::new([
                3.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ]),
            Point::new([
                0.0 + translation[0],
                4.0 + translation[1],
                0.0 + translation[2],
            ]),
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
            Point::new([0.0, 0.0, 0.0]),
            Point::new([3.0, 0.0, 0.0]),
            Point::new([0.0, 4.0, 0.0]),
        ];
        let points_order2 = vec![
            Point::new([3.0, 0.0, 0.0]),
            Point::new([0.0, 4.0, 0.0]),
            Point::new([0.0, 0.0, 0.0]),
        ];
        let points_order3 = vec![
            Point::new([0.0, 4.0, 0.0]),
            Point::new([0.0, 0.0, 0.0]),
            Point::new([3.0, 0.0, 0.0]),
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
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
            ],
            // XZ plane
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 0.0, 1.0]),
            ],
            // YZ plane
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 1.0]),
            ],
            // Diagonal plane
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 1.0, 0.0]),
                Point::new([0.0, 1.0, 1.0]),
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
        let vertices1: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]), // v1
            vertex!([1.0, 0.0, 0.0]), // v2
            vertex!([0.0, 1.0, 0.0]), // v3
            vertex!([0.0, 0.0, 1.0]), // v4
        ];
        let dt1: crate::core::delaunay_triangulation::DelaunayTriangulation<_, (), (), 3> =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices1).unwrap();
        let boundary_facets1: Vec<_> = dt1.tds().boundary_facets().unwrap().collect();

        // Find the facet opposite to v4 (triangle with v1, v2, v3) - area = 0.5
        let small_facet = boundary_facets1
            .iter()
            .find(|facet| {
                let facet_vertices: Vec<_> = facet.vertices().unwrap().collect();
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
            .expect("Should find small triangle facet");

        // Create second triangulation with large right triangle (area = 24.0)
        let vertices2: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]), // v5
            vertex!([6.0, 0.0, 0.0]), // v6
            vertex!([0.0, 8.0, 0.0]), // v7
            vertex!([0.0, 0.0, 1.0]), // v8
        ];
        let dt2: crate::core::delaunay_triangulation::DelaunayTriangulation<_, (), (), 3> =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices2).unwrap();
        let boundary_facets2: Vec<_> = dt2.tds().boundary_facets().unwrap().collect();

        // Find the facet opposite to v8 (triangle with v5, v6, v7) - area = 24.0
        let large_facet = boundary_facets2
            .iter()
            .find(|facet| {
                let facet_vertices: Vec<_> = facet.vertices().unwrap().collect();
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
            .expect("Should find large triangle facet");

        let total_surface = surface_measure(&[*small_facet, *large_facet]).unwrap();
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
        let vertices: Vec<Vertex<f64, (), 2>> = vec![
            vertex!([0.0, 0.0]), // v1
            vertex!([3.0, 0.0]), // v2
            vertex!([0.0, 4.0]), // v3
        ];

        let dt: crate::core::delaunay_triangulation::DelaunayTriangulation<_, (), (), 2> =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let boundary_facets: Vec<_> = dt.tds().boundary_facets().unwrap().collect();

        // In 2D, boundary facets are edges
        let total_perimeter = surface_measure(&boundary_facets).unwrap();

        // Perimeter should be 3 + 4 + 5 = 12 (sides of 3-4-5 triangle)
        assert_relative_eq!(total_perimeter, 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_surface_measure_4d_boundary() {
        // Test 4D surface measure (3D boundary facets)

        // Create 4D simplex (5 vertices)
        let vertices: Vec<Vertex<f64, (), 4>> = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]), // v1
            vertex!([1.0, 0.0, 0.0, 0.0]), // v2
            vertex!([0.0, 1.0, 0.0, 0.0]), // v3
            vertex!([0.0, 0.0, 1.0, 0.0]), // v4
            vertex!([0.0, 0.0, 0.0, 1.0]), // v5
        ];

        let dt: crate::core::delaunay_triangulation::DelaunayTriangulation<_, (), (), 4> =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let boundary_facets: Vec<_> = dt.tds().boundary_facets().unwrap().collect();

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
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]), // v1
            vertex!([1.0, 0.0, 0.0]), // v2
            vertex!([0.0, 1.0, 0.0]), // v3
            vertex!([0.0, 0.0, 1.0]), // v4
        ];

        let dt: crate::core::delaunay_triangulation::DelaunayTriangulation<_, (), (), 3> =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let boundary_facets: Vec<_> = dt.tds().boundary_facets().unwrap().collect();

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
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ];

        let measure_7d = facet_measure(&points_7d).unwrap();
        // Volume of 6-simplex should be 1/6! = 1/720
        assert_relative_eq!(measure_7d, 1.0 / 720.0, epsilon = 1e-10);

        let points_8d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ];

        let measure_8d = facet_measure(&points_8d).unwrap();
        // Volume of 7-simplex should be 1/7! = 1/5040
        assert_relative_eq!(measure_8d, 1.0 / 5040.0, epsilon = 1e-10);
    }

    #[test]
    fn test_surface_measure_many_facets() {
        // Test with many facets from a simple tetrahedral triangulation
        // Use a simple tetrahedron to avoid degenerate boundary facets
        let vertices: Vec<Vertex<f64, (), 3>> = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([2.0, 0.0, 0.0]),
            vertex!([0.0, 2.0, 0.0]),
            vertex!([0.0, 0.0, 2.0]),
        ];

        let dt: crate::core::delaunay_triangulation::DelaunayTriangulation<_, (), (), 3> =
            crate::core::delaunay_triangulation::DelaunayTriangulation::new(&vertices).unwrap();
        let boundary_facets: Vec<_> = dt.tds().boundary_facets().unwrap().collect();

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
                Point::new([0.0, 0.0, 0.0]),
                Point::new([side, 0.0, 0.0]),
                Point::new([side / 2.0, height, 0.0]),
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
        let v1 = Point::new([0.0, 0.0, 0.0]);
        let v2 = Point::new([side, 0.0, 0.0]);
        let v3 = Point::new([side / 2.0, side * 3.0_f64.sqrt() / 2.0, 0.0]);
        let v4 = Point::new([side / 2.0, center_offset, height]);

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
            Point::new([0.0, 0.0, 0.0]),
            Point::new([3.0, 0.0, 0.0]),
            Point::new([0.0, 4.0, 0.0]),
        ];

        // Reflect across various planes
        let reflections = [
            // Reflect x-coordinate
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([-3.0, 0.0, 0.0]),
                Point::new([0.0, 4.0, 0.0]),
            ],
            // Reflect y-coordinate
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([3.0, 0.0, 0.0]),
                Point::new([0.0, -4.0, 0.0]),
            ],
            // Reflect z-coordinate (doesn't change since all z=0)
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([3.0, 0.0, 0.0]),
                Point::new([0.0, 4.0, 0.0]),
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
        let original_points = vec![Point::new([0.0, 0.0]), Point::new([3.0, 4.0])];

        // Rotate by 90 degrees
        let rotated_points = vec![
            Point::new([0.0, 0.0]),
            Point::new([-4.0, 3.0]), // 90° rotation of (3,4)
        ];

        let original_measure = facet_measure(&original_points).unwrap();
        let rotated_measure = facet_measure(&rotated_points).unwrap();

        assert_relative_eq!(original_measure, rotated_measure, epsilon = 1e-10);
        assert_relative_eq!(original_measure, 5.0, epsilon = 1e-10); // Both should be 5.0
    }

    #[test]
    fn test_facet_measure_gram_matrix_degenerate() {
        // Test degenerate simplex (collinear points)
        let degenerate_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([2.0, 0.0, 0.0]), // All collinear
        ];

        let result = facet_measure(&degenerate_points);
        // This should either return 0 or an error depending on numerical precision
        if let Ok(measure) = result {
            assert_relative_eq!(measure, 0.0, epsilon = 1e-10);
        }
        // Also acceptable for degenerate case if Err
    }
}
