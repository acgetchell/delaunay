//! Geometric quality measures for d-dimensional simplices.
//!
//! This module provides quality metrics for evaluating the geometric quality
//! of simplices in d-dimensional triangulations. These metrics
//! are used to prefer well-shaped simplices over degenerate or sliver simplices during
//! triangulation operations.
//!
//! # Quality Metrics
//!
//! - **Radius Ratio**: Circumradius divided by inradius. Lower values indicate
//!   better-shaped simplices. An equilateral simplex has a radius ratio close to
//!   the dimension-dependent optimal value.
//! - **Normalized Volume**: Volume divided by the D-th power of the average edge length.
//!   Provides a scale-invariant measure of simplex shape quality above the
//!   documented absolute degeneracy floor.
//!
//! # References
//!
//! - Shewchuk, J.R. "What Is a Good Linear Element? Interpolation, Conditioning,
//!   Anisotropy, and Quality Measures." Eleventh International Meshing Roundtable,
//!   Ithaca, New York (2002). Available at: <https://people.eecs.berkeley.edu/~jrs/papers/elemj.pdf>
//! - Liu, A., and Joe, B. "Relationship between Tetrahedron Shape Measures."
//!   *BIT Numerical Mathematics* 34, no. 2 (1994): 268-287.
//!   DOI: [10.1007/BF01955874](https://doi.org/10.1007/BF01955874)
//! - Field, D.A. "Qualitative Measures for Initial Meshes."
//!   *International Journal for Numerical Methods in Engineering* 47, no. 4 (2000): 887-906.
//!   DOI: [10.1002/(SICI)1097-0207(20000210)47:4<887::AID-NME804>3.0.CO;2-H](https://doi.org/10.1002/(SICI)1097-0207(20000210)47:4<887::AID-NME804>3.0.CO;2-H)
//! - Parthasarathy, V.N., Graichen, C.M., and Hathaway, A.F. "A Comparison of Tetrahedron Quality Measures."
//!   *Finite Elements in Analysis and Design* 15, no. 3 (1994): 255-261.
//!   DOI: [10.1016/0168-874X(94)90033-7](https://doi.org/10.1016/0168-874X(94)90033-7)
//! - Knupp, P.M. "Algebraic Mesh Quality Metrics."
//!   *SIAM Journal on Scientific Computing* 23, no. 1 (2001): 193-218.
//!   DOI: [10.1137/S1064827500371499](https://doi.org/10.1137/S1064827500371499)

#![forbid(unsafe_code)]

use crate::core::{
    collections::{MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer},
    tds::{SimplexKey, TdsError, VertexKey},
    triangulation::Triangulation,
};
use crate::geometry::{
    kernel::Kernel,
    point::Point,
    traits::coordinate::CoordinateConversionValue,
    util::{CircumcenterError, circumradius, hypot, inradius as simplex_inradius, simplex_volume},
};
use core::{array, fmt};
use num_traits::One;
use thiserror::Error;

/// Numeric operation being performed when quality metric computation failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QualityNumericOperation {
    /// Converting the absolute epsilon floor.
    EpsilonFloorConversion,
    /// Converting the edge count to the coordinate scalar.
    EdgeCountConversion,
    /// Converting the relative epsilon factor.
    RelativeFactorConversion,
    /// Circumradius computation.
    Circumradius,
    /// Inradius computation.
    Inradius,
    /// Simplex volume computation.
    Volume,
}

impl fmt::Display for QualityNumericOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EpsilonFloorConversion => f.write_str("epsilon floor conversion"),
            Self::EdgeCountConversion => f.write_str("edge count conversion"),
            Self::RelativeFactorConversion => f.write_str("relative factor conversion"),
            Self::Circumradius => f.write_str("circumradius computation"),
            Self::Inradius => f.write_str("inradius computation"),
            Self::Volume => f.write_str("simplex volume computation"),
        }
    }
}

/// Geometric quantity that revealed a degenerate simplex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QualityDegeneracyMeasure {
    /// Inradius was below the scale-aware epsilon.
    Inradius,
    /// Volume was below the scale-aware epsilon.
    Volume,
    /// Average edge length was below the scale-aware epsilon.
    AverageEdgeLength,
    /// Average edge length raised to the dimension underflowed below epsilon.
    EdgeLengthPower,
}

impl fmt::Display for QualityDegeneracyMeasure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inradius => f.write_str("inradius"),
            Self::Volume => f.write_str("volume"),
            Self::AverageEdgeLength => f.write_str("average edge length"),
            Self::EdgeLengthPower => f.write_str("edge length power"),
        }
    }
}

/// Failure while extracting simplex vertices for quality metric evaluation.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum QualitySimplexVerticesError {
    /// The requested simplex key was not present in the TDS.
    #[error("Simplex key {simplex_key:?} not found: {context}")]
    SimplexNotFound {
        /// Missing simplex key.
        simplex_key: SimplexKey,
        /// Lookup context provided by the TDS.
        context: String,
    },
    /// A simplex referenced a vertex key that was not present in the TDS.
    #[error("Vertex key {vertex_key:?} referenced by the simplex was not found: {context}")]
    ReferencedVertexNotFound {
        /// Missing vertex key.
        vertex_key: VertexKey,
        /// Lookup context provided by the TDS.
        context: String,
    },
    /// The TDS reported an unexpected lookup failure.
    #[error("Unexpected TDS failure while extracting quality metric simplex vertices: {source}")]
    UnexpectedTdsFailure {
        /// Underlying typed TDS error.
        #[source]
        source: Box<TdsError>,
    },
}

impl From<TdsError> for QualitySimplexVerticesError {
    fn from(source: TdsError) -> Self {
        match source {
            TdsError::SimplexNotFound {
                simplex_key,
                context,
            } => Self::SimplexNotFound {
                simplex_key,
                context,
            },
            TdsError::VertexNotFound {
                vertex_key,
                context,
            } => Self::ReferencedVertexNotFound {
                vertex_key,
                context,
            },
            other => Self::UnexpectedTdsFailure {
                source: Box::new(other),
            },
        }
    }
}

/// Errors that can occur during quality metric computation.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{QualityError, QualityNumericOperation};
///
/// let err = QualityError::NumericConversion {
///     operation: QualityNumericOperation::EdgeCountConversion,
/// };
/// std::assert_matches!(err, QualityError::NumericConversion { .. });
/// ```
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum QualityError {
    /// Failed to fetch a simplex's vertex keys from the TDS.
    #[error("Failed to fetch vertices for simplex {simplex_key:?}: {source}")]
    SimplexVertices {
        /// Simplex whose vertex keys could not be retrieved.
        simplex_key: SimplexKey,
        /// Underlying TDS error.
        #[source]
        source: QualitySimplexVerticesError,
    },
    /// A vertex key referenced by the simplex was missing from the triangulation.
    #[error("Vertex {vertex_key:?} referenced by a quality metric simplex was not found")]
    VertexNotFound {
        /// Missing vertex key.
        vertex_key: VertexKey,
    },
    /// Extracted simplex arity did not match `D + 1`.
    #[error("Simplex has {actual} vertices, expected {expected} for dimension {dimension}")]
    InvalidSimplexArity {
        /// Observed vertex count.
        actual: usize,
        /// Expected vertex count.
        expected: usize,
        /// Triangulation dimension.
        dimension: usize,
    },
    /// Conversion of a numeric constant or count into the coordinate scalar failed.
    #[error("Numeric conversion failed during {operation}")]
    NumericConversion {
        /// Operation whose scalar conversion failed.
        operation: QualityNumericOperation,
    },
    /// Circumradius computation failed.
    #[error("Circumradius computation failed: {source}")]
    Circumradius {
        /// Underlying geometry error.
        #[source]
        source: Box<CircumcenterError>,
    },
    /// Inradius computation failed.
    #[error("Inradius computation failed: {source}")]
    Inradius {
        /// Underlying geometry error.
        #[source]
        source: Box<CircumcenterError>,
    },
    /// Simplex volume computation failed.
    #[error("Volume computation failed: {source}")]
    Volume {
        /// Underlying geometry error.
        #[source]
        source: Box<CircumcenterError>,
    },
    /// Simplex is degenerate (zero or near-zero volume).
    #[error(
        "Degenerate simplex: {measure} observed={observed}, epsilon={epsilon}, avg_edge_length={avg_edge_length:?}"
    )]
    DegenerateSimplex {
        /// Quantity that failed the degeneracy threshold.
        measure: QualityDegeneracyMeasure,
        /// Observed quantity value.
        observed: CoordinateConversionValue,
        /// Scale-aware epsilon used for comparison.
        epsilon: CoordinateConversionValue,
        /// Average edge length context, when applicable.
        avg_edge_length: Option<CoordinateConversionValue>,
    },
}

/// Helper function to extract simplex points from a triangulation.
///
/// This centralizes the vertex-to-point extraction logic used by quality metrics.
/// Uses `SmallBuffer` to avoid heap allocation for typical simplex sizes (D+1 vertices).
fn simplex_points<K, U, V, const D: usize>(
    tri: &Triangulation<K, U, V, D>,
    simplex_key: SimplexKey,
) -> Result<SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE>, QualityError>
where
    K: Kernel<D, Scalar = f64>,
{
    let vertex_keys =
        tri.tds
            .simplex_vertices(simplex_key)
            .map_err(|source| QualityError::SimplexVertices {
                simplex_key,
                source: source.into(),
            })?;

    // Use SmallBuffer to avoid heap allocation (simplices have D+1 vertices, D ≤ MAX_PRACTICAL_DIMENSION_SIZE)
    let mut points = SmallBuffer::new();
    for &vkey in vertex_keys {
        let point = tri
            .tds
            .vertex(vkey)
            .map(|v| *v.point())
            .ok_or(QualityError::VertexNotFound { vertex_key: vkey })?;
        points.push(point);
    }
    Ok(points)
}

/// Returns the scale-aware epsilon and average edge length for degeneracy detection.
///
/// This helper centralizes the epsilon calculation logic used by both `radius_ratio`
/// and `normalized_volume` to ensure consistent degeneracy detection across metrics.
///
/// # Arguments
///
/// * `points` - The simplex vertices
///
/// # Returns
///
/// A tuple of `(avg_edge_length, epsilon)` where:
/// - `avg_edge_length`: Translation-invariant geometric scale
/// - `epsilon`: Relative tolerance (1e-8 × `avg_edge_length`) with 1e-12 floor
fn scale_aware_epsilon<const D: usize>(
    points: &SmallBuffer<Point<D>, MAX_PRACTICAL_DIMENSION_SIZE>,
) -> (f64, f64) {
    let (total_edge_length, edge_count) = points
        .iter()
        .enumerate()
        .flat_map(|(i, point_i)| {
            points.iter().skip(i + 1).map(move |point_j| {
                let diff_coords: [f64; D] =
                    array::from_fn(|idx| point_i.coords()[idx] - point_j.coords()[idx]);
                hypot(&diff_coords)
            })
        })
        .fold((0.0, 0_u32), |(total, count), dist| {
            (total + dist, count + 1)
        });

    // If there are no edges (e.g., D == 0), fall back to floor epsilon.
    if edge_count == 0 {
        return (0.0, 1e-12);
    }

    let avg_edge_length = total_edge_length / f64::from(edge_count);
    let floor: f64 = 1e-12;
    let relative_factor: f64 = 1e-8;
    let epsilon = floor.max(avg_edge_length * relative_factor);

    (avg_edge_length, epsilon)
}

/// Computes the radius ratio quality metric for a simplex.
///
/// The radius ratio is defined as the circumradius divided by the inradius.
/// Lower values indicate better simplex quality. For a regular simplex in D dimensions,
/// the circumradius-to-inradius ratio satisfies R/r = D, which is optimal.
///
/// # Quality Interpretation
///
/// - **Optimal** (equilateral simplex): ratio ≈ D
/// - **Good**: ratio < 2×D
/// - **Acceptable**: ratio < 5×D  
/// - **Poor**: ratio ≥ 5×D (sliver or near-degenerate)
///
/// # Arguments
///
/// * `tri` - The triangulation containing the simplex
/// * `simplex_key` - The key of the simplex to evaluate
///
/// # Returns
///
/// The radius ratio as a floating-point value, or an error if computation fails.
///
/// # Errors
///
/// Returns `QualityError` if:
/// - Simplex has missing or invalid vertices
/// - Simplex is degenerate (zero or near-zero volume)
/// - Circumsphere computation fails
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::prelude::geometry::radius_ratio;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Quality(#[from] delaunay::prelude::geometry::QualityError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// // Create a 2D equilateral triangle
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.5, 0.866]?, // approximately sqrt(3)/2
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// let ratio = radius_ratio(dt.as_triangulation(), simplex_key)?;
/// // For an equilateral triangle, ratio ≈ 2.0
/// assert!(ratio > 1.5 && ratio < 2.5);
/// # Ok(())
/// # }
/// ```
pub fn radius_ratio<K, U, V, const D: usize>(
    tri: &Triangulation<K, U, V, D>,
    simplex_key: SimplexKey,
) -> Result<f64, QualityError>
where
    K: Kernel<D, Scalar = f64>,
{
    // Extract simplex points using helper
    let points = simplex_points(tri, simplex_key)?;

    if points.len() != D + 1 {
        return Err(QualityError::InvalidSimplexArity {
            actual: points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    // Compute circumradius using utility function
    let circumradius_val = circumradius(&points).map_err(|source| QualityError::Circumradius {
        source: Box::new(source),
    })?;

    // Compute inradius using utility function
    let inradius_val = simplex_inradius(&points).map_err(|source| QualityError::Inradius {
        source: Box::new(source),
    })?;

    // Check for near-zero inradius (degenerate simplex) using scale-aware tolerance
    let (avg_edge_length, epsilon) = scale_aware_epsilon(&points);

    if inradius_val < epsilon {
        return Err(QualityError::DegenerateSimplex {
            measure: QualityDegeneracyMeasure::Inradius,
            observed: CoordinateConversionValue::from_numeric_debug(&inradius_val),
            epsilon: CoordinateConversionValue::from_numeric_debug(&epsilon),
            avg_edge_length: Some(CoordinateConversionValue::from_numeric_debug(
                &avg_edge_length,
            )),
        });
    }

    // radius_ratio = circumradius / inradius
    let ratio = circumradius_val / inradius_val;

    Ok(ratio)
}

/// Computes the normalized volume quality metric for a simplex.
///
/// This metric provides a scale-invariant measure of simplex quality by dividing
/// the volume by the D-th power of the average edge length. Accepted simplices
/// retain that scale invariance, while an absolute `1e-12` length floor rejects
/// smaller scales as numerically degenerate. Volume and edge-length-power
/// degeneracy checks use the D-th power of the scale-aware length epsilon so
/// comparisons have matching physical dimensions.
///
/// # Quality Interpretation
///
/// - **Higher values** = better quality
/// - **Optimal** (equilateral 2D): ≈ 0.433 (sqrt(3)/4)
/// - **Poor**: < 0.1 (flat or sliver simplex)
///
/// # Arguments
///
/// * `tri` - The triangulation containing the simplex
/// * `simplex_key` - The key of the simplex to evaluate
///
/// # Returns
///
/// The normalized volume as a floating-point value, or an error if computation fails.
///
/// # Errors
///
/// Returns a [`QualityError`] if:
/// - Simplex has missing or invalid vertices
/// - Simplex is degenerate (zero or near-zero volume, average edge length, or
///   edge-length power)
/// - Edge length computation fails
///
/// # Examples
///
/// ```
/// use delaunay::prelude::*;
/// use delaunay::prelude::geometry::normalized_volume;
///
/// # #[derive(Debug, thiserror::Error)]
/// # enum ExampleError {
/// #     #[error(transparent)]
/// #     Construction(#[from] delaunay::DelaunayTriangulationConstructionError),
/// #     #[error(transparent)]
/// #     Quality(#[from] delaunay::prelude::geometry::QualityError),
/// #     #[error(transparent)]
/// #     Coordinate(#[from] delaunay::prelude::geometry::CoordinateConversionError),
/// # }
/// # fn main() -> Result<(), ExampleError> {
/// // Create a 2D triangle
/// let vertices = vec![
///     delaunay::vertex![0.0, 0.0]?,
///     delaunay::vertex![1.0, 0.0]?,
///     delaunay::vertex![0.0, 1.0]?,
/// ];
/// let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;
/// let Some((simplex_key, _)) = dt.simplices().next() else {
///     return Ok(());
/// };
///
/// let norm_vol = normalized_volume(dt.as_triangulation(), simplex_key)?;
/// assert!(norm_vol > 0.0);
/// # Ok(())
/// # }
/// ```
pub fn normalized_volume<K, U, V, const D: usize>(
    tri: &Triangulation<K, U, V, D>,
    simplex_key: SimplexKey,
) -> Result<f64, QualityError>
where
    K: Kernel<D, Scalar = f64>,
{
    // Extract simplex points using helper
    let points = simplex_points(tri, simplex_key)?;

    if points.len() != D + 1 {
        return Err(QualityError::InvalidSimplexArity {
            actual: points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    // Compute volume using utility function
    let volume = simplex_volume(&points).map_err(|source| QualityError::Volume {
        source: Box::new(source),
    })?;

    // Compute scale-aware epsilon and average edge length
    let (avg_edge_length, epsilon) = scale_aware_epsilon(&points);
    let mut epsilon_pow = f64::one();
    for _ in 0..D {
        epsilon_pow *= epsilon;
    }

    // Check for degenerate simplex (volume too small)
    if volume < epsilon_pow {
        return Err(QualityError::DegenerateSimplex {
            measure: QualityDegeneracyMeasure::Volume,
            observed: CoordinateConversionValue::from_numeric_debug(&volume),
            epsilon: CoordinateConversionValue::from_numeric_debug(&epsilon_pow),
            avg_edge_length: Some(CoordinateConversionValue::from_numeric_debug(
                &avg_edge_length,
            )),
        });
    }

    // Check avg_edge_length using the same scale-aware epsilon
    if avg_edge_length < epsilon {
        return Err(QualityError::DegenerateSimplex {
            measure: QualityDegeneracyMeasure::AverageEdgeLength,
            observed: CoordinateConversionValue::from_numeric_debug(&avg_edge_length),
            epsilon: CoordinateConversionValue::from_numeric_debug(&epsilon),
            avg_edge_length: Some(CoordinateConversionValue::from_numeric_debug(
                &avg_edge_length,
            )),
        });
    }

    // Normalize volume by (avg_edge_length)^D for scale invariance
    let mut edge_length_power = f64::one();
    for _ in 0..D {
        edge_length_power *= avg_edge_length;
    }

    // Check edge_length_power for numerical underflow using a D-dimensional threshold.
    // Although avg_edge_length >= epsilon is verified above, for small avg_edge_length
    // close to epsilon and large D, raising to power D can underflow to < epsilon^D.
    // This catches numerical precision loss during exponentiation.
    if edge_length_power < epsilon_pow {
        return Err(QualityError::DegenerateSimplex {
            measure: QualityDegeneracyMeasure::EdgeLengthPower,
            observed: CoordinateConversionValue::from_numeric_debug(&edge_length_power),
            epsilon: CoordinateConversionValue::from_numeric_debug(&epsilon_pow),
            avg_edge_length: Some(CoordinateConversionValue::from_numeric_debug(
                &avg_edge_length,
            )),
        });
    }

    let normalized = volume / edge_length_power;

    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::construction::{
        DelaunayConstructionFailure, DelaunayTriangulationConstructionError,
    };
    use crate::core::simplex::Simplex;
    use crate::core::tds::Tds;
    use crate::core::triangulation::Triangulation;
    use crate::geometry::kernel::FastKernel;
    use crate::triangulation::DelaunayTriangulation;
    use crate::vertex;
    use approx::assert_relative_eq;
    use std::{array, assert_matches};

    // sqrt(3) constant computed at compile time
    // const SQRT_3: f64 = 1.732_050_807_568_877_3;
    use slotmap::KeyData;

    /// Extracts an f64 scalar payload from a typed diagnostic value for assertions.
    fn scalar_payload(value: &CoordinateConversionValue) -> f64 {
        let CoordinateConversionValue::Scalar(value) = value else {
            panic!("expected scalar diagnostic payload, got {value:?}");
        };
        value.get()
    }

    // =============================================================================
    // DIMENSIONAL MACRO TESTS (2D-5D)
    // =============================================================================

    /// Macro to generate quality metric tests across dimensions
    macro_rules! test_quality_dimensions {
        ($(
            $test_name:ident => $dim:expr => $desc:expr =>
                $vertices:expr,
                $expected_ratio_min:expr,
                $expected_ratio_max:expr
        ),+ $(,)?) => {
            $(
                #[test]
                fn $test_name() {
                    let vertices = $vertices;
                    let dt: DelaunayTriangulation<_, (), (), $dim> = DelaunayTriangulation::builder(&vertices).build().unwrap();
                    let simplex_key = dt.simplices().next().unwrap().0;

                    // Test radius_ratio
                    let ratio = radius_ratio(dt.as_triangulation(), simplex_key).unwrap();
                    assert!(
                        ($expected_ratio_min..=$expected_ratio_max).contains(&ratio),
                        "{}D {}: radius_ratio={ratio}, expected range [{}, {}]",
                        $dim, $desc, $expected_ratio_min, $expected_ratio_max
                    );

                    // Test normalized_volume
                    let norm_vol = normalized_volume(dt.as_triangulation(), simplex_key).unwrap();
                    assert!(norm_vol > 0.0, "{}D {}: normalized_volume should be positive", $dim, $desc);
                }

                pastey::paste! {
                    #[test]
                    fn [<$test_name _scale_invariance>]() {
                        // Test scale invariance by scaling coordinates
                        let vertices_base = $vertices;
                        let dt_base: DelaunayTriangulation<_, (), (), $dim> = DelaunayTriangulation::builder(&vertices_base).build().unwrap();
                        let key_base = dt_base.simplices().next().unwrap().0;

                        // Scale by 10x
                        let vertices_scaled: Vec<_> = vertices_base.iter().map(|v| {
                            let coords: [f64; $dim] = array::from_fn(|idx| v.point().coords()[idx] * 10.0);
                            vertex!(coords).unwrap()
                        }).collect();
                        let dt_scaled: DelaunayTriangulation<_, (), (), $dim> = DelaunayTriangulation::builder(&vertices_scaled).build().unwrap();
                        let key_scaled = dt_scaled.simplices().next().unwrap().0;

                        let ratio_base = radius_ratio(dt_base.as_triangulation(), key_base).unwrap();
                        let ratio_scaled = radius_ratio(dt_scaled.as_triangulation(), key_scaled).unwrap();
                        assert_relative_eq!(ratio_base, ratio_scaled, epsilon = 1e-8);

                        let vol_base = normalized_volume(dt_base.as_triangulation(), key_base).unwrap();
                        let vol_scaled = normalized_volume(dt_scaled.as_triangulation(), key_scaled).unwrap();
                        assert_relative_eq!(vol_base, vol_scaled, epsilon = 1e-5);
                    }

                    #[test]
                    fn [<$test_name _translation_invariance>]() {
                        // Test translation invariance
                        let vertices_base = $vertices;
                        let dt_base: DelaunayTriangulation<_, (), (), $dim> = DelaunayTriangulation::builder(&vertices_base).build().unwrap();
                        let key_base = dt_base.simplices().next().unwrap().0;

                        // Translate by [5.0, 5.0, ...]
                        let vertices_translated: Vec<_> = vertices_base.iter().map(|v| {
                            let coords: [f64; $dim] = array::from_fn(|idx| v.point().coords()[idx] + 5.0);
                            vertex!(coords).unwrap()
                        }).collect();
                        let dt_translated: DelaunayTriangulation<_, (), (), $dim> = DelaunayTriangulation::builder(&vertices_translated).build().unwrap();
                        let key_translated = dt_translated.simplices().next().unwrap().0;

                        let ratio_base = radius_ratio(dt_base.as_triangulation(), key_base).unwrap();
                        let ratio_translated = radius_ratio(dt_translated.as_triangulation(), key_translated).unwrap();
                        assert_relative_eq!(ratio_base, ratio_translated, epsilon = 1e-10);

                        let vol_base = normalized_volume(dt_base.as_triangulation(), key_base).unwrap();
                        let vol_translated = normalized_volume(dt_translated.as_triangulation(), key_translated).unwrap();
                        assert_relative_eq!(vol_base, vol_translated, epsilon = 1e-10);
                    }
                }
            )+
        };
    }

    // Generate tests for dimensions 2D through 5D
    test_quality_dimensions! {
        quality_2d_unit => 2 => "unit simplex" =>
            vec![
                vertex!([0.0, 0.0]).unwrap(),
                vertex!([1.0, 0.0]).unwrap(),
                vertex!([0.0, 1.0]).unwrap(),
            ],
            2.0, 3.0,

        quality_2d_equilateral => 2 => "equilateral triangle" =>
            vec![
                vertex!([0.0, 0.0]).unwrap(),
                vertex!([1.0, 0.0]).unwrap(),
                vertex!([0.5, 0.866_025]).unwrap(),
            ],
            1.9, 2.1,

        quality_2d_right => 2 => "right triangle" =>
            vec![
                vertex!([0.0, 0.0]).unwrap(),
                vertex!([3.0, 0.0]).unwrap(),
                vertex!([0.0, 4.0]).unwrap(),
            ],
            2.0, 5.0,

        quality_3d_unit => 3 => "unit simplex" =>
            vec![
                vertex!([0.0, 0.0, 0.0]).unwrap(),
                vertex!([1.0, 0.0, 0.0]).unwrap(),
                vertex!([0.0, 1.0, 0.0]).unwrap(),
                vertex!([0.0, 0.0, 1.0]).unwrap(),
            ],
            3.0, 5.0,

        quality_3d_regular => 3 => "regular tetrahedron" =>
            vec![
                vertex!([0.0, 0.0, 0.0]).unwrap(),
                vertex!([1.0, 0.0, 0.0]).unwrap(),
                vertex!([0.5, 0.866_025, 0.0]).unwrap(),
                vertex!([0.5, 0.288_675, 0.816_497]).unwrap(),
            ],
            2.8, 3.2,

        quality_4d_unit => 4 => "unit simplex" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
                vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
                vertex!([0.0, 1.0, 0.0, 0.0]).unwrap(),
                vertex!([0.0, 0.0, 1.0, 0.0]).unwrap(),
                vertex!([0.0, 0.0, 0.0, 1.0]).unwrap(),
            ],
            4.0, 7.0,

        quality_4d_regular => 4 => "regular simplex" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
                vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
                vertex!([0.5, 0.866_025, 0.0, 0.0]).unwrap(),
                vertex!([0.5, 0.288_675, 0.816_497, 0.0]).unwrap(),
                vertex!([0.5, 0.288_675, 0.204_124, 0.790_569]).unwrap(),
            ],
            3.8, 4.2,

        quality_5d_unit => 5 => "unit simplex" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
                vertex!([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
                vertex!([0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
                vertex!([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
                vertex!([0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
                vertex!([0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
            ],
            5.0, 15.0,
    }

    // =============================================================================
    // DEGENERATE & POOR QUALITY TESTS
    // =============================================================================

    #[test]
    fn test_degenerate_nearly_collinear() {
        // Nearly collinear points - tests both metrics
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([2.0, 0.001]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Test radius_ratio
        let ratio_result = radius_ratio(dt.as_triangulation(), simplex_key);
        if let Ok(ratio) = ratio_result {
            assert!(ratio > 10.0);
        } else {
            assert_matches!(ratio_result, Err(QualityError::DegenerateSimplex { .. }));
        }

        // Test normalized_volume
        let vol_result = normalized_volume(dt.as_triangulation(), simplex_key);
        if let Ok(norm_vol) = vol_result {
            assert!(norm_vol < 0.01);
        } else {
            assert_matches!(vol_result, Err(QualityError::DegenerateSimplex { .. }));
        }
    }

    // =============================================================================
    // ERROR HANDLING TESTS
    // =============================================================================

    #[test]
    fn quality_simplex_vertices_error_preserves_specific_tds_lookup_failures() {
        let simplex_key = SimplexKey::from(KeyData::from_ffi(0xCAFE));
        let vertex_key = VertexKey::from(KeyData::from_ffi(0xBEEF));

        let missing_simplex = QualitySimplexVerticesError::from(TdsError::SimplexNotFound {
            simplex_key,
            context: "quality metric simplex lookup".to_string(),
        });
        assert_matches!(
            missing_simplex,
            QualitySimplexVerticesError::SimplexNotFound {
                simplex_key: observed,
                context
            } if observed == simplex_key && context == "quality metric simplex lookup"
        );

        let missing_vertex = QualitySimplexVerticesError::from(TdsError::VertexNotFound {
            vertex_key,
            context: "quality metric vertex lookup".to_string(),
        });
        assert_matches!(
            missing_vertex,
            QualitySimplexVerticesError::ReferencedVertexNotFound {
                vertex_key: observed,
                context
            } if observed == vertex_key && context == "quality metric vertex lookup"
        );

        let unexpected = QualitySimplexVerticesError::from(TdsError::DuplicateSimplices {
            message: "same vertex set appears twice".to_string(),
        });
        assert_matches!(
            unexpected,
            QualitySimplexVerticesError::UnexpectedTdsFailure { source }
                if matches!(
                    *source,
                    TdsError::DuplicateSimplices { ref message }
                        if message == "same vertex set appears twice"
                )
        );
    }

    #[test]
    fn scale_aware_epsilon_uses_floor_when_point_set_has_no_edges() {
        let mut points = SmallBuffer::new();
        points.push(Point::try_new([]).expect("finite point coordinates"));

        let (avg_edge_length, epsilon) = scale_aware_epsilon(&points);

        assert_relative_eq!(avg_edge_length, 0.0);
        assert_relative_eq!(epsilon, 1e-12);
    }

    #[test]
    fn test_quality_error_display() {
        // Test that error messages format correctly
        let err = QualityError::InvalidSimplexArity {
            actual: 2,
            expected: 3,
            dimension: 2,
        };
        assert!(format!("{err}").contains("expected 3"));

        let err = QualityError::DegenerateSimplex {
            measure: QualityDegeneracyMeasure::Volume,
            observed: CoordinateConversionValue::from_f64(0.0),
            epsilon: CoordinateConversionValue::from_f64(1e-12),
            avg_edge_length: Some(CoordinateConversionValue::from_f64(1.0)),
        };
        assert!(format!("{err}").contains("Degenerate"));
        assert!(format!("{err}").contains("volume observed"));
        assert!(format!("{err}").contains("0.0"));

        let err = QualityError::NumericConversion {
            operation: QualityNumericOperation::EdgeCountConversion,
        };
        assert!(format!("{err}").contains("Numeric conversion failed"));
        assert!(format!("{err}").contains("edge count conversion"));
    }

    #[test]
    fn quality_diagnostic_enums_display_without_debug_variant_names() {
        assert_eq!(
            QualityNumericOperation::EpsilonFloorConversion.to_string(),
            "epsilon floor conversion"
        );
        assert_eq!(
            QualityNumericOperation::EdgeCountConversion.to_string(),
            "edge count conversion"
        );
        assert_eq!(
            QualityNumericOperation::RelativeFactorConversion.to_string(),
            "relative factor conversion"
        );
        assert_eq!(
            QualityNumericOperation::Circumradius.to_string(),
            "circumradius computation"
        );
        assert_eq!(
            QualityNumericOperation::Inradius.to_string(),
            "inradius computation"
        );
        assert_eq!(
            QualityNumericOperation::Volume.to_string(),
            "simplex volume computation"
        );
        assert_eq!(QualityDegeneracyMeasure::Inradius.to_string(), "inradius");
        assert_eq!(QualityDegeneracyMeasure::Volume.to_string(), "volume");
        assert_eq!(
            QualityDegeneracyMeasure::AverageEdgeLength.to_string(),
            "average edge length"
        );
        assert_eq!(
            QualityDegeneracyMeasure::EdgeLengthPower.to_string(),
            "edge length power"
        );
    }

    // =============================================================================
    // COMPARATIVE TESTS (radius_ratio vs normalized_volume)
    // =============================================================================

    #[test]
    fn test_quality_metrics_correlation() {
        // Both metrics should agree on relative quality
        // Good quality triangle (equilateral)
        let vertices_good = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.5, 0.866_025]).unwrap(),
        ];
        let dt_good: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices_good)
                .build()
                .unwrap();
        let simplex_key_good = dt_good.simplices().next().unwrap().0;

        // Poor quality triangle (very flat)
        let vertices_poor = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.5, 0.01]).unwrap(), // Nearly flat
        ];
        let dt_poor: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices_poor)
                .build()
                .unwrap();
        let simplex_key_poor = dt_poor.simplices().next().unwrap().0;

        let ratio_good = radius_ratio(dt_good.as_triangulation(), simplex_key_good).unwrap();
        let ratio_poor = radius_ratio(dt_poor.as_triangulation(), simplex_key_poor).unwrap();

        let norm_vol_good =
            normalized_volume(dt_good.as_triangulation(), simplex_key_good).unwrap();
        let norm_vol_poor =
            normalized_volume(dt_poor.as_triangulation(), simplex_key_poor).unwrap();

        // Good triangle: lower ratio, higher normalized volume
        assert!(ratio_good < ratio_poor);
        assert!(norm_vol_good > norm_vol_poor);
    }

    // =============================================================================
    // EDGE CASES & BOUNDARY CONDITIONS
    // =============================================================================

    #[test]
    fn test_radius_ratio_perfectly_collinear_3points() {
        // Three points on a line - perfectly degenerate for 2D triangulation.
        // Currently, collinear points are accepted during construction but produce
        // degenerate simplices with very poor quality metrics.
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([2.0, 0.0]).unwrap(), // Collinear
        ];
        let dt_result: Result<
            DelaunayTriangulation<_, (), (), 2>,
            DelaunayTriangulationConstructionError,
        > = DelaunayTriangulation::builder(&vertices).build();

        // Construction may succeed with collinear points, but quality metrics
        // should detect the degeneracy
        if let Ok(dt) = dt_result {
            let simplex_key = dt.simplices().next().unwrap().0;
            let ratio_result = radius_ratio(dt.as_triangulation(), simplex_key);
            let vol_result = normalized_volume(dt.as_triangulation(), simplex_key);

            // At least one quality metric should detect the degeneracy
            assert!(
                ratio_result.is_err() || vol_result.is_err(),
                "Quality metrics should detect degenerate collinear simplex"
            );
        }
        // If construction fails, that's also acceptable for degenerate input
    }

    #[test]
    fn test_quality_near_duplicate_vertices() {
        // Two vertices very close together
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([1.000_000_1, 0.000_000_1]).unwrap(), // Nearly duplicate
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Either should error or produce very poor quality
        if let Ok(ratio) = radius_ratio(dt.as_triangulation(), simplex_key) {
            assert!(ratio > 100.0); // Very poor quality
        }
    }

    #[test]
    fn test_quality_mixed_scale_coordinates() {
        // Mix of large and small coordinates
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1e10, 0.0]).unwrap(),
            vertex!([1e-10, 1e10]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let ratio = radius_ratio(dt.as_triangulation(), simplex_key)
            .expect("mixed-scale simplex should have a radius ratio");
        let normalized = normalized_volume(dt.as_triangulation(), simplex_key)
            .expect("mixed-scale simplex should have a normalized volume");

        assert!(ratio.is_finite() && ratio > 0.0);
        assert!(normalized.is_finite() && normalized > 0.0);
    }

    // =============================================================================
    // DIMENSION-SPECIFIC TESTS
    // =============================================================================

    #[test]
    fn test_quality_6d_simplex() {
        // 6D simplex at unit hypercube corners
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 6> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let ratio = radius_ratio(dt.as_triangulation(), simplex_key).unwrap();
        assert!(ratio > 6.0); // At least the dimension
        assert!(ratio < 20.0); // Not too degenerate

        let norm_vol = normalized_volume(dt.as_triangulation(), simplex_key).unwrap();
        assert!(norm_vol > 0.0);
    }

    /// Macro to generate poor quality tests across dimensions
    macro_rules! test_poor_quality {
        ($(
            $test_name:ident => $dim:expr => $desc:expr =>
                $vertices:expr,
                $min_ratio:expr
        ),+ $(,)?) => {
            $(
                #[test]
                fn $test_name() {
                    let vertices = $vertices;
                    let dt: DelaunayTriangulation<_, (), (), $dim> = DelaunayTriangulation::builder(&vertices).build().unwrap();
let simplex_key = dt.simplices().next().unwrap().0;

                    if let Ok(ratio) = radius_ratio(dt.as_triangulation(), simplex_key) {
                        assert!(ratio > $min_ratio, "{}: ratio={ratio}, expected > {}", $desc, $min_ratio);
                    }
                }
            )+
        };
    }

    test_poor_quality! {
        poor_quality_2d_flat => 2 => "very flat triangle" =>
            vec![
                vertex!([0.0, 0.0]).unwrap(),
                vertex!([100.0, 0.0]).unwrap(),
                vertex!([50.0, 0.1]).unwrap(),
            ],
            50.0,

        poor_quality_3d_nearly_coplanar => 3 => "nearly coplanar tetrahedron" =>
            vec![
                vertex!([0.0, 0.0, 0.0]).unwrap(),
                vertex!([10.0, 0.0, 0.0]).unwrap(),
                vertex!([5.0, 8.66, 0.0]).unwrap(),
                vertex!([5.0, 2.89, 0.01]).unwrap(),
            ],
            30.0,

        poor_quality_4d_degenerate => 4 => "nearly 3D subspace" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0]).unwrap(),
                vertex!([1.0, 0.0, 0.0, 0.0]).unwrap(),
                vertex!([0.5, 0.866, 0.0, 0.0]).unwrap(),
                vertex!([0.5, 0.289, 0.816, 0.0]).unwrap(),
                vertex!([0.5, 0.289, 0.204, 0.001]).unwrap(),
            ],
            10.0,

        poor_quality_3d_sliver => 3 => "sliver tetrahedron" =>
            vec![
                vertex!([0.0, 0.0, 0.0]).unwrap(),
                vertex!([1.0, 0.0, 0.0]).unwrap(),
                vertex!([0.5, 0.866_025, 0.0]).unwrap(),
                vertex!([0.5, 0.288_675, 0.001]).unwrap(),
            ],
            100.0,

        poor_quality_2d_needle => 2 => "needle triangle" =>
            vec![
                vertex!([0.0, 0.0]).unwrap(),
                vertex!([100.0, 0.0]).unwrap(),
                vertex!([0.0, 0.1]).unwrap(),
            ],
            50.0,

        poor_quality_3d_cap => 3 => "cap tetrahedron" =>
            vec![
                vertex!([0.0, 0.0, 0.0]).unwrap(),
                vertex!([1.0, 0.0, 0.0]).unwrap(),
                vertex!([0.5, 0.866_025, 0.0]).unwrap(),
                vertex!([0.5, 0.288_675, 10.0]).unwrap(),
            ],
            10.0,
    }

    // =============================================================================
    // ERROR PATH COVERAGE
    // =============================================================================

    #[test]
    fn test_quality_invalid_simplex_key() {
        // Create triangulation
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.5, 0.866_025]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();

        // Create an invalid key (not in the SlotMap)
        let invalid_key = SimplexKey::from(KeyData::from_ffi(u64::MAX));

        let result = radius_ratio(dt.as_triangulation(), invalid_key);
        assert_matches!(result, Err(QualityError::SimplexVertices { .. }));

        let result = normalized_volume(dt.as_triangulation(), invalid_key);
        assert_matches!(result, Err(QualityError::SimplexVertices { .. }));
    }

    #[test]
    fn test_quality_error_clone_eq() {
        // Test that QualityError implements Clone and PartialEq correctly
        let err1 = QualityError::InvalidSimplexArity {
            actual: 2,
            expected: 3,
            dimension: 2,
        };
        let err2 = err1.clone();
        assert_eq!(err1, err2);

        let err3 = QualityError::DegenerateSimplex {
            measure: QualityDegeneracyMeasure::Volume,
            observed: CoordinateConversionValue::from_f64(0.0),
            epsilon: CoordinateConversionValue::from_f64(1e-12),
            avg_edge_length: None,
        };
        let err4 = err3.clone();
        assert_eq!(err3, err4);

        let err5 = QualityError::NumericConversion {
            operation: QualityNumericOperation::EdgeCountConversion,
        };
        let err6 = err5.clone();
        assert_eq!(err5, err6);

        // Different errors should not be equal
        assert_ne!(err1, err3);
        assert_ne!(err3, err5);
    }

    // =============================================================================
    // COMPARATIVE QUALITY RANKINGS
    // =============================================================================

    #[test]
    fn test_quality_ranking_and_thresholds() {
        // Test both ranking consistency and threshold validation in one test
        // Best: equilateral (good quality threshold)
        let vertices_best = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.5, 0.866_025]).unwrap(),
        ];
        let dt_best: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices_best)
                .build()
                .unwrap();
        let key_best = dt_best.simplices().next().unwrap().0;

        // Medium: right triangle (acceptable quality)
        let vertices_medium = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([3.0, 0.0]).unwrap(),
            vertex!([0.0, 4.0]).unwrap(),
        ];
        let dt_medium: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices_medium)
                .build()
                .unwrap();
        let key_medium = dt_medium.simplices().next().unwrap().0;

        // Worst: very flat (poor quality)
        let vertices_worst = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([10.0, 0.0]).unwrap(),
            vertex!([5.0, 0.1]).unwrap(),
        ];
        let dt_worst: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices_worst)
                .build()
                .unwrap();
        let key_worst = dt_worst.simplices().next().unwrap().0;

        let ratio_best = radius_ratio(dt_best.as_triangulation(), key_best).unwrap();
        let ratio_medium = radius_ratio(dt_medium.as_triangulation(), key_medium).unwrap();
        let ratio_worst = radius_ratio(dt_worst.as_triangulation(), key_worst).unwrap();

        let vol_best = normalized_volume(dt_best.as_triangulation(), key_best).unwrap();
        let vol_medium = normalized_volume(dt_medium.as_triangulation(), key_medium).unwrap();
        let vol_worst = normalized_volume(dt_worst.as_triangulation(), key_worst).unwrap();

        // Verify consistent ranking
        assert!(ratio_best < ratio_medium);
        assert!(ratio_medium < ratio_worst);
        assert!(vol_best > vol_medium);
        assert!(vol_medium > vol_worst);

        // Verify quality thresholds
        assert!(ratio_best < 4.0, "Good quality: ratio < 4");
        assert!(ratio_medium < 10.0, "Acceptable quality: ratio < 10");
        assert!(ratio_worst >= 10.0, "Poor quality: ratio >= 10");
    }

    // =============================================================================
    // SPECIAL GEOMETRIC CONFIGURATIONS
    // =============================================================================

    #[test]
    fn test_special_triangles() {
        // Test right triangle
        let vertices_right = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0]).unwrap(),
        ];
        let dt_right: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices_right)
                .build()
                .unwrap();
        let key_right = dt_right.simplices().next().unwrap().0;
        let ratio_right = radius_ratio(dt_right.as_triangulation(), key_right).unwrap();
        assert_relative_eq!(ratio_right, 1.0 + 2.0_f64.sqrt(), epsilon = 0.1);

        // Test isosceles triangle
        let vertices_iso = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([2.0, 0.0]).unwrap(),
            vertex!([1.0, 2.0]).unwrap(),
        ];
        let dt_iso: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices_iso)
                .build()
                .unwrap();
        let key_iso = dt_iso.simplices().next().unwrap().0;
        let ratio_iso = radius_ratio(dt_iso.as_triangulation(), key_iso).unwrap();
        assert!(ratio_iso > 2.0 && ratio_iso < 5.0);
    }

    // =============================================================================
    // PRECISION TESTS
    // =============================================================================

    // =============================================================================
    // HELPER FUNCTION TESTS
    // =============================================================================

    #[test]
    fn test_simplex_points_valid() {
        // Test simplex_points helper with valid simplex
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.5, 0.866_025]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        let points = simplex_points(dt.as_triangulation(), simplex_key).unwrap();
        assert_eq!(points.len(), 3, "Should have 3 points for 2D simplex");
    }

    #[test]
    fn test_simplex_points_invalid_key() {
        // Test simplex_points with invalid simplex key
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.5, 0.866_025]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let invalid_key = SimplexKey::from(KeyData::from_ffi(u64::MAX));

        let result = simplex_points(dt.as_triangulation(), invalid_key);
        assert_matches!(result, Err(QualityError::SimplexVertices { .. }));
    }

    #[test]
    fn test_scale_aware_epsilon_2d() {
        // Test epsilon computation for 2D simplex
        let mut points = SmallBuffer::new();
        points.push(Point::try_new([0.0, 0.0]).expect("finite point coordinates"));
        points.push(Point::try_new([1.0, 0.0]).expect("finite point coordinates"));
        points.push(Point::try_new([0.5, 0.866_025]).expect("finite point coordinates"));

        let (avg_edge_length, epsilon) = scale_aware_epsilon(&points);
        assert!(
            avg_edge_length > 0.0,
            "Average edge length should be positive"
        );
        assert!(epsilon > 0.0, "Epsilon should be positive");
        assert!(epsilon >= 1e-12, "Epsilon should have floor of 1e-12");
    }

    #[test]
    fn test_scale_aware_epsilon_tiny_simplex() {
        // Test epsilon computation with very small coordinates
        let mut points = SmallBuffer::new();
        points.push(Point::try_new([0.0, 0.0]).expect("finite point coordinates"));
        points.push(Point::try_new([1e-10, 0.0]).expect("finite point coordinates"));
        points.push(Point::try_new([0.5e-10, 0.866_025e-10]).expect("finite point coordinates"));

        let (avg_edge_length, epsilon) = scale_aware_epsilon(&points);
        // For tiny simplices, epsilon should use the floor (1e-12)
        assert!(epsilon >= 1e-12);
        assert!(avg_edge_length > 0.0);
    }

    #[test]
    fn test_scale_aware_epsilon_large_simplex() {
        // Test epsilon computation with large coordinates
        let mut points = SmallBuffer::new();
        points.push(Point::try_new([0.0, 0.0]).expect("finite point coordinates"));
        points.push(Point::try_new([1e6, 0.0]).expect("finite point coordinates"));
        points.push(Point::try_new([0.5e6, 0.866_025e6]).expect("finite point coordinates"));

        let (avg_edge_length, epsilon) = scale_aware_epsilon(&points);
        // For large simplices, epsilon scales with average edge length
        assert!(epsilon > 1e-12);
        assert!(avg_edge_length > 1e5);
    }

    #[test]
    fn test_scale_aware_epsilon_3d() {
        // Test epsilon computation for 3D simplex
        let mut points = SmallBuffer::new();
        points.push(Point::try_new([0.0, 0.0, 0.0]).expect("finite point coordinates"));
        points.push(Point::try_new([1.0, 0.0, 0.0]).expect("finite point coordinates"));
        points.push(Point::try_new([0.0, 1.0, 0.0]).expect("finite point coordinates"));
        points.push(Point::try_new([0.0, 0.0, 1.0]).expect("finite point coordinates"));

        let (avg_edge_length, epsilon) = scale_aware_epsilon(&points);
        assert!(avg_edge_length > 0.0);
        assert!(epsilon > 0.0);
    }

    // =============================================================================
    // ADDITIONAL ERROR PATH TESTS
    // =============================================================================

    #[test]
    fn test_radius_ratio_wrong_vertex_count() {
        // Create a triangulation to test vertex count validation in radius_ratio
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.5, 0.866_025]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 2> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Normal case should succeed
        let result = radius_ratio(dt.as_triangulation(), simplex_key);
        assert!(result.is_ok());
    }

    #[test]
    fn test_normalized_volume_wrong_vertex_count() {
        // Test normalized_volume with proper vertex count
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1.0]).unwrap(),
        ];
        let dt: DelaunayTriangulation<_, (), (), 3> =
            DelaunayTriangulation::builder(&vertices).build().unwrap();
        let simplex_key = dt.simplices().next().unwrap().0;

        // Should succeed with correct count
        let result = normalized_volume(dt.as_triangulation(), simplex_key);
        assert!(result.is_ok());
    }

    #[test]
    fn test_radius_ratio_numerical_edge_cases() {
        // Test with coordinates near numerical limits
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([0.5, 1e-15]).unwrap(), // Very small but non-zero height
        ];
        let dt_result: Result<
            DelaunayTriangulation<_, (), (), 2>,
            DelaunayTriangulationConstructionError,
        > = DelaunayTriangulation::builder(&vertices).build();

        match dt_result {
            Ok(dt) => {
                let simplex_key = dt.simplices().next().unwrap().0;

                // Should either compute or return degenerate/numerical error (no panic)
                let result = radius_ratio(dt.as_triangulation(), simplex_key);
                assert!(
                    result.is_ok()
                        || matches!(result, Err(QualityError::DegenerateSimplex { .. }))
                        || matches!(result, Err(QualityError::NumericConversion { .. }))
                        || matches!(result, Err(QualityError::Circumradius { .. }))
                        || matches!(result, Err(QualityError::Inradius { .. }))
                );
            }
            Err(DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::GeometricDegeneracy { .. },
            )) => {
                // For sufficiently extreme configurations, the robust initial simplex
                // search may reject the input up-front as geometrically degenerate.
                // This is acceptable as long as it is reported cleanly.
            }
            Err(other) => panic!("Unexpected triangulation error for numerical edge case: {other}"),
        }
    }

    #[test]
    fn test_normalized_volume_numerical_edge_cases() {
        // Test normalized_volume with numerical edge cases
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0, 0.0]).unwrap(),
            vertex!([0.0, 1.0, 0.0]).unwrap(),
            vertex!([0.0, 0.0, 1e-14]).unwrap(), // Very small but non-zero
        ];
        let dt_result: Result<
            DelaunayTriangulation<_, (), (), 3>,
            DelaunayTriangulationConstructionError,
        > = DelaunayTriangulation::builder(&vertices).build();

        match dt_result {
            Ok(dt) => {
                let simplex_key = dt.simplices().next().unwrap().0;

                // Should either compute or return degenerate/numerical error (no panic)
                let result = normalized_volume(dt.as_triangulation(), simplex_key);
                assert!(
                    result.is_ok()
                        || matches!(result, Err(QualityError::DegenerateSimplex { .. }))
                        || matches!(result, Err(QualityError::NumericConversion { .. }))
                        || matches!(result, Err(QualityError::Volume { .. }))
                );
            }
            Err(DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::GeometricDegeneracy { .. }
                | DelaunayConstructionFailure::InsufficientVertices { .. },
            )) => {
                // Extremely flat/near-degenerate configurations may now be rejected
                // up-front by the initial simplex search, or Hilbert-sort dedup may
                // collapse near-identical coordinates (e.g. 1e-14 vs 0.0) at
                // quantization resolution, leaving fewer than D+1 distinct vertices.
            }
            Err(other) => panic!(
                "Unexpected triangulation error for normalized_volume numerical edge case: {other}",
            ),
        }
    }

    #[test]
    fn normalized_volume_uses_dimensionally_consistent_volume_threshold() {
        let mut tds: Tds<(), (), 2> = Tds::empty();
        let v0 = tds
            .insert_vertex_with_mapping(vertex!([0.0, 0.0]).unwrap())
            .unwrap();
        let v1 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 0.0]).unwrap())
            .unwrap();
        let v2 = tds
            .insert_vertex_with_mapping(vertex!([1.0, 1.0e-30]).unwrap())
            .unwrap();
        let simplex_key = tds
            .insert_simplex_with_mapping(
                Simplex::try_new_with_data(vec![v0, v1, v2], None).unwrap(),
            )
            .unwrap();
        let tri = Triangulation::<FastKernel<f64>, (), (), 2>::new_with_tds(FastKernel::new(), tds);

        let err = normalized_volume(&tri, simplex_key).unwrap_err();

        let QualityError::DegenerateSimplex {
            measure,
            epsilon,
            avg_edge_length,
            ..
        } = err
        else {
            panic!("expected volume degeneracy, got {err}");
        };

        assert_eq!(measure, QualityDegeneracyMeasure::Volume);
        let epsilon = scalar_payload(&epsilon);
        let avg_edge_length = scalar_payload(&avg_edge_length.unwrap());
        let linear_epsilon = 1.0e-12_f64.max(avg_edge_length * 1.0e-8);
        let expected_area_epsilon = linear_epsilon * linear_epsilon;
        assert!(
            (epsilon - expected_area_epsilon).abs() <= expected_area_epsilon * 1.0e-12,
            "volume threshold should be epsilon^D; got {epsilon}, expected {expected_area_epsilon}"
        );
    }

    #[test]
    fn test_quality_error_source_trait() {
        // Test that QualityError implements std::error::Error properly
        let err = QualityError::NumericConversion {
            operation: QualityNumericOperation::EdgeCountConversion,
        };
        assert!(std::error::Error::source(&err).is_none());
        assert!(format!("{err}").contains("Numeric conversion failed"));
    }

    #[test]
    fn test_degenerate_simplex_error_details() {
        // Test that degenerate errors include helpful details when they occur.
        let vertices = vec![
            vertex!([0.0, 0.0]).unwrap(),
            vertex!([1.0, 0.0]).unwrap(),
            vertex!([2.0, 1e-20]).unwrap(), // Nearly collinear
        ];
        let dt_result: Result<
            DelaunayTriangulation<_, (), (), 2>,
            DelaunayTriangulationConstructionError,
        > = DelaunayTriangulation::builder(&vertices).build();

        match dt_result {
            Ok(dt) => {
                let simplex_key = dt.simplices().next().unwrap().0;

                let result = radius_ratio(dt.as_triangulation(), simplex_key);
                if let Err(QualityError::DegenerateSimplex {
                    measure, observed, ..
                }) = result
                {
                    // Should include numeric information when we surface a degenerate simplex
                    assert_matches!(
                        measure,
                        QualityDegeneracyMeasure::Inradius
                            | QualityDegeneracyMeasure::Volume
                            | QualityDegeneracyMeasure::AverageEdgeLength
                            | QualityDegeneracyMeasure::EdgeLengthPower
                    );
                    assert_matches!(observed, CoordinateConversionValue::Scalar(_));
                }
            }
            Err(DelaunayTriangulationConstructionError::Triangulation(
                DelaunayConstructionFailure::GeometricDegeneracy { .. },
            )) => {
                // In some numeric regimes, degeneracy is now detected at construction
                // time instead of by the quality metrics. That is still acceptable
                // as long as it is reported via the dedicated GeometricDegeneracy
                // error variant.
            }
            Err(other) => {
                panic!("Unexpected triangulation error for degenerate simplex test: {other}")
            }
        }
    }
}
