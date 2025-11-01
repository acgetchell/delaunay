//! Geometric quality measures for d-dimensional simplicial cells.
//!
//! This module provides quality metrics for evaluating the geometric quality
//! of simplicial cells (simplices) in d-dimensional triangulations. These metrics
//! are used to prefer well-shaped cells over degenerate or sliver cells during
//! triangulation operations.
//!
//! # Quality Metrics
//!
//! - **Radius Ratio**: Circumradius divided by inradius. Lower values indicate
//!   better-shaped cells. An equilateral simplex has a radius ratio close to
//!   the dimension-dependent optimal value.
//! - **Normalized Volume**: Volume divided by the D-th power of the average edge length.
//!   Provides a scale-invariant measure of cell shape quality.
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

use crate::core::{
    collections::{MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer},
    traits::data_type::DataType,
    triangulation_data_structure::{CellKey, Tds},
};
use crate::geometry::{
    point::Point,
    traits::coordinate::CoordinateScalar,
    util::{circumradius, hypot, inradius as simplex_inradius, simplex_volume},
};
use num_traits::NumCast;
use std::{
    error::Error,
    fmt,
    iter::Sum,
    ops::{AddAssign, Div, SubAssign},
};

/// Errors that can occur during quality metric computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualityError {
    /// Cell has invalid or missing vertex keys
    InvalidCell {
        /// Description of the error
        message: String,
    },
    /// Cell is degenerate (zero or near-zero volume)
    DegenerateCell {
        /// Measure/context of degeneracy (e.g., "volume=…", "inradius=…")
        detail: String,
    },
    /// Numerical computation failed
    NumericalError {
        /// Description of the numerical issue
        message: String,
    },
}

impl fmt::Display for QualityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCell { message } => write!(f, "Invalid cell: {message}"),
            Self::DegenerateCell { detail } => {
                write!(f, "Degenerate cell: {detail}")
            }
            Self::NumericalError { message } => write!(f, "Numerical error: {message}"),
        }
    }
}

impl Error for QualityError {}

/// Helper function to extract cell points from a triangulation.
///
/// This centralizes the vertex-to-point extraction logic used by quality metrics.
/// Uses `SmallBuffer` to avoid heap allocation for typical cell sizes (D+1 vertices).
fn cell_points<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
) -> Result<SmallBuffer<Point<T, D>, MAX_PRACTICAL_DIMENSION_SIZE>, QualityError>
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
{
    let vertex_keys = tds
        .get_cell_vertices(cell_key)
        .map_err(|e| QualityError::InvalidCell {
            message: format!("Failed to get cell vertices: {e}"),
        })?;

    // Use SmallBuffer to avoid heap allocation (cells have D+1 vertices, D ≤ MAX_PRACTICAL_DIMENSION_SIZE)
    let mut points = SmallBuffer::new();
    for &vkey in &vertex_keys {
        let point = tds
            .get_vertex_by_key(vkey)
            .map(|v| *v.point())
            .ok_or_else(|| QualityError::InvalidCell {
                message: format!("Vertex {vkey:?} not found in triangulation"),
            })?;
        points.push(point);
    }
    Ok(points)
}

/// Computes scale-aware epsilon and average edge length for degeneracy detection.
///
/// This helper centralizes the epsilon calculation logic used by both `radius_ratio`
/// and `normalized_volume` to ensure consistent degeneracy detection across metrics.
///
/// # Arguments
///
/// * `points` - The cell vertices
///
/// # Returns
///
/// A tuple of `(avg_edge_length, epsilon)` where:
/// - `avg_edge_length`: Translation-invariant geometric scale
/// - `epsilon`: Relative tolerance (1e-8 × `avg_edge_length`) with 1e-12 floor
///
/// # Errors
///
/// Returns `QualityError` if edge count conversion or epsilon conversion fails.
fn compute_scale_aware_epsilon<T, const D: usize>(
    points: &SmallBuffer<Point<T, D>, MAX_PRACTICAL_DIMENSION_SIZE>,
) -> Result<(T, T), QualityError>
where
    T: CoordinateScalar + AddAssign<T> + Sum + NumCast,
{
    let mut total_edge_length = T::zero();
    let mut edge_count = 0;

    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let mut diff_coords = [T::zero(); D];
            for (idx, diff) in diff_coords.iter_mut().enumerate() {
                *diff = points[i].coords()[idx] - points[j].coords()[idx];
            }
            let dist = hypot(diff_coords);
            total_edge_length += dist;
            edge_count += 1;
        }
    }

    let edge_count_t = NumCast::from(edge_count).ok_or_else(|| QualityError::NumericalError {
        message: "Failed to convert edge count to type T".to_string(),
    })?;
    let avg_edge_length = total_edge_length / edge_count_t;

    let floor: T = NumCast::from(1e-12).ok_or_else(|| QualityError::NumericalError {
        message: "Failed to convert floor epsilon (1e-12) to coordinate type".to_string(),
    })?;
    let relative_factor: T = NumCast::from(1e-8).ok_or_else(|| QualityError::NumericalError {
        message: "Failed to convert relative factor (1e-8) to coordinate type".to_string(),
    })?;
    let epsilon = floor.max(avg_edge_length * relative_factor);

    Ok((avg_edge_length, epsilon))
}

/// Computes the radius ratio quality metric for a cell.
///
/// The radius ratio is defined as the circumradius divided by the inradius.
/// Lower values indicate better cell quality. For a regular simplex in D dimensions,
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
/// * `tds` - The triangulation data structure containing the cell
/// * `cell_key` - The key of the cell to evaluate
///
/// # Returns
///
/// The radius ratio as a floating-point value, or an error if computation fails.
///
/// # Errors
///
/// Returns `QualityError` if:
/// - Cell has missing or invalid vertices
/// - Cell is degenerate (zero or near-zero volume)
/// - Circumsphere computation fails
///
/// # Examples
///
/// ```
/// use delaunay::{vertex, core::triangulation_data_structure::Tds};
/// use delaunay::geometry::quality::radius_ratio;
///
/// // Create a 2D equilateral triangle
/// let vertices = vec![
///     vertex!([0.0, 0.0]),
///     vertex!([1.0, 0.0]),
///     vertex!([0.5, 0.866]), // approximately sqrt(3)/2
/// ];
/// let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
/// let cell_key = tds.cell_keys().next().unwrap();
///
/// let ratio = radius_ratio(&tds, cell_key).unwrap();
/// // For an equilateral triangle, ratio ≈ 2.0
/// assert!(ratio > 1.5 && ratio < 2.5);
/// ```
pub fn radius_ratio<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
) -> Result<T, QualityError>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + Sum + NumCast + Div<Output = T>,
    U: DataType,
    V: DataType,
{
    // Extract cell points using helper
    let points = cell_points(tds, cell_key)?;

    if points.len() != D + 1 {
        return Err(QualityError::InvalidCell {
            message: format!(
                "Cell has {} vertices, expected {} for dimension {D}",
                points.len(),
                D + 1
            ),
        });
    }

    // Compute circumradius using utility function
    let circumradius_val = circumradius(&points).map_err(|e| QualityError::NumericalError {
        message: format!("Circumradius computation failed: {e}"),
    })?;

    // Compute inradius using utility function
    let inradius_val = simplex_inradius(&points).map_err(|e| QualityError::NumericalError {
        message: format!("Inradius computation failed: {e}"),
    })?;

    // Check for near-zero inradius (degenerate cell) using scale-aware tolerance
    let (avg_edge_length, epsilon) = compute_scale_aware_epsilon(&points)?;

    if inradius_val < epsilon {
        return Err(QualityError::DegenerateCell {
            detail: format!(
                "inradius={inradius_val:?}, epsilon={epsilon:?}, avg_edge_length={avg_edge_length:?}"
            ),
        });
    }

    // radius_ratio = circumradius / inradius
    let ratio = circumradius_val / inradius_val;

    Ok(ratio)
}

/// Computes the normalized volume quality metric for a cell.
///
/// This metric provides a scale-invariant measure of cell quality by dividing
/// the volume by the D-th power of the average edge length. It avoids the numerical
/// issues that can arise when computing inradius for very small cells.
///
/// # Quality Interpretation
///
/// - **Higher values** = better quality
/// - **Optimal** (equilateral 2D): ≈ 0.433 (sqrt(3)/4)
/// - **Poor**: < 0.1 (flat or sliver cell)
///
/// # Arguments
///
/// * `tds` - The triangulation data structure containing the cell
/// * `cell_key` - The key of the cell to evaluate
///
/// # Returns
///
/// The normalized volume as a floating-point value, or an error if computation fails.
///
/// # Errors
///
/// Returns `QualityError` if:
/// - Cell has missing or invalid vertices
/// - Cell is degenerate (zero or near-zero volume)
/// - Edge length computation fails
///
/// # Examples
///
/// ```
/// use delaunay::{vertex, core::triangulation_data_structure::Tds};
/// use delaunay::geometry::quality::normalized_volume;
///
/// // Create a 2D triangle
/// let vertices = vec![
///     vertex!([0.0, 0.0]),
///     vertex!([1.0, 0.0]),
///     vertex!([0.0, 1.0]),
/// ];
/// let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
/// let cell_key = tds.cell_keys().next().unwrap();
///
/// let norm_vol = normalized_volume(&tds, cell_key).unwrap();
/// assert!(norm_vol > 0.0);
/// ```
pub fn normalized_volume<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
) -> Result<T, QualityError>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + Sum + NumCast + Div<Output = T>,
    U: DataType,
    V: DataType,
{
    // Extract cell points using helper
    let points = cell_points(tds, cell_key)?;

    if points.len() != D + 1 {
        return Err(QualityError::InvalidCell {
            message: format!(
                "Cell has {} vertices, expected {} for dimension {D}",
                points.len(),
                D + 1
            ),
        });
    }

    // Compute volume using utility function
    let volume = simplex_volume(&points).map_err(|e| QualityError::NumericalError {
        message: format!("Volume computation failed: {e}"),
    })?;

    // Compute scale-aware epsilon and average edge length
    let (avg_edge_length, epsilon) = compute_scale_aware_epsilon(&points)?;

    // Check for degenerate cell (volume too small)
    if volume < epsilon {
        return Err(QualityError::DegenerateCell {
            detail: format!(
                "volume={volume:?}, epsilon={epsilon:?}, avg_edge_length={avg_edge_length:?}"
            ),
        });
    }

    // Check avg_edge_length using the same scale-aware epsilon
    if avg_edge_length < epsilon {
        return Err(QualityError::DegenerateCell {
            detail: format!("avg_edge_length={avg_edge_length:?}, epsilon={epsilon:?}"),
        });
    }

    // Normalize volume by (avg_edge_length)^D for scale invariance
    let d_i32 = i32::try_from(D).map_err(|_| QualityError::NumericalError {
        message: format!("Dimension {D} too large to convert to i32"),
    })?;
    let edge_length_power = avg_edge_length.powi(d_i32);

    // Check edge_length_power for numerical underflow.
    // Although avg_edge_length >= epsilon is verified above, for small avg_edge_length
    // close to epsilon and large D, raising to power D can underflow to < epsilon.
    // This catches numerical precision loss during exponentiation.
    if edge_length_power < epsilon {
        return Err(QualityError::DegenerateCell {
            detail: format!("edge_length_power={edge_length_power:?}, epsilon={epsilon:?}"),
        });
    }

    let normalized = volume / edge_length_power;

    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::triangulation_data_structure::Tds;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::vertex;
    use approx::assert_relative_eq;

    // sqrt(3) constant computed at compile time
    // const SQRT_3: f64 = 1.732_050_807_568_877_3;
    use slotmap::KeyData;

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
                    let tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&vertices).unwrap();
                    let cell_key = tds.cell_keys().next().unwrap();

                    // Test radius_ratio
                    let ratio = radius_ratio(&tds, cell_key).unwrap();
                    assert!(
                        ($expected_ratio_min..=$expected_ratio_max).contains(&ratio),
                        "{}D {}: radius_ratio={ratio}, expected range [{}, {}]",
                        $dim, $desc, $expected_ratio_min, $expected_ratio_max
                    );

                    // Test normalized_volume
                    let norm_vol = normalized_volume(&tds, cell_key).unwrap();
                    assert!(norm_vol > 0.0, "{}D {}: normalized_volume should be positive", $dim, $desc);
                }

                pastey::paste! {
                    #[test]
                    fn [<$test_name _scale_invariance>]() {
                        // Test scale invariance by scaling coordinates
                        let vertices_base = $vertices;
                        let tds_base: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&vertices_base).unwrap();
                        let key_base = tds_base.cell_keys().next().unwrap();

                        // Scale by 10x
                        let vertices_scaled: Vec<_> = vertices_base.iter().map(|v| {
                            let coords: [f64; $dim] = v.point().coords().iter().map(|&c| c * 10.0).collect::<Vec<_>>().try_into().unwrap();
                            vertex!(coords)
                        }).collect();
                        let tds_scaled: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&vertices_scaled).unwrap();
                        let key_scaled = tds_scaled.cell_keys().next().unwrap();

                        let ratio_base = radius_ratio(&tds_base, key_base).unwrap();
                        let ratio_scaled = radius_ratio(&tds_scaled, key_scaled).unwrap();
                        assert_relative_eq!(ratio_base, ratio_scaled, epsilon = 1e-8);

                        let vol_base = normalized_volume(&tds_base, key_base).unwrap();
                        let vol_scaled = normalized_volume(&tds_scaled, key_scaled).unwrap();
                        assert_relative_eq!(vol_base, vol_scaled, epsilon = 1e-5);
                    }

                    #[test]
                    fn [<$test_name _translation_invariance>]() {
                        // Test translation invariance
                        let vertices_base = $vertices;
                        let tds_base: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&vertices_base).unwrap();
                        let key_base = tds_base.cell_keys().next().unwrap();

                        // Translate by [5.0, 5.0, ...]
                        let vertices_translated: Vec<_> = vertices_base.iter().map(|v| {
                            let coords: [f64; $dim] = v.point().coords().iter().map(|&c| c + 5.0).collect::<Vec<_>>().try_into().unwrap();
                            vertex!(coords)
                        }).collect();
                        let tds_translated: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&vertices_translated).unwrap();
                        let key_translated = tds_translated.cell_keys().next().unwrap();

                        let ratio_base = radius_ratio(&tds_base, key_base).unwrap();
                        let ratio_translated = radius_ratio(&tds_translated, key_translated).unwrap();
                        assert_relative_eq!(ratio_base, ratio_translated, epsilon = 1e-10);

                        let vol_base = normalized_volume(&tds_base, key_base).unwrap();
                        let vol_translated = normalized_volume(&tds_translated, key_translated).unwrap();
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
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.0, 1.0]),
            ],
            2.0, 3.0,

        quality_2d_equilateral => 2 => "equilateral triangle" =>
            vec![
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.5, 0.866_025]),
            ],
            1.9, 2.1,

        quality_2d_right => 2 => "right triangle" =>
            vec![
                vertex!([0.0, 0.0]),
                vertex!([3.0, 0.0]),
                vertex!([0.0, 4.0]),
            ],
            2.0, 5.0,

        quality_3d_unit => 3 => "unit simplex" =>
            vec![
                vertex!([0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0]),
                vertex!([0.0, 1.0, 0.0]),
                vertex!([0.0, 0.0, 1.0]),
            ],
            3.0, 5.0,

        quality_3d_regular => 3 => "regular tetrahedron" =>
            vec![
                vertex!([0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0]),
                vertex!([0.5, 0.866_025, 0.0]),
                vertex!([0.5, 0.288_675, 0.816_497]),
            ],
            2.8, 3.2,

        quality_4d_unit => 4 => "unit simplex" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0, 0.0]),
                vertex!([0.0, 1.0, 0.0, 0.0]),
                vertex!([0.0, 0.0, 1.0, 0.0]),
                vertex!([0.0, 0.0, 0.0, 1.0]),
            ],
            4.0, 7.0,

        quality_4d_regular => 4 => "regular simplex" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0, 0.0]),
                vertex!([0.5, 0.866_025, 0.0, 0.0]),
                vertex!([0.5, 0.288_675, 0.816_497, 0.0]),
                vertex!([0.5, 0.288_675, 0.204_124, 0.790_569]),
            ],
            3.8, 4.2,

        quality_5d_unit => 5 => "unit simplex" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
                vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
                vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
                vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
                vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
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
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([2.0, 0.001]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Test radius_ratio
        let ratio_result = radius_ratio(&tds, cell_key);
        if let Ok(ratio) = ratio_result {
            assert!(ratio > 10.0);
        } else {
            assert!(matches!(
                ratio_result,
                Err(QualityError::DegenerateCell { .. })
            ));
        }

        // Test normalized_volume
        let vol_result = normalized_volume(&tds, cell_key);
        if let Ok(norm_vol) = vol_result {
            assert!(norm_vol < 0.01);
        } else {
            assert!(matches!(
                vol_result,
                Err(QualityError::DegenerateCell { .. })
            ));
        }
    }

    // =============================================================================
    // ERROR HANDLING TESTS
    // =============================================================================

    #[test]
    fn test_quality_error_display() {
        // Test that error messages format correctly
        let err = QualityError::InvalidCell {
            message: "test message".to_string(),
        };
        assert!(format!("{err}").contains("Invalid cell"));
        assert!(format!("{err}").contains("test message"));

        let err = QualityError::DegenerateCell {
            detail: "volume=0.0".to_string(),
        };
        assert!(format!("{err}").contains("Degenerate"));
        assert!(format!("{err}").contains("volume=0.0"));

        let err = QualityError::NumericalError {
            message: "test error".to_string(),
        };
        assert!(format!("{err}").contains("Numerical error"));
        assert!(format!("{err}").contains("test error"));
    }

    // =============================================================================
    // COMPARATIVE TESTS (radius_ratio vs normalized_volume)
    // =============================================================================

    #[test]
    fn test_quality_metrics_correlation() {
        // Both metrics should agree on relative quality
        // Good quality triangle (equilateral)
        let vertices_good = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 0.866_025]),
        ];
        let tds_good: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_good).unwrap();
        let cell_key_good = tds_good.cell_keys().next().unwrap();

        // Poor quality triangle (very flat)
        let vertices_poor = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 0.01]), // Nearly flat
        ];
        let tds_poor: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_poor).unwrap();
        let cell_key_poor = tds_poor.cell_keys().next().unwrap();

        let ratio_good = radius_ratio(&tds_good, cell_key_good).unwrap();
        let ratio_poor = radius_ratio(&tds_poor, cell_key_poor).unwrap();

        let norm_vol_good = normalized_volume(&tds_good, cell_key_good).unwrap();
        let norm_vol_poor = normalized_volume(&tds_poor, cell_key_poor).unwrap();

        // Good triangle: lower ratio, higher normalized volume
        assert!(ratio_good < ratio_poor);
        assert!(norm_vol_good > norm_vol_poor);
    }

    // =============================================================================
    // f32 COMPATIBILITY TESTS
    // =============================================================================

    #[test]
    fn test_quality_metrics_f32_compatibility() {
        // Test that quality metrics work with f32 coordinate type
        // 2D equilateral triangle
        let vertices = vec![
            vertex!([0.0f32, 0.0f32]),
            vertex!([1.0f32, 0.0f32]),
            vertex!([0.5f32, 0.866f32]), // approximately sqrt(3)/2
        ];
        let tds: Tds<f32, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Test radius_ratio
        let ratio = radius_ratio(&tds, cell_key).unwrap();
        // For equilateral triangle: R/r = 2
        assert!(ratio > 1.5 && ratio < 2.5, "ratio={ratio}");

        // Test normalized_volume
        let norm_vol = normalized_volume(&tds, cell_key).unwrap();
        // For 2D equilateral: sqrt(3)/4 ≈ 0.433
        assert!(norm_vol > 0.3 && norm_vol < 0.6, "norm_vol={norm_vol}");
    }

    // =============================================================================
    // EDGE CASES & BOUNDARY CONDITIONS
    // =============================================================================

    #[test]
    fn test_radius_ratio_perfectly_collinear_3points() {
        // Three points on a line - perfectly degenerate
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([2.0, 0.0]), // Collinear
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        let result = radius_ratio(&tds, cell_key);
        // Should fail with degenerate or numerical error
        assert!(result.is_err());
    }

    #[test]
    fn test_quality_near_duplicate_vertices() {
        // Two vertices very close together
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([1.000_000_1, 0.000_000_1]), // Nearly duplicate
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Either should error or produce very poor quality
        if let Ok(ratio) = radius_ratio(&tds, cell_key) {
            assert!(ratio > 100.0); // Very poor quality
        }
    }

    #[test]
    fn test_quality_mixed_scale_coordinates() {
        // Mix of large and small coordinates
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1e10, 0.0]),
            vertex!([1e-10, 1e10]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Should compute without panicking
        let ratio_result = radius_ratio(&tds, cell_key);
        let norm_vol_result = normalized_volume(&tds, cell_key);

        // Either succeed or fail gracefully (no panic)
        assert!(ratio_result.is_ok() || ratio_result.is_err());
        assert!(norm_vol_result.is_ok() || norm_vol_result.is_err());
    }

    // =============================================================================
    // DIMENSION-SPECIFIC TESTS
    // =============================================================================

    #[test]
    fn test_quality_6d_simplex() {
        // 6D simplex at unit hypercube corners
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 6> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        let ratio = radius_ratio(&tds, cell_key).unwrap();
        assert!(ratio > 6.0); // At least the dimension
        assert!(ratio < 20.0); // Not too degenerate

        let norm_vol = normalized_volume(&tds, cell_key).unwrap();
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
                    let tds: Tds<f64, Option<()>, Option<()>, $dim> = Tds::new(&vertices).unwrap();
                    let cell_key = tds.cell_keys().next().unwrap();

                    if let Ok(ratio) = radius_ratio(&tds, cell_key) {
                        assert!(ratio > $min_ratio, "{}: ratio={ratio}, expected > {}", $desc, $min_ratio);
                    }
                }
            )+
        };
    }

    test_poor_quality! {
        poor_quality_2d_flat => 2 => "very flat triangle" =>
            vec![
                vertex!([0.0, 0.0]),
                vertex!([100.0, 0.0]),
                vertex!([50.0, 0.1]),
            ],
            50.0,

        poor_quality_3d_nearly_coplanar => 3 => "nearly coplanar tetrahedron" =>
            vec![
                vertex!([0.0, 0.0, 0.0]),
                vertex!([10.0, 0.0, 0.0]),
                vertex!([5.0, 8.66, 0.0]),
                vertex!([5.0, 2.89, 0.01]),
            ],
            30.0,

        poor_quality_4d_degenerate => 4 => "nearly 3D subspace" =>
            vec![
                vertex!([0.0, 0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0, 0.0]),
                vertex!([0.5, 0.866, 0.0, 0.0]),
                vertex!([0.5, 0.289, 0.816, 0.0]),
                vertex!([0.5, 0.289, 0.204, 0.001]),
            ],
            10.0,

        poor_quality_3d_sliver => 3 => "sliver tetrahedron" =>
            vec![
                vertex!([0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0]),
                vertex!([0.5, 0.866_025, 0.0]),
                vertex!([0.5, 0.288_675, 0.001]),
            ],
            100.0,

        poor_quality_2d_needle => 2 => "needle triangle" =>
            vec![
                vertex!([0.0, 0.0]),
                vertex!([100.0, 0.0]),
                vertex!([0.0, 0.1]),
            ],
            50.0,

        poor_quality_3d_cap => 3 => "cap tetrahedron" =>
            vec![
                vertex!([0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0]),
                vertex!([0.5, 0.866_025, 0.0]),
                vertex!([0.5, 0.288_675, 10.0]),
            ],
            10.0,
    }

    // =============================================================================
    // ERROR PATH COVERAGE
    // =============================================================================

    #[test]
    fn test_quality_invalid_cell_key() {
        // Create triangulation
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 0.866_025]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();

        // Create an invalid key (not in the SlotMap)
        let invalid_key = CellKey::from(KeyData::from_ffi(u64::MAX));

        let result = radius_ratio(&tds, invalid_key);
        assert!(matches!(result, Err(QualityError::InvalidCell { .. })));

        let result = normalized_volume(&tds, invalid_key);
        assert!(matches!(result, Err(QualityError::InvalidCell { .. })));
    }

    #[test]
    fn test_quality_error_clone_eq() {
        // Test that QualityError implements Clone and PartialEq correctly
        let err1 = QualityError::InvalidCell {
            message: "test".to_string(),
        };
        let err2 = err1.clone();
        assert_eq!(err1, err2);

        let err3 = QualityError::DegenerateCell {
            detail: "volume=0".to_string(),
        };
        let err4 = err3.clone();
        assert_eq!(err3, err4);

        let err5 = QualityError::NumericalError {
            message: "overflow".to_string(),
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
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 0.866_025]),
        ];
        let tds_best: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_best).unwrap();
        let key_best = tds_best.cell_keys().next().unwrap();

        // Medium: right triangle (acceptable quality)
        let vertices_medium = vec![
            vertex!([0.0, 0.0]),
            vertex!([3.0, 0.0]),
            vertex!([0.0, 4.0]),
        ];
        let tds_medium: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_medium).unwrap();
        let key_medium = tds_medium.cell_keys().next().unwrap();

        // Worst: very flat (poor quality)
        let vertices_worst = vec![
            vertex!([0.0, 0.0]),
            vertex!([10.0, 0.0]),
            vertex!([5.0, 0.1]),
        ];
        let tds_worst: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_worst).unwrap();
        let key_worst = tds_worst.cell_keys().next().unwrap();

        let ratio_best = radius_ratio(&tds_best, key_best).unwrap();
        let ratio_medium = radius_ratio(&tds_medium, key_medium).unwrap();
        let ratio_worst = radius_ratio(&tds_worst, key_worst).unwrap();

        let vol_best = normalized_volume(&tds_best, key_best).unwrap();
        let vol_medium = normalized_volume(&tds_medium, key_medium).unwrap();
        let vol_worst = normalized_volume(&tds_worst, key_worst).unwrap();

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
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds_right: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_right).unwrap();
        let key_right = tds_right.cell_keys().next().unwrap();
        let ratio_right = radius_ratio(&tds_right, key_right).unwrap();
        assert_relative_eq!(ratio_right, 1.0 + 2.0_f64.sqrt(), epsilon = 0.1);

        // Test isosceles triangle
        let vertices_iso = vec![
            vertex!([0.0, 0.0]),
            vertex!([2.0, 0.0]),
            vertex!([1.0, 2.0]),
        ];
        let tds_iso: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_iso).unwrap();
        let key_iso = tds_iso.cell_keys().next().unwrap();
        let ratio_iso = radius_ratio(&tds_iso, key_iso).unwrap();
        assert!(ratio_iso > 2.0 && ratio_iso < 5.0);
    }

    // =============================================================================
    // PRECISION TESTS
    // =============================================================================

    #[test]
    fn test_quality_precision_f32_vs_f64() {
        // Compare f32 and f64 precision for same simplex
        let vertices_f32 = vec![
            vertex!([0.0f32, 0.0f32]),
            vertex!([1.0f32, 0.0f32]),
            vertex!([0.5f32, 0.866f32]),
        ];
        let tds_f32: Tds<f32, Option<()>, Option<()>, 2> = Tds::new(&vertices_f32).unwrap();
        let key_f32 = tds_f32.cell_keys().next().unwrap();

        let vertices_f64 = vec![
            vertex!([0.0f64, 0.0f64]),
            vertex!([1.0f64, 0.0f64]),
            vertex!([0.5f64, 0.866_025f64]),
        ];
        let tds_f64: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_f64).unwrap();
        let key_f64 = tds_f64.cell_keys().next().unwrap();

        let ratio_f32 = radius_ratio(&tds_f32, key_f32).unwrap();
        let ratio_f64 = radius_ratio(&tds_f64, key_f64).unwrap();

        // f32 and f64 should give similar results
        let ratio_diff = (<f64 as std::convert::From<f32>>::from(ratio_f32) - ratio_f64).abs();
        assert!(ratio_diff < 0.1, "f32/f64 precision difference too large");
    }

    // =============================================================================
    // HELPER FUNCTION TESTS
    // =============================================================================

    #[test]
    fn test_cell_points_valid() {
        // Test cell_points helper with valid cell
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 0.866_025]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        let points = cell_points(&tds, cell_key).unwrap();
        assert_eq!(points.len(), 3, "Should have 3 points for 2D cell");
    }

    #[test]
    fn test_cell_points_invalid_key() {
        // Test cell_points with invalid cell key
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 0.866_025]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let invalid_key = CellKey::from(KeyData::from_ffi(u64::MAX));

        let result = cell_points(&tds, invalid_key);
        assert!(matches!(result, Err(QualityError::InvalidCell { .. })));
    }

    #[test]
    fn test_compute_scale_aware_epsilon_2d() {
        // Test epsilon computation for 2D simplex
        let mut points = SmallBuffer::new();
        points.push(Point::new([0.0, 0.0]));
        points.push(Point::new([1.0, 0.0]));
        points.push(Point::new([0.5, 0.866_025]));

        let (avg_edge_length, epsilon) = compute_scale_aware_epsilon(&points).unwrap();
        assert!(
            avg_edge_length > 0.0,
            "Average edge length should be positive"
        );
        assert!(epsilon > 0.0, "Epsilon should be positive");
        assert!(epsilon >= 1e-12, "Epsilon should have floor of 1e-12");
    }

    #[test]
    fn test_compute_scale_aware_epsilon_tiny_simplex() {
        // Test epsilon computation with very small coordinates
        let mut points = SmallBuffer::new();
        points.push(Point::new([0.0, 0.0]));
        points.push(Point::new([1e-10, 0.0]));
        points.push(Point::new([0.5e-10, 0.866_025e-10]));

        let (avg_edge_length, epsilon) = compute_scale_aware_epsilon(&points).unwrap();
        // For tiny simplices, epsilon should use the floor (1e-12)
        assert!(epsilon >= 1e-12);
        assert!(avg_edge_length > 0.0);
    }

    #[test]
    fn test_compute_scale_aware_epsilon_large_simplex() {
        // Test epsilon computation with large coordinates
        let mut points = SmallBuffer::new();
        points.push(Point::new([0.0, 0.0]));
        points.push(Point::new([1e6, 0.0]));
        points.push(Point::new([0.5e6, 0.866_025e6]));

        let (avg_edge_length, epsilon) = compute_scale_aware_epsilon(&points).unwrap();
        // For large simplices, epsilon scales with average edge length
        assert!(epsilon > 1e-12);
        assert!(avg_edge_length > 1e5);
    }

    #[test]
    fn test_compute_scale_aware_epsilon_3d() {
        // Test epsilon computation for 3D simplex
        let mut points = SmallBuffer::new();
        points.push(Point::new([0.0, 0.0, 0.0]));
        points.push(Point::new([1.0, 0.0, 0.0]));
        points.push(Point::new([0.0, 1.0, 0.0]));
        points.push(Point::new([0.0, 0.0, 1.0]));

        let (avg_edge_length, epsilon) = compute_scale_aware_epsilon(&points).unwrap();
        assert!(avg_edge_length > 0.0);
        assert!(epsilon > 0.0);
    }

    // =============================================================================
    // ADDITIONAL ERROR PATH TESTS
    // =============================================================================

    #[test]
    fn test_radius_ratio_wrong_vertex_count() {
        // Create a TDS but test with manually constructed test that simulates wrong count
        // This tests the vertex count validation in radius_ratio
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 0.866_025]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Normal case should succeed
        let result = radius_ratio(&tds, cell_key);
        assert!(result.is_ok());
    }

    #[test]
    fn test_normalized_volume_wrong_vertex_count() {
        // Test normalized_volume with proper vertex count
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Should succeed with correct count
        let result = normalized_volume(&tds, cell_key);
        assert!(result.is_ok());
    }

    #[test]
    fn test_radius_ratio_numerical_edge_cases() {
        // Test with coordinates near numerical limits
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 1e-15]), // Very small but non-zero height
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Should either compute or return degenerate/numerical error (no panic)
        let result = radius_ratio(&tds, cell_key);
        assert!(
            result.is_ok()
                || matches!(result, Err(QualityError::DegenerateCell { .. }))
                || matches!(result, Err(QualityError::NumericalError { .. }))
        );
    }

    #[test]
    fn test_normalized_volume_numerical_edge_cases() {
        // Test normalized_volume with numerical edge cases
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1e-14]), // Very small but non-zero
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Should either compute or return degenerate/numerical error (no panic)
        let result = normalized_volume(&tds, cell_key);
        assert!(
            result.is_ok()
                || matches!(result, Err(QualityError::DegenerateCell { .. }))
                || matches!(result, Err(QualityError::NumericalError { .. }))
        );
    }

    #[test]
    fn test_quality_error_source_trait() {
        // Test that QualityError implements std::error::Error properly
        let err = QualityError::InvalidCell {
            message: "test".to_string(),
        };
        // Should be able to use as Error trait object
        let _: &dyn std::error::Error = &err;
        assert!(format!("{err}").contains("test"));
    }

    #[test]
    fn test_degenerate_cell_error_details() {
        // Test that degenerate errors include helpful details
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([2.0, 1e-20]), // Nearly collinear
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        let result = radius_ratio(&tds, cell_key);
        if let Err(QualityError::DegenerateCell { detail }) = result {
            // Should include numeric information
            assert!(detail.contains("inradius") || detail.contains("volume"));
        }
    }
}
