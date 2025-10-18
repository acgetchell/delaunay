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
//! - **Normalized Volume**: Volume normalized by edge lengths. Provides a
//!   scale-invariant measure of cell shape quality.
//!
//! # References
//!
//! - Shewchuk, J.R. "What Is a Good Linear Element? Interpolation, Conditioning,
//!   Anisotropy, and Quality Measures" (2002)
//! - Liu, A. and Joe, B. "Relationship between tetrahedron shape measures"
//!   *BIT Numerical Mathematics* 34.2 (1994): 268-287
//! - Field, D.A. "Qualitative measures for initial meshes" *International Journal
//!   for Numerical Methods in Engineering* 47.4 (2000): 887-906

use crate::core::{
    traits::data_type::DataType,
    triangulation_data_structure::{CellKey, Tds},
};
use crate::geometry::{
    point::Point,
    traits::coordinate::CoordinateScalar,
    util::{circumradius, hypot, inradius as simplex_inradius, simplex_volume},
};
use num_traits::NumCast;
use serde::{Serialize, de::DeserializeOwned};
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
        /// Approximate measure of degeneracy
        volume: String,
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
            Self::DegenerateCell { volume } => {
                write!(f, "Degenerate cell with volume ≈ {volume}")
            }
            Self::NumericalError { message } => write!(f, "Numerical error: {message}"),
        }
    }
}

impl Error for QualityError {}

/// Computes the radius ratio quality metric for a cell.
///
/// The radius ratio is defined as the circumradius divided by the inradius.
/// Lower values indicate better cell quality. An equilateral simplex in d dimensions
/// has a radius ratio of d, which is optimal.
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
/// let cell_key = tds.cells().keys().next().unwrap();
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
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + Sum + NumCast,
    U: DataType + DeserializeOwned,
    V: DataType + DeserializeOwned,
    for<'a> &'a T: Div<T>,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    // Get cell vertex keys
    let vertex_keys = tds
        .get_cell_vertices(cell_key)
        .map_err(|e| QualityError::InvalidCell {
            message: format!("Failed to get cell vertices: {e}"),
        })?;

    // Get vertex points
    let points: Vec<Point<T, D>> = vertex_keys
        .iter()
        .map(|&vkey| {
            tds.vertices()
                .get(vkey)
                .map(|v| *v.point())
                .ok_or_else(|| QualityError::InvalidCell {
                    message: format!("Vertex {vkey:?} not found in triangulation"),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

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

    // Check for near-zero inradius (degenerate cell)
    let epsilon = T::from(1e-10).unwrap_or_else(T::zero);
    if inradius_val < epsilon {
        return Err(QualityError::DegenerateCell {
            volume: format!("inradius={inradius_val:?}"),
        });
    }

    // radius_ratio = circumradius / inradius
    let ratio = circumradius_val / inradius_val;

    Ok(ratio)
}

/// Computes the normalized volume quality metric for a cell.
///
/// This metric provides a scale-invariant measure of cell quality by normalizing
/// the volume by the product of edge lengths. It avoids the numerical issues that
/// can arise when computing inradius for very small cells.
///
/// # Quality Interpretation
///
/// - **Higher values** = better quality
/// - **Optimal** (equilateral): ≈ 1.0 (normalized)
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
/// let cell_key = tds.cells().keys().next().unwrap();
///
/// let norm_vol = normalized_volume(&tds, cell_key).unwrap();
/// assert!(norm_vol > 0.0);
/// ```
pub fn normalized_volume<T, U, V, const D: usize>(
    tds: &Tds<T, U, V, D>,
    cell_key: CellKey,
) -> Result<T, QualityError>
where
    T: CoordinateScalar + AddAssign<T> + SubAssign<T> + Sum + NumCast,
    U: DataType + DeserializeOwned,
    V: DataType + DeserializeOwned,
    for<'a> &'a T: Div<T>,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    // Get cell vertex keys
    let vertex_keys = tds
        .get_cell_vertices(cell_key)
        .map_err(|e| QualityError::InvalidCell {
            message: format!("Failed to get cell vertices: {e}"),
        })?;

    // Get vertex points
    let points: Vec<Point<T, D>> = vertex_keys
        .iter()
        .map(|&vkey| {
            tds.vertices()
                .get(vkey)
                .map(|v| *v.point())
                .ok_or_else(|| QualityError::InvalidCell {
                    message: format!("Vertex {vkey:?} not found in triangulation"),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

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

    // Check for degenerate cell
    let epsilon = T::from(1e-10).unwrap_or_else(T::zero);
    if volume < epsilon {
        return Err(QualityError::DegenerateCell {
            volume: format!("{volume:?}"),
        });
    }

    // Compute average edge length for normalization
    let mut total_edge_length = T::zero();
    let mut edge_count = 0;

    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            // Compute distance between points i and j using hypot for numerical stability
            let mut diff_coords = [T::zero(); D];
            for (idx, diff) in diff_coords.iter_mut().enumerate() {
                *diff = points[i].coords()[idx] - points[j].coords()[idx];
            }
            let dist = hypot(diff_coords);
            total_edge_length += dist;
            edge_count += 1;
        }
    }

    if edge_count == 0 {
        return Err(QualityError::InvalidCell {
            message: "No edges found in cell".to_string(),
        });
    }

    let edge_count_t = T::from(edge_count).ok_or_else(|| QualityError::NumericalError {
        message: format!("Failed to convert edge count {edge_count} to coordinate type"),
    })?;

    let avg_edge_length = total_edge_length / edge_count_t;

    if avg_edge_length < epsilon {
        return Err(QualityError::DegenerateCell {
            volume: format!("avg_edge_length={avg_edge_length:?}"),
        });
    }

    // Normalize volume by edge_length^D for scale invariance
    let mut edge_length_power = T::one();
    for _ in 0..D {
        edge_length_power = edge_length_power * avg_edge_length;
    }

    if edge_length_power < epsilon {
        return Err(QualityError::DegenerateCell {
            volume: format!("edge_length_power={edge_length_power:?}"),
        });
    }

    let normalized = volume / edge_length_power;

    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::triangulation_data_structure::Tds;
    use crate::vertex;
    use approx::assert_relative_eq;

    // =============================================================================
    // RADIUS RATIO TESTS
    // =============================================================================

    #[test]
    fn test_radius_ratio_2d_equilateral_triangle() {
        // Equilateral triangle - should have ratio close to 2.0 (optimal for 2D)
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 0.866_025]), // sqrt(3)/2
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let ratio = radius_ratio(&tds, cell_key).unwrap();
        // For equilateral triangle: circumradius/inradius = 2
        assert_relative_eq!(ratio, 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_radius_ratio_2d_right_triangle() {
        // Right triangle with legs 3 and 4
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([3.0, 0.0]),
            vertex!([0.0, 4.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let ratio = radius_ratio(&tds, cell_key).unwrap();
        // Right triangle should have ratio > 2 (non-optimal)
        assert!(ratio > 2.0);
        assert!(ratio < 5.0); // But not too degenerate
    }

    #[test]
    fn test_radius_ratio_3d_regular_tetrahedron() {
        // Regular tetrahedron - should have ratio close to 3.0 (optimal for 3D)
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.5, 0.866_025, 0.0]), // equilateral triangle base
            vertex!([0.5, 0.288_675, 0.816_497]), // apex (regular tetrahedron height)
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let ratio = radius_ratio(&tds, cell_key).unwrap();
        // For regular tetrahedron: ratio should be close to 3.0
        assert_relative_eq!(ratio, 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_radius_ratio_3d_unit_cube_tetrahedron() {
        // Tetrahedron at unit cube corners - not regular, ratio > 3
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let ratio = radius_ratio(&tds, cell_key).unwrap();
        // This tetrahedron is not regular, so ratio > 3
        assert!(ratio > 3.0);
    }

    #[test]
    fn test_radius_ratio_degenerate_triangle() {
        // Nearly collinear points (degenerate triangle)
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([2.0, 0.001]), // Nearly collinear
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        // Should either compute very high ratio or fail with degenerate error
        let result = radius_ratio(&tds, cell_key);
        if let Ok(ratio) = result {
            assert!(ratio > 10.0); // Very poor quality
        } else {
            // Degenerate error is also acceptable
            assert!(matches!(result, Err(QualityError::DegenerateCell { .. })));
        }
    }

    // =============================================================================
    // NORMALIZED VOLUME TESTS
    // =============================================================================

    #[test]
    fn test_normalized_volume_2d_equilateral_triangle() {
        // Equilateral triangle - optimal normalized volume
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 0.866_025]), // sqrt(3)/2
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let norm_vol = normalized_volume(&tds, cell_key).unwrap();
        // Should be positive and reasonable for equilateral
        assert!(norm_vol > 0.3);
        assert!(norm_vol < 0.6);
    }

    #[test]
    fn test_normalized_volume_2d_right_triangle() {
        // Right triangle with legs 3 and 4
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([3.0, 0.0]),
            vertex!([0.0, 4.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let norm_vol = normalized_volume(&tds, cell_key).unwrap();
        assert!(norm_vol > 0.0);
        // Right triangle is reasonable quality but not optimal
        assert!(norm_vol < 0.6);
    }

    #[test]
    fn test_normalized_volume_3d_tetrahedron() {
        // Regular tetrahedron at unit cube corners
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let norm_vol = normalized_volume(&tds, cell_key).unwrap();
        assert!(norm_vol > 0.0);
    }

    #[test]
    fn test_normalized_volume_scale_invariance() {
        // Test that normalized volume is scale-invariant
        // Small triangle
        let vertices_small = vec![
            vertex!([0.0, 0.0]),
            vertex!([0.1, 0.0]),
            vertex!([0.05, 0.086_602_5]), // scaled by 0.1
        ];
        let tds_small: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_small).unwrap();
        let cell_key_small = tds_small.cells().keys().next().unwrap();

        // Large triangle (scaled by 10)
        let vertices_large = vec![
            vertex!([0.0, 0.0]),
            vertex!([10.0, 0.0]),
            vertex!([5.0, 8.66025]), // scaled by 10
        ];
        let tds_large: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_large).unwrap();
        let cell_key_large = tds_large.cells().keys().next().unwrap();

        let norm_vol_small = normalized_volume(&tds_small, cell_key_small).unwrap();
        let norm_vol_large = normalized_volume(&tds_large, cell_key_large).unwrap();

        // Normalized volumes should be approximately equal (scale-invariant)
        assert_relative_eq!(norm_vol_small, norm_vol_large, epsilon = 1e-5);
    }

    #[test]
    fn test_normalized_volume_degenerate() {
        // Nearly collinear points
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([2.0, 0.001]), // Nearly collinear
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let result = normalized_volume(&tds, cell_key);
        if let Ok(norm_vol) = result {
            // Very small normalized volume for degenerate cell
            assert!(norm_vol < 0.01);
        } else {
            // Degenerate error is also acceptable
            assert!(matches!(result, Err(QualityError::DegenerateCell { .. })));
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
            volume: "0.0".to_string(),
        };
        assert!(format!("{err}").contains("Degenerate"));
        assert!(format!("{err}").contains("0.0"));

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
        let cell_key_good = tds_good.cells().keys().next().unwrap();

        // Poor quality triangle (very flat)
        let vertices_poor = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 0.01]), // Nearly flat
        ];
        let tds_poor: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_poor).unwrap();
        let cell_key_poor = tds_poor.cells().keys().next().unwrap();

        let ratio_good = radius_ratio(&tds_good, cell_key_good).unwrap();
        let ratio_poor = radius_ratio(&tds_poor, cell_key_poor).unwrap();

        let norm_vol_good = normalized_volume(&tds_good, cell_key_good).unwrap();
        let norm_vol_poor = normalized_volume(&tds_poor, cell_key_poor).unwrap();

        // Good triangle: lower ratio, higher normalized volume
        assert!(ratio_good < ratio_poor);
        assert!(norm_vol_good > norm_vol_poor);
    }
}
