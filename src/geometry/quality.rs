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

    // sqrt(3) constant computed at compile time
    const SQRT_3: f64 = 1.732_050_807_568_877_3;
    use proptest::prelude::*;

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

    #[test]
    fn test_radius_ratio_4d_regular_simplex() {
        // Regular 4-simplex (5 vertices in 4D)
        // Using coordinates that form a regular 4-simplex
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.5, 0.866_025, 0.0, 0.0]),
            vertex!([0.5, 0.288_675, 0.816_497, 0.0]),
            vertex!([0.5, 0.288_675, 0.204_124, 0.790_569]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let ratio = radius_ratio(&tds, cell_key).unwrap();
        // For regular 4-simplex: ratio should be close to 4.0
        assert_relative_eq!(ratio, 4.0, epsilon = 0.2);
    }

    #[test]
    fn test_radius_ratio_5d_unit_simplex() {
        // 5D simplex at unit hypercube corners
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 5> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let ratio = radius_ratio(&tds, cell_key).unwrap();
        // Non-regular 5-simplex, but should have reasonable ratio
        assert!(ratio > 5.0); // At least the dimension
        assert!(ratio < 15.0); // Not too degenerate
    }

    #[test]
    fn test_normalized_volume_4d() {
        // 4D simplex
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let norm_vol = normalized_volume(&tds, cell_key).unwrap();
        assert!(norm_vol > 0.0);
        assert!(norm_vol < 1.0); // Reasonable normalized volume
    }

    #[test]
    fn test_normalized_volume_5d() {
        // 5D simplex
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 1.0, 0.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 1.0, 0.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 1.0, 0.0]),
            vertex!([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 5> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cells().keys().next().unwrap();

        let norm_vol = normalized_volume(&tds, cell_key).unwrap();
        assert!(norm_vol > 0.0);
        assert!(norm_vol < 1.0);
    }

    // =============================================================================
    // PROPERTY-BASED TESTS
    // =============================================================================

    proptest! {
        /// Property: Scale invariance for normalized volume
        ///
        /// For any simplex and positive scaling factor k,
        /// normalized_volume(k*S) ≈ normalized_volume(S)
        #[test]
        fn prop_normalized_volume_scale_invariant(
            scale in 0.1f64..10.0,
        ) {
            // Base equilateral triangle
            let vertices_base = vec![
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.5, 0.866_025]),
            ];
            let tds_base: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_base).unwrap();
            let cell_key_base = tds_base.cells().keys().next().unwrap();

            // Scaled triangle
            let vertices_scaled = vec![
                vertex!([0.0, 0.0]),
                vertex!([scale, 0.0]),
                vertex!([scale * 0.5, scale * 0.866_025]),
            ];
            let tds_scaled: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_scaled).unwrap();
            let cell_key_scaled = tds_scaled.cells().keys().next().unwrap();

            let norm_vol_base = normalized_volume(&tds_base, cell_key_base).unwrap();
            let norm_vol_scaled = normalized_volume(&tds_scaled, cell_key_scaled).unwrap();

            // Should be approximately equal (scale-invariant)
            prop_assert!((norm_vol_base - norm_vol_scaled).abs() < 1e-4);
        }

        /// Property: Translation invariance for both metrics
        ///
        /// Translating a simplex should not change its quality metrics
        #[test]
        fn prop_quality_translation_invariant(
            tx in -10.0f64..10.0,
            ty in -10.0f64..10.0,
        ) {
            // Base triangle
            let vertices_base = vec![
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.5, 0.866_025]),
            ];
            let tds_base: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_base).unwrap();
            let cell_key_base = tds_base.cells().keys().next().unwrap();

            // Translated triangle
            let vertices_translated = vec![
                vertex!([tx, ty]),
                vertex!([1.0 + tx, ty]),
                vertex!([0.5 + tx, 0.866_025 + ty]),
            ];
            let tds_translated: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_translated).unwrap();
            let cell_key_translated = tds_translated.cells().keys().next().unwrap();

            let ratio_base = radius_ratio(&tds_base, cell_key_base).unwrap();
            let ratio_translated = radius_ratio(&tds_translated, cell_key_translated).unwrap();

            let norm_vol_base = normalized_volume(&tds_base, cell_key_base).unwrap();
            let norm_vol_translated = normalized_volume(&tds_translated, cell_key_translated).unwrap();

            // Both metrics should be invariant under translation
            prop_assert!((ratio_base - ratio_translated).abs() < 1e-10);
            prop_assert!((norm_vol_base - norm_vol_translated).abs() < 1e-10);
        }

        /// Property: Radius ratio lower bound
        ///
        /// For any valid D-dimensional simplex, radius_ratio ≥ D
        /// (equality holds for regular simplices)
        #[test]
        fn prop_radius_ratio_lower_bound_2d(
            x1 in 0.1f64..2.0,
            _y1 in 0.1f64..2.0,
            x2 in 0.1f64..2.0,
            y2 in 0.1f64..2.0,
        ) {
            // Create a non-degenerate triangle
            let vertices = vec![
                vertex!([0.0, 0.0]),
                vertex!([x1, 0.0]),
                vertex!([x2, y2]),
            ];

            // Skip if triangle is too degenerate
            let area = 0.5 * (x1 * y2).abs();
            if area < 0.01 {
                return Ok(());
            }

            let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
            let cell_key = tds.cells().keys().next().unwrap();

            if let Ok(ratio) = radius_ratio(&tds, cell_key) {
                // For 2D, ratio ≥ 2.0 (with some numerical tolerance)
                prop_assert!(ratio >= 1.9);
            }
        }

        /// Property: Radius ratio lower bound for 3D
        #[test]
        fn prop_radius_ratio_lower_bound_3d(
            x1 in 0.5f64..2.0,
            z1 in 0.5f64..2.0,
        ) {
            // Create a tetrahedron based on unit cube
            let vertices = vec![
                vertex!([0.0, 0.0, 0.0]),
                vertex!([x1, 0.0, 0.0]),
                vertex!([0.0, x1, 0.0]),
                vertex!([0.0, 0.0, z1]),
            ];

            let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
            let cell_key = tds.cells().keys().next().unwrap();

            if let Ok(ratio) = radius_ratio(&tds, cell_key) {
                // For 3D, ratio ≥ 3.0 (with numerical tolerance)
                prop_assert!(ratio >= 2.8);
            }
        }

        /// Property: Normalized volume positivity
        ///
        /// For any non-degenerate simplex, normalized_volume > 0
        #[test]
        fn prop_normalized_volume_positive_2d(
            x1 in 0.5f64..2.0,
            y1 in 0.5f64..2.0,
            x2 in 0.1f64..2.0,
            y2 in 0.5f64..2.0,
        ) {
            let vertices = vec![
                vertex!([0.0, 0.0]),
                vertex!([x1, y1]),
                vertex!([x2, y2]),
            ];

            // Skip near-degenerate cases
            let det = x1.mul_add(y2, -(x2 * y1));
            if det.abs() < 0.1 {
                return Ok(());
            }

            let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
            let cell_key = tds.cells().keys().next().unwrap();

            if let Ok(norm_vol) = normalized_volume(&tds, cell_key) {
                prop_assert!(norm_vol > 0.0);
            }
        }

        /// Property: Metrics correlation
        ///
        /// As simplex quality decreases (flatter), radius_ratio increases
        /// and normalized_volume decreases
        #[test]
        fn prop_metrics_correlation(
            height in 0.1f64..1.0,
        ) {
            // Create two triangles with different heights (same base)
            let vertices_tall = vec![
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.5, height]),
            ];
            let vertices_short = vec![
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.5, height * 0.5]),
            ];

            let tds_tall: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_tall).unwrap();
            let tds_short: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_short).unwrap();

            let cell_key_tall = tds_tall.cells().keys().next().unwrap();
            let cell_key_short = tds_short.cells().keys().next().unwrap();

            if let (Ok(ratio_tall), Ok(ratio_short)) = (
                radius_ratio(&tds_tall, cell_key_tall),
                radius_ratio(&tds_short, cell_key_short),
            )
                && let (Ok(vol_tall), Ok(vol_short)) = (
                    normalized_volume(&tds_tall, cell_key_tall),
                    normalized_volume(&tds_short, cell_key_short),
                ) {
                    // Taller triangle should have lower ratio and higher normalized volume
                    prop_assert!(ratio_tall < ratio_short);
                    prop_assert!(vol_tall > vol_short);
                }
        }

        /// Property: Rotation invariance (2D)
        ///
        /// Rotating a simplex should not change its quality metrics
        #[test]
        fn prop_quality_rotation_invariant_2d(
            angle in 0.0f64..std::f64::consts::TAU,
        ) {

            // Base equilateral triangle
            let vertices_base = vec![
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.5, SQRT_3 / 2.0]),
            ];
            let tds_base: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_base).unwrap();
            let cell_key_base = tds_base.cells().keys().next().unwrap();

            // Rotate triangle around origin
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let rotate = |p: [f64; 2]| -> [f64; 2] {
                [p[0].mul_add(cos_a, -(p[1] * sin_a)), p[0].mul_add(sin_a, p[1] * cos_a)]
            };

            let vertices_rotated = vec![
                vertex!(rotate([0.0, 0.0])),
                vertex!(rotate([1.0, 0.0])),
                vertex!(rotate([0.5, SQRT_3 / 2.0])),
            ];
            let tds_rotated: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_rotated).unwrap();
            let cell_key_rotated = tds_rotated.cells().keys().next().unwrap();

            let ratio_base = radius_ratio(&tds_base, cell_key_base).unwrap();
            let ratio_rotated = radius_ratio(&tds_rotated, cell_key_rotated).unwrap();

            let norm_vol_base = normalized_volume(&tds_base, cell_key_base).unwrap();
            let norm_vol_rotated = normalized_volume(&tds_rotated, cell_key_rotated).unwrap();

            // Both metrics should be invariant under rotation
            prop_assert!((ratio_base - ratio_rotated).abs() < 1e-9);
            prop_assert!((norm_vol_base - norm_vol_rotated).abs() < 1e-9);
        }
    }
}
