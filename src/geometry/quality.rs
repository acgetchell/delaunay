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
    // Compute scale from average coordinate magnitude to handle varying data scales
    let mut coord_sum = T::zero();
    for point in &points {
        for &coord in point.coords() {
            coord_sum += coord.abs();
        }
    }
    let total_coords =
        NumCast::from(points.len() * D).ok_or_else(|| QualityError::NumericalError {
            message: "Failed to convert total coordinate count to type T".to_string(),
        })?;
    let scale = coord_sum / total_coords;

    // Use relative epsilon with a minimum floor to handle both tiny and huge simplices
    let floor: T = NumCast::from(1e-12).ok_or_else(|| QualityError::NumericalError {
        message: "Failed to convert floor epsilon (1e-12) to coordinate type".to_string(),
    })?;
    let relative_factor: T = NumCast::from(1e-8).ok_or_else(|| QualityError::NumericalError {
        message: "Failed to convert relative factor (1e-8) to coordinate type".to_string(),
    })?;
    let epsilon = floor.max(scale * relative_factor);

    if inradius_val < epsilon {
        return Err(QualityError::DegenerateCell {
            detail: format!("inradius={inradius_val:?}, epsilon={epsilon:?}, scale={scale:?}"),
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

    // Check for degenerate cell using scale-aware tolerance
    // Compute scale from average coordinate magnitude
    let mut coord_sum = T::zero();
    for point in &points {
        for &coord in point.coords() {
            coord_sum += coord.abs();
        }
    }
    let total_coords =
        NumCast::from(points.len() * D).ok_or_else(|| QualityError::NumericalError {
            message: "Failed to convert total coordinate count to type T".to_string(),
        })?;
    let scale = coord_sum / total_coords;

    // Use relative epsilon with a minimum floor
    let floor: T = NumCast::from(1e-12).ok_or_else(|| QualityError::NumericalError {
        message: "Failed to convert floor epsilon (1e-12) to coordinate type".to_string(),
    })?;
    let relative_factor: T = NumCast::from(1e-8).ok_or_else(|| QualityError::NumericalError {
        message: "Failed to convert relative factor (1e-8) to coordinate type".to_string(),
    })?;
    let epsilon = floor.max(scale * relative_factor);

    if volume < epsilon {
        return Err(QualityError::DegenerateCell {
            detail: format!("volume={volume:?}, epsilon={epsilon:?}, scale={scale:?}"),
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

    // Check edge_length_power using scale-aware epsilon
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
    use crate::vertex;
    use approx::assert_relative_eq;

    // sqrt(3) constant computed at compile time
    const SQRT_3: f64 = 1.732_050_807_568_877_3;
    use proptest::prelude::*;
    use slotmap::KeyData;

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
        let cell_key = tds.cell_keys().next().unwrap();

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
        let cell_key = tds.cell_keys().next().unwrap();

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
        let cell_key = tds.cell_keys().next().unwrap();

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
        let cell_key = tds.cell_keys().next().unwrap();

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
        let cell_key = tds.cell_keys().next().unwrap();

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
        let cell_key = tds.cell_keys().next().unwrap();

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
        let cell_key = tds.cell_keys().next().unwrap();

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
        let cell_key_small = tds_small.cell_keys().next().unwrap();

        // Large triangle (scaled by 10)
        let vertices_large = vec![
            vertex!([0.0, 0.0]),
            vertex!([10.0, 0.0]),
            vertex!([5.0, 8.66025]), // scaled by 10
        ];
        let tds_large: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_large).unwrap();
        let cell_key_large = tds_large.cell_keys().next().unwrap();

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
        let cell_key = tds.cell_keys().next().unwrap();

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
        let cell_key = tds.cell_keys().next().unwrap();

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
        let cell_key = tds.cell_keys().next().unwrap();

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
        let cell_key = tds.cell_keys().next().unwrap();

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
        let cell_key = tds.cell_keys().next().unwrap();

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
        let cell_key = tds.cell_keys().next().unwrap();

        let norm_vol = normalized_volume(&tds, cell_key).unwrap();
        assert!(norm_vol > 0.0);
        assert!(norm_vol < 1.0);
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
            let cell_key_base = tds_base.cell_keys().next().unwrap();

            // Scaled triangle
            let vertices_scaled = vec![
                vertex!([0.0, 0.0]),
                vertex!([scale, 0.0]),
                vertex!([scale * 0.5, scale * 0.866_025]),
            ];
            let tds_scaled: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_scaled).unwrap();
            let cell_key_scaled = tds_scaled.cell_keys().next().unwrap();

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
            let cell_key_base = tds_base.cell_keys().next().unwrap();

            // Translated triangle
            let vertices_translated = vec![
                vertex!([tx, ty]),
                vertex!([1.0 + tx, ty]),
                vertex!([0.5 + tx, 0.866_025 + ty]),
            ];
            let tds_translated: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_translated).unwrap();
            let cell_key_translated = tds_translated.cell_keys().next().unwrap();

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
            let cell_key = tds.cell_keys().next().unwrap();

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
            let cell_key = tds.cell_keys().next().unwrap();

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
            let cell_key = tds.cell_keys().next().unwrap();

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

            let cell_key_tall = tds_tall.cell_keys().next().unwrap();
            let cell_key_short = tds_short.cell_keys().next().unwrap();

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
            let cell_key_base = tds_base.cell_keys().next().unwrap();

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
            let cell_key_rotated = tds_rotated.cell_keys().next().unwrap();

            let ratio_base = radius_ratio(&tds_base, cell_key_base).unwrap();
            let ratio_rotated = radius_ratio(&tds_rotated, cell_key_rotated).unwrap();

            let norm_vol_base = normalized_volume(&tds_base, cell_key_base).unwrap();
            let norm_vol_rotated = normalized_volume(&tds_rotated, cell_key_rotated).unwrap();

            // Both metrics should be invariant under rotation
            prop_assert!((ratio_base - ratio_rotated).abs() < 1e-9);
            prop_assert!((norm_vol_base - norm_vol_rotated).abs() < 1e-9);
        }

        /// Property: Scale stability across many orders of magnitude
        ///
        /// Quality metrics should work correctly for very large and very small coordinates
        #[test]
        fn prop_quality_scale_stability(
            scale_log in -10i32..10,
        ) {
            let scale = 10.0_f64.powi(scale_log);

            // Equilateral triangle scaled by scale factor
            let vertices = vec![
                vertex!([0.0, 0.0]),
                vertex!([scale, 0.0]),
                vertex!([scale * 0.5, scale * 0.866_025]),
            ];
            let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
            let cell_key = tds.cell_keys().next().unwrap();

            // Radius ratio should be ~2 regardless of scale
            if let Ok(ratio) = radius_ratio(&tds, cell_key) {
                prop_assert!(ratio > 1.5 && ratio < 2.5, "ratio={ratio} at scale={scale}");
            }

            // Normalized volume should be ~0.433 regardless of scale
            if let Ok(norm_vol) = normalized_volume(&tds, cell_key) {
                prop_assert!(norm_vol > 0.3 && norm_vol < 0.6, "norm_vol={norm_vol} at scale={scale}");
            }
        }

        /// Property: Reflection invariance
        ///
        /// Reflecting a simplex across an axis should not change quality
        #[test]
        fn prop_quality_reflection_invariant_2d(
            reflect_x in prop::bool::ANY,
            reflect_y in prop::bool::ANY,
        ) {
            // Base equilateral triangle
            let vertices_base = vec![
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.5, SQRT_3 / 2.0]),
            ];
            let tds_base: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_base).unwrap();
            let cell_key_base = tds_base.cell_keys().next().unwrap();

            // Reflect triangle
            let reflect = |p: [f64; 2]| -> [f64; 2] {
                [
                    if reflect_x { -p[0] } else { p[0] },
                    if reflect_y { -p[1] } else { p[1] },
                ]
            };

            let vertices_reflected = vec![
                vertex!(reflect([0.0, 0.0])),
                vertex!(reflect([1.0, 0.0])),
                vertex!(reflect([0.5, SQRT_3 / 2.0])),
            ];
            let tds_reflected: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_reflected).unwrap();
            let cell_key_reflected = tds_reflected.cell_keys().next().unwrap();

            let ratio_base = radius_ratio(&tds_base, cell_key_base).unwrap();
            let ratio_reflected = radius_ratio(&tds_reflected, cell_key_reflected).unwrap();

            let norm_vol_base = normalized_volume(&tds_base, cell_key_base).unwrap();
            let norm_vol_reflected = normalized_volume(&tds_reflected, cell_key_reflected).unwrap();

            // Metrics should be invariant under reflection
            prop_assert!((ratio_base - ratio_reflected).abs() < 1e-9);
            prop_assert!((norm_vol_base - norm_vol_reflected).abs() < 1e-9);
        }

        /// Property: Aspect ratio monotonicity
        ///
        /// As aspect ratio worsens, radius_ratio increases and normalized_volume decreases
        #[test]
        fn prop_quality_aspect_ratio_monotonic(
            height1 in 0.3f64..1.0,
            height2 in 0.1f64..0.3,
        ) {
            // Ensure height1 > height2
            prop_assume!(height1 > height2 + 0.05);

            // Two triangles with same base, different heights
            let vertices_better = vec![
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.5, height1]), // Taller = better
            ];
            let vertices_worse = vec![
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.5, height2]), // Flatter = worse
            ];

            let tds_better: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_better).unwrap();
            let tds_worse: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_worse).unwrap();

            let key_better = tds_better.cell_keys().next().unwrap();
            let key_worse = tds_worse.cell_keys().next().unwrap();

            if let (Ok(ratio_better), Ok(ratio_worse)) = (
                radius_ratio(&tds_better, key_better),
                radius_ratio(&tds_worse, key_worse),
            ) {
                // Worse aspect ratio → higher radius ratio
                prop_assert!(ratio_better < ratio_worse);
            }

            if let (Ok(vol_better), Ok(vol_worse)) = (
                normalized_volume(&tds_better, key_better),
                normalized_volume(&tds_worse, key_worse),
            ) {
                // Worse aspect ratio → lower normalized volume
                prop_assert!(vol_better > vol_worse);
            }
        }

        /// Property: Quality ranking transitivity
        ///
        /// If simplex A is better than B and B is better than C, then A is better than C
        #[test]
        fn prop_quality_ranking_transitive(
            h1 in 0.7f64..1.0,
            h2 in 0.4f64..0.7,
            h3 in 0.1f64..0.4,
        ) {
            // Ensure strict ordering
            prop_assume!(h1 > h2 + 0.1 && h2 > h3 + 0.1);

            let vertices_a = vec![
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.5, h1]),
            ];
            let vertices_b = vec![
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.5, h2]),
            ];
            let vertices_c = vec![
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.5, h3]),
            ];

            let tds_a: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_a).unwrap();
            let tds_b: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_b).unwrap();
            let tds_c: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_c).unwrap();

            let key_a = tds_a.cell_keys().next().unwrap();
            let key_b = tds_b.cell_keys().next().unwrap();
            let key_c = tds_c.cell_keys().next().unwrap();

            if let (Ok(ratio_a), Ok(ratio_b), Ok(ratio_c)) = (
                radius_ratio(&tds_a, key_a),
                radius_ratio(&tds_b, key_b),
                radius_ratio(&tds_c, key_c),
            ) {
                // Verify transitivity: ratio_a < ratio_b < ratio_c
                prop_assert!(ratio_a < ratio_b);
                prop_assert!(ratio_b < ratio_c);
                prop_assert!(ratio_a < ratio_c);
            }
        }

        /// Property: Normalized volume upper bound
        ///
        /// Normalized volume should be bounded (equilateral is optimal)
        #[test]
        fn prop_normalized_volume_bounded(
            x1 in 0.5f64..2.0,
            y1 in 0.5f64..2.0,
            x2 in 0.5f64..2.0,
            y2 in 0.5f64..2.0,
        ) {
            let vertices = vec![
                vertex!([0.0, 0.0]),
                vertex!([x1, y1]),
                vertex!([x2, y2]),
            ];

            // Skip near-degenerate cases
            let det = x1.mul_add(y2, -(x2 * y1));
            if det.abs() < 0.2 {
                return Ok(());
            }

            let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
            let cell_key = tds.cell_keys().next().unwrap();

            if let Ok(norm_vol) = normalized_volume(&tds, cell_key) {
                // Should not exceed equilateral value (sqrt(3)/4 ≈ 0.433)
                prop_assert!(norm_vol <= 0.5, "norm_vol={norm_vol} exceeds reasonable bound");
                prop_assert!(norm_vol > 0.0);
            }
        }

        /// Property: Degeneracy detection
        ///
        /// Very flat simplices should be detected as degenerate or have very poor quality
        #[test]
        fn prop_quality_degeneracy_detection(
            degeneracy_factor in 1e-8f64..1e-3,
        ) {
            // Create nearly collinear points
            let vertices = vec![
                vertex!([0.0, 0.0]),
                vertex!([1.0, 0.0]),
                vertex!([0.5, degeneracy_factor]),
            ];
            let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
            let cell_key = tds.cell_keys().next().unwrap();

            match radius_ratio(&tds, cell_key) {
                Ok(ratio) => {
                    // If computed, should indicate very poor quality
                    prop_assert!(ratio > 10.0, "ratio={ratio} should indicate poor quality");
                }
                Err(QualityError::DegenerateCell { .. }) => (),
                Err(e) => return Err(proptest::test_runner::TestCaseError::fail(format!("Unexpected error: {e}")))
            }

            match normalized_volume(&tds, cell_key) {
                Ok(norm_vol) => {
                    // If computed, should be very small
                    prop_assert!(norm_vol < 0.01, "norm_vol={norm_vol} should be very small");
                }
                Err(QualityError::DegenerateCell { .. }) => (),
                Err(e) => return Err(proptest::test_runner::TestCaseError::fail(format!("Unexpected error: {e}")))
            }
        }

        /// Property: 3D rotation invariance
        ///
        /// Rotating a 3D simplex should not change quality
        #[test]
        fn prop_quality_rotation_invariant_3d(
            angle_xy in 0.0f64..std::f64::consts::TAU,
            angle_xz in 0.0f64..std::f64::consts::TAU,
        ) {
            // Regular tetrahedron
            let vertices_base = vec![
                vertex!([0.0, 0.0, 0.0]),
                vertex!([1.0, 0.0, 0.0]),
                vertex!([0.5, 0.866_025, 0.0]),
                vertex!([0.5, 0.288_675, 0.816_497]),
            ];
            let tds_base: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices_base).unwrap();
            let cell_key_base = tds_base.cell_keys().next().unwrap();

            // Rotate around x and y axes
            let cos_xy = angle_xy.cos();
            let sin_xy = angle_xy.sin();
            let cos_xz = angle_xz.cos();
            let sin_xz = angle_xz.sin();

            let rotate = |p: [f64; 3]| -> [f64; 3] {
                // Rotate in xy plane
                let x1 = p[0].mul_add(cos_xy, -(p[1] * sin_xy));
                let y1 = p[0].mul_add(sin_xy, p[1] * cos_xy);
                let z1 = p[2];

                // Rotate in xz plane
                let x2 = x1.mul_add(cos_xz, -(z1 * sin_xz));
                let y2 = y1;
                let z2 = x1.mul_add(sin_xz, z1 * cos_xz);

                [x2, y2, z2]
            };

            let vertices_rotated = vec![
                vertex!(rotate([0.0, 0.0, 0.0])),
                vertex!(rotate([1.0, 0.0, 0.0])),
                vertex!(rotate([0.5, 0.866_025, 0.0])),
                vertex!(rotate([0.5, 0.288_675, 0.816_497])),
            ];
            let tds_rotated: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices_rotated).unwrap();
            let cell_key_rotated = tds_rotated.cell_keys().next().unwrap();

            let ratio_base = radius_ratio(&tds_base, cell_key_base).unwrap();
            let ratio_rotated = radius_ratio(&tds_rotated, cell_key_rotated).unwrap();

            // Metrics should be invariant under rotation
            prop_assert!((ratio_base - ratio_rotated).abs() < 1e-8);
        }
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

    #[test]
    fn test_quality_aspect_ratio_extremes_2d() {
        // Very thin triangle (extreme aspect ratio)
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([100.0, 0.0]),
            vertex!([50.0, 0.1]), // Very flat
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Should have very poor quality
        let ratio = radius_ratio(&tds, cell_key).unwrap();
        assert!(ratio > 50.0); // Very high ratio

        let norm_vol = normalized_volume(&tds, cell_key).unwrap();
        assert!(norm_vol < 0.01); // Very low normalized volume
    }

    #[test]
    fn test_quality_aspect_ratio_extremes_3d() {
        // Very flat tetrahedron (extreme aspect ratio)
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([10.0, 0.0, 0.0]),
            vertex!([5.0, 8.66, 0.0]),
            vertex!([5.0, 2.89, 0.01]), // Nearly coplanar
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Should have very poor quality
        if let Ok(ratio) = radius_ratio(&tds, cell_key) {
            assert!(ratio > 30.0);
        }
    }

    #[test]
    fn test_quality_aspect_ratio_extremes_4d() {
        // 4D simplex with one vertex far from others
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0, 0.0]),
            vertex!([0.5, 0.866, 0.0, 0.0]),
            vertex!([0.5, 0.289, 0.816, 0.0]),
            vertex!([0.5, 0.289, 0.204, 0.001]), // Nearly in 3D subspace
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 4> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        if let Ok(ratio) = radius_ratio(&tds, cell_key) {
            assert!(ratio > 10.0);
        }
    }

    // =============================================================================
    // SLIVER DETECTION
    // =============================================================================

    #[test]
    fn test_radius_ratio_sliver_tetrahedron() {
        // Classic sliver tetrahedron - nearly coplanar vertices
        // but with large circumradius
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.5, 0.866_025, 0.0]),
            vertex!([0.5, 0.288_675, 0.001]), // Barely above the plane
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        // Sliver should have very high radius ratio
        match radius_ratio(&tds, cell_key) {
            Ok(ratio) => {
                assert!(ratio > 100.0, "Sliver should have very high radius ratio");
            }
            Err(QualityError::DegenerateCell { .. }) => (),
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_quality_needle_simplex() {
        // Needle-shaped simplex (one long edge, others short)
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([100.0, 0.0]), // Long edge
            vertex!([0.0, 0.1]),   // Short edges
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        let ratio = radius_ratio(&tds, cell_key).unwrap();
        assert!(ratio > 50.0); // Poor quality

        let norm_vol = normalized_volume(&tds, cell_key).unwrap();
        assert!(norm_vol < 0.01); // Low quality
    }

    #[test]
    fn test_quality_cap_simplex() {
        // Cap-shaped simplex in 3D
        let vertices = vec![
            vertex!([0.0, 0.0, 0.0]),
            vertex!([1.0, 0.0, 0.0]),
            vertex!([0.5, 0.866_025, 0.0]),  // Base triangle
            vertex!([0.5, 0.288_675, 10.0]), // Very tall
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 3> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        let ratio = radius_ratio(&tds, cell_key).unwrap();
        assert!(ratio > 10.0); // Poor quality due to extreme height
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
    fn test_quality_ranking_consistency() {
        // Create simplices with known quality ordering
        // Best: equilateral
        let vertices_best = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 0.866_025]),
        ];
        let tds_best: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_best).unwrap();
        let key_best = tds_best.cell_keys().next().unwrap();

        // Medium: right triangle
        let vertices_medium = vec![
            vertex!([0.0, 0.0]),
            vertex!([3.0, 0.0]),
            vertex!([0.0, 4.0]),
        ];
        let tds_medium: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_medium).unwrap();
        let key_medium = tds_medium.cell_keys().next().unwrap();

        // Worst: very flat
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
    }

    #[test]
    fn test_quality_thresholds() {
        // Test quality thresholds for mesh generation
        // Good quality threshold
        let vertices_good = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.5, 0.866_025]),
        ];
        let tds_good: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_good).unwrap();
        let key_good = tds_good.cell_keys().next().unwrap();
        let ratio_good = radius_ratio(&tds_good, key_good).unwrap();

        // Good quality: ratio < 4 (2D)
        assert!(ratio_good < 4.0);

        // Acceptable quality
        let vertices_acceptable = vec![
            vertex!([0.0, 0.0]),
            vertex!([3.0, 0.0]),
            vertex!([0.0, 4.0]),
        ];
        let tds_acceptable: Tds<f64, Option<()>, Option<()>, 2> =
            Tds::new(&vertices_acceptable).unwrap();
        let key_acceptable = tds_acceptable.cell_keys().next().unwrap();
        let ratio_acceptable = radius_ratio(&tds_acceptable, key_acceptable).unwrap();

        // Acceptable: ratio < 10 (2D)
        assert!(ratio_acceptable < 10.0);

        // Poor quality
        let vertices_poor = vec![
            vertex!([0.0, 0.0]),
            vertex!([10.0, 0.0]),
            vertex!([5.0, 0.1]),
        ];
        let tds_poor: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices_poor).unwrap();
        let key_poor = tds_poor.cell_keys().next().unwrap();
        let ratio_poor = radius_ratio(&tds_poor, key_poor).unwrap();

        // Poor: ratio >= 10 (2D)
        assert!(ratio_poor >= 10.0);
    }

    // =============================================================================
    // SPECIAL GEOMETRIC CONFIGURATIONS
    // =============================================================================

    #[test]
    fn test_right_simplex_properties() {
        // Right triangle (all edges orthogonal at origin)
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([1.0, 0.0]),
            vertex!([0.0, 1.0]),
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        let ratio = radius_ratio(&tds, cell_key).unwrap();
        // For isosceles right triangle: ratio ≈ 2.414 (1 + sqrt(2))
        assert_relative_eq!(ratio, 1.0 + 2.0_f64.sqrt(), epsilon = 0.1);

        let norm_vol = normalized_volume(&tds, cell_key).unwrap();
        assert!(norm_vol > 0.0);
    }

    #[test]
    fn test_isosceles_simplex_2d() {
        // Isosceles triangle
        let vertices = vec![
            vertex!([0.0, 0.0]),
            vertex!([2.0, 0.0]),
            vertex!([1.0, 2.0]), // Isosceles
        ];
        let tds: Tds<f64, Option<()>, Option<()>, 2> = Tds::new(&vertices).unwrap();
        let cell_key = tds.cell_keys().next().unwrap();

        let ratio = radius_ratio(&tds, cell_key).unwrap();
        assert!(ratio > 2.0); // Not equilateral, so ratio > 2
        assert!(ratio < 5.0); // But still reasonable quality

        let norm_vol = normalized_volume(&tds, cell_key).unwrap();
        assert!(norm_vol > 0.1);
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
}
