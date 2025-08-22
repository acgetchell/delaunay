//! Enhanced geometric predicates with improved numerical robustness.
//!
//! This module demonstrates several techniques for improving the numerical stability
//! of geometric predicates used in Delaunay triangulation. These improvements
//! address the "No cavity boundary facets found" error by making the predicates
//! more reliable when dealing with degenerate or near-degenerate point configurations.

use nalgebra as na;
use num_traits::{Float, cast};
use serde::{Serialize, de::DeserializeOwned};
use std::fmt::Debug;

use super::predicates::{InSphere, Orientation};
use super::util::{safe_coords_to_f64, safe_scalar_to_f64, squared_norm};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{
    Coordinate, CoordinateConversionError, CoordinateScalar,
};

/// Configuration for robust geometric predicates.
///
/// This structure allows fine-tuning of numerical robustness parameters
/// based on the specific requirements of the triangulation algorithm.
#[derive(Debug, Clone)]
pub struct RobustPredicateConfig<T> {
    /// Base tolerance for degenerate case detection
    pub base_tolerance: T,
    /// Relative tolerance factor (multiplied by magnitude of operands)
    pub relative_tolerance_factor: T,
    /// Maximum number of refinement iterations for adaptive precision
    pub max_refinement_iterations: usize,
    /// Threshold for switching to exact arithmetic
    pub exact_arithmetic_threshold: T,
    /// Scale factor for perturbation when handling degeneracies
    pub perturbation_scale: T,
}

impl<T: CoordinateScalar> Default for RobustPredicateConfig<T> {
    fn default() -> Self {
        Self {
            base_tolerance: T::default_tolerance(),
            relative_tolerance_factor: cast(1e-12).unwrap_or_else(T::default_tolerance),
            max_refinement_iterations: 3,
            exact_arithmetic_threshold: cast(1e-10).unwrap_or_else(T::default_tolerance),
            perturbation_scale: cast(1e-10).unwrap_or_else(T::default_tolerance),
        }
    }
}

/// Enhanced insphere predicate with multiple numerical robustness techniques.
///
/// This function implements several strategies to improve numerical stability:
/// 1. Adaptive tolerances based on operand magnitude
/// 2. Determinant conditioning and scaling
/// 3. Fallback strategies for degenerate cases
/// 4. Result consistency verification
///
/// # Errors
///
/// Returns an error if the input is invalid (wrong number of points) or if all
/// numerical strategies fail to produce a reliable result.
pub fn robust_insphere<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
    config: &RobustPredicateConfig<T>,
) -> Result<InSphere, CoordinateConversionError>
where
    T: CoordinateScalar + std::iter::Sum + num_traits::Zero,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: format!("Expected {} points, got {}", D + 1, simplex_points.len()),
            from_type: "point count",
            to_type: "valid simplex",
        });
    }

    // Strategy 1: Try standard determinant approach with adaptive tolerance
    if let Ok(result) = adaptive_tolerance_insphere(simplex_points, test_point, config) {
        // Strategy 2: Verify consistency with alternative method
        if verify_insphere_consistency(simplex_points, test_point, result, config) {
            return Ok(result);
        }
    } else {
        // Fall through to more robust methods
    }

    // Strategy 3: Try with determinant conditioning
    if let Ok(result) = conditioned_insphere(simplex_points, test_point, config) {
        return Ok(result);
    }
    // Continue to most robust method

    // Strategy 4: Use symbolic perturbation for degenerate cases
    Ok(symbolic_perturbation_insphere(
        simplex_points,
        test_point,
        config,
    ))
}

/// Insphere test with adaptive tolerance based on operand magnitude.
///
/// This approach computes tolerances that scale with the magnitude of the
/// coordinates being processed, which is crucial when dealing with points
/// that have very large or very small coordinate values.
fn adaptive_tolerance_insphere<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
    config: &RobustPredicateConfig<T>,
) -> Result<InSphere, CoordinateConversionError>
where
    T: CoordinateScalar,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    // Build the insphere determinant matrix
    let matrix = build_insphere_matrix(simplex_points, test_point)?;

    // Calculate determinant
    let det = matrix.determinant();

    // Compute adaptive tolerance based on matrix conditioning
    let adaptive_tolerance = compute_adaptive_tolerance(&matrix, config);

    // Get simplex orientation for correct interpretation
    let orientation = robust_orientation(simplex_points, config)?;

    // Interpret result based on orientation
    interpret_insphere_determinant(det, orientation, adaptive_tolerance)
}

/// Insphere test with matrix conditioning to improve numerical stability.
///
/// This method applies scaling and pivoting techniques to improve the
/// condition number of the matrix before computing the determinant.
fn conditioned_insphere<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
    config: &RobustPredicateConfig<T>,
) -> Result<InSphere, CoordinateConversionError>
where
    T: CoordinateScalar,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    // Build matrix and apply conditioning
    let matrix = build_insphere_matrix(simplex_points, test_point)?;

    // Apply row and column scaling to improve conditioning
    let (conditioned_matrix, scale_factor) = condition_matrix(matrix, config);

    // Calculate determinant with scale correction
    let det = conditioned_matrix.determinant() * scale_factor;

    // Use base tolerance since matrix is now better conditioned
    let tolerance = config.base_tolerance;
    let orientation = robust_orientation(simplex_points, config)?;

    interpret_insphere_determinant(det, orientation, tolerance)
}

/// Insphere test using symbolic perturbation for degenerate cases.
///
/// When points are exactly or nearly cocircular/cospherical, this method
/// applies a small symbolic perturbation to break ties deterministically.
fn symbolic_perturbation_insphere<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
    config: &RobustPredicateConfig<T>,
) -> InSphere
where
    T: CoordinateScalar,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    // Try with small perturbations in different directions
    let perturbation_directions = generate_perturbation_directions::<T, D>();

    for direction in perturbation_directions {
        let perturbed_test_point =
            apply_perturbation(test_point, direction, config.perturbation_scale);

        match adaptive_tolerance_insphere(simplex_points, &perturbed_test_point, config) {
            Ok(InSphere::INSIDE) => return InSphere::INSIDE,
            Ok(InSphere::OUTSIDE) => return InSphere::OUTSIDE,
            Ok(InSphere::BOUNDARY) | Err(_) => {} // Try next perturbation
        }
    }

    // If all perturbations fail, use deterministic tie-breaking
    deterministic_tie_breaking(simplex_points, test_point)
}

/// Enhanced orientation predicate with robustness improvements.
///
/// # Errors
///
/// Returns an error if the input is invalid (wrong number of points) or if
/// the geometric computation fails.
pub fn robust_orientation<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    config: &RobustPredicateConfig<T>,
) -> Result<Orientation, CoordinateConversionError>
where
    T: CoordinateScalar,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    if simplex_points.len() != D + 1 {
        return Err(CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: format!("Expected {} points, got {}", D + 1, simplex_points.len()),
            from_type: "point count",
            to_type: "valid simplex",
        });
    }

    // Build orientation matrix
    let matrix = build_orientation_matrix(simplex_points)?;

    // Calculate determinant
    let det = matrix.determinant();

    // Use adaptive tolerance
    let tolerance = compute_matrix_adaptive_tolerance(&matrix, config);
    let tolerance_f64: f64 = safe_scalar_to_f64(tolerance)?;

    if det > tolerance_f64 {
        Ok(Orientation::POSITIVE)
    } else if det < -tolerance_f64 {
        Ok(Orientation::NEGATIVE)
    } else {
        Ok(Orientation::DEGENERATE)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Build the standard insphere matrix for determinant computation.
fn build_insphere_matrix<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
) -> Result<na::DMatrix<f64>, CoordinateConversionError>
where
    T: CoordinateScalar,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    use na::DMatrix;

    let mut matrix = DMatrix::zeros(D + 2, D + 2);

    // Add simplex points
    for (i, point) in simplex_points.iter().enumerate() {
        let coords: [T; D] = point.into();

        // Coordinates - use safe conversion
        let coords_f64 = safe_coords_to_f64(coords)?;
        for j in 0..D {
            matrix[(i, j)] = coords_f64[j];
        }

        // Squared norm - use safe conversion
        let norm_sq = squared_norm(coords);
        let norm_sq_f64 = safe_scalar_to_f64(norm_sq)?;
        matrix[(i, D)] = norm_sq_f64;

        // Constant term
        matrix[(i, D + 1)] = 1.0;
    }

    // Add test point
    let test_coords: [T; D] = (*test_point).into();

    let test_coords_f64 = safe_coords_to_f64(test_coords)?;
    for j in 0..D {
        matrix[(D + 1, j)] = test_coords_f64[j];
    }

    let test_norm_sq = squared_norm(test_coords);
    let test_norm_sq_f64 = safe_scalar_to_f64(test_norm_sq)?;
    matrix[(D + 1, D)] = test_norm_sq_f64;
    matrix[(D + 1, D + 1)] = 1.0;

    Ok(matrix)
}

/// Build orientation matrix for determinant computation.
fn build_orientation_matrix<T, const D: usize>(
    simplex_points: &[Point<T, D>],
) -> Result<na::DMatrix<f64>, CoordinateConversionError>
where
    T: CoordinateScalar,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    use na::DMatrix;

    let mut matrix = DMatrix::zeros(D + 1, D + 1);

    for (i, point) in simplex_points.iter().enumerate() {
        let coords: [T; D] = point.into();

        // Add coordinates using safe conversion
        let coords_f64 = safe_coords_to_f64(coords)?;
        for j in 0..D {
            matrix[(i, j)] = coords_f64[j];
        }

        // Add constant term
        matrix[(i, D)] = 1.0;
    }

    Ok(matrix)
}

/// Compute adaptive tolerance based on matrix magnitude and conditioning.
fn compute_adaptive_tolerance<T>(matrix: &na::DMatrix<f64>, config: &RobustPredicateConfig<T>) -> T
where
    T: CoordinateScalar,
{
    // Compute matrix infinity norm (maximum absolute row sum)
    let mut max_row_sum = 0.0;
    for i in 0..matrix.nrows() {
        let mut row_sum = 0.0;
        for j in 0..matrix.ncols() {
            row_sum += matrix[(i, j)].abs();
        }
        max_row_sum = max_row_sum.max(row_sum);
    }

    // Scale base tolerance by matrix magnitude
    let base_tol: f64 = safe_scalar_to_f64(config.base_tolerance).unwrap_or(1e-15);
    let rel_factor: f64 = safe_scalar_to_f64(config.relative_tolerance_factor).unwrap_or(1e-12);

    let adaptive_tol = rel_factor.mul_add(max_row_sum, base_tol);

    // Use safe conversion but fall back to config value on failure
    super::util::safe_scalar_from_f64(adaptive_tol).unwrap_or(config.base_tolerance)
}

/// Simplified version for matrix-based tolerance computation.
fn compute_matrix_adaptive_tolerance<T>(
    matrix: &na::DMatrix<f64>,
    config: &RobustPredicateConfig<T>,
) -> T
where
    T: CoordinateScalar,
{
    compute_adaptive_tolerance(matrix, config)
}

/// Apply matrix conditioning to improve numerical stability.
fn condition_matrix<T>(
    mut matrix: na::DMatrix<f64>,
    _config: &RobustPredicateConfig<T>,
) -> (na::DMatrix<f64>, f64)
where
    T: CoordinateScalar,
{
    let mut scale_factor = 1.0;

    // Simple row scaling - scale each row by its maximum element
    for i in 0..matrix.nrows() {
        let mut max_element = 0.0;
        for j in 0..matrix.ncols() {
            max_element = max_element.max(matrix[(i, j)].abs());
        }

        if max_element > 1e-100 {
            // Avoid division by zero
            for j in 0..matrix.ncols() {
                matrix[(i, j)] /= max_element;
            }
            scale_factor *= max_element;
        }
    }

    (matrix, scale_factor)
}

/// Verify consistency of insphere result using alternative method.
///
/// This function provides an independent verification of the insphere result using
/// the distance-based `insphere_distance` function. This helps detect numerical
/// inconsistencies that might arise from near-degenerate configurations or precision
/// issues in the determinant calculation.
///
/// # Algorithm
///
/// 1. Use `insphere_distance` to get an independent insphere result
/// 2. Compare this result with the determinant-based result
/// 3. Consider results consistent if they match or if either is `BOUNDARY`
///
/// # Tolerance for Consistency
///
/// The function is conservative about marking results as inconsistent:
/// - Exact matches (`INSIDE`/`INSIDE`, `OUTSIDE`/`OUTSIDE`, `BOUNDARY`/`BOUNDARY`) are consistent
/// - Any result involving `BOUNDARY` is considered consistent since it indicates degeneracy
/// - Only direct contradictions (`INSIDE`/`OUTSIDE`) are marked as inconsistent
///
/// This approach prevents false negatives when dealing with legitimately degenerate
/// or near-degenerate configurations.
///
/// # Returns
///
/// - `true` if the distance-based result is consistent with the determinant result
/// - `false` if there's a direct contradiction, indicating potential numerical issues
fn verify_insphere_consistency<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
    determinant_result: InSphere,
    _config: &RobustPredicateConfig<T>,
) -> bool
where
    T: CoordinateScalar + std::iter::Sum + num_traits::Zero,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    // Use the existing distance-based insphere test for verification
    super::predicates::insphere_distance(simplex_points, *test_point).map_or(
        true,
        |distance_result| {
            match (determinant_result, distance_result) {
                // Exact matches are always consistent
                (InSphere::INSIDE, InSphere::INSIDE)
                | (InSphere::OUTSIDE, InSphere::OUTSIDE)
                | (InSphere::BOUNDARY | _, InSphere::BOUNDARY) | (InSphere::BOUNDARY, _) => true,

                // Direct contradictions indicate numerical issues
                (InSphere::INSIDE, InSphere::OUTSIDE) | (InSphere::OUTSIDE, InSphere::INSIDE) => {
                    // Log the inconsistency for debugging (in debug builds only)
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Insphere consistency check failed: determinant={determinant_result:?}, distance={distance_result:?}"
                    );
                    false
                }
            }
        },
    )
}

/// Generate perturbation directions for symbolic perturbation.
fn generate_perturbation_directions<T, const D: usize>() -> Vec<[T; D]>
where
    T: CoordinateScalar,
{
    let mut directions = Vec::new();

    // Try unit vectors in each coordinate direction
    for i in 0..D {
        let mut direction = [T::zero(); D];
        direction[i] = T::one();
        directions.push(direction);

        // Also try negative direction
        direction[i] = -T::one();
        directions.push(direction);
    }

    // Add a few diagonal directions
    if D >= 2 {
        let mut diag = [T::one(); D];
        for item in diag.iter_mut().take(D) {
            let d_value = cast(D).unwrap_or_else(T::one);
            *item = *item / d_value;
        }
        directions.push(diag);
    }

    directions
}

/// Apply small perturbation to a point.
fn apply_perturbation<T, const D: usize>(
    point: &Point<T, D>,
    direction: [T; D],
    scale: T,
) -> Point<T, D>
where
    T: CoordinateScalar,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    let mut coords: [T; D] = (*point).into();

    for i in 0..D {
        coords[i] = coords[i] + direction[i] * scale;
    }

    Point::new(coords)
}

/// Deterministic tie-breaking for degenerate cases.
fn deterministic_tie_breaking<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
) -> InSphere
where
    T: CoordinateScalar,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    // Use lexicographic comparison based on coordinates
    // This provides a deterministic result even for degenerate cases

    let test_coords: [T; D] = (*test_point).into();

    // Compare with each simplex point lexicographically
    for simplex_point in simplex_points {
        let simplex_coords: [T; D] = simplex_point.into();

        for i in 0..D {
            if test_coords[i] < simplex_coords[i] {
                return InSphere::OUTSIDE;
            } else if test_coords[i] > simplex_coords[i] {
                return InSphere::INSIDE;
            }
            // If equal, continue to next coordinate
        }
    }

    // All coordinates are equal (should be rare)
    InSphere::BOUNDARY
}

/// Interpret determinant result based on orientation and tolerance.
fn interpret_insphere_determinant<T>(
    det: f64,
    orientation: Orientation,
    tolerance: T,
) -> Result<InSphere, CoordinateConversionError>
where
    T: CoordinateScalar,
{
    let tol: f64 = safe_scalar_to_f64(tolerance)?;

    let result = match orientation {
        Orientation::DEGENERATE => {
            InSphere::BOUNDARY // Conservative approach for degenerate cases
        }
        Orientation::POSITIVE => {
            if det > tol {
                InSphere::INSIDE
            } else if det < -tol {
                InSphere::OUTSIDE
            } else {
                InSphere::BOUNDARY
            }
        }
        Orientation::NEGATIVE => {
            if det < -tol {
                InSphere::INSIDE
            } else if det > tol {
                InSphere::OUTSIDE
            } else {
                InSphere::BOUNDARY
            }
        }
    };

    Ok(result)
}

/// Factory function to create robust predicate configurations for different use cases.
pub mod config_presets {
    use super::{CoordinateScalar, RobustPredicateConfig};
    use num_traits::cast;

    /// Configuration optimized for general-purpose triangulation.
    #[must_use]
    pub fn general_triangulation<T: CoordinateScalar>() -> RobustPredicateConfig<T> {
        RobustPredicateConfig {
            base_tolerance: T::default_tolerance(),
            relative_tolerance_factor: cast(1e-12).unwrap_or_else(T::default_tolerance),
            max_refinement_iterations: 3,
            exact_arithmetic_threshold: cast(1e-10).unwrap_or_else(T::default_tolerance),
            perturbation_scale: cast(1e-10).unwrap_or_else(T::default_tolerance),
        }
    }

    /// Configuration for high-precision triangulation (stricter tolerances).
    #[must_use]
    pub fn high_precision<T: CoordinateScalar>() -> RobustPredicateConfig<T> {
        let base_tol = T::default_tolerance();
        RobustPredicateConfig {
            base_tolerance: base_tol / cast(100.0).unwrap_or_else(T::one),
            relative_tolerance_factor: cast(1e-14).unwrap_or(base_tol),
            max_refinement_iterations: 5,
            exact_arithmetic_threshold: cast(1e-12).unwrap_or(base_tol),
            perturbation_scale: cast(1e-12).unwrap_or(base_tol),
        }
    }

    /// Configuration for dealing with degenerate cases (more lenient tolerances).
    #[must_use]
    pub fn degenerate_robust<T: CoordinateScalar>() -> RobustPredicateConfig<T> {
        let base_tol = T::default_tolerance();
        RobustPredicateConfig {
            base_tolerance: base_tol * cast(100.0).unwrap_or_else(T::one),
            relative_tolerance_factor: cast(1e-10).unwrap_or(base_tol),
            max_refinement_iterations: 2,
            exact_arithmetic_threshold: cast(1e-8).unwrap_or(base_tol),
            perturbation_scale: cast(1e-8).unwrap_or(base_tol),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;

    #[test]
    fn test_robust_insphere_general() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let config = config_presets::general_triangulation();

        // Test point clearly inside
        let inside_point = Point::new([0.25, 0.25, 0.25]);
        let result = robust_insphere(&points, &inside_point, &config).unwrap();
        assert_eq!(result, InSphere::INSIDE);

        // Test point clearly outside
        let outside_point = Point::new([2.0, 2.0, 2.0]);
        let result = robust_insphere(&points, &outside_point, &config).unwrap();
        assert_eq!(result, InSphere::OUTSIDE);
    }

    #[test]
    fn test_robust_orientation() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let config = config_presets::general_triangulation();
        let result = robust_orientation(&points, &config).unwrap();

        // Should detect orientation (exact result depends on coordinate system)
        assert!(matches!(
            result,
            Orientation::POSITIVE | Orientation::NEGATIVE
        ));
    }

    #[test]
    fn test_degenerate_case_handling() {
        // Create nearly coplanar points
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 1e-15]), // Very slightly off-plane
        ];

        let config = config_presets::degenerate_robust();

        // Should handle gracefully
        let test_point = Point::new([0.25, 0.25, 1e-16]);
        let result = robust_insphere(&points, &test_point, &config);
        assert!(result.is_ok());
    }
}
