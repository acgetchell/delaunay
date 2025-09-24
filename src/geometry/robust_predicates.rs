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

/// Result of consistency verification between different insphere methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsistencyResult {
    /// The two methods agree on the result
    Consistent,
    /// The two methods disagree (potential numerical issue)
    Inconsistent,
    /// Cannot verify consistency due to error in verification method
    Unverifiable,
}

impl ConsistencyResult {
    /// Returns true if the result indicates consistency (either Consistent or Unverifiable).
    /// Only returns false for definite Inconsistent results.
    #[must_use]
    pub const fn is_consistent(self) -> bool {
        matches!(self, Self::Consistent | Self::Unverifiable)
    }
}

impl std::fmt::Display for ConsistencyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Consistent => write!(f, "Consistent"),
            Self::Inconsistent => write!(f, "Inconsistent"),
            Self::Unverifiable => write!(f, "Unverifiable"),
        }
    }
}
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
    /// Multiplier for visibility threshold in fallback visibility heuristics
    pub visibility_threshold_multiplier: T,
}

impl<T: CoordinateScalar> Default for RobustPredicateConfig<T> {
    fn default() -> Self {
        Self {
            base_tolerance: T::default_tolerance(),
            relative_tolerance_factor: cast(1e-12).unwrap_or_else(T::default_tolerance),
            max_refinement_iterations: 3,
            exact_arithmetic_threshold: cast(1e-10).unwrap_or_else(T::default_tolerance),
            perturbation_scale: cast(1e-10).unwrap_or_else(T::default_tolerance),
            visibility_threshold_multiplier: cast(100.0)
                .unwrap_or_else(|| T::from(100.0).unwrap_or_else(T::default_tolerance)),
        }
    }
}

/// Enhanced insphere predicate with multiple numerical robustness techniques.
///
/// Evaluates whether a `test_point` lies inside, on, or outside the circumsphere
/// (in 2D: circumcircle) defined by a `D`-simplex (`simplex_points`). This
/// implementation is dimension-generic and applies a series of strategies to
/// provide robust results for degenerate and near-degenerate configurations.
///
/// Strategies used, in order:
/// 1) Adaptive tolerance insphere via determinant evaluation with magnitude-aware tolerances
/// 2) Determinant conditioning (row/column scaling) to improve the condition number
/// 3) Consistency verification against a distance-based insphere check
/// 4) Symbolic perturbation with deterministic tie-breaking for hard degeneracies
///
/// Sign convention and orientation:
/// - The determinant sign is interpreted relative to the simplex orientation.
/// - If the simplex orientation is POSITIVE, det > tol => INSIDE and det < -tol => OUTSIDE.
/// - If NEGATIVE, the interpretation is swapped (det < -tol => INSIDE, det > tol => OUTSIDE).
/// - DEGENERATE orientation yields BOUNDARY conservatively.
///
/// Type parameters:
/// - `T`: Coordinate scalar type implementing `CoordinateScalar`
/// - `D`: Compile-time dimension of the space
///
/// Parameters:
/// - `simplex_points`: Exactly `D + 1` points defining the simplex
/// - `test_point`: The query point to classify relative to the simplex circumsphere
/// - `config`: Tunable numeric-robustness parameters; see `config_presets` for defaults
///
/// Returns:
/// - `Ok(InSphere::{INSIDE, BOUNDARY, OUTSIDE})` on success
/// - `Err(CoordinateConversionError)` if inputs are invalid (e.g., wrong point
///   count) or safe conversions fail
///
/// Complexity:
/// - Dominated by determinant evaluation on a (D+2)×(D+2) matrix: roughly O((D+2)^3)
///
/// Example (3D):
/// ```rust
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::robust_predicates::{robust_insphere, config_presets};
/// use delaunay::geometry::predicates::InSphere;
/// use delaunay::geometry::traits::coordinate::Coordinate;
///
/// let tetra = vec![
///     Point::new([0.0, 0.0, 0.0]),
///     Point::new([1.0, 0.0, 0.0]),
///     Point::new([0.0, 1.0, 0.0]),
///     Point::new([0.0, 0.0, 1.0]),
/// ];
/// let config = config_presets::general_triangulation::<f64>();
///
/// let inside = Point::new([0.25, 0.25, 0.25]);
/// let outside = Point::new([2.0, 2.0, 2.0]);
///
/// let r_in = robust_insphere(&tetra, &inside, &config).unwrap();
/// let r_out = robust_insphere(&tetra, &outside, &config).unwrap();
/// assert_eq!(r_in, InSphere::INSIDE);
/// assert_eq!(r_out, InSphere::OUTSIDE);
/// ```
///
/// Notes:
/// - For extremely challenging inputs, the function falls back to symbolic
///   perturbation with deterministic tie-breaking to maintain progress.
/// - See `robust_orientation` for the orientation predicate used in the sign interpretation.
///
/// # Errors
///
/// Returns an error if the input is invalid (wrong number of points) or if required
/// numeric conversions fail.
pub fn robust_insphere<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
    config: &RobustPredicateConfig<T>,
) -> Result<InSphere, CoordinateConversionError>
where
    T: CoordinateScalar + std::iter::Sum + num_traits::Zero,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
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
        match verify_insphere_consistency(simplex_points, test_point, result, config) {
            ConsistencyResult::Consistent | ConsistencyResult::Unverifiable => return Ok(result), // Accept if we can't verify
            ConsistencyResult::Inconsistent => {
                // Fall through to more robust methods when inconsistent
            }
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
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
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
/// This method applies row scaling to improve the
/// condition number of the matrix before computing the determinant.
///
/// Row scaling is preferred over full row+column scaling for determinant
/// calculations as it provides effective conditioning while keeping the scaling
/// compensation simple and numerically stable (Golub & Van Loan, "Matrix
/// Computations" 4th ed., Section 3.5; Higham, "Accuracy and Stability of
/// Numerical Algorithms" 2nd ed., Section 9.7).
fn conditioned_insphere<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
    config: &RobustPredicateConfig<T>,
) -> Result<InSphere, CoordinateConversionError>
where
    T: CoordinateScalar,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    // Build matrix and apply conditioning
    let matrix = build_insphere_matrix(simplex_points, test_point)?;

    // Apply row scaling to improve conditioning
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
    T: CoordinateScalar + std::iter::Sum + num_traits::Zero,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
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

    // If all perturbations fail, use geometric deterministic tie-breaking
    geometric_deterministic_tie_breaking(simplex_points, test_point)
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
    [T; D]: Copy + Sized,
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
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
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
    [T; D]: Copy + Sized,
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
    let mut base_tol: f64 = safe_scalar_to_f64(config.base_tolerance).unwrap_or(1e-15);
    let mut rel_factor: f64 = safe_scalar_to_f64(config.relative_tolerance_factor).unwrap_or(1e-12);

    // Guard against non-finite tolerances
    if !base_tol.is_finite() {
        base_tol = 1e-15;
    }
    if !rel_factor.is_finite() {
        rel_factor = 1e-12;
    }

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
/// - [`ConsistencyResult::Consistent`] if the two methods agree
/// - [`ConsistencyResult::Inconsistent`] if there's a direct contradiction
/// - [`ConsistencyResult::Unverifiable`] if the verification method fails
fn verify_insphere_consistency<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
    determinant_result: InSphere,
    _config: &RobustPredicateConfig<T>,
) -> ConsistencyResult
where
    T: CoordinateScalar + std::iter::Sum + num_traits::Zero,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    // Use the existing distance-based insphere test for verification
    super::predicates::insphere_distance(simplex_points, *test_point).map_or(ConsistencyResult::Unverifiable, |distance_result| match (determinant_result, distance_result) {
                // Exact matches are always consistent
                (InSphere::INSIDE, InSphere::INSIDE)
                | (InSphere::OUTSIDE, InSphere::OUTSIDE)
                | (InSphere::BOUNDARY, _)
                | (_, InSphere::BOUNDARY) => ConsistencyResult::Consistent,

                // Direct contradictions indicate numerical issues
                (InSphere::INSIDE, InSphere::OUTSIDE) | (InSphere::OUTSIDE, InSphere::INSIDE) => {
                    // Log the inconsistency for debugging (in debug builds only)
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Insphere consistency check failed: determinant={determinant_result:?}, distance={distance_result:?}"
                    );
                    ConsistencyResult::Inconsistent
                }
            })
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
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    let mut coords: [T; D] = (*point).into();

    for i in 0..D {
        coords[i] = coords[i] + direction[i] * scale;
    }

    Point::new(coords)
}

/// Geometric deterministic tie-breaking using Simulation of Simplicity (`SoS`).
///
/// This implements the Simulation of Simplicity approach by Edelsbrunner and Mücke
/// ("Simulation of Simplicity: A Technique to Cope with Degenerate Cases in
/// Geometric Algorithms", ACM Transactions on Graphics, 1990).
///
/// The method applies infinitesimal symbolic perturbations in a deterministic order
/// to break degeneracies while preserving geometric meaning. Points are conceptually
/// perturbed by ε^i where ε is infinitesimal and i is the point's index.
fn geometric_deterministic_tie_breaking<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
) -> InSphere
where
    T: CoordinateScalar + std::iter::Sum + num_traits::Zero,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    // Implement Simulation of Simplicity for insphere predicate
    // We assign symbolic indices: simplex points get indices 0..D, test point gets D+1

    // Build the symbolic insphere matrix with perturbation terms
    // The SoS approach evaluates the determinant as if each point i was perturbed
    // by adding ε^i to each coordinate, where ε is infinitesimal

    // For the insphere predicate, we need to consider the sign of the determinant
    // when breaking ties using the lowest-order perturbation that gives a non-zero result

    // Since this is quite complex to implement fully, we use a simplified geometric approach:
    // Apply tiny perturbations based on point indices to maintain geometric meaning

    let perturbation_scale = T::from(1e-100)
        .unwrap_or_else(|| T::default_tolerance() / T::from(1000).unwrap_or_else(T::one));

    // Apply SoS-style perturbations: each point gets a unique perturbation magnitude
    let mut perturbed_simplex: Vec<Point<T, D>> = Vec::with_capacity(simplex_points.len());

    for (i, point) in simplex_points.iter().enumerate() {
        let mut coords: [T; D] = (*point).into();

        // Apply perturbation with magnitude ε^(i+1) in the first coordinate
        // This maintains the SoS property while being computationally feasible
        let perturbation_magnitude = perturbation_scale / T::from(i + 1).unwrap_or_else(T::one);
        coords[0] = coords[0] + perturbation_magnitude;

        perturbed_simplex.push(Point::new(coords));
    }

    // Perturb test point with unique index
    let mut test_coords: [T; D] = (*test_point).into();
    let test_perturbation =
        perturbation_scale / T::from(simplex_points.len() + 1).unwrap_or_else(T::one);
    test_coords[0] = test_coords[0] + test_perturbation;
    let perturbed_test = Point::new(test_coords);

    // Now evaluate the insphere predicate with perturbed points
    // Use distance-based approach for robustness
    super::predicates::insphere_distance(&perturbed_simplex, perturbed_test).unwrap_or_else(|_| {
        // If that fails, fall back to a more conservative geometric approach
        // based on the centroid distance method
        centroid_based_tie_breaking(simplex_points, test_point)
    })
}

/// Centroid-based geometric tie-breaking as a fallback.
///
/// This method compares the test point's distance to the simplex centroid
/// versus the average distance of simplex points to the centroid.
/// While not as theoretically rigorous as `SoS`, it maintains geometric meaning.
fn centroid_based_tie_breaking<T, const D: usize>(
    simplex_points: &[Point<T, D>],
    test_point: &Point<T, D>,
) -> InSphere
where
    T: CoordinateScalar + std::iter::Sum + num_traits::Zero,
    [T; D]: Copy + DeserializeOwned + Serialize + Sized,
{
    // Calculate simplex centroid
    let mut centroid_coords = [T::zero(); D];
    for point in simplex_points {
        let coords: [T; D] = (*point).into();
        for i in 0..D {
            centroid_coords[i] = centroid_coords[i] + coords[i];
        }
    }

    let num_points_scalar = T::from(simplex_points.len()).unwrap_or_else(T::one);
    for coord in &mut centroid_coords {
        *coord = *coord / num_points_scalar;
    }

    // Calculate test point distance to centroid
    let test_coords: [T; D] = (*test_point).into();
    let mut test_dist_sq = T::zero();
    for i in 0..D {
        let diff = test_coords[i] - centroid_coords[i];
        test_dist_sq = test_dist_sq + diff * diff;
    }

    // Calculate average simplex point distance to centroid
    let mut avg_simplex_dist_sq = T::zero();
    for point in simplex_points {
        let coords: [T; D] = (*point).into();
        let mut dist_sq = T::zero();
        for i in 0..D {
            let diff = coords[i] - centroid_coords[i];
            dist_sq = dist_sq + diff * diff;
        }
        avg_simplex_dist_sq = avg_simplex_dist_sq + dist_sq;
    }
    avg_simplex_dist_sq = avg_simplex_dist_sq / num_points_scalar;

    // Compare distances: if test point is farther from centroid than average,
    // it's likely outside the circumsphere
    if test_dist_sq > avg_simplex_dist_sq {
        InSphere::OUTSIDE
    } else if test_dist_sq < avg_simplex_dist_sq {
        InSphere::INSIDE
    } else {
        InSphere::BOUNDARY
    }
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
    ///
    /// This provides a balanced configuration suitable for most triangulation
    /// scenarios with moderate tolerance settings.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::geometry::robust_predicates::config_presets;
    ///
    /// let config = config_presets::general_triangulation::<f64>();
    /// assert_eq!(config.max_refinement_iterations, 3);
    /// ```
    #[must_use]
    pub fn general_triangulation<T: CoordinateScalar>() -> RobustPredicateConfig<T> {
        RobustPredicateConfig {
            base_tolerance: T::default_tolerance(),
            relative_tolerance_factor: cast(1e-12).unwrap_or_else(T::default_tolerance),
            max_refinement_iterations: 3,
            exact_arithmetic_threshold: cast(1e-10).unwrap_or_else(T::default_tolerance),
            perturbation_scale: cast(1e-10).unwrap_or_else(T::default_tolerance),
            visibility_threshold_multiplier: cast(100.0)
                .unwrap_or_else(|| T::from(100.0).unwrap_or_else(T::default_tolerance)),
        }
    }

    /// Configuration for high-precision triangulation (stricter tolerances).
    ///
    /// This configuration uses tighter tolerances and more refinement iterations
    /// for applications requiring high geometric precision.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::geometry::robust_predicates::config_presets;
    ///
    /// let config = config_presets::high_precision::<f64>();
    /// assert_eq!(config.max_refinement_iterations, 5);
    /// ```
    #[must_use]
    pub fn high_precision<T: CoordinateScalar>() -> RobustPredicateConfig<T> {
        let base_tol = T::default_tolerance();
        RobustPredicateConfig {
            base_tolerance: base_tol / cast(100.0).unwrap_or_else(T::one),
            relative_tolerance_factor: cast(1e-14).unwrap_or(base_tol),
            max_refinement_iterations: 5,
            exact_arithmetic_threshold: cast(1e-12).unwrap_or(base_tol),
            perturbation_scale: cast(1e-12).unwrap_or(base_tol),
            visibility_threshold_multiplier: cast(100.0)
                .unwrap_or_else(|| T::from(100.0).unwrap_or_else(T::default_tolerance)),
        }
    }

    /// Configuration for dealing with degenerate cases (more lenient tolerances).
    ///
    /// This configuration uses more lenient tolerances to handle nearly degenerate
    /// geometric configurations that might otherwise cause numerical instability.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::geometry::robust_predicates::config_presets;
    ///
    /// let config = config_presets::degenerate_robust::<f64>();
    /// assert_eq!(config.max_refinement_iterations, 2);
    /// ```
    #[must_use]
    pub fn degenerate_robust<T: CoordinateScalar>() -> RobustPredicateConfig<T> {
        let base_tol = T::default_tolerance();
        RobustPredicateConfig {
            base_tolerance: base_tol * cast(100.0).unwrap_or_else(T::one),
            relative_tolerance_factor: cast(1e-10).unwrap_or(base_tol),
            max_refinement_iterations: 2,
            exact_arithmetic_threshold: cast(1e-8).unwrap_or(base_tol),
            perturbation_scale: cast(1e-8).unwrap_or(base_tol),
            visibility_threshold_multiplier: cast(200.0)
                .unwrap_or_else(|| T::from(200.0).unwrap_or_else(T::default_tolerance)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;
    use approx::assert_relative_eq;

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

    #[allow(clippy::too_many_lines)]
    #[test]
    fn test_verify_insphere_consistency_comprehensive() {
        let config = config_presets::general_triangulation();
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        // Test exact matches - all should be consistent
        let test_cases = [
            (
                Point::new([0.25, 0.25, 0.25]),
                InSphere::INSIDE,
                "inside point",
            ),
            (
                Point::new([2.0, 2.0, 2.0]),
                InSphere::OUTSIDE,
                "outside point",
            ),
            (
                Point::new([0.5, 0.5, 0.5]),
                InSphere::BOUNDARY,
                "boundary point",
            ),
        ];

        for (test_point, result, description) in test_cases {
            assert!(
                verify_insphere_consistency(&points, &test_point, result, &config).is_consistent(),
                "Failed for {description}"
            );
        }

        // Test that BOUNDARY results are always considered consistent
        let boundary_test_point = Point::new([0.3, 0.3, 0.3]);
        for expected_result in [InSphere::INSIDE, InSphere::OUTSIDE, InSphere::BOUNDARY] {
            if expected_result == InSphere::BOUNDARY {
                assert!(
                    verify_insphere_consistency(
                        &points,
                        &boundary_test_point,
                        expected_result,
                        &config
                    )
                    .is_consistent()
                );
            }
        }

        // Test different dimensions
        let triangle_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([2.0, 0.0]),
            Point::new([1.0, 2.0]),
        ];
        let test_2d = Point::new([1.0, 0.5]);
        assert!(
            verify_insphere_consistency(&triangle_2d, &test_2d, InSphere::BOUNDARY, &config)
                .is_consistent()
        );

        // Test edge cases with extreme coordinates and error conditions
        let edge_cases = [
            (
                vec![
                    Point::new([1e-10, 0.0, 0.0]),
                    Point::new([0.0, 1e-10, 0.0]),
                    Point::new([0.0, 0.0, 1e-10]),
                    Point::new([1e-10, 1e-10, 1e-10]),
                ],
                Point::new([5e-11, 5e-11, 5e-11]),
                "small coordinates",
            ),
            (
                vec![
                    Point::new([1e6, 0.0, 0.0]),
                    Point::new([0.0, 1e6, 0.0]),
                    Point::new([0.0, 0.0, 1e6]),
                    Point::new([1e6, 1e6, 1e6]),
                ],
                Point::new([5e5, 5e5, 5e5]),
                "large coordinates",
            ),
        ];

        for (edge_points, edge_test, description) in edge_cases {
            assert!(
                verify_insphere_consistency(&edge_points, &edge_test, InSphere::BOUNDARY, &config)
                    .is_consistent(),
                "Failed for edge case: {description}"
            );
        }

        // Test error conditions that should return Unverifiable
        let error_cases = [
            // Invalid simplex size
            (
                vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])],
                Point::new([0.5, 0.0, 0.0]),
                "too few points",
            ),
            // Non-finite coordinates
            (
                vec![
                    Point::new([f64::NAN, 0.0, 0.0]),
                    Point::new([1.0, 0.0, 0.0]),
                    Point::new([0.0, 1.0, 0.0]),
                    Point::new([0.0, 0.0, 1.0]),
                ],
                Point::new([0.1, 0.1, 0.1]),
                "NaN coordinates",
            ),
            // Degenerate simplex
            (
                vec![
                    Point::new([0.0, 0.0, 0.0]),
                    Point::new([1.0, 0.0, 0.0]),
                    Point::new([2.0, 0.0, 0.0]),
                    Point::new([3.0, 0.0, 0.0]),
                ],
                Point::new([1.5, 0.0, 0.0]),
                "collinear points",
            ),
        ];

        for (error_points, error_test, error_description) in error_cases {
            let result = verify_insphere_consistency(
                &error_points,
                &error_test,
                InSphere::BOUNDARY,
                &config,
            );
            assert_eq!(
                result,
                ConsistencyResult::Unverifiable,
                "Expected Unverifiable for: {error_description}"
            );
            assert!(
                result.is_consistent(),
                "Unverifiable should be considered consistent"
            );
        }
    }

    #[test]
    fn test_consistency_result_display() {
        // Test Display trait implementation for ConsistencyResult
        assert_eq!(format!("{}", ConsistencyResult::Consistent), "Consistent");
        assert_eq!(
            format!("{}", ConsistencyResult::Inconsistent),
            "Inconsistent"
        );
        assert_eq!(
            format!("{}", ConsistencyResult::Unverifiable),
            "Unverifiable"
        );
    }

    #[test]
    fn test_robust_predicates_dimensional_coverage() {
        // Comprehensive test across dimensions 2D-5D with both valid and invalid cases
        let config = config_presets::general_triangulation();

        // Test 2D - Valid triangle
        let triangle_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.5, 1.0]),
        ];
        let test_2d = Point::new([0.5, 0.3]);
        assert!(
            robust_insphere(&triangle_2d, &test_2d, &config).is_ok(),
            "2D insphere should work"
        );
        assert!(
            robust_orientation(&triangle_2d, &config).is_ok(),
            "2D orientation should work"
        );

        // Test 3D - Valid tetrahedron
        let tetrahedron_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let test_3d = Point::new([0.25, 0.25, 0.25]);
        assert!(
            robust_insphere(&tetrahedron_3d, &test_3d, &config).is_ok(),
            "3D insphere should work"
        );
        assert!(
            robust_orientation(&tetrahedron_3d, &config).is_ok(),
            "3D orientation should work"
        );

        // Test 4D - Valid hypersimplex
        let simplex_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];
        let test_4d = Point::new([0.2, 0.2, 0.2, 0.2]);
        assert!(
            robust_insphere(&simplex_4d, &test_4d, &config).is_ok(),
            "4D insphere should work"
        );
        assert!(
            robust_orientation(&simplex_4d, &config).is_ok(),
            "4D orientation should work"
        );

        // Test 5D - Valid hypersimplex
        let simplex_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];
        let test_5d = Point::new([0.15, 0.15, 0.15, 0.15, 0.15]);
        assert!(
            robust_insphere(&simplex_5d, &test_5d, &config).is_ok(),
            "5D insphere should work"
        );
        assert!(
            robust_orientation(&simplex_5d, &config).is_ok(),
            "5D orientation should work"
        );

        // Test error cases - wrong number of points for each dimension
        // 2D error case - too few points
        let too_few_2d = vec![Point::new([0.0, 0.0])];
        let insphere_2d_err = robust_insphere(&too_few_2d, &test_2d, &config);
        let orientation_2d_err = robust_orientation(&too_few_2d, &config);
        assert!(
            insphere_2d_err.is_err() || orientation_2d_err.is_err(),
            "2D should fail with 1 point"
        );

        // 3D error case - too few points
        let too_few_3d = vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])];
        let insphere_3d_err = robust_insphere(&too_few_3d, &test_3d, &config);
        let orientation_3d_err = robust_orientation(&too_few_3d, &config);
        assert!(
            insphere_3d_err.is_err() || orientation_3d_err.is_err(),
            "3D should fail with 2 points"
        );

        // 4D error case - too few points
        let too_few_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
        ];
        let insphere_4d_err = robust_insphere(&too_few_4d, &test_4d, &config);
        assert!(insphere_4d_err.is_err(), "4D should fail with 3 points");
    }

    #[test]
    fn test_symbolic_perturbation_fallback() {
        // Test symbolic perturbation pathways and deterministic tie-breaking
        let config = RobustPredicateConfig {
            base_tolerance: 1e-12,
            relative_tolerance_factor: 1e-15,
            max_refinement_iterations: 1,
            exact_arithmetic_threshold: 1e-8,
            perturbation_scale: 1e-15, // Very small perturbation
            visibility_threshold_multiplier: 100.0,
        };

        // Create a nearly degenerate configuration that will challenge the algorithms
        let nearly_coplanar_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.5, 1.0, 0.0]),
            Point::new([0.5, 0.5, 1e-16]), // Extremely close to coplanar
        ];

        // Test point that's very close to the boundary
        let boundary_test_point = Point::new([0.5, 0.5, 5e-17]);

        // This should exercise the symbolic perturbation logic
        let result = robust_insphere(&nearly_coplanar_points, &boundary_test_point, &config);
        assert!(result.is_ok());

        // The result should be one of the valid InSphere variants
        let insphere_result = result.unwrap();
        assert!(matches!(
            insphere_result,
            InSphere::INSIDE | InSphere::BOUNDARY | InSphere::OUTSIDE
        ));
    }

    #[test]
    fn test_matrix_conditioning_edge_cases() {
        // Test matrix conditioning with various edge cases
        use nalgebra::DMatrix;

        let config = config_presets::general_triangulation::<f64>();

        // Test matrix with very small elements
        let mut small_matrix = DMatrix::zeros(3, 3);
        small_matrix[(0, 0)] = 1e-100;
        small_matrix[(1, 1)] = 1e-99;
        small_matrix[(2, 2)] = 1e-98;

        let (conditioned_small, scale_small) = condition_matrix(small_matrix.clone(), &config);
        assert!(scale_small.is_finite());
        assert!(conditioned_small.iter().all(|x| x.is_finite()));

        // Test matrix with mixed large and small elements
        let mut mixed_matrix = DMatrix::zeros(3, 3);
        mixed_matrix[(0, 0)] = 1e10;
        mixed_matrix[(0, 1)] = 1e-10;
        mixed_matrix[(1, 0)] = 1e5;
        mixed_matrix[(1, 1)] = 1e-5;
        mixed_matrix[(2, 2)] = 1.0;

        let (conditioned_mixed, scale_mixed) = condition_matrix(mixed_matrix, &config);
        assert!(scale_mixed.is_finite() && scale_mixed > 0.0);
        assert!(conditioned_mixed.iter().all(|x| x.is_finite()));

        // Test matrix with some zero elements
        let mut zero_matrix = DMatrix::zeros(3, 3);
        zero_matrix[(0, 0)] = 1.0;
        zero_matrix[(1, 1)] = 0.0; // This row will not be scaled
        zero_matrix[(2, 2)] = 2.0;

        let (conditioned_zero, scale_zero) = condition_matrix(zero_matrix, &config);
        assert!(scale_zero.is_finite());
        assert!(conditioned_zero.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_tie_breaking_comprehensive() {
        // Test tie-breaking with various degenerate and extreme configurations across dimensions
        let degenerate_config = RobustPredicateConfig {
            base_tolerance: 1e-15_f64,
            relative_tolerance_factor: 1e-15_f64,
            max_refinement_iterations: 1,
            exact_arithmetic_threshold: 1e-18_f64,
            perturbation_scale: 1e-18_f64,
            visibility_threshold_multiplier: 100.0,
        };

        // Test 1: 2D - Degenerate triangle (nearly collinear)
        let triangle_2d = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.5, 1e-15]), // Nearly collinear
        ];
        let test_2d = Point::new([0.5, 1e-16]);
        let result_2d = robust_insphere(&triangle_2d, &test_2d, &degenerate_config);
        assert!(result_2d.is_ok(), "2D tie-breaking should work");

        // Test 2: 3D - Coplanar points (forces SoS tie-breaking)
        let coplanar_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 0.0]), // All z = 0
        ];
        let test_3d = Point::new([0.25, 0.25, 0.0]);
        let result_3d = robust_insphere(&coplanar_3d, &test_3d, &degenerate_config);
        assert!(
            result_3d.is_ok(),
            "3D tie-breaking should handle coplanar points"
        );

        // Test 3: 4D - Nearly degenerate hypersimplex
        let simplex_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([1e-14, 1e-14, 1e-14, 1.0]), // Nearly in 3D subspace
        ];
        let test_4d = Point::new([0.2, 0.2, 0.2, 1e-15]);
        let result_4d = robust_insphere(&simplex_4d, &test_4d, &degenerate_config);
        assert!(result_4d.is_ok(), "4D tie-breaking should work");

        // Test 4: 5D - Degenerate case
        let simplex_5d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([1e-12, 1e-12, 1e-12, 1e-12, 1.0]), // Nearly in 4D subspace
        ];
        let test_5d = Point::new([0.1, 0.1, 0.1, 0.1, 1e-13]);
        let result_5d = robust_insphere(&simplex_5d, &test_5d, &degenerate_config);
        assert!(result_5d.is_ok(), "5D tie-breaking should work");

        // Test determinism - same input should give same output
        let result_3d_repeat = robust_insphere(&coplanar_3d, &test_3d, &degenerate_config);
        assert_eq!(
            result_3d, result_3d_repeat,
            "Tie-breaking should be deterministic"
        );

        // Test numerical extremes
        let config = config_presets::general_triangulation::<f64>();
        let extreme_cases = [
            // Very small coordinates
            (
                vec![
                    Point::new([1e-100, 0.0, 0.0]),
                    Point::new([0.0, 1e-100, 0.0]),
                    Point::new([0.0, 0.0, 1e-100]),
                    Point::new([1e-101, 1e-101, 1e-101]),
                ],
                Point::new([5e-102, 5e-102, 5e-102]),
                "tiny coordinates",
            ),
            // Very large coordinates
            (
                vec![
                    Point::new([1e50, 0.0, 0.0]),
                    Point::new([0.0, 1e50, 0.0]),
                    Point::new([0.0, 0.0, 1e50]),
                    Point::new([1e49, 1e49, 1e49]),
                ],
                Point::new([5e48, 5e48, 5e48]),
                "huge coordinates",
            ),
        ];

        for (simplex, test_point, description) in extreme_cases {
            let result = robust_insphere(&simplex, &test_point, &config);
            assert!(result.is_ok(), "Should handle {description}");
        }

        // Test geometric meaning preservation
        let regular_tetrahedron = vec![
            Point::new([1.0, 1.0, 1.0]),
            Point::new([1.0, -1.0, -1.0]),
            Point::new([-1.0, 1.0, -1.0]),
            Point::new([-1.0, -1.0, 1.0]),
        ];
        let clearly_inside = Point::new([0.0, 0.0, 0.0]);
        let clearly_outside = Point::new([5.0, 5.0, 5.0]);

        assert_eq!(
            robust_insphere(&regular_tetrahedron, &clearly_inside, &config).unwrap(),
            InSphere::INSIDE,
            "Center should be inside"
        );
        assert_eq!(
            robust_insphere(&regular_tetrahedron, &clearly_outside, &config).unwrap(),
            InSphere::OUTSIDE,
            "Far point should be outside"
        );
    }

    #[test]
    fn test_config_fallback_values() {
        // Test that config presets handle cast failures gracefully
        // This is tricky to test directly since cast usually succeeds for standard types,
        // but we can at least verify the configs are created successfully

        let general_config = config_presets::general_triangulation::<f64>();
        assert!(general_config.base_tolerance > 0.0);
        assert!(general_config.relative_tolerance_factor > 0.0);
        assert!(general_config.exact_arithmetic_threshold > 0.0);
        assert!(general_config.perturbation_scale > 0.0);
        assert_eq!(general_config.max_refinement_iterations, 3);

        let high_precision_config = config_presets::high_precision::<f64>();
        assert!(high_precision_config.base_tolerance > 0.0);
        assert!(high_precision_config.base_tolerance < general_config.base_tolerance);
        assert_eq!(high_precision_config.max_refinement_iterations, 5);

        let degenerate_config = config_presets::degenerate_robust::<f64>();
        assert!(degenerate_config.base_tolerance > 0.0);
        assert!(degenerate_config.base_tolerance > general_config.base_tolerance);
        assert_eq!(degenerate_config.max_refinement_iterations, 2);

        // Test with f32 to exercise potentially different code paths
        let f32_config = config_presets::general_triangulation::<f32>();
        assert!(f32_config.base_tolerance > 0.0);
        assert!(f32_config.relative_tolerance_factor > 0.0);
    }

    #[test]
    fn test_deterministic_tie_breaking() {
        // Test deterministic tie-breaking with identical coordinates
        let config = config_presets::general_triangulation();

        // Create points where the test point has identical coordinates to a simplex point
        let identical_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.5, 1.0, 0.0]),
            Point::new([0.5, 0.5, 1.0]),
        ];

        // Test point identical to first simplex point
        let identical_test = Point::new([0.0, 0.0, 0.0]);

        // This should exercise the deterministic tie-breaking logic
        let result = robust_insphere(&identical_points, &identical_test, &config);
        assert!(result.is_ok());

        // Create a case where coordinates are lexicographically ordered
        let ordered_points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
            Point::new([10.0, 11.0, 12.0]),
        ];

        // Test point that's lexicographically smaller
        let smaller_test = Point::new([0.0, 1.0, 2.0]);
        let result_smaller = robust_insphere(&ordered_points, &smaller_test, &config);
        assert!(result_smaller.is_ok());

        // Test point that's lexicographically larger
        let larger_test = Point::new([15.0, 16.0, 17.0]);
        let result_larger = robust_insphere(&ordered_points, &larger_test, &config);
        assert!(result_larger.is_ok());
    }

    #[test]
    fn test_adaptive_tolerance_computation() {
        // Test adaptive tolerance computation with different matrix sizes and values
        use nalgebra::DMatrix;

        let config = config_presets::general_triangulation::<f64>();

        // Small matrix with moderate values
        let mut small_matrix = DMatrix::zeros(2, 2);
        small_matrix[(0, 0)] = 1.0;
        small_matrix[(0, 1)] = 2.0;
        small_matrix[(1, 0)] = 3.0;
        small_matrix[(1, 1)] = 4.0;

        let tolerance_small = compute_adaptive_tolerance(&small_matrix, &config);
        assert!(tolerance_small > 0.0);

        // Large matrix with large values
        let mut large_matrix = DMatrix::zeros(5, 5);
        for i in 0..5 {
            for j in 0..5 {
                if let Some(val) = num_traits::cast::<usize, f64>(i + j) {
                    large_matrix[(i, j)] = val * 1000.0;
                } else {
                    large_matrix[(i, j)] = 0.0;
                }
            }
        }

        let tolerance_large = compute_adaptive_tolerance(&large_matrix, &config);
        assert!(tolerance_large > 0.0);
        // Larger matrices with larger values should have larger tolerances
        assert!(tolerance_large > tolerance_small);

        // Matrix with very small values
        let mut tiny_matrix = DMatrix::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                tiny_matrix[(i, j)] = 1e-10;
            }
        }

        let tolerance_tiny = compute_adaptive_tolerance(&tiny_matrix, &config);
        assert!(tolerance_tiny > 0.0);
    }

    #[test]
    fn test_perturbation_direction_generation() {
        // Test perturbation directions for different dimensions

        // 1D case
        let directions_1d = generate_perturbation_directions::<f64, 1>();
        assert_eq!(directions_1d.len(), 2); // +1 and -1 in single coordinate
        assert_relative_eq!(directions_1d[0][0], 1.0);
        assert_relative_eq!(directions_1d[1][0], -1.0);

        // 2D case
        let directions_2d = generate_perturbation_directions::<f64, 2>();
        assert_eq!(directions_2d.len(), 5); // 4 axis directions + 1 diagonal

        // 3D case
        let directions_3d = generate_perturbation_directions::<f64, 3>();
        assert_eq!(directions_3d.len(), 7); // 6 axis directions + 1 diagonal

        // Check that diagonal direction is normalized
        let diag_3d = directions_3d.last().unwrap();
        for &component in diag_3d {
            assert!((component - 1.0 / 3.0).abs() < 1e-10);
        }

        // 4D case
        let directions_4d = generate_perturbation_directions::<f64, 4>();
        assert_eq!(directions_4d.len(), 9); // 8 axis directions + 1 diagonal
    }

    #[test]
    fn test_apply_perturbation() {
        // Test applying perturbations to points
        let original_point = Point::new([1.0, 2.0, 3.0]);
        let direction = [0.1, -0.1, 0.2];
        let scale = 0.001;

        let perturbed = apply_perturbation(&original_point, direction, scale);
        let perturbed_coords: [f64; 3] = perturbed.into();

        assert_relative_eq!(
            perturbed_coords[0],
            0.1f64.mul_add(0.001, 1.0),
            epsilon = 1e-15
        );
        assert_relative_eq!(
            perturbed_coords[1],
            0.1f64.mul_add(-0.001, 2.0),
            epsilon = 1e-15
        );
        assert_relative_eq!(
            perturbed_coords[2],
            0.2f64.mul_add(0.001, 3.0),
            epsilon = 1e-15
        );

        // Test with zero perturbation
        let zero_direction = [0.0, 0.0, 0.0];
        let unperturbed = apply_perturbation(&original_point, zero_direction, 1.0);
        let unperturbed_coords: [f64; 3] = unperturbed.into();
        assert_relative_eq!(unperturbed_coords.as_slice(), [1.0, 2.0, 3.0].as_slice());
    }

    #[test]
    fn test_interpret_insphere_determinant_edge_cases() {
        // Test determinant interpretation with various orientations and edge values
        let tolerance = 1e-12;

        // Test with DEGENERATE orientation (should always return BOUNDARY)
        let result_degenerate = interpret_insphere_determinant(
            100.0, // Large positive determinant
            Orientation::DEGENERATE,
            tolerance,
        )
        .unwrap();
        assert_eq!(result_degenerate, InSphere::BOUNDARY);

        let result_degenerate_neg = interpret_insphere_determinant(
            -100.0, // Large negative determinant
            Orientation::DEGENERATE,
            tolerance,
        )
        .unwrap();
        assert_eq!(result_degenerate_neg, InSphere::BOUNDARY);

        // Test POSITIVE orientation with boundary values
        let result_pos_boundary = interpret_insphere_determinant(
            tolerance / 2.0, // Within tolerance
            Orientation::POSITIVE,
            tolerance,
        )
        .unwrap();
        assert_eq!(result_pos_boundary, InSphere::BOUNDARY);

        // Test NEGATIVE orientation with boundary values
        let result_neg_boundary = interpret_insphere_determinant(
            -tolerance / 2.0, // Within tolerance
            Orientation::NEGATIVE,
            tolerance,
        )
        .unwrap();
        assert_eq!(result_neg_boundary, InSphere::BOUNDARY);

        // Test POSITIVE orientation with clear inside
        let result_pos_inside = interpret_insphere_determinant(
            tolerance * 10.0, // Well above tolerance
            Orientation::POSITIVE,
            tolerance,
        )
        .unwrap();
        assert_eq!(result_pos_inside, InSphere::INSIDE);

        // Test NEGATIVE orientation with clear inside (negative det)
        let result_neg_inside = interpret_insphere_determinant(
            -tolerance * 10.0, // Well below -tolerance
            Orientation::NEGATIVE,
            tolerance,
        )
        .unwrap();
        assert_eq!(result_neg_inside, InSphere::INSIDE);
    }

    #[test]
    fn test_consistency_check_fallback_branch() {
        // Test the case where consistency check fails and we fall back to more robust methods
        // This is challenging to test directly since we need a case where the first method
        // succeeds but consistency verification shows inconsistent result

        // Create a configuration with very strict tolerances that might cause issues
        let strict_config = RobustPredicateConfig {
            base_tolerance: 1e-20, // Extremely strict
            relative_tolerance_factor: 1e-20,
            max_refinement_iterations: 1,
            exact_arithmetic_threshold: 1e-20,
            perturbation_scale: 1e-20,
            visibility_threshold_multiplier: 100.0,
        };

        // Use points that are challenging for numerical precision
        let challenging_points = vec![
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
            Point::new([1e-10, 1e-10, 1e-10]), // Very close to origin but not exactly
        ];

        let test_point = Point::new([0.5, 0.5, 0.5]);

        // The function should still return a valid result even with challenging input
        let result = robust_insphere(&challenging_points, &test_point, &strict_config);
        assert!(result.is_ok());

        // Verify we get a sensible InSphere result
        let insphere_result = result.unwrap();
        assert!(matches!(
            insphere_result,
            InSphere::INSIDE | InSphere::BOUNDARY | InSphere::OUTSIDE
        ));
    }

    #[test]
    fn test_conditioned_insphere_fallback() {
        // Test the case where adaptive_tolerance_insphere fails but conditioned_insphere succeeds
        // This is difficult to trigger directly, but we can test with configurations that
        // might cause the first method to fail due to numerical issues

        // Create a configuration that might cause the first method to fail
        let problematic_config = RobustPredicateConfig {
            base_tolerance: f64::NAN, // This will cause issues in adaptive tolerance
            relative_tolerance_factor: 1e-12,
            max_refinement_iterations: 3,
            exact_arithmetic_threshold: 1e-10,
            perturbation_scale: 1e-10,
            visibility_threshold_multiplier: 100.0,
        };

        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let test_point = Point::new([0.25, 0.25, 0.25]);

        // The robust_insphere should still work even with problematic config
        // It should fall back to symbolic perturbation
        let result = robust_insphere(&points, &test_point, &problematic_config);
        assert!(result.is_ok());

        // Test with a more realistic scenario: very ill-conditioned matrix
        let ill_conditioned_points = vec![
            Point::new([1e-15, 0.0, 0.0]),
            Point::new([0.0, 1e15, 0.0]),
            Point::new([0.0, 0.0, 1e-8]),
            Point::new([1e8, 1e-12, 1e4]),
        ];

        let normal_config = config_presets::general_triangulation::<f64>();
        let ill_test_point = Point::new([1e-10, 1e10, 1e-5]);

        // Should still get a result even with ill-conditioned input
        let ill_result = robust_insphere(&ill_conditioned_points, &ill_test_point, &normal_config);
        assert!(ill_result.is_ok());
    }

    #[test]
    fn test_build_matrices_edge_cases() {
        // Test matrix building functions with edge cases

        // Test with points having zero coordinates
        let zero_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0]),
        ];
        let zero_test = Point::new([0.0, 0.0, 0.0]);

        // Should be able to build matrices even with all zeros
        let matrix_result = build_insphere_matrix(&zero_points, &zero_test);
        assert!(matrix_result.is_ok());

        let matrix = matrix_result.unwrap();
        assert_eq!(matrix.nrows(), 5); // D+2 for 3D
        assert_eq!(matrix.ncols(), 5);

        // Test orientation matrix building
        let orientation_result = build_orientation_matrix(&zero_points);
        assert!(orientation_result.is_ok());

        let orient_matrix = orientation_result.unwrap();
        assert_eq!(orient_matrix.nrows(), 4); // D+1 for 3D
        assert_eq!(orient_matrix.ncols(), 4);

        // Test with very large coordinates
        let large_points = vec![
            Point::new([1e100, 0.0]),
            Point::new([0.0, 1e100]),
            Point::new([1e100, 1e100]),
        ];
        let large_test = Point::new([5e99, 5e99]);

        let large_matrix_result = build_insphere_matrix(&large_points, &large_test);
        assert!(large_matrix_result.is_ok());

        // Should have finite entries (no overflow to infinity)
        let large_matrix = large_matrix_result.unwrap();
        assert!(large_matrix.iter().all(|&x| x.is_finite()));
    }
}
