//! Geometric kernel abstraction following CGAL's design.
//!
//! The Kernel trait defines the interface for geometric predicates used by
//! higher-level triangulation algorithms. This separation allows swapping
//! between fast floating-point and robust exact-arithmetic implementations.

#![forbid(unsafe_code)]

use crate::geometry::matrix::{matrix_get, matrix_set, matrix_zero_like};
use crate::geometry::point::Point;
use crate::geometry::predicates::{InSphere, Orientation, insphere_lifted, simplex_orientation};
use crate::geometry::robust_predicates::{
    RobustPredicateConfig, config_presets, robust_insphere, robust_orientation,
};
use crate::geometry::traits::coordinate::{
    Coordinate, CoordinateConversionError, CoordinateScalar, ScalarSummable,
};
use crate::geometry::util::{safe_coords_to_f64, safe_scalar_to_f64, squared_norm};
use core::marker::PhantomData;

/// Geometric kernel trait defining predicates for triangulation algorithms.
///
/// Following CGAL's architecture, the kernel encapsulates all geometric
/// operations, allowing the triangulation data structure to remain purely
/// combinatorial.
///
/// # Examples
///
/// ```
/// use delaunay::geometry::kernel::{FastKernel, Kernel};
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
///
/// let kernel = FastKernel::<f64>::new();
///
/// // Test orientation of a 2D triangle
/// let points = [
///     Point::new([0.0, 0.0]),
///     Point::new([1.0, 0.0]),
///     Point::new([0.5, 1.0]),
/// ];
/// let orientation = kernel.orientation(&points).unwrap();
/// assert!(orientation != 0); // Not degenerate
///
/// // Test if point is inside circumcircle
/// let test_point = Point::new([0.5, 0.3]);
/// let result = kernel.in_sphere(&points, &test_point).unwrap();
/// assert_eq!(result, 1); // Inside
/// ```
pub trait Kernel<const D: usize>: Clone + Default {
    /// The scalar type used for coordinates.
    type Scalar: CoordinateScalar;

    /// Compute the orientation of a simplex.
    ///
    /// Returns the sign of the determinant:
    /// - `-1`: Negative orientation
    /// - `0`: Degenerate (points are coplanar/collinear)
    /// - `+1`: Positive orientation
    ///
    /// # Arguments
    ///
    /// * `points` - Slice of exactly D+1 points forming the simplex
    ///
    /// # Returns
    ///
    /// Returns an `i32` indicating the orientation: -1, 0, or +1.
    ///
    /// # Errors
    ///
    /// Returns `CoordinateConversionError` if:
    /// - The number of points is not exactly D+1
    /// - Coordinate conversion fails
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::geometry::kernel::{FastKernel, Kernel};
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    ///
    /// let kernel = FastKernel::<f64>::new();
    ///
    /// // 3D tetrahedron
    /// let points = [
    ///     Point::new([0.0, 0.0, 0.0]),
    ///     Point::new([1.0, 0.0, 0.0]),
    ///     Point::new([0.0, 1.0, 0.0]),
    ///     Point::new([0.0, 0.0, 1.0]),
    /// ];
    /// let orientation = kernel.orientation(&points).unwrap();
    /// assert!(orientation == -1 || orientation == 1); // Non-degenerate
    /// ```
    fn orientation(
        &self,
        points: &[Point<Self::Scalar, D>],
    ) -> Result<i32, CoordinateConversionError>;

    /// Test if a point is inside, on, or outside the circumsphere of a simplex.
    ///
    /// Returns:
    /// - `-1`: Point is outside the circumsphere
    /// - `0`: Point is on the circumsphere (within numerical tolerance)
    /// - `+1`: Point is inside the circumsphere
    ///
    /// # Arguments
    ///
    /// * `simplex_points` - Slice of exactly D+1 points forming the simplex
    /// * `test_point` - The point to test for containment
    ///
    /// # Returns
    ///
    /// Returns an `i32` indicating the position: -1 (outside), 0 (boundary), or +1 (inside).
    ///
    /// # Errors
    ///
    /// Returns `CoordinateConversionError` if:
    /// - The number of simplex points is not exactly D+1
    /// - Coordinate conversion fails
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::geometry::kernel::{FastKernel, Kernel};
    /// use delaunay::geometry::point::Point;
    /// use delaunay::geometry::traits::coordinate::Coordinate;
    ///
    /// let kernel = FastKernel::<f64>::new();
    ///
    /// // 3D tetrahedron
    /// let simplex = [
    ///     Point::new([0.0, 0.0, 0.0]),
    ///     Point::new([1.0, 0.0, 0.0]),
    ///     Point::new([0.0, 1.0, 0.0]),
    ///     Point::new([0.0, 0.0, 1.0]),
    /// ];
    ///
    /// // Point inside the circumsphere
    /// let inside = Point::new([0.25, 0.25, 0.25]);
    /// assert_eq!(kernel.in_sphere(&simplex, &inside).unwrap(), 1);
    ///
    /// // Point outside the circumsphere
    /// let outside = Point::new([2.0, 2.0, 2.0]);
    /// assert_eq!(kernel.in_sphere(&simplex, &outside).unwrap(), -1);
    /// ```
    fn in_sphere(
        &self,
        simplex_points: &[Point<Self::Scalar, D>],
        test_point: &Point<Self::Scalar, D>,
    ) -> Result<i32, CoordinateConversionError>;
}

/// Fast floating-point kernel.
///
/// Uses standard floating-point arithmetic for maximum performance.
/// May produce incorrect results for degenerate or near-degenerate cases.
///
/// For applications requiring guaranteed correctness in degenerate cases,
/// use [`RobustKernel`] instead.
///
/// # ⚠️ Warning: Unreliable in 3D and Higher Dimensions
///
/// **`FastKernel` should not be used for bulk Delaunay triangulation in 3D or higher
/// dimensions.** Random point sets in 3D+ routinely produce near-co-spherical
/// configurations that cause `FastKernel`'s in-sphere predicate to misclassify
/// points, leading to incorrect conflict zones, invalid topology, and construction
/// failures.
///
/// Use [`RobustKernel`] (the default) for all 3D+ work. `FastKernel` remains
/// suitable for 2D triangulations with well-conditioned input, or when explicitly
/// opted into via [`DelaunayTriangulation::with_kernel`](crate::core::delaunay_triangulation::DelaunayTriangulation::with_kernel) for advanced use cases
/// where the caller has verified the input is non-degenerate.
///
/// # Performance
///
/// `FastKernel` wraps the standard predicates from [`crate::geometry::predicates`]
/// with zero overhead, providing excellent performance for well-conditioned input.
///
/// # Examples
///
/// ```
/// use delaunay::geometry::kernel::{FastKernel, Kernel};
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
///
/// // Create a fast kernel for f64 coordinates
/// let kernel = FastKernel::<f64>::new();
///
/// // Test with a 2D triangle
/// let points = [
///     Point::new([0.0, 0.0]),
///     Point::new([1.0, 0.0]),
///     Point::new([0.0, 1.0]),
/// ];
///
/// // Check orientation
/// let orientation = kernel.orientation(&points).unwrap();
/// assert!(orientation != 0);
///
/// // Test insphere predicate
/// let test_point = Point::new([0.25, 0.25]);
/// let result = kernel.in_sphere(&points, &test_point).unwrap();
/// assert_eq!(result, 1); // Inside circumcircle
/// ```
#[derive(Clone, Default, Debug)]
pub struct FastKernel<T: CoordinateScalar> {
    _phantom: PhantomData<T>,
}

impl<T: CoordinateScalar> FastKernel<T> {
    /// Create a new fast kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::geometry::kernel::FastKernel;
    ///
    /// let kernel = FastKernel::<f64>::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T, const D: usize> Kernel<D> for FastKernel<T>
where
    T: ScalarSummable,
{
    type Scalar = T;

    fn orientation(
        &self,
        points: &[Point<Self::Scalar, D>],
    ) -> Result<i32, CoordinateConversionError> {
        let result = simplex_orientation(points)?;
        Ok(match result {
            Orientation::NEGATIVE => -1,
            Orientation::DEGENERATE => 0,
            Orientation::POSITIVE => 1,
        })
    }

    fn in_sphere(
        &self,
        simplex_points: &[Point<Self::Scalar, D>],
        test_point: &Point<Self::Scalar, D>,
    ) -> Result<i32, CoordinateConversionError> {
        // Use insphere_lifted for optimal performance (5.3x faster in 3D)
        let result = insphere_lifted(simplex_points, *test_point).map_err(|e| {
            // Preserve original CoordinateConversionError if present
            match e {
                crate::core::cell::CellValidationError::CoordinateConversion { source } => source,
                _ => CoordinateConversionError::ConversionFailed {
                    coordinate_index: 0,
                    coordinate_value: format!("{e}"),
                    from_type: "insphere_lifted",
                    to_type: "in_sphere",
                },
            }
        })?;
        Ok(match result {
            InSphere::OUTSIDE => -1,
            InSphere::BOUNDARY => 0,
            InSphere::INSIDE => 1,
        })
    }
}

/// Robust exact-arithmetic kernel.
///
/// Uses adaptive tolerance and symbolic perturbation predicates that are
/// guaranteed to be correct even for degenerate cases. Slower than
/// [`FastKernel`] but provides better numerical stability.
///
/// # Robustness Features
///
/// - **Adaptive tolerance**: Scales with coordinate magnitude
/// - **Symbolic perturbation**: Deterministic tie-breaking for degenerate cases
/// - **Configurable**: Supports multiple precision levels via [`RobustPredicateConfig`]
///
/// # Performance
///
/// Typically 2-3x slower than `FastKernel` due to additional robustness checks,
/// but essential for:
/// - Nearly-degenerate point configurations
/// - High-precision applications
/// - Safety-critical computations
///
/// # Examples
///
/// ```
/// use delaunay::geometry::kernel::{RobustKernel, Kernel};
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
///
/// // Create with default configuration
/// let kernel = RobustKernel::<f64>::new();
///
/// // Test with a 3D tetrahedron
/// let points = [
///     Point::new([0.0, 0.0, 0.0]),
///     Point::new([1.0, 0.0, 0.0]),
///     Point::new([0.0, 1.0, 0.0]),
///     Point::new([0.0, 0.0, 1.0]),
/// ];
///
/// let orientation = kernel.orientation(&points).unwrap();
/// assert!(orientation != 0); // Non-degenerate
///
/// // Test insphere with high-precision config
/// use delaunay::geometry::robust_predicates::config_presets;
/// let precise_kernel = RobustKernel::with_config(
///     config_presets::high_precision::<f64>()
/// );
///
/// let test_point = Point::new([0.25, 0.25, 0.25]);
/// let result = precise_kernel.in_sphere(&points, &test_point).unwrap();
/// assert_eq!(result, 1); // Inside circumsphere
/// ```
#[derive(Clone, Debug)]
pub struct RobustKernel<T: CoordinateScalar> {
    config: RobustPredicateConfig<T>,
    _phantom: PhantomData<T>,
}

impl<T: CoordinateScalar> RobustKernel<T> {
    /// Create a new robust kernel with general triangulation configuration.
    ///
    /// This uses [`config_presets::general_triangulation`] which provides
    /// balanced robustness suitable for most applications.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::geometry::kernel::RobustKernel;
    ///
    /// let kernel = RobustKernel::<f64>::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: config_presets::general_triangulation(),
            _phantom: PhantomData,
        }
    }

    /// Create a robust kernel with a custom configuration.
    ///
    /// Use [`config_presets`] to access predefined configurations for
    /// different precision levels and use cases.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::geometry::kernel::RobustKernel;
    /// use delaunay::geometry::robust_predicates::config_presets;
    ///
    /// // High-precision configuration
    /// let kernel = RobustKernel::with_config(
    ///     config_presets::high_precision::<f64>()
    /// );
    ///
    /// // Degenerate-robust configuration
    /// let kernel = RobustKernel::with_config(
    ///     config_presets::degenerate_robust::<f64>()
    /// );
    /// ```
    #[must_use]
    pub const fn with_config(config: RobustPredicateConfig<T>) -> Self {
        Self {
            config,
            _phantom: PhantomData,
        }
    }
}

impl<T: CoordinateScalar> Default for RobustKernel<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const D: usize> Kernel<D> for RobustKernel<T>
where
    T: ScalarSummable,
{
    type Scalar = T;

    fn orientation(
        &self,
        points: &[Point<Self::Scalar, D>],
    ) -> Result<i32, CoordinateConversionError> {
        let result = robust_orientation(points, &self.config)?;
        Ok(match result {
            Orientation::NEGATIVE => -1,
            Orientation::DEGENERATE => 0,
            Orientation::POSITIVE => 1,
        })
    }

    fn in_sphere(
        &self,
        simplex_points: &[Point<Self::Scalar, D>],
        test_point: &Point<Self::Scalar, D>,
    ) -> Result<i32, CoordinateConversionError> {
        let result = robust_insphere(simplex_points, test_point, &self.config)?;
        Ok(match result {
            InSphere::OUTSIDE => -1,
            InSphere::BOUNDARY => 0,
            InSphere::INSIDE => 1,
        })
    }
}

/// Adaptive precision kernel with Simulation of Simplicity.
///
/// Uses a three-layer evaluation strategy for maximum robustness:
/// 1. **Fast filter**: `det_direct()` + `det_errbound()` (provable for D ≤ 4)
/// 2. **Exact arithmetic**: `det_sign_exact()` via Bareiss algorithm in `BigRational`
/// 3. **`SoS` tie-breaking**: Simulation of Simplicity for truly degenerate cases
///
/// Unlike [`RobustKernel`], this kernel:
/// - Never modifies coordinates (no perturbation)
/// - Never returns degenerate/boundary (orientation ≠ 0, insphere ≠ 0)
/// - Uses provable error bounds instead of heuristic tolerance
///
/// # Examples
///
/// ```
/// use delaunay::geometry::kernel::{AdaptiveKernel, Kernel};
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
///
/// let kernel = AdaptiveKernel::<f64>::new();
///
/// // Even collinear points get a deterministic non-zero orientation
/// let collinear = [
///     Point::new([0.0, 0.0]),
///     Point::new([1.0, 0.0]),
///     Point::new([2.0, 0.0]),
/// ];
/// let orientation = kernel.orientation(&collinear).unwrap();
/// assert!(orientation == 1 || orientation == -1); // Never 0
/// ```
#[derive(Clone, Default, Debug)]
pub struct AdaptiveKernel<T: CoordinateScalar> {
    _phantom: PhantomData<T>,
}

impl<T: CoordinateScalar> AdaptiveKernel<T> {
    /// Create a new adaptive kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::geometry::kernel::AdaptiveKernel;
    ///
    /// let kernel = AdaptiveKernel::<f64>::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T, const D: usize> Kernel<D> for AdaptiveKernel<T>
where
    T: ScalarSummable,
{
    type Scalar = T;

    fn orientation(
        &self,
        points: &[Point<Self::Scalar, D>],
    ) -> Result<i32, CoordinateConversionError> {
        if points.len() != D + 1 {
            return Err(CoordinateConversionError::ConversionFailed {
                coordinate_index: 0,
                coordinate_value: format!("Expected {} points, got {}", D + 1, points.len()),
                from_type: "point count",
                to_type: "valid simplex",
            });
        }

        let k = D + 1;

        try_with_la_stack_matrix!(k, |matrix| {
            // Build (D+1)×(D+1) homogeneous orientation matrix.
            for (i, p) in points.iter().enumerate() {
                let coords_f64 = safe_coords_to_f64(p.coords())?;
                for (j, &v) in coords_f64.iter().enumerate() {
                    matrix_set(&mut matrix, i, j, v);
                }
                matrix_set(&mut matrix, i, D, 1.0);
            }

            // Layer 1 + 2: fast filter and exact Bareiss.
            let sign = crate::geometry::sos::exact_det_sign(&matrix, k);
            if sign != 0 {
                return Ok(sign);
            }

            // Layer 3: SoS tie-breaking.
            let f64_points: Vec<Point<f64, D>> = points
                .iter()
                .map(|p| safe_coords_to_f64(p.coords()).map(Point::new))
                .collect::<Result<_, _>>()?;
            crate::geometry::sos::sos_orientation_sign(&f64_points)
        })
    }

    fn in_sphere(
        &self,
        simplex_points: &[Point<Self::Scalar, D>],
        test_point: &Point<Self::Scalar, D>,
    ) -> Result<i32, CoordinateConversionError> {
        if simplex_points.len() != D + 1 {
            return Err(CoordinateConversionError::ConversionFailed {
                coordinate_index: 0,
                coordinate_value: format!(
                    "Expected {} points, got {}",
                    D + 1,
                    simplex_points.len()
                ),
                from_type: "point count",
                to_type: "valid simplex",
            });
        }

        let k = D + 1;

        try_with_la_stack_matrix!(k, |matrix| {
            // Build (D+1)×(D+1) lifted insphere matrix using relative
            // coordinates centered on simplex_points[0].
            let ref_coords = simplex_points[0].coords();

            for (row, point) in simplex_points.iter().skip(1).enumerate() {
                let coords = point.coords();
                let mut rel_t: [T; D] = [T::zero(); D];
                for (dst, (p, r)) in rel_t.iter_mut().zip(coords.iter().zip(ref_coords.iter())) {
                    *dst = *p - *r;
                }
                let rel_f64: [f64; D] = safe_coords_to_f64(&rel_t)?;
                for (j, &v) in rel_f64.iter().enumerate() {
                    matrix_set(&mut matrix, row, j, v);
                }
                let sq_norm = squared_norm(&rel_t);
                let sq_f64 = safe_scalar_to_f64(sq_norm)?;
                matrix_set(&mut matrix, row, D, sq_f64);
            }

            // Test point row.
            let test_coords = test_point.coords();
            let mut test_rel_t: [T; D] = [T::zero(); D];
            for (dst, (p, r)) in test_rel_t
                .iter_mut()
                .zip(test_coords.iter().zip(ref_coords.iter()))
            {
                *dst = *p - *r;
            }
            let test_rel_f64: [f64; D] = safe_coords_to_f64(&test_rel_t)?;
            for (j, &v) in test_rel_f64.iter().enumerate() {
                matrix_set(&mut matrix, D, j, v);
            }
            let test_sq = squared_norm(&test_rel_t);
            let test_sq_f64 = safe_scalar_to_f64(test_sq)?;
            matrix_set(&mut matrix, D, D, test_sq_f64);

            // Compute relative orientation from D×D coordinate block
            // (same embedding as insphere_lifted).
            let mut orient_matrix = matrix_zero_like(&matrix);
            for i in 0..D {
                for j in 0..D {
                    matrix_set(&mut orient_matrix, i, j, matrix_get(&matrix, i, j));
                }
            }
            matrix_set(&mut orient_matrix, D, D, 1.0);

            // Layer 1 + 2 for both predicates.
            let rel_orient_sign = crate::geometry::sos::exact_det_sign(&orient_matrix, k);
            let insphere_det_sign = crate::geometry::sos::exact_det_sign(&matrix, k);

            // Fast path: both non-degenerate.
            if rel_orient_sign != 0 && insphere_det_sign != 0 {
                let orient_factor = -rel_orient_sign;
                return Ok((insphere_det_sign * orient_factor).signum());
            }

            // At least one predicate needs SoS → convert to f64 points.
            let f64_simplex: Vec<Point<f64, D>> = simplex_points
                .iter()
                .map(|p| safe_coords_to_f64(p.coords()).map(Point::new))
                .collect::<Result<_, _>>()?;
            let f64_test = Point::new(safe_coords_to_f64(test_point.coords())?);

            // Resolve orientation factor.
            let orient_factor: i32 = if rel_orient_sign != 0 {
                -rel_orient_sign
            } else {
                // Orientation degenerate → SoS gives absolute orientation sign.
                // rel_orient = (-1)^D × abs_orient
                // orient_factor = -rel_orient = (-1)^(D+1) × abs_orient
                let sos_abs = crate::geometry::sos::sos_orientation_sign(&f64_simplex)?;
                if D.is_multiple_of(2) {
                    -sos_abs
                } else {
                    sos_abs
                }
            };

            // Resolve insphere sign.
            let insphere_effective: i32 = if insphere_det_sign != 0 {
                insphere_det_sign
            } else {
                crate::geometry::sos::sos_insphere_sign(&f64_simplex, &f64_test)?
            };

            Ok((insphere_effective * orient_factor).signum())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;

    #[test]
    fn test_fast_kernel_orientation_3d() {
        let kernel = FastKernel::<f64>::new();

        // Create a 3D simplex (tetrahedron)
        let points = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let orientation = kernel.orientation(&points).unwrap();

        // Should have a definite orientation (not degenerate)
        assert!(orientation == -1 || orientation == 1);
    }

    #[test]
    fn test_fast_kernel_in_sphere_3d() {
        let kernel = FastKernel::<f64>::new();

        // Create a 3D simplex (tetrahedron)
        let simplex = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        // Point clearly inside the circumsphere
        let inside_point = Point::new([0.25, 0.25, 0.25]);
        let result = kernel.in_sphere(&simplex, &inside_point).unwrap();
        assert_eq!(result, 1); // INSIDE

        // Point clearly outside the circumsphere
        let outside_point = Point::new([2.0, 2.0, 2.0]);
        let result = kernel.in_sphere(&simplex, &outside_point).unwrap();
        assert_eq!(result, -1); // OUTSIDE
    }

    #[test]
    fn test_robust_kernel_orientation_3d() {
        let kernel = RobustKernel::<f64>::new();

        // Create a 3D simplex (tetrahedron)
        let points = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let orientation = kernel.orientation(&points).unwrap();

        // Should have a definite orientation (not degenerate)
        assert!(orientation == -1 || orientation == 1);
    }

    #[test]
    fn test_robust_kernel_in_sphere_3d() {
        let kernel = RobustKernel::<f64>::new();

        // Create a 3D simplex (tetrahedron)
        let simplex = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        // Point clearly inside the circumsphere
        let inside_point = Point::new([0.25, 0.25, 0.25]);
        let result = kernel.in_sphere(&simplex, &inside_point).unwrap();
        assert_eq!(result, 1); // INSIDE

        // Point clearly outside the circumsphere
        let outside_point = Point::new([2.0, 2.0, 2.0]);
        let result = kernel.in_sphere(&simplex, &outside_point).unwrap();
        assert_eq!(result, -1); // OUTSIDE
    }

    #[test]
    fn test_kernel_consistency_fast_vs_robust() {
        let fast_kernel = FastKernel::<f64>::new();
        let robust_kernel = RobustKernel::<f64>::new();

        // Create a 2D simplex (triangle)
        let simplex = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.5, 1.0]),
        ];

        // Test orientation consistency
        let fast_orientation = fast_kernel.orientation(&simplex).unwrap();
        let robust_orientation = robust_kernel.orientation(&simplex).unwrap();
        assert_eq!(fast_orientation, robust_orientation);

        // Test in_sphere consistency for clear cases
        let test_point = Point::new([0.5, 0.3]);
        let fast_result = fast_kernel.in_sphere(&simplex, &test_point).unwrap();
        let robust_result = robust_kernel.in_sphere(&simplex, &test_point).unwrap();
        assert_eq!(fast_result, robust_result);
    }

    #[test]
    fn test_fast_kernel_2d() {
        let kernel = FastKernel::<f64>::new();

        // 2D triangle
        let points = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.5, 1.0]),
        ];

        let orientation = kernel.orientation(&points).unwrap();
        assert!(orientation != 0); // Not degenerate

        // Point inside circumcircle
        let inside = Point::new([0.5, 0.3]);
        let result = kernel.in_sphere(&points, &inside).unwrap();
        assert_eq!(result, 1); // INSIDE
    }

    #[test]
    fn test_robust_kernel_with_custom_config() {
        let config = config_presets::high_precision();
        let kernel = RobustKernel::<f64>::with_config(config);

        let points = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let orientation = kernel.orientation(&points).unwrap();
        assert!(orientation != 0); // Should be non-degenerate
    }

    // =============================================================================
    // Degeneracy Detection Tests
    // =============================================================================

    #[test]
    fn test_orientation_collinear_2d_fast() {
        let kernel = FastKernel::<f64>::new();

        // Three collinear points on x-axis
        let collinear = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([2.0, 0.0]),
        ];

        let orientation = kernel.orientation(&collinear).unwrap();
        assert_eq!(
            orientation, 0,
            "Collinear points should have zero orientation"
        );
    }

    #[test]
    fn test_orientation_collinear_2d_robust() {
        let kernel = RobustKernel::<f64>::new();

        // Three collinear points on x-axis
        let collinear = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([2.0, 0.0]),
        ];

        let orientation = kernel.orientation(&collinear).unwrap();
        assert_eq!(
            orientation, 0,
            "Collinear points should have zero orientation"
        );
    }

    #[test]
    fn test_orientation_collinear_diagonal_2d() {
        let kernel = FastKernel::<f64>::new();

        // Three collinear points on diagonal
        let collinear = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 1.0]),
            Point::new([2.0, 2.0]),
        ];

        let orientation = kernel.orientation(&collinear).unwrap();
        assert_eq!(
            orientation, 0,
            "Diagonal collinear points should have zero orientation"
        );
    }

    #[test]
    fn test_orientation_valid_triangle_2d() {
        let kernel = FastKernel::<f64>::new();

        // Valid triangle (not collinear)
        let triangle = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.5, 0.866]), // ~60 degree angle
        ];

        let orientation = kernel.orientation(&triangle).unwrap();
        assert_ne!(
            orientation, 0,
            "Valid triangle should have non-zero orientation"
        );
        assert!(
            orientation == 1 || orientation == -1,
            "Orientation should be +1 or -1"
        );
    }

    #[test]
    fn test_orientation_coplanar_3d_fast() {
        let kernel = FastKernel::<f64>::new();

        // Four coplanar points in xy-plane
        let coplanar = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 0.0]), // All z=0
        ];

        let orientation = kernel.orientation(&coplanar).unwrap();
        assert_eq!(
            orientation, 0,
            "Coplanar points should have zero orientation"
        );
    }

    #[test]
    fn test_orientation_coplanar_3d_robust() {
        let kernel = RobustKernel::<f64>::new();

        // Four coplanar points in xy-plane
        let coplanar = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 0.0]), // All z=0
        ];

        let orientation = kernel.orientation(&coplanar).unwrap();
        assert_eq!(
            orientation, 0,
            "Coplanar points should have zero orientation"
        );
    }

    #[test]
    fn test_orientation_valid_tetrahedron_3d() {
        let kernel = FastKernel::<f64>::new();

        // Valid tetrahedron (not coplanar)
        let tetrahedron = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let orientation = kernel.orientation(&tetrahedron).unwrap();
        assert_ne!(
            orientation, 0,
            "Valid tetrahedron should have non-zero orientation"
        );
        assert!(
            orientation == 1 || orientation == -1,
            "Orientation should be +1 or -1"
        );
    }

    #[test]
    fn test_orientation_nearly_collinear_2d_robust() {
        let kernel = RobustKernel::<f64>::new();

        // Nearly collinear points (small perturbation)
        let nearly_collinear = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([2.0, 1e-10]), // Tiny deviation from collinearity
        ];

        let orientation = kernel.orientation(&nearly_collinear).unwrap();
        // Robust predicates should detect this as non-degenerate
        // (though it may be very close to zero, it should return definite answer)
        assert!(orientation == -1 || orientation == 0 || orientation == 1);
    }

    #[test]
    fn test_orientation_extreme_coordinates_2d() {
        let kernel = RobustKernel::<f64>::new();

        // Triangle with large coordinates
        let large_triangle = [
            Point::new([1e6, 1e6]),
            Point::new([1e6 + 1.0, 1e6]),
            Point::new([1e6, 1e6 + 1.0]),
        ];

        let orientation = kernel.orientation(&large_triangle).unwrap();
        assert_ne!(
            orientation, 0,
            "Triangle with large coordinates should be non-degenerate"
        );
    }

    #[test]
    fn test_orientation_4d_valid_simplex() {
        let kernel = FastKernel::<f64>::new();

        // 4D simplex (5 points)
        let simplex_4d = [
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];

        let orientation = kernel.orientation(&simplex_4d).unwrap();
        assert_ne!(
            orientation, 0,
            "4D simplex should have non-zero orientation"
        );
    }

    #[test]
    fn test_orientation_4d_degenerate() {
        let kernel = FastKernel::<f64>::new();

        // 4D degenerate simplex (all points in 3D hyperplane)
        let degenerate_4d = [
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 0.5, 0.0]), // All w=0
        ];

        let orientation = kernel.orientation(&degenerate_4d).unwrap();
        assert_eq!(
            orientation, 0,
            "Degenerate 4D simplex should have zero orientation"
        );
    }

    #[test]
    fn test_orientation_small_but_valid_2d() {
        let kernel = FastKernel::<f64>::new();

        // Very small but valid triangle
        let small_triangle = [
            Point::new([0.0, 0.0]),
            Point::new([1e-6, 0.0]),
            Point::new([0.0, 1e-6]),
        ];

        let orientation = kernel.orientation(&small_triangle).unwrap();
        assert_ne!(
            orientation, 0,
            "Small but valid triangle should be non-degenerate"
        );
    }

    #[test]
    fn test_orientation_consistency_both_kernels() {
        let fast = FastKernel::<f64>::new();
        let robust = RobustKernel::<f64>::new();

        // Test case 1: Collinear
        let collinear = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([2.0, 0.0]),
        ];
        assert_eq!(
            fast.orientation(&collinear).unwrap(),
            robust.orientation(&collinear).unwrap(),
            "Both kernels should agree on collinear points"
        );

        // Test case 2: Valid triangle
        let triangle1 = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];
        assert_eq!(
            fast.orientation(&triangle1).unwrap(),
            robust.orientation(&triangle1).unwrap(),
            "Both kernels should agree on valid triangle"
        );

        // Test case 3: Another valid triangle
        let triangle2 = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.5, 0.5]),
        ];
        assert_eq!(
            fast.orientation(&triangle2).unwrap(),
            robust.orientation(&triangle2).unwrap(),
            "Both kernels should agree on another valid triangle"
        );
    }

    #[test]
    fn test_kernel_default_trait() {
        // Test that both kernels implement Default (required for simplex validation)
        let _fast: FastKernel<f64> = FastKernel::default();
        let _robust: RobustKernel<f64> = RobustKernel::default();
    }

    #[test]
    fn test_fast_kernel_in_sphere_insufficient_vertices() {
        // Exercises the non-CoordinateConversion error path (InsufficientVertices)
        let kernel = FastKernel::<f64>::new();
        let simplex: [Point<f64, 3>; 2] =
            [Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])];
        let test_point = Point::new([0.5, 0.5, 0.5]);
        let result = kernel.in_sphere(&simplex, &test_point);
        assert!(result.is_err(), "Should error with insufficient vertices");
    }

    #[test]
    fn test_fast_kernel_in_sphere_degenerate_simplex() {
        // Exercises the DegenerateSimplex error → CoordinateConversion path
        let kernel = FastKernel::<f64>::new();
        let simplex = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([1.0, 1.0, 0.0]), // Coplanar — degenerate
        ];
        let test_point = Point::new([0.5, 0.5, 0.5]);
        let result = kernel.in_sphere(&simplex, &test_point);
        assert!(result.is_err(), "Should error with degenerate simplex");
    }

    // =============================================================================
    // AdaptiveKernel Tests
    // =============================================================================

    #[test]
    fn test_adaptive_kernel_orientation_3d() {
        let adaptive = AdaptiveKernel::<f64>::new();
        let fast = FastKernel::<f64>::new();

        let points = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let adaptive_result = adaptive.orientation(&points).unwrap();
        let fast_result = fast.orientation(&points).unwrap();
        assert_eq!(
            adaptive_result, fast_result,
            "AdaptiveKernel should agree with FastKernel on non-degenerate inputs"
        );
    }

    #[test]
    fn test_adaptive_kernel_in_sphere_3d() {
        let adaptive = AdaptiveKernel::<f64>::new();
        let fast = FastKernel::<f64>::new();

        let simplex = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let inside = Point::new([0.25, 0.25, 0.25]);
        assert_eq!(
            adaptive.in_sphere(&simplex, &inside).unwrap(),
            fast.in_sphere(&simplex, &inside).unwrap(),
            "Should agree on clearly inside point"
        );

        let outside = Point::new([2.0, 2.0, 2.0]);
        assert_eq!(
            adaptive.in_sphere(&simplex, &outside).unwrap(),
            fast.in_sphere(&simplex, &outside).unwrap(),
            "Should agree on clearly outside point"
        );
    }

    #[test]
    fn test_adaptive_kernel_collinear_2d_nonzero() {
        let kernel = AdaptiveKernel::<f64>::new();

        let collinear = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([2.0, 0.0]),
        ];

        let result = kernel.orientation(&collinear).unwrap();
        assert!(
            result == 1 || result == -1,
            "AdaptiveKernel must never return 0 for orientation, got {result}"
        );
    }

    #[test]
    fn test_adaptive_kernel_coplanar_3d_nonzero() {
        let kernel = AdaptiveKernel::<f64>::new();

        let coplanar = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 0.0]),
        ];

        let result = kernel.orientation(&coplanar).unwrap();
        assert!(
            result == 1 || result == -1,
            "AdaptiveKernel must never return 0 for orientation, got {result}"
        );
    }

    #[test]
    fn test_adaptive_kernel_cospherical_3d_nonzero() {
        let kernel = AdaptiveKernel::<f64>::new();

        let simplex = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        // (1,1,0) is exactly on the circumsphere.
        let test = Point::new([1.0, 1.0, 0.0]);

        let result = kernel.in_sphere(&simplex, &test).unwrap();
        assert!(
            result == 1 || result == -1,
            "AdaptiveKernel must never return 0 for insphere, got {result}"
        );
    }

    #[test]
    fn test_adaptive_kernel_cospherical_2d_nonzero() {
        let kernel = AdaptiveKernel::<f64>::new();

        let simplex = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];
        // (1,1) is on the circumcircle of this right triangle.
        let test = Point::new([1.0, 1.0]);

        let result = kernel.in_sphere(&simplex, &test).unwrap();
        assert!(
            result == 1 || result == -1,
            "AdaptiveKernel must never return 0 for insphere, got {result}"
        );
    }

    #[test]
    fn test_adaptive_kernel_deterministic() {
        let kernel = AdaptiveKernel::<f64>::new();

        let collinear = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([2.0, 0.0]),
        ];

        let results: Vec<i32> = (0..10)
            .map(|_| kernel.orientation(&collinear).unwrap())
            .collect();
        assert!(
            results.iter().all(|&r| r == results[0]),
            "AdaptiveKernel orientation must be deterministic"
        );
    }

    #[test]
    fn test_adaptive_kernel_consistency_all_kernels() {
        let fast = FastKernel::<f64>::new();
        let robust = RobustKernel::<f64>::new();
        let adaptive = AdaptiveKernel::<f64>::new();

        let triangle = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        let fast_orient = fast.orientation(&triangle).unwrap();
        let robust_orient = robust.orientation(&triangle).unwrap();
        let adaptive_orient = adaptive.orientation(&triangle).unwrap();
        assert_eq!(fast_orient, robust_orient);
        assert_eq!(fast_orient, adaptive_orient);

        let test_point = Point::new([0.25, 0.25]);
        let fast_insphere = fast.in_sphere(&triangle, &test_point).unwrap();
        let robust_insphere = robust.in_sphere(&triangle, &test_point).unwrap();
        let adaptive_insphere = adaptive.in_sphere(&triangle, &test_point).unwrap();
        assert_eq!(fast_insphere, robust_insphere);
        assert_eq!(fast_insphere, adaptive_insphere);
    }

    #[test]
    fn test_adaptive_kernel_4d_orientation() {
        let kernel = AdaptiveKernel::<f64>::new();

        // Non-degenerate 4D simplex
        let simplex = [
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];
        let result = kernel.orientation(&simplex).unwrap();
        assert!(result == 1 || result == -1);

        // Degenerate 4D simplex (all w=0)
        let degenerate = [
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.5, 0.5, 0.5, 0.0]),
        ];
        let result = kernel.orientation(&degenerate).unwrap();
        assert!(
            result == 1 || result == -1,
            "AdaptiveKernel must resolve 4D degeneracy, got {result}"
        );
    }

    #[test]
    fn test_adaptive_kernel_default_trait() {
        let _adaptive: AdaptiveKernel<f64> = AdaptiveKernel::default();
    }

    #[test]
    fn test_adaptive_kernel_wrong_point_count() {
        let kernel = AdaptiveKernel::<f64>::new();
        let points = [Point::new([0.0, 0.0]), Point::new([1.0, 0.0])];
        assert!(kernel.orientation(&points).is_err());

        let simplex = [Point::new([0.0, 0.0]), Point::new([1.0, 0.0])];
        let test = Point::new([0.5, 0.5]);
        assert!(kernel.in_sphere(&simplex, &test).is_err());
    }
}
