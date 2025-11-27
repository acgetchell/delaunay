//! Geometric kernel abstraction following CGAL's design.
//!
//! The Kernel trait defines the interface for geometric predicates used by
//! higher-level triangulation algorithms. This separation allows swapping
//! between fast floating-point and robust exact-arithmetic implementations.

use core::iter::Sum;
use core::marker::PhantomData;

use crate::geometry::point::Point;
use crate::geometry::predicates::{InSphere, Orientation, insphere, simplex_orientation};
use crate::geometry::robust_predicates::{
    RobustPredicateConfig, config_presets, robust_insphere, robust_orientation,
};
use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateScalar};

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
pub trait Kernel<const D: usize>: Clone {
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
    T: CoordinateScalar + Sum,
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
        let result = insphere(simplex_points, *test_point)?;
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
    T: CoordinateScalar + Sum + num_traits::Zero,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;

    #[test]
    fn test_fast_kernel_orientation_3d() {
        let kernel = FastKernel::<f64>::new();

        // Create a 3D simplex (tetrahedron)
        let points = vec![
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
        let simplex = vec![
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
        let points = vec![
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
        let simplex = vec![
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
        let simplex = vec![
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
        let points = vec![
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

        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let orientation = kernel.orientation(&points).unwrap();
        assert!(orientation != 0); // Should be non-degenerate
    }
}
