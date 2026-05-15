//! Geometric kernel abstraction following CGAL's design.
//!
//! The `Kernel` trait defines the interface for geometric predicates used by
//! higher-level triangulation algorithms. This separation allows swapping
//! between fast floating-point and robust exact-arithmetic implementations.
//!
//! # Choosing a kernel
//!
//! **`AdaptiveKernel`** (default) — best for Delaunay triangulation.
//! Provably correct predicates with zero configuration. Insphere uses
//! Simulation of Simplicity (`SoS`) to break cospherical ties
//! deterministically, so every query returns a definitive ±1.
//!
//! **`RobustKernel`** — exact-arithmetic predicates that preserve explicit
//! `BOUNDARY`/`DEGENERATE` signals and can run opt-in diagnostic consistency
//! checks. Prefer this when your application needs to detect cospherical or
//! coplanar configurations directly. Use the explicit-kernel constructors
//! (`with_kernel`, `build_with_kernel`, etc.) to opt in.
//!
//! **`FastKernel`** — raw f64 arithmetic, no robustness guarantees.
//! Only suitable for 2D with well-conditioned input.

#![forbid(unsafe_code)]

use crate::core::cell::CellValidationError;
use crate::core::collections::{MAX_PRACTICAL_DIMENSION_SIZE, SmallBuffer};
use crate::geometry::point::Point;
use crate::geometry::predicates::{
    InSphere, Orientation, insphere_lifted, relative_insphere_determinant_sign,
    relative_insphere_effective_sign, relative_insphere_signs, simplex_orientation,
};
use crate::geometry::robust_predicates::{
    robust_insphere, robust_insphere_positive_oriented, robust_orientation,
};
use crate::geometry::sos::{sos_insphere_sign, sos_orientation_sign};
use crate::geometry::traits::coordinate::{
    Coordinate, CoordinateConversionError, CoordinateScalar, DegenerateSimplexReason,
};
use crate::geometry::util::safe_coords_to_f64;
use core::marker::PhantomData;

/// Converts an insphere classification into the [`Kernel::in_sphere`] integer convention.
#[inline]
const fn insphere_to_i32(result: InSphere) -> i32 {
    match result {
        InSphere::OUTSIDE => -1,
        InSphere::BOUNDARY => 0,
        InSphere::INSIDE => 1,
    }
}

/// Geometric kernel trait defining predicates for triangulation algorithms.
///
/// Following CGAL's architecture, the kernel encapsulates all geometric
/// operations, allowing the triangulation data structure to remain purely
/// combinatorial.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::geometry::{FastKernel, Kernel};
/// use delaunay::prelude::geometry::Point;
/// use delaunay::prelude::geometry::Coordinate;
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
    /// **Note:** [`AdaptiveKernel`] resolves degenerate cases via Simulation
    /// of Simplicity and returns `0` only when points are identical in `f64`
    /// representation (a case `SoS` cannot resolve). For all distinct-point
    /// inputs, `AdaptiveKernel` returns ±1. Other kernels (`FastKernel`,
    /// `RobustKernel`) can return `0` for any degenerate input. Generic
    /// code should handle all three values.
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
    /// use delaunay::prelude::geometry::{FastKernel, Kernel};
    /// use delaunay::prelude::geometry::Point;
    /// use delaunay::prelude::geometry::Coordinate;
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
    /// **Note:** [`AdaptiveKernel`] resolves boundary cases via Simulation
    /// of Simplicity and never returns `0` for insphere. Generic code
    /// should handle all three values but must not *rely* on receiving `0`
    /// for boundary inputs.
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
    /// use delaunay::prelude::geometry::{FastKernel, Kernel};
    /// use delaunay::prelude::geometry::Point;
    /// use delaunay::prelude::geometry::Coordinate;
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

    /// Test circumsphere containment when the simplex is already known to have
    /// positive orientation.
    ///
    /// The default implementation preserves existing kernel semantics by calling
    /// [`Kernel::in_sphere`]. Kernels with an exact positive-orientation fast path
    /// can override this to avoid recomputing the orientation determinant.
    ///
    /// Callers must only use this when `simplex_points` are ordered with positive
    /// geometric orientation under the same coordinate frame used for the predicate.
    /// If that precondition is not true, optimized implementations can return an
    /// inverted classification. Check [`Kernel::orientation`] first unless the
    /// ordering was canonicalized by the triangulation.
    ///
    /// Returns:
    /// - `-1`: Point is outside the circumsphere
    /// - `0`: Point is on the circumsphere
    /// - `+1`: Point is inside the circumsphere
    ///
    /// # Errors
    ///
    /// Returns [`CoordinateConversionError`] if the simplex has the wrong number
    /// of points, contains non-convertible coordinates, or predicate evaluation
    /// cannot complete.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::geometry::{Coordinate, CoordinateConversionError, FastKernel};
    /// use delaunay::prelude::geometry::{Kernel, Point};
    ///
    /// # fn main() -> Result<(), CoordinateConversionError> {
    /// let kernel = FastKernel::<f64>::new();
    /// let simplex = [
    ///     Point::new([0.0, 0.0]),
    ///     Point::new([1.0, 0.0]),
    ///     Point::new([0.0, 1.0]),
    /// ];
    ///
    /// assert_eq!(kernel.orientation(&simplex)?, 1);
    ///
    /// let inside = Point::new([0.25, 0.25]);
    /// assert_eq!(kernel.in_sphere_positive_oriented(&simplex, &inside)?, 1);
    /// # Ok(())
    /// # }
    /// ```
    fn in_sphere_positive_oriented(
        &self,
        simplex_points: &[Point<Self::Scalar, D>],
        test_point: &Point<Self::Scalar, D>,
    ) -> Result<i32, CoordinateConversionError> {
        self.in_sphere(simplex_points, test_point)
    }
}

/// Marker trait for kernels that provide exact geometric predicates.
///
/// Exact-predicate kernels guarantee that [`Kernel::orientation`] and
/// [`Kernel::in_sphere`] return the mathematically correct sign for all
/// inputs in the supported dimension `D`, including near-degenerate configurations. This
/// eliminates the silent misclassification that can occur with floating-point-only kernels like
/// [`FastKernel`].
///
/// The exact stack-matrix path currently supports insphere predicates through `D <= 5`.
/// Higher dimensions can still use deterministic robust fallbacks, but they do not implement this
/// marker because flip repair uses it as a compile-time exactness gate.
///
/// # Implementors
///
/// - [`AdaptiveKernel`] — exact arithmetic + Simulation of Simplicity
///   (never returns `0` for distinct points)
/// - [`RobustKernel`] — exact Bareiss arithmetic (may return `0` for
///   boundary/degenerate, but never a *wrong* non-zero sign)
///
/// [`FastKernel`] does **not** implement this trait because its raw
/// floating-point arithmetic can produce incorrect signs for
/// near-degenerate inputs.
///
/// # Usage
///
/// This marker inherits [`Kernel`] so exactness and kernel usability remain one
/// contract. Functions that require predicate correctness for safety — such as
/// flip-based Delaunay repair — should bound their kernel parameter with this
/// trait:
///
/// ```rust,ignore
/// fn repair<K, const D: usize>(kernel: &K)
/// where
///     K: ExactPredicates<D>,
/// { /* ... */ }
/// ```
///
/// # Negative example
///
/// [`FastKernel`] does not implement `ExactPredicates`, so this fails:
///
/// ```compile_fail
/// use delaunay::prelude::geometry::{ExactPredicates, FastKernel};
/// fn requires_exact<T: ExactPredicates<3>>() {}
/// requires_exact::<FastKernel<f64>>(); // ERROR: FastKernel doesn't implement ExactPredicates
/// ```
///
/// Dimension-bound exactness is also enforced:
///
/// ```compile_fail
/// use delaunay::prelude::geometry::{AdaptiveKernel, ExactPredicates};
/// fn requires_exact<T: ExactPredicates<6>>() {}
/// requires_exact::<AdaptiveKernel<f64>>(); // ERROR: exact insphere is not available for D=6
/// ```
pub trait ExactPredicates<const D: usize>: Kernel<D> {}

macro_rules! impl_exact_predicates_for_supported_dims {
    ($($dim:literal),* $(,)?) => {
        $(
            impl<T: CoordinateScalar> ExactPredicates<$dim> for RobustKernel<T> {}
            impl<T: CoordinateScalar> ExactPredicates<$dim> for AdaptiveKernel<T> {}
        )*
    };
}

impl_exact_predicates_for_supported_dims!(0, 1, 2, 3, 4, 5);

/// Fast floating-point kernel.
///
/// Uses standard floating-point arithmetic for maximum performance.
/// May produce incorrect results for degenerate or near-degenerate cases.
///
/// For applications requiring guaranteed correctness in degenerate cases,
/// use [`AdaptiveKernel`] (the default) instead.
///
/// # ⚠️ Warning: Unreliable in 3D and Higher Dimensions
///
/// **`FastKernel` should not be used for bulk Delaunay triangulation in 3D or higher
/// dimensions.** Random point sets in 3D+ routinely produce near-co-spherical
/// configurations that cause `FastKernel`'s in-sphere predicate to misclassify
/// points, leading to incorrect conflict zones, invalid topology, and construction
/// failures.
///
/// Use [`AdaptiveKernel`] (the default) for all 3D+ work. `FastKernel` remains
/// suitable for 2D triangulations with well-conditioned input, or when explicitly
/// opted into via [`DelaunayTriangulation::with_kernel`](crate::triangulation::delaunay::DelaunayTriangulation::with_kernel) for advanced use cases
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
/// use delaunay::prelude::geometry::{FastKernel, Kernel};
/// use delaunay::prelude::geometry::Point;
/// use delaunay::prelude::geometry::Coordinate;
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
pub struct FastKernel<T> {
    _phantom: PhantomData<T>,
}

impl<T> FastKernel<T> {
    /// Create a new fast kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::geometry::FastKernel;
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
    T: CoordinateScalar,
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
                CellValidationError::CoordinateConversion { source } => source,
                CellValidationError::InsufficientVertices {
                    actual,
                    expected,
                    dimension,
                } => CoordinateConversionError::InvalidSimplexPointCount {
                    actual,
                    expected,
                    dimension,
                },
                CellValidationError::DegenerateSimplex => {
                    CoordinateConversionError::DegenerateSimplex {
                        dimension: D,
                        reason: DegenerateSimplexReason::ZeroOrientation,
                    }
                }
                _ => CoordinateConversionError::ConversionFailed {
                    coordinate_index: 0,
                    coordinate_value: format!("{e}"),
                    from_type: "insphere_lifted",
                    to_type: "in_sphere",
                },
            }
        })?;
        Ok(insphere_to_i32(result))
    }

    fn in_sphere_positive_oriented(
        &self,
        simplex_points: &[Point<Self::Scalar, D>],
        test_point: &Point<Self::Scalar, D>,
    ) -> Result<i32, CoordinateConversionError> {
        let result = robust_insphere_positive_oriented(simplex_points, test_point)?;
        Ok(insphere_to_i32(result))
    }
}

/// Robust exact-arithmetic kernel.
///
/// Uses exact Bareiss arithmetic backed by provable error bounds.
/// Slower than [`FastKernel`] but provides robust numerical stability.
///
/// # When to use `RobustKernel` over [`AdaptiveKernel`]
///
/// Prefer `RobustKernel` when you need:
/// - **Explicit degeneracy signals** — returns `DEGENERATE`/`BOUNDARY` (`0`)
///   instead of forcing a decision, useful when your application needs to
///   detect and handle cospherical or coplanar configurations directly
/// - **Opt-in diagnostic consistency checks** — cross-validates insphere results
///   against a distance-based check when `DELAUNAY_STRICT_INSPHERE_CONSISTENCY`
///   is set
///
/// For standard Delaunay triangulation, [`AdaptiveKernel`] is the better
/// default: zero configuration, provable error bounds, and `SoS`
/// tie-breaking on insphere eliminates `BOUNDARY` ambiguity.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::geometry::{RobustKernel, Kernel};
/// use delaunay::prelude::geometry::Point;
/// use delaunay::prelude::geometry::Coordinate;
///
/// let kernel = RobustKernel::<f64>::new();
///
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
/// let test_point = Point::new([0.25, 0.25, 0.25]);
/// let result = kernel.in_sphere(&points, &test_point).unwrap();
/// assert_eq!(result, 1); // Inside circumsphere
/// ```
#[derive(Clone, Default, Debug)]
pub struct RobustKernel<T> {
    _phantom: PhantomData<T>,
}

impl<T> RobustKernel<T> {
    /// Create a new robust kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::geometry::RobustKernel;
    ///
    /// let kernel = RobustKernel::<f64>::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T, const D: usize> Kernel<D> for RobustKernel<T>
where
    T: CoordinateScalar,
{
    type Scalar = T;

    fn orientation(
        &self,
        points: &[Point<Self::Scalar, D>],
    ) -> Result<i32, CoordinateConversionError> {
        let result = robust_orientation(points)?;
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
        let result = robust_insphere(simplex_points, test_point)?;
        Ok(insphere_to_i32(result))
    }

    fn in_sphere_positive_oriented(
        &self,
        simplex_points: &[Point<Self::Scalar, D>],
        test_point: &Point<Self::Scalar, D>,
    ) -> Result<i32, CoordinateConversionError> {
        let result = robust_insphere_positive_oriented(simplex_points, test_point)?;
        Ok(insphere_to_i32(result))
    }
}

/// Adaptive precision kernel with Simulation of Simplicity.
///
/// This is the **default kernel** for [`DelaunayTriangulation`] convenience
/// constructors (`new`, `empty`, `new_with_options`, etc.).
///
/// [`DelaunayTriangulation`]: crate::triangulation::delaunay::DelaunayTriangulation
///
/// # When to use `AdaptiveKernel`
///
/// Use this kernel (the default) for Delaunay triangulation. It provides:
/// - **Zero configuration** — no tolerance to tune or get wrong
/// - **Provable error bounds** on the fast filter (no heuristic tolerance)
/// - **`SoS` orientation** — degenerate ties are broken deterministically;
///   returns ±1 for all distinct-point inputs (returns 0 only when points
///   are identical in `f64` representation)
/// - **`SoS` insphere** — cospherical ties are broken deterministically,
///   so every insphere query returns ±1 (never 0/BOUNDARY)
///
/// If you need configurable tolerance, explicit `BOUNDARY`/`DEGENERATE`
/// signals, or diagnostic consistency checks, use [`RobustKernel`] via
/// the explicit-kernel constructors (`with_kernel`, `build_with_kernel`).
///
/// # Evaluation strategy
///
/// **Orientation** (exact + `SoS` tie-breaking):
/// 1. **Fast filter**: `det_direct()` + `det_errbound()` (provable for D ≤ 4)
/// 2. **Exact arithmetic**: `det_sign_exact()` via Bareiss algorithm in `BigRational`
/// 3. **`SoS` tie-breaking**: Simulation of Simplicity for degenerate cases
///
/// **Insphere** (exact + `SoS` tie-breaking):
/// 1. **Fast filter** + **exact arithmetic** (same as orientation)
/// 2. **`SoS` tie-breaking**: Simulation of Simplicity for cospherical cases
///
/// # Examples
///
/// ```
/// use delaunay::prelude::geometry::{AdaptiveKernel, Kernel};
/// use delaunay::prelude::geometry::Point;
/// use delaunay::prelude::geometry::Coordinate;
///
/// let kernel = AdaptiveKernel::<f64>::new();
///
/// // Collinear points get a deterministic SoS sign (never 0)
/// let collinear = [
///     Point::new([0.0, 0.0]),
///     Point::new([1.0, 0.0]),
///     Point::new([2.0, 0.0]),
/// ];
/// let orientation = kernel.orientation(&collinear).unwrap();
/// assert!(orientation == 1 || orientation == -1); // SoS: always non-zero
/// ```
#[derive(Clone, Default, Debug)]
pub struct AdaptiveKernel<T> {
    _phantom: PhantomData<T>,
}

impl<T> AdaptiveKernel<T> {
    /// Create a new adaptive kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use delaunay::prelude::geometry::AdaptiveKernel;
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
    T: CoordinateScalar,
{
    type Scalar = T;

    fn orientation(
        &self,
        points: &[Point<Self::Scalar, D>],
    ) -> Result<i32, CoordinateConversionError> {
        if points.len() != D + 1 {
            return Err(CoordinateConversionError::InvalidSimplexPointCount {
                actual: points.len(),
                expected: D + 1,
                dimension: D,
            });
        }

        // Layer 1+2: exact sign via fast filter + Bareiss in BigRational.
        // Delegates to robust_orientation to avoid duplicating the homogeneous
        // matrix build + exact_det_sign pipeline.
        let exact = robust_orientation(points)?;
        match exact {
            Orientation::POSITIVE => return Ok(1),
            Orientation::NEGATIVE => return Ok(-1),
            Orientation::DEGENERATE => {}
        }

        // Layer 3: SoS tie-breaking for truly degenerate orientation.
        // Same pattern as in_sphere() — convert to f64 points for SoS.
        let mut f64_points: SmallBuffer<Point<f64, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(points.len());
        for point in points {
            f64_points.push(Point::new(safe_coords_to_f64(point.coords())?));
        }

        // SoS guarantees a non-zero sign for distinct points.  If SoS
        // fails (all cofactors vanish) the points are identical in f64
        // representation — a true degeneracy that cannot be resolved
        // symbolically.  Return 0 so callers' existing degenerate-
        // orientation handling applies.
        sos_orientation_sign(&f64_points).map_or(Ok(0), Ok)
    }

    fn in_sphere(
        &self,
        simplex_points: &[Point<Self::Scalar, D>],
        test_point: &Point<Self::Scalar, D>,
    ) -> Result<i32, CoordinateConversionError> {
        if simplex_points.len() != D + 1 {
            return Err(CoordinateConversionError::InvalidSimplexPointCount {
                actual: simplex_points.len(),
                expected: D + 1,
                dimension: D,
            });
        }

        let signs = relative_insphere_signs(simplex_points, test_point)?;
        let rel_orient_sign = signs.relative_orientation;
        let insphere_det_sign = signs.insphere_determinant;

        // Fast path: both non-degenerate.
        if rel_orient_sign != 0 && insphere_det_sign != 0 {
            return Ok(relative_insphere_effective_sign(signs));
        }

        // At least one predicate needs SoS → convert to f64 points.
        let mut f64_simplex: SmallBuffer<Point<f64, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
            SmallBuffer::with_capacity(simplex_points.len());
        for point in simplex_points {
            f64_simplex.push(Point::new(safe_coords_to_f64(point.coords())?));
        }
        let f64_test = Point::new(safe_coords_to_f64(test_point.coords())?);

        // Resolve orientation factor.
        let orient_factor: i32 = if rel_orient_sign != 0 {
            -rel_orient_sign
        } else {
            // Orientation degenerate → SoS gives absolute orientation sign.
            // rel_orient = (-1)^D × abs_orient
            // orient_factor = -rel_orient = (-1)^(D+1) × abs_orient
            let sos_abs = sos_orientation_sign(&f64_simplex)?;
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
            sos_insphere_sign(&f64_simplex, &f64_test)?
        };

        Ok((insphere_effective * orient_factor).signum())
    }

    fn in_sphere_positive_oriented(
        &self,
        simplex_points: &[Point<Self::Scalar, D>],
        test_point: &Point<Self::Scalar, D>,
    ) -> Result<i32, CoordinateConversionError> {
        if simplex_points.len() != D + 1 {
            return Err(CoordinateConversionError::InvalidSimplexPointCount {
                actual: simplex_points.len(),
                expected: D + 1,
                dimension: D,
            });
        }

        let determinant_sign = relative_insphere_determinant_sign(simplex_points, test_point)?;
        let orient_factor = if D.is_multiple_of(2) { -1 } else { 1 };
        let insphere_effective = if determinant_sign != 0 {
            determinant_sign
        } else {
            let mut f64_simplex: SmallBuffer<Point<f64, D>, MAX_PRACTICAL_DIMENSION_SIZE> =
                SmallBuffer::with_capacity(simplex_points.len());
            for point in simplex_points {
                f64_simplex.push(Point::new(safe_coords_to_f64(point.coords())?));
            }
            let f64_test = Point::new(safe_coords_to_f64(test_point.coords())?);
            sos_insphere_sign(&f64_simplex, &f64_test)?
        };

        Ok((insphere_effective * orient_factor).signum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;

    // =========================================================================
    // GENERIC HELPER FUNCTIONS
    // =========================================================================

    /// Standard D-simplex: origin + D unit vectors (non-degenerate).
    fn standard_simplex<const D: usize>() -> Vec<Point<f64, D>> {
        let mut points = Vec::with_capacity(D + 1);
        points.push(Point::new([0.0; D]));
        for i in 0..D {
            let mut coords = [0.0; D];
            coords[i] = 1.0;
            points.push(Point::new(coords));
        }
        points
    }

    /// Degenerate D-simplex: all points have last coordinate = 0.
    fn degenerate_simplex<const D: usize>() -> Vec<Point<f64, D>> {
        let mut points = Vec::with_capacity(D + 1);
        points.push(Point::new([0.0; D]));
        for i in 0..D.saturating_sub(1) {
            let mut coords = [0.0; D];
            coords[i] = 1.0;
            points.push(Point::new(coords));
        }
        let mut bary = [0.0; D];
        for c in bary.iter_mut().take(D.saturating_sub(1)) {
            *c = 0.5;
        }
        points.push(Point::new(bary));
        points
    }

    /// Point clearly inside the circumsphere of the standard simplex.
    fn inside_point<const D: usize>() -> Point<f64, D> {
        Point::new([0.1; D])
    }

    /// Point clearly outside the circumsphere of the standard simplex.
    fn outside_point<const D: usize>() -> Point<f64, D> {
        Point::new([2.0; D])
    }

    /// Co-spherical test point: (1,1,…,1) lies on the circumsphere of the
    /// standard simplex for all D ≥ 2.
    fn cospherical_test<const D: usize>() -> Point<f64, D> {
        Point::new([1.0; D])
    }

    /// Test point off the degenerate hyperplane (last coord nonzero).
    fn off_plane_test<const D: usize>() -> Point<f64, D> {
        let mut coords = [0.0; D];
        coords[D - 1] = 1.0;
        Point::new(coords)
    }

    /// Test point in the degenerate hyperplane but far from the simplex.
    fn coplanar_far_test<const D: usize>() -> Point<f64, D> {
        let mut coords = [0.0; D];
        coords[0] = 3.0;
        Point::new(coords)
    }

    // =========================================================================
    // MACRO — FastKernel + RobustKernel PER-DIMENSION TESTS (2D–5D)
    // =========================================================================

    /// Generate orientation + insphere tests for a standard (non-`SoS`) kernel.
    macro_rules! gen_standard_kernel_tests {
        ($dim:literal, $name:ident, $kernel:expr) => {
            pastey::paste! {
                #[test]
                fn [<test_ $name _orientation_ $dim d_nondegenerate>]() {
                    let kernel = $kernel;
                    let simplex = standard_simplex::<$dim>();
                    let result = kernel.orientation(&simplex).unwrap();
                    assert!(
                        result == 1 || result == -1,
                        "expected ±1, got {result}"
                    );
                }

                #[test]
                fn [<test_ $name _orientation_ $dim d_degenerate>]() {
                    let kernel = $kernel;
                    let simplex = degenerate_simplex::<$dim>();
                    let result = kernel.orientation(&simplex).unwrap();
                    assert_eq!(result, 0, "degenerate simplex must give 0");
                }

                #[test]
                fn [<test_ $name _insphere_ $dim d_inside>]() {
                    let kernel = $kernel;
                    let simplex = standard_simplex::<$dim>();
                    let test = inside_point::<$dim>();
                    let result = kernel.in_sphere(&simplex, &test).unwrap();
                    assert_eq!(result, 1, "point should be INSIDE");
                }

                #[test]
                fn [<test_ $name _insphere_ $dim d_outside>]() {
                    let kernel = $kernel;
                    let simplex = standard_simplex::<$dim>();
                    let test = outside_point::<$dim>();
                    let result = kernel.in_sphere(&simplex, &test).unwrap();
                    assert_eq!(result, -1, "point should be OUTSIDE");
                }
            }
        };
    }

    gen_standard_kernel_tests!(2, fast, FastKernel::<f64>::new());
    gen_standard_kernel_tests!(3, fast, FastKernel::<f64>::new());
    gen_standard_kernel_tests!(4, fast, FastKernel::<f64>::new());
    gen_standard_kernel_tests!(5, fast, FastKernel::<f64>::new());

    gen_standard_kernel_tests!(2, robust, RobustKernel::<f64>::new());
    gen_standard_kernel_tests!(3, robust, RobustKernel::<f64>::new());
    gen_standard_kernel_tests!(4, robust, RobustKernel::<f64>::new());
    gen_standard_kernel_tests!(5, robust, RobustKernel::<f64>::new());

    // =========================================================================
    // NON-MACRO — EDGE CASES AND SPECIAL CONFIGURATIONS
    // =========================================================================

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
    fn test_kernel_default_trait() {
        // Test that both kernels implement Default (required for simplex validation)
        let _fast: FastKernel<f64> = FastKernel::default();
        let _robust: RobustKernel<f64> = RobustKernel::default();
    }

    #[test]
    fn test_fast_kernel_in_sphere_insufficient_vertices() {
        // Exercises the CellValidationError::InsufficientVertices mapping.
        let kernel = FastKernel::<f64>::new();
        let simplex: [Point<f64, 3>; 2] =
            [Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])];
        let test_point = Point::new([0.5, 0.5, 0.5]);
        let result = kernel.in_sphere(&simplex, &test_point);
        assert_eq!(
            result,
            Err(CoordinateConversionError::InvalidSimplexPointCount {
                actual: 2,
                expected: 4,
                dimension: 3,
            })
        );
    }

    #[test]
    fn test_fast_kernel_in_sphere_degenerate_simplex() {
        // Exercises the CellValidationError::DegenerateSimplex mapping.
        let kernel = FastKernel::<f64>::new();
        let simplex = [
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([1.0, 1.0, 0.0]), // Coplanar — degenerate
        ];
        let test_point = Point::new([0.5, 0.5, 0.5]);
        let result = kernel.in_sphere(&simplex, &test_point);
        assert_eq!(
            result,
            Err(CoordinateConversionError::DegenerateSimplex {
                dimension: 3,
                reason: DegenerateSimplexReason::ZeroOrientation,
            })
        );
    }

    #[test]
    fn test_robust_kernel_positive_oriented_insphere_boundary_maps_to_zero() {
        let kernel = RobustKernel::<f64>::new();
        let simplex = [
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];

        assert_eq!(kernel.orientation(&simplex).unwrap(), 1);
        assert_eq!(
            kernel
                .in_sphere_positive_oriented(&simplex, &Point::new([1.0, 1.0]))
                .unwrap(),
            0
        );
    }

    #[test]
    fn test_robust_kernel_positive_oriented_insphere_wrong_arity_errors() {
        let kernel = RobustKernel::<f64>::new();
        let simplex: [Point<f64, 2>; 2] = [Point::new([0.0, 0.0]), Point::new([1.0, 0.0])];

        let err = kernel
            .in_sphere_positive_oriented(&simplex, &Point::new([0.25, 0.25]))
            .unwrap_err();

        assert_eq!(
            err,
            CoordinateConversionError::InvalidSimplexPointCount {
                actual: 2,
                expected: 3,
                dimension: 2,
            }
        );
    }

    // =========================================================================
    // MACRO — AdaptiveKernel PER-DIMENSION TESTS (2D–5D)
    // =========================================================================

    /// Generate `AdaptiveKernel` tests for a given dimension: agreement with
    /// `FastKernel` on non-degenerate inputs, `SoS` resolution of degenerate
    /// orientation and cospherical insphere, determinism, and branch coverage
    /// for orient-degenerate and both-degenerate insphere paths.
    macro_rules! gen_adaptive_kernel_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_adaptive_orientation_ $dim d_agrees>]() {
                    let adaptive = AdaptiveKernel::<f64>::new();
                    let fast = FastKernel::<f64>::new();
                    let simplex = standard_simplex::<$dim>();
                    assert_eq!(
                        adaptive.orientation(&simplex).unwrap(),
                        fast.orientation(&simplex).unwrap(),
                        "AdaptiveKernel must agree with FastKernel on non-degenerate"
                    );
                }

                #[test]
                fn [<test_adaptive_orientation_ $dim d_degenerate_sos_nonzero>]() {
                    // SoS orientation returns ±1 for degenerate inputs (never 0).
                    let kernel = AdaptiveKernel::<f64>::new();
                    let simplex = degenerate_simplex::<$dim>();
                    let result = kernel.orientation(&simplex).unwrap();
                    assert!(
                        result == 1 || result == -1,
                        "AdaptiveKernel SoS orientation must never return 0, got {result}"
                    );
                }

                #[test]
                fn [<test_adaptive_orientation_ $dim d_degenerate_deterministic>]() {
                    let kernel = AdaptiveKernel::<f64>::new();
                    let simplex = degenerate_simplex::<$dim>();
                    let results: Vec<i32> = (0..10)
                        .map(|_| kernel.orientation(&simplex).unwrap())
                        .collect();
                    assert!(
                        results.iter().all(|&r| r == results[0]),
                        "Degenerate SoS orientation must be deterministic"
                    );
                }

                #[test]
                fn [<test_adaptive_insphere_ $dim d_agrees>]() {
                    let adaptive = AdaptiveKernel::<f64>::new();
                    let fast = FastKernel::<f64>::new();
                    let simplex = standard_simplex::<$dim>();
                    let inside = inside_point::<$dim>();
                    assert_eq!(
                        adaptive.in_sphere(&simplex, &inside).unwrap(),
                        fast.in_sphere(&simplex, &inside).unwrap(),
                        "Must agree on clearly inside point"
                    );
                    let outside = outside_point::<$dim>();
                    assert_eq!(
                        adaptive.in_sphere(&simplex, &outside).unwrap(),
                        fast.in_sphere(&simplex, &outside).unwrap(),
                        "Must agree on clearly outside point"
                    );
                }

                #[test]
                fn [<test_adaptive_positive_oriented_insphere_ $dim d_agrees>]() {
                    let kernel = AdaptiveKernel::<f64>::new();
                    let mut simplex = standard_simplex::<$dim>();
                    if kernel.orientation(&simplex).unwrap() < 0 {
                        simplex.swap(0, 1);
                    }
                    assert_eq!(kernel.orientation(&simplex).unwrap(), 1);

                    for test in [
                        inside_point::<$dim>(),
                        outside_point::<$dim>(),
                        cospherical_test::<$dim>(),
                    ] {
                        assert_eq!(
                            kernel.in_sphere_positive_oriented(&simplex, &test).unwrap(),
                            kernel.in_sphere(&simplex, &test).unwrap(),
                            "positive-oriented fast path must preserve AdaptiveKernel semantics"
                        );
                    }
                }

                #[test]
                fn [<test_adaptive_insphere_ $dim d_cospherical_nonzero>]() {
                    let kernel = AdaptiveKernel::<f64>::new();
                    let simplex = standard_simplex::<$dim>();
                    let test = cospherical_test::<$dim>();
                    let result = kernel.in_sphere(&simplex, &test).unwrap();
                    assert!(
                        result == 1 || result == -1,
                        "AdaptiveKernel insphere must never return 0, got {result}"
                    );
                }

                #[test]
                fn [<test_adaptive_insphere_ $dim d_cospherical_deterministic>]() {
                    let kernel = AdaptiveKernel::<f64>::new();
                    let simplex = standard_simplex::<$dim>();
                    let test = cospherical_test::<$dim>();
                    let results: Vec<i32> = (0..10)
                        .map(|_| kernel.in_sphere(&simplex, &test).unwrap())
                        .collect();
                    assert!(
                        results.iter().all(|&r| r == results[0]),
                        "Degenerate insphere must be deterministic"
                    );
                }

                #[test]
                fn [<test_adaptive_insphere_ $dim d_orient_degenerate>]() {
                    // Degenerate simplex + off-plane test → orientation is
                    // degenerate but insphere determinant may not be.
                    let kernel = AdaptiveKernel::<f64>::new();
                    let simplex = degenerate_simplex::<$dim>();
                    let test = off_plane_test::<$dim>();
                    let result = kernel.in_sphere(&simplex, &test).unwrap();
                    assert!(
                        result == 1 || result == -1,
                        "Must resolve orient-degenerate insphere, got {result}"
                    );
                }

                #[test]
                fn [<test_adaptive_insphere_ $dim d_both_degenerate>]() {
                    // Degenerate simplex + coplanar test → both orientation
                    // and insphere determinants may be zero.
                    let kernel = AdaptiveKernel::<f64>::new();
                    let simplex = degenerate_simplex::<$dim>();
                    let test = coplanar_far_test::<$dim>();
                    let result = kernel.in_sphere(&simplex, &test).unwrap();
                    assert!(
                        result == 1 || result == -1,
                        "Must resolve both-degenerate insphere, got {result}"
                    );
                }
            }
        };
    }

    gen_adaptive_kernel_tests!(2);
    gen_adaptive_kernel_tests!(3);
    gen_adaptive_kernel_tests!(4);
    gen_adaptive_kernel_tests!(5);

    // =========================================================================
    // MACRO — Kernel Consistency Across All Three Implementations (2D–5D)
    // =========================================================================

    /// Verify that `FastKernel`, `RobustKernel`, and `AdaptiveKernel` agree on
    /// clearly non-degenerate inputs.
    macro_rules! gen_kernel_consistency_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_kernel_consistency_ $dim d_orientation>]() {
                    let fast = FastKernel::<f64>::new();
                    let robust = RobustKernel::<f64>::new();
                    let adaptive = AdaptiveKernel::<f64>::new();
                    let simplex = standard_simplex::<$dim>();
                    let f = fast.orientation(&simplex).unwrap();
                    let r = robust.orientation(&simplex).unwrap();
                    let a = adaptive.orientation(&simplex).unwrap();
                    assert_eq!(f, r, "Fast vs Robust");
                    assert_eq!(f, a, "Fast vs Adaptive");
                }

                #[test]
                fn [<test_kernel_consistency_ $dim d_insphere>]() {
                    let fast = FastKernel::<f64>::new();
                    let robust = RobustKernel::<f64>::new();
                    let adaptive = AdaptiveKernel::<f64>::new();
                    let simplex = standard_simplex::<$dim>();
                    let test = inside_point::<$dim>();
                    let f = fast.in_sphere(&simplex, &test).unwrap();
                    let r = robust.in_sphere(&simplex, &test).unwrap();
                    let a = adaptive.in_sphere(&simplex, &test).unwrap();
                    assert_eq!(f, r, "Fast vs Robust");
                    assert_eq!(f, a, "Fast vs Adaptive");
                }
            }
        };
    }

    gen_kernel_consistency_tests!(2);
    gen_kernel_consistency_tests!(3);
    gen_kernel_consistency_tests!(4);
    gen_kernel_consistency_tests!(5);

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

    // =========================================================================
    // SoS IDENTICAL-POINTS REGRESSION
    // =========================================================================

    /// When all D+1 input points are identical in f64, every `SoS` cofactor
    /// vanishes and `sos_orientation_sign` returns `Err`.  `AdaptiveKernel`
    /// must map that to `Ok(0)` so callers' degenerate-orientation handling
    /// applies.  This is a regression guard for the fallback at kernel.rs:518.
    macro_rules! gen_sos_identical_points_test {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_adaptive_sos_identical_points_ $dim d>]() {
                    let kernel = AdaptiveKernel::<f64>::new();
                    let points: Vec<Point<f64, $dim>> =
                        vec![Point::new([0.42; $dim]); $dim + 1];
                    let result = kernel.orientation(&points).unwrap();
                    assert_eq!(
                        result, 0,
                        "{}D: identical points must yield orientation 0, got {result}",
                        $dim
                    );
                }
            }
        };
    }

    gen_sos_identical_points_test!(2);
    gen_sos_identical_points_test!(3);
    gen_sos_identical_points_test!(4);
    gen_sos_identical_points_test!(5);

    // =========================================================================
    // ExactPredicates MARKER TRAIT — COMPILE-TIME ASSERTIONS
    // =========================================================================

    /// Helper that requires `T: ExactPredicates<D>` — compilation succeeds only
    /// for types that implement the marker trait in the requested dimension.
    const fn assert_exact_predicates<T, const D: usize>()
    where
        T: ExactPredicates<D>,
    {
    }

    #[test]
    fn test_adaptive_kernel_implements_exact_predicates() {
        assert_exact_predicates::<AdaptiveKernel<f64>, 2>();
        assert_exact_predicates::<AdaptiveKernel<f64>, 5>();
        assert_exact_predicates::<AdaptiveKernel<f32>, 2>();
        assert_exact_predicates::<AdaptiveKernel<f32>, 5>();
    }

    #[test]
    fn test_robust_kernel_implements_exact_predicates() {
        assert_exact_predicates::<RobustKernel<f64>, 2>();
        assert_exact_predicates::<RobustKernel<f64>, 5>();
        assert_exact_predicates::<RobustKernel<f32>, 2>();
        assert_exact_predicates::<RobustKernel<f32>, 5>();
    }

    /// Negative compile-time assertion: `FastKernel` must NOT implement
    /// `ExactPredicates`.  This is verified by the `compile_fail` doctest
    /// on [`ExactPredicates`] (see trait doc) and by the absence of an impl
    /// block; this test documents the intent.
    #[test]
    fn test_fast_kernel_does_not_implement_exact_predicates() {
        // If `FastKernel` ever gains an `ExactPredicates` impl, the
        // compile_fail doctest will break.  This test serves as a
        // human-readable reminder of the design invariant.
        fn _requires_exact<T: ExactPredicates<3>>() {}
        // Uncomment the next line to verify it fails to compile:
        // _requires_exact::<FastKernel<f64>>();
    }
}
