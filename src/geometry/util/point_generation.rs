//! Random point generation functions.
//!
//! This module provides utilities for generating random points in d-dimensional
//! space with various distribution strategies.

#![forbid(unsafe_code)]

use super::conversions::safe_usize_to_scalar;
use super::norms::hypot;
use crate::geometry::coordinate_range::{
    CoordinateRange, CoordinateRangeError, InvalidCoordinateValue,
};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateValidationError};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::{any::type_name, env, num::NonZeroUsize};

/// Reason a scalar that must be finite and positive failed validation.
///
/// This error is used by public generators whose raw scalar inputs are parsed
/// into positive internal values before sampling starts, such as grid spacing,
/// periodic-domain periods, and ball radii.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::generators::InvalidPositiveScalar;
///
/// let err = InvalidPositiveScalar::NonPositive { value: 0.0 };
/// std::assert_matches!(err, InvalidPositiveScalar::NonPositive { .. });
/// ```
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum InvalidPositiveScalar<T = f64> {
    /// The value is non-finite.
    #[error("non-finite value {value}")]
    NonFinite {
        /// The non-finite value category.
        value: InvalidCoordinateValue,
    },
    /// The finite value is zero or negative.
    #[error("non-positive value {value:?}")]
    NonPositive {
        /// The finite non-positive value.
        value: T,
    },
}

/// Errors that can occur during random point generation.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::generators::{
///     CoordinateRangeError, CoordinateRangeOrdering, InvalidPositiveScalar,
///     RandomPointGenerationError,
/// };
///
/// let err = RandomPointGenerationError::InvalidCoordinateRange {
///     source: CoordinateRangeError::NonIncreasing {
///         ordering: CoordinateRangeOrdering::Decreasing,
///         min: 1.0,
///         max: 0.0,
///     },
/// };
/// std::assert_matches!(
///     err,
///     RandomPointGenerationError::InvalidCoordinateRange { .. }
/// );
///
/// let grid_err = RandomPointGenerationError::InvalidGridSpacing {
///     reason: InvalidPositiveScalar::NonPositive { value: 0.0 },
/// };
/// std::assert_matches!(
///     grid_err,
///     RandomPointGenerationError::InvalidGridSpacing { .. }
/// );
/// ```
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum RandomPointGenerationError<T = f64> {
    /// Invalid coordinate range provided.
    #[error("Invalid coordinate range for random point generation: {source}")]
    InvalidCoordinateRange {
        /// The coordinate range validation failure.
        #[from]
        source: CoordinateRangeError<T>,
    },

    /// Validated coordinate bounds are too wide for floating-point uniform sampling.
    #[error("Coordinate range width overflows for random point generation: {min:?}..{max:?}")]
    CoordinateRangeWidthOverflow {
        /// Lower sampling bound.
        min: T,
        /// Upper sampling bound.
        max: T,
    },

    /// Invalid Poisson disk minimum distance provided.
    #[error("Invalid minimum distance for Poisson point generation: {distance} must be finite")]
    InvalidMinimumDistance {
        /// The non-finite minimum distance.
        distance: InvalidCoordinateValue,
    },

    /// Invalid periodic domain period provided.
    #[error(
        "Invalid periodic domain period at axis {axis}: {reason}; period must be finite and positive"
    )]
    InvalidPeriodicDomain {
        /// Axis whose period failed validation.
        axis: usize,
        /// Why the period failed validation.
        reason: InvalidPositiveScalar<T>,
    },

    /// Invalid ball radius provided.
    #[error("Invalid ball radius: {reason}; radius must be finite and positive")]
    InvalidBallRadius {
        /// Why the radius failed validation.
        reason: InvalidPositiveScalar<T>,
    },

    /// Squaring a validated finite ball radius produced a non-finite value.
    #[error("Invalid squared ball radius: {value}; squared radius must be finite")]
    InvalidBallRadiusSquared {
        /// The non-finite squared radius value.
        value: InvalidCoordinateValue,
    },

    /// Ball rejection sampling could not produce the requested point count.
    #[error(
        "Could not generate {requested_points} ball points in dimension {dimension} with radius {radius:?} after {attempts} attempts; generated {generated_points}"
    )]
    BallSamplingFailed {
        /// Number of points requested.
        requested_points: usize,
        /// Number of points generated before attempts were exhausted.
        generated_points: usize,
        /// Ball dimension.
        dimension: usize,
        /// Ball radius used by the sampler.
        radius: T,
        /// Number of candidate samples attempted.
        attempts: usize,
    },

    /// Failed to convert a discrete count, index, or scalar into a numeric target type.
    #[error("Failed to convert {value} to {target_type}: {source}")]
    CoordinateConversionFailed {
        /// The value that failed to convert.
        value: usize,
        /// Target numeric type.
        target_type: &'static str,
        /// The coordinate conversion failure.
        source: CoordinateConversionError,
    },

    /// Generated coordinates failed the point validation boundary.
    #[error("Generated point coordinates were invalid: {source}")]
    GeneratedPointCoordinateRejected {
        /// The coordinate validation failure.
        source: CoordinateValidationError,
    },

    /// Requested grid dimensions overflowed `usize`.
    #[error("Requested grid size {points_per_dim}^{dimension} overflows usize point count")]
    GridSizeOverflow {
        /// Number of points requested along each grid axis.
        points_per_dim: usize,
        /// Grid dimension.
        dimension: usize,
    },

    /// Requested grid allocation exceeds the configured safety cap.
    #[error(
        "Requested grid allocation {required_bytes} bytes exceeds safety cap {cap_bytes} bytes"
    )]
    GridAllocationTooLarge {
        /// Estimated bytes required by the requested grid.
        required_bytes: usize,
        /// Configured safety cap in bytes.
        cap_bytes: usize,
    },

    /// Grid spacing was non-finite or non-positive.
    #[error("Invalid grid spacing: {reason}; spacing must be finite and positive")]
    InvalidGridSpacing {
        /// Why the spacing failed validation.
        reason: InvalidPositiveScalar<T>,
    },

    /// Grid offset was non-finite.
    #[error("Invalid grid offset at axis {axis}: {value}; offset coordinates must be finite")]
    InvalidGridOffset {
        /// Axis whose offset coordinate failed validation.
        axis: usize,
        /// The non-finite offset value.
        value: InvalidCoordinateValue,
    },

    /// A generated grid coordinate was non-finite.
    #[error("Invalid generated grid coordinate at axis {axis}: {value}")]
    InvalidGeneratedGridCoordinate {
        /// Axis whose generated coordinate failed validation.
        axis: usize,
        /// The non-finite generated coordinate value.
        value: InvalidCoordinateValue,
    },

    /// Poisson disk sampling could not satisfy the requested spacing.
    #[error(
        "Could not generate {requested_points} Poisson points in {bounds} with minimum distance {min_distance:?} after {attempts} attempts; generated {generated_points}"
    )]
    PoissonSamplingFailed {
        /// Number of points requested.
        requested_points: usize,
        /// Number of points generated before attempts were exhausted.
        generated_points: usize,
        /// Minimum distance used by the sampler.
        min_distance: T,
        /// Validated coordinate bounds used by the sampler.
        bounds: CoordinateRange<T>,
        /// Number of candidate samples attempted.
        attempts: usize,
    },
}

/// Default maximum bytes allowed for grid allocation to prevent OOM in CI.
///
/// This default safety cap prevents excessive memory allocation when generating grid points.
/// The limit of 4 GiB provides reasonable headroom for modern systems (GitHub Actions
/// runners have 7GB) while still protecting against extreme allocations.
///
/// The actual cap can be overridden via the `MAX_GRID_BYTES_SAFETY_CAP` environment variable.
const MAX_GRID_BYTES_SAFETY_CAP_DEFAULT: usize = 4_294_967_296; // 4 GiB

/// Base number of Poisson disk sampling attempts per requested point.
const POISSON_ATTEMPTS_PER_POINT: usize = 30;

/// Base number of ball rejection-sampling attempts per requested point.
const BALL_ATTEMPTS_PER_POINT: usize = 1_024;

/// Get the maximum bytes allowed for grid allocation.
///
/// Reads the `MAX_GRID_BYTES_SAFETY_CAP` environment variable if set,
/// otherwise returns the default value of 4 GiB. This allows CI environments
/// with different memory limits to tune the safety cap as needed.
///
/// # Returns
///
/// The maximum number of bytes allowed for grid allocation
fn max_grid_bytes_safety_cap() -> usize {
    if let Ok(v) = env::var("MAX_GRID_BYTES_SAFETY_CAP")
        && let Ok(n) = v.parse::<usize>()
    {
        return n;
    }
    MAX_GRID_BYTES_SAFETY_CAP_DEFAULT
}

/// Compute symmetric coordinate bounds scaled by point count.
///
/// This helper is intended for generating random point sets for triangulation construction
/// without "cramming" many points into a tiny bounding box like `[-1, 1]^D`, which can
/// increase the likelihood of near-degenerate configurations when using absolute tolerances.
///
/// It returns symmetric bounds `[-s/2, +s/2]` where `s = max(1, n_points)`.
///
/// # Errors
///
/// Returns [`RandomPointGenerationError::CoordinateConversionFailed`] if
/// `n_points` cannot be converted to the coordinate type `T` without loss of
/// precision.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::generators::{
///     RandomPointGenerationError, scaled_bounds_by_point_count,
/// };
///
/// # fn main() -> Result<(), RandomPointGenerationError> {
/// // 100 points -> side length 100, i.e. [-50, 50].
/// let bounds = scaled_bounds_by_point_count(100)?;
/// assert_eq!(bounds.bounds(), (-50.0, 50.0));
/// # Ok(())
/// # }
/// ```
pub fn scaled_bounds_by_point_count(
    n_points: usize,
) -> Result<CoordinateRange<f64>, RandomPointGenerationError> {
    let side_len = n_points.max(1);
    let side = safe_usize_to_scalar(side_len).map_err(|source| {
        RandomPointGenerationError::CoordinateConversionFailed {
            value: side_len,
            target_type: type_name::<f64>(),
            source,
        }
    })?;

    let half = side / 2.0;
    Ok(CoordinateRange::from_validated_bounds(-half, half))
}

/// Converts generated coordinates through the public [`Point`] validation boundary.
fn point_from_generated_coords<const D: usize>(
    coords: [f64; D],
) -> Result<Point<D>, RandomPointGenerationError> {
    Point::try_new(coords)
        .map_err(|source| RandomPointGenerationError::GeneratedPointCoordinateRejected { source })
}

/// Coordinate bounds proven finite and safe for `rand::Rng::random_range`.
#[derive(Clone, Copy, Debug)]
struct SamplerRange {
    range: CoordinateRange<f64>,
}

impl SamplerRange {
    /// Parses validated coordinate bounds into sampler-safe bounds.
    fn try_new(range: CoordinateRange<f64>) -> Result<Self, RandomPointGenerationError> {
        let width = range.max() - range.min();
        if width.is_finite() {
            Ok(Self { range })
        } else {
            Err(RandomPointGenerationError::CoordinateRangeWidthOverflow {
                min: range.min(),
                max: range.max(),
            })
        }
    }

    /// Returns the lower sampling bound.
    fn min(self) -> f64 {
        self.range.min()
    }

    /// Returns the upper sampling bound.
    fn max(self) -> f64 {
        self.range.max()
    }
}

/// Samples one point from an already-validated and sampler-safe coordinate range.
fn sample_point_in_range<R, const D: usize>(
    range: SamplerRange,
    rng: &mut R,
) -> Result<Point<D>, RandomPointGenerationError>
where
    R: rand::Rng + ?Sized,
{
    let coords = [0.0; D].map(|_| rng.random_range(range.min()..range.max()));
    point_from_generated_coords(coords)
}

/// Generates points after proving the validated range is safe for uniform sampling.
fn generate_random_points_in_range_with_rng<R, const D: usize>(
    n_points: usize,
    range: CoordinateRange<f64>,
    rng: &mut R,
) -> Result<Vec<Point<D>>, RandomPointGenerationError>
where
    R: rand::Rng + ?Sized,
{
    let range = SamplerRange::try_new(range)?;
    let mut points = Vec::with_capacity(n_points);

    for _ in 0..n_points {
        points.push(sample_point_in_range(range, rng)?);
    }

    Ok(points)
}

/// Parsed Poisson disk spacing policy for public Poisson generators.
///
/// Non-positive finite distances intentionally disable the spacing constraint,
/// while positive distances carry a [`PositiveScalar`] proof into the sampler.
#[derive(Clone, Copy, Debug, PartialEq)]
enum PoissonSpacing {
    /// No minimum-distance constraint should be enforced.
    Disabled,
    /// A finite, positive minimum distance.
    Minimum(PositiveScalar),
}

/// Scalar proven to be finite and strictly positive.
///
/// This private proof type backs public grid-spacing, periodic-domain, and
/// ball-radius error contracts so sampling code can consume positive values
/// without repeating raw scalar validation.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PositiveScalar(f64);

impl PositiveScalar {
    /// Parses a raw scalar into a finite positive scalar proof.
    ///
    /// The returned [`InvalidPositiveScalar`] is embedded directly in public
    /// [`RandomPointGenerationError`] variants by the caller.
    fn try_new(value: f64) -> Result<Self, InvalidPositiveScalar> {
        if !value.is_finite() {
            return Err(InvalidPositiveScalar::NonFinite {
                value: InvalidCoordinateValue::from_debug(&value),
            });
        }

        if value <= 0.0 {
            Err(InvalidPositiveScalar::NonPositive { value })
        } else {
            Ok(Self(value))
        }
    }

    /// Returns the proven finite positive scalar.
    const fn get(self) -> f64 {
        self.0
    }

    /// Builds symmetric coordinate bounds from a positive radius.
    ///
    /// The positive-radius proof guarantees `-radius < radius`, making the
    /// internal [`CoordinateRange`] construction infallible.
    fn symmetric_range(self) -> CoordinateRange<f64> {
        let radius = self.get();
        CoordinateRange::from_validated_bounds(-radius, radius)
    }
}

impl PoissonSpacing {
    /// Parses raw Poisson disk spacing before the sampler starts.
    fn try_new(min_distance: f64) -> Result<Self, RandomPointGenerationError> {
        if !min_distance.is_finite() {
            return Err(RandomPointGenerationError::InvalidMinimumDistance {
                distance: InvalidCoordinateValue::from_debug(&min_distance),
            });
        }

        if min_distance <= 0.0 {
            Ok(Self::Disabled)
        } else {
            Ok(Self::Minimum(PositiveScalar(min_distance)))
        }
    }
}

/// Scales Poisson disk sampling attempts by dimension.
const fn poisson_dimension_attempt_scaling(dimension: usize) -> usize {
    match dimension {
        0..=2 => 1,
        3..=4 => 2,
        5..=6 => 4,
        _ => 8,
    }
}

/// Computes the Poisson disk sampling attempt budget without panicking on large inputs.
const fn poisson_max_attempts(n_points: usize, dimension: usize) -> usize {
    n_points
        .saturating_mul(POISSON_ATTEMPTS_PER_POINT)
        .saturating_mul(poisson_dimension_attempt_scaling(dimension))
}

/// Computes the ball rejection-sampling attempt budget without panicking on large inputs.
const fn ball_max_attempts(n_points: usize, dimension: usize) -> usize {
    n_points
        .saturating_mul(BALL_ATTEMPTS_PER_POINT)
        .saturating_mul(poisson_dimension_attempt_scaling(dimension))
}

/// Generate random points in D-dimensional space with uniform distribution.
///
/// This function provides a flexible way to generate random points for testing,
/// benchmarking, or example applications. Points are generated with coordinates
/// uniformly distributed within the specified range.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate
/// * `range` - Range for coordinate values (min, max)
///
/// # Returns
///
/// Vector of random points with coordinates in the specified range, or a
/// [`RandomPointGenerationError`] if the parameters are invalid.
///
/// # Errors
///
/// * [`RandomPointGenerationError::InvalidCoordinateRange`] if either bound is
///   non-finite, equal, decreasing, or incomparable.
/// * [`RandomPointGenerationError::CoordinateRangeWidthOverflow`] if the
///   validated bounds are too wide for uniform floating-point sampling.
/// * [`RandomPointGenerationError::GeneratedPointCoordinateRejected`] if a
///   generated coordinate is rejected while constructing a [`Point`].
///
/// # Examples
///
/// ```
/// use delaunay::prelude::generators::{
///     RandomPointGenerationError, try_generate_random_points,
/// };
///
/// # fn main() -> Result<(), RandomPointGenerationError> {
/// // Generate 100 random 2D points with coordinates in [-10.0, 10.0]
/// let points_2d = try_generate_random_points::<2>(100, (-10.0, 10.0))?;
/// assert_eq!(points_2d.len(), 100);
///
/// // Generate 3D points with coordinates in [0.0, 1.0] (unit cube)
/// let points_3d = try_generate_random_points::<3>(50, (0.0, 1.0))?;
/// assert_eq!(points_3d.len(), 50);
///
/// // Generate 4D points centered around origin
/// let points_4d = try_generate_random_points::<4>(25, (-1.0, 1.0))?;
/// assert_eq!(points_4d.len(), 25);
///
/// // Error handling
/// let result = try_generate_random_points::<2>(100, (10.0, -10.0));
/// std::assert_matches!(
///     result,
///     Err(RandomPointGenerationError::InvalidCoordinateRange { .. })
/// );
/// # Ok(())
/// # }
/// ```
pub fn try_generate_random_points<const D: usize>(
    n_points: usize,
    range: (f64, f64),
) -> Result<Vec<Point<D>>, RandomPointGenerationError> {
    #[cfg(debug_assertions)]
    if env::var_os("DELAUNAY_DEBUG_UNUSED_IMPORTS").is_some() {
        tracing::debug!(
            n_points,
            dimension = D,
            "point_generation::try_generate_random_points called"
        );
    }
    let range = CoordinateRange::try_from(range)?;

    let mut rng = rand::rng();
    let points = generate_random_points_in_range_with_rng(n_points, range, &mut rng)?;

    Ok(points)
}

/// Generate random points in D-dimensional space from validated coordinate bounds.
///
/// This variant accepts a [`CoordinateRange`] so callers that already parsed raw
/// bounds can carry that evidence into generation.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate.
/// * `range` - Validated coordinate bounds shared by every coordinate axis.
///
/// # Returns
///
/// Vector of random points with coordinates sampled from `range`.
///
/// # Errors
///
/// Returns [`RandomPointGenerationError::GeneratedPointCoordinateRejected`] if
/// a generated coordinate is rejected while constructing a [`Point`], or
/// [`RandomPointGenerationError::CoordinateRangeWidthOverflow`] if the validated
/// bounds are too wide for uniform floating-point sampling.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::generators::{
///     CoordinateRange, RandomPointGenerationError, generate_random_points_in_range,
/// };
///
/// # fn main() -> Result<(), RandomPointGenerationError> {
/// let range = CoordinateRange::try_new(-1.0_f64, 1.0)?;
/// let points = generate_random_points_in_range::<2>(8, range)?;
/// assert_eq!(points.len(), 8);
/// # Ok(())
/// # }
/// ```
pub fn generate_random_points_in_range<const D: usize>(
    n_points: usize,
    range: CoordinateRange<f64>,
) -> Result<Vec<Point<D>>, RandomPointGenerationError> {
    let mut rng = rand::rng();
    generate_random_points_in_range_with_rng(n_points, range, &mut rng)
}

/// Generate random points with a seeded RNG for reproducible results.
///
/// This function is useful when you need consistent point generation across
/// multiple runs for testing, benchmarking, or debugging purposes.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate
/// * `range` - Range for coordinate values (min, max)
/// * `seed` - Seed for the random number generator
///
/// # Returns
///
/// Vector of random points with coordinates in the specified range, or a
/// [`RandomPointGenerationError`] if the parameters are invalid.
///
/// # Errors
///
/// * [`RandomPointGenerationError::InvalidCoordinateRange`] if either bound is
///   non-finite, equal, decreasing, or incomparable.
/// * [`RandomPointGenerationError::CoordinateRangeWidthOverflow`] if the
///   validated bounds are too wide for uniform floating-point sampling.
/// * [`RandomPointGenerationError::GeneratedPointCoordinateRejected`] if a
///   generated coordinate is rejected while constructing a [`Point`].
///
/// # Examples
///
/// ```
/// use delaunay::prelude::generators::{
///     RandomPointGenerationError, try_generate_random_points_seeded,
/// };
///
/// # fn main() -> Result<(), RandomPointGenerationError> {
/// // Generate reproducible random points
/// let points1 = try_generate_random_points_seeded::<3>(100, (-5.0, 5.0), 42)?;
/// let points2 = try_generate_random_points_seeded::<3>(100, (-5.0, 5.0), 42)?;
/// assert_eq!(points1, points2); // Same seed produces identical results
///
/// // Different seeds produce different results
/// let points3 = try_generate_random_points_seeded::<3>(100, (-5.0, 5.0), 123)?;
/// assert_ne!(points1, points3);
///
/// // Common ranges - unit cube [0,1]
/// let unit_points = try_generate_random_points_seeded::<3>(50, (0.0, 1.0), 42)?;
///
/// // Centered around origin [-1,1]
/// let centered_points = try_generate_random_points_seeded::<3>(50, (-1.0, 1.0), 42)?;
/// # Ok(())
/// # }
/// ```
pub fn try_generate_random_points_seeded<const D: usize>(
    n_points: usize,
    range: (f64, f64),
    seed: u64,
) -> Result<Vec<Point<D>>, RandomPointGenerationError> {
    #[cfg(debug_assertions)]
    if env::var_os("DELAUNAY_DEBUG_UNUSED_IMPORTS").is_some() {
        tracing::debug!(
            n_points,
            dimension = D,
            seed,
            "point_generation::try_generate_random_points_seeded called"
        );
    }

    let range = CoordinateRange::try_from(range)?;

    let mut rng = StdRng::seed_from_u64(seed);
    let points = generate_random_points_in_range_with_rng(n_points, range, &mut rng)?;

    Ok(points)
}

/// Generate random points from validated coordinate bounds with a seeded RNG.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate.
/// * `range` - Validated coordinate bounds shared by every coordinate axis.
/// * `seed` - Seed for reproducible point generation.
///
/// # Returns
///
/// Vector of random points with coordinates sampled from `range`.
///
/// # Errors
///
/// Returns [`RandomPointGenerationError::GeneratedPointCoordinateRejected`] if
/// a generated coordinate is rejected while constructing a [`Point`], or
/// [`RandomPointGenerationError::CoordinateRangeWidthOverflow`] if the validated
/// bounds are too wide for uniform floating-point sampling.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::generators::{
///     CoordinateRange, RandomPointGenerationError, generate_random_points_in_range_seeded,
/// };
///
/// # fn main() -> Result<(), RandomPointGenerationError> {
/// let range = CoordinateRange::try_new(0.0_f64, 1.0)?;
/// let points_a = generate_random_points_in_range_seeded::<3>(4, range, 42)?;
/// let points_b = generate_random_points_in_range_seeded::<3>(4, range, 42)?;
/// assert_eq!(points_a, points_b);
/// # Ok(())
/// # }
/// ```
pub fn generate_random_points_in_range_seeded<const D: usize>(
    n_points: usize,
    range: CoordinateRange<f64>,
    seed: u64,
) -> Result<Vec<Point<D>>, RandomPointGenerationError> {
    let mut rng = StdRng::seed_from_u64(seed);
    generate_random_points_in_range_with_rng(n_points, range, &mut rng)
}

/// Generate random points in a periodic (toroidal) domain with seeded RNG.
///
/// Each coordinate is independently sampled from `[T::zero(), domain[i])` using a seeded
/// random number generator. All `domain[i]` must be strictly positive.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate
/// * `domain` - Period length for each dimension (all must be > 0)
/// * `seed` - Seed for the random number generator
///
/// # Returns
///
/// Vector of random points with coordinates in `[0, domain[i])` per axis.
///
/// # Errors
///
/// Returns [`RandomPointGenerationError::InvalidPeriodicDomain`] if any
/// `domain[i]` is non-finite or non-positive.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::generators::{
///     RandomPointGenerationError, generate_random_points_periodic,
/// };
///
/// # fn main() -> Result<(), RandomPointGenerationError> {
/// // Generate 100 random 2D points in [0,1) × [0,2)
/// let points = generate_random_points_periodic::<2>(100, [1.0, 2.0], 42)?;
/// assert_eq!(points.len(), 100);
///
/// // Reproducible generation
/// let points1 = generate_random_points_periodic::<3>(50, [1.0, 1.0, 1.0], 123)?;
/// let points2 = generate_random_points_periodic::<3>(50, [1.0, 1.0, 1.0], 123)?;
/// assert_eq!(points1, points2);
/// # Ok(())
/// # }
/// ```
pub fn generate_random_points_periodic<const D: usize>(
    n_points: usize,
    domain: [f64; D],
    seed: u64,
) -> Result<Vec<Point<D>>, RandomPointGenerationError> {
    // Parse domain periods before sampling so later code consumes only positive periods.
    let mut periods = [PositiveScalar(1.0); D];
    for (axis, period) in domain.into_iter().enumerate() {
        periods[axis] = PositiveScalar::try_new(period)
            .map_err(|reason| RandomPointGenerationError::InvalidPeriodicDomain { axis, reason })?;
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut points = Vec::with_capacity(n_points);

    for _ in 0..n_points {
        let coords = core::array::from_fn(|axis| rng.random_range(0.0..periods[axis].get()));
        points.push(point_from_generated_coords(coords)?);
    }

    Ok(points)
}

/// Generates ball samples through an injected RNG after validating the public radius input.
///
/// Both seeded and unseeded public ball generators delegate here so radius
/// parsing and rejection-sampling behavior stay identical across entry points.
fn generate_random_points_in_ball_with_rng<R, const D: usize>(
    n_points: usize,
    radius: f64,
    rng: &mut R,
) -> Result<Vec<Point<D>>, RandomPointGenerationError>
where
    R: rand::Rng + ?Sized,
{
    generate_random_points_in_ball_with_rng_and_budget(
        n_points,
        radius,
        rng,
        ball_max_attempts(n_points, D),
    )
}

/// Generates ball samples through an injected RNG with an explicit attempt budget.
fn generate_random_points_in_ball_with_rng_and_budget<R, const D: usize>(
    n_points: usize,
    radius: f64,
    rng: &mut R,
    max_attempts: usize,
) -> Result<Vec<Point<D>>, RandomPointGenerationError>
where
    R: rand::Rng + ?Sized,
{
    let radius = PositiveScalar::try_new(radius)
        .map_err(|reason| RandomPointGenerationError::InvalidBallRadius { reason })?;

    if n_points == 0 {
        return Ok(Vec::new());
    }

    // Rejection sampling from `[-radius, radius]^D` yields a uniform ball sample.
    let bounds = radius.symmetric_range();
    let radius = radius.get();
    let radius_sq = radius * radius;
    if !radius_sq.is_finite() {
        return Err(RandomPointGenerationError::InvalidBallRadiusSquared {
            value: InvalidCoordinateValue::from_debug(&radius_sq),
        });
    }
    let bounds = SamplerRange::try_new(bounds)?;

    let mut points = Vec::with_capacity(n_points);
    let mut attempts = 0;

    while points.len() < n_points && attempts < max_attempts {
        attempts += 1;
        let coords = [0.0; D].map(|_| rng.random_range(bounds.min()..bounds.max()));
        let norm_sq = coords.iter().fold(0.0, |acc, &c| c.mul_add(c, acc));
        if norm_sq <= radius_sq {
            points.push(point_from_generated_coords(coords)?);
        }
    }

    if points.len() < n_points {
        return Err(RandomPointGenerationError::BallSamplingFailed {
            requested_points: n_points,
            generated_points: points.len(),
            dimension: D,
            radius,
            attempts,
        });
    }

    Ok(points)
}

/// Generate random points uniformly distributed in a D-dimensional ball.
///
/// Points are generated inside the ball of radius `radius` centered at the origin.
///
/// ## Distribution
///
/// This uses rejection sampling from the axis-aligned cube `[-radius, radius]^D`.
/// Since candidates are drawn uniformly from the cube and accepted only when they
/// lie inside the ball, the accepted points are i.i.d. and uniformly distributed
/// within the ball.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate
/// * `radius` - Ball radius (must be finite and > 0)
///
/// # Errors
///
/// Returns [`RandomPointGenerationError::InvalidBallRadius`] if `radius` is
/// non-finite or non-positive,
/// [`RandomPointGenerationError::InvalidBallRadiusSquared`] if squaring a
/// finite radius overflows to a non-finite value,
/// [`RandomPointGenerationError::CoordinateRangeWidthOverflow`] if the
/// radius-derived sampling cube is too wide for uniform floating-point
/// sampling, or
/// [`RandomPointGenerationError::BallSamplingFailed`] if rejection sampling
/// exhausts its attempt budget before producing `n_points`.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::generators::{
///     RandomPointGenerationError, generate_random_points_in_ball,
/// };
///
/// # fn main() -> Result<(), RandomPointGenerationError> {
/// // Generate 100 random 4D points in a radius-10 ball.
/// let points = generate_random_points_in_ball::<4>(100, 10.0)?;
/// assert_eq!(points.len(), 100);
/// # Ok(())
/// # }
/// ```
pub fn generate_random_points_in_ball<const D: usize>(
    n_points: usize,
    radius: f64,
) -> Result<Vec<Point<D>>, RandomPointGenerationError> {
    let mut rng = rand::rng();
    generate_random_points_in_ball_with_rng(n_points, radius, &mut rng)
}

/// Generate random points uniformly distributed in a D-dimensional ball, using a seeded RNG.
///
/// See [`generate_random_points_in_ball`] for distribution and semantics.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate
/// * `radius` - Ball radius (must be finite and > 0)
/// * `seed` - Seed for the random number generator
///
/// # Errors
///
/// Returns [`RandomPointGenerationError::InvalidBallRadius`] if `radius` is
/// non-finite or non-positive,
/// [`RandomPointGenerationError::InvalidBallRadiusSquared`] if squaring a
/// finite radius overflows to a non-finite value,
/// [`RandomPointGenerationError::CoordinateRangeWidthOverflow`] if the
/// radius-derived sampling cube is too wide for uniform floating-point
/// sampling, or
/// [`RandomPointGenerationError::BallSamplingFailed`] if rejection sampling
/// exhausts its attempt budget before producing `n_points`.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::generators::{
///     RandomPointGenerationError, generate_random_points_in_ball_seeded,
/// };
///
/// # fn main() -> Result<(), RandomPointGenerationError> {
/// let points1 = generate_random_points_in_ball_seeded::<4>(10, 1.0, 42)?;
/// let points2 = generate_random_points_in_ball_seeded::<4>(10, 1.0, 42)?;
/// assert_eq!(points1, points2);
/// # Ok(())
/// # }
/// ```
pub fn generate_random_points_in_ball_seeded<const D: usize>(
    n_points: usize,
    radius: f64,
    seed: u64,
) -> Result<Vec<Point<D>>, RandomPointGenerationError> {
    let mut rng = StdRng::seed_from_u64(seed);
    generate_random_points_in_ball_with_rng(n_points, radius, &mut rng)
}

/// Generate points arranged in a regular grid pattern.
///
/// This function creates points in D-dimensional space arranged in a regular grid
/// (Cartesian product of equally spaced coordinates),
/// which provides a structured, predictable point distribution useful for
/// benchmarking and testing geometric algorithms under best-case scenarios.
///
/// The implementation uses an efficient mixed-radix counter to generate
/// coordinates on-the-fly without allocating intermediate index vectors,
/// making it memory-efficient for large grids and high dimensions.
///
/// # Arguments
///
/// * `points_per_dim` - Non-zero number of points along each dimension
/// * `spacing` - Positive distance between adjacent grid points
/// * `offset` - Translation offset for the entire grid
///
/// # Returns
///
/// Vector of grid points, or a `RandomPointGenerationError` if parameters are invalid.
///
/// # Errors
///
/// Returns [`RandomPointGenerationError::GridSizeOverflow`] if the requested
/// grid size overflows `usize`,
/// [`RandomPointGenerationError::GridAllocationTooLarge`] if the grid exceeds
/// the allocation safety cap, or
/// [`RandomPointGenerationError::InvalidGridSpacing`] if `spacing` is
/// non-finite or non-positive,
/// [`RandomPointGenerationError::InvalidGridOffset`] if the grid parameters
/// contain non-finite offsets,
/// [`RandomPointGenerationError::InvalidGeneratedGridCoordinate`] if a generated
/// coordinate overflows to a non-finite value, or
/// [`RandomPointGenerationError::CoordinateConversionFailed`] if a grid index
/// cannot be represented exactly by the supported coordinate scalar.
///
/// # References
///
/// The mixed-radix counter algorithm is described in:
/// - D. E. Knuth, *The Art of Computer Programming, Vol. 4A: Combinatorial Algorithms*, Addison-Wesley, 2011.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::generators::{
///     RandomPointGenerationError, generate_grid_points,
/// };
/// use std::num::NonZeroUsize;
///
/// # fn main() -> Result<(), RandomPointGenerationError> {
/// let Some(four) = NonZeroUsize::new(4) else {
///     return Ok(());
/// };
/// let Some(three) = NonZeroUsize::new(3) else {
///     return Ok(());
/// };
/// let Some(two) = NonZeroUsize::new(2) else {
///     return Ok(());
/// };
///
/// // Generate 2D grid: 4x4 = 16 points with unit spacing
/// let grid_2d = generate_grid_points::<2>(four, 1.0, [0.0, 0.0])?;
/// assert_eq!(grid_2d.len(), 16);
///
/// // Generate 3D grid: 3x3x3 = 27 points with spacing 2.0
/// let grid_3d = generate_grid_points::<3>(three, 2.0, [0.0, 0.0, 0.0])?;
/// assert_eq!(grid_3d.len(), 27);
///
/// // Generate 4D grid centered at origin
/// let grid_4d = generate_grid_points::<4>(two, 1.0, [-0.5, -0.5, -0.5, -0.5])?;
/// assert_eq!(grid_4d.len(), 16); // 2^4 = 16 points
/// # Ok(())
/// # }
/// ```
pub fn generate_grid_points<const D: usize>(
    points_per_dim: NonZeroUsize,
    spacing: f64,
    offset: [f64; D],
) -> Result<Vec<Point<D>>, RandomPointGenerationError> {
    let points_per_dim = points_per_dim.get();

    let spacing = PositiveScalar::try_new(spacing)
        .map_err(|reason| RandomPointGenerationError::InvalidGridSpacing { reason })?
        .get();

    for (axis, coordinate) in offset.iter().enumerate() {
        if !coordinate.is_finite() {
            return Err(RandomPointGenerationError::InvalidGridOffset {
                axis,
                value: InvalidCoordinateValue::from_debug(coordinate),
            });
        }
    }

    // Compute total_points with overflow checking (avoids debug panic or release wrap)
    let mut total_points: usize = 1;
    for _ in 0..D {
        total_points = total_points.checked_mul(points_per_dim).ok_or({
            RandomPointGenerationError::GridSizeOverflow {
                points_per_dim,
                dimension: D,
            }
        })?;
    }

    // Dimension/type-aware memory cap: total_points * D * size_of::<T>()
    let per_point_bytes = D.saturating_mul(core::mem::size_of::<f64>());
    let total_bytes = total_points.saturating_mul(per_point_bytes);
    let cap = max_grid_bytes_safety_cap();
    if total_bytes > cap {
        return Err(RandomPointGenerationError::GridAllocationTooLarge {
            required_bytes: total_bytes,
            cap_bytes: cap,
        });
    }
    let mut points = Vec::with_capacity(total_points);

    // Use mixed-radix counter over D dimensions (see Knuth TAOCP Vol 4A)
    // This avoids O(N) memory allocation for intermediate index vectors
    let mut idx = [0usize; D];
    for _ in 0..total_points {
        let mut coords = [0.0; D];
        for d in 0..D {
            let index_as_scalar = grid_index_as_scalar::<D>(&idx, d, points_per_dim)?;
            coords[d] = index_as_scalar.mul_add(spacing, offset[d]);
            if !coords[d].is_finite() {
                return Err(RandomPointGenerationError::InvalidGeneratedGridCoordinate {
                    axis: d,
                    value: InvalidCoordinateValue::from_debug(&coords[d]),
                });
            }
        }
        points.push(point_from_generated_coords(coords)?);

        // Increment mixed-radix counter
        for d in (0..D).rev() {
            idx[d] += 1;
            if idx[d] < points_per_dim {
                break;
            }
            idx[d] = 0;
        }
    }

    Ok(points)
}

/// Converts one mixed-radix grid index component into the coordinate scalar type.
///
/// This keeps [`generate_grid_points`] from silently rounding large grid indices
/// when a scalar type cannot represent every `usize` exactly.
fn grid_index_as_scalar<const D: usize>(
    idx: &[usize; D],
    coordinate_index: usize,
    _points_per_dim: usize,
) -> Result<f64, RandomPointGenerationError> {
    safe_usize_to_scalar(idx[coordinate_index]).map_err(|source| {
        RandomPointGenerationError::CoordinateConversionFailed {
            value: idx[coordinate_index],
            target_type: type_name::<f64>(),
            source,
        }
    })
}

/// Generate points using Poisson disk sampling for uniform distribution.
///
/// This function creates points with approximately uniform spacing using a
/// simplified Poisson disk sampling algorithm. This provides a more natural
/// point distribution than pure random sampling, useful for benchmarking
/// algorithms under realistic scenarios.
///
/// **Important**: The algorithm may terminate early if `min_distance` is too tight
/// for the given bounds and dimension, resulting in fewer points than requested.
/// In higher dimensions, tight spacing constraints become exponentially more
/// difficult to satisfy.
///
/// **Complexity**: The current implementation uses O(n²) distance checks per candidate,
/// which is efficient for typical test/benchmark sizes but may be slow for very large
/// point sets (e.g., n > 10,000).
///
/// # Arguments
///
/// * `n_points` - Target number of points to generate
/// * `bounds` - Bounding box as (min, max) coordinates
/// * `min_distance` - Minimum distance between any two points; finite non-positive
///   values disable spacing constraints
/// * `seed` - Seed for reproducible results
///
/// # Returns
///
/// Vector of Poisson-distributed points, or a [`RandomPointGenerationError`] if parameters are invalid.
/// Note: The actual number of points may be less than `n_points` due to spacing constraints.
///
/// # Errors
///
/// * [`RandomPointGenerationError::InvalidCoordinateRange`] if either bound is
///   non-finite, equal, decreasing, or incomparable in `bounds`.
/// * [`RandomPointGenerationError::InvalidMinimumDistance`] if `min_distance`
///   is non-finite.
/// * [`RandomPointGenerationError::CoordinateRangeWidthOverflow`] if the
///   validated bounds are too wide for uniform floating-point sampling.
/// * [`RandomPointGenerationError::PoissonSamplingFailed`] if `min_distance`
///   is too large for the bounds or if no points can be generated within the
///   attempt limit
///
/// # Examples
///
/// ```
/// use delaunay::prelude::generators::{
///     RandomPointGenerationError, try_generate_poisson_points,
/// };
///
/// # fn main() -> Result<(), RandomPointGenerationError> {
/// // Generate ~100 2D points with minimum distance 0.1 in unit square
/// let poisson_2d = try_generate_poisson_points::<2>(100, (0.0, 1.0), 0.1, 42)?;
/// // Actual count may be less than 100 due to spacing constraints
///
/// // Generate 3D points in a cube
/// let poisson_3d = try_generate_poisson_points::<3>(50, (-1.0, 1.0), 0.2, 123)?;
/// # Ok(())
/// # }
/// ```
pub fn try_generate_poisson_points<const D: usize>(
    n_points: usize,
    bounds: (f64, f64),
    min_distance: f64,
    seed: u64,
) -> Result<Vec<Point<D>>, RandomPointGenerationError> {
    let bounds = CoordinateRange::try_from(bounds)?;

    generate_poisson_points_in_range(n_points, bounds, min_distance, seed)
}

/// Generate Poisson disk points from validated coordinate bounds.
///
/// # Arguments
///
/// * `n_points` - Target number of points to generate.
/// * `bounds` - Validated coordinate bounds shared by every coordinate axis.
/// * `min_distance` - Minimum distance between accepted points; finite
///   non-positive values disable spacing constraints.
/// * `seed` - Seed for reproducible point generation.
///
/// # Returns
///
/// Vector of Poisson-distributed points. The actual number of points may be
/// less than `n_points` when the spacing constraint is too tight for the bounds.
///
/// # Errors
///
/// Returns [`RandomPointGenerationError::InvalidMinimumDistance`] if
/// `min_distance` is non-finite. Returns
/// [`RandomPointGenerationError::PoissonSamplingFailed`] if `min_distance` is
/// too large for the bounds or if no point can be generated within the attempt
/// limit. Returns
/// [`RandomPointGenerationError::CoordinateRangeWidthOverflow`] if the validated
/// bounds are too wide for uniform floating-point sampling.
///
/// # Examples
///
/// ```
/// use delaunay::prelude::generators::{
///     CoordinateRange, RandomPointGenerationError, generate_poisson_points_in_range,
/// };
///
/// # fn main() -> Result<(), RandomPointGenerationError> {
/// let range = CoordinateRange::try_new(0.0_f64, 1.0)?;
/// let points = generate_poisson_points_in_range::<2>(16, range, 0.1, 42)?;
/// assert!(!points.is_empty());
/// # Ok(())
/// # }
/// ```
pub fn generate_poisson_points_in_range<const D: usize>(
    n_points: usize,
    bounds: CoordinateRange<f64>,
    min_distance: f64,
    seed: u64,
) -> Result<Vec<Point<D>>, RandomPointGenerationError> {
    let spacing = PoissonSpacing::try_new(min_distance)?;

    if n_points == 0 {
        return Ok(Vec::new());
    }

    let mut rng = StdRng::seed_from_u64(seed);

    let min_distance = match spacing {
        PoissonSpacing::Disabled => {
            return generate_random_points_in_range_with_rng(n_points, bounds, &mut rng);
        }
        PoissonSpacing::Minimum(min_distance) => min_distance,
    };
    let min_distance = min_distance.get();

    let raw_bounds = bounds;
    let bounds = SamplerRange::try_new(bounds)?;
    let mut points: Vec<Point<D>> = Vec::new();

    // Simple Poisson disk sampling: rejection method
    // Scale max attempts with dimension since higher dimensions make spacing harder
    // Base: 30 attempts per point, scaled exponentially with dimension to account
    // for the curse of dimensionality in Poisson disk sampling
    let max_attempts = poisson_max_attempts(n_points, D);
    let mut attempts = 0;

    while points.len() < n_points && attempts < max_attempts {
        attempts += 1;

        // Generate candidate point
        let candidate = sample_point_in_range(bounds, &mut rng)?;

        // Check distance to all existing points
        let mut valid = true;
        let candidate_coords: [f64; D] = *candidate.coords();
        for existing_point in &points {
            let existing_coords: [f64; D] = *existing_point.coords();

            // Calculate distance using hypot for numerical stability
            let mut diff_coords = [0.0; D];
            for i in 0..D {
                diff_coords[i] = candidate_coords[i] - existing_coords[i];
            }
            let distance = hypot(&diff_coords);

            if distance < min_distance {
                valid = false;
                break;
            }
        }

        if valid {
            points.push(candidate);
        }
    }

    if points.is_empty() {
        return Err(RandomPointGenerationError::PoissonSamplingFailed {
            requested_points: n_points,
            generated_points: points.len(),
            min_distance,
            bounds: raw_bounds,
            attempts,
        });
    }

    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::coordinate_range::{
        CoordinateRangeBound, CoordinateRangeError, CoordinateRangeOrdering, InvalidCoordinateValue,
    };
    use crate::geometry::traits::coordinate::CoordinateConversionError;
    use approx::assert_relative_eq;
    use std::assert_matches;

    /// Builds non-zero test literals so grid-generation tests exercise the typed boundary.
    const fn nonzero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test point count must be non-zero")
    }

    fn assert_invalid_coordinate_range<const D: usize>(
        result: &Result<Vec<Point<D>>, RandomPointGenerationError<f64>>,
        expected_ordering: CoordinateRangeOrdering,
        expected_min: f64,
        expected_max: f64,
    ) {
        let Err(RandomPointGenerationError::InvalidCoordinateRange {
            source: CoordinateRangeError::NonIncreasing { ordering, min, max },
        }) = result
        else {
            panic!("expected non-increasing coordinate range");
        };
        assert_eq!(*ordering, expected_ordering);
        assert_relative_eq!(*min, expected_min, epsilon = f64::EPSILON);
        assert_relative_eq!(*max, expected_max, epsilon = f64::EPSILON);
    }

    fn assert_generated_point_coordinate_rejected<const D: usize>(
        coords: [f64; D],
        expected_index: usize,
        expected_value: &InvalidCoordinateValue,
    ) {
        let Err(RandomPointGenerationError::GeneratedPointCoordinateRejected {
            source:
                CoordinateValidationError::InvalidCoordinate {
                    coordinate_index,
                    coordinate_value,
                    dimension,
                },
        }) = point_from_generated_coords(coords)
        else {
            panic!("expected generated point coordinate rejection");
        };

        assert_eq!(coordinate_index, expected_index);
        assert_eq!(&coordinate_value, expected_value);
        assert_eq!(dimension, D);
    }

    macro_rules! gen_generated_point_coordinate_rejected_tests {
        ($dim:literal) => {
            pastey::paste! {
                #[test]
                fn [<test_generated_point_coordinate_rejected_nan_ $dim d>]() {
                    let mut coords = [0.0; $dim];
                    coords[0] = f64::NAN;
                    assert_generated_point_coordinate_rejected::<$dim>(
                        coords,
                        0,
                        &InvalidCoordinateValue::Nan,
                    );
                }

                #[test]
                fn [<test_generated_point_coordinate_rejected_infinity_ $dim d>]() {
                    let mut coords = [0.0; $dim];
                    coords[$dim - 1] = f64::INFINITY;
                    assert_generated_point_coordinate_rejected::<$dim>(
                        coords,
                        $dim - 1,
                        &InvalidCoordinateValue::PositiveInfinity,
                    );
                }
            }
        };
    }

    gen_generated_point_coordinate_rejected_tests!(2);
    gen_generated_point_coordinate_rejected_tests!(3);
    gen_generated_point_coordinate_rejected_tests!(4);
    gen_generated_point_coordinate_rejected_tests!(5);

    #[test]
    fn random_point_generation_error_display_names_variants() {
        let range_error: RandomPointGenerationError<f64> =
            RandomPointGenerationError::InvalidCoordinateRange {
                source: CoordinateRangeError::NonIncreasing {
                    ordering: CoordinateRangeOrdering::Decreasing,
                    min: 10.0,
                    max: 5.0,
                },
            };
        let display = format!("{range_error}");
        assert!(display.contains("Invalid coordinate range"));

        let range_width_error: RandomPointGenerationError<f64> =
            RandomPointGenerationError::CoordinateRangeWidthOverflow {
                min: -f64::MAX,
                max: f64::MAX,
            };
        let display = format!("{range_width_error}");
        assert!(display.contains("Coordinate range width overflows"));

        let generated_point_error: RandomPointGenerationError<f64> =
            RandomPointGenerationError::GeneratedPointCoordinateRejected {
                source: CoordinateValidationError::InvalidCoordinate {
                    coordinate_index: 1,
                    coordinate_value: InvalidCoordinateValue::PositiveInfinity,
                    dimension: 3,
                },
            };
        let display = format!("{generated_point_error}");
        assert!(display.contains("Generated point coordinates were invalid"));
        assert!(display.contains("inf"));

        let grid_allocation_error: RandomPointGenerationError<f64> =
            RandomPointGenerationError::GridAllocationTooLarge {
                required_bytes: 8_589_934_592,
                cap_bytes: 4_294_967_296,
            };
        let display = format!("{grid_allocation_error}");
        assert!(display.contains("exceeds safety cap"));

        let grid_spacing_error: RandomPointGenerationError<f64> =
            RandomPointGenerationError::InvalidGridSpacing {
                reason: InvalidPositiveScalar::NonPositive { value: 0.0 },
            };
        let display = format!("{grid_spacing_error}");
        assert!(display.contains("Invalid grid spacing"));
        assert!(display.contains("non-positive"));

        let radius_error: RandomPointGenerationError<f64> =
            RandomPointGenerationError::InvalidBallRadius {
                reason: InvalidPositiveScalar::NonFinite {
                    value: InvalidCoordinateValue::Nan,
                },
            };
        let display = format!("{radius_error}");
        assert!(display.contains("Invalid ball radius"));
        assert!(display.contains("NaN"));

        let squared_radius_error: RandomPointGenerationError<f64> =
            RandomPointGenerationError::InvalidBallRadiusSquared {
                value: InvalidCoordinateValue::PositiveInfinity,
            };
        let display = format!("{squared_radius_error}");
        assert!(display.contains("Invalid squared ball radius"));
        assert!(display.contains("inf"));

        let ball_error = RandomPointGenerationError::BallSamplingFailed {
            requested_points: 4,
            generated_points: 1,
            dimension: 12,
            radius: 2.0,
            attempts: 8_192,
        };
        let display = format!("{ball_error}");
        assert!(display.contains("Could not generate 4 ball points"));
        assert!(display.contains("dimension 12"));
        assert!(display.contains("generated 1"));

        let generated_grid_coordinate_error: RandomPointGenerationError<f64> =
            RandomPointGenerationError::InvalidGeneratedGridCoordinate {
                axis: 2,
                value: InvalidCoordinateValue::PositiveInfinity,
            };
        let display = format!("{generated_grid_coordinate_error}");
        assert!(display.contains("Invalid generated grid coordinate at axis 2"));
        assert!(display.contains("inf"));

        let poisson_error = RandomPointGenerationError::PoissonSamplingFailed {
            requested_points: 10,
            generated_points: 0,
            min_distance: 2.5,
            bounds: CoordinateRange::try_new(-1.0, 1.0).unwrap(),
            attempts: 300,
        };
        let RandomPointGenerationError::PoissonSamplingFailed {
            min_distance,
            bounds,
            ..
        } = poisson_error
        else {
            panic!("expected PoissonSamplingFailed");
        };
        assert_relative_eq!(min_distance, 2.5, epsilon = f64::EPSILON);
        assert_relative_eq!(bounds.min(), -1.0, epsilon = f64::EPSILON);
        assert_relative_eq!(bounds.max(), 1.0, epsilon = f64::EPSILON);
    }

    // =============================================================================
    // RANDOM POINT GENERATION TESTS
    // =============================================================================

    #[test]
    fn test_generate_random_points_rejects_nonfinite_tuple_bounds() {
        assert_matches!(
            try_generate_random_points::<2>(4, (f64::NAN, 1.0)),
            Err(RandomPointGenerationError::InvalidCoordinateRange {
                source: CoordinateRangeError::NonFiniteBound { bound, value }
            }) if bound == CoordinateRangeBound::Minimum && value == InvalidCoordinateValue::Nan
        );
        assert_matches!(
            try_generate_random_points_seeded::<2>(4, (0.0, f64::INFINITY), 42),
            Err(RandomPointGenerationError::InvalidCoordinateRange {
                source: CoordinateRangeError::NonFiniteBound { bound, value }
            }) if bound == CoordinateRangeBound::Maximum
                && value == InvalidCoordinateValue::PositiveInfinity
        );
        assert_matches!(
            try_generate_poisson_points::<2>(4, (f64::NEG_INFINITY, 1.0), 0.1, 42),
            Err(RandomPointGenerationError::InvalidCoordinateRange {
                source: CoordinateRangeError::NonFiniteBound { bound, value }
            }) if bound == CoordinateRangeBound::Minimum
                && value == InvalidCoordinateValue::NegativeInfinity
        );
    }

    #[test]
    fn test_range_based_generators_use_validated_bounds() {
        let range = CoordinateRange::try_new(-1.0_f64, 1.0).unwrap();

        let points = generate_random_points_in_range::<3>(8, range)
            .expect("validated range should generate finite points");
        assert_eq!(points.len(), 8);
        for point in points {
            for &coord in point.coords() {
                assert!((-1.0..1.0).contains(&coord));
            }
        }

        let seeded_a = generate_random_points_in_range_seeded::<3>(8, range, 42)
            .expect("validated range should generate finite points");
        let seeded_b = generate_random_points_in_range_seeded::<3>(8, range, 42)
            .expect("validated range should generate finite points");
        assert_eq!(seeded_a, seeded_b);
        for point in seeded_a {
            for &coord in point.coords() {
                assert!((-1.0..1.0).contains(&coord));
            }
        }

        let empty = generate_random_points_in_range::<3>(0, range)
            .expect("validated range should generate finite points");
        assert!(empty.is_empty());

        let seeded_empty = generate_random_points_in_range_seeded::<3>(0, range, 42)
            .expect("validated range should generate finite points");
        assert!(seeded_empty.is_empty());

        let poisson = generate_poisson_points_in_range::<2>(8, range, 0.1, 42).unwrap();
        assert!(!poisson.is_empty());

        let unconstrained_poisson =
            generate_poisson_points_in_range::<2>(8, range, 0.0, 42).unwrap();
        assert_eq!(unconstrained_poisson.len(), 8);

        let empty_poisson = generate_poisson_points_in_range::<2>(0, range, 0.1, 42).unwrap();
        assert!(empty_poisson.is_empty());

        let invalid_empty_poisson = generate_poisson_points_in_range::<2>(0, range, f64::NAN, 42);
        assert_matches!(
            invalid_empty_poisson,
            Err(RandomPointGenerationError::InvalidMinimumDistance { distance })
                if distance == InvalidCoordinateValue::Nan
        );
    }

    #[test]
    fn test_generate_random_points_in_range_rejects_overflowing_width() {
        let range = CoordinateRange::try_new(-f64::MAX, f64::MAX).unwrap();

        assert_matches!(
            try_generate_random_points::<2>(1, (-f64::MAX, f64::MAX)),
            Err(RandomPointGenerationError::CoordinateRangeWidthOverflow { min, max })
                if min.to_bits() == (-f64::MAX).to_bits() && max.to_bits() == f64::MAX.to_bits()
        );

        assert_matches!(
            try_generate_random_points_seeded::<2>(1, (-f64::MAX, f64::MAX), 42),
            Err(RandomPointGenerationError::CoordinateRangeWidthOverflow { min, max })
                if min.to_bits() == (-f64::MAX).to_bits() && max.to_bits() == f64::MAX.to_bits()
        );

        assert_matches!(
            generate_random_points_in_range::<2>(1, range),
            Err(RandomPointGenerationError::CoordinateRangeWidthOverflow { min, max })
                if min.to_bits() == (-f64::MAX).to_bits() && max.to_bits() == f64::MAX.to_bits()
        );

        assert_matches!(
            generate_random_points_in_range_seeded::<2>(1, range, 42),
            Err(RandomPointGenerationError::CoordinateRangeWidthOverflow { min, max })
                if min.to_bits() == (-f64::MAX).to_bits() && max.to_bits() == f64::MAX.to_bits()
        );
    }

    #[test]
    fn test_unconstrained_poisson_rejects_overflowing_width() {
        let range = CoordinateRange::try_new(-f64::MAX, f64::MAX).unwrap();

        assert_matches!(
            generate_poisson_points_in_range::<2>(1, range, 0.0, 42),
            Err(RandomPointGenerationError::CoordinateRangeWidthOverflow { min, max })
                if min.to_bits() == (-f64::MAX).to_bits() && max.to_bits() == f64::MAX.to_bits()
        );
    }

    #[test]
    fn test_scaled_bounds_by_point_count_returns_validated_range() {
        let hundred = scaled_bounds_by_point_count(100).unwrap();
        assert_relative_eq!(hundred.min(), -50.0, epsilon = f64::EPSILON);
        assert_relative_eq!(hundred.max(), 50.0, epsilon = f64::EPSILON);

        let empty = scaled_bounds_by_point_count(0).unwrap();
        assert_relative_eq!(empty.min(), -0.5, epsilon = f64::EPSILON);
        assert_relative_eq!(empty.max(), 0.5, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_poisson_spacing_parses_raw_distance_once() {
        assert_eq!(
            PoissonSpacing::try_new(0.0_f64),
            Ok(PoissonSpacing::Disabled)
        );
        assert_eq!(
            PoissonSpacing::try_new(-1.0_f64),
            Ok(PoissonSpacing::Disabled)
        );
        assert_eq!(
            PoissonSpacing::try_new(0.25_f64),
            Ok(PoissonSpacing::Minimum(PositiveScalar(0.25)))
        );

        assert_matches!(
            PoissonSpacing::try_new(f64::NAN),
            Err(RandomPointGenerationError::InvalidMinimumDistance { distance })
                if distance == InvalidCoordinateValue::Nan
        );
        assert_matches!(
            PoissonSpacing::try_new(f64::INFINITY),
            Err(RandomPointGenerationError::InvalidMinimumDistance { distance })
                if distance == InvalidCoordinateValue::PositiveInfinity
        );
        assert_matches!(
            PoissonSpacing::try_new(f64::NEG_INFINITY),
            Err(RandomPointGenerationError::InvalidMinimumDistance { distance })
                if distance == InvalidCoordinateValue::NegativeInfinity
        );
    }

    #[test]
    fn test_positive_scalar_preserves_finite_positive_proof() {
        assert_eq!(PositiveScalar::try_new(2.5_f64), Ok(PositiveScalar(2.5)));
        assert_eq!(
            PositiveScalar::try_new(0.0_f64),
            Err(InvalidPositiveScalar::NonPositive { value: 0.0 })
        );
        assert_eq!(
            PositiveScalar::try_new(-1.0_f64),
            Err(InvalidPositiveScalar::NonPositive { value: -1.0 })
        );
        assert_eq!(
            PositiveScalar::try_new(f64::NAN),
            Err(InvalidPositiveScalar::NonFinite {
                value: InvalidCoordinateValue::Nan
            })
        );
        assert_eq!(
            PositiveScalar::try_new(f64::INFINITY),
            Err(InvalidPositiveScalar::NonFinite {
                value: InvalidCoordinateValue::PositiveInfinity
            })
        );
    }

    #[test]
    fn test_poisson_attempt_budget_saturates_for_large_point_counts() {
        assert_eq!(poisson_max_attempts(10, 2), 300);
        assert_eq!(poisson_max_attempts(10, 4), 600);
        assert_eq!(poisson_max_attempts(10, 6), 1_200);
        assert_eq!(poisson_max_attempts(10, 7), 2_400);
        assert_eq!(poisson_max_attempts(usize::MAX, 7), usize::MAX);
    }

    #[test]
    fn test_ball_attempt_budget_saturates_for_large_point_counts() {
        assert_eq!(ball_max_attempts(10, 2), 10_240);
        assert_eq!(ball_max_attempts(10, 4), 20_480);
        assert_eq!(ball_max_attempts(10, 6), 40_960);
        assert_eq!(ball_max_attempts(10, 7), 81_920);
        assert_eq!(ball_max_attempts(usize::MAX, 7), usize::MAX);
    }

    #[test]
    fn test_generate_random_points_2d() {
        // Test 2D random point generation
        let points = try_generate_random_points::<2>(100, (-10.0, 10.0)).unwrap();

        assert_eq!(points.len(), 100);

        // Check that all points are within range
        for point in &points {
            let coords = *point.coords();
            assert!(coords[0] >= -10.0 && coords[0] < 10.0);
            assert!(coords[1] >= -10.0 && coords[1] < 10.0);
        }
    }

    #[test]
    fn test_generate_random_points_3d() {
        // Test 3D random point generation
        let points = try_generate_random_points::<3>(75, (0.0, 5.0)).unwrap();

        assert_eq!(points.len(), 75);

        for point in &points {
            let coords = *point.coords();
            assert!(coords[0] >= 0.0 && coords[0] < 5.0);
            assert!(coords[1] >= 0.0 && coords[1] < 5.0);
            assert!(coords[2] >= 0.0 && coords[2] < 5.0);
        }
    }

    #[test]
    fn test_generate_random_points_4d() {
        // Test 4D random point generation
        let points = try_generate_random_points::<4>(50, (-2.0, 2.0)).unwrap();

        assert_eq!(points.len(), 50);

        for point in &points {
            let coords = *point.coords();
            for &coord in &coords {
                assert!((-2.0..2.0).contains(&coord));
            }
        }
    }

    #[test]
    fn test_generate_random_points_5d() {
        // Test 5D random point generation
        let points = try_generate_random_points::<5>(25, (-1.0, 1.0)).unwrap();

        assert_eq!(points.len(), 25);

        for point in &points {
            let coords = *point.coords();
            for &coord in &coords {
                assert!((-1.0..1.0).contains(&coord));
            }
        }
    }

    #[test]
    fn test_generate_random_points_error_handling() {
        // Test invalid range (non-increasing bounds) across all dimensions

        // 2D
        let result = try_generate_random_points::<2>(100, (10.0, -10.0));
        assert_invalid_coordinate_range(&result, CoordinateRangeOrdering::Decreasing, 10.0, -10.0);

        // 3D
        let result = try_generate_random_points::<3>(50, (5.0, 5.0));
        assert_invalid_coordinate_range(&result, CoordinateRangeOrdering::Equal, 5.0, 5.0);

        // 4D
        let result = try_generate_random_points::<4>(25, (1.0, 0.5));
        assert_invalid_coordinate_range(&result, CoordinateRangeOrdering::Decreasing, 1.0, 0.5);

        // 5D
        let result = try_generate_random_points::<5>(10, (2.0, 2.0));
        assert_invalid_coordinate_range(&result, CoordinateRangeOrdering::Equal, 2.0, 2.0);

        // Test valid edge case - very small range
        let points = try_generate_random_points::<2>(10, (0.0, 0.001)).unwrap();
        assert_eq!(points.len(), 10);
        for point in points {
            for &coord in point.coords() {
                assert!((0.0..0.001).contains(&coord));
            }
        }
    }

    #[test]
    fn test_generate_random_points_zero_points() {
        // Test generating zero points across all dimensions
        let points_2d = try_generate_random_points::<2>(0, (-1.0, 1.0)).unwrap();
        assert_eq!(points_2d.len(), 0);

        let points_3d = try_generate_random_points::<3>(0, (-1.0, 1.0)).unwrap();
        assert_eq!(points_3d.len(), 0);

        let points_4d = try_generate_random_points::<4>(0, (-1.0, 1.0)).unwrap();
        assert_eq!(points_4d.len(), 0);

        let points_5d = try_generate_random_points::<5>(0, (-1.0, 1.0)).unwrap();
        assert_eq!(points_5d.len(), 0);
    }

    #[test]
    fn test_generate_random_points_seeded_2d() {
        // Test seeded 2D generation reproducibility
        let seed = 42_u64;
        let points1 = try_generate_random_points_seeded::<2>(50, (-5.0, 5.0), seed).unwrap();
        let points2 = try_generate_random_points_seeded::<2>(50, (-5.0, 5.0), seed).unwrap();

        assert_eq!(points1.len(), points2.len());

        // Points should be identical with same seed
        for (p1, p2) in points1.iter().zip(points2.iter()) {
            let coords1 = *p1.coords();
            let coords2 = *p2.coords();

            for (c1, c2) in coords1.iter().zip(coords2.iter()) {
                assert_relative_eq!(c1, c2, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn test_generate_random_points_seeded_3d() {
        // Test seeded 3D generation reproducibility
        let seed = 123_u64;
        let points1 = try_generate_random_points_seeded::<3>(40, (0.0, 10.0), seed).unwrap();
        let points2 = try_generate_random_points_seeded::<3>(40, (0.0, 10.0), seed).unwrap();

        assert_eq!(points1.len(), points2.len());

        for (p1, p2) in points1.iter().zip(points2.iter()) {
            let coords1 = *p1.coords();
            let coords2 = *p2.coords();

            for (c1, c2) in coords1.iter().zip(coords2.iter()) {
                assert_relative_eq!(c1, c2, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn test_generate_random_points_seeded_4d() {
        // Test seeded 4D generation reproducibility
        let seed = 789_u64;
        let points1 = try_generate_random_points_seeded::<4>(30, (-2.5, 2.5), seed).unwrap();
        let points2 = try_generate_random_points_seeded::<4>(30, (-2.5, 2.5), seed).unwrap();

        assert_eq!(points1.len(), points2.len());

        for (p1, p2) in points1.iter().zip(points2.iter()) {
            let coords1 = *p1.coords();
            let coords2 = *p2.coords();

            for (c1, c2) in coords1.iter().zip(coords2.iter()) {
                assert_relative_eq!(c1, c2, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn test_generate_random_points_seeded_5d() {
        // Test seeded 5D generation reproducibility
        let seed = 456_u64;
        let points1 = try_generate_random_points_seeded::<5>(20, (-1.0, 3.0), seed).unwrap();
        let points2 = try_generate_random_points_seeded::<5>(20, (-1.0, 3.0), seed).unwrap();

        assert_eq!(points1.len(), points2.len());

        for (p1, p2) in points1.iter().zip(points2.iter()) {
            let coords1 = *p1.coords();
            let coords2 = *p2.coords();

            for (c1, c2) in coords1.iter().zip(coords2.iter()) {
                assert_relative_eq!(c1, c2, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn test_generate_random_points_seeded_different_seeds() {
        // Test that different seeds produce different results across all dimensions

        // 2D
        let points1_2d = try_generate_random_points_seeded::<2>(50, (0.0, 1.0), 42).unwrap();
        let points2_2d = try_generate_random_points_seeded::<2>(50, (0.0, 1.0), 123).unwrap();
        assert_ne!(points1_2d, points2_2d);

        // 3D
        let points1_3d = try_generate_random_points_seeded::<3>(30, (-5.0, 5.0), 42).unwrap();
        let points2_3d = try_generate_random_points_seeded::<3>(30, (-5.0, 5.0), 999).unwrap();
        assert_ne!(points1_3d, points2_3d);

        // 4D
        let points1_4d = try_generate_random_points_seeded::<4>(25, (-1.0, 1.0), 1337).unwrap();
        let points2_4d = try_generate_random_points_seeded::<4>(25, (-1.0, 1.0), 7331).unwrap();
        assert_ne!(points1_4d, points2_4d);

        // 5D
        let points1_5d = try_generate_random_points_seeded::<5>(15, (0.0, 10.0), 2021).unwrap();
        let points2_5d = try_generate_random_points_seeded::<5>(15, (0.0, 10.0), 2024).unwrap();
        assert_ne!(points1_5d, points2_5d);
    }
    #[test]
    fn test_generate_random_points_periodic_2d_in_domain() {
        let points = generate_random_points_periodic::<2>(100, [1.0, 2.0], 42).unwrap();
        assert_eq!(points.len(), 100);

        for point in &points {
            let coords = *point.coords();
            assert!((0.0..1.0).contains(&coords[0]));
            assert!((0.0..2.0).contains(&coords[1]));
        }
    }

    #[test]
    fn test_generate_random_points_periodic_seeded_reproducible() {
        let points1 = generate_random_points_periodic::<3>(50, [1.0, 1.0, 1.0], 123).unwrap();
        let points2 = generate_random_points_periodic::<3>(50, [1.0, 1.0, 1.0], 123).unwrap();
        assert_eq!(points1, points2);
    }

    #[test]
    fn test_generate_random_points_periodic_invalid_domain() {
        let zero_period = generate_random_points_periodic::<2>(10, [1.0, 0.0], 7);
        let Err(RandomPointGenerationError::InvalidPeriodicDomain {
            axis,
            reason: InvalidPositiveScalar::NonPositive { value },
        }) = zero_period
        else {
            panic!("expected non-positive periodic domain error");
        };
        assert_eq!(axis, 1);
        assert_relative_eq!(value, 0.0, epsilon = f64::EPSILON);

        let negative_period = generate_random_points_periodic::<2>(10, [1.0, -2.0], 7);
        let Err(RandomPointGenerationError::InvalidPeriodicDomain {
            axis,
            reason: InvalidPositiveScalar::NonPositive { value },
        }) = negative_period
        else {
            panic!("expected non-positive periodic domain error");
        };
        assert_eq!(axis, 1);
        assert_relative_eq!(value, -2.0, epsilon = f64::EPSILON);

        let nan_period = generate_random_points_periodic::<2>(10, [1.0, f64::NAN], 7);
        assert_matches!(
            nan_period,
            Err(RandomPointGenerationError::InvalidPeriodicDomain {
                axis,
                reason: InvalidPositiveScalar::NonFinite { value }
            }) if axis == 1 && value == InvalidCoordinateValue::Nan
        );

        let infinite_period = generate_random_points_periodic::<2>(10, [1.0, f64::INFINITY], 7);
        assert_matches!(
            infinite_period,
            Err(RandomPointGenerationError::InvalidPeriodicDomain {
                axis,
                reason: InvalidPositiveScalar::NonFinite { value }
            }) if axis == 1 && value == InvalidCoordinateValue::PositiveInfinity
        );
    }

    #[test]
    fn test_generate_random_points_periodic_zero_points() {
        let points = generate_random_points_periodic::<4>(0, [1.0, 2.0, 3.0, 4.0], 9).unwrap();
        assert!(points.is_empty());
    }

    // =============================================================================
    // RANDOM POINT GENERATION (UNIFORM IN BALL) TESTS
    // =============================================================================

    #[test]
    fn test_generate_random_points_in_ball_4d() {
        let radius = 3.0_f64;
        let points = generate_random_points_in_ball::<4>(200, radius).unwrap();
        assert_eq!(points.len(), 200);

        let radius_sq = radius * radius;
        for point in &points {
            let coords = *point.coords();
            let mut norm_sq = 0.0_f64;
            for &c in &coords {
                assert!(c >= -radius && c <= radius);
                norm_sq = c.mul_add(c, norm_sq);
            }
            assert!(norm_sq <= radius_sq + 1e-12);
        }
    }

    #[test]
    fn test_generate_random_points_in_ball_seeded_reproducible_4d() {
        let points1 = generate_random_points_in_ball_seeded::<4>(50, 2.5, 42).unwrap();
        let points2 = generate_random_points_in_ball_seeded::<4>(50, 2.5, 42).unwrap();
        assert_eq!(points1, points2);
    }

    #[test]
    fn test_generate_random_points_in_ball_seeded_different_seeds_4d() {
        let points1 = generate_random_points_in_ball_seeded::<4>(50, 2.5, 42).unwrap();
        let points2 = generate_random_points_in_ball_seeded::<4>(50, 2.5, 123).unwrap();
        assert_ne!(points1, points2);
    }

    #[test]
    fn test_generate_random_points_in_ball_rejects_zero_radius() {
        let result = generate_random_points_in_ball::<4>(10, 0.0);
        let Err(RandomPointGenerationError::InvalidBallRadius {
            reason: InvalidPositiveScalar::NonPositive { value },
        }) = result
        else {
            panic!("expected non-positive ball radius error");
        };
        assert_relative_eq!(value, 0.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_generate_random_points_in_ball_rejects_negative_radius() {
        let result = generate_random_points_in_ball::<4>(10, -1.0);
        let Err(RandomPointGenerationError::InvalidBallRadius {
            reason: InvalidPositiveScalar::NonPositive { value },
        }) = result
        else {
            panic!("expected non-positive ball radius error");
        };
        assert_relative_eq!(value, -1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_generate_random_points_in_ball_zero_points() {
        let points = generate_random_points_in_ball::<4>(0, 1.0).unwrap();
        assert!(points.is_empty());
    }

    #[test]
    fn test_generate_random_points_in_ball_seeded_zero_points() {
        let points = generate_random_points_in_ball_seeded::<4>(0, 1.0, 7).unwrap();
        assert!(points.is_empty());
    }

    #[test]
    fn test_generate_random_points_in_ball_rejects_nan_radius() {
        let result = generate_random_points_in_ball::<4>(10, f64::NAN);
        assert_matches!(
            result,
            Err(RandomPointGenerationError::InvalidBallRadius {
                reason: InvalidPositiveScalar::NonFinite { value }
            }) if value == InvalidCoordinateValue::Nan
        );
    }

    #[test]
    fn test_generate_random_points_in_ball_rejects_infinite_radius() {
        let result = generate_random_points_in_ball::<4>(10, f64::INFINITY);
        assert_matches!(
            result,
            Err(RandomPointGenerationError::InvalidBallRadius {
                reason: InvalidPositiveScalar::NonFinite { value }
            }) if value == InvalidCoordinateValue::PositiveInfinity
        );
    }

    #[test]
    fn test_generate_random_points_in_ball_rejects_overflowing_radius_squared() {
        let result = generate_random_points_in_ball::<2>(1, f64::MAX);
        assert_matches!(
            result,
            Err(RandomPointGenerationError::InvalidBallRadiusSquared { value })
                if value == InvalidCoordinateValue::PositiveInfinity
        );
    }

    #[test]
    fn test_generate_random_points_in_ball_seeded_in_ball_constraints_4d() {
        let radius = 1.25;
        let points = generate_random_points_in_ball_seeded::<4>(100, radius, 99).unwrap();
        assert_eq!(points.len(), 100);

        let radius_sq = radius * radius;
        for point in &points {
            let coords = *point.coords();
            let mut norm_sq = 0.0;
            for &c in &coords {
                assert!(c >= -radius && c <= radius);
                norm_sq = c.mul_add(c, norm_sq);
            }
            assert!(norm_sq <= radius_sq + 1e-12);
        }
    }

    #[test]
    fn test_generate_random_points_in_ball_returns_typed_error_when_budget_exhausts() {
        let mut rng = StdRng::seed_from_u64(42);
        let result =
            generate_random_points_in_ball_with_rng_and_budget::<_, 2>(1, 1.0, &mut rng, 0);

        let Err(RandomPointGenerationError::BallSamplingFailed {
            requested_points,
            generated_points,
            dimension,
            radius,
            attempts,
        }) = result
        else {
            panic!("expected ball sampling budget exhaustion");
        };
        assert_eq!(requested_points, 1);
        assert_eq!(generated_points, 0);
        assert_eq!(dimension, 2);
        assert_relative_eq!(radius, 1.0, epsilon = f64::EPSILON);
        assert_eq!(attempts, 0);
    }

    #[test]
    fn test_generate_random_points_in_ball_seeded_same_seed_is_deterministic_4d() {
        // This is a small smoke test that ensures deterministic output for fixed seed.
        let points1 = generate_random_points_in_ball_seeded::<4>(10, 1.0, 0xBEEF).unwrap();
        let points2 = generate_random_points_in_ball_seeded::<4>(10, 1.0, 0xBEEF).unwrap();
        assert_eq!(points1, points2);
    }

    #[test]
    fn test_generate_random_points_distribution_coverage_all_dimensions() {
        // Test that points cover the range reasonably well across all dimensions

        // 2D coverage test
        let points_2d = try_generate_random_points::<2>(500, (0.0, 10.0)).unwrap();
        let mut min_2d = [f64::INFINITY; 2];
        let mut max_2d = [f64::NEG_INFINITY; 2];

        for point in &points_2d {
            let coords = *point.coords();
            for (i, &coord) in coords.iter().enumerate() {
                min_2d[i] = min_2d[i].min(coord);
                max_2d[i] = max_2d[i].max(coord);
            }
        }

        // Should cover most of the range in each dimension
        for i in 0..2 {
            assert!(
                min_2d[i] < 2.0,
                "Min in dimension {i} should be close to lower bound"
            );
            assert!(
                max_2d[i] > 8.0,
                "Max in dimension {i} should be close to upper bound"
            );
        }

        // 5D coverage test (smaller sample)
        let points_5d = try_generate_random_points::<5>(200, (-5.0, 5.0)).unwrap();
        let mut min_5d = [f64::INFINITY; 5];
        let mut max_5d = [f64::NEG_INFINITY; 5];

        for point in &points_5d {
            let coords = *point.coords();
            for (i, &coord) in coords.iter().enumerate() {
                min_5d[i] = min_5d[i].min(coord);
                max_5d[i] = max_5d[i].max(coord);
            }
        }

        // Should have reasonable coverage in each dimension
        for i in 0..5 {
            assert!(
                min_5d[i] < -2.0,
                "Min in 5D dimension {i} should be reasonably low"
            );
            assert!(
                max_5d[i] > 2.0,
                "Max in 5D dimension {i} should be reasonably high"
            );
        }
    }

    #[test]
    fn test_generate_random_points_common_ranges() {
        // Test common useful ranges across dimensions

        // Unit cube [0,1] for all dimensions
        let unit_2d = try_generate_random_points::<2>(50, (0.0, 1.0)).unwrap();
        let unit_3d = try_generate_random_points::<3>(50, (0.0, 1.0)).unwrap();
        let unit_4d = try_generate_random_points::<4>(50, (0.0, 1.0)).unwrap();
        let unit_5d = try_generate_random_points::<5>(50, (0.0, 1.0)).unwrap();

        assert_eq!(unit_2d.len(), 50);
        assert_eq!(unit_3d.len(), 50);
        assert_eq!(unit_4d.len(), 50);
        assert_eq!(unit_5d.len(), 50);

        // Centered cube [-1,1] for all dimensions
        let centered_2d = try_generate_random_points::<2>(30, (-1.0, 1.0)).unwrap();
        let centered_3d = try_generate_random_points::<3>(30, (-1.0, 1.0)).unwrap();
        let centered_4d = try_generate_random_points::<4>(30, (-1.0, 1.0)).unwrap();
        let centered_5d = try_generate_random_points::<5>(30, (-1.0, 1.0)).unwrap();

        assert_eq!(centered_2d.len(), 30);
        assert_eq!(centered_3d.len(), 30);
        assert_eq!(centered_4d.len(), 30);
        assert_eq!(centered_5d.len(), 30);

        // Verify ranges for centered points
        for point in &centered_5d {
            let coords = *point.coords();
            for &coord in &coords {
                assert!((-1.0..1.0).contains(&coord));
            }
        }
    }

    // =============================================================================
    // GRID POINT GENERATION TESTS
    // =============================================================================

    #[test]
    fn test_generate_grid_points_2d() {
        // Test 2D grid generation
        let grid = generate_grid_points::<2>(nonzero(3), 1.0, [0.0, 0.0]).unwrap();

        assert_eq!(grid.len(), 9); // 3^2 = 9 points

        // Check that we get the expected coordinates
        let expected_coords = [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ];

        for point in &grid {
            let coords = *point.coords();
            // Grid generation order might vary, so check if point exists in expected set
            assert!(
                expected_coords.iter().any(|&expected| {
                    (coords[0] - expected[0]).abs() < 1e-10
                        && (coords[1] - expected[1]).abs() < 1e-10
                }),
                "Point {coords:?} not found in expected coordinates"
            );
        }
    }

    #[test]
    fn test_generate_grid_points_3d() {
        // Test 3D grid generation
        let grid = generate_grid_points::<3>(nonzero(2), 2.0, [1.0, 1.0, 1.0]).unwrap();

        assert_eq!(grid.len(), 8); // 2^3 = 8 points

        // Check that all points are within expected bounds
        for point in &grid {
            let coords = *point.coords();
            for &coord in &coords {
                assert!((1.0..=3.0).contains(&coord)); // offset 1.0 + (0 or 1) * spacing 2.0
            }
        }
    }

    #[test]
    fn test_generate_grid_points_4d() {
        // Test 4D grid generation
        let grid = generate_grid_points::<4>(nonzero(2), 0.5, [-0.5, -0.5, -0.5, -0.5]).unwrap();

        assert_eq!(grid.len(), 16); // 2^4 = 16 points

        // Check coordinate ranges
        for point in &grid {
            let coords = *point.coords();
            for &coord in &coords {
                assert!((-0.5..=0.0).contains(&coord)); // offset -0.5 + (0 or 1) * spacing 0.5
            }
        }
    }

    #[test]
    fn test_generate_grid_points_5d() {
        // Test 5D grid generation
        let grid = generate_grid_points::<5>(nonzero(2), 1.0, [0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();

        assert_eq!(grid.len(), 32); // 2^5 = 32 points

        // Check coordinate ranges
        for point in &grid {
            let coords = *point.coords();
            for &coord in &coords {
                assert!((0.0..=1.0).contains(&coord)); // offset 0.0 + (0 or 1) * spacing 1.0
            }
        }
    }

    #[test]
    fn test_generate_grid_points_edge_cases() {
        // Test single point grid
        let grid = generate_grid_points::<3>(nonzero(1), 1.0, [0.0, 0.0, 0.0]).unwrap();
        assert_eq!(grid.len(), 1);
        let coords = *grid[0].coords();
        // Use approx for floating point comparison
        for (actual, expected) in coords.iter().zip([0.0, 0.0, 0.0].iter()) {
            assert!((actual - expected).abs() < 1e-15);
        }

        let zero_spacing = generate_grid_points::<2>(nonzero(2), 0.0, [5.0, 5.0]);
        let Err(RandomPointGenerationError::InvalidGridSpacing {
            reason: InvalidPositiveScalar::NonPositive { value },
        }) = zero_spacing
        else {
            panic!("expected non-positive grid spacing error for zero spacing");
        };
        assert_relative_eq!(value, 0.0, epsilon = f64::EPSILON);

        let negative_spacing = generate_grid_points::<2>(nonzero(2), -1.0, [5.0, 5.0]);
        let Err(RandomPointGenerationError::InvalidGridSpacing {
            reason: InvalidPositiveScalar::NonPositive { value },
        }) = negative_spacing
        else {
            panic!("expected non-positive grid spacing error for negative spacing");
        };
        assert_relative_eq!(value, -1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_generate_grid_points_error_handling() {
        // Zero points per dimension cannot cross the typed API boundary.
        assert_eq!(NonZeroUsize::new(0), None);

        let nan_spacing = generate_grid_points::<2>(nonzero(2), f64::NAN, [0.0, 0.0]);
        assert_matches!(
            nan_spacing,
            Err(RandomPointGenerationError::InvalidGridSpacing {
                reason: InvalidPositiveScalar::NonFinite { value }
            }) if value == InvalidCoordinateValue::Nan
        );

        let infinite_offset = generate_grid_points::<3>(nonzero(2), 1.0, [0.0, f64::INFINITY, 0.0]);
        assert_matches!(
            infinite_offset,
            Err(RandomPointGenerationError::InvalidGridOffset { axis, value })
                if axis == 1 && value == InvalidCoordinateValue::PositiveInfinity
        );

        // Test safety cap for excessive points (prevents OOM)
        let result = generate_grid_points::<3>(nonzero(1000), 1.0, [0.0, 0.0, 0.0]);
        assert_matches!(
            result,
            Err(RandomPointGenerationError::GridAllocationTooLarge {
                required_bytes,
                cap_bytes,
            }) if required_bytes > cap_bytes
        );
    }

    #[test]
    fn test_generate_grid_points_rejects_non_finite_generated_coordinate() {
        let result = generate_grid_points::<1>(nonzero(2), f64::MAX, [f64::MAX]);

        assert_matches!(
            result,
            Err(RandomPointGenerationError::InvalidGeneratedGridCoordinate { axis, value })
                if axis == 0 && value == InvalidCoordinateValue::PositiveInfinity
        );
    }

    #[test]
    fn test_generate_grid_points_overflow_detection() {
        // Test overflow detection when points_per_dim^D would overflow usize
        // We'll use a dimension that would cause overflow
        const LARGE_D: usize = 64; // This will definitely cause overflow
        let offset = [0.0; LARGE_D];
        let spacing = 0.1; // This would require 10^64 points which overflows usize
        let points_per_dim = 10;

        let result = generate_grid_points::<LARGE_D>(nonzero(points_per_dim), spacing, offset);
        assert_matches!(
            result,
            Err(RandomPointGenerationError::GridSizeOverflow {
                points_per_dim: actual_points_per_dim,
                dimension,
            }) if actual_points_per_dim == points_per_dim && dimension == LARGE_D
        );
    }

    #[test]
    fn test_generate_grid_points_index_conversion_failure() {
        let first_inexact_f64_index = (1_usize << f64::MANTISSA_DIGITS) + 1;
        let idx = [first_inexact_f64_index];
        let result = grid_index_as_scalar::<1>(&idx, 0, first_inexact_f64_index.saturating_add(1));

        assert_matches!(
            result,
            Err(RandomPointGenerationError::CoordinateConversionFailed {
                value,
                target_type,
                ref source,
            }) if value == first_inexact_f64_index
                && target_type == "f64"
                && matches!(
                    source,
                    CoordinateConversionError::ConversionFailed {
                        coordinate_index: 0,
                        from_type: "usize",
                        to_type: "f64",
                        ..
                    }
                )
        );
    }

    // =============================================================================
    // POISSON POINT GENERATION TESTS
    // =============================================================================

    #[test]
    fn test_generate_poisson_points_2d() {
        // Test 2D Poisson disk sampling
        let points = try_generate_poisson_points::<2>(50, (0.0, 10.0), 0.5, 42).unwrap();

        // Should generate some points (exact count depends on spacing constraints)
        assert!(!points.is_empty());
        assert!(points.len() <= 50); // May be less than requested due to spacing constraints

        // Check that all points are within bounds
        for point in &points {
            let coords = *point.coords();
            assert!((0.0..10.0).contains(&coords[0]));
            assert!((0.0..10.0).contains(&coords[1]));
        }

        // Check minimum distance constraint
        for (i, p1) in points.iter().enumerate() {
            for (j, p2) in points.iter().enumerate() {
                if i != j {
                    let coords1 = *p1.coords();
                    let coords2 = *p2.coords();
                    let diff = [coords1[0] - coords2[0], coords1[1] - coords2[1]];
                    let distance = hypot(&diff);
                    assert!(
                        distance >= 0.5 - 1e-10,
                        "Distance {distance} violates minimum distance constraint"
                    );
                }
            }
        }
    }

    #[test]
    fn test_generate_poisson_points_3d() {
        // Test 3D Poisson disk sampling
        let points = try_generate_poisson_points::<3>(30, (-1.0, 1.0), 0.2, 123).unwrap();

        assert!(!points.is_empty());

        // Check bounds and minimum distance
        for point in &points {
            let coords = *point.coords();
            for &coord in &coords {
                assert!((-1.0..1.0).contains(&coord));
            }
        }

        // Check minimum distance constraint in 3D
        for (i, p1) in points.iter().enumerate() {
            for (j, p2) in points.iter().enumerate() {
                if i != j {
                    let coords1 = *p1.coords();
                    let coords2 = *p2.coords();
                    let diff = [
                        coords1[0] - coords2[0],
                        coords1[1] - coords2[1],
                        coords1[2] - coords2[2],
                    ];
                    let distance = hypot(&diff);
                    assert!(distance >= 0.2 - 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_generate_poisson_points_4d() {
        // Test 4D Poisson disk sampling
        let points = try_generate_poisson_points::<4>(15, (0.0, 5.0), 0.5, 333).unwrap();

        assert!(!points.is_empty());

        for point in &points {
            let coords = *point.coords();
            for &coord in &coords {
                assert!((0.0..5.0).contains(&coord));
            }
        }

        // Check minimum distance constraint in 4D
        for (i, p1) in points.iter().enumerate() {
            for (j, p2) in points.iter().enumerate() {
                if i != j {
                    let coords1 = *p1.coords();
                    let coords2 = *p2.coords();
                    let diff = [
                        coords1[0] - coords2[0],
                        coords1[1] - coords2[1],
                        coords1[2] - coords2[2],
                        coords1[3] - coords2[3],
                    ];
                    let distance = hypot(&diff);
                    assert!(distance >= 0.5 - 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_generate_poisson_points_5d() {
        // Test 5D Poisson disk sampling
        let points = try_generate_poisson_points::<5>(10, (-2.0, 2.0), 0.4, 777).unwrap();

        assert!(!points.is_empty());

        for point in &points {
            let coords = *point.coords();
            for &coord in &coords {
                assert!((-2.0..2.0).contains(&coord));
            }
        }

        // Check minimum distance constraint in 5D
        for (i, p1) in points.iter().enumerate() {
            for (j, p2) in points.iter().enumerate() {
                if i != j {
                    let coords1 = *p1.coords();
                    let coords2 = *p2.coords();
                    let diff = [
                        coords1[0] - coords2[0],
                        coords1[1] - coords2[1],
                        coords1[2] - coords2[2],
                        coords1[3] - coords2[3],
                        coords1[4] - coords2[4],
                    ];
                    let distance = hypot(&diff);
                    assert!(distance >= 0.4 - 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_generate_poisson_points_reproducible() {
        // Test that same seed produces same results
        let points1 = try_generate_poisson_points::<2>(25, (0.0, 5.0), 0.3, 456).unwrap();
        let points2 = try_generate_poisson_points::<2>(25, (0.0, 5.0), 0.3, 456).unwrap();

        assert_eq!(points1.len(), points2.len());

        for (p1, p2) in points1.iter().zip(points2.iter()) {
            let coords1 = *p1.coords();
            let coords2 = *p2.coords();

            for (c1, c2) in coords1.iter().zip(coords2.iter()) {
                assert_relative_eq!(c1, c2, epsilon = 1e-15);
            }
        }

        // Different seeds should produce different results
        let points3 = try_generate_poisson_points::<2>(25, (0.0, 5.0), 0.3, 789).unwrap();
        assert_ne!(points1, points3);
    }

    #[test]
    fn test_generate_poisson_points_error_handling() {
        // Test invalid range
        let result = try_generate_poisson_points::<2>(50, (10.0, 5.0), 0.1, 42);
        assert_invalid_coordinate_range(&result, CoordinateRangeOrdering::Decreasing, 10.0, 5.0);

        // Test non-finite minimum distance rejects at the public boundary.
        let nan_distance = try_generate_poisson_points::<2>(0, (0.0, 1.0), f64::NAN, 42);
        assert_matches!(
            nan_distance,
            Err(RandomPointGenerationError::InvalidMinimumDistance { distance })
                if distance == InvalidCoordinateValue::Nan
        );

        let infinite_distance = try_generate_poisson_points::<2>(50, (0.0, 1.0), f64::INFINITY, 42);
        assert_matches!(
            infinite_distance,
            Err(RandomPointGenerationError::InvalidMinimumDistance { distance })
                if distance == InvalidCoordinateValue::PositiveInfinity
        );

        // Test minimum distance too large for bounds (should produce few/no points)
        let result = try_generate_poisson_points::<2>(100, (0.0, 1.0), 10.0, 42);
        match result {
            Ok(points) => {
                // Should produce very few points or fail
                assert!(points.len() < 5);
            }
            Err(RandomPointGenerationError::PoissonSamplingFailed { .. }) => {
                // This is also acceptable - can't fit points with such large spacing
            }
            _ => panic!("Unexpected error type"),
        }

        // Test zero distance optimization (should return exact count without spacing checks)
        let points = try_generate_poisson_points::<2>(100, (0.0, 10.0), 0.0, 42).unwrap();
        assert_eq!(points.len(), 100); // Should get exactly the requested number

        // Test negative distance optimization (should return exact count without spacing checks)
        let points = try_generate_poisson_points::<2>(50, (0.0, 10.0), -1.0, 42).unwrap();
        assert_eq!(points.len(), 50); // Should get exactly the requested number
    }

    // =============================================================================
    // SAFETY CAP AND UTILITY FUNCTION TESTS
    // =============================================================================

    #[test]
    fn test_generate_random_points_invalid_range() {
        // Test invalid range (non-increasing bounds)
        let result = try_generate_random_points::<2>(100, (10.0, 5.0));
        assert_matches!(
            result,
            Err(RandomPointGenerationError::InvalidCoordinateRange { .. })
        );

        // Test equal min and max
        let result = try_generate_random_points::<2>(100, (5.0, 5.0));
        assert_matches!(
            result,
            Err(RandomPointGenerationError::InvalidCoordinateRange { .. })
        );

        // Test valid range
        let points = try_generate_random_points::<2>(10, (0.0, 1.0)).unwrap();
        assert_eq!(points.len(), 10);
    }

    #[test]
    fn test_generate_random_points_seeded_invalid_range() {
        // Test invalid range with seed
        let result = try_generate_random_points_seeded::<3>(50, (100.0, 10.0), 42);
        assert_matches!(
            result,
            Err(RandomPointGenerationError::InvalidCoordinateRange { .. })
        );

        // Test valid range produces consistent results
        let points1 = try_generate_random_points_seeded::<3>(5, (0.0, 1.0), 42).unwrap();
        let points2 = try_generate_random_points_seeded::<3>(5, (0.0, 1.0), 42).unwrap();
        assert_eq!(points1, points2);

        // Different seeds produce different results
        let points3 = try_generate_random_points_seeded::<3>(5, (0.0, 1.0), 123).unwrap();
        assert_ne!(points1, points3);
    }

    #[test]
    fn test_generate_grid_points_overflow_detection_edge_cases() {
        // Test cases that would cause potential memory issues in grid point calculation
        // Generate small grid to test the function works
        let points = generate_grid_points::<2>(nonzero(10), 0.1, [0.0, 0.0]).unwrap();
        assert!(!points.is_empty());

        // Test very fine spacing which would generate lots of points
        let result = generate_grid_points::<2>(nonzero(1000), 0.0001, [0.0, 0.0]);
        // Should either succeed or fail gracefully
        if let Ok(points) = result {
            assert!(!points.is_empty());
        }
        // Expected to fail with large point counts
    }

    #[test]
    fn test_generate_poisson_points_edge_cases() {
        // Test very small spacing with valid number of points
        let result = try_generate_poisson_points::<2>(100, (0.0, 1.0), 0.001, 42);
        if let Ok(points) = result {
            assert!(!points.is_empty());
        }
        // May fail due to too many points

        // Test with zero points (should succeed with empty result)
        let result = try_generate_poisson_points::<2>(0, (0.0, 1.0), 0.1, 42);
        if let Ok(points) = result {
            assert!(points.is_empty());
        }
        // Also acceptable if Err

        // Test very large spacing (should work but produce fewer points)
        let result = try_generate_poisson_points::<2>(10, (0.0, 1.0), 2.0, 42);
        if let Ok(points) = result {
            assert!(points.len() <= 10);
        }
        // May fail if spacing is too large for domain
    }

    #[test]
    fn test_max_grid_bytes_safety_cap() {
        let expected = env::var("MAX_GRID_BYTES_SAFETY_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(MAX_GRID_BYTES_SAFETY_CAP_DEFAULT);

        let cap = max_grid_bytes_safety_cap();
        assert!(cap > 0);
        assert_eq!(cap, expected);

        // Test that the default constant remains a reasonable value (4 GiB).
        assert_eq!(MAX_GRID_BYTES_SAFETY_CAP_DEFAULT, 4_294_967_296);
    }
}
