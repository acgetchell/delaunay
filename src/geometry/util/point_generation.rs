//! Random point generation functions.
//!
//! This module provides utilities for generating random points in d-dimensional
//! space with various distribution strategies.

use rand::Rng;
use rand::distr::uniform::SampleUniform;

use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{Coordinate, CoordinateScalar};

use super::conversions::safe_usize_to_scalar;
use super::norms::hypot;

// Re-export error type
pub use super::RandomPointGenerationError;

/// Default maximum bytes allowed for grid allocation to prevent OOM in CI.
///
/// This default safety cap prevents excessive memory allocation when generating grid points.
/// The limit of 4 GiB provides reasonable headroom for modern systems (GitHub Actions
/// runners have 7GB) while still protecting against extreme allocations.
///
/// The actual cap can be overridden via the `MAX_GRID_BYTES_SAFETY_CAP` environment variable.
const MAX_GRID_BYTES_SAFETY_CAP_DEFAULT: usize = 4_294_967_296; // 4 GiB

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
    if let Ok(v) = std::env::var("MAX_GRID_BYTES_SAFETY_CAP")
        && let Ok(n) = v.parse::<usize>()
    {
        return n;
    }
    MAX_GRID_BYTES_SAFETY_CAP_DEFAULT
}

/// Format bytes in human-readable form (e.g., "4.2 GiB", "512 MiB").
///
/// This helper function converts byte counts to human-readable strings
/// using binary prefixes (1024-based) for better UX in error messages.
fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];

    // Use safe cast to avoid precision loss warnings
    let Ok(mut size) = safe_usize_to_scalar::<f64>(bytes) else {
        // Fallback for extremely large values
        return format!("{bytes} B");
    };

    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Compute symmetric coordinate bounds scaled by point count.
///
/// This helper is intended for generating random point sets for triangulation construction
/// without "cramming" many points into a tiny bounding box like `[-1, 1]^D`, which can
/// increase the likelihood of near-degenerate configurations when using absolute tolerances.
///
/// It returns symmetric bounds `(-s/2, +s/2)` where `s = max(1, n_points)`.
///
/// # Errors
///
/// Returns `RandomPointGenerationError::RandomGenerationFailed` if `n_points` cannot be
/// converted to the coordinate type `T` without loss of precision.
///
/// # Examples
///
/// ```
/// use delaunay::geometry::util::scaled_bounds_by_point_count;
///
/// // 100 points -> side length 100, i.e. [-50, 50]
/// let bounds = scaled_bounds_by_point_count::<f64>(100).unwrap();
/// assert_eq!(bounds, (-50.0, 50.0));
/// ```
pub fn scaled_bounds_by_point_count<T: CoordinateScalar>(
    n_points: usize,
) -> Result<(T, T), RandomPointGenerationError> {
    let side_len = n_points.max(1);
    let side = safe_usize_to_scalar::<T>(side_len).map_err(|e| {
        RandomPointGenerationError::RandomGenerationFailed {
            min: "n/a".to_string(),
            max: "n/a".to_string(),
            details: format!(
                "Failed to convert n_points={side_len} to coordinate type {}: {e}",
                std::any::type_name::<T>()
            ),
        }
    })?;

    let half = side / (T::one() + T::one());
    Ok((-half, half))
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
/// Vector of random points with coordinates in the specified range,
/// or a `RandomPointGenerationError` if the parameters are invalid.
///
/// # Errors
///
/// * `RandomPointGenerationError::InvalidRange` if min >= max
///
/// # Examples
///
/// ```
/// use delaunay::geometry::util::generate_random_points;
///
/// // Generate 100 random 2D points with coordinates in [-10.0, 10.0]
/// let points_2d = generate_random_points::<f64, 2>(100, (-10.0, 10.0)).unwrap();
/// assert_eq!(points_2d.len(), 100);
///
/// // Generate 3D points with coordinates in [0.0, 1.0] (unit cube)
/// let points_3d = generate_random_points::<f64, 3>(50, (0.0, 1.0)).unwrap();
/// assert_eq!(points_3d.len(), 50);
///
/// // Generate 4D points centered around origin
/// let points_4d = generate_random_points::<f32, 4>(25, (-1.0, 1.0)).unwrap();
/// assert_eq!(points_4d.len(), 25);
///
/// // Error handling
/// let result = generate_random_points::<f64, 2>(100, (10.0, -10.0));
/// assert!(result.is_err()); // Invalid range
/// ```
pub fn generate_random_points<T: CoordinateScalar + SampleUniform, const D: usize>(
    n_points: usize,
    range: (T, T),
) -> Result<Vec<Point<T, D>>, RandomPointGenerationError> {
    #[cfg(debug_assertions)]
    if std::env::var_os("DELAUNAY_DEBUG_UNUSED_IMPORTS").is_some() {
        eprintln!("point_generation::generate_random_points called (n_points={n_points}, D={D})");
    }
    // Validate range
    if range.0 >= range.1 {
        return Err(RandomPointGenerationError::InvalidRange {
            min: format!("{:?}", range.0),
            max: format!("{:?}", range.1),
        });
    }

    let mut rng = rand::rng();
    let mut points = Vec::with_capacity(n_points);

    for _ in 0..n_points {
        let coords = [T::zero(); D].map(|_| rng.random_range(range.0..range.1));
        points.push(Point::new(coords));
    }

    Ok(points)
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
/// Vector of random points with coordinates in the specified range,
/// or a `RandomPointGenerationError` if the parameters are invalid.
///
/// # Errors
///
/// * `RandomPointGenerationError::InvalidRange` if min >= max
///
/// # Examples
///
/// ```
/// use delaunay::geometry::util::generate_random_points_seeded;
///
/// // Generate reproducible random points
/// let points1 = generate_random_points_seeded::<f64, 3>(100, (-5.0, 5.0), 42).unwrap();
/// let points2 = generate_random_points_seeded::<f64, 3>(100, (-5.0, 5.0), 42).unwrap();
/// assert_eq!(points1, points2); // Same seed produces identical results
///
/// // Different seeds produce different results
/// let points3 = generate_random_points_seeded::<f64, 3>(100, (-5.0, 5.0), 123).unwrap();
/// assert_ne!(points1, points3);
///
/// // Common ranges - unit cube [0,1]
/// let unit_points = generate_random_points_seeded::<f64, 3>(50, (0.0, 1.0), 42).unwrap();
///
/// // Centered around origin [-1,1]
/// let centered_points = generate_random_points_seeded::<f64, 3>(50, (-1.0, 1.0), 42).unwrap();
/// ```
pub fn generate_random_points_seeded<T: CoordinateScalar + SampleUniform, const D: usize>(
    n_points: usize,
    range: (T, T),
    seed: u64,
) -> Result<Vec<Point<T, D>>, RandomPointGenerationError> {
    use rand::SeedableRng;

    #[cfg(debug_assertions)]
    if std::env::var_os("DELAUNAY_DEBUG_UNUSED_IMPORTS").is_some() {
        eprintln!(
            "point_generation::generate_random_points_seeded called (n_points={n_points}, D={D}, seed={seed})"
        );
    }

    // Validate range
    if range.0 >= range.1 {
        return Err(RandomPointGenerationError::InvalidRange {
            min: format!("{:?}", range.0),
            max: format!("{:?}", range.1),
        });
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut points = Vec::with_capacity(n_points);

    for _ in 0..n_points {
        let coords = [T::zero(); D].map(|_| rng.random_range(range.0..range.1));
        points.push(Point::new(coords));
    }

    Ok(points)
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
/// * `points_per_dim` - Number of points along each dimension
/// * `spacing` - Distance between adjacent grid points
/// * `offset` - Translation offset for the entire grid
///
/// # Returns
///
/// Vector of grid points, or a `RandomPointGenerationError` if parameters are invalid.
///
/// # Errors
///
/// * `RandomPointGenerationError::InvalidPointCount` if `points_per_dim` is zero
///
/// # References
///
/// The mixed-radix counter algorithm is described in:
/// - D. E. Knuth, *The Art of Computer Programming, Vol. 4A: Combinatorial Algorithms*, Addison-Wesley, 2011.
///
/// # Examples
///
/// ```
/// use delaunay::geometry::util::generate_grid_points;
///
/// // Generate 2D grid: 4x4 = 16 points with unit spacing
/// let grid_2d = generate_grid_points::<f64, 2>(4, 1.0, [0.0, 0.0]).unwrap();
/// assert_eq!(grid_2d.len(), 16);
///
/// // Generate 3D grid: 3x3x3 = 27 points with spacing 2.0
/// let grid_3d = generate_grid_points::<f64, 3>(3, 2.0, [0.0, 0.0, 0.0]).unwrap();
/// assert_eq!(grid_3d.len(), 27);
///
/// // Generate 4D grid centered at origin
/// let grid_4d = generate_grid_points::<f64, 4>(2, 1.0, [-0.5, -0.5, -0.5, -0.5]).unwrap();
/// assert_eq!(grid_4d.len(), 16); // 2^4 = 16 points
/// ```
pub fn generate_grid_points<T: CoordinateScalar, const D: usize>(
    points_per_dim: usize,
    spacing: T,
    offset: [T; D],
) -> Result<Vec<Point<T, D>>, RandomPointGenerationError> {
    if points_per_dim == 0 {
        return Err(RandomPointGenerationError::InvalidPointCount { n_points: 0 });
    }

    // Compute total_points with overflow checking (avoids debug panic or release wrap)
    let mut total_points: usize = 1;
    for _ in 0..D {
        total_points = total_points.checked_mul(points_per_dim).ok_or_else(|| {
            RandomPointGenerationError::RandomGenerationFailed {
                min: "0".into(),
                max: format!("{}", points_per_dim.saturating_sub(1)),
                details: format!("Requested grid size {points_per_dim}^{D} overflows usize"),
            }
        })?;
    }

    // Dimension/type-aware memory cap: total_points * D * size_of::<T>()
    let per_point_bytes = D.saturating_mul(core::mem::size_of::<T>());
    let total_bytes = total_points.saturating_mul(per_point_bytes);
    let cap = max_grid_bytes_safety_cap();
    if total_bytes > cap {
        return Err(RandomPointGenerationError::RandomGenerationFailed {
            min: "n/a".into(),
            max: "n/a".into(),
            details: format!(
                "Requested grid requires {} (> cap {})",
                format_bytes(total_bytes),
                format_bytes(cap)
            ),
        });
    }
    let mut points = Vec::with_capacity(total_points);

    // Use mixed-radix counter over D dimensions (see Knuth TAOCP Vol 4A)
    // This avoids O(N) memory allocation for intermediate index vectors
    let mut idx = [0usize; D];
    for _ in 0..total_points {
        let mut coords = [T::zero(); D];
        for d in 0..D {
            let index_as_scalar = safe_usize_to_scalar::<T>(idx[d]).map_err(|_| {
                RandomPointGenerationError::RandomGenerationFailed {
                    min: "0".to_string(),
                    max: format!("{}", points_per_dim - 1),
                    details: format!("Failed to convert grid index {idx:?} to coordinate type"),
                }
            })?;
            coords[d] = offset[d] + index_as_scalar * spacing;
        }
        points.push(Point::new(coords));

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
/// * `min_distance` - Minimum distance between any two points
/// * `seed` - Seed for reproducible results
///
/// # Returns
///
/// Vector of Poisson-distributed points, or a `RandomPointGenerationError` if parameters are invalid.
/// Note: The actual number of points may be less than `n_points` due to spacing constraints.
///
/// # Errors
///
/// * `RandomPointGenerationError::InvalidRange` if min >= max in bounds
/// * `RandomPointGenerationError::RandomGenerationFailed` if `min_distance` is too large for the bounds
///   or if no points can be generated within the attempt limit
///
/// # Examples
///
/// ```
/// use delaunay::geometry::util::generate_poisson_points;
///
/// // Generate ~100 2D points with minimum distance 0.1 in unit square
/// let poisson_2d = generate_poisson_points::<f64, 2>(100, (0.0, 1.0), 0.1, 42).unwrap();
/// // Actual count may be less than 100 due to spacing constraints
///
/// // Generate 3D points in a cube
/// let poisson_3d = generate_poisson_points::<f64, 3>(50, (-1.0, 1.0), 0.2, 123).unwrap();
/// ```
pub fn generate_poisson_points<T: CoordinateScalar + SampleUniform, const D: usize>(
    n_points: usize,
    bounds: (T, T),
    min_distance: T,
    seed: u64,
) -> Result<Vec<Point<T, D>>, RandomPointGenerationError> {
    use rand::Rng;
    use rand::SeedableRng;

    // Validate bounds
    if bounds.0 >= bounds.1 {
        return Err(RandomPointGenerationError::InvalidRange {
            min: format!("{:?}", bounds.0),
            max: format!("{:?}", bounds.1),
        });
    }

    if n_points == 0 {
        return Ok(Vec::new());
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Early validation: if min_distance is non-positive, skip spacing constraints
    if min_distance <= T::zero() {
        let mut points = Vec::with_capacity(n_points);
        for _ in 0..n_points {
            let coords = [T::zero(); D].map(|_| rng.random_range(bounds.0..bounds.1));
            points.push(Point::new(coords));
        }
        return Ok(points);
    }

    let mut points: Vec<Point<T, D>> = Vec::new();

    // Simple Poisson disk sampling: rejection method
    // Scale max attempts with dimension since higher dimensions make spacing harder
    // Base: 30 attempts per point, scaled exponentially with dimension to account
    // for the curse of dimensionality in Poisson disk sampling
    let dimension_scaling = match D {
        0..=2 => 1,
        3..=4 => 2,
        5..=6 => 4,
        _ => 8, // Very high dimensions need much more attempts
    };
    let max_attempts = (n_points * 30).saturating_mul(dimension_scaling);
    let mut attempts = 0;

    while points.len() < n_points && attempts < max_attempts {
        attempts += 1;

        // Generate candidate point
        let coords = [T::zero(); D].map(|_| rng.random_range(bounds.0..bounds.1));
        let candidate = Point::new(coords);

        // Check distance to all existing points
        let mut valid = true;
        let candidate_coords: [T; D] = *candidate.coords();
        for existing_point in &points {
            let existing_coords: [T; D] = *existing_point.coords();

            // Calculate distance using hypot for numerical stability
            let mut diff_coords = [T::zero(); D];
            for i in 0..D {
                diff_coords[i] = candidate_coords[i] - existing_coords[i];
            }
            let distance = hypot(diff_coords);

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
        return Err(RandomPointGenerationError::RandomGenerationFailed {
            min: format!("{:?}", bounds.0),
            max: format!("{:?}", bounds.1),
            details: format!(
                "Could not generate any points with minimum distance {min_distance:?} in given bounds"
            ),
        });
    }

    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // =============================================================================
    // RANDOM POINT GENERATION TESTS
    // =============================================================================

    #[test]
    fn test_generate_random_points_2d() {
        // Test 2D random point generation
        let points = generate_random_points::<f64, 2>(100, (-10.0, 10.0)).unwrap();

        assert_eq!(points.len(), 100);

        // Check that all points are within range
        for point in &points {
            let coords: [f64; 2] = point.into();
            assert!(coords[0] >= -10.0 && coords[0] < 10.0);
            assert!(coords[1] >= -10.0 && coords[1] < 10.0);
        }
    }

    #[test]
    fn test_generate_random_points_3d() {
        // Test 3D random point generation
        let points = generate_random_points::<f64, 3>(75, (0.0, 5.0)).unwrap();

        assert_eq!(points.len(), 75);

        for point in &points {
            let coords: [f64; 3] = point.into();
            assert!(coords[0] >= 0.0 && coords[0] < 5.0);
            assert!(coords[1] >= 0.0 && coords[1] < 5.0);
            assert!(coords[2] >= 0.0 && coords[2] < 5.0);
        }
    }

    #[test]
    fn test_generate_random_points_4d() {
        // Test 4D random point generation
        let points = generate_random_points::<f32, 4>(50, (-2.0, 2.0)).unwrap();

        assert_eq!(points.len(), 50);

        for point in &points {
            let coords: [f32; 4] = point.into();
            for &coord in &coords {
                assert!((-2.0..2.0).contains(&coord));
            }
        }
    }

    #[test]
    fn test_generate_random_points_5d() {
        // Test 5D random point generation
        let points = generate_random_points::<f64, 5>(25, (-1.0, 1.0)).unwrap();

        assert_eq!(points.len(), 25);

        for point in &points {
            let coords: [f64; 5] = point.into();
            for &coord in &coords {
                assert!((-1.0..1.0).contains(&coord));
            }
        }
    }

    #[test]
    fn test_generate_random_points_error_handling() {
        // Test invalid range (min >= max) across all dimensions

        // 2D
        let result = generate_random_points::<f64, 2>(100, (10.0, -10.0));
        assert!(result.is_err());
        match result {
            Err(RandomPointGenerationError::InvalidRange { min, max }) => {
                assert_eq!(min, "10.0");
                assert_eq!(max, "-10.0");
            }
            _ => panic!("Expected InvalidRange error"),
        }

        // 3D
        let result = generate_random_points::<f64, 3>(50, (5.0, 5.0));
        assert!(result.is_err());

        // 4D
        let result = generate_random_points::<f32, 4>(25, (1.0, 0.5));
        assert!(result.is_err());

        // 5D
        let result = generate_random_points::<f64, 5>(10, (2.0, 2.0));
        assert!(result.is_err());

        // Test valid edge case - very small range
        let result = generate_random_points::<f64, 2>(10, (0.0, 0.001));
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_random_points_zero_points() {
        // Test generating zero points across all dimensions
        let points_2d = generate_random_points::<f64, 2>(0, (-1.0, 1.0)).unwrap();
        assert_eq!(points_2d.len(), 0);

        let points_3d = generate_random_points::<f64, 3>(0, (-1.0, 1.0)).unwrap();
        assert_eq!(points_3d.len(), 0);

        let points_4d = generate_random_points::<f64, 4>(0, (-1.0, 1.0)).unwrap();
        assert_eq!(points_4d.len(), 0);

        let points_5d = generate_random_points::<f64, 5>(0, (-1.0, 1.0)).unwrap();
        assert_eq!(points_5d.len(), 0);
    }

    #[test]
    fn test_generate_random_points_seeded_2d() {
        // Test seeded 2D generation reproducibility
        let seed = 42_u64;
        let points1 = generate_random_points_seeded::<f64, 2>(50, (-5.0, 5.0), seed).unwrap();
        let points2 = generate_random_points_seeded::<f64, 2>(50, (-5.0, 5.0), seed).unwrap();

        assert_eq!(points1.len(), points2.len());

        // Points should be identical with same seed
        for (p1, p2) in points1.iter().zip(points2.iter()) {
            let coords1: [f64; 2] = p1.into();
            let coords2: [f64; 2] = p2.into();

            for (c1, c2) in coords1.iter().zip(coords2.iter()) {
                assert_relative_eq!(c1, c2, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn test_generate_random_points_seeded_3d() {
        // Test seeded 3D generation reproducibility
        let seed = 123_u64;
        let points1 = generate_random_points_seeded::<f64, 3>(40, (0.0, 10.0), seed).unwrap();
        let points2 = generate_random_points_seeded::<f64, 3>(40, (0.0, 10.0), seed).unwrap();

        assert_eq!(points1.len(), points2.len());

        for (p1, p2) in points1.iter().zip(points2.iter()) {
            let coords1: [f64; 3] = p1.into();
            let coords2: [f64; 3] = p2.into();

            for (c1, c2) in coords1.iter().zip(coords2.iter()) {
                assert_relative_eq!(c1, c2, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn test_generate_random_points_seeded_4d() {
        // Test seeded 4D generation reproducibility
        let seed = 789_u64;
        let points1 = generate_random_points_seeded::<f32, 4>(30, (-2.5, 2.5), seed).unwrap();
        let points2 = generate_random_points_seeded::<f32, 4>(30, (-2.5, 2.5), seed).unwrap();

        assert_eq!(points1.len(), points2.len());

        for (p1, p2) in points1.iter().zip(points2.iter()) {
            let coords1: [f32; 4] = p1.into();
            let coords2: [f32; 4] = p2.into();

            for (c1, c2) in coords1.iter().zip(coords2.iter()) {
                assert_relative_eq!(c1, c2, epsilon = 1e-6); // f32 precision
            }
        }
    }

    #[test]
    fn test_generate_random_points_seeded_5d() {
        // Test seeded 5D generation reproducibility
        let seed = 456_u64;
        let points1 = generate_random_points_seeded::<f64, 5>(20, (-1.0, 3.0), seed).unwrap();
        let points2 = generate_random_points_seeded::<f64, 5>(20, (-1.0, 3.0), seed).unwrap();

        assert_eq!(points1.len(), points2.len());

        for (p1, p2) in points1.iter().zip(points2.iter()) {
            let coords1: [f64; 5] = p1.into();
            let coords2: [f64; 5] = p2.into();

            for (c1, c2) in coords1.iter().zip(coords2.iter()) {
                assert_relative_eq!(c1, c2, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn test_generate_random_points_seeded_different_seeds() {
        // Test that different seeds produce different results across all dimensions

        // 2D
        let points1_2d = generate_random_points_seeded::<f64, 2>(50, (0.0, 1.0), 42).unwrap();
        let points2_2d = generate_random_points_seeded::<f64, 2>(50, (0.0, 1.0), 123).unwrap();
        assert_ne!(points1_2d, points2_2d);

        // 3D
        let points1_3d = generate_random_points_seeded::<f64, 3>(30, (-5.0, 5.0), 42).unwrap();
        let points2_3d = generate_random_points_seeded::<f64, 3>(30, (-5.0, 5.0), 999).unwrap();
        assert_ne!(points1_3d, points2_3d);

        // 4D
        let points1_4d = generate_random_points_seeded::<f32, 4>(25, (-1.0, 1.0), 1337).unwrap();
        let points2_4d = generate_random_points_seeded::<f32, 4>(25, (-1.0, 1.0), 7331).unwrap();
        assert_ne!(points1_4d, points2_4d);

        // 5D
        let points1_5d = generate_random_points_seeded::<f64, 5>(15, (0.0, 10.0), 2021).unwrap();
        let points2_5d = generate_random_points_seeded::<f64, 5>(15, (0.0, 10.0), 2024).unwrap();
        assert_ne!(points1_5d, points2_5d);
    }

    #[test]
    fn test_generate_random_points_distribution_coverage_all_dimensions() {
        // Test that points cover the range reasonably well across all dimensions

        // 2D coverage test
        let points_2d = generate_random_points::<f64, 2>(500, (0.0, 10.0)).unwrap();
        let mut min_2d = [f64::INFINITY; 2];
        let mut max_2d = [f64::NEG_INFINITY; 2];

        for point in &points_2d {
            let coords: [f64; 2] = point.into();
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
        let points_5d = generate_random_points::<f64, 5>(200, (-5.0, 5.0)).unwrap();
        let mut min_5d = [f64::INFINITY; 5];
        let mut max_5d = [f64::NEG_INFINITY; 5];

        for point in &points_5d {
            let coords: [f64; 5] = point.into();
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
        let unit_2d = generate_random_points::<f64, 2>(50, (0.0, 1.0)).unwrap();
        let unit_3d = generate_random_points::<f64, 3>(50, (0.0, 1.0)).unwrap();
        let unit_4d = generate_random_points::<f64, 4>(50, (0.0, 1.0)).unwrap();
        let unit_5d = generate_random_points::<f64, 5>(50, (0.0, 1.0)).unwrap();

        assert_eq!(unit_2d.len(), 50);
        assert_eq!(unit_3d.len(), 50);
        assert_eq!(unit_4d.len(), 50);
        assert_eq!(unit_5d.len(), 50);

        // Centered cube [-1,1] for all dimensions
        let centered_2d = generate_random_points::<f64, 2>(30, (-1.0, 1.0)).unwrap();
        let centered_3d = generate_random_points::<f64, 3>(30, (-1.0, 1.0)).unwrap();
        let centered_4d = generate_random_points::<f64, 4>(30, (-1.0, 1.0)).unwrap();
        let centered_5d = generate_random_points::<f64, 5>(30, (-1.0, 1.0)).unwrap();

        assert_eq!(centered_2d.len(), 30);
        assert_eq!(centered_3d.len(), 30);
        assert_eq!(centered_4d.len(), 30);
        assert_eq!(centered_5d.len(), 30);

        // Verify ranges for centered points
        for point in &centered_5d {
            let coords: [f64; 5] = point.into();
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
        let grid = generate_grid_points::<f64, 2>(3, 1.0, [0.0, 0.0]).unwrap();

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
            let coords: [f64; 2] = point.into();
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
        let grid = generate_grid_points::<f64, 3>(2, 2.0, [1.0, 1.0, 1.0]).unwrap();

        assert_eq!(grid.len(), 8); // 2^3 = 8 points

        // Check that all points are within expected bounds
        for point in &grid {
            let coords: [f64; 3] = point.into();
            for &coord in &coords {
                assert!((1.0..=3.0).contains(&coord)); // offset 1.0 + (0 or 1) * spacing 2.0
            }
        }
    }

    #[test]
    fn test_generate_grid_points_4d() {
        // Test 4D grid generation
        let grid = generate_grid_points::<f32, 4>(2, 0.5, [-0.5, -0.5, -0.5, -0.5]).unwrap();

        assert_eq!(grid.len(), 16); // 2^4 = 16 points

        // Check coordinate ranges
        for point in &grid {
            let coords: [f32; 4] = point.into();
            for &coord in &coords {
                assert!((-0.5..=0.0).contains(&coord)); // offset -0.5 + (0 or 1) * spacing 0.5
            }
        }
    }

    #[test]
    fn test_generate_grid_points_5d() {
        // Test 5D grid generation
        let grid = generate_grid_points::<f64, 5>(2, 1.0, [0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();

        assert_eq!(grid.len(), 32); // 2^5 = 32 points

        // Check coordinate ranges
        for point in &grid {
            let coords: [f64; 5] = point.into();
            for &coord in &coords {
                assert!((0.0..=1.0).contains(&coord)); // offset 0.0 + (0 or 1) * spacing 1.0
            }
        }
    }

    #[test]
    fn test_generate_grid_points_edge_cases() {
        // Test single point grid
        let grid = generate_grid_points::<f64, 3>(1, 1.0, [0.0, 0.0, 0.0]).unwrap();
        assert_eq!(grid.len(), 1);
        let coords: [f64; 3] = (&grid[0]).into();
        // Use approx for floating point comparison
        for (actual, expected) in coords.iter().zip([0.0, 0.0, 0.0].iter()) {
            assert!((actual - expected).abs() < 1e-15);
        }

        // Test zero spacing
        let grid = generate_grid_points::<f64, 2>(2, 0.0, [5.0, 5.0]).unwrap();
        assert_eq!(grid.len(), 4);
        for point in &grid {
            let coords: [f64; 2] = point.into();
            // Use approx for floating point comparison
            for (actual, expected) in coords.iter().zip([5.0, 5.0].iter()) {
                assert!((actual - expected).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_generate_grid_points_error_handling() {
        // Test zero points per dimension
        let result = generate_grid_points::<f64, 2>(0, 1.0, [0.0, 0.0]);
        assert!(result.is_err());
        match result {
            Err(RandomPointGenerationError::InvalidPointCount { n_points }) => {
                assert_eq!(n_points, 0);
            }
            _ => panic!("Expected InvalidPointCount error"),
        }

        // Test safety cap for excessive points (prevents OOM)
        let result = generate_grid_points::<f64, 3>(1000, 1.0, [0.0, 0.0, 0.0]);
        assert!(result.is_err());
        let error_msg = format!("{}", result.unwrap_err());
        assert!(error_msg.contains("cap"));
        // Should contain human-readable byte formatting (no longer contains raw "bytes")
        assert!(
            error_msg.contains("GiB") || error_msg.contains("MiB") || error_msg.contains("KiB"),
            "Error message should contain human-readable byte units: {error_msg}"
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

        let result = generate_grid_points::<f64, LARGE_D>(points_per_dim, spacing, offset);
        assert!(result.is_err(), "Expected error due to usize overflow");

        if let Err(RandomPointGenerationError::RandomGenerationFailed {
            min: _,
            max: _,
            details,
        }) = result
        {
            assert!(
                details.contains("overflows usize"),
                "Expected overflow error, got: {details}"
            );
        } else {
            panic!("Expected RandomGenerationFailed error due to overflow");
        }
    }

    // =============================================================================
    // POISSON POINT GENERATION TESTS
    // =============================================================================

    #[test]
    fn test_generate_poisson_points_2d() {
        // Test 2D Poisson disk sampling
        let points = generate_poisson_points::<f64, 2>(50, (0.0, 10.0), 0.5, 42).unwrap();

        // Should generate some points (exact count depends on spacing constraints)
        assert!(!points.is_empty());
        assert!(points.len() <= 50); // May be less than requested due to spacing constraints

        // Check that all points are within bounds
        for point in &points {
            let coords: [f64; 2] = point.into();
            assert!((0.0..10.0).contains(&coords[0]));
            assert!((0.0..10.0).contains(&coords[1]));
        }

        // Check minimum distance constraint
        for (i, p1) in points.iter().enumerate() {
            for (j, p2) in points.iter().enumerate() {
                if i != j {
                    let coords1: [f64; 2] = p1.into();
                    let coords2: [f64; 2] = p2.into();
                    let diff = [coords1[0] - coords2[0], coords1[1] - coords2[1]];
                    let distance = hypot(diff);
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
        let points = generate_poisson_points::<f64, 3>(30, (-1.0, 1.0), 0.2, 123).unwrap();

        assert!(!points.is_empty());

        // Check bounds and minimum distance
        for point in &points {
            let coords: [f64; 3] = point.into();
            for &coord in &coords {
                assert!((-1.0..1.0).contains(&coord));
            }
        }

        // Check minimum distance constraint in 3D
        for (i, p1) in points.iter().enumerate() {
            for (j, p2) in points.iter().enumerate() {
                if i != j {
                    let coords1: [f64; 3] = p1.into();
                    let coords2: [f64; 3] = p2.into();
                    let diff = [
                        coords1[0] - coords2[0],
                        coords1[1] - coords2[1],
                        coords1[2] - coords2[2],
                    ];
                    let distance = hypot(diff);
                    assert!(distance >= 0.2 - 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_generate_poisson_points_4d() {
        // Test 4D Poisson disk sampling
        let points =
            generate_poisson_points::<f32, 4>(15, (0.0_f32, 5.0_f32), 0.5_f32, 333).unwrap();

        assert!(!points.is_empty());

        for point in &points {
            let coords: [f32; 4] = point.into();
            for &coord in &coords {
                assert!((0.0..5.0).contains(&coord));
            }
        }

        // Check minimum distance constraint in 4D
        for (i, p1) in points.iter().enumerate() {
            for (j, p2) in points.iter().enumerate() {
                if i != j {
                    let coords1: [f32; 4] = p1.into();
                    let coords2: [f32; 4] = p2.into();
                    let diff = [
                        coords1[0] - coords2[0],
                        coords1[1] - coords2[1],
                        coords1[2] - coords2[2],
                        coords1[3] - coords2[3],
                    ];
                    let distance = hypot(diff);
                    assert!(distance >= 0.5 - 1e-6); // f32 precision
                }
            }
        }
    }

    #[test]
    fn test_generate_poisson_points_5d() {
        // Test 5D Poisson disk sampling
        let points = generate_poisson_points::<f64, 5>(10, (-2.0, 2.0), 0.4, 777).unwrap();

        assert!(!points.is_empty());

        for point in &points {
            let coords: [f64; 5] = point.into();
            for &coord in &coords {
                assert!((-2.0..2.0).contains(&coord));
            }
        }

        // Check minimum distance constraint in 5D
        for (i, p1) in points.iter().enumerate() {
            for (j, p2) in points.iter().enumerate() {
                if i != j {
                    let coords1: [f64; 5] = p1.into();
                    let coords2: [f64; 5] = p2.into();
                    let diff = [
                        coords1[0] - coords2[0],
                        coords1[1] - coords2[1],
                        coords1[2] - coords2[2],
                        coords1[3] - coords2[3],
                        coords1[4] - coords2[4],
                    ];
                    let distance = hypot(diff);
                    assert!(distance >= 0.4 - 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_generate_poisson_points_reproducible() {
        // Test that same seed produces same results
        let points1 = generate_poisson_points::<f64, 2>(25, (0.0, 5.0), 0.3, 456).unwrap();
        let points2 = generate_poisson_points::<f64, 2>(25, (0.0, 5.0), 0.3, 456).unwrap();

        assert_eq!(points1.len(), points2.len());

        for (p1, p2) in points1.iter().zip(points2.iter()) {
            let coords1: [f64; 2] = p1.into();
            let coords2: [f64; 2] = p2.into();

            for (c1, c2) in coords1.iter().zip(coords2.iter()) {
                assert_relative_eq!(c1, c2, epsilon = 1e-15);
            }
        }

        // Different seeds should produce different results
        let points3 = generate_poisson_points::<f64, 2>(25, (0.0, 5.0), 0.3, 789).unwrap();
        assert_ne!(points1, points3);
    }

    #[test]
    fn test_generate_poisson_points_error_handling() {
        // Test invalid range
        let result = generate_poisson_points::<f64, 2>(50, (10.0, 5.0), 0.1, 42);
        assert!(result.is_err());
        match result {
            Err(RandomPointGenerationError::InvalidRange { min, max }) => {
                assert_eq!(min, "10.0");
                assert_eq!(max, "5.0");
            }
            _ => panic!("Expected InvalidRange error"),
        }

        // Test minimum distance too large for bounds (should produce few/no points)
        let result = generate_poisson_points::<f64, 2>(100, (0.0, 1.0), 10.0, 42);
        match result {
            Ok(points) => {
                // Should produce very few points or fail
                assert!(points.len() < 5);
            }
            Err(RandomPointGenerationError::RandomGenerationFailed { .. }) => {
                // This is also acceptable - can't fit points with such large spacing
            }
            _ => panic!("Unexpected error type"),
        }

        // Test zero distance optimization (should return exact count without spacing checks)
        let result = generate_poisson_points::<f64, 2>(100, (0.0, 10.0), 0.0, 42);
        assert!(result.is_ok());
        let points = result.unwrap();
        assert_eq!(points.len(), 100); // Should get exactly the requested number

        // Test negative distance optimization (should return exact count without spacing checks)
        let result = generate_poisson_points::<f64, 2>(50, (0.0, 10.0), -1.0, 42);
        assert!(result.is_ok());
        let points = result.unwrap();
        assert_eq!(points.len(), 50); // Should get exactly the requested number
    }

    // =============================================================================
    // SAFETY CAP AND UTILITY FUNCTION TESTS
    // =============================================================================

    #[test]
    fn test_generate_random_points_invalid_range() {
        // Test invalid range (min >= max)
        let result = generate_random_points::<f64, 2>(100, (10.0, 5.0));
        assert!(matches!(
            result,
            Err(RandomPointGenerationError::InvalidRange { .. })
        ));

        // Test equal min and max
        let result = generate_random_points::<f64, 2>(100, (5.0, 5.0));
        assert!(matches!(
            result,
            Err(RandomPointGenerationError::InvalidRange { .. })
        ));

        // Test valid range
        let result = generate_random_points::<f64, 2>(10, (0.0, 1.0));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 10);
    }

    #[test]
    fn test_generate_random_points_seeded_invalid_range() {
        // Test invalid range with seed
        let result = generate_random_points_seeded::<f64, 3>(50, (100.0, 10.0), 42);
        assert!(matches!(
            result,
            Err(RandomPointGenerationError::InvalidRange { .. })
        ));

        // Test valid range produces consistent results
        let points1 = generate_random_points_seeded::<f64, 3>(5, (0.0, 1.0), 42).unwrap();
        let points2 = generate_random_points_seeded::<f64, 3>(5, (0.0, 1.0), 42).unwrap();
        assert_eq!(points1, points2);

        // Different seeds produce different results
        let points3 = generate_random_points_seeded::<f64, 3>(5, (0.0, 1.0), 123).unwrap();
        assert_ne!(points1, points3);
    }

    #[test]
    fn test_generate_grid_points_overflow_detection_edge_cases() {
        // Test cases that would cause potential memory issues in grid point calculation
        // Generate small grid to test the function works
        let result = generate_grid_points::<f64, 2>(10, 0.1, [0.0, 0.0]);
        assert!(result.is_ok());

        // Test very fine spacing which would generate lots of points
        let result = generate_grid_points::<f64, 2>(1000, 0.0001, [0.0, 0.0]);
        // Should either succeed or fail gracefully
        if let Ok(points) = result {
            assert!(!points.is_empty());
        }
        // Expected to fail with large point counts
    }

    #[test]
    fn test_generate_poisson_points_edge_cases() {
        // Test very small spacing with valid number of points
        let result = generate_poisson_points::<f64, 2>(100, (0.0, 1.0), 0.001, 42);
        if let Ok(points) = result {
            assert!(!points.is_empty());
        }
        // May fail due to too many points

        // Test with zero points (should succeed with empty result)
        let result = generate_poisson_points::<f64, 2>(0, (0.0, 1.0), 0.1, 42);
        if let Ok(points) = result {
            assert!(points.is_empty());
        }
        // Also acceptable if Err

        // Test very large spacing (should work but produce fewer points)
        let result = generate_poisson_points::<f64, 2>(10, (0.0, 1.0), 2.0, 42);
        if let Ok(points) = result {
            assert!(points.len() <= 10);
        }
        // May fail if spacing is too large for domain
    }

    #[test]
    fn test_format_bytes_edge_cases() {
        // Test additional edge cases for byte formatting
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1), "1 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1.0 KiB");
        assert_eq!(format_bytes(1536), "1.5 KiB"); // 1024 + 512
        assert_eq!(format_bytes(1024 * 1024), "1.0 MiB");

        // Test larger values
        let large_bytes = 7 * 1024 * 1024 * 1024; // 7 GiB
        let formatted = format_bytes(large_bytes);
        assert!(formatted.contains("GiB"));
        assert!(formatted.contains("7."));
    }

    #[test]
    fn test_max_grid_bytes_safety_cap() {
        let expected = std::env::var("MAX_GRID_BYTES_SAFETY_CAP")
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
