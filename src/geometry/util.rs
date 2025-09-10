//! Geometric utility functions for d-dimensional geometry calculations.
//!
//! This module contains utility functions for computing distances, norms, and
//! circumsphere properties of geometric objects. These functions are used by
//! both predicates and other geometric algorithms.

use num_traits::{Float, Zero};
use peroxide::fuga::{LinearAlgebra, MatrixTrait, zeros};
use peroxide::statistics::ops::factorial;
use rand::Rng;
use rand::distr::uniform::SampleUniform;
use std::iter::Sum;

use crate::core::traits::data_type::DataType;
use crate::core::triangulation_data_structure::Tds;
use crate::core::vertex::Vertex;
use crate::geometry::matrix::{MatrixError, invert};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{
    Coordinate, CoordinateConversionError, CoordinateScalar,
};
use num_traits::cast;

/// Errors that can occur during value type conversions.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ValueConversionError {
    /// Failed to convert a value from one type to another
    #[error("Cannot convert {value} from {from_type} to {to_type}: {details}")]
    ConversionFailed {
        /// The value that failed to convert (as string for display)
        value: String,
        /// Source type name
        from_type: &'static str,
        /// Target type name
        to_type: &'static str,
        /// Additional details about the failure
        details: String,
    },
}

/// Errors that can occur during random point generation.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum RandomPointGenerationError {
    /// Invalid coordinate range provided
    #[error("Invalid coordinate range: minimum {min} must be less than maximum {max}")]
    InvalidRange {
        /// The minimum value of the range
        min: String,
        /// The maximum value of the range
        max: String,
    },

    /// Failed to generate random value within range
    #[error("Failed to generate random value in range [{min}, {max}]: {details}")]
    RandomGenerationFailed {
        /// The minimum value of the range
        min: String,
        /// The maximum value of the range
        max: String,
        /// Additional details about the failure
        details: String,
    },

    /// Invalid number of points requested
    #[error("Invalid number of points: {n_points} (must be non-negative)")]
    InvalidPointCount {
        /// The invalid number of points requested
        n_points: isize,
    },
}

/// Errors that can occur during circumcenter calculation.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum CircumcenterError {
    /// Empty point set provided
    #[error("Empty point set")]
    EmptyPointSet,

    /// Points do not form a valid simplex
    #[error(
        "Points do not form a valid simplex: expected {expected} points for dimension {dimension}, got {actual}"
    )]
    InvalidSimplex {
        /// Number of points provided
        actual: usize,
        /// Number of points expected (D+1)
        expected: usize,
        /// Dimension
        dimension: usize,
    },

    /// Matrix inversion failed (degenerate simplex)
    #[error("Matrix inversion failed: {details}")]
    MatrixInversionFailed {
        /// Details about the matrix inversion failure
        details: String,
    },

    /// Matrix operation error
    #[error("Matrix error: {0}")]
    MatrixError(#[from] MatrixError),

    /// Array conversion failed
    #[error("Array conversion failed: {details}")]
    ArrayConversionFailed {
        /// Details about the array conversion failure
        details: String,
    },

    /// Coordinate conversion error
    #[error("Coordinate conversion error: {0}")]
    CoordinateConversion(#[from] CoordinateConversionError),

    /// Value conversion error
    #[error("Value conversion error: {0}")]
    ValueConversion(#[from] ValueConversionError),
}

// =============================================================================
// CONSTANTS AND HELPERS
// =============================================================================

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

// =============================================================================
// SAFE COORDINATE CONVERSION FUNCTIONS
// =============================================================================

/// Compute 2D hypot using numerically stable scaled algorithm.
///
/// This is used as a fallback when standard library hypot conversion fails.
/// It implements the same scaling approach used in the general hypot algorithm.
///
/// # Arguments
///
/// * `x` - First coordinate
/// * `y` - Second coordinate
///
/// # Returns
///
/// The computed hypot value using scaled computation
fn scaled_hypot_2d<T: CoordinateScalar + num_traits::Zero>(x: T, y: T) -> T {
    let max_abs = Float::abs(x).max(Float::abs(y));
    if max_abs == T::zero() {
        return T::zero();
    }
    // Use scaled computation for numerical stability
    let x_scaled = x / max_abs;
    let y_scaled = y / max_abs;
    max_abs * Float::sqrt(x_scaled * x_scaled + y_scaled * y_scaled)
}

/// Safely convert a coordinate value from type T to f64.
/// This function provides proper error handling for coordinate type conversions,
/// replacing the unsafe `cast(x).unwrap_or(fallback)` pattern with explicit
/// error reporting. It also checks for non-finite values (NaN, infinity).
///
/// # Arguments
///
/// * `value` - The coordinate value to convert
/// * `coordinate_index` - Index of the coordinate for error reporting
///
/// # Returns
///
/// The converted f64 value or a `CoordinateConversionError`
///
/// # Errors
///
/// Returns `CoordinateConversionError::NonFiniteValue` if the value is NaN or infinite
/// Returns `CoordinateConversionError::ConversionFailed` if the conversion fails
fn safe_cast_to_f64<T: CoordinateScalar>(
    value: T,
    coordinate_index: usize,
) -> Result<f64, CoordinateConversionError> {
    // Check for non-finite values first
    if !value.is_finite_generic() {
        return Err(CoordinateConversionError::NonFiniteValue {
            coordinate_index,
            coordinate_value: format!("{value:?}"),
        });
    }

    cast(value).ok_or_else(|| CoordinateConversionError::ConversionFailed {
        coordinate_index,
        coordinate_value: format!("{value:?}"),
        from_type: std::any::type_name::<T>(),
        to_type: "f64",
    })
}

/// Safely convert a coordinate value from f64 to type T.
///
/// This function provides proper error handling for coordinate type conversions,
/// replacing the unsafe `cast(x).unwrap_or(fallback)` pattern with explicit
/// error reporting. It also checks for non-finite values (NaN, infinity).
///
/// # Arguments
///
/// * `value` - The f64 value to convert
/// * `coordinate_index` - Index of the coordinate for error reporting
///
/// # Returns
///
/// The converted T value or a `CoordinateConversionError`
///
/// # Errors
///
/// Returns `CoordinateConversionError::NonFiniteValue` if the value is NaN or infinite
/// Returns `CoordinateConversionError::ConversionFailed` if the conversion fails
fn safe_cast_from_f64<T: CoordinateScalar>(
    value: f64,
    coordinate_index: usize,
) -> Result<T, CoordinateConversionError> {
    // Check for non-finite values first
    if !value.is_finite() {
        return Err(CoordinateConversionError::NonFiniteValue {
            coordinate_index,
            coordinate_value: format!("{value:?}"),
        });
    }

    cast(value).ok_or_else(|| CoordinateConversionError::ConversionFailed {
        coordinate_index,
        coordinate_value: format!("{value:?}"),
        from_type: "f64",
        to_type: std::any::type_name::<T>(),
    })
}

/// Safely convert an array of coordinates from type T to f64.
///
/// This function converts each coordinate in the array, providing detailed
/// error information if any conversion fails.
///
/// # Arguments
///
/// * `coords` - Array of coordinates to convert
///
/// # Returns
///
/// Array of f64 coordinates or a `CoordinateConversionError`
///
/// # Errors
///
/// Returns `CoordinateConversionError::ConversionFailed` if any coordinate conversion fails
///
/// # Examples
///
/// ```
/// use delaunay::geometry::util::safe_coords_to_f64;
///
/// // Convert f32 coordinates to f64
/// let coords_f32 = [1.5f32, 2.5f32, 3.5f32];
/// let coords_f64 = safe_coords_to_f64(coords_f32).unwrap();
/// assert_eq!(coords_f64, [1.5f64, 2.5f64, 3.5f64]);
///
/// // Works with different array sizes - 4D example
/// let coords_4d = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
/// let result_4d = safe_coords_to_f64(coords_4d).unwrap();
/// assert_eq!(result_4d, [1.0f64, 2.0f64, 3.0f64, 4.0f64]);
/// ```
pub fn safe_coords_to_f64<T: CoordinateScalar, const D: usize>(
    coords: [T; D],
) -> Result<[f64; D], CoordinateConversionError> {
    let mut result = [0.0_f64; D];
    for (i, &coord) in coords.iter().enumerate() {
        result[i] = safe_cast_to_f64(coord, i)?;
    }
    Ok(result)
}

/// Safely convert an array of coordinates from f64 to type T.
///
/// This function converts each coordinate in the array, providing detailed
/// error information if any conversion fails.
///
/// # Arguments
///
/// * `coords` - Array of f64 coordinates to convert
///
/// # Returns
///
/// Array of T coordinates or a `CoordinateConversionError`
///
/// # Errors
///
/// Returns `CoordinateConversionError::ConversionFailed` if any coordinate conversion fails
///
/// # Examples
///
/// ```
/// use delaunay::geometry::util::safe_coords_from_f64;
///
/// // Convert f64 coordinates to f32
/// let coords_f64 = [1.5f64, 2.5f64, 3.5f64];
/// let coords_f32: [f32; 3] = safe_coords_from_f64(coords_f64).unwrap();
/// assert_eq!(coords_f32, [1.5f32, 2.5f32, 3.5f32]);
///
/// // Works with different array sizes - 4D example
/// let coords_4d = [1.0f64, 2.0f64, 3.0f64, 4.0f64];
/// let result_4d: [f32; 4] = safe_coords_from_f64(coords_4d).unwrap();
/// assert_eq!(result_4d, [1.0f32, 2.0f32, 3.0f32, 4.0f32]);
/// ```
pub fn safe_coords_from_f64<T: CoordinateScalar, const D: usize>(
    coords: [f64; D],
) -> Result<[T; D], CoordinateConversionError> {
    let mut result = [T::zero(); D];
    for (i, &coord) in coords.iter().enumerate() {
        result[i] = safe_cast_from_f64(coord, i)?;
    }
    Ok(result)
}

/// Safely convert a single scalar value from type T to f64.
///
/// This is a convenience function for converting single values with proper error handling.
/// Unlike basic casting, this function checks for non-finite values (NaN, infinity) and
/// provides detailed error information if the conversion fails.
///
/// # Arguments
///
/// * `value` - The value to convert
///
/// # Returns
///
/// The converted f64 value or a `CoordinateConversionError`
///
/// # Errors
///
/// Returns `CoordinateConversionError::NonFiniteValue` if the value is NaN or infinite
/// Returns `CoordinateConversionError::ConversionFailed` if the conversion fails
///
/// # Example
///
/// ```
/// use delaunay::geometry::util::safe_scalar_to_f64;
///
/// let value_f32 = 42.5f32;
/// let value_f64 = safe_scalar_to_f64(value_f32).unwrap();
/// assert_eq!(value_f64, 42.5f64);
/// ```
pub fn safe_scalar_to_f64<T: CoordinateScalar>(value: T) -> Result<f64, CoordinateConversionError> {
    safe_cast_to_f64(value, 0)
}

/// Safely convert a single scalar value from f64 to type T.
///
/// This is a convenience function for converting single values with proper error handling.
///
/// # Arguments
///
/// * `value` - The f64 value to convert
///
/// # Returns
///
/// The converted T value or a `CoordinateConversionError`
///
/// # Errors
///
/// Returns `CoordinateConversionError::ConversionFailed` if the conversion fails
///
/// # Examples
///
/// ```
/// use delaunay::geometry::util::safe_scalar_from_f64;
///
/// // Convert f64 to f32
/// let value_f64 = 123.456f64;
/// let value_f32: f32 = safe_scalar_from_f64(value_f64).unwrap();
/// assert!((value_f32 - 123.456f32).abs() < 1e-6);
///
/// // Convert f64 to f64 (identity)
/// let value: f64 = safe_scalar_from_f64(42.0f64).unwrap();
/// assert_eq!(value, 42.0f64);
/// ```
pub fn safe_scalar_from_f64<T: CoordinateScalar>(
    value: f64,
) -> Result<T, CoordinateConversionError> {
    safe_cast_from_f64(value, 0)
}

/// Safely convert a `usize` value to a coordinate scalar type T.
///
/// This function handles the conversion from `usize` to coordinate scalar types
/// with proper precision checking. Since `f64` has only 52 bits of precision
/// for the mantissa, `usize` values larger than 2^52 could lose precision when
/// converted through `f64`. This function checks for this condition and returns
/// an error if precision loss would occur.
///
/// # Arguments
///
/// * `value` - The `usize` value to convert
///
/// # Returns
///
/// The converted T value or a `CoordinateConversionError`
///
/// # Errors
///
/// Returns `CoordinateConversionError::ConversionFailed` if:
/// - The `usize` value is too large and would lose precision when converted through `f64`
/// - The conversion from `f64` to type T fails
///
/// # Examples
///
/// ```
/// use delaunay::geometry::util::safe_usize_to_scalar;
///
/// // Normal case - small usize values
/// let result: Result<f64, _> = safe_usize_to_scalar(42_usize);
/// assert_eq!(result.unwrap(), 42.0);
///
/// // Large values that fit within f64 precision
/// let large_value = (1_u64 << 50) as usize; // 2^50, well within f64 precision
/// let result: Result<f64, _> = safe_usize_to_scalar(large_value);
/// assert!(result.is_ok());
///
/// // Values that would lose precision (if usize is large enough)
/// // This test may not trigger on all platforms depending on usize size
/// ```
///
/// # Precision Limits
///
/// - `f64` mantissa has 52 bits of precision
/// - `usize` values larger than 2^52 (4,503,599,627,370,496) may lose precision
/// - On 32-bit platforms, `usize` is only 32 bits, so precision loss is impossible
/// - On 64-bit platforms, `usize` can be up to 64 bits, so precision loss is possible
pub fn safe_usize_to_scalar<T: CoordinateScalar>(
    value: usize,
) -> Result<T, CoordinateConversionError> {
    // Check for potential precision loss when converting usize to f64
    // f64 has 52 bits of precision in the mantissa, so values larger than 2^52 may lose precision
    const MAX_PRECISE_USIZE_IN_F64: u64 = 1_u64 << 52; // 2^52 = 4,503,599,627,370,496

    // Use try_from to safely convert usize to u64 for comparison
    let value_u64 =
        u64::try_from(value).map_err(|_| CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: format!("{value}"),
            from_type: "usize",
            to_type: std::any::type_name::<T>(),
        })?;

    if value_u64 > MAX_PRECISE_USIZE_IN_F64 {
        return Err(CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: format!("{value}"),
            from_type: "usize",
            to_type: std::any::type_name::<T>(),
        });
    }

    // Safe to convert to f64 without precision loss, then convert to T
    // Use cast from num_traits for safe conversion
    let f64_value: f64 =
        cast(value).ok_or_else(|| CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: format!("{value}"),
            from_type: "usize",
            to_type: "f64",
        })?;

    safe_scalar_from_f64(f64_value)
}

/// Helper function to compute squared norm using generic arithmetic on T.
///
/// This function computes the sum of squares of coordinates using generic
/// arithmetic operations on type T, avoiding premature conversion to f64.
///
/// # Arguments
///
/// * `coords` - Array of coordinates of type T
///
/// # Returns
///
/// The squared norm (sum of squares) as type T
///
/// # Examples
///
/// ```
/// use delaunay::geometry::util::squared_norm;
///
/// // 2D vector
/// let coords_2d = [3.0, 4.0];
/// let norm_sq = squared_norm(coords_2d);
/// assert_eq!(norm_sq, 25.0); // 3² + 4² = 9 + 16 = 25
///
/// // 3D vector
/// let coords_3d = [1.0, 2.0, 2.0];
/// let norm_sq_3d = squared_norm(coords_3d);
/// assert_eq!(norm_sq_3d, 9.0); // 1² + 2² + 2² = 1 + 4 + 4 = 9
///
/// // 4D vector
/// let coords_4d = [1.0, 1.0, 1.0, 1.0];
/// let norm_sq_4d = squared_norm(coords_4d);
/// assert_eq!(norm_sq_4d, 4.0); // 1² + 1² + 1² + 1² = 4
/// ```
pub fn squared_norm<T, const D: usize>(coords: [T; D]) -> T
where
    T: CoordinateScalar + num_traits::Zero,
{
    coords.iter().fold(T::zero(), |acc, &x| acc + x * x)
}

/// Compute the d-dimensional hypot (Euclidean norm) of a coordinate array.
///
/// This function provides a numerically stable way to compute the Euclidean distance
/// (L2 norm) of a d-dimensional vector. For 2D, it uses the standard library's
/// `f64::hypot` function which provides optimal numerical stability. For higher
/// dimensions, it implements a generalized hypot calculation.
///
/// # Numerical Stability
///
/// The 2D case uses `f64::hypot(a, b)` which avoids overflow and underflow
/// issues when computing `sqrt(a² + b²)`. For higher dimensions, the function
/// implements a similar approach by finding the maximum absolute value and
/// scaling all coordinates relative to it.
///
/// # Arguments
///
/// * `coords` - Array of coordinates of type T
///
/// # Returns
///
/// The Euclidean norm (hypot) as type T
///
/// # Examples
///
/// ```
/// use delaunay::geometry::util::hypot;
///
/// // 2D case - uses std::f64::hypot internally
/// let distance_2d = hypot([3.0, 4.0]);
/// assert_eq!(distance_2d, 5.0);
///
/// // 3D case - uses generalized algorithm
/// let distance_3d = hypot([1.0, 2.0, 2.0]);
/// assert_eq!(distance_3d, 3.0);
///
/// // Higher dimensions
/// let distance_4d = hypot([1.0, 1.0, 1.0, 1.0]);
/// assert_eq!(distance_4d, 2.0);
/// ```
pub fn hypot<T, const D: usize>(coords: [T; D]) -> T
where
    T: CoordinateScalar + num_traits::Zero,
{
    match D {
        0 => T::zero(),
        1 => Float::abs(coords[0]),
        2 => {
            // Use standard library hypot for optimal 2D performance and stability
            // Use safe conversion with proper error handling
            // If conversion fails, fall back to general algorithm
            if let (Ok(a_f64), Ok(b_f64)) =
                (safe_scalar_to_f64(coords[0]), safe_scalar_to_f64(coords[1]))
            {
                let result_f64 = a_f64.hypot(b_f64);
                safe_scalar_from_f64(result_f64).unwrap_or_else(|_| {
                    // Fall back to scaled algorithm if conversion back fails
                    scaled_hypot_2d(coords[0], coords[1])
                })
            } else {
                // Fall back to scaled algorithm if conversion fails
                scaled_hypot_2d(coords[0], coords[1])
            }
        }
        _ => {
            // For higher dimensions, implement generalized hypot
            // Find the maximum absolute value to avoid overflow/underflow
            let max_abs = coords
                .iter()
                .map(|&x| Float::abs(x))
                .fold(T::zero(), |acc, x| if x > acc { x } else { acc });

            if max_abs == T::zero() {
                return T::zero();
            }

            // Scale all coordinates by max_abs and compute sum of squares
            let sum_of_scaled_squares = coords
                .iter()
                .map(|&x| {
                    let scaled = x / max_abs;
                    scaled * scaled
                })
                .fold(T::zero(), |acc, x| acc + x);

            // Result is max_abs * sqrt(sum_of_scaled_squares)
            max_abs * Float::sqrt(sum_of_scaled_squares)
        }
    }
}

/// Calculate the circumcenter of a set of points forming a simplex.
///
/// The circumcenter is the unique point equidistant from all points of
/// the simplex. Returns an error if the points do not form a valid simplex or
/// if the computation fails due to degeneracy or numerical issues.
///
/// Using the approach from:
///
/// Lévy, Bruno, and Yang Liu.
/// "Lp Centroidal Voronoi Tessellation and Its Applications."
/// ACM Transactions on Graphics 29, no. 4 (July 26, 2010): 119:1-119:11.
/// <https://doi.org/10.1145/1778765.1778856>.
///
/// The circumcenter C of a simplex with points `x_0`, `x_1`, ..., `x_n` is the
/// solution to the system:
///
/// C = 1/2 (A^-1*B)
///
/// Where:
///
/// A is a matrix (to be inverted) of the form:
///     (x_1-x0) for all coordinates in x1, x0
///     (x2-x0) for all coordinates in x2, x0
///     ... for all `x_n` in the simplex
///
/// These are the perpendicular bisectors of the edges of the simplex.
///
/// And:
///
/// B is a vector of the form:
///     (x_1^2-x0^2) for all coordinates in x1, x0
///     (x_2^2-x0^2) for all coordinates in x2, x0
///     ... for all `x_n` in the simplex
///
/// The resulting vector gives the coordinates of the circumcenter.
///
/// # Arguments
///
/// * `points` - A slice of points that form the simplex
///
/// # Returns
/// The circumcenter as a Point<T, D> if successful, or an error if the
/// simplex is degenerate or the matrix inversion fails.
///
/// # Errors
///
/// Returns an error if:
/// - The points do not form a valid simplex
/// - The matrix inversion fails due to degeneracy
/// - Array conversion fails
///
/// # Example
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::geometry::util::circumcenter;
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let points = vec![point1, point2, point3, point4];
/// let center = circumcenter(&points).unwrap();
/// assert_eq!(center, Point::new([0.5, 0.5, 0.5]));
/// ```
pub fn circumcenter<T, const D: usize>(
    points: &[Point<T, D>],
) -> Result<Point<T, D>, CircumcenterError>
where
    T: CoordinateScalar + Sum + Zero,
{
    if points.is_empty() {
        return Err(CircumcenterError::EmptyPointSet);
    }

    let dim = points.len() - 1;
    if dim != D {
        return Err(CircumcenterError::InvalidSimplex {
            actual: points.len(),
            expected: D + 1,
            dimension: D,
        });
    }

    // Build matrix A and vector B for the linear system
    let mut matrix = zeros(dim, dim);
    let mut b = zeros(dim, 1);
    let coords_0: [T; D] = (&points[0]).into();

    // Use safe coordinate conversion
    let coords_0_f64: [f64; D] = safe_coords_to_f64(coords_0)?;

    for i in 0..dim {
        let coords_point: [T; D] = (&points[i + 1]).into();

        // Use safe coordinate conversion
        let coords_point_f64: [f64; D] = safe_coords_to_f64(coords_point)?;

        // Fill matrix row
        for j in 0..dim {
            matrix[(i, j)] = coords_point_f64[j] - coords_0_f64[j];
        }

        // Calculate squared distance using squared_norm for consistency
        let mut diff_coords = [T::zero(); D];
        for j in 0..D {
            diff_coords[j] = coords_point[j] - coords_0[j];
        }
        let squared_distance = squared_norm(diff_coords);

        // Use safe coordinate conversion for squared distance
        let squared_distance_f64: f64 = safe_scalar_to_f64(squared_distance)?;
        b[(i, 0)] = squared_distance_f64;
    }

    let a_inv = invert(&matrix)?;

    let solution = a_inv * b * 0.5;
    let solution_vec = solution.col(0);

    // Convert solution vector to array
    let solution_slice: &[f64] = &solution_vec;
    let solution_array: [f64; D] =
        solution_slice
            .try_into()
            .map_err(|_| CircumcenterError::ArrayConversionFailed {
                details: "Failed to convert solution vector to array".to_string(),
            })?;

    // Use safe coordinate conversion for solution and add back the first point
    let mut circumcenter_coords = [T::zero(); D];
    for i in 0..D {
        let relative_coord: T = safe_scalar_from_f64(solution_array[i])?;
        circumcenter_coords[i] = coords_0[i] + relative_coord;
    }
    Ok(Point::new(circumcenter_coords))
}

/// Calculate the circumradius of a set of points forming a simplex.
///
/// The circumradius is the distance from the circumcenter to any point of the simplex.
///
/// # Arguments
///
/// * `points` - A slice of points that form the simplex
///
/// # Returns
/// The circumradius as a value of type T if successful, or an error if the
/// circumcenter calculation fails.
///
/// # Errors
///
/// Returns an error if the circumcenter calculation fails. See [`circumcenter`] for details.
///
/// # Example
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::geometry::util::circumradius;
/// use approx::assert_relative_eq;
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let points = vec![point1, point2, point3, point4];
/// let radius = circumradius(&points).unwrap();
/// let expected_radius = (3.0_f64.sqrt() / 2.0);
/// assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
/// ```
pub fn circumradius<T, const D: usize>(points: &[Point<T, D>]) -> Result<T, CircumcenterError>
where
    T: CoordinateScalar + Sum + Zero,
{
    let circumcenter = circumcenter(points)?;
    circumradius_with_center(points, &circumcenter).map_err(|e| {
        CircumcenterError::MatrixInversionFailed {
            details: format!("Failed to calculate circumradius: {e}"),
        }
    })
}

/// Calculate the circumradius given a precomputed circumcenter.
///
/// This is a helper function that calculates the circumradius when the circumcenter
/// is already known, avoiding redundant computation.
///
/// # Arguments
///
/// * `points` - A slice of points that form the simplex
/// * `circumcenter` - The precomputed circumcenter
///
/// # Returns
/// The circumradius as a value of type T if successful, or an error if the
/// simplex is degenerate or the distance calculation fails.
///
/// # Errors
///
/// Returns an error if:
/// - The points slice is empty
/// - Coordinate conversion fails
/// - Distance calculation fails
///
/// # Example
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::geometry::util::{circumcenter, circumradius_with_center};
/// use approx::assert_relative_eq;
/// let point1 = Point::new([0.0, 0.0, 0.0]);
/// let point2 = Point::new([1.0, 0.0, 0.0]);
/// let point3 = Point::new([0.0, 1.0, 0.0]);
/// let point4 = Point::new([0.0, 0.0, 1.0]);
/// let points = vec![point1, point2, point3, point4];
/// let center = circumcenter(&points).unwrap();
/// let radius = circumradius_with_center(&points, &center).unwrap();
/// let expected_radius = (3.0_f64.sqrt() / 2.0);
/// assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
/// ```
pub fn circumradius_with_center<T, const D: usize>(
    points: &[Point<T, D>],
    circumcenter: &Point<T, D>,
) -> Result<T, CircumcenterError>
where
    T: CoordinateScalar + Sum + Zero,
{
    if points.is_empty() {
        return Err(CircumcenterError::EmptyPointSet);
    }

    let point_coords: [T; D] = (&points[0]).into();
    let circumcenter_coords: [T; D] = circumcenter.to_array();

    // Calculate distance using hypot for numerical stability
    let mut diff_coords = [T::zero(); D];
    for i in 0..D {
        diff_coords[i] = circumcenter_coords[i] - point_coords[i];
    }
    let distance = hypot(diff_coords);
    Ok(distance)
}

/// Calculate the area/volume of a facet defined by a set of points.
///
/// This function calculates the (D-1)-dimensional "area" of a facet in D-dimensional space:
/// - 1D: Distance between two points (length)
/// - 2D: Area of triangle using cross product  
/// - 3D: Volume of tetrahedron using scalar triple product
/// - 4D+: Generalized volume using Gram matrix method
///
/// For dimensions 4 and higher, this function uses the Gram matrix method for
/// mathematically accurate volume computation.
///
/// # Arguments
///
/// * `points` - Points defining the facet (should have exactly D points for (D-1)-dimensional facet)
///
/// # Returns
///
/// The area/volume of the facet, or an error if calculation fails
///
/// # Errors
///
/// Returns an error if:
/// - Wrong number of points provided
/// - Points are degenerate (collinear/coplanar)
/// - Coordinate conversion fails
///
/// # Examples
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::traits::coordinate::Coordinate;
/// use delaunay::geometry::util::facet_measure;
/// use approx::assert_relative_eq;
///
/// // 2D: Line segment length (1D facet in 2D space)
/// let line_segment = vec![
///     Point::new([0.0, 0.0]),
///     Point::new([3.0, 4.0]),
/// ];
/// let length = facet_measure(&line_segment).unwrap();
/// assert_relative_eq!(length, 5.0, epsilon = 1e-10); // sqrt(3² + 4²) = 5
///
/// // 3D: Triangle area (2D facet in 3D space)
/// let triangle = vec![
///     Point::new([0.0, 0.0, 0.0]),
///     Point::new([3.0, 0.0, 0.0]),
///     Point::new([0.0, 4.0, 0.0]),
/// ];
/// let area = facet_measure(&triangle).unwrap();
/// assert_relative_eq!(area, 6.0, epsilon = 1e-10); // 3*4/2 = 6
/// ```
pub fn facet_measure<T, const D: usize>(points: &[Point<T, D>]) -> Result<T, CircumcenterError>
where
    T: CoordinateScalar + Sum + Zero,
{
    if points.len() != D {
        return Err(CircumcenterError::InvalidSimplex {
            actual: points.len(),
            expected: D,
            dimension: D,
        });
    }

    match D {
        1 => {
            // 1D: Distance between two points
            if points.len() != 1 {
                return Err(CircumcenterError::InvalidSimplex {
                    actual: points.len(),
                    expected: 1,
                    dimension: 1,
                });
            }
            // For 1D, a "facet" is a single point, so "area" is 1 (or we could return 0)
            // This is somewhat arbitrary - in practice 1D facets aren't commonly used
            Ok(T::one())
        }
        2 => {
            // 2D: Length of line segment (1D facet in 2D space)
            let p0 = points[0].to_array();
            let p1 = points[1].to_array();

            let diff = [p1[0] - p0[0], p1[1] - p0[1]];
            Ok(hypot(diff))
        }
        3 => {
            // 3D: Area of triangle (2D facet in 3D space) using cross product
            let p0 = points[0].to_array();
            let p1 = points[1].to_array();
            let p2 = points[2].to_array();

            // Vectors from p0 to p1 and p0 to p2
            let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

            // Cross product v1 × v2
            let cross = [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ];

            // Area is |cross product| / 2
            let cross_magnitude = hypot(cross);
            Ok(cross_magnitude / (T::one() + T::one())) // Divide by 2
        }
        4 => {
            // 4D: Volume of tetrahedron (3D facet in 4D space)
            // Use Gram matrix method for correct calculation
            facet_measure_gram_matrix::<T, D>(points)
        }
        _ => {
            // Higher dimensions: Use Gram matrix method for correct calculation
            facet_measure_gram_matrix::<T, D>(points)
        }
    }
}

/// Calculate the area/volume of a (D-1)-dimensional simplex using the Gram matrix method.
///
/// This function implements the mathematically rigorous approach for computing the volume
/// of a (D-1)-dimensional simplex embedded in D-dimensional space using the Gram matrix
/// determinant formula:
///
/// **Volume = (1/(D-1)!) × √(det(G))**
///
/// where G is the Gram matrix of edge vectors from one vertex to all other vertices.
///
/// # Mathematical Background
///
/// The Gram matrix method is the standard approach for computing simplex volumes in
/// high-dimensional spaces, as described in:
///
/// - Coxeter, H.S.M. "Introduction to Geometry" (2nd ed., 1969), Chapter 13
/// - Richter-Gebert, Jürgen. "Perspectives on Projective Geometry" (2011), Section 14.3
/// - Edelsbrunner, Herbert. "Geometry and Topology for Mesh Generation" (2001), Chapter 2
///
/// The method constructs the Gram matrix G where G\[i,j\] = `v_i` · `v_j` (dot product of
/// edge vectors), then computes the volume as the square root of the determinant
/// divided by the appropriate factorial.
///
/// This approach is numerically stable and generalizes correctly to arbitrary dimensions,
/// unlike methods based on recursive determinant expansion which become computationally
/// intractable in high dimensions.
///
/// # Arguments
///
/// * `points` - Points defining the simplex (should have exactly D points for (D-1)-dimensional facet)
///
/// # Returns
///
/// The volume of the simplex, or an error if calculation fails
///
/// # Errors
///
/// Returns an error if:
/// - Matrix operations fail (singular Gram matrix indicates degenerate simplex)
/// - Coordinate conversion fails
/// - Gram matrix determinant is negative (should never happen for valid input)
fn facet_measure_gram_matrix<T, const D: usize>(
    points: &[Point<T, D>],
) -> Result<T, CircumcenterError>
where
    T: CoordinateScalar + Sum + Zero,
{
    // Convert points to f64 and create edge vectors from first point to all others
    let p0_coords = points[0].to_array();
    let p0_f64 = safe_coords_to_f64(p0_coords)?;

    // Create matrix of edge vectors (each row is an edge vector)
    let mut edge_matrix = zeros(D - 1, D);
    for i in 1..D {
        let point_coords = points[i].to_array();
        let point_f64 = safe_coords_to_f64(point_coords)?;

        for j in 0..D {
            edge_matrix[(i - 1, j)] = point_f64[j] - p0_f64[j];
        }
    }

    // Compute Gram matrix G where G[i,j] = edge_i · edge_j
    let mut gram_matrix = zeros(D - 1, D - 1);
    for i in 0..(D - 1) {
        for j in 0..(D - 1) {
            let mut dot_product = 0.0;
            for k in 0..D {
                dot_product += edge_matrix[(i, k)] * edge_matrix[(j, k)];
            }
            gram_matrix[(i, j)] = dot_product;
        }
    }

    // Calculate determinant of Gram matrix
    let det = gram_matrix.det();

    // Volume = (1/(D-1)!) × √(det(G))
    if det < 0.0 {
        return Err(CircumcenterError::MatrixInversionFailed {
            details: "Gram matrix has negative determinant (degenerate simplex)".to_string(),
        });
    }

    let volume_f64 = if det == 0.0 {
        0.0 // Degenerate case
    } else {
        let sqrt_det = det.sqrt();

        // Calculate (D-1)! factorial - peroxide's factorial function returns usize
        let factorial_usize = factorial(D - 1);
        let factorial_val = safe_usize_to_scalar::<f64>(factorial_usize).map_err(|_| {
            CircumcenterError::ValueConversion(ValueConversionError::ConversionFailed {
                value: factorial_usize.to_string(),
                from_type: "usize",
                to_type: "f64",
                details: "Factorial value too large for f64 precision".to_string(),
            })
        })?;
        sqrt_det / factorial_val
    };

    safe_scalar_from_f64(volume_f64).map_err(CircumcenterError::CoordinateConversion)
}

/// Calculate the surface area of a triangulated boundary by summing facet measures.
///
/// This function calculates the total surface area of a boundary defined by
/// a collection of facets. Each facet's measure (area/volume) is calculated
/// and summed to give the total surface measure.
///
/// # Arguments
///
/// * `facets` - Collection of facets defining the boundary surface
///
/// # Returns
///
/// Total surface area/volume, or error if any facet calculation fails
///
/// # Errors
///
/// Returns an error if any individual facet measure calculation fails
///
/// # Examples
///
/// ```
/// use delaunay::geometry::point::Point;
/// use delaunay::geometry::util::surface_measure;
/// use delaunay::core::facet::Facet;
/// use delaunay::core::vertex::Vertex;
/// use delaunay::core::cell::Cell;
/// use delaunay::{cell, vertex};
///
/// // Create triangular facets for a cube surface
/// let v1: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 0.0]);
/// let v2: Vertex<f64, Option<()>, 3> = vertex!([1.0, 0.0, 0.0]);
/// let v3: Vertex<f64, Option<()>, 3> = vertex!([0.0, 1.0, 0.0]);
/// let v4: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 1.0]);
///
/// let cell: Cell<f64, Option<()>, Option<()>, 3> = cell!(vec![v1, v2, v3, v4]);
/// let facet = Facet::new(cell, v1).unwrap();
///
/// // Calculate surface area (this example shows the API pattern)
/// // let surface_area = surface_measure(&[facet]).unwrap();
/// ```
pub fn surface_measure<T, U, V, const D: usize>(
    facets: &[crate::core::facet::Facet<T, U, V, D>],
) -> Result<T, CircumcenterError>
where
    T: CoordinateScalar + Sum + Zero,
    U: crate::core::traits::data_type::DataType,
    V: crate::core::traits::data_type::DataType,
    [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
{
    let mut total_measure = T::zero();

    for facet in facets {
        let facet_vertices = facet.vertices();

        // Convert vertices to Points for measure calculation
        let points: Vec<Point<T, D>> = facet_vertices
            .iter()
            .map(|v| {
                let coords: [T; D] = v.into();
                Point::new(coords)
            })
            .collect();

        let measure = facet_measure(&points)?;
        total_measure = total_measure + measure;
    }

    Ok(total_measure)
}

// ============================================================================
// Random Point Generation Utilities
// ============================================================================

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
        let coords = [T::zero(); D].map(|_| {
            use rand::Rng;
            rng.random_range(range.0..range.1)
        });
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
        let candidate_coords: [T; D] = candidate.to_array();
        for existing_point in &points {
            let existing_coords: [T; D] = existing_point.to_array();

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

/// Generate a random Delaunay triangulation with specified parameters.
///
/// This utility function combines random point generation and triangulation creation
/// in a single convenient function. It generates random points using either seeded
/// or unseeded random generation, converts them to vertices, and creates a Delaunay
/// triangulation using the Bowyer-Watson algorithm.
///
/// This function is particularly useful for testing, benchmarking, and creating
/// triangulations for analysis or visualization purposes.
///
/// # Type Parameters
///
/// * `T` - Coordinate scalar type (must implement `CoordinateScalar + SampleUniform`)
/// * `U` - Vertex data type (must implement `DataType`)
/// * `V` - Cell data type (must implement `DataType`)
/// * `D` - Dimensionality (const generic parameter)
///
/// # Arguments
///
/// * `n_points` - Number of random points to generate
/// * `bounds` - Coordinate bounds as `(min, max)` tuple
/// * `vertex_data` - Optional data to attach to each generated vertex
/// * `seed` - Optional seed for reproducible results. If `None`, uses thread-local RNG
///
/// # Returns
///
/// A `Result` containing either:
/// - `Ok(Tds<T, U, V, D>)` - The successfully created triangulation
/// - `Err` - An error from point generation or triangulation construction
///
/// # Errors
///
/// This function can fail with:
/// - `RandomPointGenerationError` if point generation fails (invalid bounds, etc.)
/// - `TdsError` if triangulation construction fails (degenerate points, etc.)
///
/// # Panics
///
/// This function can panic if:
/// - Vertex construction fails due to invalid data types or constraints
/// - This should not happen with valid inputs and supported data types
///
/// # Examples
///
/// ```
/// use delaunay::geometry::util::generate_random_triangulation;
///
/// // Generate a 2D triangulation with 100 points, no seed (random each time)
/// let triangulation_2d = generate_random_triangulation::<f64, (), (), 2>(
///     100,
///     (-10.0, 10.0),
///     None,
///     None
/// ).unwrap();
///
/// // Generate a 3D triangulation with 50 points, seeded for reproducibility  
/// let triangulation_3d = generate_random_triangulation::<f64, (), (), 3>(
///     50,
///     (-5.0, 5.0),
///     None,
///     Some(42)
/// ).unwrap();
///
/// // Generate a 4D triangulation with custom vertex data
/// let triangulation_4d = generate_random_triangulation::<f64, i32, (), 4>(
///     25,
///     (0.0, 1.0),
///     Some(123),
///     Some(456)
/// ).unwrap();
///
/// // For string-like data, use fixed-size character arrays (Copy types)
/// let triangulation_with_strings = generate_random_triangulation::<f64, [char; 8], (), 2>(
///     20,
///     (0.0, 1.0),
///     Some(['v', 'e', 'r', 't', 'e', 'x', '_', 'A']),
///     Some(789)
/// ).unwrap();
/// ```
///
/// # Note on String Data
///
/// Due to the `DataType` trait requiring `Copy`, `String` and `&str` cannot be used directly
/// as vertex data. For string-like data, consider using:
/// - Fixed-size character arrays: `[char; N]`
/// - Small integer types that can be mapped to strings: `u32`, `u64`
/// - Custom Copy types that wrap string-like data
///
/// # Performance Notes
///
/// - Point generation is O(n) and typically fast
/// - Triangulation construction complexity varies by dimension:
///   - 2D, 3D: O(n log n) expected with Bowyer-Watson algorithm
///   - 4D+: O(n²) worst case, significantly slower for large point sets
/// - Consider using smaller point counts for dimensions ≥ 4
///
/// # See Also
///
/// - [`generate_random_points`] - For generating points without triangulation
/// - [`generate_random_points_seeded`] - For seeded random point generation only
/// - [`Tds::new`] - For creating triangulations from existing vertices
pub fn generate_random_triangulation<T, U, V, const D: usize>(
    n_points: usize,
    bounds: (T, T),
    vertex_data: Option<U>,
    seed: Option<u64>,
) -> Result<Tds<T, U, V, D>, Box<dyn std::error::Error>>
where
    T: CoordinateScalar
        + SampleUniform
        + std::ops::AddAssign<T>
        + std::ops::SubAssign<T>
        + std::iter::Sum
        + num_traits::cast::NumCast,
    U: DataType,
    V: DataType,
    for<'a> &'a T: std::ops::Div<T>,
    [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
{
    // Generate random points (seeded or unseeded)
    let points: Vec<Point<T, D>> = match seed {
        Some(seed_value) => generate_random_points_seeded(n_points, bounds, seed_value)?,
        None => generate_random_points(n_points, bounds)?,
    };

    // Convert points to vertices using the vertex! macro pattern
    let vertices: Vec<Vertex<T, U, D>> = points
        .into_iter()
        .map(|point| {
            use crate::core::vertex::VertexBuilder;
            vertex_data.map_or_else(
                || {
                    VertexBuilder::default()
                        .point(point)
                        .build()
                        .expect("Failed to build vertex without data")
                },
                |data| {
                    VertexBuilder::default()
                        .point(point)
                        .data(data)
                        .build()
                        .expect("Failed to build vertex with data")
                },
            )
        })
        .collect();

    // Create and return triangulation
    let triangulation =
        Tds::<T, U, V, D>::new(&vertices).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    Ok(triangulation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cell::Cell;
    use crate::core::vertex::Vertex;
    use crate::geometry::point::Point;
    use approx::assert_relative_eq;

    #[test]
    fn test_hypot_2d() {
        // Test 2D case - should use std::f64::hypot
        let distance = hypot([3.0, 4.0]);
        assert_relative_eq!(distance, 5.0, epsilon = 1e-10);

        // Test with zero
        let distance_zero = hypot([0.0, 0.0]);
        assert_relative_eq!(distance_zero, 0.0, epsilon = 1e-10);

        // Test with negative values
        let distance_neg = hypot([-3.0, 4.0]);
        assert_relative_eq!(distance_neg, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hypot_3d() {
        // Test 3D case - uses generalized algorithm
        let distance = hypot([1.0, 2.0, 2.0]);
        assert_relative_eq!(distance, 3.0, epsilon = 1e-10);

        // Test unit vector in 3D
        let distance_unit = hypot([1.0, 0.0, 0.0]);
        assert_relative_eq!(distance_unit, 1.0, epsilon = 1e-10);

        // Test with all equal components
        let distance_equal = hypot([1.0, 1.0, 1.0]);
        assert_relative_eq!(distance_equal, 3.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_hypot_4d() {
        // Test 4D case
        let distance = hypot([1.0, 1.0, 1.0, 1.0]);
        assert_relative_eq!(distance, 2.0, epsilon = 1e-10);

        // Test with zero vector
        let distance_zero = hypot([0.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(distance_zero, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hypot_edge_cases() {
        // Test 0D case
        let distance_0d = hypot::<f64, 0>([]);
        assert_relative_eq!(distance_0d, 0.0, epsilon = 1e-10);

        // Test 1D case
        let distance_1d_pos = hypot([5.0]);
        assert_relative_eq!(distance_1d_pos, 5.0, epsilon = 1e-10);

        let distance_1d_neg = hypot([-5.0]);
        assert_relative_eq!(distance_1d_neg, 5.0, epsilon = 1e-10);

        // Test large values that might cause overflow with naive sqrt(x² + y²)
        let distance_large = hypot([1e200, 1e200]);
        assert!(distance_large.is_finite());
        assert!(distance_large > 0.0);
    }

    #[test]
    fn predicates_circumcenter() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let center = circumcenter(&points).unwrap();

        assert_eq!(center, Point::new([0.5, 0.5, 0.5]));
    }

    #[test]
    fn predicates_circumcenter_fail() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
        ];
        let center = circumcenter(&points);

        assert!(center.is_err());
    }

    #[test]
    fn predicates_circumradius() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let radius = circumradius(&points).unwrap();
        let expected_radius: f64 = 3.0_f64.sqrt() / 2.0;

        assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
    }

    #[test]
    fn predicates_circumcenter_2d() {
        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([2.0, 0.0]),
            Point::new([1.0, 2.0]),
        ];
        let center = circumcenter(&points).unwrap();

        // For this triangle, circumcenter should be at (1.0, 0.75)
        assert_relative_eq!(center.to_array()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(center.to_array()[1], 0.75, epsilon = 1e-10);
    }

    #[test]
    fn predicates_circumradius_2d() {
        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, 0.0]),
            Point::new([0.0, 1.0]),
        ];
        let radius = circumradius(&points).unwrap();

        // For a right triangle with legs of length 1, circumradius is sqrt(2)/2
        let expected_radius = 2.0_f64.sqrt() / 2.0;
        assert_relative_eq!(radius, expected_radius, epsilon = 1e-10);
    }

    #[test]
    fn predicates_circumradius_with_center() {
        // Test the circumradius_with_center function
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let center = circumcenter(&points).unwrap();
        let radius_with_center = circumradius_with_center(&points, &center);
        let radius_direct = circumradius(&points).unwrap();

        assert_relative_eq!(radius_with_center.unwrap(), radius_direct, epsilon = 1e-10);
    }

    // =============================================================================
    // COORDINATE CONVERSION FUNCTION TESTS
    // =============================================================================

    // =============================================================================
    // SAFE USIZE TO SCALAR CONVERSION TESTS
    // =============================================================================

    #[test]
    fn test_safe_usize_to_scalar_basic_success() {
        // Test successful conversion of small usize values
        let small_value = 42_usize;
        let result: Result<f64, _> = safe_usize_to_scalar(small_value);
        assert!(result.is_ok());
        assert_relative_eq!(result.unwrap(), 42.0f64, epsilon = 1e-15);

        // Test with f32 target type
        let result_f32: Result<f32, _> = safe_usize_to_scalar(small_value);
        assert!(result_f32.is_ok());
        assert_relative_eq!(result_f32.unwrap(), 42.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_safe_usize_to_scalar_zero() {
        let zero_value = 0_usize;
        let result: Result<f64, _> = safe_usize_to_scalar(zero_value);
        assert!(result.is_ok());
        assert_relative_eq!(result.unwrap(), 0.0f64, epsilon = 1e-15);
    }

    #[test]
    fn test_safe_usize_to_scalar_small_values() {
        // Test various small values that should always work
        let test_values = [1, 10, 100, 1000, 10_000, 100_000, 1_000_000];

        for &value in &test_values {
            let result: Result<f64, _> = safe_usize_to_scalar(value);
            assert!(result.is_ok(), "Failed to convert {value}");
            let expected_f64: f64 = cast(value).expect("Small values should convert safely");
            assert_relative_eq!(result.unwrap(), expected_f64, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_safe_usize_to_scalar_within_f64_precision() {
        // Test values that are within f64 precision limits but relatively large
        let safe_large_values = [
            usize::try_from(1_u64 << 50).unwrap_or(usize::MAX), // 2^50, well within f64 precision
            usize::try_from(1_u64 << 51).unwrap_or(usize::MAX), // 2^51, still safe
            usize::try_from((1_u64 << 52) - 1).unwrap_or(usize::MAX), // Just under 2^52, should be safe
        ];

        for &value in &safe_large_values {
            let result: Result<f64, _> = safe_usize_to_scalar(value);
            assert!(result.is_ok(), "Failed to convert safe large value {value}");

            // Verify that the conversion is exact
            let converted = result.unwrap();
            let back_converted: usize =
                cast(converted).expect("f64 should convert back to usize exactly");
            assert_eq!(
                back_converted, value,
                "Conversion was not exact for {value}"
            );
        }
    }

    #[test]
    fn test_safe_usize_to_scalar_precision_boundary() {
        // Test the exact boundary value: 2^52
        const MAX_PRECISE_USIZE_IN_F64: u64 = 1_u64 << 52;

        // This should succeed (exactly at the boundary)
        if usize::try_from(MAX_PRECISE_USIZE_IN_F64).is_ok() {
            let boundary_value = usize::try_from(MAX_PRECISE_USIZE_IN_F64).unwrap();
            let result: Result<f64, _> = safe_usize_to_scalar(boundary_value);
            assert!(result.is_ok(), "Boundary value 2^52 should be convertible");

            let converted = result.unwrap();
            let back_converted: usize =
                cast(converted).expect("Boundary f64 should convert back to usize exactly");
            assert_eq!(
                back_converted, boundary_value,
                "Boundary conversion should be exact"
            );
        }
    }

    #[test]
    fn test_safe_usize_to_scalar_precision_loss_detection() {
        // Test values that would lose precision (only on 64-bit platforms where usize can exceed 2^52)
        const MAX_PRECISE_USIZE_IN_F64: u64 = 1_u64 << 52;

        if std::mem::size_of::<usize>() >= 8 {
            // On 64-bit platforms, test values that would lose precision
            let precision_loss_values = [
                usize::try_from(MAX_PRECISE_USIZE_IN_F64 + 1).unwrap_or(usize::MAX), // Just over 2^52
                usize::try_from(MAX_PRECISE_USIZE_IN_F64 + 100).unwrap_or(usize::MAX), // Well over 2^52
            ];

            for &value in &precision_loss_values {
                // Skip if the value would overflow usize on this platform
                let value_u64 = u64::try_from(value).unwrap_or(u64::MAX);
                if value_u64 > u64::try_from(usize::MAX).unwrap_or(u64::MAX) {
                    continue;
                }

                let result: Result<f64, _> = safe_usize_to_scalar(value);
                assert!(
                    result.is_err(),
                    "Value {value} should fail conversion due to precision loss"
                );

                // Verify error details
                if let Err(CoordinateConversionError::ConversionFailed {
                    coordinate_index,
                    coordinate_value,
                    from_type,
                    to_type,
                }) = result
                {
                    assert_eq!(coordinate_index, 0);
                    assert_eq!(coordinate_value, format!("{value}"));
                    assert_eq!(from_type, "usize");
                    assert_eq!(to_type, "f64");
                } else {
                    panic!("Expected ConversionFailed error for value {value}");
                }
            }
        } else {
            // On 32-bit platforms, usize cannot exceed 2^52, so all values should succeed
            println!("Skipping precision loss test on 32-bit platform");
        }
    }

    #[test]
    fn test_safe_usize_to_scalar_different_target_types() {
        let test_value = 1000_usize;

        // Test conversion to f64
        let result_f64: Result<f64, _> = safe_usize_to_scalar(test_value);
        assert!(result_f64.is_ok());
        assert_relative_eq!(result_f64.unwrap(), 1000.0f64, epsilon = 1e-15);

        // Test conversion to f32
        let result_f32: Result<f32, _> = safe_usize_to_scalar(test_value);
        assert!(result_f32.is_ok());
        assert_relative_eq!(result_f32.unwrap(), 1000.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_safe_usize_to_scalar_max_safe_values() {
        // Test maximum values that should be safe for common vertex/facet counts
        let realistic_values = [
            1_000,      // Small mesh
            10_000,     // Medium mesh
            100_000,    // Large mesh
            1_000_000,  // Very large mesh
            10_000_000, // Extremely large mesh (but still practical)
        ];

        for &value in &realistic_values {
            let result: Result<f64, _> = safe_usize_to_scalar(value);
            assert!(result.is_ok(), "Failed to convert realistic value {value}");

            // Verify precision is maintained
            let converted = result.unwrap();
            let back_converted: usize =
                cast(converted).expect("Realistic f64 values should convert back to usize exactly");
            assert_eq!(
                back_converted, value,
                "Precision lost for realistic value {value}"
            );
        }
    }

    #[test]
    fn test_safe_usize_to_scalar_error_message_format() {
        // Test that error messages are properly formatted
        const MAX_PRECISE_USIZE_IN_F64: u64 = 1_u64 << 52;

        if std::mem::size_of::<usize>() >= 8 {
            let large_value = usize::try_from(MAX_PRECISE_USIZE_IN_F64 + 1).unwrap_or(usize::MAX);
            if u64::try_from(large_value).unwrap_or(u64::MAX)
                <= u64::try_from(usize::MAX).unwrap_or(u64::MAX)
            {
                let result: Result<f64, _> = safe_usize_to_scalar(large_value);
                assert!(result.is_err());

                let error_message = format!("{}", result.unwrap_err());
                assert!(error_message.contains("Failed to convert"));
                assert!(error_message.contains(&format!("{large_value}")));
                assert!(error_message.contains("usize"));
                assert!(error_message.contains("f64"));
            }
        }
    }

    #[test]
    fn test_safe_usize_to_scalar_consistency_with_direct_cast() {
        // For values that should not lose precision, verify consistency with direct casting
        let safe_values = [0, 1, 42, 100, 1000, 10_000, 100_000];

        for &value in &safe_values {
            let safe_result: Result<f64, _> = safe_usize_to_scalar(value);
            assert!(safe_result.is_ok());

            let safe_converted = safe_result.unwrap();
            let direct_cast: f64 = cast(value).expect("Small values should convert to f64 safely");

            assert_relative_eq!(safe_converted, direct_cast, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_safe_usize_to_scalar_platform_independence() {
        // Test that the function behaves correctly on different platforms
        println!(
            "Testing on platform with usize size: {} bytes",
            std::mem::size_of::<usize>()
        );
        println!("usize::MAX = {}", usize::MAX);
        println!("2^52 = {}", 1_u64 << 52);

        // Values that should work on any platform
        let universal_safe_values = [0, 1, 100, 10000];

        for &value in &universal_safe_values {
            let result: Result<f64, _> = safe_usize_to_scalar(value);
            assert!(
                result.is_ok(),
                "Universal safe value {value} should convert on any platform"
            );
        }

        // Test the maximum safe value for this platform
        let _usize_max_u64 = u64::try_from(usize::MAX).unwrap_or(u64::MAX);
    }

    // =============================================================================
    // COMPREHENSIVE CIRCUMCENTER TESTS
    // =============================================================================

    #[test]
    fn test_circumcenter_regular_simplex_3d() {
        // Test with a regular tetrahedron - use simpler vertices
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.5, 3.0_f64.sqrt() / 2.0, 0.0]),
            Point::new([0.5, 3.0_f64.sqrt() / 6.0, (2.0 / 3.0_f64).sqrt()]),
        ];
        let center = circumcenter(&points).unwrap();

        // For this tetrahedron, verify circumcenter exists and is finite
        let center_coords = center.to_array();
        for coord in center_coords {
            assert!(
                coord.is_finite(),
                "Circumcenter coordinates should be finite"
            );
        }

        // Verify all points are equidistant from circumcenter
        let distances: Vec<f64> = points
            .iter()
            .map(|p| {
                let p_coords: [f64; 3] = p.into();
                let diff = [
                    p_coords[0] - center_coords[0],
                    p_coords[1] - center_coords[1],
                    p_coords[2] - center_coords[2],
                ];
                hypot(diff)
            })
            .collect();

        // All distances should be equal
        for i in 1..distances.len() {
            assert_relative_eq!(distances[0], distances[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_circumcenter_regular_simplex_4d() {
        // Test 4D simplex - use orthonormal basis plus origin
        let points = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0]),
        ];
        let center = circumcenter(&points).unwrap();

        // For this symmetric configuration, circumcenter should be at equal coordinates
        let center_coords = center.to_array();
        for coord in center_coords {
            assert!(
                coord.is_finite(),
                "Circumcenter coordinates should be finite"
            );
            // Should be around 0.5 for this configuration
            assert_relative_eq!(coord, 0.5, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_circumcenter_right_triangle_2d() {
        // Test with right triangle - circumcenter should be at hypotenuse midpoint
        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([4.0, 0.0]),
            Point::new([0.0, 3.0]),
        ];
        let center = circumcenter(&points).unwrap();

        // For right triangle, circumcenter is at midpoint of hypotenuse
        let center_coords = center.to_array();
        assert_relative_eq!(center_coords[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(center_coords[1], 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_circumcenter_scaled_simplex() {
        // Test that scaling preserves circumcenter properties
        let scale = 10.0;
        let points = vec![
            Point::new([0.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([1.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([0.0 * scale, 1.0 * scale, 0.0 * scale]),
            Point::new([0.0 * scale, 0.0 * scale, 1.0 * scale]),
        ];
        let center = circumcenter(&points).unwrap();

        // Scaled simplex should have scaled circumcenter
        let expected_center = Point::new([0.5 * scale, 0.5 * scale, 0.5 * scale]);
        let center_coords = center.to_array();
        let expected_coords = expected_center.to_array();

        for i in 0..3 {
            assert_relative_eq!(center_coords[i], expected_coords[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_circumcenter_translated_simplex() {
        // Test that translation preserves relative circumcenter position
        let translation = [10.0, 20.0, 30.0];
        let points = vec![
            Point::new([
                0.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ]),
            Point::new([
                1.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ]),
            Point::new([
                0.0 + translation[0],
                1.0 + translation[1],
                0.0 + translation[2],
            ]),
            Point::new([
                0.0 + translation[0],
                0.0 + translation[1],
                1.0 + translation[2],
            ]),
        ];
        let center = circumcenter(&points).unwrap();

        // Get the circumcenter of the untranslated simplex for comparison
        let untranslated_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let untranslated_center = circumcenter(&untranslated_points).unwrap();

        // Translated circumcenter should be untranslated circumcenter + translation
        let center_coords = center.to_array();
        let untranslated_coords = untranslated_center.to_array();

        for i in 0..3 {
            assert_relative_eq!(
                center_coords[i],
                untranslated_coords[i] + translation[i],
                epsilon = 1e-9
            );
        }

        // Also verify the expected absolute values for this specific tetrahedron
        let expected = [10.5, 20.5, 30.5];
        for i in 0..3 {
            assert_relative_eq!(center_coords[i], expected[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_circumcenter_nearly_degenerate_simplex() {
        // Test with points that are nearly collinear (may succeed or fail gracefully)
        let eps = 1e-3; // Use larger epsilon for more robustness
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.5, eps, 0.0]), // Slightly off the line
            Point::new([0.5, 0.0, eps]), // Slightly off the plane
        ];

        let result = circumcenter(&points);
        // Should either succeed or fail gracefully (don't require success)
        if let Ok(center) = result {
            // If it succeeds, center should have finite coordinates
            let coords = center.to_array();
            assert!(
                coords.iter().all(|&x| x.is_finite()),
                "Circumcenter coordinates should be finite"
            );
        } else {
            // If it fails, that's acceptable for this nearly degenerate case
        }
    }

    #[test]
    fn test_circumcenter_empty_points() {
        let points: Vec<Point<f64, 3>> = vec![];
        let result = circumcenter(&points);

        assert!(result.is_err());
        match result.unwrap_err() {
            CircumcenterError::EmptyPointSet => {}
            other => panic!("Expected EmptyPointSet error, got: {other:?}"),
        }
    }

    #[test]
    fn test_circumcenter_wrong_dimension() {
        // Test with 2 points for 3D (need 4 points for 3D circumcenter)
        let points = vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])];
        let result = circumcenter(&points);

        assert!(result.is_err());
        match result.unwrap_err() {
            CircumcenterError::InvalidSimplex {
                actual,
                expected,
                dimension,
            } => {
                assert_eq!(actual, 2);
                assert_eq!(expected, 4); // D + 1 where D = 3
                assert_eq!(dimension, 3);
            }
            other => panic!("Expected InvalidSimplex error, got: {other:?}"),
        }
    }

    #[test]
    fn test_circumcenter_large_coordinates() {
        // Test with large but finite coordinates
        let large_val = 1e6;
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([large_val, 0.0, 0.0]),
            Point::new([0.0, large_val, 0.0]),
            Point::new([0.0, 0.0, large_val]),
        ];

        let result = circumcenter(&points);
        assert!(result.is_ok(), "Large coordinates should work");

        let center = result.unwrap();
        let center_coords = center.to_array();

        // Should be approximately at (large_val/2, large_val/2, large_val/2)
        for &coord in &center_coords {
            assert_relative_eq!(coord, large_val / 2.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_circumcenter_small_coordinates() {
        // Test with very small but non-zero coordinates
        let small_val = 1e-3; // Use larger value for numerical stability
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([small_val, 0.0, 0.0]),
            Point::new([0.0, small_val, 0.0]),
            Point::new([0.0, 0.0, small_val]),
        ];

        let result = circumcenter(&points);
        assert!(result.is_ok(), "Small coordinates should work");

        let center = result.unwrap();
        let center_coords = center.to_array();

        // Verify coordinates are finite and reasonable
        for coord in center_coords {
            assert!(
                coord.is_finite(),
                "Circumcenter coordinates should be finite"
            );
            // Should be on the order of small_val
            assert!(
                coord.abs() < small_val * 10.0,
                "Coordinates should be reasonably small"
            );
        }
    }

    #[test]
    fn test_circumcenter_equilateral_triangle_properties() {
        // Test that circumcenter has expected properties for equilateral triangle
        let side_length = 2.0;
        let height = side_length * 3.0_f64.sqrt() / 2.0;

        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([side_length, 0.0]),
            Point::new([side_length / 2.0, height]),
        ];

        let center = circumcenter(&points).unwrap();
        let center_coords = center.to_array();

        // For equilateral triangle, circumcenter should be at centroid
        let expected_x = side_length / 2.0;
        let expected_y = height / 3.0;

        assert_relative_eq!(center_coords[0], expected_x, epsilon = 1e-10);
        assert_relative_eq!(center_coords[1], expected_y, epsilon = 1e-10);

        // Verify all vertices are equidistant from circumcenter
        let _center_point = Point::new([center_coords[0], center_coords[1]]);
        let distances: Vec<f64> = points
            .iter()
            .map(|p| {
                let p_coords: [f64; 2] = p.into();
                let diff = [
                    p_coords[0] - center_coords[0],
                    p_coords[1] - center_coords[1],
                ];
                hypot(diff)
            })
            .collect();

        // All distances should be equal
        for i in 1..distances.len() {
            assert_relative_eq!(distances[0], distances[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_circumcenter_consistency_with_circumradius() {
        // Test that circumcenter and circumradius are consistent
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let center = circumcenter(&points).unwrap();
        let radius = circumradius(&points).unwrap();
        let radius_with_center = circumradius_with_center(&points, &center).unwrap();

        // Both methods should give the same radius
        assert_relative_eq!(radius, radius_with_center, epsilon = 1e-10);

        // Verify all points are at circumradius distance from circumcenter
        let center_coords = center.to_array();
        for point in &points {
            let point_coords: [f64; 3] = point.into();
            let diff = [
                point_coords[0] - center_coords[0],
                point_coords[1] - center_coords[1],
                point_coords[2] - center_coords[2],
            ];
            let distance = hypot(diff);
            assert_relative_eq!(distance, radius, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_circumcenter_numerical_stability() {
        // Test with points that could cause numerical instability
        let points = vec![
            Point::new([1.0, 0.0]),
            Point::new([1.000_000_1, 0.0]), // Very close to first point
            Point::new([1.000_000_1, 0.000_000_1]), // Forms very thin triangle
        ];

        let result = circumcenter(&points);
        // Should either succeed or fail gracefully (not panic)
        if let Ok(center) = result {
            // If it succeeds, center should have finite coordinates
            let coords = center.to_array();
            assert!(
                coords.iter().all(|&x| x.is_finite()),
                "Circumcenter coordinates should be finite"
            );
        } else {
            // If it fails, that's acceptable for this degenerate case
        }
    }

    #[test]
    fn test_circumcenter_1d_case() {
        // Test 1D case (2 points)
        let points = vec![Point::new([0.0]), Point::new([2.0])];

        let center = circumcenter(&points).unwrap();
        let center_coords = center.to_array();

        // 1D circumcenter should be at midpoint
        assert_relative_eq!(center_coords[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_circumcenter_high_dimension() {
        // Test higher dimensional case (5D)
        let points = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]),
        ];

        let result = circumcenter(&points);
        assert!(result.is_ok(), "5D circumcenter should work");

        let center = result.unwrap();
        let center_coords = center.to_array();

        // Verify circumcenter has finite coordinates
        for coord in center_coords {
            assert!(
                coord.is_finite(),
                "Circumcenter coordinates should be finite"
            );
        }

        // For this configuration, all points are equidistant from circumcenter
        // Verify all points are at same distance from circumcenter
        let distances: Vec<f64> = points
            .iter()
            .map(|p| {
                let p_coords: [f64; 5] = p.into();
                let diff = [
                    p_coords[0] - center_coords[0],
                    p_coords[1] - center_coords[1],
                    p_coords[2] - center_coords[2],
                    p_coords[3] - center_coords[3],
                    p_coords[4] - center_coords[4],
                ];
                hypot(diff)
            })
            .collect();

        // All distances should be equal
        for i in 1..distances.len() {
            assert_relative_eq!(distances[0], distances[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn predicates_circumcenter_precise_values() {
        // Test with precisely known circumcenter values
        // Using a simplex where we can calculate the circumcenter analytically
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([6.0, 0.0, 0.0]),
            Point::new([0.0, 8.0, 0.0]),
            Point::new([0.0, 0.0, 10.0]),
        ];

        let center = circumcenter(&points).unwrap();
        let center_coords = center.to_array();

        // For this configuration, circumcenter should be at (3, 4, 5)
        assert_relative_eq!(center_coords[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(center_coords[1], 4.0, epsilon = 1e-10);
        assert_relative_eq!(center_coords[2], 5.0, epsilon = 1e-10);
    }

    // =============================================================================
    // BASIC FACET MEASURE TESTS (BY DIMENSION)
    // =============================================================================

    #[test]
    fn test_facet_measure_1d_point() {
        // 1D facet is a single point - measure should be 1.0 by convention
        let points = vec![Point::new([5.0])];
        let measure = facet_measure(&points).unwrap();
        assert_relative_eq!(measure, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_2d_line_segment() {
        // 2D: Line segment (1D facet in 2D space) - 3-4-5 triangle
        let points = vec![Point::new([0.0, 0.0]), Point::new([3.0, 4.0])];
        let measure = facet_measure(&points).unwrap();
        // Length should be sqrt(3² + 4²) = 5.0
        assert_relative_eq!(measure, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_3d_triangle_right_angle() {
        // 3D: Right triangle (area = 1/2 * base * height)
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([3.0, 0.0, 0.0]),
            Point::new([0.0, 4.0, 0.0]),
        ];
        let measure = facet_measure(&points).unwrap();
        // Area should be 3 * 4 / 2 = 6.0
        assert_relative_eq!(measure, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_4d_tetrahedron() {
        // 4D: Unit tetrahedron (3D facet in 4D space)
        let points = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
        ];
        let measure = facet_measure(&points).unwrap();

        // For a unit tetrahedron in 4D with vertices at origin and 3 unit vectors,
        // the volume should be 1/3! = 1/6
        // This is a 3-dimensional simplex in 4D space
        assert_relative_eq!(measure, 1.0 / 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gram_matrix_debug() {
        // Test the Gram matrix method against known simple cases

        // Test 1: Unit right triangle in 3D - we know this should be 0.5
        let triangle_3d = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
        ];
        let area_3d = facet_measure(&triangle_3d).unwrap();
        println!("3D triangle area: {area_3d} (expected: 0.5)");

        // Test 2: Same triangle but use direct Gram matrix calculation
        let area_3d_gram = facet_measure_gram_matrix::<f64, 3>(&triangle_3d).unwrap();
        println!("3D triangle area (Gram): {area_3d_gram} (expected: 0.5)");

        // Test 3: Unit tetrahedron in 4D - should be 1/6 ≈ 0.167
        let tetrahedron_4d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
        ];
        let volume_4d = facet_measure(&tetrahedron_4d).unwrap();
        println!(
            "4D tetrahedron volume: {} (expected: {})",
            volume_4d,
            1.0 / 6.0
        );

        // Test 4: Manual calculation for the 4D tetrahedron
        let volume_4d_gram = facet_measure_gram_matrix::<f64, 4>(&tetrahedron_4d).unwrap();
        println!(
            "4D tetrahedron volume (Gram): {} (expected: {})",
            volume_4d_gram,
            1.0 / 6.0
        );
    }

    #[test]
    fn test_facet_measure_5d_simplex() {
        // 5D: 4-dimensional facet in 5D space (4-simplex volume)
        let points = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]),
        ];
        let measure = facet_measure(&points).unwrap();

        // Volume of 4-simplex with vertices at origin and unit vectors
        // Should be 1/4! = 1/24 (generalized determinant formula)
        assert_relative_eq!(measure, 1.0 / 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_6d_simplex() {
        // 6D: 5-dimensional facet in 6D space
        let points = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ];
        let measure = facet_measure(&points).unwrap();

        // Volume of 5-simplex with vertices at origin and unit vectors
        // Should be 1/5! = 1/120
        assert_relative_eq!(measure, 1.0 / 120.0, epsilon = 1e-10);
    }

    // =============================================================================
    // FACET MEASURE ERROR HANDLING TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_wrong_point_count() {
        // Test error when wrong number of points provided
        // 3D expects 3 points, but provide 2
        let points = vec![Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])];
        let result = facet_measure::<f64, 3>(&points);

        assert!(result.is_err());
        match result.unwrap_err() {
            CircumcenterError::InvalidSimplex {
                actual,
                expected,
                dimension,
            } => {
                assert_eq!(actual, 2);
                assert_eq!(expected, 3);
                assert_eq!(dimension, 3);
            }
            other => panic!("Expected InvalidSimplex error, got: {other:?}"),
        }
    }

    // =============================================================================
    // FACET MEASURE DEGENERATE CASE TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_zero_area_triangle() {
        // Degenerate triangle (collinear points) - should have zero area
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([2.0, 0.0, 0.0]), // Collinear
        ];
        let measure = facet_measure(&points).unwrap();

        // Should be zero (or very close to zero due to numerical precision)
        assert!(
            measure < 1e-10,
            "Collinear points should have zero area, got: {measure}"
        );
    }

    #[test]
    fn test_facet_measure_nearly_collinear_points_2d() {
        // Test with points that are nearly collinear in 2D
        let eps = 1e-10;
        let points = vec![
            Point::new([0.0, 0.0]),
            Point::new([1.0, eps]), // Slightly off the x-axis
        ];

        let measure = facet_measure(&points).unwrap();
        let expected = eps.mul_add(eps, 1.0).sqrt(); // Length of line segment
        assert_relative_eq!(measure, expected, epsilon = 1e-9);
    }

    #[test]
    fn test_facet_measure_nearly_coplanar_points_3d() {
        // Test with points that are truly nearly coplanar in 3D
        let eps = 1e-8;
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([1.0, eps, eps]), // Very close to the line from (0,0,0) to (1,0,0)
        ];

        let measure = facet_measure(&points).unwrap();
        // Should be small but non-zero area
        assert!(
            measure > 0.0,
            "Nearly coplanar triangle should have positive area"
        );
        // With points very close to being collinear, area should be very small
        assert!(
            measure < 1e-6,
            "Nearly coplanar triangle should have very small area, got: {measure}"
        );
    }

    #[test]
    fn test_facet_measure_degenerate_4d_tetrahedron() {
        // Test with points that are nearly coplanar in 4D (3 points in 3D subspace)
        let points = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.5, 0.5, 0.0, 0.0]), // In the same 3D subspace
        ];

        let measure = facet_measure(&points).unwrap();
        // Should be zero (or very close to zero) since all points lie in 3D subspace
        assert!(
            measure < 1e-10,
            "Degenerate 4D tetrahedron should have zero volume, got: {measure}"
        );
    }

    // =============================================================================
    // SURFACE MEASURE TESTS
    // =============================================================================

    #[test]
    fn test_surface_measure_empty_facets() {
        // Test with empty facet collection
        use crate::core::facet::Facet;
        let facets: Vec<Facet<f64, Option<()>, Option<()>, 3>> = vec![];
        let result = surface_measure(&facets).unwrap();

        assert_relative_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_surface_measure_single_facet() {
        // Test with single triangular facet
        use crate::core::facet::Facet;
        use crate::{cell, vertex};

        // Create a right triangle
        let v1: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 0.0]);
        let v2: Vertex<f64, Option<()>, 3> = vertex!([3.0, 0.0, 0.0]);
        let v3: Vertex<f64, Option<()>, 3> = vertex!([0.0, 4.0, 0.0]);
        let v4: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 1.0]); // Fourth vertex for 3D cell

        let cell: Cell<f64, Option<()>, Option<()>, 3> = cell!(vec![v1, v2, v3, v4]);
        let facet = Facet::new(cell, v4).unwrap(); // Facet opposite to v4 (the triangle)

        let surface_area = surface_measure(&[facet]).unwrap();

        // Should be area of right triangle: 3 * 4 / 2 = 6.0
        assert_relative_eq!(surface_area, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_surface_measure_consistency_with_facet_measure() {
        // Test that surface_measure sum equals sum of individual facet_measures
        use crate::core::facet::Facet;
        use crate::{cell, vertex};

        // Create several facets
        let v1: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 0.0]);
        let v2: Vertex<f64, Option<()>, 3> = vertex!([1.0, 0.0, 0.0]);
        let v3: Vertex<f64, Option<()>, 3> = vertex!([0.0, 1.0, 0.0]);
        let v4: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 1.0]);
        let v5: Vertex<f64, Option<()>, 3> = vertex!([1.0, 1.0, 1.0]);

        let cell1: Cell<f64, Option<()>, Option<()>, 3> = cell!(vec![v1, v2, v3, v4]);
        let cell2 = cell!(vec![v2, v3, v4, v5]);

        let facet1 = Facet::new(cell1, v4).unwrap();
        let facet2 = Facet::new(cell2, v5).unwrap();

        // Calculate surface measure
        let total_surface = surface_measure(&[facet1.clone(), facet2.clone()]).unwrap();

        // Calculate individual facet measures and sum them
        let points1: Vec<Point<f64, 3>> = facet1
            .vertices()
            .iter()
            .map(|v| {
                let coords: [f64; 3] = v.into();
                Point::new(coords)
            })
            .collect();
        let points2: Vec<Point<f64, 3>> = facet2
            .vertices()
            .iter()
            .map(|v| {
                let coords: [f64; 3] = v.into();
                Point::new(coords)
            })
            .collect();

        let measure1 = facet_measure(&points1).unwrap();
        let measure2 = facet_measure(&points2).unwrap();
        let sum_individual = measure1 + measure2;

        // Should be equal
        assert_relative_eq!(total_surface, sum_individual, epsilon = 1e-10);
    }

    // =============================================================================
    // FACET MEASURE SCALING PROPERTY TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_scaled_simplex_2d() {
        // Test scaling property: measure should scale by |λ|^(D-1)
        let scale = 3.0;
        let original_points = vec![Point::new([0.0, 0.0]), Point::new([1.0, 0.0])];
        let scaled_points = vec![
            Point::new([0.0 * scale, 0.0 * scale]),
            Point::new([1.0 * scale, 0.0 * scale]),
        ];

        let original_measure = facet_measure(&original_points).unwrap();
        let scaled_measure = facet_measure(&scaled_points).unwrap();

        // For 2D (D=2), measure scales by |λ|^(2-1) = |λ|^1 = λ
        assert_relative_eq!(scaled_measure, original_measure * scale, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_scaled_simplex_3d() {
        // Test scaling property for 3D triangle (D=3, measure scales by λ^2)
        let scale = 2.5;
        let original_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([2.0, 0.0, 0.0]),
            Point::new([0.0, 3.0, 0.0]),
        ];
        let scaled_points = vec![
            Point::new([0.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([2.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([0.0 * scale, 3.0 * scale, 0.0 * scale]),
        ];

        let original_measure = facet_measure(&original_points).unwrap();
        let scaled_measure = facet_measure(&scaled_points).unwrap();

        // For 3D (D=3), measure scales by |λ|^(3-1) = λ^2
        assert_relative_eq!(
            scaled_measure,
            original_measure * scale * scale,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_facet_measure_scaled_simplex_4d() {
        // Test scaling property for 4D tetrahedron (D=4, measure scales by λ^3)
        let scale = 2.0;
        let original_points = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0]),
        ];
        let scaled_points = vec![
            Point::new([0.0 * scale, 0.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([1.0 * scale, 0.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([0.0 * scale, 1.0 * scale, 0.0 * scale, 0.0 * scale]),
            Point::new([0.0 * scale, 0.0 * scale, 1.0 * scale, 0.0 * scale]),
        ];

        let original_measure = facet_measure(&original_points).unwrap();
        let scaled_measure = facet_measure(&scaled_points).unwrap();

        // For 4D (D=4), measure scales by |λ|^(4-1) = λ^3
        assert_relative_eq!(
            scaled_measure,
            original_measure * scale.powi(3),
            epsilon = 1e-10
        );
    }

    // =============================================================================
    // EDGE CASE AND NUMERICAL STABILITY TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_very_large_coordinates() {
        // Test with very large but finite coordinates
        let large_val = 1e8;
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([large_val, 0.0, 0.0]),
            Point::new([0.0, large_val, 0.0]),
        ];

        let result = facet_measure(&points);
        assert!(result.is_ok(), "Large coordinates should work");

        let measure = result.unwrap();
        assert!(measure.is_finite(), "Measure should be finite");
        // Should be area of right triangle: large_val * large_val / 2
        let expected = large_val * large_val / 2.0;
        assert_relative_eq!(measure, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_facet_measure_very_small_coordinates() {
        // Test with very small but non-zero coordinates
        let small_val = 1e-6; // Use reasonable small value to avoid underflow
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([small_val, 0.0, 0.0]),
            Point::new([0.0, small_val, 0.0]),
        ];

        let result = facet_measure(&points);
        assert!(result.is_ok(), "Small coordinates should work");

        let measure = result.unwrap();
        assert!(measure.is_finite(), "Measure should be finite");
        // Should be area of right triangle: small_val * small_val / 2
        let expected = small_val * small_val / 2.0;
        assert_relative_eq!(measure, expected, epsilon = 1e-12);
    }

    #[test]
    fn test_facet_measure_mixed_positive_negative_coordinates() {
        // Test with mixed positive and negative coordinates
        let points = vec![
            Point::new([-1.0, -1.0, 0.0]),
            Point::new([2.0, -1.0, 0.0]),
            Point::new([-1.0, 3.0, 0.0]),
        ];

        let measure = facet_measure(&points).unwrap();
        // Triangle with base=3, height=4, area=6
        assert_relative_eq!(measure, 6.0, epsilon = 1e-10);
    }

    // =============================================================================
    // COORDINATE TYPE TESTS (f32 vs f64)
    // =============================================================================

    #[test]
    fn test_facet_measure_f32_vs_f64_consistency() {
        // Test that f32 and f64 give similar results (within tolerance)
        let points_f64 = vec![
            Point::new([0.0_f64, 0.0_f64, 0.0_f64]),
            Point::new([3.0_f64, 0.0_f64, 0.0_f64]),
            Point::new([0.0_f64, 4.0_f64, 0.0_f64]),
        ];
        let points_f32 = vec![
            Point::new([0.0_f32, 0.0_f32, 0.0_f32]),
            Point::new([3.0_f32, 0.0_f32, 0.0_f32]),
            Point::new([0.0_f32, 4.0_f32, 0.0_f32]),
        ];

        let measure_f64 = facet_measure(&points_f64).unwrap();
        let measure_f32 = facet_measure(&points_f32).unwrap();

        // Convert f32 result to f64 for comparison
        let measure_f32_as_f64 = f64::from(measure_f32);

        // Should be approximately equal (within f32 precision)
        assert_relative_eq!(measure_f64, measure_f32_as_f64, epsilon = 1e-6);
        assert_relative_eq!(measure_f64, 6.0, epsilon = 1e-10);
        assert_relative_eq!(measure_f32_as_f64, 6.0, epsilon = 1e-6);
    }

    // =============================================================================
    // GEOMETRIC INVARIANCE TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_translation_invariance() {
        // Test that translation doesn't change facet measure
        let translation = [10.0, 20.0, 30.0];
        let original_points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([3.0, 0.0, 0.0]),
            Point::new([0.0, 4.0, 0.0]),
        ];
        let translated_points = vec![
            Point::new([
                0.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ]),
            Point::new([
                3.0 + translation[0],
                0.0 + translation[1],
                0.0 + translation[2],
            ]),
            Point::new([
                0.0 + translation[0],
                4.0 + translation[1],
                0.0 + translation[2],
            ]),
        ];

        let original_measure = facet_measure(&original_points).unwrap();
        let translated_measure = facet_measure(&translated_points).unwrap();

        assert_relative_eq!(original_measure, translated_measure, epsilon = 1e-10);
        assert_relative_eq!(original_measure, 6.0, epsilon = 1e-10); // Area of 3-4-5 triangle / 2
    }

    #[test]
    fn test_facet_measure_vertex_permutation_invariance() {
        // Test that vertex order doesn't change facet measure (absolute value)
        let points_order1 = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([3.0, 0.0, 0.0]),
            Point::new([0.0, 4.0, 0.0]),
        ];
        let points_order2 = vec![
            Point::new([3.0, 0.0, 0.0]),
            Point::new([0.0, 4.0, 0.0]),
            Point::new([0.0, 0.0, 0.0]),
        ];
        let points_order3 = vec![
            Point::new([0.0, 4.0, 0.0]),
            Point::new([0.0, 0.0, 0.0]),
            Point::new([3.0, 0.0, 0.0]),
        ];

        let measure1 = facet_measure(&points_order1).unwrap();
        let measure2 = facet_measure(&points_order2).unwrap();
        let measure3 = facet_measure(&points_order3).unwrap();

        // All should give same area (measure is absolute value)
        assert_relative_eq!(measure1, measure2, epsilon = 1e-10);
        assert_relative_eq!(measure1, measure3, epsilon = 1e-10);
        assert_relative_eq!(measure1, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_facet_measure_various_triangle_orientations() {
        // Test triangles in different orientations in 3D space
        let triangles = [
            // XY plane
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
            ],
            // XZ plane
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 0.0, 1.0]),
            ],
            // YZ plane
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 1.0]),
            ],
            // Diagonal plane
            vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 1.0, 0.0]),
                Point::new([0.0, 1.0, 1.0]),
            ],
        ];

        let expected_areas = [0.5, 0.5, 0.5]; // First three are right triangles with legs of length 1

        for (i, triangle) in triangles.iter().take(3).enumerate() {
            let measure = facet_measure(triangle).unwrap();
            // Triangle should have expected area
            assert_relative_eq!(measure, expected_areas[i], epsilon = 1e-10);
        }

        // Fourth triangle has a different but computable area
        let measure4 = facet_measure(&triangles[3]).unwrap();
        assert!(
            measure4 > 0.0,
            "Diagonal triangle should have positive area"
        );
        assert!(
            measure4.is_finite(),
            "Diagonal triangle area should be finite"
        );
    }

    // =============================================================================
    // ADDITIONAL SURFACE MEASURE TESTS
    // =============================================================================

    #[test]
    fn test_surface_measure_multiple_facets_different_sizes() {
        // Test with facets of different sizes
        use crate::core::facet::Facet;
        use crate::{cell, vertex};

        // Small triangle
        let v1: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 0.0]);
        let v2: Vertex<f64, Option<()>, 3> = vertex!([1.0, 0.0, 0.0]);
        let v3: Vertex<f64, Option<()>, 3> = vertex!([0.0, 1.0, 0.0]);
        let v4: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 1.0]);

        // Large triangle
        let v5: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 0.0]);
        let v6: Vertex<f64, Option<()>, 3> = vertex!([6.0, 0.0, 0.0]);
        let v7: Vertex<f64, Option<()>, 3> = vertex!([0.0, 8.0, 0.0]);
        let v8: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 1.0]);

        let cell1: Cell<f64, Option<()>, Option<()>, 3> = cell!(vec![v1, v2, v3, v4]);
        let cell2: Cell<f64, Option<()>, Option<()>, 3> = cell!(vec![v5, v6, v7, v8]);

        let facet1 = Facet::new(cell1, v4).unwrap(); // Area = 0.5
        let facet2 = Facet::new(cell2, v8).unwrap(); // Area = 24.0

        let total_surface = surface_measure(&[facet1, facet2]).unwrap();
        let expected_total = 0.5 + 24.0;

        assert_relative_eq!(total_surface, expected_total, epsilon = 1e-10);
    }

    // =============================================================================
    // 2D AND 4D+ SURFACE MEASURE TESTS
    // =============================================================================

    #[test]
    fn test_surface_measure_2d_perimeter() {
        // Test 2D surface measure (perimeter of polygon)
        use crate::core::facet::Facet;
        use crate::{cell, vertex};

        // Create 2D triangle
        let v1: Vertex<f64, Option<()>, 2> = vertex!([0.0, 0.0]);
        let v2: Vertex<f64, Option<()>, 2> = vertex!([3.0, 0.0]);
        let v3: Vertex<f64, Option<()>, 2> = vertex!([0.0, 4.0]);

        let cell: Cell<f64, Option<()>, Option<()>, 2> = cell!(vec![v1, v2, v3]);

        // Create facets for each edge
        let edge1 = Facet::new(cell.clone(), v3).unwrap(); // Edge v1-v2
        let edge2 = Facet::new(cell.clone(), v1).unwrap(); // Edge v2-v3
        let edge3 = Facet::new(cell, v2).unwrap(); // Edge v3-v1

        let total_perimeter = surface_measure(&[edge1, edge2, edge3]).unwrap();

        // Perimeter should be 3 + 4 + 5 = 12 (sides of 3-4-5 triangle)
        assert_relative_eq!(total_perimeter, 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_surface_measure_4d_boundary() {
        // Test 4D surface measure (3D boundary facets)
        use crate::core::facet::Facet;
        use crate::{cell, vertex};

        // Create 4D simplex (5 vertices)
        let v1: Vertex<f64, Option<()>, 4> = vertex!([0.0, 0.0, 0.0, 0.0]);
        let v2: Vertex<f64, Option<()>, 4> = vertex!([1.0, 0.0, 0.0, 0.0]);
        let v3: Vertex<f64, Option<()>, 4> = vertex!([0.0, 1.0, 0.0, 0.0]);
        let v4: Vertex<f64, Option<()>, 4> = vertex!([0.0, 0.0, 1.0, 0.0]);
        let v5: Vertex<f64, Option<()>, 4> = vertex!([0.0, 0.0, 0.0, 1.0]);

        let cell: Cell<f64, Option<()>, Option<()>, 4> = cell!(vec![v1, v2, v3, v4, v5]);

        // Create boundary facets (tetrahedra)
        let facets = vec![
            Facet::new(cell.clone(), v5).unwrap(), // Tetrahedron without v5
            Facet::new(cell.clone(), v4).unwrap(), // Tetrahedron without v4
            Facet::new(cell.clone(), v3).unwrap(), // Tetrahedron without v3
            Facet::new(cell.clone(), v2).unwrap(), // Tetrahedron without v2
            Facet::new(cell, v1).unwrap(),         // Tetrahedron without v1
        ];

        let total_surface = surface_measure(&facets).unwrap();

        // The correct total surface area is 1.0, not 5/6 as originally expected
        // This is because the boundary facets have different volumes:
        // - 4 facets that include the origin: each has volume 1/6
        // - 1 facet that excludes the origin: has volume 1/3
        // Total: 4×(1/6) + 1×(1/3) = 4/6 + 2/6 = 1.0
        let expected_total = 1.0;
        assert_relative_eq!(total_surface, expected_total, epsilon = 1e-10);
    }

    // =============================================================================
    // ERROR PROPAGATION TESTS
    // =============================================================================

    #[test]
    fn test_surface_measure_with_invalid_facet() {
        // Test error handling when facet measure calculation fails
        use crate::core::facet::Facet;
        use crate::{cell, vertex};

        // Create a valid facet
        let v1: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 0.0]);
        let v2: Vertex<f64, Option<()>, 3> = vertex!([1.0, 0.0, 0.0]);
        let v3: Vertex<f64, Option<()>, 3> = vertex!([0.0, 1.0, 0.0]);
        let v4: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 1.0]);

        let cell: Cell<f64, Option<()>, Option<()>, 3> = cell!(vec![v1, v2, v3, v4]);
        let facet = Facet::new(cell, v4).unwrap();

        // Test with valid facets - should work
        let result = surface_measure(&[facet]);
        assert!(result.is_ok(), "Valid facets should work");
        assert_relative_eq!(result.unwrap(), 0.5, epsilon = 1e-10);
    }

    // =============================================================================
    // PERFORMANCE AND STRESS TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_performance_many_dimensions() {
        // Test performance with higher dimensions (7D, 8D)
        let points_7d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ];

        let measure_7d = facet_measure(&points_7d).unwrap();
        // Volume of 6-simplex should be 1/6! = 1/720
        assert_relative_eq!(measure_7d, 1.0 / 720.0, epsilon = 1e-10);

        let points_8d = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ];

        let measure_8d = facet_measure(&points_8d).unwrap();
        // Volume of 7-simplex should be 1/7! = 1/5040
        assert_relative_eq!(measure_8d, 1.0 / 5040.0, epsilon = 1e-10);
    }

    #[test]
    fn test_surface_measure_many_facets() {
        // Test with many facets to ensure linear scaling
        use crate::core::facet::Facet;
        use crate::{cell, vertex};

        let mut facets = Vec::new();
        let mut expected_total = 0.0;

        // Create 10 different triangular facets
        for i in 0..10 {
            let scale = f64::from(i + 1);
            let v1: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 0.0]);
            let v2: Vertex<f64, Option<()>, 3> = vertex!([scale, 0.0, 0.0]);
            let v3: Vertex<f64, Option<()>, 3> = vertex!([0.0, scale, 0.0]);
            let v4: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 1.0]);

            let cell: Cell<f64, Option<()>, Option<()>, 3> = cell!(vec![v1, v2, v3, v4]);
            let facet = Facet::new(cell, v4).unwrap();

            // Each triangle has area scale * scale / 2
            expected_total += scale * scale / 2.0;
            facets.push(facet);
        }

        let total_surface = surface_measure(&facets).unwrap();
        assert_relative_eq!(total_surface, expected_total, epsilon = 1e-10);
    }

    // =============================================================================
    // ADVANCED GEOMETRIC PROPERTY TESTS
    // =============================================================================

    #[test]
    fn test_facet_measure_equilateral_triangles() {
        // Test equilateral triangles of various sizes
        let side_lengths = [1.0, 2.0, 5.0, 10.0];

        for &side in &side_lengths {
            let height = side * 3.0_f64.sqrt() / 2.0;
            let points = vec![
                Point::new([0.0, 0.0, 0.0]),
                Point::new([side, 0.0, 0.0]),
                Point::new([side / 2.0, height, 0.0]),
            ];

            let measure = facet_measure(&points).unwrap();
            let expected_area = side * side * 3.0_f64.sqrt() / 4.0; // Formula for equilateral triangle area

            // Equilateral triangle should have expected area
            assert_relative_eq!(measure, expected_area, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_facet_measure_regular_tetrahedron_faces() {
        // Test faces of regular tetrahedron
        let side = 2.0;
        let height = side * (2.0_f64 / 3.0).sqrt();
        let center_offset = side / (2.0 * 3.0_f64.sqrt());

        // Regular tetrahedron vertices
        let v1 = Point::new([0.0, 0.0, 0.0]);
        let v2 = Point::new([side, 0.0, 0.0]);
        let v3 = Point::new([side / 2.0, side * 3.0_f64.sqrt() / 2.0, 0.0]);
        let v4 = Point::new([side / 2.0, center_offset, height]);

        // Test each face
        let faces = [
            vec![v1, v2, v3], // Base
            vec![v1, v2, v4], // Face 1
            vec![v2, v3, v4], // Face 2
            vec![v3, v1, v4], // Face 3
        ];

        let expected_face_area = side * side * 3.0_f64.sqrt() / 4.0; // Equilateral triangle area

        for face in &faces {
            let measure = facet_measure(face).unwrap();
            // Face of regular tetrahedron should have expected area
            assert_relative_eq!(measure, expected_face_area, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_facet_measure_reflection_invariance() {
        // Test that reflection doesn't change facet measure
        let original_points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
        ];

        // Reflect across various planes
        let reflections = [
            // Reflect x-coordinate
            vec![
                Point::new([-1.0, 2.0, 3.0]),
                Point::new([-4.0, 5.0, 6.0]),
                Point::new([-7.0, 8.0, 9.0]),
            ],
            // Reflect y-coordinate
            vec![
                Point::new([1.0, -2.0, 3.0]),
                Point::new([4.0, -5.0, 6.0]),
                Point::new([7.0, -8.0, 9.0]),
            ],
            // Reflect z-coordinate
            vec![
                Point::new([1.0, 2.0, -3.0]),
                Point::new([4.0, 5.0, -6.0]),
                Point::new([7.0, 8.0, -9.0]),
            ],
        ];

        let original_measure = facet_measure(&original_points).unwrap();

        for reflected_points in &reflections {
            let reflected_measure = facet_measure(reflected_points).unwrap();
            // Reflection should preserve facet measure
            assert_relative_eq!(original_measure, reflected_measure, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_facet_measure_rotation_invariance_2d() {
        // Test that rotation doesn't change 2D facet measure (line length)
        let original_points = vec![Point::new([0.0, 0.0]), Point::new([3.0, 4.0])];

        // Rotate by 90 degrees
        let rotated_points = vec![
            Point::new([0.0, 0.0]),
            Point::new([-4.0, 3.0]), // 90° rotation of (3,4)
        ];

        let original_measure = facet_measure(&original_points).unwrap();
        let rotated_measure = facet_measure(&rotated_points).unwrap();

        assert_relative_eq!(original_measure, rotated_measure, epsilon = 1e-10);
        assert_relative_eq!(original_measure, 5.0, epsilon = 1e-10); // Both should be 5.0
    }

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
    fn test_generate_random_points_coordinate_types() {
        // Test with different coordinate scalar types across dimensions

        // f64 tests
        let points_f64_2d = generate_random_points::<f64, 2>(20, (0.0, 1.0)).unwrap();
        let points_f64_3d = generate_random_points::<f64, 3>(20, (0.0, 1.0)).unwrap();
        let points_f64_4d = generate_random_points::<f64, 4>(20, (0.0, 1.0)).unwrap();
        let points_f64_5d = generate_random_points::<f64, 5>(20, (0.0, 1.0)).unwrap();

        // f32 tests
        let points_f32_2d = generate_random_points::<f32, 2>(20, (0.0_f32, 1.0_f32)).unwrap();
        let points_f32_3d = generate_random_points::<f32, 3>(20, (0.0_f32, 1.0_f32)).unwrap();
        let points_f32_4d = generate_random_points::<f32, 4>(20, (0.0_f32, 1.0_f32)).unwrap();
        let points_f32_5d = generate_random_points::<f32, 5>(20, (0.0_f32, 1.0_f32)).unwrap();

        // Verify lengths
        assert_eq!(points_f64_2d.len(), 20);
        assert_eq!(points_f64_3d.len(), 20);
        assert_eq!(points_f64_4d.len(), 20);
        assert_eq!(points_f64_5d.len(), 20);
        assert_eq!(points_f32_2d.len(), 20);
        assert_eq!(points_f32_3d.len(), 20);
        assert_eq!(points_f32_4d.len(), 20);
        assert_eq!(points_f32_5d.len(), 20);

        // Check ranges for f64 points
        for point in &points_f64_2d {
            let coords: [f64; 2] = point.into();
            for &coord in &coords {
                assert!((0.0..1.0).contains(&coord));
            }
        }

        // Check ranges for f32 points
        for point in &points_f32_5d {
            let coords: [f32; 5] = point.into();
            for &coord in &coords {
                assert!((0.0..1.0).contains(&coord));
            }
        }
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

    #[test]
    fn test_generate_grid_points_large_coordinates() {
        // Test with large coordinate values
        let grid = generate_grid_points::<f64, 2>(2, 1000.0, [1e6, 1e6]).unwrap();
        assert_eq!(grid.len(), 4);

        for point in &grid {
            let coords: [f64; 2] = point.into();
            assert!((1e6..=1e6 + 1000.0).contains(&coords[0]));
            assert!((1e6..=1e6 + 1000.0).contains(&coords[1]));
        }
    }

    #[test]
    fn test_generate_grid_points_negative_spacing() {
        // Test with negative spacing
        let grid = generate_grid_points::<f64, 2>(3, -1.0, [2.0, 2.0]).unwrap();
        assert_eq!(grid.len(), 9);

        // With negative spacing, coordinates should decrease from offset
        // Grid indices are 0, 1, 2, so coordinates are:
        // offset + index * spacing = 2.0 + {0,1,2} * (-1.0) = {2.0, 1.0, 0.0}
        let mut found_coords = std::collections::HashSet::new();
        for point in &grid {
            let coords: [f64; 2] = point.into();
            // Each coordinate should be exactly one of: 2.0, 1.0, 0.0
            for &coord in &coords {
                assert!(
                    (coord - 2.0).abs() < 1e-15
                        || (coord - 1.0).abs() < 1e-15
                        || coord.abs() < 1e-15,
                    "Unexpected coordinate: {coord}"
                );
            }
            found_coords.insert(format!("{:.1}_{:.1}", coords[0], coords[1]));
        }

        // Should find all 9 expected coordinate combinations
        assert_eq!(
            found_coords.len(),
            9,
            "Should have 9 unique coordinate pairs"
        );
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

    #[test]
    fn test_generate_poisson_points_small_spacing() {
        // Test with very small minimum distance (should behave more like random sampling)
        let points = generate_poisson_points::<f64, 2>(20, (0.0, 10.0), 0.01, 999).unwrap();

        // Should be able to generate close to the requested number
        assert!(points.len() >= 15); // Allow some margin for randomness

        // All points should still respect the minimum distance
        for (i, p1) in points.iter().enumerate() {
            for (j, p2) in points.iter().enumerate() {
                if i != j {
                    let coords1: [f64; 2] = p1.into();
                    let coords2: [f64; 2] = p2.into();
                    let diff = [coords1[0] - coords2[0], coords1[1] - coords2[1]];
                    let distance = hypot(diff);
                    assert!(distance >= 0.01 - 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_generate_poisson_points_coordinate_types() {
        // Test with f32 coordinates
        let points_f32 =
            generate_poisson_points::<f32, 3>(15, (0.0_f32, 5.0_f32), 0.5_f32, 333).unwrap();

        assert!(!points_f32.is_empty());

        for point in &points_f32 {
            let coords: [f32; 3] = point.into();
            for &coord in &coords {
                assert!((0.0..5.0).contains(&coord));
            }
        }

        // Test with f64 coordinates
        let points_f64 = generate_poisson_points::<f64, 4>(10, (-2.0, 2.0), 0.4, 777).unwrap();

        assert!(!points_f64.is_empty());

        for point in &points_f64 {
            let coords: [f64; 4] = point.into();
            for &coord in &coords {
                assert!((-2.0..2.0).contains(&coord));
            }
        }
    }

    // =============================================================================
    // SAFETY CAP AND UTILITY FUNCTION TESTS
    // =============================================================================

    #[test]
    fn test_max_grid_bytes_safety_cap_default_value() {
        // Test that the default value is as expected
        assert_eq!(MAX_GRID_BYTES_SAFETY_CAP_DEFAULT, 4_294_967_296); // 4 GiB

        // Test that the function doesn't crash and returns a reasonable value
        // (We can't safely test env var manipulation in a crate that forbids unsafe)
        let cap = max_grid_bytes_safety_cap();
        assert!(cap > 0, "Safety cap should be positive");
        // The actual value depends on environment, but function should not crash
    }

    #[test]
    fn test_format_bytes() {
        // Test byte formatting with various sizes
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KiB");
        assert_eq!(format_bytes(1536), "1.5 KiB");
        assert_eq!(format_bytes(1_048_576), "1.0 MiB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GiB");
        assert_eq!(format_bytes(MAX_GRID_BYTES_SAFETY_CAP_DEFAULT), "4.0 GiB");
        assert_eq!(format_bytes(5_368_709_120), "5.0 GiB");
        assert_eq!(format_bytes(1_099_511_627_776), "1.0 TiB");
    }

    #[test]
    fn test_generate_grid_points_safety_cap_enforcement() {
        // Test that very large grids fail with the current safety cap
        // We can't safely modify environment variables in a crate that forbids unsafe,
        // so we test with a grid that would exceed the default 4 GiB cap
        let result = generate_grid_points::<f64, 3>(1000, 1.0, [0.0, 0.0, 0.0]);
        assert!(result.is_err(), "Should fail with safety cap exceeded");

        if let Err(RandomPointGenerationError::RandomGenerationFailed { details, .. }) = result {
            assert!(
                details.contains("cap"),
                "Error should mention cap: {details}"
            );
            // Should contain human-readable formatting
            assert!(
                details.contains('B'),
                "Error should use human-readable bytes: {details}"
            );
            // Should contain one of the byte unit suffixes
            assert!(
                details.contains("GiB") || details.contains("MiB") || details.contains("KiB"),
                "Error should use human-readable byte units: {details}"
            );
        } else {
            panic!("Expected RandomGenerationFailed error");
        }
    }

    #[test]
    fn test_scaled_hypot_2d() {
        // Test our fallback scaled hypot implementation
        let result = scaled_hypot_2d(3.0, 4.0);
        assert_relative_eq!(result, 5.0, epsilon = 1e-10);

        // Test with zero values
        let zero_result = scaled_hypot_2d(0.0, 0.0);
        assert_relative_eq!(zero_result, 0.0, epsilon = 1e-10);

        // Test with negative values
        let neg_result = scaled_hypot_2d(-3.0, 4.0);
        assert_relative_eq!(neg_result, 5.0, epsilon = 1e-10);

        // Test with very large values to check scaling behavior
        let large_result = scaled_hypot_2d(1e10, 1e10);
        let expected = (2.0_f64).sqrt() * 1e10;
        assert_relative_eq!(large_result, expected, epsilon = 1e-5);
    }

    // =============================================================================
    // RANDOM TRIANGULATION GENERATION TESTS
    // =============================================================================

    #[test]
    fn test_generate_random_triangulation_basic() {
        // Test 2D triangulation creation
        let triangulation_2d =
            generate_random_triangulation::<f64, (), (), 2>(10, (-5.0, 5.0), None, Some(42))
                .unwrap();

        assert_eq!(triangulation_2d.number_of_vertices(), 10);
        assert_eq!(triangulation_2d.dim(), 2);
        assert!(triangulation_2d.is_valid().is_ok());

        // Test 3D triangulation creation with data
        let triangulation_3d =
            generate_random_triangulation::<f64, i32, (), 3>(8, (0.0, 1.0), Some(123), Some(456))
                .unwrap();

        assert_eq!(triangulation_3d.number_of_vertices(), 8);
        assert_eq!(triangulation_3d.dim(), 3);
        assert!(triangulation_3d.is_valid().is_ok());

        // Test seeded vs unseeded (should get different results)
        let triangulation_seeded =
            generate_random_triangulation::<f64, (), (), 2>(5, (-1.0, 1.0), None, Some(789))
                .unwrap();

        let triangulation_unseeded =
            generate_random_triangulation::<f64, (), (), 2>(5, (-1.0, 1.0), None, None).unwrap();

        // Both should be valid
        assert!(triangulation_seeded.is_valid().is_ok());
        assert!(triangulation_unseeded.is_valid().is_ok());
        assert_eq!(triangulation_seeded.number_of_vertices(), 5);
        assert_eq!(triangulation_unseeded.number_of_vertices(), 5);
    }

    #[test]
    fn test_generate_random_triangulation_error_cases() {
        // Test invalid bounds
        let result = generate_random_triangulation::<f64, (), (), 2>(
            10,
            (5.0, 1.0), // min > max
            None,
            Some(42),
        );
        assert!(result.is_err());

        // Test zero points
        let result =
            generate_random_triangulation::<f64, (), (), 2>(0, (-1.0, 1.0), None, Some(42));
        assert!(result.is_ok()); // Should succeed with empty triangulation
        let triangulation = result.unwrap();
        assert_eq!(triangulation.number_of_vertices(), 0);
        assert_eq!(triangulation.dim(), -1);
    }

    #[test]
    fn test_generate_random_triangulation_reproducibility() {
        // Same seed should produce identical triangulations
        let triangulation1 =
            generate_random_triangulation::<f64, (), (), 3>(6, (-2.0, 2.0), None, Some(12345))
                .unwrap();

        let triangulation2 =
            generate_random_triangulation::<f64, (), (), 3>(6, (-2.0, 2.0), None, Some(12345))
                .unwrap();

        // Should have same structural properties
        assert_eq!(
            triangulation1.number_of_vertices(),
            triangulation2.number_of_vertices()
        );
        assert_eq!(
            triangulation1.number_of_cells(),
            triangulation2.number_of_cells()
        );
        assert_eq!(triangulation1.dim(), triangulation2.dim());
    }

    #[test]
    fn test_generate_random_triangulation_dimensions() {
        // Test different dimensional triangulations

        // 2D with sufficient points for full triangulation
        let tri_2d =
            generate_random_triangulation::<f64, (), (), 2>(15, (0.0, 10.0), None, Some(555))
                .unwrap();
        assert_eq!(tri_2d.dim(), 2);
        assert!(tri_2d.number_of_cells() > 0);

        // 3D with sufficient points for full triangulation
        let tri_3d =
            generate_random_triangulation::<f64, (), (), 3>(20, (-3.0, 3.0), None, Some(666))
                .unwrap();
        assert_eq!(tri_3d.dim(), 3);
        assert!(tri_3d.number_of_cells() > 0);

        // 4D with sufficient points for full triangulation
        let tri_4d =
            generate_random_triangulation::<f64, (), (), 4>(12, (-1.0, 1.0), None, Some(777))
                .unwrap();
        assert_eq!(tri_4d.dim(), 4);
        assert!(tri_4d.number_of_cells() > 0);
    }

    #[test]
    fn test_generate_random_triangulation_with_data() {
        // Test with different data types for vertices

        // Test with fixed-size character array (Copy type that can represent strings)
        // NOTE: This is a workaround for the DataType trait requiring Copy, which
        // prevents using String or &str directly due to lifetime/ownership constraints
        let tri_with_char_array = generate_random_triangulation::<f64, [char; 8], (), 2>(
            6,
            (-2.0, 2.0),
            Some(['v', 'e', 'r', 't', 'e', 'x', '_', 'd']),
            Some(888),
        )
        .unwrap();

        assert_eq!(tri_with_char_array.number_of_vertices(), 6);
        assert!(tri_with_char_array.is_valid().is_ok());

        // Convert the char array to a string to demonstrate string-like usage
        let char_array_data = ['v', 'e', 'r', 't', 'e', 'x', '_', 'd'];
        let string_representation: String = char_array_data.iter().collect();
        assert_eq!(string_representation, "vertex_d");

        // Test with integer data
        let tri_with_int_data =
            generate_random_triangulation::<f64, u32, (), 3>(8, (0.0, 5.0), Some(42u32), Some(999))
                .unwrap();

        assert_eq!(tri_with_int_data.number_of_vertices(), 8);
        assert!(tri_with_int_data.is_valid().is_ok());

        // Test without data (None)
        let tri_no_data =
            generate_random_triangulation::<f64, (), (), 2>(5, (-1.0, 1.0), None, Some(111))
                .unwrap();

        assert_eq!(tri_no_data.number_of_vertices(), 5);
        assert!(tri_no_data.is_valid().is_ok());
    }
}
