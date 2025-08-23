//! Geometric utility functions for d-dimensional geometry calculations.
//!
//! This module contains utility functions for computing distances, norms, and
//! circumsphere properties of geometric objects. These functions are used by
//! both predicates and other geometric algorithms.

use num_traits::{Float, Zero};
use peroxide::fuga::{MatrixTrait, zeros};
use std::iter::Sum;

use crate::geometry::matrix::{MatrixError, invert};
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{
    Coordinate, CoordinateConversionError, CoordinateScalar,
};
use num_traits::cast;

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
}

// ============================================================================
// Safe Coordinate Conversion Functions
// ============================================================================

/// Safely convert a coordinate value from type T to f64.
///
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
                if let Ok(result) = safe_scalar_from_f64(result_f64) {
                    result
                } else {
                    // Fall back to general algorithm if conversion back fails
                    let max_abs = Float::abs(coords[0]).max(Float::abs(coords[1]));
                    if max_abs == T::zero() {
                        return T::zero();
                    }
                    // Use scaled computation for numerical stability
                    let x_scaled = coords[0] / max_abs;
                    let y_scaled = coords[1] / max_abs;
                    max_abs * Float::sqrt(x_scaled * x_scaled + y_scaled * y_scaled)
                }
            } else {
                // Fall back to general algorithm if conversion fails
                let max_abs = Float::abs(coords[0]).max(Float::abs(coords[1]));
                if max_abs == T::zero() {
                    return T::zero();
                }
                // Use scaled computation for numerical stability
                let x_scaled = coords[0] / max_abs;
                let y_scaled = coords[1] / max_abs;
                max_abs * Float::sqrt(x_scaled * x_scaled + y_scaled * y_scaled)
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

#[cfg(test)]
mod tests {
    use super::*;
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
}
