//! Vector norm and distance computations.
//!
//! This module provides numerically stable functions for computing norms and
//! distances of d-dimensional vectors.

use num_traits::Float;

use crate::geometry::traits::coordinate::CoordinateScalar;

use super::conversions::{safe_scalar_from_f64, safe_scalar_to_f64};

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
pub(in crate::geometry::util) fn scaled_hypot_2d<T: CoordinateScalar>(x: T, y: T) -> T {
    let max_abs = Float::abs(x).max(Float::abs(y));
    if max_abs == T::zero() {
        return T::zero();
    }
    // Use scaled computation for numerical stability
    let x_scaled = x / max_abs;
    let y_scaled = y / max_abs;
    max_abs * Float::sqrt(x_scaled * x_scaled + y_scaled * y_scaled)
}

/// Helper function to compute squared norm using generic arithmetic on T.
///
/// This function computes the sum of squares of coordinates using generic
/// arithmetic operations on type T, avoiding premature conversion to f64.
///
/// # Arguments
///
/// * `coords` - Reference to array of coordinates of type T
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
/// let norm_sq = squared_norm(&coords_2d);
/// assert_eq!(norm_sq, 25.0); // 3² + 4² = 9 + 16 = 25
///
/// // 3D vector
/// let coords_3d = [1.0, 2.0, 2.0];
/// let norm_sq_3d = squared_norm(&coords_3d);
/// assert_eq!(norm_sq_3d, 9.0); // 1² + 2² + 2² = 1 + 4 + 4 = 9
///
/// // 4D vector
/// let coords_4d = [1.0, 1.0, 1.0, 1.0];
/// let norm_sq_4d = squared_norm(&coords_4d);
/// assert_eq!(norm_sq_4d, 4.0); // 1² + 1² + 1² + 1² = 4
/// ```
pub fn squared_norm<T, const D: usize>(coords: &[T; D]) -> T
where
    T: CoordinateScalar,
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
/// * `coords` - Reference to array of coordinates of type T
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
/// let distance_2d = hypot(&[3.0, 4.0]);
/// assert_eq!(distance_2d, 5.0);
///
/// // 3D case - uses generalized algorithm
/// let distance_3d = hypot(&[1.0, 2.0, 2.0]);
/// assert_eq!(distance_3d, 3.0);
///
/// // Higher dimensions
/// let distance_4d = hypot(&[1.0, 1.0, 1.0, 1.0]);
/// assert_eq!(distance_4d, 2.0);
/// ```
pub fn hypot<T, const D: usize>(coords: &[T; D]) -> T
where
    T: CoordinateScalar,
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hypot_2d() {
        // Test 2D case - should use std::f64::hypot
        let distance = hypot(&[3.0, 4.0]);
        assert_relative_eq!(distance, 5.0, epsilon = 1e-10);

        // Test with zero
        let distance_zero = hypot(&[0.0, 0.0]);
        assert_relative_eq!(distance_zero, 0.0, epsilon = 1e-10);

        // Test with negative values
        let distance_neg = hypot(&[-3.0, 4.0]);
        assert_relative_eq!(distance_neg, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hypot_3d() {
        // Test 3D case - uses generalized algorithm
        let distance = hypot(&[1.0, 2.0, 2.0]);
        assert_relative_eq!(distance, 3.0, epsilon = 1e-10);

        // Test unit vector in 3D
        let distance_unit = hypot(&[1.0, 0.0, 0.0]);
        assert_relative_eq!(distance_unit, 1.0, epsilon = 1e-10);

        // Test with all equal components
        let distance_equal = hypot(&[1.0, 1.0, 1.0]);
        assert_relative_eq!(distance_equal, 3.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_hypot_4d() {
        // Test 4D case
        let distance = hypot(&[1.0, 1.0, 1.0, 1.0]);
        assert_relative_eq!(distance, 2.0, epsilon = 1e-10);

        // Test with zero vector
        let distance_zero = hypot(&[0.0, 0.0, 0.0, 0.0]);
        assert_relative_eq!(distance_zero, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hypot_edge_cases() {
        // Test 0D case
        let distance_0d = hypot::<f64, 0>(&[]);
        assert_relative_eq!(distance_0d, 0.0, epsilon = 1e-10);

        // Test 1D case
        let distance_1d_pos = hypot(&[5.0]);
        assert_relative_eq!(distance_1d_pos, 5.0, epsilon = 1e-10);

        let distance_1d_neg = hypot(&[-5.0]);
        assert_relative_eq!(distance_1d_neg, 5.0, epsilon = 1e-10);

        // Test large values that might cause overflow with naive sqrt(x² + y²)
        let distance_large = hypot(&[1e200, 1e200]);
        assert!(distance_large.is_finite());
        assert!(distance_large > 0.0);
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

    #[test]
    fn test_scaled_hypot_2d_edge_cases() {
        // Test with zeros
        let result = scaled_hypot_2d(0.0, 0.0);
        assert_relative_eq!(result, 0.0);

        // Test with one zero
        let result = scaled_hypot_2d(0.0, 5.0);
        assert_relative_eq!(result, 5.0);

        // Test with standard case
        let result = scaled_hypot_2d(3.0, 4.0);
        assert_relative_eq!(result, 5.0, epsilon = 1e-10);

        // Test with negative values
        let result = scaled_hypot_2d(-3.0, -4.0);
        assert_relative_eq!(result, 5.0, epsilon = 1e-10);

        // Test with very large values (avoid overflow)
        let large_val = 1e100;
        let result = scaled_hypot_2d(large_val, large_val);
        assert!(result.is_finite());
        assert!(result > 0.0);
    }

    #[test]
    fn test_squared_norm_dimensions_2_to_5() {
        // Test across dimensions 2-5
        // Test 2D
        let norm_2d = squared_norm(&[3.0, 4.0]);
        assert_relative_eq!(norm_2d, 25.0);

        // Test 3D
        let norm_3d = squared_norm(&[1.0, 2.0, 2.0]);
        assert_relative_eq!(norm_3d, 9.0);

        // Test 4D
        let norm_4d = squared_norm(&[1.0, 1.0, 1.0, 1.0]);
        assert_relative_eq!(norm_4d, 4.0);

        // Test 5D
        let norm_5d = squared_norm(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_relative_eq!(norm_5d, 5.0);

        // Test with zeros
        let norm_zero = squared_norm(&[0.0, 0.0, 0.0]);
        assert_relative_eq!(norm_zero, 0.0);
    }

    #[test]
    fn test_hypot_dimensions_2_to_5() {
        // Test hypot across dimensions 2-5

        // Test 2D case (uses optimized scaled_hypot_2d)
        let distance_2d = hypot(&[3.0, 4.0]);
        assert_relative_eq!(distance_2d, 5.0, epsilon = 1e-10);

        // Test 3D case
        let distance_3d = hypot(&[1.0, 2.0, 2.0]);
        assert_relative_eq!(distance_3d, 3.0, epsilon = 1e-10);

        // Test 4D case
        let distance_4d = hypot(&[1.0, 1.0, 1.0, 1.0]);
        assert_relative_eq!(distance_4d, 2.0, epsilon = 1e-10);

        // Test 5D case (uses general algorithm)
        let distance_5d = hypot(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_relative_eq!(distance_5d, 5.0_f64.sqrt(), epsilon = 1e-10);

        // Test with mixed large and small values
        let distance_mixed = hypot(&[1e10, 1e-10, 1e5]);
        assert!(distance_mixed.is_finite());
        assert!(distance_mixed > 0.0);
    }
}
