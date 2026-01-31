//! Safe coordinate conversion functions between numeric types.
//!
//! This module provides functions for safely converting coordinate values
//! between different numeric types with proper error handling and finite-value
//! checking.

use num_traits::cast;

use crate::geometry::traits::coordinate::{CoordinateConversionError, CoordinateScalar};

// Re-export error type
pub use super::ValueConversionError;

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
pub(in crate::geometry::util) fn safe_cast_to_f64<T: CoordinateScalar>(
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
pub(in crate::geometry::util) fn safe_cast_from_f64<T: CoordinateScalar>(
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
/// with proper precision checking. The conversion goes through `f64` as an intermediate,
/// so we must guard against precision loss in both the f64 conversion and the final
/// conversion to type T. The function uses the minimum mantissa bits of f64 (53 bits)
/// and the target type T to determine the safe conversion limit.
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
/// - The `usize` value is too large and would lose precision in the conversion chain
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
/// The function uses the minimum mantissa bits between f64 and the target type:
/// - `f64` integers are exact up to 2^53 − 1 (9,007,199,254,740,991)
/// - `f32` integers are exact up to 2^24 − 1 (16,777,215)
/// - For f32 targets: `usize` values larger than 2^24 − 1 will cause an error
/// - For f64 targets: `usize` values larger than 2^53 − 1 will cause an error
/// - On 32-bit platforms, `usize` is only 32 bits, so f32 precision loss is possible
/// - On 64-bit platforms, `usize` can be up to 64 bits, so both f32 and f64 precision loss is possible
pub fn safe_usize_to_scalar<T: CoordinateScalar>(
    value: usize,
) -> Result<T, CoordinateConversionError> {
    // Guard precision for both the f64 intermediate and the target T.
    // Use mantissa bits min(53 (f64), T::mantissa_digits()) to bound exact integers.
    const F64_MANTISSA_BITS: u32 = 53;
    let t_mantissa_bits: u32 = T::mantissa_digits();
    let max_precise_bits = core::cmp::min(F64_MANTISSA_BITS, t_mantissa_bits);
    let max_precise_u128: u128 = (1u128 << max_precise_bits) - 1;

    // Use try_from to safely convert usize to u64 for comparison
    let value_u64 =
        u64::try_from(value).map_err(|_| CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: format!("{value}"),
            from_type: "usize",
            to_type: std::any::type_name::<T>(),
        })?;

    if u128::from(value_u64) > max_precise_u128 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::traits::coordinate::CoordinateConversionError;
    use crate::geometry::util::{CircumcenterError, RandomPointGenerationError};
    use approx::assert_relative_eq;
    use num_traits::cast;

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
    fn test_safe_usize_to_scalar_within_f64_precision() {
        // Test values that are within f64 precision limits but relatively large
        let safe_large_values = [
            usize::try_from(1_u64 << 50).unwrap_or(usize::MAX), // 2^50, well within f64 precision
            usize::try_from(1_u64 << 51).unwrap_or(usize::MAX), // 2^51, still safe
            usize::try_from((1_u64 << 53) - 2).unwrap_or(usize::MAX), // Just under 2^53-1, should be safe
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
        // Test the exact boundary value: 2^53-1
        const MAX_PRECISE_USIZE_IN_F64: u64 = (1_u64 << 53) - 1;

        // This should succeed (exactly at the boundary)
        if usize::try_from(MAX_PRECISE_USIZE_IN_F64).is_ok() {
            let boundary_value = usize::try_from(MAX_PRECISE_USIZE_IN_F64).unwrap();
            let result: Result<f64, _> = safe_usize_to_scalar(boundary_value);
            assert!(
                result.is_ok(),
                "Boundary value 2^53-1 should be convertible"
            );

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
        // Test values that would lose precision (only on 64-bit platforms where usize can exceed 2^53-1)
        const MAX_PRECISE_USIZE_IN_F64: u64 = (1_u64 << 53) - 1;

        if std::mem::size_of::<usize>() >= 8 {
            // On 64-bit platforms, test values that would lose precision
            let precision_loss_values = [
                usize::try_from(MAX_PRECISE_USIZE_IN_F64 + 1).unwrap_or(usize::MAX), // Just over 2^53-1
                usize::try_from(MAX_PRECISE_USIZE_IN_F64 + 100).unwrap_or(usize::MAX), // Well over 2^53-1
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
            // On 32-bit platforms, usize cannot exceed 2^53-1, so all values should succeed
            println!("Skipping precision loss test on 32-bit platform");
        }
    }

    #[test]
    fn test_safe_usize_to_scalar_error_message_format() {
        // Test that error messages are properly formatted
        const MAX_PRECISE_USIZE_IN_F64: u64 = (1_u64 << 53) - 1;

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
    fn test_safe_usize_to_scalar_f32_precision_limit() {
        // Test that f32 precision limit is correctly detected
        // f32 has 24 mantissa bits, so max exact integer is 2^24 - 1 = 16,777,215
        const F32_MAX_EXACT_INT: usize = 16_777_215;
        const F32_FIRST_INEXACT_INT: usize = 16_777_216;

        // This should succeed
        let result_within: Result<f32, _> = safe_usize_to_scalar(F32_MAX_EXACT_INT);
        assert!(
            result_within.is_ok(),
            "Value {F32_MAX_EXACT_INT} should be within f32 precision"
        );
        // Convert the expected value properly using num_traits::cast
        let expected_f32: f32 =
            cast(F32_MAX_EXACT_INT).expect("This value should be exactly representable in f32");
        assert_relative_eq!(result_within.unwrap(), expected_f32, epsilon = 1e-6);

        // This should fail precision check
        let result_beyond: Result<f32, _> = safe_usize_to_scalar(F32_FIRST_INEXACT_INT);
        assert!(
            result_beyond.is_err(),
            "Value {F32_FIRST_INEXACT_INT} should exceed f32 precision"
        );

        // For f64, the same value should still work since f64 has higher precision
        let result_f64: Result<f64, _> = safe_usize_to_scalar(F32_FIRST_INEXACT_INT);
        assert!(
            result_f64.is_ok(),
            "Value {F32_FIRST_INEXACT_INT} should be within f64 precision"
        );
        // Convert the expected value properly using num_traits::cast
        let expected_f64: f64 =
            cast(F32_FIRST_INEXACT_INT).expect("This value should be exactly representable in f64");
        assert_relative_eq!(result_f64.unwrap(), expected_f64, epsilon = 1e-15);
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
        println!("2^53-1 = {}", (1_u64 << 53) - 1);

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

    #[test]
    fn test_safe_cast_to_f64_non_finite() {
        // Test NaN handling
        let result = safe_cast_to_f64(f64::NAN, 0);
        assert!(matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        ));

        // Test infinity handling
        let result = safe_cast_to_f64(f64::INFINITY, 1);
        assert!(matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        ));

        // Test negative infinity
        let result = safe_cast_to_f64(f64::NEG_INFINITY, 2);
        assert!(matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        ));

        // Test valid finite value
        let result = safe_cast_to_f64(42.5f64, 0);
        assert!(result.is_ok());
        assert_relative_eq!(result.unwrap(), 42.5);
    }

    #[test]
    fn test_safe_cast_from_f64_non_finite() {
        // Test NaN handling
        let result: Result<f64, _> = safe_cast_from_f64(f64::NAN, 0);
        assert!(matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        ));

        // Test infinity handling
        let result: Result<f64, _> = safe_cast_from_f64(f64::INFINITY, 1);
        assert!(matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        ));

        // Test negative infinity
        let result: Result<f64, _> = safe_cast_from_f64(f64::NEG_INFINITY, 2);
        assert!(matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        ));

        // Test valid conversion
        let result: Result<f32, _> = safe_cast_from_f64(42.5f64, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_safe_coords_conversion_with_non_finite() {
        // Test array with NaN
        let coords_nan = [1.0f32, f32::NAN, 3.0f32];
        let result = safe_coords_to_f64(coords_nan);
        assert!(matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        ));

        // Test array with infinity
        let coords_inf = [1.0f32, 2.0f32, f32::INFINITY];
        let result = safe_coords_to_f64(coords_inf);
        assert!(matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        ));

        // Test successful conversion
        let coords_valid = [1.0f32, 2.0f32, 3.0f32];
        let result = safe_coords_to_f64(coords_valid);
        assert!(result.is_ok());
        assert_relative_eq!(
            result.unwrap().as_slice(),
            [1.0f64, 2.0f64, 3.0f64].as_slice()
        );
    }

    #[test]
    fn test_safe_coords_from_f64_with_non_finite() {
        // Test array with NaN
        let coords_nan = [1.0f64, f64::NAN, 3.0f64];
        let result: Result<[f32; 3], _> = safe_coords_from_f64(coords_nan);
        assert!(matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        ));

        // Test array with infinity
        let coords_inf = [1.0f64, 2.0f64, f64::INFINITY];
        let result: Result<[f32; 3], _> = safe_coords_from_f64(coords_inf);
        assert!(matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        ));

        // Test successful conversion
        let coords_valid = [1.0f64, 2.0f64, 3.0f64];
        let result: Result<[f32; 3], _> = safe_coords_from_f64(coords_valid);
        assert!(result.is_ok());
    }

    #[test]
    fn test_safe_scalar_conversion_edge_cases() {
        // Test scalar to f64 with NaN
        let result = safe_scalar_to_f64(f64::NAN);
        assert!(matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        ));

        // Test scalar from f64 with NaN
        let result: Result<f32, _> = safe_scalar_from_f64(f64::NAN);
        assert!(matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        ));

        // Test successful conversions
        let result = safe_scalar_to_f64(42.5f32);
        assert!(result.is_ok());
        assert_relative_eq!(result.unwrap(), 42.5f64);

        let result: Result<f32, _> = safe_scalar_from_f64(42.5f64);
        assert!(result.is_ok());
    }

    #[test]
    fn test_safe_usize_to_scalar_precision_boundary_extended() {
        // Test values at the precision boundary (additional edge cases)
        const MAX_PRECISE: u64 = (1_u64 << 53) - 1;

        // Test maximum precise value (should work)
        let result: Result<f64, _> =
            safe_usize_to_scalar(usize::try_from(MAX_PRECISE).unwrap_or(usize::MAX));
        assert!(result.is_ok());

        // Test beyond precision limit (if usize is large enough)
        if std::mem::size_of::<usize>() >= 8 {
            // Only test on 64-bit platforms
            let large_value = usize::try_from(MAX_PRECISE + 1).unwrap_or(usize::MAX);
            let result: Result<f64, _> = safe_usize_to_scalar(large_value);
            assert!(matches!(
                result,
                Err(CoordinateConversionError::ConversionFailed { .. })
            ));
        }

        // Test normal values
        let result: Result<f64, _> = safe_usize_to_scalar(42_usize);
        assert!(result.is_ok());
        assert_relative_eq!(result.unwrap(), 42.0);

        // Test zero
        let result: Result<f64, _> = safe_usize_to_scalar(0_usize);
        assert!(result.is_ok());
        assert_relative_eq!(result.unwrap(), 0.0);
    }

    #[test]
    fn test_error_types_display() {
        // Test ValueConversionError display
        let value_error = ValueConversionError::ConversionFailed {
            value: "42".to_string(),
            from_type: "i32",
            to_type: "u32",
            details: "overflow".to_string(),
        };
        let display = format!("{value_error}");
        assert!(display.contains("Cannot convert 42 from i32 to u32"));

        // Test RandomPointGenerationError display
        let range_error = RandomPointGenerationError::InvalidRange {
            min: "10.0".to_string(),
            max: "5.0".to_string(),
        };
        let display = format!("{range_error}");
        assert!(display.contains("Invalid coordinate range"));

        let gen_error = RandomPointGenerationError::RandomGenerationFailed {
            min: "0.0".to_string(),
            max: "1.0".to_string(),
            details: "test error".to_string(),
        };
        let display = format!("{gen_error}");
        assert!(display.contains("Failed to generate random value"));

        let count_error = RandomPointGenerationError::InvalidPointCount { n_points: -5 };
        let display = format!("{count_error}");
        assert!(display.contains("Invalid number of points: -5"));

        // Test CircumcenterError variants
        let empty_error = CircumcenterError::EmptyPointSet;
        let display = format!("{empty_error}");
        assert!(display.contains("Empty point set"));

        let simplex_error = CircumcenterError::InvalidSimplex {
            actual: 2,
            expected: 3,
            dimension: 2,
        };
        let display = format!("{simplex_error}");
        assert!(display.contains("Points do not form a valid simplex"));
    }
}
