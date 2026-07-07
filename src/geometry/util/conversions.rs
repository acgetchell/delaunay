//! Safe coordinate conversion functions between numeric types.
//!
//! This module provides functions for safely converting coordinate values
//! between different numeric types with proper error handling and finite-value
//! checking.

#![forbid(unsafe_code)]

use crate::geometry::traits::coordinate::{
    CoordinateConversionError, CoordinateConversionValue, F64_MANTISSA_DIGITS,
    InvalidCoordinateValue,
};
use num_traits::ToPrimitive;

/// Structured reason why a direct value conversion failed.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValueConversionFailureReason {
    /// The target coordinate type rejected the input value.
    #[error("target type rejected value")]
    TargetTypeRejected,
}

/// Errors that can occur during value type conversions.
///
/// # Examples
///
/// ```rust
/// use delaunay::prelude::geometry::{
///     CoordinateConversionValue, ValueConversionError, ValueConversionFailureReason,
/// };
///
/// let err = ValueConversionError::ConversionFailed {
///     value: CoordinateConversionValue::from_f64(1.0),
///     from_type: "f64",
///     to_type: "u32",
///     reason: ValueConversionFailureReason::TargetTypeRejected,
/// };
/// std::assert_matches!(err, ValueConversionError::ConversionFailed { .. });
/// ```
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum ValueConversionError {
    /// Failed to convert a value from one type to another.
    #[error("Cannot convert {value} from {from_type} to {to_type}: {reason}")]
    ConversionFailed {
        /// The value that failed to convert.
        value: CoordinateConversionValue,
        /// Source type name.
        from_type: &'static str,
        /// Target type name.
        to_type: &'static str,
        /// Structured reason for the conversion failure.
        reason: ValueConversionFailureReason,
    },
    /// A lower-level coordinate conversion failed while converting a derived value.
    #[error("Cannot convert {value} from {from_type} to {to_type}: {source}")]
    CoordinateConversion {
        /// The value that failed to convert.
        value: CoordinateConversionValue,
        /// Source type name.
        from_type: &'static str,
        /// Target type name.
        to_type: &'static str,
        /// Structured source conversion failure.
        #[source]
        source: Box<CoordinateConversionError>,
    },
}

/// Safely convert a coordinate value from type T to f64.
/// This function provides proper error handling for coordinate type conversions,
/// replacing silent fallback after `cast(x)` with explicit
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
pub(super) fn safe_cast_to_f64(
    value: f64,
    coordinate_index: usize,
) -> Result<f64, CoordinateConversionError> {
    // Check for non-finite values first
    if !value.is_finite() {
        return Err(CoordinateConversionError::NonFiniteValue {
            coordinate_index,
            coordinate_value: InvalidCoordinateValue::from_debug(&value),
        });
    }

    Ok(value)
}

/// Safely convert a coordinate value from f64 to type T.
///
/// This function provides proper error handling for coordinate type conversions,
/// replacing silent fallback after `cast(x)` with explicit
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
pub(super) fn safe_cast_from_f64(
    value: f64,
    coordinate_index: usize,
) -> Result<f64, CoordinateConversionError> {
    // Check for non-finite values first
    if !value.is_finite() {
        return Err(CoordinateConversionError::NonFiniteValue {
            coordinate_index,
            coordinate_value: InvalidCoordinateValue::from_debug(&value),
        });
    }

    Ok(value)
}

/// Safely convert an array of coordinates from type T to f64.
///
/// This function converts each coordinate in the array, providing detailed
/// error information if any conversion fails.
///
/// # Arguments
///
/// * `coords` - Reference to array of coordinates to convert
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
/// use delaunay::prelude::geometry::{CoordinateConversionError, safe_coords_to_f64};
///
/// # fn main() -> Result<(), CoordinateConversionError> {
/// // Convert f64 coordinates to f64 after validation
/// let coords = [1.5, 2.5, 3.5];
/// let coords_f64 = safe_coords_to_f64(&coords)?;
/// assert_eq!(coords_f64, [1.5, 2.5, 3.5]);
///
/// // Works with different array sizes - 4D example
/// let coords_4d = [1.0, 2.0, 3.0, 4.0];
/// let result_4d = safe_coords_to_f64(&coords_4d)?;
/// assert_eq!(result_4d, [1.0, 2.0, 3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
pub fn safe_coords_to_f64<const D: usize>(
    coords: &[f64; D],
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
/// * `coords` - Reference to array of f64 coordinates to convert
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
/// use delaunay::prelude::geometry::{CoordinateConversionError, safe_coords_from_f64};
///
/// # fn main() -> Result<(), CoordinateConversionError> {
/// // Convert f64 coordinates to validated f64 coordinates
/// let coords_f64 = [1.5, 2.5, 3.5];
/// let coords_checked: [f64; 3] = safe_coords_from_f64(&coords_f64)?;
/// assert_eq!(coords_checked, [1.5, 2.5, 3.5]);
///
/// // Works with different array sizes - 4D example
/// let coords_4d = [1.0, 2.0, 3.0, 4.0];
/// let result_4d: [f64; 4] = safe_coords_from_f64(&coords_4d)?;
/// assert_eq!(result_4d, [1.0, 2.0, 3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
pub fn safe_coords_from_f64<const D: usize>(
    coords: &[f64; D],
) -> Result<[f64; D], CoordinateConversionError> {
    let mut result = [0.0_f64; D];
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
/// use delaunay::prelude::geometry::{CoordinateConversionError, safe_scalar_to_f64};
///
/// # fn main() -> Result<(), CoordinateConversionError> {
/// let value = 42.5;
/// let value_f64 = safe_scalar_to_f64(value)?;
/// assert_eq!(value_f64, 42.5);
/// # Ok(())
/// # }
/// ```
pub fn safe_scalar_to_f64(value: f64) -> Result<f64, CoordinateConversionError> {
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
/// use delaunay::prelude::geometry::{CoordinateConversionError, safe_scalar_from_f64};
///
/// # fn main() -> Result<(), CoordinateConversionError> {
/// // Convert f64 to f64 (identity)
/// let value: f64 = safe_scalar_from_f64(42.0)?;
/// assert_eq!(value, 42.0);
/// # Ok(())
/// # }
/// ```
pub fn safe_scalar_from_f64(value: f64) -> Result<f64, CoordinateConversionError> {
    safe_cast_from_f64(value, 0)
}

/// Safely convert a `usize` value to the supported coordinate scalar type.
///
/// This function handles the conversion from `usize` to `f64` with proper
/// precision checking.
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
/// use delaunay::prelude::geometry::{CoordinateConversionError, safe_usize_to_scalar};
///
/// # fn main() -> Result<(), CoordinateConversionError> {
/// // Normal case - small usize values
/// let result: Result<f64, _> = safe_usize_to_scalar(42_usize);
/// assert_eq!(result?, 42.0);
///
/// // Large values that fit within f64 precision
/// let large_value = (1_u64 << 50) as usize; // 2^50, well within f64 precision
/// let result: Result<f64, _> = safe_usize_to_scalar(large_value);
/// assert!(result.is_ok());
///
/// // Values that would lose precision (if usize is large enough)
/// // This test may not trigger on all platforms depending on usize size
/// # Ok(())
/// # }
/// ```
///
/// # Precision Limits
///
/// The function uses the minimum mantissa bits between f64 and the target type:
/// - `f64` integers are exact up to and including 2^53 (9,007,199,254,740,992)
/// - For f64 targets: `usize` values larger than 2^53 will cause an error
/// - On 64-bit platforms, `usize` can be up to 64 bits, so f64 precision loss is possible
pub fn safe_usize_to_scalar(value: usize) -> Result<f64, CoordinateConversionError> {
    let max_precise_u128: u128 = 1u128 << F64_MANTISSA_DIGITS;

    // Use try_from to safely convert usize to u64 for comparison
    let value_u64 =
        u64::try_from(value).map_err(|_| CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: CoordinateConversionValue::from_usize(value),
            from_type: "usize",
            to_type: "f64",
        })?;

    if u128::from(value_u64) > max_precise_u128 {
        return Err(CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: CoordinateConversionValue::from_usize(value),
            from_type: "usize",
            to_type: "f64",
        });
    }

    value
        .to_f64()
        .ok_or_else(|| CoordinateConversionError::ConversionFailed {
            coordinate_index: 0,
            coordinate_value: CoordinateConversionValue::from_usize(value),
            from_type: "usize",
            to_type: "f64",
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::traits::coordinate::CoordinateConversionError;
    use approx::assert_relative_eq;
    use num_traits::cast;
    use std::assert_matches;

    #[test]
    fn test_safe_usize_to_scalar_basic_success() {
        // Test successful conversion of small usize values
        let small_value = 42_usize;
        let result: Result<f64, _> = safe_usize_to_scalar(small_value);
        assert!(result.is_ok());
        assert_relative_eq!(result.unwrap(), 42.0f64, epsilon = 1e-15);
    }

    #[test]
    fn test_safe_usize_to_scalar_within_f64_precision() {
        // Test values that are within f64 precision limits but relatively large
        let safe_large_values = [
            usize::try_from(1_u64 << 50).unwrap_or(usize::MAX), // 2^50, well within f64 precision
            usize::try_from(1_u64 << 51).unwrap_or(usize::MAX), // 2^51, still safe
            usize::try_from(1_u64 << 53).unwrap_or(usize::MAX), // 2^53 is still exactly representable
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
        // Test the exact boundary value: 2^53.
        const MAX_PRECISE_USIZE_IN_F64: u64 = 1_u64 << 53;

        // This should succeed (exactly at the boundary)
        if usize::try_from(MAX_PRECISE_USIZE_IN_F64).is_ok() {
            let boundary_value = usize::try_from(MAX_PRECISE_USIZE_IN_F64).unwrap();
            let result: Result<f64, _> = safe_usize_to_scalar(boundary_value);
            assert!(result.is_ok(), "Boundary value 2^53 should be convertible");

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
        // Test values that would lose precision (only on 64-bit platforms where usize can exceed 2^53)
        const MAX_PRECISE_USIZE_IN_F64: u64 = 1_u64 << 53;

        if std::mem::size_of::<usize>() >= 8 {
            // On 64-bit platforms, test values that would lose precision
            let precision_loss_values = [
                usize::try_from(MAX_PRECISE_USIZE_IN_F64 + 1).unwrap_or(usize::MAX), // Just over 2^53
                usize::try_from(MAX_PRECISE_USIZE_IN_F64 + 100).unwrap_or(usize::MAX), // Well over 2^53
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

                assert_matches!(
                    result,
                    Err(CoordinateConversionError::ConversionFailed {
                        coordinate_index: 0,
                        coordinate_value,
                        from_type: "usize",
                        to_type: "f64",
                    }) if coordinate_value == CoordinateConversionValue::from_usize(value)
                );
            }
        }
    }

    #[test]
    fn test_safe_usize_to_scalar_error_message_format() {
        // Test that error messages are properly formatted
        const MAX_PRECISE_USIZE_IN_F64: u64 = 1_u64 << 53;

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
        tracing::debug!(
            "Testing on platform with usize size: {} bytes",
            std::mem::size_of::<usize>()
        );
        tracing::debug!("usize::MAX = {}", usize::MAX);
        tracing::debug!("2^53 = {}", 1_u64 << 53);

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
        assert_matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        );

        // Test infinity handling
        let result = safe_cast_to_f64(f64::INFINITY, 1);
        assert_matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        );

        // Test negative infinity
        let result = safe_cast_to_f64(f64::NEG_INFINITY, 2);
        assert_matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        );

        // Test valid finite value
        let result = safe_cast_to_f64(42.5f64, 0);
        assert!(result.is_ok());
        assert_relative_eq!(result.unwrap(), 42.5);
    }

    #[test]
    fn test_safe_cast_from_f64_non_finite() {
        // Test NaN handling
        let result: Result<f64, _> = safe_cast_from_f64(f64::NAN, 0);
        assert_matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        );

        // Test infinity handling
        let result: Result<f64, _> = safe_cast_from_f64(f64::INFINITY, 1);
        assert_matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        );

        // Test negative infinity
        let result: Result<f64, _> = safe_cast_from_f64(f64::NEG_INFINITY, 2);
        assert_matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        );

        // Test valid conversion
        let result: Result<f64, _> = safe_cast_from_f64(42.5, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_safe_coords_conversion_with_non_finite() {
        // Test array with NaN
        let coords_nan = [1.0, f64::NAN, 3.0];
        let result = safe_coords_to_f64(&coords_nan);
        assert_matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        );

        // Test array with infinity
        let coords_inf = [1.0, 2.0, f64::INFINITY];
        let result = safe_coords_to_f64(&coords_inf);
        assert_matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        );

        // Test successful conversion
        let coords_valid = [1.0, 2.0, 3.0];
        let result = safe_coords_to_f64(&coords_valid);
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
        let result: Result<[f64; 3], _> = safe_coords_from_f64(&coords_nan);
        assert_matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        );

        // Test array with infinity
        let coords_inf = [1.0f64, 2.0f64, f64::INFINITY];
        let result: Result<[f64; 3], _> = safe_coords_from_f64(&coords_inf);
        assert_matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        );

        // Test successful conversion
        let coords_valid = [1.0f64, 2.0f64, 3.0f64];
        let result: Result<[f64; 3], _> = safe_coords_from_f64(&coords_valid);
        assert!(result.is_ok());
    }

    #[test]
    fn test_safe_scalar_conversion_edge_cases() {
        // Test scalar to f64 with NaN
        let result = safe_scalar_to_f64(f64::NAN);
        assert_matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        );

        // Test scalar from f64 with NaN
        let result: Result<f64, _> = safe_scalar_from_f64(f64::NAN);
        assert_matches!(
            result,
            Err(CoordinateConversionError::NonFiniteValue { .. })
        );

        // Test successful conversions
        let result = safe_scalar_to_f64(42.5);
        assert!(result.is_ok());
        assert_relative_eq!(result.unwrap(), 42.5f64);

        let result: Result<f64, _> = safe_scalar_from_f64(42.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_safe_usize_to_scalar_precision_boundary_extended() {
        // Test values at the precision boundary (additional edge cases)
        const MAX_PRECISE: u64 = 1_u64 << 53;

        // Test maximum precise value (should work)
        let result: Result<f64, _> =
            safe_usize_to_scalar(usize::try_from(MAX_PRECISE).unwrap_or(usize::MAX));
        assert!(result.is_ok());

        // Test beyond precision limit (if usize is large enough)
        if std::mem::size_of::<usize>() >= 8 {
            // Only test on 64-bit platforms
            let large_value = usize::try_from(MAX_PRECISE + 1).unwrap_or(usize::MAX);
            let result: Result<f64, _> = safe_usize_to_scalar(large_value);
            assert_matches!(
                result,
                Err(CoordinateConversionError::ConversionFailed { .. })
            );
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
    fn value_conversion_error_display_names_conversion() {
        let value_error = ValueConversionError::ConversionFailed {
            value: CoordinateConversionValue::from_usize(42),
            from_type: "i32",
            to_type: "u32",
            reason: ValueConversionFailureReason::TargetTypeRejected,
        };
        let display = format!("{value_error}");
        assert!(display.contains("Cannot convert 42 from i32 to u32"));
        assert!(display.contains("target type rejected value"));
    }
}
