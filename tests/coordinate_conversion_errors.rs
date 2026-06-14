//! Integration tests for coordinate conversion error handling.
//!
//! This module tests that functions properly handle coordinate conversion errors
//! when dealing with extreme values, NaN, infinity, etc.

use delaunay::prelude::geometry::*;

fn assert_invalid_coordinate<const D: usize>(
    coords: [f64; D],
    coordinate_index: usize,
    coordinate_value: InvalidCoordinateValue,
) {
    assert_eq!(
        Point::<D>::try_new(coords),
        Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value,
            dimension: D,
        })
    );
}

#[test]
fn test_point_try_new_rejects_non_finite_coordinates() {
    assert_invalid_coordinate([f64::NAN, 0.5, 0.5], 0, InvalidCoordinateValue::Nan);
    assert_invalid_coordinate(
        [f64::INFINITY, 0.5, 0.5],
        0,
        InvalidCoordinateValue::PositiveInfinity,
    );
    assert_invalid_coordinate([f64::NAN, 1.0], 0, InvalidCoordinateValue::Nan);
    assert_invalid_coordinate(
        [f64::NEG_INFINITY, 1.0],
        0,
        InvalidCoordinateValue::NegativeInfinity,
    );
    assert_invalid_coordinate(
        [0.0, f64::INFINITY],
        1,
        InvalidCoordinateValue::PositiveInfinity,
    );
}

#[test]
fn test_hypot_distance_with_mixed_problematic_coordinates() {
    // Test distance calculation using hypot with mixed NaN and infinity
    let point1_coords: [f64; 2] = [f64::NAN, 1.0];
    let point2_coords: [f64; 2] = [1.0, f64::INFINITY];

    // Calculate difference vector
    let diff_coords = [
        point1_coords[0] - point2_coords[0],
        point1_coords[1] - point2_coords[1],
    ];

    // The hypot function should handle problematic coordinates properly
    // Since hypot returns T directly (not a Result), we expect it to return NaN or infinity
    let result = hypot(&diff_coords);

    // Verify that the result contains non-finite values
    assert!(
        !result.is_finite(),
        "Expected non-finite result from hypot with NaN/infinity coordinates"
    );
}

#[test]
fn test_hypot_with_nan_values() {
    // Test hypot with NaN values
    let result = hypot(&[f64::NAN, 1.0]);

    // hypot returns T directly, so we check that the result is NaN
    assert!(
        result.is_nan(),
        "Expected NaN result from hypot with NaN input"
    );
}

#[test]
fn test_hypot_with_infinity_values() {
    // Test hypot with infinity values
    let result = hypot(&[f64::INFINITY, 1.0]);

    // With our new safe conversion, hypot falls back to general algorithm when conversion fails
    // The result should still be infinity due to the general algorithm handling infinity properly
    assert!(
        result.is_infinite() || result.is_nan(),
        "Expected infinite or NaN result from hypot with infinity input"
    );
}

// =============================================================================
// ERROR MESSAGE VERIFICATION TESTS
// =============================================================================

#[test]
fn test_error_message_contains_context() {
    // Test that error messages contain useful context information
    let Err(error) = Point::<2>::try_new([f64::NAN, 1.0]) else {
        panic!("Expected an error, but got Ok");
    };
    let error_msg = error.to_string();

    // Error message should contain useful context
    assert!(error_msg.contains("NaN") || error_msg.contains("non-finite"));

    // Should identify the problematic value type
    assert!(error_msg.contains("coordinate") || error_msg.contains("value"));
}

#[test]
fn test_infinity_error_message_contains_context() {
    let Err(error) = Point::<2>::try_new([f64::INFINITY, 1.0]) else {
        panic!("Expected an error, but got Ok");
    };
    let error_msg = error.to_string();

    // Error message should contain useful context about infinity or non-finite values
    assert!(
        error_msg.contains("inf")
            || error_msg.contains("infinite")
            || error_msg.contains("non-finite")
    );
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

#[test]
fn test_subnormal_values_handling() {
    // Test that subnormal values are handled correctly (should not error)
    let subnormal = f64::MIN_POSITIVE / 2.0;
    let points = vec![
        Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([subnormal, 1.0]).expect("finite point coordinates"), // Subnormal value
    ];

    // Subnormal values should be handled without error (they're finite)
    let result = circumcenter(&points);

    match result {
        Ok(_) => {
            // Expected - subnormal values should be processed normally
        }
        Err(error) => panic!("Subnormal values should not cause errors: {error:?}"),
    }
}

#[test]
fn test_zero_and_negative_zero() {
    // Test that positive and negative zero are handled correctly
    let points = vec![
        Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([-0.0, 1.0]).expect("finite point coordinates"), // Negative zero
    ];

    // Both positive and negative zero should be processed normally
    let result = circumcenter(&points);

    match result {
        Ok(_) => {
            // Expected - zero values should be processed normally
        }
        Err(error) => panic!("Zero values should not cause errors: {error:?}"),
    }
}

#[test]
fn test_very_large_finite_values() {
    // Test that very large but finite values are handled correctly
    // Use a large value that won't overflow when squared (f64::MAX would become infinity when squared)
    let large_value = 1e100; // Large but safe value
    let points = vec![
        Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([large_value, 1.0]).expect("finite point coordinates"), // Very large but finite value
    ];

    // Large finite values should be handled without error as long as they don't overflow in calculations
    let result = circumcenter(&points);

    match result {
        Ok(_)
        | Err(CircumcenterError::CoordinateConversion {
            source: CoordinateConversionError::NonFiniteValue { .. },
        }) => {}
        Err(other_error) => {
            panic!("Unexpected error with large finite values: {other_error:?}");
        }
    }
}

#[test]
fn test_very_small_finite_values() {
    // Test that very small but finite values are handled correctly
    let points = vec![
        Point::try_new([0.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([1.0, 0.0]).expect("finite point coordinates"),
        Point::try_new([f64::MIN_POSITIVE, 1.0]).expect("finite point coordinates"), // Very small but finite value
    ];

    // Small finite values should be handled without error
    let result = circumcenter(&points);

    match result {
        Ok(_) => {
            // Expected - finite values should be processed normally
        }
        Err(error) => panic!("Small finite values should not cause errors: {error:?}"),
    }
}
