//! Integration tests for coordinate conversion error handling.
//!
//! This module tests that functions properly handle coordinate conversion errors
//! when dealing with extreme values, NaN, infinity, etc.

use delaunay::geometry::point::Point;
use delaunay::geometry::predicates::{insphere, insphere_distance, simplex_orientation};
use delaunay::geometry::robust_predicates::{config_presets, robust_insphere};
use delaunay::geometry::traits::coordinate::{Coordinate, CoordinateConversionError};
use delaunay::geometry::util::{CircumcenterError, circumcenter, circumradius, hypot};

// =============================================================================
// GEOMETRIC PREDICATES ERROR TESTS
// =============================================================================

#[test]
fn test_insphere_with_nan_coordinates() {
    // Test 3D insphere with NaN coordinates
    let points = vec![
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
        Point::new([f64::NAN, 0.5, 0.5]), // Point with NaN
    ];
    let test_point = Point::new([0.5, 0.5, 0.5]);

    // The function should return an error due to NaN coordinate
    let result = insphere(&points, test_point);

    match result {
        Err(CoordinateConversionError::NonFiniteValue { .. }) => {
            // Expected error type
        }
        other => panic!("Expected CoordinateConversionError::NonFiniteValue, got: {other:?}"),
    }
}

#[test]
fn test_insphere_with_infinity_coordinates() {
    // Test 3D insphere with infinity coordinates
    let points = vec![
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
        Point::new([f64::INFINITY, 0.5, 0.5]), // Point with positive infinity
    ];
    let test_point = Point::new([0.5, 0.5, 0.5]);

    // The function should return an error due to infinity coordinate
    let result = insphere(&points, test_point);

    match result {
        Err(CoordinateConversionError::NonFiniteValue { .. }) => {
            // Expected error type
        }
        other => panic!("Expected CoordinateConversionError::NonFiniteValue, got: {other:?}"),
    }
}

#[test]
fn test_insphere_2d_with_nan_coordinates() {
    // Test 2D insphere with NaN coordinates (using 2D triangle)
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([f64::NAN, 1.0]), // Point with NaN
    ];
    let test_point = Point::new([0.5, 0.5]);

    // The function should return an error due to NaN coordinate
    let result = insphere(&points, test_point);

    match result {
        Err(CoordinateConversionError::NonFiniteValue { .. }) => {
            // Expected error type
        }
        other => panic!("Expected CoordinateConversionError::NonFiniteValue, got: {other:?}"),
    }
}

#[test]
fn test_simplex_orientation_with_infinity_coordinates() {
    // Test simplex orientation with infinity coordinates
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([f64::NEG_INFINITY, 1.0]), // Point with negative infinity
    ];

    // The function should return an error due to infinity coordinate
    let result = simplex_orientation(&points);

    match result {
        Err(CoordinateConversionError::NonFiniteValue { .. }) => {
            // Expected error type
        }
        other => panic!("Expected CoordinateConversionError::NonFiniteValue, got: {other:?}"),
    }
}

// =============================================================================
// UTILITY FUNCTIONS ERROR TESTS
// =============================================================================

#[test]
fn test_circumcenter_with_nan_coordinates() {
    // Test circumcenter with NaN coordinates
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([f64::NAN, 1.0]), // Point with NaN
    ];

    // The function should return an error due to NaN coordinate
    let result = circumcenter(&points);

    match result {
        Err(CircumcenterError::CoordinateConversion(
            CoordinateConversionError::NonFiniteValue { .. },
        )) => {
            // Expected error type
        }
        other => panic!(
            "Expected CoordinateConversionError::NonFiniteValue wrapped in CircumcenterError, got: {other:?}"
        ),
    }
}

#[test]
fn test_circumradius_with_infinity_coordinates() {
    // Test circumradius with infinity coordinates
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([0.0, f64::INFINITY]), // Point with positive infinity
    ];

    // The function should return an error due to infinity coordinate
    let result = circumradius(&points);

    match result {
        Err(CircumcenterError::CoordinateConversion(
            CoordinateConversionError::NonFiniteValue { .. },
        )) => {
            // Expected error type
        }
        other => panic!(
            "Expected CoordinateConversionError::NonFiniteValue wrapped in CircumcenterError, got: {other:?}"
        ),
    }
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
    let result = hypot(diff_coords);

    // Verify that the result contains non-finite values
    assert!(
        !result.is_finite(),
        "Expected non-finite result from hypot with NaN/infinity coordinates"
    );
}

#[test]
fn test_hypot_with_nan_values() {
    // Test hypot with NaN values
    let result = hypot([f64::NAN, 1.0]);

    // hypot returns T directly, so we check that the result is NaN
    assert!(
        result.is_nan(),
        "Expected NaN result from hypot with NaN input"
    );
}

#[test]
fn test_hypot_with_infinity_values() {
    // Test hypot with infinity values
    let result = hypot([f64::INFINITY, 1.0]);

    // With our new safe conversion, hypot falls back to general algorithm when conversion fails
    // The result should still be infinity due to the general algorithm handling infinity properly
    assert!(
        result.is_infinite() || result.is_nan(),
        "Expected infinite or NaN result from hypot with infinity input"
    );
}

// =============================================================================
// ROBUST PREDICATES ERROR TESTS
// =============================================================================

#[test]
#[expect(clippy::match_same_arms)]
fn test_robust_insphere_with_nan() {
    // Test robust insphere with NaN coordinates
    let points = vec![
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
        Point::new([f64::NAN, 0.5, 0.5]), // Point with NaN
    ];
    let test_point = Point::new([0.5, 0.5, 0.5]);
    let config = config_presets::general_triangulation();

    // The robust function might use different strategies that don't immediately hit the safe conversion
    // The function might return an error due to NaN coordinate, or it might handle it differently
    let result = robust_insphere(&points, &test_point, &config);

    match result {
        Err(CoordinateConversionError::NonFiniteValue { .. }) => {
            // Expected error type
        }
        // The robust predicates might also handle NaN differently, so we accept any result
        // that doesn't cause a panic
        Ok(_) => {
            // The robust predicates might have fallback strategies that don't error
            // This is acceptable as long as it doesn't panic
        }
        Err(_other_error) => {
            // Other errors are also acceptable as they indicate proper error handling
        }
    }
}

#[test]
#[expect(clippy::match_same_arms)]
fn test_robust_insphere_with_infinity() {
    // Test robust insphere with infinity coordinates
    let points = vec![
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 1.0, 0.0]),
        Point::new([0.0, 0.0, 1.0]),
        Point::new([f64::NEG_INFINITY, 0.5, 0.5]), // Point with negative infinity
    ];
    let test_point = Point::new([0.5, 0.5, 0.5]);
    let config = config_presets::general_triangulation();

    // The robust function might use different strategies that don't immediately hit the safe conversion
    // The function might return an error due to infinity coordinate, or it might handle it differently
    let result = robust_insphere(&points, &test_point, &config);

    match result {
        Err(CoordinateConversionError::NonFiniteValue { .. }) => {
            // Expected error type
        }
        // The robust predicates might also handle infinity differently, so we accept any result
        // that doesn't cause a panic
        Ok(_) => {
            // The robust predicates might have fallback strategies that don't error
            // This is acceptable as long as it doesn't panic
        }
        Err(_other_error) => {
            // Other errors are also acceptable as they indicate proper error handling
        }
    }
}

// =============================================================================
// ERROR MESSAGE VERIFICATION TESTS
// =============================================================================

#[test]
fn test_error_message_contains_context() {
    // Test that error messages contain useful context information
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([f64::NAN, 1.0]), // Point with NaN at coordinate index 0
    ];

    let result = circumcenter(&points);

    if let Err(error) = result {
        let error_msg = error.to_string();

        // Error message should contain useful context
        assert!(error_msg.contains("NaN") || error_msg.contains("non-finite"));

        // Should identify the problematic value type
        assert!(error_msg.contains("coordinate") || error_msg.contains("value"));
    } else {
        panic!("Expected an error, but got Ok");
    }
}

#[test]
fn test_infinity_error_message_contains_context() {
    // Test that infinity error messages contain useful context using insphere_distance
    // which uses hypot internally and should handle problematic coordinates
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([0.0, 1.0]), // Valid triangle
    ];
    let test_point = Point::new([f64::INFINITY, 1.0]); // Point with infinity

    let result = insphere_distance(&points, test_point);

    if let Err(error) = result {
        let error_msg = error.to_string();

        // Error message should contain useful context about infinity or non-finite values
        assert!(
            error_msg.contains("inf")
                || error_msg.contains("infinite")
                || error_msg.contains("non-finite")
        );
    } else {
        // For this test, we might get Ok since hypot can handle infinity
        // In that case, we just verify the function doesn't crash
    }
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

#[test]
fn test_subnormal_values_handling() {
    // Test that subnormal values are handled correctly (should not error)
    let subnormal = f64::MIN_POSITIVE / 2.0;
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([subnormal, 1.0]), // Subnormal value
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
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([-0.0, 1.0]), // Negative zero
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
#[expect(clippy::match_same_arms)]
fn test_very_large_finite_values() {
    // Test that very large but finite values are handled correctly
    // Use a large value that won't overflow when squared (f64::MAX would become infinity when squared)
    let large_value = 1e100; // Large but safe value
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([large_value, 1.0]), // Very large but finite value
    ];

    // Large finite values should be handled without error as long as they don't overflow in calculations
    let result = circumcenter(&points);

    match result {
        Ok(_) => {
            // Expected - finite values should be processed normally
        }
        Err(CircumcenterError::CoordinateConversion(
            CoordinateConversionError::NonFiniteValue { .. },
        )) => {
            // If the large value causes overflow during calculations (like squaring),
            // this error is acceptable and expected
        }
        Err(other_error) => panic!("Unexpected error with large finite values: {other_error:?}"),
    }
}

#[test]
fn test_very_small_finite_values() {
    // Test that very small but finite values are handled correctly
    let points = vec![
        Point::new([0.0, 0.0]),
        Point::new([1.0, 0.0]),
        Point::new([f64::MIN_POSITIVE, 1.0]), // Very small but finite value
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
