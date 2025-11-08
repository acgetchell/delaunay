//! Property-based tests for safe conversion functions.
//!
//! This module uses proptest to verify the correctness and safety of coordinate
//! conversion functions, including:
//! - `safe_usize_to_scalar` - Convert usize to float with precision checking
//! - `safe_scalar_to_f64` - Convert coordinate scalar to f64
//! - `safe_scalar_from_f64` - Convert f64 to coordinate scalar
//! - `safe_coords_to_f64` - Convert coordinate arrays to f64
//! - `safe_coords_from_f64` - Convert f64 arrays to coordinate scalars
//!
//! Properties verified:
//! - Conversions succeed for values within safe ranges
//! - Conversions fail gracefully for overflow/precision loss
//! - Round-trip conversions preserve values where possible
//! - Non-finite values (NaN, Infinity) are properly rejected

#![allow(unused_imports)] // Imports used in test functions

use approx::assert_relative_eq;
use delaunay::geometry::util::{
    safe_coords_from_f64, safe_coords_to_f64, safe_scalar_from_f64, safe_scalar_to_f64,
    safe_usize_to_scalar,
};
use num_traits::cast;
use proptest::prelude::*;
// =============================================================================
// TEST CONFIGURATION
// =============================================================================

/// Strategy for generating finite f64 coordinates
fn finite_f64() -> impl Strategy<Value = f64> {
    (-1000.0..1000.0).prop_filter("must be finite", |x: &f64| x.is_finite())
}

/// Strategy for generating finite f32 coordinates  
fn finite_f32() -> impl Strategy<Value = f32> {
    (-1000.0..1000.0_f32).prop_filter("must be finite", |x: &f32| x.is_finite())
}

/// Strategy for generating non-finite f64 values (NaN, Infinity)
fn non_finite_f64() -> impl Strategy<Value = f64> {
    prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY),]
}

/// Strategy for generating safe usize values (within f64 precision)
/// f64 can represent integers exactly up to 2^53 - 1
fn safe_usize() -> impl Strategy<Value = usize> {
    0..=9_007_199_254_740_991_usize // 2^53 - 1
}

/// Strategy for generating safe usize values for f32 targets
/// f32 can represent integers exactly up to 2^24 - 1
fn safe_usize_for_f32() -> impl Strategy<Value = usize> {
    0..=16_777_215_usize // 2^24 - 1
}

// =============================================================================
// SCALAR CONVERSION TESTS
// =============================================================================

proptest! {
    /// Property: safe_scalar_to_f64 succeeds for finite f64 values
    #[test]
    fn prop_safe_scalar_to_f64_succeeds_for_finite(value in finite_f64()) {
        let result: Result<f64, _> = safe_scalar_to_f64(value);
        prop_assert!(result.is_ok(), "Conversion should succeed for finite f64: {}", value);
        prop_assert!((result.unwrap() - value).abs() < 1e-12, "Converted value should match original");
    }

    /// Property: safe_scalar_to_f64 rejects non-finite values
    #[test]
    fn prop_safe_scalar_to_f64_rejects_non_finite(value in non_finite_f64()) {
        let result: Result<f64, _> = safe_scalar_to_f64(value);
        prop_assert!(result.is_err(), "Conversion should fail for non-finite f64: {}", value);
    }

    /// Property: safe_scalar_from_f64 succeeds for finite values
    #[test]
    fn prop_safe_scalar_from_f64_succeeds_for_finite(value in finite_f64()) {
        let result: Result<f64, _> = safe_scalar_from_f64(value);
        prop_assert!(result.is_ok(), "Conversion should succeed for finite f64: {}", value);
        prop_assert!((result.unwrap() - value).abs() < 1e-10, "Converted value should match");
    }

    /// Property: safe_scalar_from_f64 rejects non-finite values
    #[test]
    fn prop_safe_scalar_from_f64_rejects_non_finite(value in non_finite_f64()) {
        let result: Result<f32, _> = safe_scalar_from_f64(value);
        prop_assert!(result.is_err(), "Conversion should fail for non-finite f64: {}", value);
    }

    /// Property: Round-trip f64 -> f64 preserves value exactly
    #[test]
    fn prop_round_trip_f64_f64(value in finite_f64()) {
        let converted: Result<f64, _> = safe_scalar_to_f64(value);
        prop_assert!(converted.is_ok());

        let back: Result<f64, _> = safe_scalar_from_f64(converted.unwrap());
        prop_assert!(back.is_ok());
        prop_assert!((back.unwrap() - value).abs() < 1e-12, "Round-trip should preserve f64 value");
    }

    /// Property: Round-trip f32 -> f64 -> f32 preserves value (within f32 precision)
    #[test]
    fn prop_round_trip_f32_f64_f32(value in finite_f32()) {
        let to_f64: Result<f64, _> = safe_scalar_to_f64(value);
        prop_assert!(to_f64.is_ok());

        let back: Result<f32, _> = safe_scalar_from_f64(to_f64.unwrap());
        prop_assert!(back.is_ok());

        let difference = (back.unwrap() - value).abs();
        prop_assert!(
            difference < 1e-6,
            "Round-trip should preserve f32 value: diff = {}",
            difference
        );
    }
}

// =============================================================================
// USIZE CONVERSION TESTS
// =============================================================================

proptest! {
    /// Property: safe_usize_to_scalar succeeds for small usize values (< 2^53)
    #[test]
    fn prop_safe_usize_to_f64_succeeds_small_values(value in safe_usize()) {
        let result: Result<f64, _> = safe_usize_to_scalar(value);
        prop_assert!(result.is_ok(), "Conversion should succeed for usize {} < 2^53", value);

        let converted = result.unwrap();
        let back: usize = cast(converted).expect("f64 should convert back to usize exactly");
        prop_assert_eq!(back, value, "Converted value should match original");
    }

    /// Property: safe_usize_to_scalar succeeds for small usize values targeting f32 (< 2^24)
    #[test]
    fn prop_safe_usize_to_f32_succeeds_small_values(value in safe_usize_for_f32()) {
        let result: Result<f32, _> = safe_usize_to_scalar(value);
        prop_assert!(result.is_ok(), "Conversion should succeed for usize {} < 2^24", value);

        let converted = result.unwrap();
        let back: usize = cast(converted).expect("f32 should convert back to usize exactly");
        prop_assert_eq!(back, value, "Converted f32 value should match original");
    }

    /// Property: safe_usize_to_scalar is exact for small values
    #[test]
    fn prop_safe_usize_exact_for_small_values(value in 0..=1000_usize) {
        let result: Result<f64, _> = safe_usize_to_scalar(value);
        prop_assert!(result.is_ok(), "Conversion should succeed for small usize: {}", value);

        let converted = result.unwrap();
        let back: usize = cast(converted).expect("f64 should convert back to usize exactly");
        prop_assert_eq!(back, value, "Small values should convert exactly");
    }


    /// Property: safe_usize_to_scalar is monotonic for safe values
    #[test]
    fn prop_safe_usize_monotonic(value1 in safe_usize_for_f32(), value2 in safe_usize_for_f32()) {
        let result1: Result<f32, _> = safe_usize_to_scalar(value1);
        let result2: Result<f32, _> = safe_usize_to_scalar(value2);

        prop_assert!(result1.is_ok() && result2.is_ok());

        let converted1 = result1.unwrap();
        let converted2 = result2.unwrap();

        match value1.cmp(&value2) {
            core::cmp::Ordering::Less => {
                prop_assert_eq!(converted1.partial_cmp(&converted2), Some(core::cmp::Ordering::Less), "Conversion should be monotonic");
            }
            core::cmp::Ordering::Equal => {
                prop_assert_eq!(converted1.partial_cmp(&converted2), Some(core::cmp::Ordering::Equal), "Equal inputs should give equal outputs");
            }
            core::cmp::Ordering::Greater => {
                prop_assert_eq!(converted1.partial_cmp(&converted2), Some(core::cmp::Ordering::Greater), "Conversion should be monotonic");
            }
        }
    }
}

/// Property: `safe_usize_to_scalar` preserves zero
#[test]
fn prop_safe_usize_preserves_zero() {
    let result: Result<f64, _> = safe_usize_to_scalar(0_usize);
    assert!(result.is_ok());
    assert!(
        (result.unwrap() - 0.0).abs() < 1e-15,
        "Zero should convert to 0.0"
    );
}

/// Property: `safe_usize_to_scalar` preserves one
#[test]
fn prop_safe_usize_preserves_one() {
    let result: Result<f64, _> = safe_usize_to_scalar(1_usize);
    assert!(result.is_ok());
    assert!(
        (result.unwrap() - 1.0).abs() < 1e-15,
        "One should convert to 1.0"
    );
}

// =============================================================================
// COORDINATE ARRAY CONVERSION TESTS
// =============================================================================

proptest! {
    /// Property: safe_coords_to_f64 succeeds for 2D finite coordinates
    #[test]
    fn prop_safe_coords_to_f64_2d(coords in prop::array::uniform2(finite_f32())) {
        let result = safe_coords_to_f64(coords);
        prop_assert!(result.is_ok(), "Conversion should succeed for finite 2D coords");

        let converted = result.unwrap();
        for i in 0..2 {
            let diff = (converted[i] - f64::from(coords[i])).abs();
            prop_assert!(
                diff < 1e-6,
                "Coordinate {} should match: {} vs {}",
                i, converted[i], coords[i]
            );
        }
    }

    /// Property: safe_coords_to_f64 succeeds for 3D finite coordinates
    #[test]
    fn prop_safe_coords_to_f64_3d(coords in prop::array::uniform3(finite_f32())) {
        let result = safe_coords_to_f64(coords);
        prop_assert!(result.is_ok(), "Conversion should succeed for finite 3D coords");

        let converted = result.unwrap();
        for i in 0..3 {
            let diff = (converted[i] - f64::from(coords[i])).abs();
            prop_assert!(
                diff < 1e-6,
                "Coordinate {} should match: {} vs {}",
                i, converted[i], coords[i]
            );
        }
    }

    /// Property: safe_coords_from_f64 succeeds for 2D finite coordinates
    #[test]
    fn prop_safe_coords_from_f64_2d(coords in prop::array::uniform2(finite_f64())) {
        let result: Result<[f64; 2], _> = safe_coords_from_f64(coords);
        prop_assert!(result.is_ok(), "Conversion should succeed for finite 2D coords");

        let converted = result.unwrap();
        for i in 0..2 {
            let diff = (converted[i] - coords[i]).abs();
            prop_assert!(
                diff < 1e-10,
                "Coordinate {} should match: {} vs {}",
                i, converted[i], coords[i]
            );
        }
    }

    /// Property: safe_coords_from_f64 succeeds for 3D finite coordinates
    #[test]
    fn prop_safe_coords_from_f64_3d(coords in prop::array::uniform3(finite_f64())) {
        let result: Result<[f64; 3], _> = safe_coords_from_f64(coords);
        prop_assert!(result.is_ok(), "Conversion should succeed for finite 3D coords");

        let converted = result.unwrap();
        for i in 0..3 {
            let diff = (converted[i] - coords[i]).abs();
            prop_assert!(
                diff < 1e-10,
                "Coordinate {} should match: {} vs {}",
                i, converted[i], coords[i]
            );
        }
    }

    /// Property: Round-trip coordinate conversion 2D (f32 -> f64 -> f32)
    #[test]
    fn prop_round_trip_coords_2d(coords in prop::array::uniform2(finite_f32())) {
        let to_f64 = safe_coords_to_f64(coords);
        prop_assert!(to_f64.is_ok());

        let back: Result<[f32; 2], _> = safe_coords_from_f64(to_f64.unwrap());
        prop_assert!(back.is_ok());

        let converted = back.unwrap();
        for i in 0..2 {
            let diff = (converted[i] - coords[i]).abs();
            prop_assert!(
                diff < 1e-6,
                "Round-trip coordinate {} should match: diff = {}",
                i, diff
            );
        }
    }

    /// Property: Round-trip coordinate conversion 3D (f32 -> f64 -> f32)
    #[test]
    fn prop_round_trip_coords_3d(coords in prop::array::uniform3(finite_f32())) {
        let to_f64 = safe_coords_to_f64(coords);
        prop_assert!(to_f64.is_ok());

        let back: Result<[f32; 3], _> = safe_coords_from_f64(to_f64.unwrap());
        prop_assert!(back.is_ok());

        let converted = back.unwrap();
        for i in 0..3 {
            let diff = (converted[i] - coords[i]).abs();
            prop_assert!(
                diff < 1e-6,
                "Round-trip coordinate {} should match: diff = {}",
                i, diff
            );
        }
    }

    /// Property: Round-trip coordinate conversion 4D (f64 -> f64 -> f64)
    #[test]
    fn prop_round_trip_coords_4d_exact(coords in prop::array::uniform4(finite_f64())) {
        let to_f64 = safe_coords_to_f64(coords);
        prop_assert!(to_f64.is_ok());

        let back: Result<[f64; 4], _> = safe_coords_from_f64(to_f64.unwrap());
        prop_assert!(back.is_ok());

        let converted = back.unwrap();
        for i in 0..4 {
            prop_assert!(
                (converted[i] - coords[i]).abs() < 1e-12,
                "Round-trip f64 coordinate {} should be exact (within tolerance)",
                i
            );
        }
    }

    /// Property: Coordinate array with one NaN is rejected
    #[test]
    fn prop_coords_reject_nan_3d(good_coord in finite_f64(), nan_index in 0..3_usize) {
        let mut coords = [good_coord; 3];
        coords[nan_index] = f64::NAN;

        let result: Result<[f64; 3], _> = safe_coords_from_f64(coords);
        prop_assert!(result.is_err(), "Array with NaN should be rejected");
    }

    /// Property: Coordinate array with one Infinity is rejected
    #[test]
    fn prop_coords_reject_infinity_3d(good_coord in finite_f64(), inf_index in 0..3_usize) {
        let mut coords = [good_coord; 3];
        coords[inf_index] = f64::INFINITY;

        let result: Result<[f64; 3], _> = safe_coords_from_f64(coords);
        prop_assert!(result.is_err(), "Array with Infinity should be rejected");
    }
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

/// Property: Zero coordinates convert correctly
#[test]
fn prop_zero_coords_convert_correctly_3d() {
    let zeros = [0.0_f64; 3];
    let result: Result<[f64; 3], _> = safe_coords_from_f64(zeros);
    assert!(result.is_ok());

    let converted = result.unwrap();
    for coord in &converted {
        assert_relative_eq!(*coord, 0.0, epsilon = 1e-15);
    }
}

proptest! {

    /// Property: Negative coordinates convert correctly
    #[test]
    fn prop_negative_coords_convert_3d(coords in prop::array::uniform3(finite_f64())) {
        let negative_coords = coords.map(|x| -x.abs());
        let result: Result<[f64; 3], _> = safe_coords_from_f64(negative_coords);
        prop_assert!(result.is_ok());

        let converted = result.unwrap();
        for (i, &coord) in converted.iter().enumerate() {
            prop_assert!(
                coord.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater),
                "Negative coordinate {} should remain negative or zero",
                i
            );
        }
    }

    /// Property: Very small values (near zero) convert correctly
    #[test]
    fn prop_small_values_convert(value in -1e-100..1e-100_f64) {
        if !value.is_finite() {
            return Ok(());
        }

        let result: Result<f64, _> = safe_scalar_from_f64(value);
        prop_assert!(result.is_ok(), "Small finite value should convert");

        let converted = result.unwrap();
        let diff = (converted - value).abs();
        prop_assert!(
            diff < 1e-10 || (converted == 0.0 && value.abs() < 1e-100),
            "Small value conversion: {} vs {} (diff: {})",
            value, converted, diff
        );
    }
}
