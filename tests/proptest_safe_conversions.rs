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

use approx::assert_relative_eq;
use delaunay::prelude::geometry::*;
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
// SCALAR CONVERSION TESTS (MACRO-GENERATED)
// =============================================================================

// f64 input -> f64 output should match exactly
macro_rules! gen_safe_scalar_to_f64_ok_f64 {
    () => {
        proptest! {
            #[test]
            fn prop_safe_scalar_to_f64_succeeds_for_finite_f64(value in finite_f64()) {
                let result: Result<f64, _> = safe_scalar_to_f64(value);
                prop_assert!(result.is_ok(), "Conversion should succeed for finite f64");
                prop_assert!((result.unwrap() - value).abs() < 1e-12, "Converted value should match original");
            }
        }
    };
}

// f32 input -> f64 output should equal f64::from(input)
macro_rules! gen_safe_scalar_to_f64_ok_f32 {
    () => {
        proptest! {
            #[test]
            fn prop_safe_scalar_to_f64_succeeds_for_finite_f32(value in finite_f32()) {
                let result: Result<f64, _> = safe_scalar_to_f64(value);
                prop_assert!(result.is_ok(), "Conversion should succeed for finite f32");
                let expected = f64::from(value);
                prop_assert!((result.unwrap() - expected).abs() < 1e-12, "Converted value should match original");
            }
        }
    };
}

// f64 input -> f64 output
macro_rules! gen_safe_scalar_from_f64_ok_f64 {
    () => {
        proptest! {
            #[test]
            fn prop_safe_scalar_from_f64_succeeds_for_finite_f64(value in finite_f64()) {
                let result: Result<f64, _> = safe_scalar_from_f64(value);
                prop_assert!(result.is_ok(), "Conversion should succeed for finite f64");
                prop_assert!((result.unwrap() - value).abs() < 1e-12, "Converted value should match original");
            }
        }
    };
}

// f64 input -> f32 output (within tolerance)
macro_rules! gen_safe_scalar_from_f64_ok_f32 {
    () => {
        proptest! {
            #[test]
            fn prop_safe_scalar_from_f64_succeeds_for_finite_f32(value in finite_f64()) {
                let result: Result<f32, _> = safe_scalar_from_f64(value);
                prop_assert!(result.is_ok(), "Conversion should succeed for finite f64");
                let diff = (f64::from(result.unwrap()) - value).abs();
                prop_assert!(diff <= 1e-4, "Converted value should match within tolerance: diff={}", diff);
            }
        }
    };
}

// Round-trip scalar tests
macro_rules! gen_round_trip_scalar_f64_f64 {
    () => {
        proptest! {
            #[test]
            fn prop_round_trip_f64_f64(value in finite_f64()) {
                let converted: Result<f64, _> = safe_scalar_to_f64(value);
                prop_assert!(converted.is_ok());
                let back: Result<f64, _> = safe_scalar_from_f64(converted.unwrap());
                prop_assert!(back.is_ok());
                prop_assert!((back.unwrap() - value).abs() < 1e-12, "Round-trip should preserve f64 value");
            }
        }
    };
}

macro_rules! gen_round_trip_scalar_f32_f64_f32 {
    () => {
        proptest! {
            #[test]
            fn prop_round_trip_f32_f64_f32(value in finite_f32()) {
                let to_f64: Result<f64, _> = safe_scalar_to_f64(value);
                prop_assert!(to_f64.is_ok());
                let back: Result<f32, _> = safe_scalar_from_f64(to_f64.unwrap());
                prop_assert!(back.is_ok());
                let difference = (back.unwrap() - value).abs();
                prop_assert!(difference < 1e-6, "Round-trip should preserve f32 value: diff = {}", difference);
            }
        }
    };
}

// Non-finite rejection tests
proptest! {
    /// Property: safe_scalar_to_f64 rejects non-finite values
    #[test]
    fn prop_safe_scalar_to_f64_rejects_non_finite(value in non_finite_f64()) {
        let result: Result<f64, _> = safe_scalar_to_f64(value);
        prop_assert!(result.is_err(), "Conversion should fail for non-finite f64: {}", value);
    }

    /// Property: safe_scalar_from_f64 rejects non-finite values
    #[test]
    fn prop_safe_scalar_from_f64_rejects_non_finite(value in non_finite_f64()) {
        let result: Result<f32, _> = safe_scalar_from_f64(value);
        prop_assert!(result.is_err(), "Conversion should fail for non-finite f64: {}", value);
    }
}

// Invoke macros
gen_safe_scalar_to_f64_ok_f64!();
gen_safe_scalar_to_f64_ok_f32!();

gen_safe_scalar_from_f64_ok_f64!();
gen_safe_scalar_from_f64_ok_f32!();

gen_round_trip_scalar_f64_f64!();
gen_round_trip_scalar_f32_f64_f32!();

// =============================================================================
// USIZE CONVERSION TESTS (MACRO-GENERATED)
// =============================================================================

macro_rules! gen_safe_usize_to_scalar_succeeds {
    ($name:ident, $ty:ty, $strategy:ident) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<$name>](value in $strategy()) {
                    let result: Result<$ty, _> = safe_usize_to_scalar(value);
                    prop_assert!(result.is_ok(), "Conversion should succeed for usize {}", value);
                    let converted = result.unwrap();
                    let back: usize = cast(converted).expect("round-trip usize should be exact");
                    prop_assert_eq!(back, value, "Converted value should match original");
                }
            }
        }
    };
}

macro_rules! gen_safe_usize_exact_small_values {
    ($name:ident, $ty:ty, $upper:expr) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<$name>](value in 0..=$upper) {
                    let result: Result<$ty, _> = safe_usize_to_scalar(value);
                    prop_assert!(result.is_ok(), "Conversion should succeed for small usize: {}", value);
                    let converted = result.unwrap();
                    let back: usize = cast(converted).expect("round-trip usize should be exact");
                    prop_assert_eq!(back, value, "Small values should convert exactly");
                }
            }
        }
    };
}

macro_rules! gen_safe_usize_preserves_const {
    ($name:ident, $val:expr, $expected:expr, $ty:ty, $eps:expr) => {
        pastey::paste! {
            #[test]
            fn [<$name>]() {
                let result: Result<$ty, _> = safe_usize_to_scalar($val);
                assert!(result.is_ok());
                let v = result.unwrap();
                assert!((f64::from(v) - $expected).abs() < $eps);
            }
        }
    };
}

fn usize_range_f64() -> impl Strategy<Value = usize> {
    safe_usize()
}
fn usize_range_f32() -> impl Strategy<Value = usize> {
    safe_usize_for_f32()
}

gen_safe_usize_to_scalar_succeeds!(
    prop_safe_usize_to_f64_succeeds_small_values,
    f64,
    usize_range_f64
);

gen_safe_usize_to_scalar_succeeds!(
    prop_safe_usize_to_f32_succeeds_small_values,
    f32,
    usize_range_f32
);

gen_safe_usize_exact_small_values!(prop_safe_usize_exact_for_small_values, f64, 1000_usize);

gen_safe_usize_preserves_const!(prop_safe_usize_preserves_zero, 0_usize, 0.0_f64, f64, 1e-15);

gen_safe_usize_preserves_const!(prop_safe_usize_preserves_one, 1_usize, 1.0_f64, f64, 1e-15);

// Monotonicity test retained (f32)
proptest! {
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

// =============================================================================
// COORDINATE ARRAY CONVERSION TESTS
// =============================================================================

// Macros to generate repeated coordinate array tests across dimensions
macro_rules! gen_safe_coords_to_f64 {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_safe_coords_to_f64_ $dim d>](coords in prop::array::[<uniform $dim>](finite_f32())) {
                    let result = safe_coords_to_f64(&coords);
                    prop_assert!(result.is_ok(), "Conversion should succeed for finite {}D coords", $dim);
                    let converted = result.unwrap();
                    for i in 0..$dim {
                        let diff = (converted[i] - f64::from(coords[i])).abs();
                        prop_assert!(diff < 1e-6, "Coordinate {} should match: {} vs {}", i, converted[i], coords[i]);
                    }
                }
            }
        }
    };
}

macro_rules! gen_safe_coords_from_f64 {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_safe_coords_from_f64_ $dim d>](coords in prop::array::[<uniform $dim>](finite_f64())) {
                    let result: Result<[f64; $dim], _> = safe_coords_from_f64(&coords);
                    prop_assert!(result.is_ok(), "Conversion should succeed for finite {}D coords", $dim);
                    let converted = result.unwrap();
                    for i in 0..$dim {
                        let diff = (converted[i] - coords[i]).abs();
                        prop_assert!(diff < 1e-10, "Coordinate {} should match: {} vs {}", i, converted[i], coords[i]);
                    }
                }
            }
        }
    };
}

macro_rules! gen_round_trip_coords_f32 {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_round_trip_coords_ $dim d>](coords in prop::array::[<uniform $dim>](finite_f32())) {
                    let to_f64 = safe_coords_to_f64(&coords);
                    prop_assert!(to_f64.is_ok());
                    let back: Result<[f32; $dim], _> = safe_coords_from_f64(&to_f64.unwrap());
                    prop_assert!(back.is_ok());
                    let converted = back.unwrap();
                    for i in 0..$dim {
                        let diff = (converted[i] - coords[i]).abs();
                        prop_assert!(diff < 1e-6, "Round-trip coordinate {} should match: diff = {}", i, diff);
                    }
                }
            }
        }
    };
}

macro_rules! gen_round_trip_coords_f64_exact {
    ($dim:literal) => {
        pastey::paste! {
            proptest! {
                #[test]
                fn [<prop_round_trip_coords_ $dim d _exact>](coords in prop::array::[<uniform $dim>](finite_f64())) {
                    let to_f64 = safe_coords_to_f64(&coords);
                    prop_assert!(to_f64.is_ok());
                    let back: Result<[f64; $dim], _> = safe_coords_from_f64(&to_f64.unwrap());
                    prop_assert!(back.is_ok());
                    let converted = back.unwrap();
                    for i in 0..$dim {
                        prop_assert!((converted[i] - coords[i]).abs() < 1e-12,
                            "Round-trip f64 coordinate {} should be exact (within tolerance)", i);
                    }
                }
            }
        }
    };
}

// Instantiate macros for specific dimensions
gen_safe_coords_to_f64!(2);
gen_safe_coords_to_f64!(3);

gen_safe_coords_from_f64!(2);
gen_safe_coords_from_f64!(3);

gen_round_trip_coords_f32!(2);
gen_round_trip_coords_f32!(3);

gen_round_trip_coords_f64_exact!(4);

// Keep 3D rejection tests
proptest! {
    /// Property: Coordinate array with one NaN is rejected
    #[test]
    fn prop_coords_reject_nan_3d(good_coord in finite_f64(), nan_index in 0..3_usize) {
        let mut coords = [good_coord; 3];
        coords[nan_index] = f64::NAN;

        let result: Result<[f64; 3], _> = safe_coords_from_f64(&coords);
        prop_assert!(result.is_err(), "Array with NaN should be rejected");
    }

    /// Property: Coordinate array with one Infinity is rejected
    #[test]
    fn prop_coords_reject_infinity_3d(good_coord in finite_f64(), inf_index in 0..3_usize) {
        let mut coords = [good_coord; 3];
        coords[inf_index] = f64::INFINITY;

        let result: Result<[f64; 3], _> = safe_coords_from_f64(&coords);
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
    let result: Result<[f64; 3], _> = safe_coords_from_f64(&zeros);
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
        let result: Result<[f64; 3], _> = safe_coords_from_f64(&negative_coords);
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
