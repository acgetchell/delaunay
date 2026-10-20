//! Property tests for finite-coordinate checks and checked integer conversions.
//!
//! Finite checks must preserve input bits and retain typed, indexed failures.
//! Integer conversions must preserve values within their supported precision.

#[macro_use]
#[path = "common/proptest_config.rs"]
mod proptest_config;

use delaunay::prelude::geometry::{
    CoordinateConversionError, InvalidCoordinateValue, safe_usize_to_scalar, try_finite_coords,
    try_finite_scalar,
};
use num_traits::cast;
use proptest::{prelude::*, test_runner::TestCaseError};

/// Samples the full finite f64 bit-pattern space.
fn finite_f64() -> impl Strategy<Value = f64> {
    any::<u64>()
        .prop_map(f64::from_bits)
        .prop_filter("must be finite", |value| value.is_finite())
}

/// Pairs non-finite inputs with their expected typed diagnostic.
fn non_finite_f64() -> impl Strategy<Value = (f64, InvalidCoordinateValue)> {
    prop_oneof![
        Just((f64::NAN, InvalidCoordinateValue::Nan)),
        Just((f64::INFINITY, InvalidCoordinateValue::PositiveInfinity)),
        Just((f64::NEG_INFINITY, InvalidCoordinateValue::NegativeInfinity)),
    ]
}

/// Strategy for generating safe usize values (within f64 precision)
/// f64 can represent integers exactly up to and including 2^53.
fn safe_usize() -> impl Strategy<Value = usize> {
    0..=9_007_199_254_740_992_usize // 2^53
}

repo_proptest! {
    #[test]
    fn prop_finite_scalar_preserves_bits(value in finite_f64()) {
        let checked = try_finite_scalar(value)
            .map_err(|error| TestCaseError::fail(format!("finite scalar rejected: {error}")))?;
        prop_assert_eq!(checked.to_bits(), value.to_bits());
    }
}

#[test]
fn finite_scalar_classifies_non_finite_values() {
    for (value, expected) in [
        (f64::NAN, InvalidCoordinateValue::Nan),
        (
            f64::from_bits(0x7ff0_0000_0000_0001),
            InvalidCoordinateValue::Nan,
        ),
        (
            f64::from_bits(0xfff8_0000_0000_0001),
            InvalidCoordinateValue::Nan,
        ),
        (f64::INFINITY, InvalidCoordinateValue::PositiveInfinity),
        (f64::NEG_INFINITY, InvalidCoordinateValue::NegativeInfinity),
    ] {
        assert_eq!(
            try_finite_scalar(value),
            Err(CoordinateConversionError::NonFiniteValue {
                coordinate_index: 0,
                coordinate_value: expected,
            })
        );
    }
}

// =============================================================================
// USIZE CONVERSION TESTS (MACRO-GENERATED)
// =============================================================================

macro_rules! gen_safe_usize_to_scalar_succeeds {
    ($name:ident, $ty:ty, $strategy:ident) => {
        pastey::paste! {
            repo_proptest! {
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
            repo_proptest! {
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

gen_safe_usize_to_scalar_succeeds!(
    prop_safe_usize_to_f64_succeeds_small_values,
    f64,
    usize_range_f64
);

gen_safe_usize_exact_small_values!(prop_safe_usize_exact_for_small_values, f64, 1000_usize);

gen_safe_usize_preserves_const!(prop_safe_usize_preserves_zero, 0_usize, 0.0_f64, f64, 1e-15);

gen_safe_usize_preserves_const!(prop_safe_usize_preserves_one, 1_usize, 1.0_f64, f64, 1e-15);

// Monotonicity test retained for f64
repo_proptest! {
    /// Property: safe_usize_to_scalar is monotonic for safe values
    #[test]
    fn prop_safe_usize_monotonic(value1 in safe_usize(), value2 in safe_usize()) {
        let result1: Result<f64, _> = safe_usize_to_scalar(value1);
        let result2: Result<f64, _> = safe_usize_to_scalar(value2);

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
// FINITE COORDINATE CHECKS
// =============================================================================

macro_rules! finite_coords_properties {
    ($dim:literal) => {
        pastey::paste! {
            repo_proptest! {
                #[test]
                fn [<prop_finite_coords_preserve_bits_ $dim d>](
                    coords in prop::array::[<uniform $dim>](finite_f64())
                ) {
                    let checked = try_finite_coords(&coords)
                        .map_err(|error| TestCaseError::fail(format!("finite coordinates rejected: {error}")))?;
                    prop_assert_eq!(checked.map(f64::to_bits), coords.map(f64::to_bits));
                }

                #[test]
                fn [<prop_finite_coords_report_first_invalid_index_ $dim d>](
                    mut coords in prop::array::[<uniform $dim>](finite_f64()),
                    first_invalid in 0_usize..$dim,
                    (invalid, expected) in non_finite_f64(),
                ) {
                    // Later invalid coordinates must not replace the first diagnostic.
                    coords[first_invalid..].fill(f64::NAN);
                    coords[first_invalid] = invalid;
                    prop_assert_eq!(
                        try_finite_coords(&coords),
                        Err(CoordinateConversionError::NonFiniteValue {
                            coordinate_index: first_invalid,
                            coordinate_value: expected,
                        })
                    );
                }
            }
        }
    };
}

finite_coords_properties!(2);
finite_coords_properties!(3);
finite_coords_properties!(4);
finite_coords_properties!(5);

#[test]
fn finite_checks_preserve_signed_zero_and_extreme_bits() -> Result<(), CoordinateConversionError> {
    let coords = [
        0.0,
        -0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
    ];
    assert_eq!(
        try_finite_coords(&coords)?.map(f64::to_bits),
        coords.map(f64::to_bits)
    );
    for value in coords {
        assert_eq!(try_finite_scalar(value)?.to_bits(), value.to_bits());
    }
    Ok(())
}

#[test]
fn finite_coords_accept_empty_array() -> Result<(), CoordinateConversionError> {
    assert_eq!(try_finite_coords(&[])?, []);
    Ok(())
}
