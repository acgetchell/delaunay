#![allow(dead_code, unused_imports)]

use num_traits::NumCast;

pub fn production_stdio() {
    // ruleid: delaunay.rust.no-stdio-diagnostics-in-src
    println!("debug output");

    // ruleid: delaunay.rust.no-stdio-diagnostics-in-src
    eprintln!("debug output");
}

pub fn nonfinite_defaults(value: Option<f64>) -> f64 {
    // ruleid: delaunay.rust.no-nonfinite-unwrap-defaults
    value.unwrap_or(f64::NAN)
}

pub fn silent_conversion_fallback(value: u64) -> f64 {
    // ruleid: delaunay.rust.no-silent-conversion-fallbacks
    NumCast::from(value).unwrap_or(0.0)
}

fn safe_f64(_value: u64) -> Option<f64> {
    Some(1.0)
}

pub fn safe_conversion_fallback(value: u64) -> f64 {
    // ruleid: delaunay.rust.no-silent-conversion-fallbacks
    safe_f64(value).unwrap_or(0.0)
}

pub fn public_unwrap_bypass(value: Option<u8>) -> u8 {
    // ruleid: delaunay.rust.no-production-unwrap-panic
    value.unwrap()
}

pub fn public_expect_bypass(value: Option<u8>) -> u8 {
    // ruleid: delaunay.rust.no-production-unwrap-panic
    value.expect("public APIs should return typed errors instead")
}

fn private_documented_invariant(value: Option<u8>) -> u8 {
    // ok: delaunay.rust.no-production-unwrap-panic
    value.expect("private helper documents an internal invariant")
}

pub fn env_gated_stdio() {
    // ruleid: delaunay.rust.no-env-gated-stdio-diagnostics
    if std::env::var_os("DELAUNAY_DEBUG").is_some() {
        // ruleid: delaunay.rust.no-stdio-diagnostics-in-src
        println!("debug output");
    }
}

// ruleid: delaunay.rust.no-clippy-allow-lints
#[allow(clippy::too_many_lines)]
fn clippy_allow_fixture() {}

// ruleid: delaunay.rust.expect-requires-reason
#[expect(clippy::too_many_lines)]
fn expect_without_reason_fixture() {}

// ok: delaunay.rust.expect-requires-reason
#[expect(clippy::too_many_lines, reason = "fixture documents the suppression")]
fn expect_with_reason_fixture() {}

// ruleid: delaunay.rust.no-box-dyn-error-in-src
type ProductionBoxedError = Box<dyn std::error::Error>;

trait ProductionDynamicErrors {
    // ruleid: delaunay.rust.no-box-dyn-error-in-src
    fn boxed_error_result(&self) -> Result<(), Box<dyn std::error::Error>>;

    // ruleid: delaunay.rust.no-box-dyn-error-in-src
    fn borrowed_error(&self, error: &dyn std::error::Error);

    // ruleid: delaunay.rust.no-box-dyn-error-in-src
    fn anyhow_error(&self, error: anyhow::Error);
}

/// # Ok::<(), Box<dyn std::error::Error>>(())
fn doctest_style_error_is_ignored() {}
