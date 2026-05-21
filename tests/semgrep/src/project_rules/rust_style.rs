#![allow(dead_code, unused_imports)]

use num_traits::NumCast;

// ruleid: delaunay.rust.prefer-prelude-imports-in-examples-benches
use delaunay::core::vertex::Vertex as DeepVertex;
// ok: delaunay.rust.prefer-prelude-imports-in-examples-benches
use delaunay::prelude::Vertex as PreludeVertex;

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
    // ruleid: delaunay.rust.no-silent-conversion-fallbacks, delaunay.rust.no-silent-conversion-fallbacks-in-public-samples
    NumCast::from(value).unwrap_or(0.0)
}

fn safe_f64(_value: u64) -> Option<f64> {
    Some(1.0)
}

pub fn safe_conversion_fallback(value: u64) -> f64 {
    // ruleid: delaunay.rust.no-silent-conversion-fallbacks, delaunay.rust.no-silent-conversion-fallbacks-in-public-samples
    safe_f64(value).unwrap_or(0.0)
}

pub fn public_sample_silent_conversion_fallback(value: u64) -> f64 {
    // ruleid: delaunay.rust.no-silent-conversion-fallbacks, delaunay.rust.no-silent-conversion-fallbacks-in-public-samples
    NumCast::from(value).unwrap_or(0.0)
}

pub fn partial_cmp_ordering_default(left: f64, right: f64) -> std::cmp::Ordering {
    // ruleid: delaunay.rust.no-partial-cmp-ordering-defaults
    left.partial_cmp(&right).unwrap_or(std::cmp::Ordering::Equal)
}

pub fn function_local_use_fixture() {
    // ruleid: delaunay.rust.no-function-local-use-in-src
    use std::cmp::Ordering;

    let _ordering = Ordering::Equal;
}

pub fn deep_crate_path_fixture() {
    // ruleid: delaunay.rust.no-deep-crate-paths-in-functions
    let _buffer = crate::core::collections::SimplexKeyBuffer::new();
}

pub fn public_unwrap_bypass(value: Option<u8>) -> u8 {
    // ruleid: delaunay.rust.no-production-unwrap-panic, delaunay.rust.no-public-surface-unwrap-panic
    value.unwrap()
}

pub fn public_expect_bypass(value: Option<u8>) -> u8 {
    // ruleid: delaunay.rust.no-production-unwrap-panic, delaunay.rust.no-public-surface-unwrap-panic
    value.expect("public APIs should return typed errors instead")
}

pub fn public_panic_bypass() {
    // ruleid: delaunay.rust.no-production-unwrap-panic, delaunay.rust.no-public-surface-unwrap-panic
    panic!("public APIs should return typed errors instead");
}

fn private_documented_invariant(value: Option<u8>) -> u8 {
    // ok: delaunay.rust.no-production-unwrap-panic
    // ruleid: delaunay.rust.no-public-surface-unwrap-panic
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

// ruleid: delaunay.rust.no-ignored-tests
#[ignore = "Slow (>10s); use the slow-tests feature instead"]
fn slow_ignore_fixture() {}

// ok: delaunay.rust.no-ignored-tests
#[cfg(feature = "slow-tests")]
fn slow_cfg_fixture() {}

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

// ruleid: delaunay.rust.public-error-enums-non-exhaustive
pub enum PublicFixtureError {
    Invalid,
}

// ok: delaunay.rust.public-error-enums-non-exhaustive
#[non_exhaustive]
pub enum PublicNonExhaustiveFixtureError {
    Invalid,
}

// ok: delaunay.rust.public-error-enums-non-exhaustive
enum PrivateFixtureError {
    Invalid,
}

/// # Ok::<(), Box<dyn std::error::Error>>(())
fn doctest_style_error_is_ignored() {}

/// ```rust
/// // ruleid: delaunay.rust.prefer-prelude-imports-in-delaunay-doctests
/// use delaunay::flips::BistellarFlips;
/// // ok: delaunay.rust.prefer-prelude-imports-in-delaunay-doctests
/// use delaunay::prelude::DelaunayTriangulation;
/// // ok: delaunay.rust.prefer-prelude-imports-in-delaunay-doctests
/// # use delaunay::prelude::DelaunayTriangulation as HiddenPreludeImport;
/// // ruleid: delaunay.rust.prefer-prelude-imports-in-delaunay-doctests
/// # use delaunay::flips::BistellarFlips as HiddenDeepImport;
/// ```
fn triangulation_doctest_deep_import_fixture() {}
