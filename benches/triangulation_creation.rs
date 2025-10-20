//! DEPRECATED: This benchmark is deprecated and will be removed in the next release.
//!
//! # Migration Guide
//!
//! **For CI/regression testing:** Use `ci_performance_suite.rs`
//! ```bash
//! cargo bench --bench ci_performance_suite
//! # or
//! just bench
//! ```
//!
//! **For Phase 4 `SlotMap` evaluation:** Use `large_scale_performance.rs`
//! ```bash
//! cargo bench --bench large_scale_performance
//! ```
//!
//! See `benches/README.md` for detailed benchmark selection guidance.
//!
//! # Deprecation Rationale
//!
//! This benchmark is redundant with `ci_performance_suite.rs` and provides no unique
//! functionality. The CI suite is the canonical source for triangulation creation
//! benchmarks and is used by all CI workflows for baseline generation and regression testing.

#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};

/// Deprecated benchmark - prints migration notice
fn deprecated_notice(_c: &mut Criterion) {
    eprintln!();
    eprintln!("==============================================================================");
    eprintln!("  ⚠️  WARNING: triangulation_creation benchmark is DEPRECATED");
    eprintln!("==============================================================================");
    eprintln!();
    eprintln!("This benchmark will be removed in the next release.");
    eprintln!();
    eprintln!("Migration guide:");
    eprintln!();
    eprintln!("  For CI/regression testing:");
    eprintln!("    cargo bench --bench ci_performance_suite");
    eprintln!("    # or");
    eprintln!("    just bench");
    eprintln!();
    eprintln!("  For Phase 4 SlotMap evaluation:");
    eprintln!("    cargo bench --bench large_scale_performance");
    eprintln!();
    eprintln!("See benches/README.md for detailed benchmark selection guidance.");
    eprintln!();
    eprintln!("==============================================================================");
    eprintln!();
}

criterion_group!(benches, deprecated_notice);
criterion_main!(benches);
