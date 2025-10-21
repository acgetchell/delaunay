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
//! This benchmark (1,000 vertices per dimension) has been intentionally split into two
//! specialized benchmarks with different purposes and scales:
//!
//! - **`ci_performance_suite.rs`**: Small-scale regression testing (10-50 vertices)
//!   optimized for fast CI execution and baseline generation
//! - **`large_scale_performance.rs`**: Large-scale Phase 4 evaluation (1K-10K vertices)
//!   focused on `SlotMap` comparison with memory/validation/iteration metrics
//!
//! The original 1,000-vertex scale is no longer needed as a standalone benchmark:
//! - Fast regression detection is better served by smaller scales in CI suite
//! - Phase 4 evaluation requires larger scales with comprehensive metrics
//! - Maintaining three overlapping benchmarks creates unnecessary CI overhead
//!
//! Migration depends on your use case (see Migration Guide above).

#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};

/// Deprecated benchmark - prints migration notice
fn deprecated_notice(_c: &mut Criterion) {
    eprintln!();
    eprintln!("==============================================================================");
    eprintln!("  ⚠️  WARNING: triangulation_creation benchmark is DEPRECATED");
    eprintln!("==============================================================================");
    eprintln!();
    eprintln!("This benchmark (1,000 vertices) has been split into specialized benchmarks:");
    eprintln!();
    eprintln!("  - ci_performance_suite (10-50 vertices): Fast CI regression testing");
    eprintln!("  - large_scale_performance (1K-10K vertices): Phase 4 evaluation");
    eprintln!();
    eprintln!("Migration guide:");
    eprintln!();
    eprintln!("  For CI/regression testing (fast, small scale):");
    eprintln!("    cargo bench --bench ci_performance_suite");
    eprintln!("    # or");
    eprintln!("    just bench");
    eprintln!();
    eprintln!("  For Phase 4 SlotMap evaluation (large scale + metrics):");
    eprintln!("    cargo bench --bench large_scale_performance");
    eprintln!();
    eprintln!("  The original 1,000-vertex scale is not directly replicated.");
    eprintln!("  Choose based on your use case (see benches/README.md for guidance).");
    eprintln!();
    eprintln!("==============================================================================");
    eprintln!();
}

criterion_group!(benches, deprecated_notice);
criterion_main!(benches);
