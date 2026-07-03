# Diagnostics

This document covers the crate's diagnostic surfaces: structured validation
reports, construction and repair telemetry, opt-in debug helpers, and runtime
debug switches.

Keep this separate from [`property_testing_summary.md`](property_testing_summary.md).
Property testing explains which randomized invariants the repository checks;
diagnostics explains what evidence to collect when a construction, validation,
repair, or test failure needs investigation.

## Diagnostic Surfaces

The crate exposes two kinds of diagnostics.

Always available:

- `validate()` and `validation_report()` for cumulative Levels 1-5 validation.
- Typed construction, insertion, validation, topology, and repair errors.
- Repair diagnostics attached to non-convergence and repair-neighbor failures.
- Construction statistics and telemetry through
  `delaunay::prelude::diagnostics`.

Feature-gated with `diagnostics`:

- `delaunay::prelude::diagnostics::delaunay_violation_report`.
- `DelaunayViolationReport` and `DelaunayViolationDetail`.
- `debug_print_first_delaunay_violation`.
- `verify_conflict_region_completeness`.
- Extra test/debug tracing in selected integration tests and debug harnesses.

The `diagnostics` feature does not change triangulation construction,
validation, repair, or query semantics. It only compiles in additional
inspection tools.

Examples that derive `thiserror::Error` assume the example crate includes
`thiserror`; run `cargo add thiserror` alongside `delaunay` when copying those
snippets into an application.

## Commands

Use the repository recipe for the standard diagnostics harness:

```bash
just test-diagnostics
```

Run the diagnostics example:

```bash
cargo run --release --features diagnostics --example diagnostics
```

Run a diagnostics-enabled test directly:

```bash
cargo test --test circumsphere_debug_tools --features diagnostics -- --nocapture
```

In downstream application code:

```toml
[dependencies]
delaunay = { version = "...", features = ["diagnostics"] }
```

## Validation Reports

For most validation work, start with the always-available APIs:

```rust
use delaunay::prelude::construction::{DelaunayResult, DelaunayTriangulationBuilder, vertex};

fn main() -> DelaunayResult<()> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    assert!(dt.validate().is_ok());
    let report = dt.validation_report();
    assert!(report.is_ok());
    Ok(())
}
```

Use `validate()` when a cumulative pass/fail result is enough. Use
`validation_report()` when you need all violated invariants across the stack
instead of the first error.

Layer-local diagnostics follow a standard naming pattern:

- `is_valid()` for unambiguous element/TDS owners, and `is_valid_*` for
  higher-level owners with multiple validation layers.
- `*_diagnostic`: first actionable repair/retry diagnostic for that layer.
- `*_report`: all checkable layer-local failures.

For Level 4 embedding failures specifically, use
`dt.as_triangulation().embedding_diagnostic()` for the first repair-oriented
failure and `dt.as_triangulation().embedding_report()` for all checkable
embedding failures. These report invalid simplices or simplex pairs with
simplex keys, simplex UUIDs, offending vertex keys, and offending vertex UUIDs.
That key-oriented payload is the intended starting point for explicit rollback,
vertex deletion, or future repair workflows; the report itself is pure and does
not mutate the triangulation.

## Construction Telemetry

Construction statistics expose aggregate insertion and repair behavior for batch
construction workflows. Telemetry is intentionally coarse enough to be useful in
release-mode characterization without turning every insertion into a trace.

Useful fields include:

- skipped duplicate and degeneracy counts
- representative skipped-vertex samples
- insertion, locate, conflict-region, and cavity counters
- local repair timings and slow repair samples
- final topology and Delaunay validation timings

For large-scale reproducible diagnostics, prefer the documented debug recipes in
[`benches/README.md`](../benches/README.md) and
[`docs/dev/debug_env_vars.md`](dev/debug_env_vars.md).

## Delaunay Violation Reports

Use `delaunay_violation_report` when you want key-based data about Level 5
empty-circumsphere violations:

```rust
use delaunay::prelude::diagnostics::delaunay_violation_report;
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError, vertex,
};
use delaunay::prelude::geometry::CoordinateConversionError;
use delaunay::prelude::DelaunayValidationError;

#[derive(Debug, thiserror::Error)]
enum DiagnosticsExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Coordinate(#[from] CoordinateConversionError),
    #[error(transparent)]
    Validation(#[from] DelaunayValidationError),
}

fn main() -> Result<(), DiagnosticsExampleError> {
    let vertices = vec![
        vertex![0.0, 0.0, 0.0]?,
        vertex![1.0, 0.0, 0.0]?,
        vertex![0.0, 1.0, 0.0]?,
        vertex![0.0, 0.0, 1.0]?,
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices).build()?;

    let report = delaunay_violation_report(dt.tds(), None)?;
    assert!(report.is_valid());
    Ok(())
}
```

Reports store `SimplexKey` and `VertexKey` values rather than copying every
coordinate. This keeps diagnostics compact and lets callers recover coordinates,
UUIDs, or attached data from the original `Tds`.

Useful fields:

- `number_of_vertices`, `number_of_simplices`: size of the inspected TDS.
- `checked_simplices`: number of requested simplices considered by the scan.
- `violating_simplices`: all simplices that violate the Delaunay property.
- `violation_details`: per-violation repair seeds with the simplex vertices,
  neighbor slots, and one offending external vertex when identified.
- `first_violation()`: borrowed view of the first `violation_details` entry,
  avoiding a duplicate stored detail that could drift from the aggregate report.

## Tracing Output

The logging helper is useful when a failure is easier to inspect as a trace:

```rust
use delaunay::prelude::diagnostics::debug_print_first_delaunay_violation;

debug_print_first_delaunay_violation(dt.tds(), None);
```

Install a `tracing` subscriber in tests or applications to see output. In tests,
`-- --nocapture` is often useful.

## Conflict Region Cross-Checks

`verify_conflict_region_completeness` is a deliberately expensive brute-force
check. It compares the conflict region discovered by local traversal against a
full scan of all simplices. The insertion path can invoke this check when both are
true:

- the crate was compiled with `diagnostics`
- `DELAUNAY_DEBUG_CONFLICT_VERIFY=1` is set

Use it when investigating missed conflict simplices, broken neighbor traversal, or
cavity construction failures.

## Debug Environment Variables

Runtime debug switches are documented in
[`docs/dev/debug_env_vars.md`](dev/debug_env_vars.md). The most relevant ones
for diagnostics work are:

- `DELAUNAY_DEBUG_CONFLICT_VERIFY`: enable conflict-region completeness checks.
- `DELAUNAY_DEBUG_CAVITY`: trace cavity boundary extraction and filling.
- `DELAUNAY_DEBUG_NEIGHBORS`: trace neighbor wiring checks.
- `DELAUNAY_DEBUG_RIDGE_LINK`: trace ridge-link validation failures.
- `DELAUNAY_DEBUG_RETRYABLE_SKIP`: trace retryable conflict skips after rollback.
- `DELAUNAY_DUPLICATE_METRICS`: emit duplicate-detection grid metrics.

These switches should remain investigation tools. Normal user-facing validation
should use `validate()`, `validation_report()`, and typed errors.

## Property-Test Failures

When a property test fails, start with the reproduction seed and minimized input
from `proptest`. If the failure is geometric or topological, rerun the narrowed
test with `-- --nocapture`; then enable the `diagnostics` feature or one of the
debug environment variables above only when the ordinary typed error/report is
not enough.

See [`property_testing_summary.md`](property_testing_summary.md) for the property
test map and reproduction workflow.
