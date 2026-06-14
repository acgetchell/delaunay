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

- `validate()` and `validation_report()` for cumulative Levels 1-4 validation.
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
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
};

fn main() -> Result<(), DelaunayTriangulationConstructionError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

    assert!(dt.validate().is_ok());
    let report = dt.validation_report();
    assert!(report.is_valid());
    Ok(())
}
```

Use `validate()` when a pass/fail result is enough. Use `validation_report()`
when you need all violated invariants instead of the first error.

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

Use `delaunay_violation_report` when you want key-based data about Level 4
empty-circumsphere violations:

```rust
use delaunay::prelude::diagnostics::delaunay_violation_report;
use delaunay::prelude::construction::{
    DelaunayTriangulationBuilder, DelaunayTriangulationConstructionError,
};
use delaunay::prelude::DelaunayValidationError;

#[derive(Debug, thiserror::Error)]
enum DiagnosticsExampleError {
    #[error(transparent)]
    Construction(#[from] DelaunayTriangulationConstructionError),
    #[error(transparent)]
    Validation(#[from] DelaunayValidationError),
}

fn main() -> Result<(), DiagnosticsExampleError> {
    let vertices = vec![
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([1.0, 0.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 1.0, 0.0])?,
        delaunay::prelude::Vertex::<(), _>::try_new([0.0, 0.0, 1.0])?,
    ];
    let dt = DelaunayTriangulationBuilder::new(&vertices).build::<()>()?;

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
- `first_violation`: first violating simplex, its vertex keys, neighbor slots, and
  one offending external vertex when identified.

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
