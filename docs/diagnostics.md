# Diagnostics Feature

The `diagnostics` feature exposes opt-in helpers for investigating Delaunay and
topology failures without adding them to the default public API surface. It is
intended for tests, bug reports, local debugging, and advanced verification
workflows.

Enable it when you need detailed evidence about why construction, insertion, or
validation behaved unexpectedly:

```bash
cargo test --features diagnostics
cargo test --features diagnostics -- --nocapture
cargo run --features diagnostics --example diagnostics
```

In application code:

```toml
[dependencies]
delaunay = { version = "...", features = ["diagnostics"] }
```

## What It Provides

The feature currently exposes `delaunay::prelude::diagnostics`, which includes:

- `delaunay_violation_report`: returns structured, key-based data about
  Delaunay empty-circumsphere violations.
- `DelaunayViolationReport` and `DelaunayViolationDetail`: compact report types
  suitable for assertions, logs, or issue templates.
- `debug_print_first_delaunay_violation`: emits verbose `tracing` diagnostics
  for the first detected Delaunay violation.
- `verify_conflict_region_completeness`: brute-force cross-checks a
  Bowyer-Watson conflict region against all cells.

The feature does not change triangulation construction, validation, repair, or
query semantics. It only compiles in additional inspection tools.

## Structured Delaunay Violation Reports

Use `delaunay_violation_report` when you want data instead of only log output:

```rust
use delaunay::prelude::diagnostics::delaunay_violation_report;
use delaunay::prelude::triangulation::*;

let vertices = vec![
    vertex!([0.0, 0.0, 0.0]),
    vertex!([1.0, 0.0, 0.0]),
    vertex!([0.0, 1.0, 0.0]),
    vertex!([0.0, 0.0, 1.0]),
];
let dt = DelaunayTriangulation::new(&vertices).unwrap();

let report = delaunay_violation_report(dt.tds(), None).unwrap();
assert!(report.is_valid());
```

Reports store `CellKey` and `VertexKey` values rather than copying every
coordinate. This keeps diagnostics compact and lets callers recover coordinates,
UUIDs, or attached data from the original `Tds`.

Useful fields:

- `number_of_vertices`, `number_of_cells`: size of the inspected TDS.
- `checked_cells`: number of requested cells considered by the scan.
- `violating_cells`: all cells that violate the Delaunay property.
- `first_violation`: first violating cell, its vertex keys, neighbor slots, and
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
full scan of all cells. The insertion path can invoke this check when both are
true:

- the crate was compiled with `diagnostics`
- `DELAUNAY_DEBUG_CONFLICT_VERIFY=1` is set

Use it when investigating missed conflict cells, broken neighbor traversal, or
cavity construction failures.

## Related Debug Environment Variables

Runtime debug switches are documented in
[`docs/dev/debug_env_vars.md`](dev/debug_env_vars.md). The most relevant ones
for diagnostics work are:

- `DELAUNAY_DEBUG_CONFLICT_VERIFY`: enable conflict-region completeness checks.
- `DELAUNAY_DEBUG_CAVITY`: trace cavity boundary extraction and filling.
- `DELAUNAY_DEBUG_NEIGHBORS`: trace neighbor wiring checks.
- `DELAUNAY_DEBUG_RIDGE_LINK`: trace ridge-link validation failures.

These switches should remain investigation tools. Normal user-facing validation
should use `validate()`, `validation_report()`, and typed errors.
