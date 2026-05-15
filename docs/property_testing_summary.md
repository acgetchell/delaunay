# Property-Based Testing

This repository uses property-based testing with `proptest` to exercise
structural, topological, and geometric invariants across dimensions.

Keep this document separate from [`diagnostics.md`](diagnostics.md). This page
answers "what randomized invariants do we test, and how do I run or reproduce
them?" Diagnostics answers "what evidence do I collect when something fails?"

For the theory behind the invariants, see [`invariants.md`](invariants.md). For
validation API semantics, see [`validation.md`](validation.md).

This document is intentionally current-only. Historical notes belong in
[`docs/archive/`](archive/).

## Scope

Property tests should assert durable invariants rather than only checking that a
function does not panic. In this crate, that usually means one of:

- predicate sign agreement, determinism, or exact-degeneracy behavior
- TDS incidence, mapping, neighbor, and orientation invariants
- topology guarantees such as manifoldness, Euler characteristic, and links
- Delaunay-layer insertion, repair, rollback, and validation behavior
- serialization round trips and public API surface stability

Dimension-generic properties should cover 2D through 5D when feasible. Heavier
or intentionally slow properties may be gated by `slow-tests` or temporarily
ignored while the v0.7.8 slow-test taxonomy cleanup is in progress.

## Where Tests Live

Most property tests are Rust integration tests under [`tests/`](../tests/).

Core property modules:

- `tests/proptest_predicates.rs`: orientation and in-sphere predicate
  properties.
- `tests/proptest_sos.rs`: Simulation of Simplicity non-degeneracy,
  determinism, and translation-invariance properties.
- `tests/proptest_point.rs`: `Point` equality, ordering, hashing, and
  conversion semantics.
- `tests/proptest_geometry.rs`: geometric utility and measure properties.
- `tests/proptest_safe_conversions.rs`: checked conversion behavior.
- `tests/proptest_cell.rs`, `tests/proptest_facet.rs`,
  `tests/proptest_vertex.rs`: low-level simplex component invariants.
- `tests/proptest_tds.rs`: TDS mapping, neighbor, duplicate-cell, and structure
  invariants.
- `tests/proptest_orientation.rs`: coherent orientation across construction,
  tamper detection, and incremental insertion.
- `tests/proptest_triangulation.rs`: triangulation-layer validation and quality
  invariants.
- `tests/proptest_euler_characteristic.rs`: Euler characteristic behavior by
  dimension and topology.
- `tests/proptest_toroidal.rs`: toroidal/canonicalization and periodic topology
  properties.
- `tests/proptest_convex_hull.rs`: convex-hull extraction and boundary
  consistency.
- `tests/proptest_flips.rs`: flip/Pachner invariant checks.
- `tests/proptest_delaunay_triangulation.rs`: Delaunay-layer insertion, repair,
  rollback, validation, and duplicate/degeneracy behavior.
- `tests/proptest_serialization.rs`: serde round trips and neighbor/data
  preservation.

The detailed per-file test inventory lives in [`tests/README.md`](../tests/README.md).

## Running Property Tests

Recommended repository workflows:

```bash
# All Rust integration tests, including proptests, in release mode.
just test-integration

# Integration tests excluding tests whose names contain `prop_`.
just test-integration-fast
```

Target one property module while iterating:

```bash
cargo test --release --test proptest_predicates
cargo test --release --test proptest_orientation
cargo test --release --test proptest_tds
cargo test --release --test proptest_sos
```

Use release mode for representative property-test performance. Exact predicate
fallbacks can make higher-dimensional property tests much slower in debug mode.

Slow/stress coverage:

```bash
just test-slow
just test-slow-release
```

The intended contract for `slow-tests` versus `#[ignore]` is being cleaned up in
the v0.7.8 test taxonomy work (#380). Until that lands, prefer the documented
recipes in [`tests/README.md`](../tests/README.md) for one-off ignored tests.

## Configuration

The repository default is defined in [`proptest.toml`](../proptest.toml):

```toml
cases = 32
```

Override it locally with environment variables:

```bash
PROPTEST_CASES=128 cargo test --release --test proptest_point
PROPTEST_SEED=12345 cargo test --release --test proptest_triangulation -- --nocapture
```

Common knobs:

- `PROPTEST_CASES`: number of randomized cases per property.
- `PROPTEST_SEED`: reproduce a specific random run from failure output.
- `-- --nocapture`: show tracing/diagnostic output emitted by the test.
- `-- --test-threads=1`: remove cross-test interleaving while debugging.

## Interpreting Failures

When a property fails, `proptest` prints a minimized counterexample and a
reproduction seed. A good debugging pass is:

1. Re-run the exact failing module with `PROPTEST_SEED=...`.
2. Add `-- --nocapture` if the test emits useful tracing.
3. Narrow the property or promote the minimized case into a fixed regression
   test when it represents a real bug.
4. Enable the `diagnostics` feature or debug environment variables only when
   the ordinary error and validation report are not enough.

See [`diagnostics.md`](diagnostics.md) for the diagnostic tools and feature
flags available during that last step.

## Regression Files

Proptest automatically records minimized failures in
`tests/*.proptest-regressions`. These files are part of the regression suite:

- Commit generated regression files when they represent real failures.
- Do not hand-edit them; let `proptest` manage the format.
- Keep fixed-bug integration tests in `tests/regressions.rs` when the scenario
  deserves a clear, named test in addition to the generated corpus.

Current generated corpus:

- `tests/proptest_delaunay_triangulation.proptest-regressions`

## Writing New Properties

- Tie each property to a named invariant from [`invariants.md`](invariants.md),
  [`validation.md`](validation.md), or a focused issue.
- Prefer dimension-generic coverage across 2D-5D when the runtime is reasonable.
- Use generated inputs that satisfy the preconditions being tested; reject
  out-of-scope cases with `prop_assume!` instead of weakening assertions.
- Keep debug output behind `tracing` and the `diagnostics` feature when it is
  noisy or expensive.
- Add fixed regression tests for minimized failures that should be readable to a
  future maintainer without replaying the whole property suite.
