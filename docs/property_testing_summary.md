# Property-Based Testing

This repository uses property-based testing (via `proptest`) to exercise structural,
topological, and geometric invariants across multiple dimensions.

For a curated list of invariants (with rationale and implementation pointers), see
[`invariants.md`](invariants.md).

This document is intentionally *current-only*. Historical notes belong in
`docs/archive/`.

## Where the tests live

Most property tests are Rust integration tests under `tests/`.

Core modules (by layer):

- `tests/proptest_predicates.rs`: geometric predicate properties (orientation, insphere).
- `tests/proptest_point.rs`: `Point` semantics (NaN-aware equality/ordering, hashing).
- `tests/proptest_tds.rs`: TDS combinatorial invariants (mappings, neighbor consistency, duplicate-cell prevention).
- `tests/proptest_orientation.rs`: coherent-orientation invariants (construction, tamper detection, incremental insertion coherence).
- `tests/proptest_triangulation.rs`: triangulation-layer invariants and quality metrics.
- `tests/proptest_delaunay_triangulation.rs`: Delaunay-layer properties (insertion, repair, validation).

There are also additional `tests/proptest_*.rs` modules covering specific types and
algorithms (cell, facet, vertex, convex hull, serialization, etc.).

## Running property tests

Recommended (matches project workflows):

```bash
# All Rust integration tests (includes proptests)
just test-integration
```

Target a single proptest module:

```bash
cargo test --test proptest_predicates
cargo test --test proptest_tds
```

For quick local iteration, you can skip proptests whose names are prefixed with `prop_`:

```bash
just test-integration-fast
```

## Configuration

`proptest` can be configured via:

- Environment variables (e.g. `PROPTEST_CASES`, `PROPTEST_SEED`)
- The repository config file `proptest.toml`

Example:

```bash
PROPTEST_CASES=1000 cargo test --test proptest_point
```

## Interpreting failures

When a property fails, `proptest` shrinks the input to a minimal counterexample and
prints a reproduction seed. Re-run with `-- --nocapture` to see full output.
