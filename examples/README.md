# Examples

This directory contains runnable examples that show crate users how to compose
the public APIs. Performance measurement lives in `benches/`; invariant and
regression checks live in `tests/`.

## Running Examples

For best performance and to match CI, run examples in release mode:

```bash
cargo run --release --example <example_name>
```

To run every example:

```bash
just examples
```

Feature-gated examples should be run with the relevant feature:

```bash
cargo run --release --features diagnostics --example diagnostics
```

## Guide Coverage

| Guide | Primary example(s) |
|---|---|
| [`docs/api_design.md`](../docs/api_design.md) | `topology_editing`, `delaunayize_repair` |
| [`docs/diagnostics.md`](../docs/diagnostics.md) | `diagnostics` |
| [`docs/invariants.md`](../docs/invariants.md) | `triangulation_and_hull`, `delaunayize_repair` |
| [`docs/numerical_robustness_guide.md`](../docs/numerical_robustness_guide.md) | `numerical_robustness` |
| [`docs/topology.md`](../docs/topology.md) | `topology_editing` |
| [`docs/validation.md`](../docs/validation.md) | `diagnostics` |
| [`docs/workflows.md`](../docs/workflows.md) | `triangulation_and_hull` |

`docs/code_organization.md` and `docs/property_testing_summary.md` are
contributor/testing references rather than user-facing API guides, so they do
not have dedicated examples here.

## Example Index

### `delaunayize_repair`

Demonstrates the `delaunayize_by_flips` workflow: bounded topology repair
followed by flip-based Delaunay repair, with no-op and flip-then-repair cases
plus custom configuration with fallback.

- Run: `cargo run --release --example delaunayize_repair`
- Source: [`delaunayize_repair.rs`](./delaunayize_repair.rs)

### `diagnostics`

Demonstrates the opt-in `diagnostics` feature, including structured
`DelaunayViolationReport` data and verbose tracing helpers for a deliberately
non-Delaunay TDS.

- Run: `cargo run --release --features diagnostics --example diagnostics`
- Source: [`diagnostics.rs`](./diagnostics.rs)

### `into_from_conversions`

Shows `Into`/`From` conversions between `Vertex`/`Point` and coordinate arrays
for ergonomic coordinate access.

- Run: `cargo run --release --example into_from_conversions`
- Source: [`into_from_conversions.rs`](./into_from_conversions.rs)

### `numerical_robustness`

Compares `FastKernel`, `RobustKernel`, and `AdaptiveKernel` on degenerate
orientation and cospherical insphere queries, then builds a small triangulation
with the default adaptive kernel.

- Run: `cargo run --release --example numerical_robustness`
- Source: [`numerical_robustness.rs`](./numerical_robustness.rs)

### `point_comparison_and_hashing`

Demonstrates total ordering, equality, and hashing for `Point` values,
including NaN/infinity handling suitable for `HashMap`/`HashSet`.

- Run: `cargo run --release --example point_comparison_and_hashing`
- Source: [`point_comparison_and_hashing.rs`](./point_comparison_and_hashing.rs)

### `topology_editing`

Contrasts the Builder API and Edit API in 2D and 3D, including bistellar flips
(k=1, k=2, and k=3 where applicable) and how Delaunay preservation differs
between the two tracks.

- Run: `cargo run --release --example topology_editing`
- Source: [`topology_editing.rs`](./topology_editing.rs)

### `triangulation_and_hull`

Builds seeded 3D and 4D Delaunay triangulations, traverses edges and boundary
facets, extracts convex hulls, and runs containment/visibility queries.

- Run: `cargo run --release --example triangulation_and_hull`
- Source: [`triangulation_and_hull.rs`](./triangulation_and_hull.rs)
