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

This recipe discovers the example programs, runs ordinary examples with the
default feature set, and runs feature-gated examples with their required
features.

Feature-gated examples should be run with the relevant feature:

```bash
cargo run --release --features diagnostics --example diagnostics
```

## Coverage Model

Runnable Rust examples and notebooks have complementary ownership:

- Rust examples teach copyable, compile-checked application workflows whose
  primary result is typed Rust state.
- Notebooks orchestrate the Rust CLI and interpret generated visual or tabular
  artifacts.
- Rustdoc covers individual public items and short method-level snippets.
- Tests, rather than tutorials, own exhaustive dimensions, invalid inputs, and
  invariant edge cases.

Small API idioms such as `Point`/`Vertex` coordinate conversion, comparison,
and hashing live as doctests on their owning items rather than as standalone
examples.

Completeness here means that every advertised public workflow has a primary
learning artifact. It does not require a separate example for every public
method or every supported dimension.

## Runnable Workflow Coverage

| Public workflow | Primary runnable artifact |
|---|---|
| Euclidean construction, options, queries, quality, and hulls in 3D–5D | `triangulation_and_hull` |
| Incremental insertion statistics, location, and deletion | `dynamic_lifecycle` |
| Periodic `T^2`/`T^3` image-point construction | `toroidal_construction` |
| Direct `SphericalDelaunayBuilder` use for `S^2` and `S^3` | `spherical_construction` |
| Payloads, secondary maps, JSON persistence, and set similarity | `data_and_serialization` |
| Builder-managed insertion and explicit Pachner moves | `topology_editing` |
| PL-manifold and Delaunay repair | `delaunayize_repair` |
| Typed validation reports and opt-in diagnostics | `diagnostics` |
| Predicate kernels, degeneracy handling, and circumcenter fallback | `numerical_robustness` |
| Stable mesh and hull export plus 3D visualization | [`00_quickstart.ipynb`](../notebooks/00_quickstart.ipynb) |

The visual complements are
[`01_validation.ipynb`](../notebooks/01_validation.ipynb) for the five-level
validation model and
[`02_spherical_hero.ipynb`](../notebooks/02_spherical_hero.ipynb) for the
`S^2` result. The direct Rust export snippet remains in
[`docs/mesh_export.md`](../docs/mesh_export.md). Detailed workflow contracts
remain in [`docs/workflows.md`](../docs/workflows.md),
[`docs/topology.md`](../docs/topology.md), and
[`docs/numerical_robustness_guide.md`](../docs/numerical_robustness_guide.md).

`docs/code_organization.md` and `docs/property_testing_summary.md` remain
contributor/testing references rather than user-facing workflow guides.

## Example Index

### `data_and_serialization`

Stores typed vertex/simplex payloads, uses caller-owned secondary maps, and
round-trips the serialized TDS through checked Delaunay reconstruction while
comparing the before/after coordinate sets.

- Run: `cargo run --release --example data_and_serialization`
- Source: [`data_and_serialization.rs`](./data_and_serialization.rs)

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

### `dynamic_lifecycle`

Inserts a vertex with observable statistics, locates the inserted point,
deletes the vertex transactionally, and validates the final triangulation.

- Run: `cargo run --release --example dynamic_lifecycle`
- Source: [`dynamic_lifecycle.rs`](./dynamic_lifecycle.rs)

### `numerical_robustness`

Compares `FastKernel`, `RobustKernel`, and `AdaptiveKernel` on degenerate
orientation and cospherical insphere queries, demonstrates the near-singular
circumcenter exact-solve fallback, then builds a small triangulation with the
default adaptive kernel.

- Run: `cargo run --release --example numerical_robustness`
- Source: [`numerical_robustness.rs`](./numerical_robustness.rs)

### `spherical_construction`

Uses `SphericalDelaunayBuilder` directly to construct and validate the bounded
`S^2` and `S^3` prototypes. Visualization remains in the spherical notebook.

- Run: `cargo run --release --example spherical_construction`
- Source: [`spherical_construction.rs`](./spherical_construction.rs)

### `topology_editing`

Contrasts Delaunay vertex lifecycle APIs and the Pachner Move API in 2D and
3D, including k=1, k=2, and k=3 moves and how Delaunay preservation differs
between the workflows.

- Run: `cargo run --release --example topology_editing`
- Source: [`topology_editing.rs`](./topology_editing.rs)

### `toroidal_construction`

Demonstrates closed, boundary-free `T^2` and `T^3` image-point quotients.

- Run: `cargo run --release --example toroidal_construction`
- Source: [`toroidal_construction.rs`](./toroidal_construction.rs)

### `triangulation_and_hull`

Builds seeded 3D through 5D Delaunay triangulations, traverses edges and
boundary facets, locates points, evaluates simplex quality, extracts convex
hulls, and runs containment/visibility queries.

- Run: `cargo run --release --example triangulation_and_hull`
- Source: [`triangulation_and_hull.rs`](./triangulation_and_hull.rs)
