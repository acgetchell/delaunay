# Examples

This directory contains runnable examples demonstrating different parts of the `delaunay` crate.

## Running examples

For best performance (and to match CI), run examples in release mode:

```bash
cargo run --release --example <example_name>
```

Debug mode is useful during development, but can be much slower:

```bash
cargo run --example <example_name>
```

To run all examples in one shot:

```bash
just examples
```

Some examples use random inputs but are seeded for reproducibility. For those examples you can
override the seed with `DELAUNAY_EXAMPLE_SEED`:

```bash
DELAUNAY_EXAMPLE_SEED=666 cargo run --release --example triangulation_3d_100_points
```

## Example index (lexicographic)

### `convex_hull_3d_100_points`

Extracts a 3D convex hull from a Delaunay triangulation built from a stable 100-point random
configuration. Demonstrates hull validation, containment queries, visible-facet queries, and
basic timing.

- Run: `cargo run --release --example convex_hull_3d_100_points`
- Source: [`convex_hull_3d_100_points.rs`](./convex_hull_3d_100_points.rs)

### `into_from_conversions`

Shows `Into`/`From` conversions between `Vertex`/`Point` and coordinate arrays for ergonomic
coordinate access.

- Run: `cargo run --release --example into_from_conversions`
- Source: [`into_from_conversions.rs`](./into_from_conversions.rs)

### `memory_analysis`

Computes coarse construction + convex hull extraction timings across dimensions, and can
optionally report allocation counts when built with `count-allocations`.

- Run: `cargo run --release --example memory_analysis`
- Run (with allocation tracking): `cargo run --release --features count-allocations --example memory_analysis`
- Source: [`memory_analysis.rs`](./memory_analysis.rs)

### `pachner_roundtrip_4d`

Builds a small, deterministic 4D PL-manifold triangulation and roundtrips Pachner flips
(k=1,2,3 plus inverses), asserting the triangulation is unchanged and remains valid.

- Run: `cargo run --release --example pachner_roundtrip_4d`
- Source: [`pachner_roundtrip_4d.rs`](./pachner_roundtrip_4d.rs)

### `point_comparison_and_hashing`

Demonstrates total ordering, equality, and hashing for `Point` values, including NaN/infinity
handling suitable for `HashMap`/`HashSet`.

- Run: `cargo run --release --example point_comparison_and_hashing`
- Source: [`point_comparison_and_hashing.rs`](./point_comparison_and_hashing.rs)

### `topology_editing_2d_3d`

Contrasts the Builder API and Edit API in 2D and 3D, including bistellar flips (k=1, k=2, and
k=3 where applicable) and how Delaunay preservation differs between the two tracks.

- Run: `cargo run --release --example topology_editing_2d_3d`
- Source: [`topology_editing_2d_3d.rs`](./topology_editing_2d_3d.rs)

### `triangulation_3d_100_points`

Constructs a 3D Delaunay triangulation from a stable 100-point random configuration and
demonstrates validation and boundary analysis.

- Run: `cargo run --release --example triangulation_3d_100_points`
- Source: [`triangulation_3d_100_points.rs`](./triangulation_3d_100_points.rs)

### `zero_allocation_iterator_demo`

Compares `vertex_uuids()` against the zero-allocation `vertex_uuid_iter()` and demonstrates
iterator ergonomics and performance.

- Run: `cargo run --release --example zero_allocation_iterator_demo`
- Source: [`zero_allocation_iterator_demo.rs`](./zero_allocation_iterator_demo.rs)
