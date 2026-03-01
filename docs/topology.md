# Topology

This document describes the topology-related parts of the `delaunay` crate:
Level 3 manifold validation, Euler characteristic checks, and support for
different topological spaces (Euclidean and toroidal, with spherical planned).

If you want the user-facing guide to the full validation stack (Levels 1–4), start
with `docs/validation.md`.

For the theoretical background and rationale behind the invariants themselves, see
[`invariants.md`](invariants.md).

## Code layout

Relevant modules (lexicographically sorted):

```text
src/
├── core/
│   └── triangulation.rs
├── geometry/
├── lib.rs
├── topology/
│   ├── characteristics/
│   │   ├── euler.rs
│   │   └── validation.rs
│   ├── manifold.rs
│   ├── spaces/
│   │   ├── euclidean.rs
│   │   ├── spherical.rs
│   │   └── toroidal.rs
│   └── traits/
│       ├── global_topology_model.rs
│       └── topological_space.rs
└── triangulation/
    └── flips.rs
```

Notes:

- This repository does not use `mod.rs`; module declarations live in `src/lib.rs`.
- Topology is combinatorial: core types (`Tds`, `Cell`, `Vertex`) do not require
  `T: CoordinateScalar` at the type level; geometric operations/validation are
  gated behind `T: CoordinateScalar`.

## Level 3 topology validation (`Triangulation::is_valid()`)

`Triangulation::is_valid()` validates *topology-only* invariants (Level 3). It
intentionally does **not** validate lower layers (elements or TDS structure).

For cumulative validation, use `Triangulation::validate()` (Levels 1–3) or
`DelaunayTriangulation::validate()` (Levels 1–4).

### Always-checked invariants

Level 3 always checks:

- **Codimension-1 facet degree** (pseudomanifold / manifold-with-boundary):
  every (D−1)-facet is incident to exactly 1 (boundary) or 2 (interior) D-cells.
  (`topology::manifold::validate_facet_degree`)
- **Codimension-2 boundary manifoldness**: if a boundary exists, it is closed
  ("no boundary of boundary"). (`topology::manifold::validate_closed_boundary`)
- **Connectedness**: a single connected component in the cell neighbor graph.
- **No isolated vertices**: every vertex is incident to at least one cell.
- **Euler characteristic** for the full D-dimensional simplicial complex.
  (`topology::characteristics::validation::validate_triangulation_euler_with_facet_to_cells_map`)

### `TopologyGuarantee`-dependent checks

`TopologyGuarantee` controls which additional PL-manifold checks Level 3 runs:

- `TopologyGuarantee::Pseudomanifold`: no additional link checks.
- `TopologyGuarantee::PLManifold`: runs ridge-link validation during insertion and
  requires a completion-time vertex-link pass for full certification.
- `TopologyGuarantee::PLManifoldStrict`: runs vertex-link validation after every
  insertion (slowest, maximum safety).

Implementation pointers:

- Level 3 entry point: `src/core/triangulation.rs` (`Triangulation::is_valid`)
- Manifold validators: `src/topology/manifold.rs`
- Euler characteristic helpers: `src/topology/characteristics/{euler.rs,validation.rs}`

## Euler characteristic (`topology::characteristics`)

Level 3 uses Euler characteristic (χ) as a global combinatorial consistency check.

### What is computed?

`topology::characteristics::euler::count_simplices` computes the f-vector
(f₀…f_D) for the **full** simplicial complex induced by all D-cells in the TDS.
The Euler characteristic is:

```text
χ = Σ(k=0..D) (-1)^k · f_k
```

### Expected values (simple classification)

`topology::characteristics::validation::validate_triangulation_euler*` produces a
`TopologyCheckResult` containing χ, an expected value (when known), a coarse
classification, the full f-vector, and diagnostic notes.

The expected χ is determined from a simple classification:

- `Empty` (no cells): expected χ = 0
- `SingleSimplex(D)`: expected χ = 1
- `Ball(D)` (has boundary): expected χ = 1
- `ClosedSphere(D)` (no boundary): expected χ = 1 + (-1)^D
- `Unknown`: no expected χ (treated as "can't decide")

For most finite Delaunay triangulations in Euclidean space, the complex has a
boundary (convex hull), so the expected classification is `Ball(D)` and χ = 1.

### Boundary-only χ (not used by Level 3)

For research/debugging, `topology::characteristics::euler::count_boundary_simplices`
computes simplex counts for the boundary complex only (a (D−1)-dimensional
simplicial complex). This is currently not part of Level 3 validation.

## PL-manifold validators (`topology::manifold`)

`src/topology/manifold.rs` contains combinatorial validators for manifold and
PL-manifold invariants (no geometric predicates):

- `validate_facet_degree`
- `validate_closed_boundary`
- `validate_ridge_links`
- `validate_vertex_links`

The module docs explain which conditions are necessary vs sufficient, especially
for D ≥ 3.

## Topological spaces

### Architecture: Metadata vs Behavior Models

The topology system uses a two-layer architecture to separate public API stability from
internal implementation flexibility:

**Public metadata layer**: `GlobalTopology<D>` (in `src/topology/traits/topological_space.rs`)

- Runtime enum exposed through public API
- Stores topology configuration (e.g., toroidal domain periods)
- Remains stable across implementation refactoring
- Accessible via `Triangulation::global_topology()` and `DelaunayTriangulation::global_topology()`

**Internal behavior layer**: `GlobalTopologyModel<D>` trait
(in `src/topology/traits/global_topology_model.rs`)

- Scalar-generic trait defining topology-specific operations:
  - `canonicalize_point_in_place`: normalize coordinates to fundamental domain
  - `lift_for_orientation`: apply periodic offsets for orientation predicates
  - `periodic_domain`: expose domain periods for periodic topologies
  - `supports_periodic_facet_signatures`: indicate periodic cell support
- Concrete implementations:
  - `EuclideanModel`: identity operations (no wrapping or lifting)
  - `ToroidalModel`: domain wrapping and lattice-offset lifting
  - `SphericalModel`: scaffold for sphere-constrained operations (future work)
  - `HyperbolicModel`: scaffold for hyperbolic model operations (future work)
- Accessed internally via `GlobalTopology::model()` adapter method

This separation allows core triangulation and builder code to delegate topology-specific
behavior to model implementations without branching on the `GlobalTopology` enum throughout
the codebase.

**Space helper types** (in `src/topology/spaces/`):

- `EuclideanSpace`, `ToroidalSpace`, `SphericalSpace`: `f64`-oriented helper types
- Not directly used by scalar-generic core algorithms
- Provide utilities for specific topology computations

### Toroidal topology support

Toroidal (periodic) triangulations are **fully implemented and functional**. You can
construct toroidal triangulations using `DelaunayTriangulationBuilder`:

```rust
use delaunay::prelude::triangulation::*;

// 2D periodic triangulation
let vertices = vec![
    vertex!([0.1, 0.1]),
    vertex!([0.9, 0.9]),
    // ...
];

let dt = DelaunayTriangulationBuilder::new(&vertices)
    .toroidal([1.0, 1.0]) // Phase 1: canonicalized toroidal construction
    .build::<()>()
    .unwrap();
```

Toroidal triangulations handle point canonicalization (wrapping coordinates to the
fundamental domain) and distance computations across periodic boundaries. The
implementation supports both 2D and 3D toroidal spaces. For true periodic
image-point construction, use `.toroidal_periodic([..])`.

For more examples, see the toroidal section in the main `README.md`.

### Future work

Spherical space topology is defined but not yet fully integrated with the
construction and validation pipeline.

## Triangulation editing (`src/triangulation/`)

`src/triangulation/flips.rs` exposes explicit bistellar-flip editing APIs
(`BistellarFlips`) built on `core::algorithms::flips`. These operations:

- are topological edits (they can change manifold structure), and
- do not automatically restore the Delaunay property.

After batch edits, consider repair and/or validation (see `docs/api_design.md`
and `docs/validation.md`).

## Further reading

Primary references used throughout the topology and validation code:

1. A. Hatcher, *Algebraic Topology*, Cambridge University Press, 2002.
   - <https://pi.math.cornell.edu/~hatcher/AT/ATpage.html>
2. J. R. Munkres, *Elements of Algebraic Topology*, Addison–Wesley, 1984.
3. C. P. Rourke & B. J. Sanderson, *Introduction to Piecewise-Linear Topology*,
   Springer, 1972.
4. H. Edelsbrunner & J. Harer, *Computational Topology: An Introduction*, AMS, 2010.
5. A. Zomorodian, *Topology for Computing*, Cambridge University Press, 2005.
6. J. Stillwell, *Euler’s Gem: The Polyhedron Formula and the Birth of Topology*,
   Princeton University Press, 2010.
7. CGAL documentation (triangulation validation patterns and invariants):
   <https://doc.cgal.org/latest/Triangulation_3/index.html>
