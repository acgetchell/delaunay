# Topology

This document describes the topology-related parts of the `delaunay` crate:
Level 3 Intrinsic PL Topology validation, Euler characteristic checks, and
support for different topological spaces. Euclidean and toroidal workflows are
fully integrated; spherical support provides a real `S^D` coordinate/metric
backend plus a bounded `S^2`/`S^3` construction and validation prototype, while
full spherical integration and hyperbolic support remain future work.

If you want the user-facing guide to the full validation stack (Levels 1–5), start
with `docs/validation.md`.

For the theoretical background and rationale behind the invariants themselves, see
[`invariants.md`](invariants.md).

## Code layout

Relevant modules (lexicographically sorted):

```text
src/
├── core/
│   ├── facet_incidence.rs
│   ├── query.rs
│   ├── triangulation.rs
│   └── validation.rs
├── delaunay/
│   ├── flips.rs
│   ├── repair.rs
│   └── validation.rs
├── geometry/
├── lib.rs
├── topology/
│   ├── characteristics/
│   │   ├── euler.rs
│   │   └── validation.rs
│   ├── manifold.rs
│   ├── ridge.rs
│   ├── spaces/
│   │   ├── euclidean.rs
│   │   ├── spherical.rs
│   │   └── toroidal.rs
│   └── traits/
│       ├── global_topology_model.rs
│       └── topological_space.rs
```

Notes:

- This repository does not use `mod.rs`; module declarations live in `src/lib.rs`.
- Topology is combinatorial: core types (`Tds`, `Simplex`, `Vertex`) keep
  topology and payload data separate from coordinate parsing. The currently
  supported caller-visible coordinate scalar is finite `f64`, stored through
  validated `Point` coordinates; exact-coordinate input, if added in the future,
  should be an explicit documented API rather than incidental generic support.

## Level 3 Intrinsic PL Topology validation (`Triangulation::is_valid_topology()`)

`Triangulation::is_valid_topology()` validates realization-independent PL-topology
invariants (Level 3). It intentionally does **not** validate lower layers
(elements or combinatorial TDS consistency), nor does it certify Level 4
realization validity or Level 5 geometric predicates.

For cumulative validation, use `Triangulation::validate()` (Levels 1–3) or
`DelaunayTriangulation::validate()` (Levels 1–5).

### Always-checked invariants

Level 3 always checks:

- **Codimension-1 facet degree** (pseudomanifold / manifold-with-boundary):
  every (D−1)-facet is incident to exactly 1 or 2 D-simplices. Public facet
  incidence APIs parse this into the owner-bound `FacetToSimplicesIndex` via
  `Tds::build_facet_to_simplices_index`; Level 3 validation builds one raw
  `FacetToSimplicesMap`, parses it into `ValidatedFacetDegreeMap`, and reuses
  that proof-bearing map so boundary, ridge-link, vertex-link, and Euler checks do not
  rebuild or revalidate the same facet-degree evidence. Boundary classification
  additionally excludes admissible periodic self-identifications, which are
  closed quotient topology rather than boundary.
- **Codimension-2 boundary manifoldness**: if a boundary exists, it is closed
  ("no boundary of boundary"). (`topology::manifold::validate_closed_boundary`)
- **Connectedness**: a single connected component in the simplex neighbor graph.
- **No isolated vertices**: every vertex is incident to at least one simplex.
- **Euler characteristic** for the full D-dimensional simplicial complex.
  (`topology::characteristics::validation::validate_triangulation_euler_from_validated_facet_map`)

### `TopologyGuarantee`-dependent checks

`TopologyGuarantee` controls which additional PL-manifold checks Level 3 runs:

- `TopologyGuarantee::Pseudomanifold`: no additional link checks.
- `TopologyGuarantee::PLManifold`: runs ridge-link validation during insertion and
  requires a completion-time vertex-link pass for full certification.
- `TopologyGuarantee::PLManifoldStrict`: runs vertex-link validation after every
  insertion (slowest, maximum safety).

Implementation pointers:

- Level 3 entry points and validation vocabulary: `src/core/validation.rs`
  (`Triangulation::is_valid_topology`, `Triangulation::validate`)
- Owner-level topology validators: `src/core/validation.rs` and
  `src/delaunay/query.rs`
  (`Triangulation::validate_ridge_links`,
  `Triangulation::validate_ridge_links_for_simplices`,
  `Triangulation::validate_vertex_links`, and the matching
  `DelaunayTriangulation` forwarding methods)
- Storage-level manifold validators: `src/topology/manifold.rs`
  (`validate_closed_boundary`, `validate_vertex_links`, `validate_ridge_links`)
- Internal raw-map reuse helpers: `src/topology/manifold.rs`
  (`ValidatedFacetDegreeMap::try_from_facet_map`,
  `validate_closed_boundary_from_validated_facet_map`,
  `validate_vertex_links_from_validated_facet_map`)
- Euler characteristic helpers: `src/topology/characteristics/{euler.rs,validation.rs}`

## Boundary semantics

Facet incidence by itself does **not** prove that a facet is a manifold boundary.
It only describes how many D-simplices share a canonical facet key in the TDS.
The current API keeps this distinction explicit:

- `Triangulation::facet_incidence_index()` and
  `DelaunayTriangulation::facet_incidence_index()` report raw facet incidence;
  `FacetIncidenceView::is_one_sided()` identifies one-sided incidences. This is
  a Level 1–2 incidence fact, not a topology-aware boundary classification.
- `Triangulation::boundary_facets()` and `DelaunayTriangulation::boundary_facets()`
  report true boundary facets after interpreting the incidence under the
  triangulation's `GlobalTopology`.
- `topology::manifold::classify_boundary_facet` is the semantic boundary
  classifier used by validation and query code.

This matters for periodic quotient topology. In a true toroidal triangulation,
a facet can be one-sided in the raw incidence index because the owning simplex
has an admissible periodic self-neighbor. That facet is a closed
self-identification, not a boundary. Conversely, an open one-sided facet in a
closed topology (`Toroidal`, `Spherical`, or `Hyperbolic`) is an invariant error,
not a valid boundary.

The validation order is therefore:

1. Parse facet incidence and reject non-manifold multiplicity.
2. Classify one-sided incidences against `GlobalTopology`.
3. Validate boundary closure, ridge links, vertex links, and connectedness.
4. Use Euler characteristic as a compatibility check for the already classified
   topology.

## Euler characteristic (`topology::characteristics`)

Level 3 uses Euler characteristic (χ) as a global combinatorial consistency check.

### What is computed?

`topology::characteristics::euler::count_simplices` computes the f-vector
(f₀…f_D) for the **full** simplicial complex induced by all D-simplices in the TDS.
The Euler characteristic is:

```text
χ = Σ(k=0..D) (-1)^k · f_k
```

### Expected values (simple classification)

`topology::characteristics::validation::validate_triangulation_euler*` produces a
`TopologyCheckResult` containing χ, an expected value (when known), a coarse
classification, the full f-vector, and diagnostic notes.

The expected χ is determined from declared topology metadata plus the
topology-aware boundary classification described above:

- `Empty` (no simplices): expected χ = 0
- `SingleSimplex(D)`: expected χ = 1
- `Ball(D)` (has boundary): expected χ = 1
- `ClosedSphere(D)` (no boundary): expected χ = 1 + (-1)^D
- `ClosedToroid(D)` (periodic quotient): expected χ = 0
- `Unknown`: no expected χ (treated as "can't decide")

For most finite Delaunay triangulations in Euclidean space, the complex has a
boundary (convex hull), so the expected classification is `Ball(D)` and χ = 1.

Euler characteristic is not a topology detector by itself. It is an invariant
used after the manifold and boundary checks above. Many non-homeomorphic
manifolds share the same χ, especially in higher dimensions, so χ must not bless
an arbitrary gluing as toroidal or spherical without the corresponding
topological construction and local manifold checks.

### Boundary-only χ (not used by Level 3)

For research/debugging, `topology::characteristics::euler::count_boundary_simplices`
computes simplex counts for the boundary complex only (a (D−1)-dimensional
simplicial complex). This is currently not part of Level 3 validation.

## PL-manifold validators (`topology::manifold`)

The public owner-level entry points for PL-manifold link checks are:

- `Triangulation::validate_ridge_links()` and
  `DelaunayTriangulation::validate_ridge_links()` for the global codimension-2
  ridge-link screen.
- `Triangulation::validate_ridge_links_for_simplices()` and
  `DelaunayTriangulation::validate_ridge_links_for_simplices()` for localized
  post-edit diagnostics over a touched simplex frontier.
- `Triangulation::validate_vertex_links()` and
  `DelaunayTriangulation::validate_vertex_links()` for the canonical
  vertex-link PL-manifold certification.

These owner methods are the preferred API for papers, examples, tests, and
application code because they carry the triangulation's topology metadata and
avoid exposing raw TDS storage.

`src/topology/manifold.rs` also contains storage-level combinatorial validators
for manifold and PL-manifold invariants (no geometric predicates):

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
  - `supports_periodic_facet_signatures`: indicate periodic simplex support
- Concrete implementations:
  - `EuclideanModel`: identity operations (no wrapping or lifting)
  - `ToroidalModel`: domain wrapping and lattice-offset lifting
  - `SphericalModel`: unit-sphere projection for finite nonzero
    `GlobalTopology` coordinate arrays
  - `HyperbolicModel`: scaffold for hyperbolic model operations (future work)
- Accessed internally via `GlobalTopology::model()` adapter method

This separation allows core triangulation and builder code to delegate topology-specific
behavior to model implementations without branching on the `GlobalTopology` enum throughout
the codebase.

**Topology helpers and coordinate backends** (in `src/topology/spaces/`):

- `EuclideanSpace`, `ToroidalSpace`: `f64`-oriented `TopologicalSpace` helper types
- Not directly used by scalar-generic core algorithms
- Provide utilities for specific topology computations
- Spherical Delaunay construction uses `SphericalPoint<D>` and `SphericalMetric<D>`
  as the coordinate/metric backend: `D` is the intrinsic dimension of `S^D`, and
  points carry `D + 1` ambient coordinates in `R^(D+1)`. `SphericalModel`
  remains internal metadata/model plumbing for `GlobalTopology::Spherical`, so
  its `D` is the coordinate-array length seen by `GlobalTopologyModel<D>` rather
  than the spherical Delaunay intrinsic dimension.

### Toroidal topology support

Toroidal workflows use two explicit `DelaunayTriangulationBuilder` modes:
canonicalized coordinate wrapping and true periodic quotient construction. The
periodic path is release-validated in 2D and for compact 3D inputs:

```rust
use delaunay::prelude::construction::{DelaunayTriangulationBuilder, vertex};

// 2D Euclidean triangulation after wrapping inputs into a toroidal domain
let vertices = vec![
    vertex![0.1, 0.1]?,
    vertex![0.9, 0.9]?,
    // ...
];

let dt = DelaunayTriangulationBuilder::new(&vertices)
    .try_canonicalized_toroidal([1.0, 1.0])? // Canonicalized toroidal construction
    .build()?;
```

Canonicalized toroidal construction wraps coordinates into the fundamental
domain before building a Euclidean triangulation of the wrapped point set. It
does not attach toroidal manifold topology to the output and does not rewire
opposite boundary facets. For a true periodic quotient, use
`.try_toroidal([..])`; the validated image-point path currently covers 2D
and compact 3D fixtures. 4D/5D periodic quotients fail fast pending scalable
construction work in issue #416.

For more examples, see the toroidal section in the main `README.md`.

### Future work

Spherical topology is defined in metadata/behavior-model layers and
canonicalizes finite nonzero coordinates onto the unit sphere. The bounded
`SphericalDelaunayBuilder` prototype constructs `S^2`/`S^3` Delaunay simplices from
ambient `R^3`/`R^4` points by convex-hull duality, while keeping Level 3
PL-topology validation separate from spherical Level 4/5 geometry. Full
spherical integration across 2D-5D and hyperbolic integration remain future
work.

## Triangulation editing (`src/delaunay/`)

`src/delaunay/flips.rs` exposes explicit bistellar-flip editing APIs
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
