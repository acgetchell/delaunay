# Module Map

This document owns the `src/` module map, layer boundaries, public namespace
policy, and architecture principles. For import guidance, see
[`prelude_reference.md`](prelude_reference.md). For file-internal section
ordering, see [`module_patterns.md`](module_patterns.md).

## Core Layer

`src/core/` contains triangulation data structures and algorithm machinery:

- `tds/storage.rs` - main `Tds` storage, accessors, identity helpers, and
  construction tests.
- `tds/errors.rs` - TDS error/report vocabulary re-export boundary.
- `tds/equality.rs` - TDS equality implementation and stable simplex identity
  helpers.
- `tds/incidence.rs` - invariant-bearing vertex-to-simplices incidence index.
- `tds/keys.rs` - slotmap-backed `VertexKey` and `SimplexKey` handle types.
- `tds/mutation.rs` - TDS topology mutation, orientation repair, and neighbor
  maintenance.
- `tds/snapshot.rs` - persistence boundary from raw codec records into
  validated UUID snapshots before hydration allocates fresh slotmap keys.
- `tds/validation.rs` - Level 2 Combinatorial Consistency validation and adjacency checks.
- `triangulation.rs` - generic triangulation layer with kernel.
- `construction.rs` - generic construction helpers and initial-simplex setup.
- `insertion.rs` - generic transactional insertion, duplicate detection, and
  insertion telemetry.
- `orientation.rs` - simplex orientation validation, lifted-coordinate
  handling, and positive-orientation canonicalization.
- `query.rs` - read-only generic triangulation accessors, adjacency indices,
  and topology traversal helpers.
- `repair.rs` - generic local topology repair, stale incident-simplex repair,
  and vertex-deletion cavity retriangulation.
- `validation.rs` - generic validation vocabulary and Level 3 orchestration.
- `vertex.rs`, `simplex.rs`, and `facet.rs` - core geometric primitives.
- `edge.rs` - canonical `EdgeKey` for topology traversal.
- `adjacency.rs` - optional lifetime-bound topology indexes:
  `IncidenceView`, `EdgeIndex`, `SimplexNeighborIndex`, and
  `TriangulationAdjacency`.
- `facet_incidence.rs` - TDS-level one-sided/two-sided facet incidence
  analysis; topology-aware boundary classification lives in the
  `Triangulation`/manifold layer.
- `collections/` - optimized collection aliases, key maps, buffers, and
  spatial acceleration structures.
- `algorithms/` - incremental insertion, flips, point location, and
  PL-manifold repair algorithms.
- `traits/` - core boundary/data trait definitions and internal facet-cache
  plumbing.
- `util/` - shared helpers for UUIDs, hashing, deduplication, allocation
  measurement, facet keys, Jaccard diagnostics, Hilbert ordering, and
  canonical point order.
- `operations.rs` - semantic classification and telemetry for topological
  operations.

`edge.rs` and `facet.rs` stay in `src/core/` because they are direct TDS
traversal primitives. A ridge is different: its codimension-2 shape depends on
`D`, and its query/view types support ridge stars, lifted toroidal links, and
PL-manifold validation. Ridge ownership therefore belongs in `src/topology/`.

## Geometry Layer

`src/geometry/` owns geometric primitives and predicates:

- `coordinate_range.rs` - validated coordinate-range value type for random
  point and triangulation generator APIs.
- `embedding.rs` - pure labeled-simplex embedding predicates and witnesses
  used by generic Level 4 validation.
- `kernel.rs` - kernel abstraction (`AdaptiveKernel`, `RobustKernel`,
  `FastKernel`) and `ExactPredicates` marker trait.
- `point.rs` - finite/NaN-aware point operations.
- `predicates.rs` and `robust_predicates.rs` - geometric predicates and robust
  predicate plumbing.
- `sos.rs` - Simulation of Simplicity for deterministic degeneracy resolution.
- `quality.rs` - simplex quality metrics such as radius ratio and normalized
  volume.
- `matrix.rs` - linear algebra support.
- `algorithms/convex_hull.rs` - convex-hull extraction.
- `traits/coordinate.rs` - coordinate abstractions and typed coordinate
  diagnostic payloads.
- `util/` - coordinate conversions, vector norms, circumsphere computations,
  geometric measures, point generation, and random triangulation generation.

The currently supported caller-visible coordinate scalar is `f64`. Exact
arithmetic is used internally by robust predicate fallbacks. If exact
coordinates become caller-visible in the future, add an explicit documented
coordinate model/API rather than loosening ordinary `f64` APIs.

## Delaunay-Facing Layer

`src/delaunay/` owns public Delaunay workflows:

- `builder.rs` - fluent builder API for Euclidean and toroidal/periodic
  construction.
- `construction.rs` - batch construction options, errors, statistics, and
  high-level constructors.
- `insertion.rs` - post-construction vertex insertion and repair orchestration.
- `deletion.rs` - post-construction vertex deletion errors and transactional
  rollback-facing API support.
- `query.rs` - read-only `DelaunayTriangulation` accessors and traversal
  helpers.
- `triangulation.rs` - `DelaunayTriangulation` storage type and insertion-state
  cache.
- `delaunayize.rs` - bounded topology repair plus flip-based Delaunay repair,
  with optional fallback rebuild.
- `flips.rs` - high-level bistellar flip primitive trait and supporting public
  types.
- `pachner.rs` - unified Pachner move enum, result, and dispatch trait over the
  primitive flip layer.
- `locality.rs` - local seed/frontier helpers for Hilbert-local construction
  and repair.
- `repair.rs` - Delaunay repair policies, rebuild config, and repair outcomes.
- `serialization.rs` - conversion to/from `Tds` with topology metadata reset
  rules.
- `spherical.rs` - bounded `S^2`/`S^3` spherical construction,
  embedding-validation, and empty-cap Delaunay backend using the topology
  space coordinate/metric backend.
- `validation.rs` - implemented Level 5 Geometric Predicate APIs for Delaunay
  validation errors and construction validation cadence helpers.
- `property_validation.rs` - TDS-level Delaunay empty-circumsphere scans and
  repair-oriented violation reports used by Level 5 validation APIs.

`src/lib.rs` wires public modules, root re-exports, focused preludes, and the
crate-level documentation map. Delaunay-facing modules are exposed directly as
`delaunay::builder`, `delaunay::construction`, `delaunay::flips`,
`delaunay::repair`, `delaunay::validation`, and focused preludes rather than
through a nested `delaunay::delaunay` facade.

## I/O And Export Layer

`src/io/` owns public downstream-facing export data models:

- `visualization.rs` - generic simplicial-complex primitives for notebooks,
  visualization tools, analysis pipelines, and downstream crates.

This layer is distinct from the TDS snapshot/hydration boundary. TDS serde
remains the canonical validated persistence path; `io::visualization` exposes
stable UUID-based records for consumers that should not depend on runtime
slotmap handles.

## Topology Layer

`src/topology/` owns topology analysis and validation:

- `characteristics/euler.rs` - Euler characteristic computation for full
  complexes and boundaries.
- `characteristics/validation.rs` - topological validation functions.
- `manifold.rs` - topology-only manifold invariants and boundary
  classification over declared global topology.
- `ridge.rs` - ridge candidates, borrowed ridge queries/views, lifted
  ridge-link views, and ridge-star map builders.
- `spaces/euclidean.rs` and `spaces/toroidal.rs` - concrete
  `TopologicalSpace` helper implementations.
- `spaces/spherical.rs` - spherical coordinate and metric backend for points on
  `S^D` embedded in `R^(D+1)`.
- `traits/topological_space.rs` - public `GlobalTopology<D>` metadata enum and
  `TopologyKind`.
- `traits/global_topology_model.rs` - internal scalar-generic
  `GlobalTopologyModel<D>` trait and concrete topology models.

Dependency direction should remain `topology -> core`, not the reverse.

## Public Namespace Policy

`crate::core` is the internal implementation namespace for the low-level TDS
and algorithm layer. The public low-level surface is exposed through curated
modules and focused preludes:

- `delaunay::tds`
- `delaunay::collections`
- `delaunay::algorithms`
- `delaunay::query`
- matching `delaunay::prelude::*` modules

Keep raw bistellar flip primitives out of preludes. User-facing local move
workflows should use `delaunay::prelude::pachner`.

## Architectural Principles

- Separate concerns between topology storage (`core`), geometric predicates
  (`geometry`), Delaunay workflows (`delaunay`), and topology validation
  (`topology`).
- Keep all core types const-generic over dimension.
- Preserve the f64-oriented caller surface until an explicit coordinate model
  broadens it.
- Keep performance infrastructure in `benches/` and benchmark utilities; do not
  turn timing measurements into correctness tests.
- Keep public namespaces curated and focused so examples, doctests, benchmarks,
  and downstream-style tests communicate intent at the import site.
