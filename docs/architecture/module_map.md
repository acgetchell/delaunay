# Module Map

This document owns the `src/` module map, layer boundaries, public namespace
policy, and architecture principles. For import guidance, see
[`prelude_reference.md`](prelude_reference.md). For file-internal section
ordering, see [`module_patterns.md`](module_patterns.md).

## Shared Refinement Boundary

- `src/refinement.rs` - generic recoverable `RefinementError<T, E>` carrier for
  consuming transitions between proof-bearing owners. It keeps the accepted
  lower-layer owner attached to the typed rejection reason on failure.

## Core Layer

`src/core/` contains the Levels 1–2 triangulation data structure and low-level
algorithm machinery that operates on it:

- `tds/model.rs` - proof-bearing Levels 1–2 `Tds` storage, read-only
  accessors, identity helpers, and construction tests. Canonical fields remain
  visible only inside `core::tds`.
- `tds/builder.rs` - explicit-connectivity `TdsBuilder` publication boundary.
  It owns raw vertex/simplex specifications and returns a constructed TDS only
  after cumulative Levels 1–2 validation succeeds.
- `tds/draft.rs` - mutable unpublished `TdsDraft` representation for explicitly
  supplied vertices and maximal simplices, plus its checked Levels 1–2
  `finish()` publication boundary.
- `tds/errors.rs` - TDS error/report vocabulary re-export boundary.
- `tds/equality.rs` - TDS equality implementation and stable simplex identity
  helpers.
- `tds/incidence.rs` - invariant-bearing vertex-to-simplices incidence index.
- `tds/keys.rs` - slotmap-backed `VertexKey` and `SimplexKey` handle types.
- `tds/mutation.rs` - checked TDS topology transitions, construction
  completion, orientation repair, incidence updates, and neighbor maintenance.
  Higher proof owners delegate storage edits here.
- `tds/snapshot.rs` - persistence boundary from raw codec records into
  validated UUID snapshots before hydration allocates fresh slotmap keys, plus
  the owner-borrowed Levels 1–2 evidence required for unchecked snapshot
  encoding.
- `tds/validation.rs` - Level 2 Combinatorial Consistency validation and adjacency checks.
- `tds/rollback.rs` - canonical TDS snapshot ownership plus the shared
  transaction window used by nested proof refinements without duplicate
  snapshots.
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
- `algorithms/flips/` - bistellar move vocabulary, mutation and orientation
  engines, validation-context construction, typed errors, queued Delaunay
  repair, and shared local topology/predicate support.
- `algorithms/insertion.rs` - shared cavity topology primitives, neighbor
  wiring and repair helpers, and insertion failure types. The Levels 3–4
  insertion algorithm lives in `triangulation/insertion.rs`; this core module
  does not own a construction workflow.
- `algorithms/` also owns point location and PL-manifold repair algorithms
  alongside the focused flip module.
- `traits/` - core boundary/data trait definitions and facet-incidence analysis.
- `util/` - shared helpers for UUIDs, hashing, deduplication, allocation
  measurement, facet keys, Jaccard diagnostics, Hilbert ordering, and
  canonical point order.
- `operations.rs` - semantic classification and telemetry for topological
  operations.

`edge.rs` and `facet.rs` stay in `src/core/` because they are direct TDS
traversal primitives. A ridge is different: its codimension-2 shape depends on
`D`, and its query/view types support ridge stars, lifted toroidal links, and
PL-manifold validation. Ridge ownership therefore belongs in `src/topology/`.

## Triangulation Layer

`src/triangulation/` owns the proof-bearing Levels 3–4 `Triangulation` domain
and operations over it. It consumes the checked Levels 1–2 TDS surface without
receiving raw access to canonical TDS storage:

- `model.rs` - `Triangulation` owner plus kernel and topology metadata.
- `builder.rs` - `TriangulationBuilder` publication boundary. It consumes a
  TDS plus kernel/topology options and returns a triangulation only after
  cumulative Levels 1–4 validation succeeds. Strict mode is non-mutating and
  snapshot-free; canonicalizing mode normalizes orientation transactionally.
  Both return the exact input TDS in a recoverable failure.
- `draft.rs` - crate-internal `TriangulationDraft` publication state shared by
  `TriangulationBuilder` and higher-layer construction. Its crate-private
  `UnverifiedTriangulation` wrapper is removed only after Levels 3–4 certification.
  It is not a separate public workflow because a proof-bearing TDS already has
  complete connectivity and the generic layer exposes no further staged
  mutation.
- `construction.rs` - generic construction helpers and initial-simplex setup.
- `insertion.rs` - generic transactional insertion, duplicate detection, and
  insertion telemetry.
- `orientation.rs` - simplex orientation validation, lifted-coordinate
  handling, and positive-orientation canonicalization.
- `query.rs` - read-only generic triangulation accessors, adjacency indices,
  barycenters, and topology traversal helpers.
- `repair.rs` - generic local topology repair, stale incident-simplex repair,
  and vertex-deletion cavity retriangulation.
- `rollback.rs` - rollback guards for generic mutation windows.
- `validation.rs` - Level 3 topology validation vocabulary and orchestration.
- `realization.rs` - Level 4 realization validation and the shared Levels 3–4
  certification used by both strict TDS-to-`Triangulation` refinement and
  canonicalizing builder publication.
- `flips.rs` - public primitive bistellar-edit contract, defined only for the
  generic triangulation owner.
- `pachner.rs` - composed Pachner workflow over generic triangulations.
- `jaccard.rs` and `locality.rs` - owner-level query locality and diagnostic
  helpers.

## Geometry Layer

`src/geometry/` owns geometric primitives and predicates:

- `coordinate_range.rs` - validated coordinate-range value type for random
  point and triangulation generator APIs.
- `realization.rs` - pure labeled-simplex realization predicates and witnesses
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
  geometric measures, point generation, random triangulation generation, and
  the private `simplex_lp.rs` implementation used by realization validation.

The currently supported caller-visible coordinate scalar is `f64`. Exact
arithmetic is used internally by robust predicate fallbacks. If exact
coordinates become caller-visible in the future, add an explicit documented
coordinate model/API rather than loosening ordinary `f64` APIs.

## Delaunay-Facing Layer

`src/delaunay/` owns public Delaunay workflows:

- `builder.rs` - fluent builder API for Euclidean and toroidal/periodic
  construction. Explicit-connectivity construction delegates through
  `TdsBuilder` and `TriangulationBuilder` before Level 5 certification.
- `construction.rs` - batch construction options, errors, statistics, and
  high-level constructors. Its private `DelaunayBatchWorkspace` owns mutable
  construction caches and repair state until it forms the final Level 5 draft.
- `draft.rs` - private, mutation-free `DelaunayTriangulationDraft`. It stores
  exactly a Levels 1–4 `Triangulation` plus publication metadata while the
  single Level 5 promotion is attempted.
- `incremental_builder.rs` - public `DelaunayIncrementalBuilder` workflow.
  Its private `DelaunayBootstrapWorkspace` owns an
  unpublished `TdsDraft` plus kernel/topology context; the first maximal simplex
  crosses Levels 1–5 failure-atomically without remapping staged vertex keys,
  and later mutations operate on a verified owner before the checked `finish()`
  boundary returns it.
- `insertion.rs` - post-construction vertex insertion and repair orchestration;
  published empty owners reject bootstrap insertion, which belongs to
  `DelaunayIncrementalBuilder`.
- `deletion.rs` - post-construction vertex deletion errors and transactional
  rollback-facing API support.
- `query.rs` - read-only `DelaunayTriangulation` accessors and traversal
  helpers.
- `model.rs` - Level 5 `DelaunayTriangulation` storage type and insertion-state
  cache.
- `delaunayize.rs` - `DelaunayRefinementBuilder`, the sole consuming promotion
  from a Levels 1–4 `Triangulation` to a Levels 1–5
  `DelaunayTriangulation`. Strict mode certifies only Level 5; flip-repair mode
  is transactional and optionally rebuilds. Raw-TDS PL-manifold repair remains
  an orthogonal core transformation before Levels 1–4 restoration.
- `repair.rs` - Delaunay repair policies, rebuild config, and repair outcomes.
- `serialization.rs` - versioned owner-level persistence that stores the
  canonical `Tds` as an embedded CBOR byte image plus topology
  guarantee, global topology, validation policy, and a versioned scientific
  integrity manifest. It owns bounded envelope parsing, typed load/migration
  APIs, canonical UUID-ordered streaming hashes with map-only buffering,
  owner-bound reuse of Level-3 f-vector/Euler evidence, and independently
  replayed D4/D5 Euclidean construction proof before Levels 3–5 restoration.
- `spherical.rs` - bounded `S^2`/`S^3` construction,
  realization-validation, and empty-cap Delaunay backend using the topology
  space coordinate/metric backend.
- `validation.rs` - implemented Level 5 Geometric Predicate APIs, validation
  errors, reports, and reusable checks consumed by the private draft. Fast-path
  certificates are created only by the batch workspace in `construction.rs`.
- `property_validation.rs` - TDS-level Delaunay empty-circumsphere scans and
  repair-oriented violation reports used by Level 5 validation APIs.

`src/lib.rs` wires public modules, root re-exports, focused preludes, and the
crate-level documentation map. Public workflow modules are exposed directly as
`delaunay::builder`, `delaunay::construction`, `delaunay::flips`,
`delaunay::checkpoint`, `delaunay::incremental_builder`, `delaunay::pachner`,
`delaunay::repair`, `delaunay::validation`, and focused preludes rather than through a nested
`delaunay::delaunay` facade. The physical
location of `flips` and `pachner` under `src/triangulation/` records that these
operations require only the Levels 1–4 owner.

## I/O And Export Layer

`src/io/` owns public downstream-facing export data models:

- `visualization.rs` - generic simplicial-complex primitives for notebooks,
  visualization tools, analysis pipelines, and downstream crates.

This layer is distinct from the TDS snapshot/hydration boundary. TDS serde
persists the Levels 1–2 owner; Delaunay serde wraps that snapshot with the
higher-layer proof context needed for validated restoration.
`io::visualization` exposes stable UUID-based records for consumers that should
not depend on runtime slotmap handles.

## Optional Binaries

The opt-in `cli` feature builds two repository-facing binaries without adding
command-line dependencies to the default library feature set:

- `src/bin/delaunay/main.rs` owns only artifact-process startup and exit status.
- `src/bin/delaunay/config.rs` owns clap parsing, semantic validation, dispatch,
  and the artifact binary's typed error boundary.
- `src/bin/delaunay/generate.rs`, `spherical_hero.rs`, and
  `validation_demo.rs` each own one direct artifact subcommand's raw arguments,
  validated configuration, runner, and stable exported schema.
- `src/bin/pachner_stress/main.rs` owns the independent maintainer diagnostic,
  including its parser, validated configuration, same-file output conflict
  checks, telemetry, and runner.
- `src/bin/shared/cli_output.rs` owns the two binaries' shared validated
  destinations and path-rich serialization/output errors;
  `src/bin/delaunay/cli_output.rs` owns the shared component's unit tests.

The binary target is the workflow namespace: `delaunay` exposes `generate`,
`spherical-hero`, and `validation-demo` directly, while `pachner-stress` accepts
its diagnostic options directly. There are no redundant `artifact` or
`diagnose` wrapper subcommands.

Raw paths remain `PathBuf` values at the clap boundary and become
`ArtifactPath` values during semantic validation. This keeps filesystem
identity and destination errors out of parsing while preventing distinct path
spellings from selecting one output file. Machine-readable command-owned JSON
objects carry both `schema` and `schema_version`; Cargo-backed notebook,
diagnostic, and `just run` workflows execute with the repository lockfile.

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
  `S^D` realized in `R^(D+1)`.
- `traits/topological_space.rs` - public `GlobalTopology<D>` metadata enum and
  `TopologyKind`.
- `traits/global_topology_model.rs` - internal scalar-generic
  `GlobalTopologyModel<D>` trait and concrete topology models.

Proof-owner dependencies should remain
`core::tds <- triangulation <- delaunay`; lower proof layers must not import
higher proof owners.

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
