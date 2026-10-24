# Limitations and Scope

This page summarizes the operational limits of the code being prepared for
v0.8.3. It distinguishes construction coverage, validated owner contracts, and
operations that require a particular geometric model.
Historical investigations and closed reproducers live in [`archive/`](archive/).

## Intended Use

`delaunay` provides general triangulations, Delaunay triangulations, and convex
hull snapshots over finite `f64` coordinates. Euclidean construction has the
strongest test and benchmark coverage in 2D through 5D. Toroidal construction
supports `T^2` and compact `T^3`, and a separate bounded spherical builder
supports `S^2`/`S^3`.

The default `AdaptiveKernel` uses exact predicates and deterministic Simulation
of Simplicity (SoS) tie-breaking in its supported dimension range.

Use the crate when you need:

- Deterministic construction controls for regression tests and experiments.
- Dimension-generic general/Delaunay triangulations and convex hull snapshots.
- Explicit element, combinatorial, topology, realization, and Delaunay checks.
- PL-manifold-aware editing, flip-based Delaunay repair, and Euclidean or
  toroidal construction experiments.

Consider a specialized meshing tool instead when you need constrained
Delaunay triangulations, direct Voronoi extraction, out-of-core meshing,
GPU/parallel meshing, or production-scale dynamic remeshing.

## Ownership and Operation Scope

| Owner | Cumulative contract | Intended operations |
|---|---|---|
| `Tds` | Levels 1–2: elements and connectivity. | Checked storage and combinatorial transitions. |
| `Triangulation` | Levels 1–4: adds intrinsic topology and valid realization. | General queries, geometry, persistence, and topology editing. |
| `DelaunayTriangulation` | Levels 1–5: adds Delaunay certification. | Construction and vertex maintenance that preserve the Delaunay property. |

`DelaunayTriangulationBuilder::build_triangulation()` stops at Levels 1–4;
`build()` also establishes Level 5. `TriangulationBuilder` certifies an existing
TDS through Levels 1–4. For Pachner editing, consume a Delaunay owner with
`into_triangulation()` and explicitly certify or repair Level 5 afterward with
`DelaunayRefinementBuilder`.

Construction support does not imply that every operation supports that model.
Point location (`locate` and `locate_with_stats`) currently requires Euclidean
geometry and returns `LocateError::UnsupportedTopology` for periodic owners.
General `Triangulation::insert_vertex()` and `delete_vertex()` also require
Euclidean geometry. Periodic editing uses chart-bound Pachner proposals, as
described below.

Persistence re-establishes the owning validation levels on load. It does not
trust stored construction provenance or repair invalid connectivity. Some
high-dimensional PL-manifold snapshots cannot be restored when their link
proof cannot be re-established. Payload codec restrictions and kernel selection
are documented in the [checkpoint and adapter contracts](construction_and_validation.md#level-4-adapter-workflows).

## Dimension and Predicate Limits

| Dimension | Status |
|---|---|
| 2D | Primary supported path; broad unit, integration, property, and large-scale coverage. |
| 3D | Supported and covered; current large-scale acceptance uses thousands of vertices. |
| 4D | Supported and covered; large batch runs are exact-predicate-heavy and should use release mode. |
| 5D | Supported with bounded fixtures; construction and exact fallback costs can be high. |
| 6D | Built-in exact predicates and SoS are available; construction coverage is more limited than in 2D–5D. |
| 7D+ | Outside the built-in point-driven construction and `ExactPredicates` contracts. |

The current geometric predicate adapters enforce the following limits:

- Orientation f64 fast filter: D ≤ 3, because `det_direct_with_errbound()`
  supports matrices up to 4×4 and orientation uses a `(D + 1) × (D + 1)` matrix.
- Relative in-sphere interval filter: D ≤ 6, using outward-rounded intervals
  in a `(D + 1) × (D + 1)` matrix. Uncertain intervals request exact arithmetic.
- Exact orientation: D ≤ 6, because orientation uses a `(D + 1) × (D + 1)`
  determinant.
- Exact insphere: D ≤ 6, because the translation-invariant relative formulation
  uses a `(D + 1) × (D + 1)` determinant. Its cold path forms differences and
  squared norms directly as exact rationals from the IEEE-754 inputs.

These are geometric predicate limits, not a general limit on all `la-stack`
matrix types or on the separate exact simplex-intersection adapter.
For D=4 through D=6, orientation goes directly to exact arithmetic, while the
relative in-sphere determinant can still be interval-certified. For D ≥ 7,
these predicate adapters do not provide an exact sign. The `robust_insphere` distance fallback,
also used by `RobustKernel::in_sphere`, does not carry the `ExactPredicates`
proof; `AdaptiveKernel` reports the unsupported dimension instead. Built-in
point-driven constructors require `ExactPredicates` and are supported through D=6.

## Numerical Robustness

The default `AdaptiveKernel` is the recommended default. It combines a
provable f64 fast filter, exact Bareiss determinant signs, and SoS
tie-breaking. Use `RobustKernel` when you need explicit
`BOUNDARY`/`DEGENERATE` signals instead of SoS resolving ties.

`FastKernel` is the lean filtered-exact policy: it preserves explicit
`BOUNDARY`/`DEGENERATE` results without `RobustKernel`'s opt-in diagnostic
cross-check or higher-dimensional fallback. It implements `ExactPredicates`
through D ≤ 6 and can use the explicit public repair APIs in those dimensions.

See [`numerical_robustness_guide.md`](numerical_robustness_guide.md) for kernel
selection, duplicate handling, exact predicate details, and retry semantics.

## Topology and Domain Limits

Euclidean construction is the default and best-covered path.

`.try_toroidal([..])` uses the 3^D image-point method to construct a periodic
quotient with rewired neighbor pointers. This path is release covered on `T^2`
and compact `T^3`, where periodic triangulations validate as closed tori through
Levels 1-5, including translation-normalized Level 2 stored-orientation
coherence and Level 3 intrinsic orientability. Construction moves each
canonical coordinate by at most about 2^-32 of its domain period using a
deterministic perturbation that resolves periodic covering-space degeneracies
while preserving vertex UUIDs and payloads.
`T^4`/`T^5` periodic construction fails fast until issue #416 makes quotient
selection scalable and diagnosable enough for release validation.

Spherical topologies provide public metadata and behavior-model support for
projecting finite nonzero coordinates onto the unit sphere. The bounded
`SphericalDelaunayBuilder` prototype additionally supports `S^2` and `S^3`
construction from points in `R^3`/`R^4` by ambient convex-hull duality. Its
validation surface keeps Level 3 Intrinsic PL Topology separate from spherical
Level 4 realization validation and spherical Level 5 empty-cap / hull-facet predicates. Full
`S^2`-`S^5` integration, richer spherical realization diagnostics, and
integration with the ordinary mutable triangulation/editing surface remain
tracked by issue #414. Hyperbolic topologies remain public metadata and
behavior-model scaffolds.

The spherical builder normalizes finite nonzero input vectors and requires
their ambient convex hull to enclose the origin. Its result is a separate
spherical owner; it does not expose the ordinary mutable `Triangulation`
editing workflow.

For periodic stellar insertion/deletion, use `PachnerMove::K1Insert` with a
selected simplex and `PachnerMove::K1Remove` for the inverse move. Other Pachner
moves can reshape a vertex star before removal. Arbitrary periodic cavity
deletion is not implemented by the general Euclidean fan routine.
Each checked move preserves its promised validation scope; a sequence of
separately committed moves is not one atomic transaction. Refine the edited
`Triangulation` explicitly before relying on a Level 5 Delaunay proof. See
[adapter workflows](construction_and_validation.md#level-4-adapter-workflows).

## Large-Scale Behavior

Large-scale construction is single-threaded and in-memory, and its cost rises
rapidly with dimension and exact-predicate frequency. The
`debug-large-scale-{2,3,4,5}d` recipes are release-mode acceptance and
profiling runs, not Criterion benchmarks or portable performance promises.
Current workloads, calibrated defaults, and maintainer-hardware timings live in
the [benchmark guide](../benches/README.md#release-mode-debug-defaults); the
`justfile` remains authoritative for command defaults.

Benchmark evidence is retained with its owning workflow rather than copied
here. [`../benches/PERFORMANCE_RESULTS.md`](../benches/PERFORMANCE_RESULTS.md)
is the committed generated snapshot, with prior snapshots preserved in Git
history. Release-to-release reports and raw Criterion data use the workflow
documented in the [benchmark guide](../benches/README.md#how-the-workflow-fits-together).
Dated large-scale correctness and performance investigations are preserved in
the [May 2026 characterization snapshot](archive/performance/large-scale-debug-characterization-2026-05.md),
with the underlying 3D/4D correctness investigations in
[`archive/known_issues_4d_2026-04-23.md`](archive/known_issues_4d_2026-04-23.md)
and [`archive/issue_204_investigation.md`](archive/issue_204_investigation.md).

The historical 3D and 4D seeded correctness reproducers described there have
been fixed. Thousands-point 4D construction remains a manual characterization
workload because of runtime, while 5D acceptance uses smaller bounded fixtures.
The completed [5D feasibility investigation #342](https://github.com/acgetchell/delaunay/issues/342)
is historical context, not an active implementation target or a guarantee that
1,000-point workloads are practical. Fixture sizes describe current test and
profiling coverage, not maximum supported input sizes.

For reproducible diagnostic controls, see
[`dev/debug_env_vars.md`](dev/debug_env_vars.md).

## Feature Gaps

These are not currently implemented:

- Constrained Delaunay triangulations.
- Multi-threaded construction, proposal coordination, or concurrent topology
  mutation APIs. Runtime owner/generation provenance exists for caches and
  detached Pachner proposals, but parallel execution still requires a dedicated
  synchronization and transaction design.
- Massively parallel, GPU, or out-of-core construction.
- Full spherical integration beyond the bounded `S^2`/`S^3` prototype, or
  hyperbolic triangulation semantics.
- Native Rust plotting or viewer APIs. Notebook visualization and UUID-based
  JSON export are available through the existing examples and CLI.
- Voronoi diagram extraction.

The roadmap for active follow-up work is in [`roadmap.md`](roadmap.md).
