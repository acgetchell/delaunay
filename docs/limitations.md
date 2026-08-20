# Limitations and Scope

This page summarizes the current operational limits of the `delaunay` crate.
Historical investigations and closed reproducers live in [`archive/`](archive/).

## Intended Use

`delaunay` is designed for finite point sets in Euclidean space, with optional
toroidal construction through the builder APIs. Its strongest test and
benchmark coverage is in 2D through 5D. The default `AdaptiveKernel` is the
recommended kernel for scientific work because it uses exact predicates and
deterministic Simulation of Simplicity (SoS) tie-breaking.

Use the crate when you need:

- Dimension-generic Delaunay triangulations and convex hull snapshots in Rust.
- Explicit validation of element, structural, topological, and Delaunay
  invariants.
- Deterministic construction controls for regression tests and experiments.
- PL-manifold-aware editing, flip-based Delaunay repair, and Euclidean or
  toroidal construction experiments.

Consider a specialized meshing tool instead when you need constrained
Delaunay triangulations, direct Voronoi extraction, out-of-core meshing,
GPU/parallel meshing, or production-scale dynamic remeshing.

## Dimension and Predicate Limits

| Dimension | Status |
|---|---|
| 2D | Primary supported path; broad unit, integration, property, and large-scale coverage. |
| 3D | Supported and covered; current large-scale acceptance uses thousands of vertices. |
| 4D | Supported and covered; large batch runs are exact-predicate-heavy and should use release mode. |
| 5D | Supported, but high cost; exact insphere still fits the stack matrix limit, but most predicate calls take the exact path. |
| 6D+ | Experimental. Exact orientation remains available through D=6, but exact insphere does not; routine construction coverage stops below this range. |

The stack-allocated exact determinant path supports matrices up to 7×7:

- f64 fast filter: D ≤ 4, because `det_errbound()` is unavailable above 4D.
- Exact orientation: D ≤ 6, because orientation uses a `(D + 1) × (D + 1)`
  determinant.
- Exact insphere: D ≤ 5, because insphere uses a `(D + 2) × (D + 2)`
  determinant.

For D ≥ 5, predicate evaluation falls through to exact arithmetic more often.
For D ≥ 6, exact insphere determinants are unavailable: classification first
uses the floating-point circumcenter/radius distance predicate, then applies
symbolic perturbation only when that predicate reports a boundary or fails.
This fallback is deterministic but does not provide exact-sign protection for
near-degenerate D ≥ 6 inputs.

## Numerical Robustness

The default `AdaptiveKernel` is the recommended default. It combines a
provable f64 fast filter, exact Bareiss determinant signs, and SoS
tie-breaking. Use `RobustKernel` when you need explicit
`BOUNDARY`/`DEGENERATE` signals instead of SoS resolving ties.

`FastKernel` is the lean filtered-exact policy: it preserves explicit
`BOUNDARY`/`DEGENERATE` results without `RobustKernel`'s opt-in diagnostic
cross-check or higher-dimensional fallback. It implements `ExactPredicates`
through D ≤ 5 and can use the explicit public repair APIs in those dimensions.

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

Manual topological editing APIs are intentionally low level. After bistellar
flips or direct TDS-oriented work, callers should run repair and validation
before relying on the Delaunay property.

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
workload because of runtime, while 5D acceptance uses much smaller bounded
fixtures and 1,000-point feasibility remains tracked by
[#342](https://github.com/acgetchell/delaunay/issues/342). These fixture sizes
describe current test and profiling coverage, not maximum supported input
sizes.

For reproducible diagnostic controls, see
[`dev/debug_env_vars.md`](dev/debug_env_vars.md).

## Feature Gaps

These are not currently implemented:

- Constrained Delaunay triangulations.
- Voronoi diagram extraction.
- Built-in visualization.
- Multi-threaded construction, proposal coordination, or concurrent topology
  mutation APIs. Runtime owner/generation provenance exists for caches and
  detached Pachner proposals, but parallel execution still requires a dedicated
  synchronization and transaction design.
- Massively parallel, GPU, or out-of-core construction.
- Full spherical integration beyond the bounded `S^2`/`S^3` prototype, or
  hyperbolic triangulation semantics.

The roadmap for active follow-up work is in [`roadmap.md`](roadmap.md).
