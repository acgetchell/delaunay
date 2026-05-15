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

For D ≥ 5, predicate evaluation falls through to exact arithmetic more often;
for D ≥ 6, insphere classification relies on symbolic perturbation and
deterministic tie-breaking rather than exact insphere determinants.

## Numerical Robustness

The default `AdaptiveKernel` is the recommended default. It combines a
provable f64 fast filter, exact Bareiss determinant signs, and SoS
tie-breaking. Use `RobustKernel` when you need explicit
`BOUNDARY`/`DEGENERATE` signals instead of SoS resolving ties.

`FastKernel` is raw floating-point arithmetic. It is useful for exploratory
well-conditioned 2D work and for low-level tests, but it does not implement
`ExactPredicates` and cannot call the explicit public repair APIs.

See [`numerical_robustness_guide.md`](numerical_robustness_guide.md) for kernel
selection, duplicate handling, exact predicate details, and retry semantics.

## Topology and Domain Limits

Euclidean construction is the default and best-covered path.

Toroidal support has two modes:

- `.toroidal([..])` canonicalizes vertices into the fundamental domain, then
  builds the Euclidean Delaunay triangulation of the wrapped point set. It does
  not identify opposite boundary facets.
- `.toroidal_periodic([..])` uses the 3^D image-point method to construct a
  true periodic quotient with rewired neighbor pointers. This path is strongest
  in 2D. Compact 3D smoke coverage is active, while full 3D validation and
  4D/5D periodic validation are ignored slow tests because image expansion grows
  quickly with dimension.

Spherical and hyperbolic topologies are public metadata and behavior-model
scaffolds today. They do not yet provide full construction or validation
semantics for points constrained to a sphere or hyperbolic model.

Manual topological editing APIs are intentionally low level. After bistellar
flips or direct TDS-oriented work, callers should run repair and validation
before relying on the Delaunay property.

## Large-Scale Behavior

The historical 3D 35/1000-point and 4D 100/500-point seeded correctness
reproducers have been rechecked and fixed. The later 4D 3000-point work is now
a runtime/observability characterization target rather than a known correctness
failure. Details are archived in
[`archive/known_issues_4d_2026-04-23.md`](archive/known_issues_4d_2026-04-23.md)
and [`archive/issue_204_investigation.md`](archive/issue_204_investigation.md).

The current 2D–5D large-scale debug envelope is an operational baseline for
release characterization, not a portable performance promise:

- `just debug-large-scale-{2,3,4,5}d [n] [repair_every]` runs the same
  release-mode ignored harness shape across dimensions: deterministic point
  generation, batch construction, final flip repair, and `validation_report`
  for Levels 1–4.
- The `just` helper defaults are dimension-aware rather than identical: 2D
  defaults to 36,000 vertices, 3D defaults to 8,000 vertices, 4D defaults to
  900 vertices, and 5D defaults to 140 vertices. Pass `n` explicitly when a
  run must match a documented scale exactly.
- The raw ignored tests use slightly heavier defaults for some dimensions
  (currently 40,000 vertices in 2D and 150 vertices in 5D). Prefer the `just`
  helpers for routine acceptance/profiling runs.

Current 2D scale envelope:

- `just debug-large-scale-2d` defaults to 36,000 vertices for the
  near-one-minute 2D path.
- `just debug-large-scale-2d 40000 1` is the current heavier release-mode
  acceptance probe for the 40,000-vertex 2D path.
- The 2026-05-14 local run inserted all 40,000 vertices with zero skips, final
  repair performed 0 flips, and `validation_report` passed in about 66 seconds
  total wall time.

Current 3D scale envelope:

- `just debug-large-scale-3d 8000 1` is the current release-mode acceptance
  harness for the 8,000-vertex 3D path.
- Recent maintainer-hardware runs insert all 8,000 vertices with zero skips,
  run a clean final flip repair, and pass `validation_report` for Levels 1–4.
- Wall time is hardware- and load-sensitive. Recent Apple M4 Max-class local
  runs complete in roughly 56 seconds; treat that as an envelope, not a
  portable guarantee.
- `just debug-large-scale-3d 10000 1` is a heavier characterization probe that
  has also passed final Levels 1–4 validation; use it when the 10,000-vertex
  envelope matters more than one-minute feedback.

Current 4D scale envelope:

- `just debug-large-scale-4d 900 1` is the current release-mode acceptance
  harness for the 900-vertex 4D path. The 2026-05-14 local run inserted all
  900 vertices with zero skips, final repair performed 0 flips, and
  `validation_report` passed in about 60 seconds total wall time.
- `just debug-large-scale-4d 3000 1` remains a manual characterization scale for
  #340 rather than a default acceptance run.
- Keep 4D thousands-point runs out of routine CI unless they are reduced to a
  bounded fixture; use release mode and the large-scale debug harness when
  characterizing this regime.
- Exact predicates can dominate runtime on near-degenerate inputs. Improving
  adaptive predicate performance is tracked separately from correctness.

Current 5D scale envelope:

- `just debug-large-scale-5d` defaults to 140 vertices for the near-one-minute
  5D path.
- `just debug-large-scale-5d 150 1` is the current heavier practical 5D
  acceptance probe. The 2026-05-14 local run inserted all 150 vertices with
  zero skips, final repair performed 0 flips, and `validation_report` passed in
  about 62 seconds total wall time.
- The 50-vertex 5D probe remains useful for quick local checks and measured
  about 7 seconds as a single release-mode debug run; Criterion uses an even
  smaller 25-vertex canary so repeated samples stay practical.
- `just debug-large-scale-5d 1000 1` remains the #342 feasibility target. On
  2026-05-14 it exceeded the 1800-second manual harness timeout before the
  construction summary completed, so do not treat 1000-point 5D as a default
  acceptance scale until the shared high-dimensional bottleneck is reduced.

For reproducible large-scale diagnostics, see
[`dev/debug_env_vars.md`](dev/debug_env_vars.md).

## Feature Gaps

These are not currently implemented:

- Constrained Delaunay triangulations.
- Voronoi diagram extraction.
- Built-in visualization.
- Massively parallel, GPU, or out-of-core construction.
- Full spherical or hyperbolic triangulation semantics.

The roadmap for active follow-up work is in [`roadmap.md`](roadmap.md).
