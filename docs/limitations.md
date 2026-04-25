# Limitations and Scope

This page summarizes the current operational limits of the `delaunay` crate.
Historical investigations and closed reproducers live in [`archive/`](archive/).

## Intended Use

`delaunay` is designed for finite point sets in small-to-medium dimensions,
with the strongest test coverage in 2D through 5D. The default
`AdaptiveKernel` is the recommended kernel for scientific work because it uses
exact predicates and deterministic Simulation of Simplicity (SoS) tie-breaking.

Use the crate when you need:

- Dimension-generic Delaunay triangulations and convex hulls in Rust.
- Explicit validation of element, structural, topological, and Delaunay
  invariants.
- Deterministic construction controls for regression tests and experiments.
- PL-manifold-aware editing and flip-based Delaunay repair.

Consider a specialized meshing tool instead when you need constrained
Delaunay triangulations, out-of-core meshing, GPU/parallel meshing, or
production-scale dynamic remeshing.

## Dimension and Predicate Limits

| Dimension | Status |
|---|---|
| 2D | Primary supported path; broad unit, integration, and property coverage. |
| 3D | Supported and covered; use `AdaptiveKernel` or `RobustKernel` for research workflows. |
| 4D | Supported and covered; large batch runs are more expensive and should use release mode. |
| 5D | Supported, but higher cost; exact insphere predicates still fit the stack matrix limit. |
| 6D+ | Experimental. Exact orientation remains available through D=6, but exact insphere does not. |

The stack-allocated exact determinant path supports matrices up to 7×7:

- Exact orientation: D ≤ 6, because orientation uses a `(D + 1) × (D + 1)`
  determinant.
- Exact insphere: D ≤ 5, because insphere uses a `(D + 2) × (D + 2)`
  determinant.

For D ≥ 6, insphere classification relies on symbolic perturbation and
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

## Large-Scale Behavior

The historical 3D 35/1000-point and 4D 100/500-point seeded correctness
reproducers have been rechecked and fixed. Their details are archived in
[`archive/known_issues_4d_2026-04-23.md`](archive/known_issues_4d_2026-04-23.md)
and [`archive/issue_204_investigation.md`](archive/issue_204_investigation.md).

Remaining large-scale caution:

- 4D batch construction at thousands of points can be expensive to investigate.
  Use release mode and the large-scale debug harness when characterizing this
  regime.
- Exact predicates can dominate runtime on near-degenerate inputs. Improving
  adaptive predicate performance is tracked separately from correctness.

For reproducible large-scale diagnostics, see
[`dev/debug_env_vars.md`](dev/debug_env_vars.md).

## Feature Gaps

These are not currently implemented:

- Constrained Delaunay triangulations.
- Voronoi diagram extraction.
- Built-in visualization.
- Massively parallel, GPU, or out-of-core construction.

The roadmap for active follow-up work is in [`roadmap.md`](roadmap.md).
