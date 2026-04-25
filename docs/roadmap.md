# Roadmap

This page tracks current follow-up work at a high level. Historical task
snapshots live in [`archive/`](archive/).

## Correctness and Robustness

- **Predicate performance (#256):** the provable fast filters correctly route
  many hard cases to exact Bareiss arithmetic. A future adaptive expansion layer
  could reduce exact-path frequency without weakening correctness.
- **Selective property-test restoration:** re-enable the fastest disabled
  near-degenerate proptests once predicate performance allows them to run within
  CI budgets.

## Performance

- **3D construction profiling (#310):** profile construction and query hot paths
  before making targeted optimizations.
- **4D large-scale characterization (#204):** characterize the 3000-point
  release-mode debug harness and decide whether that scenario belongs in
  automated regression coverage or remains a manual investigation recipe.

## API and Documentation

- **Doctest migration to builder examples (#214):** prefer
  `DelaunayTriangulationBuilder` in examples that need explicit configuration,
  while keeping `DelaunayTriangulation::new` for the minimal path.
- **Prelude guidance:** keep examples aligned with the focused preludes
  (`triangulation`, `triangulation::repair`, `triangulation::flips`,
  `geometry`, `query`, `collections`, and `topology::validation`).
- **Builder decomposition:** reduce large helper functions in `builder.rs`
  where doing so clarifies topology or selection logic.

## Out of Scope for Now

- Constrained Delaunay triangulations.
- Voronoi diagram extraction.
- Built-in visualization.
- Massively parallel, GPU, or out-of-core meshing.

See [`limitations.md`](limitations.md) for current operational limits and
[`archive/todo_2026-04-23.md`](archive/todo_2026-04-23.md) for the retired
post-v0.7.5 task snapshot.
