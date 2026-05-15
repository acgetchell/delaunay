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

- **2D-5D shared large-scale monitoring (#340/#341/#342):** keep
  `just debug-large-scale-{2,3,4,5}d [n] [repair_every]` aligned so
  performance work is measured across the supported small-dimensional range
  instead of tuned for one dimension at another's expense. The current defaults
  are calibrated as roughly one-minute release-mode runs on maintainer hardware:
  2D=40,000, 3D=8,000, 4D=900, and 5D=150.
- **Criterion performance canaries:** keep smaller `ci_performance_suite`
  canaries for the same construction path so PR regression checks remain
  practical under Criterion's repeated sampling model.
- **4D large-scale monitoring (#204/#340):** keep the 3000-point release-mode
  debug harness as an optional manual investigation recipe; its multi-minute
  runtime is too large for routine CI.
- **5D feasibility (#342):** keep the 150-point release-mode harness as the
  current practical 5D baseline while optimizing toward the 1000-point target.
  The 200-vertex case is a useful heavier probe but currently sits closer to
  two minutes than one.

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
