# Roadmap

This page tracks current follow-up work at a high level. Historical task
snapshots live in [`archive/`](archive/).

## Release Sequence

### v0.7.8 pre-v0.8.0 cleanup (released)

The v0.7.8 line was the final pre-v0.8.0 cleanup baseline. It kept the
performance-summary work from v0.7.7, closes the release-facing documentation
and doctest hygiene pass, tightens default test-suite budgets, validates compact
`T^3` quotient construction, and restores failed topology-repair
workflows to their pre-call state when fallback rebuild does not succeed.

Key takeaways from v0.7.8:

- Default correctness tests now stay under the routine 10-second budget, with
  long-running correctness work routed through `slow-tests`.
- Public examples and doctests avoid release-hostile unwrap/expect patterns in
  favor of typed `Result` wrappers or non-degenerate examples.
- Compact `T^3` quotients are validated through topology and Delaunay
  checks before being returned; higher-dimensional quotient construction fails
  fast until scalable follow-up work lands.
- `delaunayize` now consumes a Levels 1–4 triangulation, keeps intermediate
  flip-repair state private, and publishes only a Levels 1–5 result; raw-TDS
  PL-manifold repair is an orthogonal pre-restoration workflow.
- The release benchmark summary remains the current public performance snapshot
  for construction throughput, generated simplex counts, and circumsphere
  predicate behavior.

### v0.8.0 paper-facing API and topology push (released)

v0.8.0 delivered the paper-facing API and topology work deferred from v0.7.8.
It requires Rust 1.97.1, including the completed audit of the 1.97.1 toolchain
surface.

- **Pachner/Edit API shape (#252/#253/#350/#337):** unified the Pachner move API,
  expanded public flip benchmark coverage, added Monte-Carlo stress benchmarks,
  and supported periodic external-simplex parity constraints in bistellar flips.
- **Linear algebra and API boundaries (#424):** kept `la-stack` details behind
  `src/geometry/matrix.rs`, preserved typed backend errors in geometry helpers,
  and kept README/API guidance aligned with the focused prelude reference in
  `docs/code_organization.md`.
- **Topology and incidence surface (#359/#304):** added stable incidence queries
  for simplex-local topology and dedicated targeted topology repair stages for
  ridge and vertex-link failures.
- **Transactional rollback architecture (#364):** replaced or centralized
  full-TDS clone rollback where benchmarks show it matters, while preserving
  the current strong failed-mutation rollback guarantee.
- **Naming cleanup (#323):** made the breaking `Cell` → `Simplex` rename.
- **Iterator cleanup (#353):** preferred iterator-based collection-building paths
  where that improves clarity and allocation behavior.
- **Rust 1.97.1 release gate (#329/#496):** raised the v0.8.0 MSRV to Rust
  1.97.1, finished the baseline `assert_matches!` cleanup, audited the new
  integer/`NonZero` bit helpers against Hilbert bit-depth/index invariants,
  reviewed `RepeatN::default` and Cargo 1.97 tooling changes for useful adoption,
  and re-benchmarked predicate `cold_path` decisions under the 1.97.1 compiler.
- **Notebook/export artifact (#64/#408):** shipped the generic simplicial-complex
  JSON export, reproducible quickstart and validation notebooks, tracked
  validation diagrams, and reviewer-facing artifact instructions.

### v0.8.1 validation evidence and performance follow-up

- **Level 4 realization validation (#482/#559):** completed the independent
  randomized and degenerate 2D–5D agreement campaign for the revised
  simplex-intersection narrow phase and added focused narrow-phase and
  whole-realization benchmarks.
- **Level 5 Delaunay reporting (#483/#560):** replaced the certified Euclidean
  all-vertices/all-simplices report bottleneck with an O(simplices) local-flip
  path while preserving exhaustive fallback diagnostics, with 2D–5D agreement
  tests and performance canaries.

### v0.9.0 and later horizon

v0.9.0 is the right parking lot for work that is valuable but larger or less
tightly coupled to the v0.8.0 paper/API push:

- **Broader geometry features (#299/#63/#136):** constrained Delaunay
  triangulations, Voronoi diagrams, and weakly-visible hull facets.
- **Built-in visualization and high-dimensional tuning (#106):** any native
  plotting layer and convex-hull buffer allocation work for D > 7.

## Ongoing Performance Monitoring

- **2D-5D shared large-scale monitoring (#340/#341/#342):** keep
  `just debug-large-scale-{2,3,4,5}d [n] [repair_every]` aligned so
  performance work is measured across the supported small-dimensional range
  instead of tuned for one dimension at another's expense. The current defaults
  are calibrated as roughly one-minute release-mode runs on maintainer hardware:
  2D=36,000, 3D=7,500, 4D=800, and 5D=140. Heavier explicit probes such as
  2D=40,000, 3D=10,000, and 5D=150 remain useful for release characterization.
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

## Not Implemented Today

- Constrained Delaunay triangulations.
- Voronoi diagram extraction.
- Built-in visualization.
- Massively parallel, GPU, or out-of-core meshing.

See [`limitations.md`](limitations.md) for current operational limits and
[`archive/todo_2026-04-23.md`](archive/todo_2026-04-23.md) for the retired
post-v0.7.5 task snapshot.
