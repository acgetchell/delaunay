# Roadmap

This page tracks current follow-up work at a high level. Historical task
snapshots live in [`archive/`](archive/).

## Release Sequence

### v0.7.8 pre-v0.8.0 cleanup (released)

The v0.7.8 line is the current pre-v0.8.0 cleanup baseline. It keeps the
performance-summary work from v0.7.7, closes the release-facing documentation
and doctest hygiene pass, tightens default test-suite budgets, validates compact
3D toroidal quotient construction, and restores failed topology-repair
workflows to their pre-call state when fallback rebuild does not succeed.

Key takeaways from v0.7.8:

- Default correctness tests now stay under the routine 10-second budget, with
  long-running correctness work routed through `slow-tests`.
- Public examples and doctests avoid release-hostile unwrap/expect patterns in
  favor of typed `Result` wrappers or non-degenerate examples.
- Compact 3D toroidal quotients are validated through topology and Delaunay
  checks before being returned; higher-dimensional quotient construction fails
  fast until scalable follow-up work lands.
- `delaunayize_by_flips` preserves the incoming triangulation on topology
  repair failure unless fallback rebuild succeeds.
- The release benchmark summary remains the current public performance snapshot
  for construction throughput, generated simplex counts, and circumsphere
  predicate behavior.

### v0.8.0 paper-facing API and topology push

v0.8.0 is the next feature-bearing release and is expected to carry the larger
work intentionally deferred from v0.7.8 cleanup. It will require Rust 1.97.0;
the final release gate is an explicit audit of the 1.97.0 toolchain surface
before shipping.

- **Pachner/Edit API shape (#252/#253/#350/#337):** unify the Pachner move API,
  expand public flip benchmark coverage, add Monte-Carlo stress benchmarks, and
  support periodic external-simplex parity constraints in bistellar flips.
- **Linear algebra and API boundaries (#424):** keep `la-stack` details behind
  `src/geometry/matrix.rs`, preserve typed backend errors in geometry helpers,
  and keep README/API guidance aligned with the focused prelude reference in
  `docs/code_organization.md`.
- **Topology and incidence surface (#359/#304):** add stable incidence queries
  for simplex-local topology and dedicated targeted topology repair stages for
  ridge and vertex-link failures.
- **Transactional rollback architecture (#364):** replace or centralize
  full-TDS clone rollback where benchmarks show it matters, while preserving
  the current strong failed-mutation rollback guarantee.
- **Naming cleanup (#323):** make the breaking `Cell` → `Simplex` rename.
- **Iterator cleanup (#353):** prefer iterator-based collection-building paths
  where that improves clarity and allocation behavior.
- **Rust 1.97.0 release gate (#329/#496):** raise the v0.8.0 MSRV to Rust
  1.97.0, finish the baseline `assert_matches!` cleanup, audit the new
  integer/`NonZero` bit helpers against Hilbert bit-depth/index invariants,
  review `RepeatN::default` and Cargo 1.97 tooling changes for useful adoption,
  and re-benchmark predicate `cold_path` decisions under the 1.97.0 compiler.

### v0.9.0 and later horizon

v0.9.0 is the right parking lot for work that is valuable but larger or less
tightly coupled to the v0.8.0 paper/API push:

- **Broader geometry features (#299/#63/#136):** constrained Delaunay
  triangulations, Voronoi diagrams, and weakly-visible hull facets.
- **Visualization and high-dimensional tuning (#64/#106):** built-in
  visualization and convex-hull buffer allocation work for D > 7.

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
