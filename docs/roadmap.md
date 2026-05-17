# Roadmap

This page tracks current follow-up work at a high level. Historical task
snapshots live in [`archive/`](archive/).

## Release Sequence

### v0.7.7 performance baseline (released)

The just-released v0.7.7 line closed a large amount of performance and
benchmark-documentation cleanup. It should be treated as the current baseline
for construction throughput, generated simplex counts, release benchmark
summaries, and the roughly one-minute large-scale debug envelopes.

Key takeaways from v0.7.7:

- The public `ci_performance_suite` contract now reports current Criterion run
  metadata and generated simplex counts instead of stale baseline-only context.
- The release summary flow highlights the user-facing construction path first,
  then the public API contract and circumsphere predicate data.
- The `just debug-large-scale-{2,3,4,5}d` helpers are calibrated across
  supported dimensions instead of focusing only on one scale target.
- Several previous performance pain points were reduced enough to make v0.7.8
  a cleanup release rather than another performance stabilization release.

### v0.7.8 pre-v0.8.0 cleanup

- **Rust-native tooling cleanup (#379):** plan the switch from the current
  Node-backed Markdown/YAML tooling to `pretty_yaml` plus a Rust Markdown
  linter, while also cleaning up `just` recipe naming and adding a Semgrep guard
  for check-before-fix ordering.
- **Predicate and slow-test cleanup (#256/#380):** continue the predicate
  performance work needed to re-enable ignored proptests, re-measure ignored
  tests, move unattended tests over the 30-second threshold into the
  `slow-tests` flow, and keep shorter ignored tests eligible for the default
  suite.
- **Module and prelude cleanup (#381):** rename the public high-level
  triangulation module layout to `delaunay`, split the current
  `triangulation/delaunay.rs` implementation into clearer components, and
  consolidate focused preludes while the API churn is still isolated from
  v0.8.0 feature work.
- **Documentation and doctest hygiene (#214/#365):** move configuration-heavy
  examples to `DelaunayTriangulationBuilder`, remove public doctest
  `.unwrap()`/`.expect(...)` patterns, and keep examples tied to typed errors.
- **Production-review API and hygiene (#382-#388):** resolve cfg-only feature
  labels, stale comments, safe-code invariant wording, UUID panic-helper scope,
  strict insphere test control, error enum clone/boxing policy, the
  topology-guarantee/validation-policy API split, skipped-insertion semantics,
  TDS neighbor/incidence encapsulation, and public `core` module naming.

### v0.8.0 paper-facing API and topology push

v0.8.0 is the next feature-bearing release and is expected to carry the larger
work that should not be tangled with v0.7.8 cleanup churn:

- **Pachner/Edit API shape (#252/#253/#350/#337):** unify the Pachner move API,
  expand public flip benchmark coverage, add Monte-Carlo stress benchmarks, and
  support periodic external-simplex parity constraints in bistellar flips.
- **Topology and incidence surface (#359/#304):** add stable incidence queries
  for simplex-local topology and dedicated targeted topology repair stages for
  ridge and vertex-link failures.
- **Transactional rollback architecture (#364):** replace or centralize
  full-TDS clone rollback where benchmarks show it matters, while preserving
  the current strong failed-mutation rollback guarantee.
- **Naming and spherical topology (#323/#188):** make the breaking
  `Simplex` → `Simplex` rename and implement unit-sphere normalization for
  `SphericalSpace::canonicalize_point()`.
- **Iterator cleanup (#353):** prefer iterator-based collection-building paths
  where that improves clarity and allocation behavior.

### v0.9.0 and later horizon

v0.9.0 is the right parking lot for work that is valuable but larger or less
tightly coupled to the v0.8.0 paper/API push:

- **Broader geometry features (#299/#63/#136):** constrained Delaunay
  triangulations, Voronoi diagrams, and weakly-visible hull facets.
- **Visualization and high-dimensional tuning (#64/#106):** built-in
  visualization and convex-hull buffer allocation work for D > 7.
- **Future Rust cleanup (#329):** adopt `assert_matches!` in tests once it is
  stable.

## Ongoing Performance Monitoring

- **2D-5D shared large-scale monitoring (#340/#341/#342):** keep
  `just debug-large-scale-{2,3,4,5}d [n] [repair_every]` aligned so
  performance work is measured across the supported small-dimensional range
  instead of tuned for one dimension at another's expense. The current defaults
  are calibrated as roughly one-minute release-mode runs on maintainer hardware:
  2D=36,000, 3D=8,000, 4D=900, and 5D=140. Heavier explicit probes such as
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
