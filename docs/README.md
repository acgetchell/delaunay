# Documentation

This directory contains current (non-historical) documentation for the `delaunay` crate.

Historical design notes, investigations, and completed optimization roadmaps live under
`docs/archive/`.

## Start here

- [`api_design.md`](api_design.md): construction, vertex lifecycle, and Pachner move APIs.
- [`topology.md`](topology.md): Level 3 topology invariants (manifold checks, Euler characteristic).
- [`validation.md`](validation.md): the four-level validation model (Levels 1–4) and how to configure it.
- [`diagnostics.md`](diagnostics.md): opt-in diagnostic helpers, structured reports, and debug switches.
- [`mesh_export.md`](mesh_export.md): stable simplicial-complex export schema for notebooks and downstream tools.
- [`workflows.md`](workflows.md): practical recipes for construction, deletion, and local Pachner moves.
- [`limitations.md`](limitations.md): supported dimensions, predicate limits, large-scale cautions, and feature gaps.

## Reference guides

- [`code_organization.md`](code_organization.md): short architecture hub.
- [`architecture/`](architecture/): focused project-structure, module-map,
  prelude, and module-pattern references.
- [`invariants.md`](invariants.md): theoretical background and rationale for the topological and geometric invariants.
- [`numerical_robustness_guide.md`](numerical_robustness_guide.md): robustness strategies, kernels, and retry/repair behavior.
- [`property_testing_summary.md`](property_testing_summary.md): property-based testing with proptest (where tests live, how to run).
- [`../benches/README.md`](../benches/README.md): benchmark suites, perf-profile workflow, release summaries, and canary sizes.
- [`RELEASING.md`](RELEASING.md): release workflow (changelog + benchmarks + publish).
- [`roadmap.md`](roadmap.md): current follow-up work and deferred features.

## Templates and archive

- `archive/`: historical documentation only, including completed investigations and retired task snapshots.
- `templates/`: tooling templates (e.g., changelog templates).
