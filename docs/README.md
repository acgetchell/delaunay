# Documentation

This directory contains current (non-historical) documentation for the `delaunay` crate.

Historical design notes, investigations, and completed optimization roadmaps live under
`docs/archive/`.

## Start here

- [`scientific_basis.md`](scientific_basis.md): API selection, scientific scope, assumptions, and method/source map.
- [`../papers/ARTIFACT.md`](../papers/ARTIFACT.md): v0.8.0 reviewer reproduction paths, claim map, evidence, and limits.
- [`../examples/README.md`](../examples/README.md): public-workflow coverage across runnable Rust examples and notebooks.
- [`api_design.md`](api_design.md): construction, vertex lifecycle, and Pachner move APIs.
- [`topology.md`](topology.md): Level 3 Intrinsic PL Topology invariants (manifold checks, orientability, Euler characteristic).
- [`construction_and_validation.md`](construction_and_validation.md): proof-bearing construction, the five validation levels, and their configuration.
- [`../papers/validation.pdf`](../papers/validation.pdf): reviewer-facing construction and validation architecture paper.
- [`diagnostics.md`](diagnostics.md): opt-in diagnostic helpers, structured reports, and debug switches.
- [`mesh_export.md`](mesh_export.md): stable simplicial-complex export schema for notebooks and downstream tools.
- [`USING_TRIANGULATIONS.md`](USING_TRIANGULATIONS.md): practical recipes for construction, deletion, and local Pachner moves.
- [`limitations.md`](limitations.md): supported dimensions, predicate limits, large-scale cautions, and feature gaps.

## Reference guides

- [`code_organization.md`](code_organization.md): short architecture hub.
- [`architecture/`](architecture/): focused project-structure, module-map,
  prelude, and module-pattern references.
- [`invariants.md`](invariants.md): theoretical background and rationale for the topological and geometric invariants.
- [`numerical_robustness_guide.md`](numerical_robustness_guide.md): robustness strategies, kernels, and retry/repair behavior.
- [`orientation_spec.md`](orientation_spec.md): coherent combinatorial and geometric orientation specification.
- [`performance.md`](performance.md): retained release comparisons and their provenance qualifications.
- [`property_testing_summary.md`](property_testing_summary.md): property-based testing with proptest (where tests live, how to run).
- [`../benches/README.md`](../benches/README.md): benchmark suites, perf-profile workflow, release summaries, and canary sizes.
- [`RELEASING.md`](RELEASING.md): release workflow (changelog + benchmarks + publish).
- [`dev/TUNING-PERFORMANCE.md`](dev/TUNING-PERFORMANCE.md): benchmark-before/after workflow for performance changes.
- [`roadmap.md`](roadmap.md): current follow-up work and deferred features.

## Templates and archive

- `archive/`: historical documentation only, including completed investigations and retired task snapshots.
- `templates/`: tooling templates (e.g., changelog templates).
