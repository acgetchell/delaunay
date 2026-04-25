# Documentation

This directory contains current (non-historical) documentation for the `delaunay` crate.

Historical design notes, investigations, and completed optimization roadmaps live under
`docs/archive/`.

## Start here

- [`api_design.md`](api_design.md): builder vs edit APIs (bistellar flips) and how to combine them.
- [`topology.md`](topology.md): Level 3 topology invariants (manifold checks, Euler characteristic).
- [`validation.md`](validation.md): the four-level validation model (Levels 1–4) and how to configure it.
- [`workflows.md`](workflows.md): practical recipes (Builder API vs Edit API), including minimal flip examples.
- [`limitations.md`](limitations.md): supported dimensions, predicate limits, large-scale cautions, and feature gaps.

## Reference guides

- [`code_organization.md`](code_organization.md): module layout and contributor-oriented orientation.
- [`invariants.md`](invariants.md): theoretical background and rationale for the topological and geometric invariants.
- [`numerical_robustness_guide.md`](numerical_robustness_guide.md): robustness strategies, kernels, and retry/repair behavior.
- [`property_testing_summary.md`](property_testing_summary.md): property-based testing with proptest (where tests live, how to run).
- [`RELEASING.md`](RELEASING.md): release workflow (changelog + benchmarks + publish).
- [`roadmap.md`](roadmap.md): current follow-up work and deferred features.

## Templates and archive

- `archive/`: historical documentation only, including completed investigations and retired task snapshots.
- `templates/`: tooling templates (e.g., changelog templates).
