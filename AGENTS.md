# AGENTS.md

Essential guidance for AI assistants working in this repository.

This file is the **entry point for all coding agents**. It should stay short
enough to load and remember. Detailed operational rules are indexed in
`docs/dev/README.md`; focused architecture references are indexed from
`docs/code_organization.md`.

---

## Contents

- [Required Reading](#required-reading)
- [Non-Negotiable Rules](#non-negotiable-rules)
- [Project Context](#project-context)
- [Design Priority](#design-priority)
- [Scientific Invariants](#scientific-invariants)
  - [Numerical Correctness](#numerical-correctness)
  - [Topological Correctness](#topological-correctness)
  - [Validation Layers](#validation-layers)
  - [Symbolic Perturbation](#symbolic-perturbation)
- [Design Principles](#design-principles)
- [Where Detailed Rules Live](#where-detailed-rules-live)
- [Agent Expectations](#agent-expectations)

---

## Required Reading

Before modifying code, agents MUST read:

- `AGENTS.md` (this file)
- `docs/dev/README.md` - development guidance index
- **All top-level files in `docs/dev/*.md`** - focused repository development
  rules and routing indexes
- `docs/code_organization.md` - short architecture hub; follow its focused
  links when touching module layout, public namespace, preludes, or file
  organization

`docs/dev/README.md` explains which focused document owns each development
workflow. Agents must load every top-level file in `docs/dev/` before making
changes, then follow any workflow-specific links to deeper focused references
for the files they will touch.

## Non-Negotiable Rules

- **Do not mutate version-control state** unless the maintainer explicitly asks.
  Never run `git commit`, `git push`, `git tag`, or ref/index-mutating Git
  commands by default. Use `git --no-pager` for read-only Git commands.
  Detailed Git/GitHub rules live in `docs/dev/git.md`.
- **Use patch editing for manual edits.** Never use `sed`, `awk`, `perl`, or
  Python to modify code. Shell text tools are fine for read-only analysis.
- **Do not revert user changes.** The worktree may be dirty; preserve unrelated
  changes and work with any overlapping edits.
- **Unsafe Rust is forbidden.** The crate enforces `#![forbid(unsafe_code)]`.
- **Validate with the right command.** Core Rust changes require final
  `just ci`. Documentation, configuration, Python, notebook, Rust
  unit-test-only, doctest-only, integration-test-only, benchmark-only, and
  example-only changes use the matching focused validators from
  `docs/dev/commands.md`; compose those validators once each when multiple
  focused surfaces changed.
- **Do not edit generated changelogs manually.** Changelog and documentation
  maintenance rules live in `docs/dev/docs.md`.
- **Treat paper prose as author-owned.** Agents must not add substantive
  publication prose to `papers/`; local paper maintenance rules live in
  `docs/dev/docs.md`.
- **Keep README and citation prose mirrored.** The first paragraph under
  `README.md`'s Introduction is mirrored by the `abstract` field in
  `CITATION.cff`; update both together. The invariant is checked by
  `scripts/tests/test_readme_citation_mirror.py`.

## Project Context

- **Language**: Rust
- **Project**: d-dimensional Delaunay triangulation library
- **MSRV**: 1.96.0
- **Edition**: 2024
- **Primary architecture hub**: `docs/code_organization.md`

## Design Priority

This is a scientific d-dimensional Delaunay triangulation library. Design
decisions trade off in this order:

```text
numerical correctness -> topological correctness -> API stability ->
composability -> idiomatic Rust -> performance within scope
```

When in doubt, favor the invariant over the convenient edit.

## Scientific Invariants

### Numerical Correctness

- Geometric predicates (`insphere`, `insphere_lifted`, `orientation`) use the
  three-stage pattern from `src/geometry/predicates.rs`:
  - Stage 1: provable f64 fast filter with a Shewchuk-style error bound.
  - Stage 2: exact sign via Bareiss in `la-stack`.
  - Stage 3: deterministic `InSphere::BOUNDARY` /
    `Orientation::DEGENERATE` fallback for non-finite inputs.
- No f64 operation may silently lose sign information. Avoid patterns such as
  `unwrap_or(NaN)`, `unwrap_or(f64::INFINITY)`, or "return `true` on error."
- Algorithms cite their source in `REFERENCES.md` and document conditioning
  behavior.
- When two predicate implementations answer the same question, property tests
  verify agreement on the domain where both are defined.

### Topological Correctness

- Every mutating operation preserves the invariants checked by
  `Tds::is_valid` / `validate` (Levels 1-2),
  `Triangulation::is_valid_topology` / `validate` (Level 3),
  `Triangulation::is_valid_embedding` / `validate_embedding` (Level 4), and
  `DelaunayTriangulation::is_valid_delaunay` / `validate` (Level 5). An
  operation that cannot preserve them must fail explicitly rather than leave
  inconsistent state behind.
- PL-manifold invariants: facets have multiplicity 1 (boundary) or 2
  (interior), ridges are linked consistently, and Euler characteristic matches
  the triangulation's `TopologyGuarantee`.
- Repair paths (`repair_delaunay_with_flips`, `repair_facet_oversharing`,
  `delaunayize_by_flips`) bound work through explicit budgets and surface
  non-convergence as typed errors.

### Validation Layers

The library exposes five validation levels, each a superset of the last:

1. **Level 1 - Element Validity**: individual simplices, vertices, facets,
   coordinates, and canonical local data are internally consistent.
2. **Level 2 - Combinatorial Consistency**: adjacency pointers, incidence
   relations, neighbor links, simplex/ridge connectivity, and TDS integrity are
   coherent.
3. **Level 3 - Intrinsic PL Topology**: the abstract simplicial complex has the
   requested PL topology, including manifold/pseudomanifold conditions, links,
   Euler characteristic, orientability, and connected components.
4. **Level 4 - Embedding Validity**: the complex is faithfully realized in the
   chosen ambient model, including Euclidean affine charts, toroidal periodic
   charts, spherical `S^d` embeddings, and embedding-specific constraints.
5. **Level 5 - Geometric Predicates**: the embedding satisfies the selected
   geometric predicate family, currently Euclidean/toroidal/spherical Delaunay
   checks and eventually regular, weighted, constrained, Gabriel, alpha, or
   related predicates.

Level 4 uses orientation and exact barycentric geometry for affine-chart
embedding, with backend-specific realization checks for non-Euclidean models.
Level 5 uses geometry-specific predicates. Levels 1-3 are embedding-independent
graph/topology checks. Validation code belongs at the lowest layer that owns the
invariant.
Each layer should expose the standard validation surface. Use plain
`is_valid()` when the owner already names the invariant scope (`Vertex`,
`Simplex`, and `Tds`); use `is_valid_*` when higher-level owners expose
multiple validation layers. Use `*_diagnostic` for the first actionable
repair/retry diagnostic, `*_report` for layer-local aggregate diagnostics, and
`validate()` / `validation_report()` for cumulative roll-up through the owning
layer. Report names should identify the layer being checked, e.g.
`structure_report`, `topology_report`, `embedding_report`, and
`delaunay_report`. Higher layers should roll lower diagnostics up without
stringifying them.

### Symbolic Perturbation

Degenerate configurations (cospherical, collinear, coplanar inputs) are
resolved by Simulation of Simplicity in `src/geometry/sos.rs`. This is a
first-class invariant: callers can rely on predicates returning a consistent,
total ordering even on degenerate input. Tests under `tests/proptest_sos.rs`
enforce that contract.

## Design Principles

- Const-generic `D` belongs on every core type; do not introduce runtime
  dimension where the existing API is dimension-generic.
- Public APIs return typed `Result` for recoverable failures. Library code in
  `src/` must not panic on user input.
- Borrow by default and return lifetime-bound views where possible.
- Error values should be typed, orthogonal, and debuggable; avoid stringly
  failure modes.
- Focused preludes stay narrow and workflow-specific. Use
  `delaunay::prelude::pachner::*` for Pachner workflows; import primitive
  bistellar flips directly only for testing, benchmarking, or documenting that
  primitive layer.
- Pre-1.0 breaking changes are allowed when correctness, orthogonality, or
  performance require them, but they must be intentional and documented through
  the commit/changelog workflow.
- Performance is subordinate to scientific invariants. Use the
  benchmark-before/after workflow in `docs/dev/perf-tuning.md` for
  performance-sensitive changes, and treat timings from invariant-violating
  runs as invalid evidence.
- Tests should verify mathematical, geometric, and topological invariants.
  Dimension-generic tests should cover 2D through 5D whenever feasible.

## Where Detailed Rules Live

| Need | Read |
|-----|-----|
| Development guidance index | `docs/dev/README.md` |
| Git, GitHub CLI, issue dependencies, branch names, commit messages | `docs/dev/git.md` |
| Validation command selection and `just` recipes | `docs/dev/commands.md` |
| Rust API, naming, error, panic, prelude, and implementation conventions | `docs/dev/rust.md`, then its focused links |
| Tests, doctests, proptests, slow tests, and dimension coverage | `docs/dev/testing.md` |
| Performance tuning and benchmark evidence | `docs/dev/perf-tuning.md` |
| Documentation, references, changelog, and scientific notation | `docs/dev/docs.md` |
| Python support scripts | `docs/dev/python.md` |
| Debug environment variables | `docs/dev/debug_env_vars.md` |
| Tooling alignment and workflow/config rationale | `docs/dev/tooling-alignment.md` |
| Architecture hub and focused architecture links | `docs/code_organization.md` |

## Agent Expectations

- Prefer small, focused patches.
- Search `docs/`, `docs/dev/README.md`, and `docs/architecture/README.md`
  before inventing new conventions.
- Opportunistically fix nearby issues discovered while working in a touched
  area, even when they predate the current patch, when the fix is small,
  clearly related, and improves correctness, clarity, tests, or
  maintainability. Avoid broad mechanical churn; split repo-wide cleanup into
  separate work.
- Keep code simple and maintainable when multiple correct solutions exist.
- Preserve numerical and topological invariants first; optimize only inside
  that envelope.
