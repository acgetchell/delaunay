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
- **All files in `docs/dev/*.md`** - focused repository development rules
- `docs/code_organization.md` - short architecture hub; follow its focused
  links when touching module layout, public namespace, preludes, or file
  organization

`docs/dev/README.md` explains which focused document owns each development
workflow. Agents must load every file in `docs/dev/` before making changes.

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
- **Validate with the right command.** Non-test Rust changes require final
  `just ci`. Rust unit-test-only, integration-test-only, and benchmark-only
  changes use the matching focused validators from `docs/dev/commands.md`.
  Docs/config changes use `just check`; Python-only changes use
  `just python-check`.
- **Do not edit generated changelogs manually.** Changelog and documentation
  maintenance rules live in `docs/dev/docs.md`.

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

- Every mutating operation preserves the invariants checked by `Tds::is_valid`
  (Levels 1-3) and `DelaunayTriangulation::is_valid` (Level 4). An operation
  that cannot preserve them must fail explicitly rather than leave inconsistent
  state behind.
- PL-manifold invariants: facets have multiplicity 1 (boundary) or 2
  (interior), ridges are linked consistently, and Euler characteristic matches
  the triangulation's `TopologyGuarantee`.
- Repair paths (`repair_delaunay_with_flips`, `repair_facet_oversharing`,
  `delaunayize_by_flips`) bound work through explicit budgets and surface
  non-convergence as typed errors.

### Validation Layers

The library exposes four validation levels, each a superset of the last:

1. **Level 1 - elements**: individual simplices, vertices, and facets are
   internally consistent.
2. **Level 2 - structure**: adjacency pointers and neighbor links form a valid
   incidence graph.
3. **Level 3 - topology**: PL-manifold-with-boundary, Euler characteristic, and
   ridge-link consistency.
4. **Level 4 - Delaunay property**: every facet is locally Delaunay.

Only Level 4 requires predicate evaluation. Levels 1-3 are pure graph checks.
Validation code belongs at the lowest layer that owns the invariant.

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
- Performance is subordinate to correctness. Use the benchmark-before/after
  workflow in `docs/dev/perf-tuning.md` for performance-sensitive changes.
- Tests should verify mathematical, geometric, and topological invariants.
  Dimension-generic tests should cover 2D through 5D whenever feasible.

## Where Detailed Rules Live

| Need | Read |
|-----|-----|
| Development guidance index | `docs/dev/README.md` |
| Git, GitHub CLI, issue dependencies, branch names, commit messages | `docs/dev/git.md` |
| Validation command selection and `just` recipes | `docs/dev/commands.md` |
| Rust API, naming, error, panic, prelude, and implementation conventions | `docs/dev/rust.md` |
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
- Keep code simple and maintainable when multiple correct solutions exist.
- Preserve numerical and topological invariants first; optimize only inside
  that envelope.
