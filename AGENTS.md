# AGENTS.md

Essential guidance for AI assistants working in this repository.

This file is the **entry point for all coding agents**. Detailed rules are
split into additional documents under `docs/dev/`. Agents MUST read the
referenced files before making changes.

---

## Contents

- [Required Reading](#required-reading)
- [Core Rules](#core-rules)
  - [Git Operations](#git-operations)
  - [GitHub CLI (`gh`)](#github-cli-gh)
  - [Code Editing](#code-editing)
  - [Commit Message Generation](#commit-message-generation)
- [Validation Workflow](#validation-workflow)
- [Project Context](#project-context)
- [Design Principles](#design-principles)
  - [Numerical correctness as an invariant](#numerical-correctness-as-an-invariant)
  - [Topological correctness as an invariant](#topological-correctness-as-an-invariant)
  - [Validation layering](#validation-layering)
  - [Symbolic perturbation (SoS)](#symbolic-perturbation-sos)
  - [Public‑API stability](#publicapi-stability)
  - [Composability](#composability)
  - [Idiomatic Rust as a proxy for mathematical clarity](#idiomatic-rust-as-a-proxy-for-mathematical-clarity)
  - [Scientific notation in docs](#scientific-notation-in-docs)
  - [Performance within scope](#performance-within-scope)
  - [Testing mirrors the principles](#testing-mirrors-the-principles)
- [Testing](#testing)
- [Documentation Maintenance](#documentation-maintenance)
- [Agent Behavior Expectations](#agent-behavior-expectations)

---

## Required Reading

Before modifying code, agents MUST read:

- `AGENTS.md` (this file)
- **All files in `docs/dev/*.md`** – repository development rules
- `docs/code_organization.md` – module layout and architecture

The `docs/dev/` directory contains the authoritative development guidance
for this repository. Agents must load every file in that directory before
making changes.

---

## Core Rules

### Git Operations

- **NEVER** run `git commit`, `git push`, `git tag`, or any git commands that modify version control state
- **ALLOWED**: read‑only git commands (`git --no-pager status`, `git --no-pager diff`, `git --no-pager log`, `git --no-pager show`, `git --no-pager blame`)
- **ALWAYS** use `git --no-pager` when reading git output
- When suggesting branch names, prefer `{type}/{issue}-descriptor-or-two`,
  e.g. `fix/307-oriented-flips`, `perf/315-bench-profile`, or
  `doc/329-branch-guidance`. If an environment requires an owner/tool prefix,
  keep this structure after the prefix, e.g. `codex/fix/307-oriented-flips`.
  Typical types are `fix`, `feat`, `perf`, `doc`, `test`, `refactor`, `ci`,
  `build`, `chore`, and `style`.
- Suggest git commands that modify version control state for the user to run manually

### GitHub CLI (`gh`)

When using the `gh` CLI to view issues, PRs, or other GitHub objects:

- **ALWAYS** use `--json` with `| cat` to avoid pager and scope errors:

  ```bash
  gh issue view 212 --repo acgetchell/delaunay --json title,body | cat
  ```

- To extract specific fields cleanly, combine `--json` with `--jq`:

  ```bash
  gh issue view 212 --repo acgetchell/delaunay --json title,body --jq '.title + "\n" + .body' | cat
  ```

- **AVOID** plain `gh issue view N` — it may fail with `read:project`
  scope errors or open a pager.

- To manage **issue dependencies** (Blocks / Is Blocked By), use the
  GitHub REST API via `gh api`. The endpoint requires the **internal
  issue ID** (not the issue number).

  To get an issue's internal ID:

  ```bash
  gh api repos/acgetchell/delaunay/issues/233 --jq '.id'
  ```

  To add a "blocked by" dependency (e.g. #254 is blocked by #233):

  ```bash
  gh api repos/acgetchell/delaunay/issues/254/dependencies/blocked_by \
    -X POST -F issue_id=<BLOCKING_ISSUE_ID>
  ```

  To list existing blocked‑by dependencies:

  ```bash
  gh api repos/acgetchell/delaunay/issues/254/dependencies/blocked_by \
    --jq '[.[].number]' | cat
  ```

  **Note**: Use `-F` (not `-f`) for `issue_id` so it is sent as an
  integer. The API returns HTTP 422 if the dependency already exists.

- When updating issues, use explicit `comment`/`edit` commands.
  For **arbitrary Markdown** (backticks, quotes, special characters),
  prefer `--body-file -` with a heredoc:

  ```bash
  gh issue comment 242 --repo acgetchell/delaunay --body-file - <<'EOF'
  ## Heading

  Body with `backticks`, **bold**, and apostrophes that's safe.
  EOF
  ```

  For **simple text only** (no apostrophes or special characters),
  single‑quoted `--body` is fine:

  ```bash
  gh issue comment 242 --repo acgetchell/delaunay --body 'Simple update text'
  ```

### Code Editing

- **NEVER** use `sed`, `awk`, `perl`, or `python` to modify code
- **ALWAYS** use the patch editing mechanism provided by the agent
- Shell text tools may be used for **read‑only analysis only**

### Commit Message Generation

When generating commit messages:

1. Run `git --no-pager diff --cached --stat`
2. Use conventional commits: `<type>: <summary>`
3. Valid types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`, `style`, `ci`, `build`
4. Include bullet‑point body describing key changes
5. Present inside a code block so the user can commit manually

#### Changelog‑Aware Body Text

Commit bodies appear **verbatim** in `CHANGELOG.md` (indented by
git‑cliff's template). Write them as clean, readable prose:

- Keep the **subject line** concise — it becomes the changelog entry.
- The **type** determines the changelog section (`feat` → Added,
  `fix` → Fixed, `refactor`/`test`/`style` → Changed, `perf` →
  Performance, `docs` → Documentation, `build`/`chore`/`ci` →
  Maintenance).
- Include **PR references** as `(#N)` in the subject — cliff auto‑links
  them (e.g. `feat: add foo (#42)`).
- **Avoid headings** `#`–`###` in the body — they conflict with
  changelog structure (`##` = release, `###` = section). Use `####` if
  a heading is truly needed.
- Body text should be **plain prose or simple lists**. Numbered lists
  and sub‑items are fine but avoid deep nesting.

#### Breaking Changes

Breaking changes **must** use one of these conventional commit markers so
that `git‑cliff` can detect them and generate the
`### ⚠️ Breaking Changes` section in `CHANGELOG.md`:

- **Bang notation**: `feat!: remove deprecated API` (append `!` after the type/scope)
- **Footer trailer**: add `BREAKING CHANGE: <description>` as a
  [git trailer](https://git-scm.com/docs/git-interpret-trailers) at the
  end of the commit body

Examples of breaking changes: removing/renaming public API items, changing
default behaviour, bumping MSRV, altering serialisation formats.

---

## Validation Workflow

After modifying files, run appropriate validators.

Common commands:

```bash
just fix
just check
just ci
```

Refer to `docs/dev/commands.md` for full details.

---

## Project Context

- **Language**: Rust
- **Project**: d‑dimensional Delaunay triangulation library
- **MSRV**: 1.95
- **Edition**: 2024
- **Unsafe code**: forbidden (`#![forbid(unsafe_code)]`)

Architecture details are documented in:

```text
docs/code_organization.md
```

---

## Design Principles

This is a scientific d‑dimensional Delaunay triangulation library.  Design
decisions trade off in roughly this priority: **numerical correctness →
topological correctness → API stability → composability → idiomatic Rust →
performance within scope**.  The sections below spell out what each means
in practice; when in doubt, favour the invariant over the convenient edit.

### Numerical correctness as an invariant

- Geometric predicates (`insphere`, `insphere_lifted`, `orientation`) use
  the three‑stage pattern from `src/geometry/predicates.rs`:
  **Stage 1** — provable f64 fast filter with a Shewchuk‑style errbound;
  **Stage 2** — exact sign via Bareiss in `la-stack`;
  **Stage 3** — deterministic `InSphere::BOUNDARY` / `Orientation::DEGENERATE`
  fallback for non‑finite inputs.  A new predicate must either plug into
  this pattern or justify the deviation in its docs.
- No f64 operation may silently lose sign information.  `unwrap_or(NaN)`,
  `unwrap_or(f64::INFINITY)`, or "return `true` on error" are anti‑patterns.
- Algorithms cite their source (Shewchuk, Bowyer–Watson, Edelsbrunner,
  Preparata–Shamos, …) in `REFERENCES.md` and document their
  conditioning behaviour.
- When two predicate implementations cover the same question
  (e.g. `insphere` vs `insphere_distance` vs `insphere_lifted`), a
  proptest verifies they agree on the domain where all are defined.

### Topological correctness as an invariant

- Every mutating operation preserves the invariants checked by
  `Tds::is_valid` (Level 1–3) and `DelaunayTriangulation::is_valid`
  (Level 4).  An operation that cannot preserve them must fail explicitly
  rather than leave the triangulation in an inconsistent state.
- PL‑manifold invariants: facets have multiplicity 1 (boundary) or 2
  (interior); ridges are linked consistently; the Euler characteristic
  matches the triangulation's `TopologyGuarantee`.  The checks live in
  `src/topology/manifold.rs` and `src/topology/characteristics/`.
- Repair paths (`repair_delaunay_with_flips`, `repair_facet_oversharing`,
  `delaunayize_by_flips`) bound their work via explicit budgets
  (`max_flips`, `max_iterations`, `max_cells_removed`) and surface
  non‑convergence as a typed error — never by logging and proceeding.

### Validation layering

The library exposes four validation levels, each a superset of the last:

1. **Level 1 — elements**: individual cells, vertices, facets are
   internally consistent (dimensions, UUIDs, coordinate finiteness).
2. **Level 2 — structure**: adjacency pointers and neighbour links form a
   valid incidence graph; no dangling keys.
3. **Level 3 — topology**: PL‑manifold‑with‑boundary, Euler characteristic,
   ridge‑link consistency.
4. **Level 4 — Delaunay property**: every facet is locally Delaunay.

Only Level 4 requires predicate evaluation; Levels 1–3 are pure graph
checks.  Agents adding validation code should place it at the correct
layer and avoid reaching into lower layers unnecessarily.

### Symbolic perturbation (SoS)

Degenerate configurations (cospherical, collinear, coplanar inputs) are
resolved by Simulation‑of‑Simplicity in `src/geometry/sos.rs`.  This is
a first‑class invariant, not an implementation detail — callers can
rely on predicates returning a consistent, total ordering even on
degenerate input, and tests under `tests/proptest_sos.rs` enforce that.

### Public‑API stability

- Error enums are `#[non_exhaustive]`; public wrapper types are
  `#[must_use]`.  New variants are additive.
- New functionality is additive: use `crate::prelude::*` (or the focused
  `prelude::triangulation`, `prelude::query`, etc.) for ergonomic
  re‑exports; never silently rename or remove a public item.
- Pre‑1.0 semver: `0.x.Y` is a patch‑level additive bump, `0.X.y` is a
  minor bump that may include breaking changes.  Conventional‑commit
  types (`feat`, `fix`, `refactor`, …) mirror this convention.
- Publish documentation changes *before* bumping the crates.io version
  (crates.io does not allow re‑publishing docs without a version bump).

### Composability

- Const‑generic `D` on every core type (`DelaunayTriangulation<K, U, V, D>`,
  `Tds<T, U, V, D>`, `Cell<T, U, V, D>`, `Vertex<T, U, D>`, `Point<T, D>`).
  No runtime dimension.
- Per‑simplex data is stack‑allocated (`[T; D]` coordinates,
  `SmallBuffer<VertexKey, MAX_PRACTICAL_DIMENSION_SIZE>`).  The
  triangulation's topology is stored in `SlotMap` — heap‑backed by
  necessity, not by accident.
- Feature flags isolate optional dependency weight.  Default builds stay
  dep‑minimal.  Known flags: `dense-slotmap` (default),
  `count-allocations`, `bench`, `bench-logging`, `test-debug`,
  `slow-tests`.

### Idiomatic Rust as a proxy for mathematical clarity

- `#![forbid(unsafe_code)]` is a hard constraint, not a guideline.
- `const fn` for pure‑math helpers (`sign_to_orientation`,
  `sign_to_insphere`, coordinate conversions) where the inputs allow.
  Do not twist mutating APIs into `const fn` for its own sake.
- `Result<_, _Error>` for every fallible operation.  Panics are reserved
  for documented, debug‑only precondition violations; library code in
  `src/` must not panic on user input.
- Borrow by default (`&T`, `&mut T`, `&[T]`); return borrowed views where
  possible.  `FacetView`, `AdjacencyIndex`, and the `cells()`/`vertices()`
  iterators are examples.
- Type and function names match the textbook vocabulary: `Triangulation`,
  `Vertex`, `Cell`, `Facet`, `Ridge`, `InSphere`, `Orientation`,
  `insphere`, `circumcenter`, `circumradius`.  Avoid Rust‑ecosystem
  abstractions that obscure the math.
- Use `tracing::{debug,info,warn,error}!` for all runtime diagnostics.
  Never `eprintln!` / `println!` outside examples and benches.

### Scientific notation in docs

- Unicode math (×, ≤, ≥, ∈, Σ, ², `2^-50`, …) is welcome in doc
  comments — readability trumps ASCII‑only preference.
- Reference literature via `REFERENCES.md` numbered citations.
- State invariants mathematically where possible (e.g.
  `χ(S^d) = 1 + (−1)^d`) rather than prose‑only.

### Performance within scope

- Performance is a design goal but strictly subordinate to the principles
  above.  Never trade correctness, stability, or clarity for speed; if
  the two conflict, re‑scope the problem rather than compromise the
  invariant.
- **In scope**: d‑dimensional Delaunay triangulations for small‑to‑medium
  dimensions (typically 2 ≤ D ≤ 7), single‑threaded in‑memory
  construction, `SlotMap`‑backed topology, Hilbert‑ordered insertion.
- **Out of scope**: massively parallel / GPU meshing, out‑of‑core
  triangulations, sparse sampling, dynamic remeshing at scale.  Those
  belong to specialised tools (CGAL, TetGen, Gmsh).
- Within scope, prefer:
  - allocation‑free hot paths (`SmallBuffer`, stack arrays, iterators)
  - Shewchuk‑style f64 fast filters with `core::hint::cold_path()` on
    exact‑arithmetic fallbacks (see `src/geometry/predicates.rs`)
  - `const fn` where the inputs allow
  - typed flip/insertion budgets rather than heuristic timeouts
- Validate any performance claim against one of the benchmark suites in
  `benches/` (`ci_performance_suite`, `large_scale_performance`,
  `profiling_suite`, `cold_path_predicates`, `microbenchmarks`,
  `circumsphere_containment`, `topology_guarantee_construction`) before
  relying on it.

### Testing mirrors the principles

- Unit tests cover known values, error paths, and dimension‑generic
  correctness.  Dimension‑generic tests **must cover D=2 through D=5**
  whenever feasible; use `pastey` macros to generate per‑dimension tests
  (see `src/core/cell.rs`, `src/core/tds.rs` for patterns).
- Proptests under `tests/proptest_*.rs` cover algebraic and
  topological invariants — round‑trips, Euler characteristic, orientation
  sign agreement, SoS consistency — not just "does it not panic".
- Adversarial inputs (near‑boundary points, cospherical sets, degenerate
  simplices, large coordinates) accompany well‑conditioned inputs in
  both tests and benchmarks.
- When a public API has two paths for the same question (fast filter +
  exact fallback, or two alternative predicates), a proptest verifies
  they agree on the domain where all are defined.

---

## Testing

For *what* tests cover and why, see **Design Principles → Testing mirrors the principles** above.
For detailed rules (proptest conventions, adversarial inputs, etc.), see `docs/dev/testing.md`.

Typical commands:

```bash
just test
just test-integration
just test-all
just examples
```

---

## Documentation Maintenance

- Never edit `CHANGELOG.md` or `docs/archive/changelog/*.md` manually
- Run `just changelog` to regenerate the root changelog and archive files from commits
- The root `CHANGELOG.md` contains only Unreleased + the active minor series; completed minors are archived in `docs/archive/changelog/X.Y.md`

---

## Agent Behavior Expectations

The invariants in **Design Principles** above are authoritative; this
section only lists expectations that are not already codified there.

- Prefer small, focused patches.
- Search `docs/` and `docs/dev/` before inventing new conventions.
- When a design question is ambiguous, default to the trade‑off ordering
  in Design Principles (numerical correctness → topological correctness
  → API stability → composability → idiomatic Rust → performance).
- Keep code simple and maintainable when multiple correct solutions exist.
