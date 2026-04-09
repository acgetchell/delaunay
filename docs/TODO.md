# TODO — Weaknesses & Risks

Identified during a full codebase evaluation (v0.7.4, branch
`feat/diagnostic-infrastructure-v076`). Items are grouped by category and
prioritized by severity.

Legend: **🔴 High** · **🟡 Medium** · **🟢 Low**

---

## 1 · Correctness

### 🔴 3D flip-cycle non-convergence (#306, #204)

Flip-based Delaunay repair enters cycles at ≥35 vertices (seed-dependent).
SoS eliminates predicate ambiguity; root cause is cavity/topology
interactions. This is the primary open correctness issue.

**v0.7.5 scope:** the `feat/diagnostic-infrastructure-v076` branch adds
conflict-region completeness verification and orientation audits. A fix is
unlikely for v0.7.5 but improved diagnostics will ship, narrowing the root
cause for a subsequent release.

### 🔴 4D bulk construction vertex skipping (#307, #204)

Batch 4D construction (100 points, specific seed) produces a
negative-orientation cell early, causing 88% of subsequent insertions to be
skipped as degeneracies. Incremental insertion is a viable workaround.

**v0.7.5 scope:** same diagnostic branch. Orientation-audit improvements
may expose the root cause but a fix is unlikely for this release.

---

## 2 · Performance

### 🟡 Exact predicate fast-filter rejection rate (#256)

The provable `det_errbound()` bounds correctly reject more cases to exact
Bareiss than the old heuristic, causing ~47 proptests to exceed CI
timeouts. A Shewchuk-style multi-stage adaptive expansion would reduce
exact-path frequency without sacrificing correctness.

**v0.7.5 scope:** not feasible — requires upstream `la-stack` work or a
new adaptive expansion layer. Track in #256.

### 🟡 3D triangulation performance (#310)

General 3D construction and query performance. Profiling and targeted
optimization needed.

**v0.7.5 scope:** profiling can begin; targeted fixes possible if
bottlenecks are clear.

---

## 3 · Codebase Complexity

### 🟡 Monolithic implementation blocks (#302)

`DelaunayTriangulation` has large `impl` blocks with broad trait bounds
that make the code harder to navigate and increase compile times.

**v0.7.5 scope:** feasible — mechanical refactor to split `impl` blocks by
concern (construction, query, mutation, validation). Low risk.

### 🟡 Module structure (#288)

`delaunay_triangulation.rs` and `builder.rs` live under `core/` but could
be moved to `triangulation/` for clearer layering.

**v0.7.5 scope:** not recommended for a patch release — high churn, no
functional benefit.

### 🟢 Builder function size

`builder.rs` has `#[expect(clippy::too_many_lines)]` suppressions on
`search_closed_2d_selection` and related functions. These should be
decomposed into smaller helpers.

**v0.7.5 scope:** feasible — localized refactor with low risk.

---

## 4 · API & Ergonomics

### 🟡 Prelude nesting depth

The prelude has 7+ sub-modules (`triangulation`, `geometry`, `query`,
`collections`, `topology::validation`, `triangulation::flips`,
`triangulation::delaunayize`). New users may struggle to find the right
import path.

**v0.7.5 scope:** documentation improvements are feasible (e.g., a
"which import do I need?" section in the crate docs). Structural changes
should wait for a minor release.

### 🟡 Doctest migration to builder API (#214)

Many doctests still use `DelaunayTriangulation::new()` directly. Migrating
to `DelaunayTriangulationBuilder` would dogfood the recommended API.

**v0.7.5 scope:** feasible — mechanical, low risk, improves documentation
quality.

### 🟢 Unified Pachner move API (#252)

The current flip API has separate methods for each k value. A unified
`flip(move)` entry point would simplify the surface.

**v0.7.5 scope:** not for a patch release — API design work needed.

---

## 5 · Testing Gaps

### 🟡 Disabled proptests (~47 tests, #256)

Property tests disabled due to exact-predicate timeouts. These represent
real coverage gaps for near-degenerate inputs.

**v0.7.5 scope:** blocked by predicate performance (#256). Could
selectively re-enable the fastest subset.

### 🟢 Constrained Delaunay triangulations (#299)

No CDT support yet. This is a feature gap rather than a bug, but it limits
applicability for mesh generation workflows.

**v0.7.5 scope:** out of scope — significant new feature.

---

## 6 · Infrastructure

### 🟢 Visualization (#64)

No built-in visualization for debugging or user inspection. Third-party
tools (e.g., plotting point sets) are the current workaround.

**v0.7.5 scope:** out of scope.

### 🟢 Voronoi diagrams (#63)

The dual Voronoi diagram is a natural extension but not yet implemented.

**v0.7.5 scope:** out of scope — new feature.

---

## v0.7.5 Release Candidates

Items feasible for the v0.7.5 release, ordered by impact:

1. **Diagnostic infrastructure for #306/#307** — already in progress on
   this branch. Ship improved conflict-region verification and orientation
   audits.
2. **Split monolithic `impl` blocks (#302)** — mechanical refactor,
   improves maintainability and compile times.
3. **Builder function decomposition** — eliminate `too_many_lines`
   suppressions in `builder.rs`.
4. **Doctest migration to builder API (#214)** — improves documentation
   quality and dogfoods the recommended API.
5. **Prelude import guidance** — add a "which import?" section to crate-level
   docs.
6. **Selectively re-enable fastest proptests** — partial recovery of
   coverage from #256 disabled tests.
7. **3D performance profiling (#310)** — begin profiling; ship fixes if
   bottlenecks are clear.
