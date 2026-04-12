# TODO — Weaknesses & Risks

**As of:** 2026-04-12 · post-v0.7.5 (main, unreleased)

Identified during a full codebase evaluation. Items are grouped by category
and prioritized by severity. Scoping notes reference the *next release*
(version TBD).

Legend: **🔴 High** · **🟡 Medium** · **🟢 Low**

---

## 1 · Correctness

### 🔴 3D flip-cycle non-convergence (#306, #204)

Flip-based Delaunay repair enters cycles at ≥35 vertices (seed-dependent).
SoS eliminates predicate ambiguity; root cause is cavity/topology
interactions. This is the primary open correctness issue.

**Status:** Diagnostic infrastructure (conflict-region verification,
orientation audits) shipped in #309 and #319. Repair constants unified
across build profiles in #319. Root cause narrowed but not yet fixed.

### 🔴 4D bulk construction vertex skipping (#307, #204)

Batch 4D construction (100 points, specific seed) produces a
negative-orientation cell early, causing 88% of subsequent insertions to be
skipped as degeneracies. Incremental insertion is a viable workaround.

**Status:** Same diagnostic infrastructure as above. Orientation-audit
improvements shipped. Root cause not yet fixed.

---

## 2 · Performance

### 🟡 Exact predicate fast-filter rejection rate (#256)

The provable `det_errbound()` bounds correctly reject more cases to exact
Bareiss than the old heuristic, causing ~47 proptests to exceed CI
timeouts. A Shewchuk-style multi-stage adaptive expansion would reduce
exact-path frequency without sacrificing correctness.

**Status:** not feasible yet — requires upstream `la-stack` work or a
new adaptive expansion layer. Track in #256.

### 🟡 3D triangulation performance (#310)

General 3D construction and query performance. Profiling and targeted
optimization needed.

**Status:** profiling can begin; targeted fixes possible if
bottlenecks are clear.

---

## 3 · Codebase Complexity

### ✅ ~~Monolithic implementation blocks (#302)~~ — DONE

Completed in #311: `impl` blocks split by concern (construction, query,
mutation, validation).

### ✅ ~~Module structure (#288)~~ — DONE

Completed in #317: `delaunay_triangulation.rs` → `triangulation/delaunay.rs`,
`builder.rs` → `triangulation/builder.rs`. Public API preserved via re-exports.

### 🟢 Builder function size

`builder.rs` has `#[expect(clippy::too_many_lines)]` suppressions on
`search_closed_2d_selection` and related functions. These should be
decomposed into smaller helpers.

**Status:** feasible — localized refactor with low risk.

---

## 4 · API & Ergonomics

### 🟡 Prelude nesting depth

The prelude has 7+ sub-modules (`triangulation`, `geometry`, `query`,
`collections`, `topology::validation`, `triangulation::flips`,
`triangulation::delaunayize`). New users may struggle to find the right
import path.

**Status:** documentation improvements are feasible (e.g., a
"which import do I need?" section in the crate docs). Structural changes
should wait for a minor release.

### 🟡 Doctest migration to builder API (#214)

Many doctests still use `DelaunayTriangulation::new()` directly. Migrating
to `DelaunayTriangulationBuilder` would dogfood the recommended API.

**Status:** feasible — mechanical, low risk, improves documentation
quality.

### 🟢 Unified Pachner move API (#252)

The current flip API has separate methods for each k value. A unified
`flip(move)` entry point would simplify the surface.

**Status:** not for a patch release — API design work needed.

---

## 5 · Testing Gaps

### 🟡 Disabled proptests (~47 tests, #256)

Property tests disabled due to exact-predicate timeouts. These represent
real coverage gaps for near-degenerate inputs.

**Status:** blocked by predicate performance (#256). Could
selectively re-enable the fastest subset.

### 🟢 Constrained Delaunay triangulations (#299)

No CDT support yet. This is a feature gap rather than a bug, but it limits
applicability for mesh generation workflows.

**Status:** out of scope — significant new feature.

---

## 6 · Infrastructure

### 🟢 Visualization (#64)

No built-in visualization for debugging or user inspection. Third-party
tools (e.g., plotting point sets) are the current workaround.

**Status:** out of scope.

### 🟢 Voronoi diagrams (#63)

The dual Voronoi diagram is a natural extension but not yet implemented.

**Status:** out of scope.

---

## Next Release Candidates

Remaining items for the next release, ordered by impact:

1. ~~**Diagnostic infrastructure for #306/#307**~~ — ✅ shipped in #309, #319
2. ~~**Split monolithic `impl` blocks (#302)**~~ — ✅ shipped in #311
3. ~~**Module structure (#288)**~~ — ✅ shipped in #317
4. **Builder function decomposition** — eliminate `too_many_lines`
   suppressions in `builder.rs`.
5. **Doctest migration to builder API (#214)** — improves documentation
   quality and dogfoods the recommended API.
6. **Prelude import guidance** — add a "which import?" section to crate-level
   docs.
7. **Selectively re-enable fastest proptests** — partial recovery of
   coverage from #256 disabled tests.
8. **3D performance profiling (#310)** — begin profiling; ship fixes if
   bottlenecks are clear.
