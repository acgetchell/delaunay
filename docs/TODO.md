# TODO — Weaknesses & Risks

**As of:** 2026-04-12 · post-v0.7.5 (main, unreleased)

Identified during a full codebase evaluation. Items are grouped by category
and prioritized by severity. Scoping notes reference the *next release*
(version TBD).

Legend: **🔴 High** · **🟡 Medium** · **🟢 Low**

---

## 1 · Correctness

### ✅ ~~3D flip-cycle non-convergence (#306, #204)~~ — FIXED

The historical 35-vertex and 1000-vertex release-mode repros no longer fail on
the current branch. The original seed `0xE30C78582376677C` now passes at 35
vertices, at 1000 vertices, and the 1000-prefix bisect reports no failing
prefix.

**Status:** release-mode recheck completed on 2026-04-23; keep #204 focused on
larger-scale monitoring and regression detection rather than the old #306
correctness repro.

### ✅ ~~4D bulk construction vertex skipping (#307, #204)~~ — FIXED

The historical 100-point 4D release-mode repro no longer skips vertices on the
current branch. The original seed `0x9B7786C999C56A16` now inserts all 100
vertices with zero skips and passes validation.

**Status:** release-mode recheck completed on 2026-04-23; keep #204 focused on
larger 4D batch-runtime/observability work rather than the old #307
orientation-skip repro.

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

### 🟡 4D large-scale batch runtime / observability (#204)

The known 100-point correctness repro is fixed, but larger seeded 4D release
batch runs still degrade into skip-heavy retries and can fail all shuffled
attempts. The clearest bounded repro is now the 500-point seed
`0xD225B8A07E274AE6`, which spent ~595.9s exhausting attempts 0..6 before
failing with `Cell violates Delaunay property: cell contains vertex that is
inside circumsphere`.

**Status:** 2026-04-23 rechecks confirmed the 100-point case is healthy and the
new retry-boundary instrumentation is working. The 500-point seeded repro shows
attempts ending around `inserted≈266–300`, `skipped≈200–234`, with skip samples
dominated by `Conflict region error: Ridge fan detected: 4 facets share ridge
with 3 vertices`. Continue #204 by tracing that conflict-region ridge-fan path
through the retryable skip logic rather than treating the issue as pure
observability.

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
